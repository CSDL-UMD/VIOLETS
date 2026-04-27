"""
server/log_store.py
===================
Async SQLite log store for chat request records.

Uses asyncio.Queue + a background worker task to write log entries without
blocking the response path. The chatbot returns its answer first; the DB
write happens afterward in a background coroutine.

DB location: logs/chat_logs.db (sibling of logs/crawl.log)

Privacy contract
----------------
- query          : only populated when LOG_QUERIES=True AND no PII was found
                   (the early-return guardrail ensures a PII-containing query
                   never reaches this path with text attached)
- pii_type_*     : entity type label only (e.g. "EMAIL_ADDRESS") — never the
                   actual PII value
- response       : full text stored — always post-PII-sanitization, so safe to log
- retrieved_sources : URL, title, and similarity score — no chunk text
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File path
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DB_PATH: Path = _LOG_DIR / "chat_logs.db"

# Drop entries rather than blocking the response path if the worker falls behind.
_QUEUE_MAXSIZE = 1_000


# ---------------------------------------------------------------------------
# Log entry schema
# ---------------------------------------------------------------------------

@dataclass
class ChatLogEntry:
    """One row in chat_logs — one row per /chat request."""

    # ---- Identity ----
    timestamp: float                        # unix epoch (time.time())
    user_id: str
    session_turn: int                       # 1-based turn count for this session

    # ---- Query ----
    query_len: int                          # character length of raw query
    query: str | None                       # text only when LOG_QUERIES=True and no PII
    rephrased_query: str | None             # standalone question sent to retriever

    # ---- Guardrail outcomes ----
    query_category: str | None             # "normal" | "out_of_scope" | "partisan"
    guardrail_blocked_by: str | None       # "pii" | "out_of_scope" | "partisan" | None
    pii_in_query: bool = False             # PII was detected in the input
    pii_type_in_query: str | None = None   # entity type label, NOT the value

    # ---- Retrieval ----
    retrieved_sources: list[dict] = field(default_factory=list)
    # Each dict: {source_number, source_url, title, score}

    # ---- Response ----
    response: str | None = None            # final response text (post-PII sanitization, always safe)
    response_len: int = 0                  # character length (derived, kept for quick queries)
    pii_in_response: bool = False          # PII was detected and replaced with fallback
    pii_type_in_response: str | None = None  # entity type label only, NOT the PII value

    # ---- Partisan check ----
    partisan_detected: bool = False        # LLM response was flagged partisan
    partisan_retried: bool = False         # chain was re-run with strict prompt

    # ---- Token / cost ----
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    # ---- Latency (milliseconds) ----
    retrieval_latency_ms: float | None = None
    llm_latency_ms: float | None = None    # cumulative across all LLM calls
    total_latency_ms: float | None = None  # full end-to-end request time

    # ---- Model / errors ----
    model: str | None = None
    error: str | None = None               # exception message if chain failed


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             REAL    NOT NULL,
    user_id               TEXT    NOT NULL,
    session_turn          INTEGER,
    query_len             INTEGER,
    query                 TEXT,
    rephrased_query       TEXT,
    query_category        TEXT,
    guardrail_blocked_by  TEXT,
    pii_in_query          INTEGER NOT NULL DEFAULT 0,
    pii_type_in_query     TEXT,
    retrieved_sources     TEXT,
    response              TEXT,
    response_len          INTEGER,
    pii_in_response       INTEGER NOT NULL DEFAULT 0,
    pii_type_in_response  TEXT,
    partisan_detected     INTEGER NOT NULL DEFAULT 0,
    partisan_retried      INTEGER NOT NULL DEFAULT 0,
    prompt_tokens         INTEGER,
    completion_tokens     INTEGER,
    total_tokens          INTEGER,
    estimated_cost_usd    REAL,
    retrieval_latency_ms  REAL,
    llm_latency_ms        REAL,
    total_latency_ms      REAL,
    model                 TEXT,
    error                 TEXT
)
"""

_INSERT_SQL = """
INSERT INTO chat_logs (
    timestamp, user_id, session_turn, query_len, query,
    rephrased_query, query_category, guardrail_blocked_by,
    pii_in_query, pii_type_in_query, retrieved_sources,
    response, response_len, pii_in_response, pii_type_in_response,
    partisan_detected, partisan_retried,
    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
    retrieval_latency_ms, llm_latency_ms, total_latency_ms,
    model, error
) VALUES (
    ?,?,?,?,?,
    ?,?,?,
    ?,?,?,
    ?,?,?,?,
    ?,?,
    ?,?,?,?,
    ?,?,?,
    ?,?
)
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AsyncLogStore:
    """
    Non-blocking SQLite log store.

    enqueue() is called on the response path — it is instant (O(1) queue put).
    A background asyncio task drains the queue and writes to SQLite via
    asyncio.to_thread(), keeping the event loop free.

    Usage
    -----
    In server lifespan:
        log_store = AsyncLogStore()
        log_store.start()
        ...
        await log_store.stop()

    In request handler:
        log_store.enqueue(ChatLogEntry(...))
    """

    def __init__(self, db_path: Path = LOG_DB_PATH):
        self._db_path = db_path
        self._queue: asyncio.Queue[ChatLogEntry | None] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        """Create DB + table if needed, then start background worker."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._worker_task = asyncio.create_task(
            self._worker(), name="log-store-worker"
        )
        logger.info("AsyncLogStore ready — db=%s", self._db_path)

    async def stop(self) -> None:
        """Drain remaining entries and shut down the worker."""
        await self._queue.put(None)  # sentinel
        if self._worker_task:
            await self._worker_task

    def enqueue(self, entry: ChatLogEntry) -> None:
        """
        Put a log entry on the queue. Never blocks.
        Drops silently (with a warning) if the queue is at capacity.
        """
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning(
                "Log queue full — dropping entry for user=%s", entry.user_id
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    async def _worker(self) -> None:
        while True:
            entry = await self._queue.get()
            if entry is None:  # shutdown sentinel
                self._queue.task_done()
                break
            try:
                await asyncio.to_thread(self._write_entry, entry)
            except Exception as exc:
                logger.error("Log write failed: %s", exc)
            finally:
                self._queue.task_done()

    def _write_entry(self, entry: ChatLogEntry) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(_INSERT_SQL, (
                entry.timestamp,
                entry.user_id,
                entry.session_turn,
                entry.query_len,
                entry.query,
                entry.rephrased_query,
                entry.query_category,
                entry.guardrail_blocked_by,
                int(entry.pii_in_query),
                entry.pii_type_in_query,
                json.dumps(entry.retrieved_sources),
                entry.response,
                entry.response_len,
                int(entry.pii_in_response),
                entry.pii_type_in_response,
                int(entry.partisan_detected),
                int(entry.partisan_retried),
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.total_tokens,
                entry.estimated_cost_usd,
                entry.retrieval_latency_ms,
                entry.llm_latency_ms,
                entry.total_latency_ms,
                entry.model,
                entry.error,
            ))
            conn.commit()
        finally:
            conn.close()
