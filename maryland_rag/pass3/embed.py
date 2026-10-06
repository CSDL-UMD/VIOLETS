"""
Pass 3: Embed chunks and insert into PostgreSQL with pgvector.

Reads data/chunks.jsonl (output from Pass 2), embeds each chunk's text
using OpenAI text-embedding-3-large (3072-dim), and inserts vectors +
metadata into a PostgreSQL table with the pgvector extension.

Env vars (set in .env or shell):
    OPENAI_API_KEY
    DATABASE_URL          (e.g. postgresql://user:pass@localhost:5432/violets)

Usage:
    python -m maryland_rag pass3
    python -m maryland_rag pass3 --chunks data/chunks.jsonl
    python -m maryland_rag pass3 --resume        # skip already-inserted chunk IDs
"""
import json
import logging
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Must match server/rag_chain.py EMBED_MODEL. Changing the model means a full
# drop-and-reingest: the stored vectors and the column dimension both change.
# 3-large beat 3-small on server/eval_retrieval.py (hit@1 0.77 -> 0.82,
# 2026-10-05) at ~equal query latency.
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# OpenAI allows up to 2048 inputs per request; 100 is conservative and
# keeps individual request payloads small.
EMBED_BATCH_SIZE = 100

# Number of rows to insert per transaction.
INSERT_BATCH_SIZE = 100

# Seconds to wait after a retryable API error before re-attempting.
RETRY_DELAY = 5


def run_embed(
    chunks_path: str = "data/chunks.jsonl",
    resume: bool = False,
) -> int:
    """
    Embed all chunks and insert into PostgreSQL.

    Args:
        chunks_path: Path to the JSONL file produced by Pass 2.
        resume: If True, skip chunk IDs already present in the database.

    Returns:
        Number of vectors inserted.

    Raises:
        RuntimeError: If any chunk permanently failed to embed or insert,
            so the pipeline exits non-zero instead of silently shipping a
            corpus with holes.
    """
    _setup_logging()
    load_dotenv()

    openai_api_key = _require_env("OPENAI_API_KEY")
    database_url = _require_env("DATABASE_URL")

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    conn = _setup_pgvector(database_url)

    if resume:
        done_ids = _get_existing_ids(conn)
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        logger.info(
            "Resume mode: skipping %d already-inserted, %d remaining",
            before - len(chunks),
            len(chunks),
        )

    if not chunks:
        logger.info("Nothing to insert.")
        conn.close()
        return 0

    from openai import OpenAI
    # Same Enterprise endpoint as server/config.py: the project key is
    # rejected (401 incorrect_hostname) on the default api.openai.com.
    oai = OpenAI(
        api_key=openai_api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1"),
    )

    total = len(chunks)
    inserted = 0
    # (chunk_id, reason) for every chunk whose embedding or INSERT
    # permanently failed — either kind must fail the run.
    failed: list[tuple[str, str]] = []

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]

        # --- Embed (bisecting on deterministic 4xx to isolate bad inputs) ---
        pairs, batch_failed = _embed_or_bisect(oai, batch)
        for chunk_id, reason in batch_failed:
            logger.error("Embedding permanently failed for chunk %s: %s", chunk_id, reason)
        failed.extend(batch_failed)
        if not pairs:
            continue

        # --- Insert into PostgreSQL ---
        rows = []
        for chunk, embedding in pairs:
            # Separate known columns from extra metadata
            chunk_id = chunk["chunk_id"]
            text = chunk.get("text", "")
            source_url = chunk.get("source_url", "")
            title = chunk.get("title", "")
            meta = {
                k: v for k, v in chunk.items()
                if k not in ("chunk_id", "text", "source_url", "title")
            }
            rows.append((chunk_id, embedding, text, source_url, title, json.dumps(meta)))

        count, insert_failed = _insert_batch(conn, rows)
        inserted += count
        failed.extend(insert_failed)

        pct = 100.0 * (batch_start + len(batch)) / total
        logger.info(
            "Progress: %d/%d (%.1f%%) — %d inserted",
            batch_start + len(batch),
            total,
            pct,
            inserted,
        )

    conn.close()

    if failed:
        sample = [chunk_id for chunk_id, _ in failed[:10]]
        logger.error(
            "Pass 3 FAILED: %d of %d chunk(s) could not be embedded or inserted "
            "(%d inserted). Sample failed chunk_ids: %s",
            len(failed), total, inserted, sample,
        )
        # The 'all' command must exit non-zero — a silent gap in the vector
        # store is worse than a failed run the operator can retry.
        raise RuntimeError(
            f"Pass 3: {len(failed)} chunk(s) permanently failed to embed or insert "
            f"({inserted} inserted). Sample chunk_ids: {sample}"
        )

    logger.info("Pass 3 complete. %d vectors inserted into pgvector.", inserted)
    return inserted


# ---------------------------------------------------------------------------
# PostgreSQL / pgvector setup
# ---------------------------------------------------------------------------


def _setup_pgvector(database_url: str):
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise ImportError(
            "psycopg or pgvector not installed. Run: pip install 'psycopg[binary]' pgvector"
        )

    conn = psycopg.connect(database_url)

    # CREATE EXTENSION must run (and commit) before register_vector: on a
    # freshly created database — the operator's drop-and-reingest workflow —
    # the vector type doesn't exist yet and register_vector would fail.
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.errors.InsufficientPrivilege:
        raise RuntimeError(
            "Cannot create the pgvector extension with this role. Run "
            "'CREATE EXTENSION vector;' once as a superuser on this database, "
            "then re-run pass 3."
        )
    conn.commit()
    register_vector(conn)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            embedding  vector({EMBED_DIM}),
            text       TEXT,
            source_url TEXT,
            title      TEXT,
            metadata   JSONB DEFAULT '{{}}'::jsonb
        )
    """)
    conn.commit()
    # CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so after a
    # model swap the old column dimension would survive and every insert would
    # fail. Check up front and say exactly what to do.
    existing = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()[0]
    if existing != f"vector({EMBED_DIM})":
        conn.close()
        raise RuntimeError(
            f"chunks.embedding is {existing} but {EMBED_MODEL} produces "
            f"vector({EMBED_DIM}). Back up and DROP TABLE chunks, then re-run "
            "pass 3 for chunks.jsonl and box_chunks.jsonl."
        )
    logger.info("pgvector table 'chunks' is ready (dim=%d).", EMBED_DIM)
    return conn


def _get_existing_ids(conn) -> set[str]:
    """Return the set of chunk_ids already in the database."""
    rows = conn.execute("SELECT chunk_id FROM chunks").fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Batch insert
# ---------------------------------------------------------------------------


def _insert_batch(conn, rows: list[tuple]) -> tuple[int, list[tuple[str, str]]]:
    """Insert a batch of rows, using ON CONFLICT to upsert.

    Each row is wrapped in a savepoint so a single failure doesn't
    roll back previously-committed rows in the same batch.

    Returns (count, failures): failures is (chunk_id, reason) for every row
    that could not be inserted — the caller adds them to the fatal `failed`
    list so a run with holes cannot report success.
    """
    count = 0
    failures: list[tuple[str, str]] = []
    for row in rows:
        try:
            conn.execute("SAVEPOINT insert_row")
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, embedding, text, source_url, title, metadata)
                VALUES (%s, %s::vector, %s, %s, %s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    text       = EXCLUDED.text,
                    source_url = EXCLUDED.source_url,
                    title      = EXCLUDED.title,
                    metadata   = EXCLUDED.metadata
                """,
                row,
            )
            conn.execute("RELEASE SAVEPOINT insert_row")
            count += 1
        except Exception as exc:
            logger.error("Insert failed for chunk %s: %s", row[0], exc)
            failures.append((row[0], f"insert failed: {exc}"))
            conn.execute("ROLLBACK TO SAVEPOINT insert_row")
            continue
    conn.commit()
    return count, failures


# ---------------------------------------------------------------------------
# Embedding with retry + bisection
# ---------------------------------------------------------------------------


def _is_deterministic_client_error(exc: Exception | None) -> bool:
    """True for 4xx API errors that will fail identically on retry
    (e.g. an input exceeding the model's token limit). 408 (request
    timeout) and 429 (rate limit) are transient and excluded."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


def _embed_with_retry(
    oai, texts: list[str], retries: int = 3
) -> tuple[list | None, Exception | None]:
    """Embed texts, retrying transient errors with jittered backoff.

    Returns (embeddings, None) on success or (None, last_error) on failure.
    Deterministic 4xx errors are returned immediately — retrying them can
    never succeed, and the caller bisects the batch instead.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data], None
        except Exception as exc:
            last_exc = exc
            if _is_deterministic_client_error(exc):
                logger.warning(
                    "Embedding request rejected with non-retryable status %s: %s",
                    getattr(exc, "status_code", "4xx"), exc,
                )
                return None, exc
            logger.warning(
                "Embedding attempt %d/%d failed: %s", attempt + 1, retries, exc
            )
            if attempt < retries - 1:
                # Jitter the backoff so retries don't land in lockstep.
                time.sleep(RETRY_DELAY * (attempt + 1) + random.uniform(0, RETRY_DELAY))
    return None, last_exc


def _embed_or_bisect(
    oai, batch: list[dict]
) -> tuple[list[tuple[dict, list]], list[tuple[str, str]]]:
    """Embed a batch of chunks, isolating bad inputs by bisection.

    One over-long text 400s the entire request, so on a deterministic 4xx
    the batch is recursively halved until the offending chunk(s) are
    isolated — the good halves still get embedded instead of being dropped.

    Returns (successes, failures): successes pairs each chunk with its
    embedding; failures is (chunk_id, reason) for permanently-failed chunks.
    """
    embeddings, err = _embed_with_retry(oai, [c["text"] for c in batch])
    if embeddings is not None:
        return list(zip(batch, embeddings)), []
    if len(batch) > 1 and _is_deterministic_client_error(err):
        mid = len(batch) // 2
        left_ok, left_bad = _embed_or_bisect(oai, batch[:mid])
        right_ok, right_bad = _embed_or_bisect(oai, batch[mid:])
        return left_ok + right_ok, left_bad + right_bad
    # A single offending chunk, or a transient error that exhausted retries.
    return [], [(c["chunk_id"], str(err)) for c in batch]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def _load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Add it to your .env file or shell environment."
        )
    return val


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
