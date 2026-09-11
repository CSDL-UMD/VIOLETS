"""
server/main.py
==============
FastAPI server for the VIOLETS Election Chatbot.

MIDDLEWARE PIPELINE (inside /chat, in order):
---------------------------------------------
    [1] detect_pii()                — Presidio, 0 tokens
    [2] classify_query()            — small LLM, ~70 tokens
    [3] Main RAG chain              — only if [1] and [2] pass
    [4] check_partisan_response()   — small LLM, ~100 tokens

HOW TO RUN:
    python -m server.main            # preferred: pins workers=1 (see below)

    # The app keeps rate-limit + session state in-process, so it MUST run as a
    # single worker. Launching via `python -m server.main` enforces workers=1
    # regardless of WEB_CONCURRENCY / --workers. If you launch uvicorn directly
    # (uvicorn server.main:app --host 0.0.0.0 --port 8000) do NOT pass
    # --workers >1 and do NOT set WEB_CONCURRENCY.

HOW TO TEST:
    # Health check
    curl http://localhost:8000/health

    # Normal query
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "test", "query": "When is the voter registration deadline?"}'

    # PII test (should be blocked)
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "test", "query": "My SSN is 123-45-6789"}'

    # Out-of-scope test (should be blocked)
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "test", "query": "Who is running for Senate in Virginia?"}'

    # Partisan test (should be blocked)
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "test", "query": "Which party is better for Maryland voters?"}'
"""

import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import config
from .logging_setup import setup_logging, new_request_id
from .metrics import METRICS
from .rag_chain import build_chain, to_langchain_messages
from .rag_logger import (
    RAGCallbackHandler,
    log_request,
    LOG_PROMPTS,
    LOG_RESPONSES,
)
from .session import SessionStore
from .timing import begin_request, format_line, stage
from .middleware import (
    QueryContext,
    classify_query,
    detect_pii,
    check_partisan_response,
)

logger = logging.getLogger(__name__)

# Hard ceiling on the main RAG chain so a hung upstream LLM/DB call can't
# pin the request (and its threadpool worker) indefinitely.
RAG_CHAIN_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(key: str | None = Security(_api_key_header)):
    if key is None or not hmac.compare_digest(key, config.VIOLETS_API_KEY):
        # WARNING, not ERROR — a single bad key is routine, but a burst of these
        # is worth a lead's attention (misconfigured client or probing).
        logger.warning("Auth failed — missing or invalid API key")
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple sliding-window rate limiter keyed by user_id."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            # Evict idle keys whose most recent request is outside the window
            # so the dict stays bounded (it is never otherwise pruned).
            stale = [
                k
                for k, ts in self._requests.items()
                if k != key and (not ts or now - ts[-1] >= self._window)
            ]
            for k in stale:
                del self._requests[k]

            timestamps = [t for t in self._requests[key] if now - t < self._window]
            if len(timestamps) >= self._max:
                self._requests[key] = timestamps
                return False
            timestamps.append(now)
            self._requests[key] = timestamps
            return True


_rate_limiter = _RateLimiter(max_requests=config.RATE_LIMIT_PER_MINUTE)

# Global backstop — the per-user limiter above is keyed on the client-supplied
# user_id, so rotating ids bypasses it. This second limiter uses one fixed key
# for ALL requests and caps total throughput regardless of how many ids a
# client invents.
_global_rate_limiter = _RateLimiter(max_requests=config.RATE_LIMIT_GLOBAL_PER_MINUTE)
_GLOBAL_RATE_KEY = "__global__"


# ---------------------------------------------------------------------------
# Globals (initialized at startup)
# ---------------------------------------------------------------------------

store: SessionStore | None = None
chain = None
_rag_callback: RAGCallbackHandler | None = None
_pool = None


# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------


HEARTBEAT_INTERVAL = 300  # seconds


def _pool_gauge() -> str:
    """Best-effort 'in-use/max' pool snapshot for the heartbeat. Never raises.

    Denominator is pool_max (the 25-connection ceiling), not the currently-open
    count — the pool keeps only min_size (4) connections warm when idle and
    grows toward max under load, so dividing by the open count would make an
    idle pool read "0/4" and hide the real headroom. requests_waiting > 0 means
    every connection is checked out and callers are queuing — the saturation
    signal worth watching.
    """
    try:
        stats = _pool.get_stats() if _pool is not None else {}
        maximum = stats.get("pool_max", 0)
        size = stats.get("pool_size", 0)  # connections currently open
        available = stats.get("pool_available", 0)  # open and idle
        waiting = stats.get("requests_waiting", 0)
        in_use = size - available
        gauge = f"pool={in_use}/{maximum} in use ({size} open)"
        if waiting:
            gauge += f", {waiting} waiting"
        return gauge
    except Exception:
        return "pool=?"


async def _periodic_maintenance():
    """Every interval: expire old sessions and emit a heartbeat summary.

    This is a routine health line, NOT an error channel — errors are logged the
    instant they occur elsewhere. The heartbeat only answers "how's it doing?"
    (volume, cost, pool utilization) at a glance.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            if store:
                store.cleanup_expired()
        except Exception:
            # Never let a transient error kill the task permanently.
            logger.exception("Periodic session cleanup failed; continuing.")

        try:
            m = METRICS.drain_rolling()
            logger.info(
                "HEARTBEAT last %dm — requests=%d errors=%d blocked=%d cost=~$%.4f "
                "| since boot: requests=%d errors=%d cost=~$%.4f | %s",
                HEARTBEAT_INTERVAL // 60,
                m["requests"],
                m["errors"],
                m["blocked"],
                m["cost"],
                m["total_requests"],
                m["total_errors"],
                m["total_cost"],
                _pool_gauge(),
            )
        except Exception:
            logger.exception("Heartbeat emission failed; continuing.")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, store, _rag_callback, _pool

    # Idempotent — also called from __main__ before uvicorn starts, but configure
    # here too so launching via `uvicorn server.main:app` still gets file logging.
    setup_logging()

    logger.info(
        "Starting VIOLETS server — model=%s  k=%d  session_ttl=%dm",
        config.LLM_MODEL,
        config.RETRIEVER_K,
        config.SESSION_TTL_MINUTES,
    )

    if LOG_PROMPTS or LOG_RESPONSES:
        logger.warning(
            "Verbose prompt/response logging enabled — disable for production."
        )

    # PII detection still runs in the threadpool via asyncio.to_thread, so keep
    # the limiter above the default 40 tokens. (Retrieval no longer uses the
    # threadpool — PgVectorRetriever is fully async via AsyncConnectionPool.)
    try:
        import anyio

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 100
    except Exception:
        logger.warning("Could not raise anyio thread limiter; using default.")

    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async

    _pool = AsyncConnectionPool(
        conninfo=config.DATABASE_URL,
        min_size=4,
        max_size=25,
        # Fail fast on connection checkout instead of silently eating the 60s
        # chain budget if every connection is busy.
        timeout=10,
        configure=register_vector_async,
        open=False,
    )
    # Block until the pool is ready so the first request doesn't race startup.
    try:
        await _pool.open(wait=True, timeout=10)
    except Exception:
        logger.exception(
            "Database unreachable at startup — check DATABASE_URL and that "
            "Postgres/pgvector is running. Server cannot start."
        )
        raise

    store = SessionStore(
        ttl_minutes=config.SESSION_TTL_MINUTES,
        max_turns=config.MAX_HISTORY_TURNS,
    )

    chain = build_chain(_pool)
    _rag_callback = RAGCallbackHandler()

    cleanup_task = asyncio.create_task(_periodic_maintenance())

    logger.info("Server ready.")
    yield

    cleanup_task.cancel()
    await _pool.close(timeout=30)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VIOLETS Election Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "https://umdsurvey.umd.edu", "https://verasight.qualtrics.com"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    query: str = Field(min_length=1, max_length=2000)


class SourceReference(BaseModel):
    source_number: int
    source_url: str
    # All URLs this chunk's content appears at (deduplicated content surfaces
    # multiple sources). Defaults to [source_url] for single-source chunks.
    source_urls: list[str] = []
    title: str
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceReference] = []


class ResetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
async def chat(req: ChatRequest):
    # Bind a correlation id for this request — every log line below (across
    # guardrails, chain, and callbacks) is stamped with it via the log filter.
    new_request_id()

    # Per-stage wall-clock accounting. Stages inside the chain (rephrase,
    # embed, retrieve, generate) record themselves into this same dict via the
    # timing contextvar. One TIMINGS line is logged per request outcome.
    timings = begin_request()
    req_start = time.time()

    def _log_timings(outcome: str) -> None:
        logger.info(
            "TIMINGS [user=%s outcome=%s] %s",
            req.user_id,
            outcome,
            format_line(timings, time.time() - req_start),
        )

    # Shared context object — travels through all guardrails
    ctx = QueryContext(user_id=req.user_id)

    # Guard against requests arriving before lifespan startup finished.
    if store is None or chain is None:
        raise HTTPException(status_code=503, detail="Service starting, retry shortly")

    # Per-user rate limit check first (before any LLM calls) — checked before
    # the global limiter so a client hammering a single user_id is rejected
    # here without consuming global slots (at most RATE_LIMIT_PER_MINUTE of
    # them per minute) instead of starving every other user.
    if not _rate_limiter.check(req.user_id):
        logger.warning("Rate limit exceeded [user=%s]", req.user_id)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Global backstop — a client rotating user_ids slips past the per-user
    # check above but still hits this hard ceiling.
    if not _global_rate_limiter.check(_GLOBAL_RATE_KEY):
        logger.warning("Global rate limit exceeded [user=%s]", req.user_id)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # ------------------------------------------------------------------
    # GUARDRAIL 1: PII detection (Presidio, zero tokens)
    # Runs first because it costs nothing. If PII is found we never
    # spend tokens on classification or the RAG chain.
    # ------------------------------------------------------------------
    # Presidio analysis is CPU-bound and blocks the event loop; offload it.
    with stage("pii"):
        pii_response = await asyncio.to_thread(detect_pii, req.query, ctx)
    if pii_response:
        logger.warning(
            "Request blocked — PII detected [user=%s type=%s]",
            req.user_id,
            ctx.pii_type,
        )
        METRICS.record_request("blocked:pii")
        _log_timings("blocked:pii")
        return ChatResponse(response=pii_response)

    # ------------------------------------------------------------------
    # GUARDRAIL 2: Query classification (~70 tokens)
    # Runs after PII check. Blocks out-of-scope and partisan queries
    # before the expensive RAG chain is invoked.
    # ------------------------------------------------------------------
    with stage("classify"):
        classification_response = await classify_query(req.query, ctx)
    if classification_response:
        logger.info(
            "Request blocked — query not in scope [user=%s category=%s]",
            req.user_id,
            ctx.query_category,
        )
        METRICS.record_request("blocked:scope")
        _log_timings("blocked:scope")
        return ChatResponse(response=classification_response)

    # ------------------------------------------------------------------
    # Session created only after guardrails pass (avoids wasting memory
    # on blocked PII / out-of-scope / partisan queries).
    # ------------------------------------------------------------------
    store.get_or_create(req.user_id)
    chat_history = to_langchain_messages(store.get_history(req.user_id))
    last_sources = store.get_last_sources(req.user_id)

    # ------------------------------------------------------------------
    # MAIN RAG CHAIN
    # Only reached if both guardrails above passed.
    # ------------------------------------------------------------------
    start = time.time()
    try:
        result = await asyncio.wait_for(
            chain.ainvoke(
                {
                    "input": req.query,
                    "chat_history": chat_history,
                    "query_category": ctx.query_category,
                    "last_sources": last_sources,
                },
                config={"callbacks": [_rag_callback]},
            ),
            timeout=RAG_CHAIN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "RAG chain timed out after %ds [user=%s]", RAG_CHAIN_TIMEOUT, req.user_id
        )
        METRICS.record_request("error:timeout")
        _log_timings("error:timeout")
        raise HTTPException(status_code=502, detail="Failed to generate response.")
    except Exception:
        # logger.exception captures the full traceback — this is the line a lead
        # forwards to a developer, so it must carry more than the message string.
        logger.exception("RAG chain error [user=%s]", req.user_id)
        METRICS.record_request("error:chain")
        _log_timings("error:chain")
        raise HTTPException(status_code=502, detail="Failed to generate response.")

    answer = result["answer"]
    sources = [SourceReference(**s) for s in result.get("sources", [])]

    # ------------------------------------------------------------------
    # GUARDRAIL 3: Partisan response check (~100 tokens)
    # Runs after the chain so it can inspect the output.
    # Retries up to MAX_PARTISAN_RETRIES times with a stricter prompt
    # if partisan content is detected. If all retries fail, fails closed —
    # discards the response and returns a nonpartisan fallback.
    # ------------------------------------------------------------------
    retry_retrieved: list | None = None
    partisan_fallback = False
    if result.get("skip_partisan"):
        # The cached-links short-circuit is a deterministic template over
        # already-vetted titles/URLs, not model output — nothing to vet, so
        # don't burn a checker LLM call (or risk a retry) on it.
        logger.debug("Partisan check skipped — deterministic cached-links answer")
    else:
        # NOTE: a partisan retry re-invokes the full chain, so its rephrase/
        # embed/retrieve/generate time lands BOTH in those stages and inside
        # "partisan" — on retry requests the stages sum to more than total.
        with stage("partisan"):
            answer, new_sources, retry_retrieved, partisan_fallback = (
                await check_partisan_response(
                    query=req.query,
                    response=answer,
                    chat_history=chat_history,
                    chain=chain,
                    ctx=ctx,
                    callbacks=[_rag_callback] if _rag_callback else None,
                )
            )
        if new_sources is not None:
            sources = [SourceReference(**s) for s in new_sources]

    METRICS.record_request("ok")
    _log_timings("ok")
    log_request(
        req.user_id, req.query, answer, elapsed=time.time() - start, outcome="ok"
    )
    store.add_exchange(req.user_id, req.query, answer)
    # Cache this turn's retrieved sources so a link follow-up ("can you give
    # me the links?") is answered from real URLs instead of model memory.
    # Conversational turns don't retrieve and omit the key, leaving the cache
    # untouched — a "thanks" doesn't wipe the links to the previous answer.
    if partisan_fallback:
        # Partisan check failed closed — the shown answer is a canned
        # fallback, so never cache the suppressed answer's sources, and clear
        # any previously cached links so a "give me the links" follow-up
        # can't serve sources for an answer the user never saw.
        store.set_last_sources(req.user_id, [])
    elif retry_retrieved is not None:
        # A partisan retry produced the final answer — cache ITS retrieval,
        # not the discarded pre-retry attempt's.
        store.set_last_sources(req.user_id, retry_retrieved)
    elif "retrieved_sources" in result:
        store.set_last_sources(req.user_id, result["retrieved_sources"])

    return ChatResponse(response=answer, sources=sources)


@app.post("/reset", dependencies=[Depends(_verify_api_key)])
async def reset_session(req: ResetRequest):
    if store is None:
        raise HTTPException(status_code=503, detail="Service starting, retry shortly")
    if not _global_rate_limiter.check(_GLOBAL_RATE_KEY):
        logger.warning("Global rate limit exceeded [user=%s]", req.user_id)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    store.reset(req.user_id)
    return {"status": "session cleared"}


@app.get("/health")
async def health():
    # Actually ping the database so an uptime monitor catches a dead pool /
    # unreachable Postgres instead of a false "ok". Returns 503 when the DB is
    # down (or the pool hasn't finished starting) so load balancers pull the node.
    db_ok = False
    try:
        if _pool is not None:
            async with _pool.connection() as conn:
                await conn.execute("SELECT 1")
            db_ok = True
    except Exception as exc:
        # One concise WARNING (no traceback) — during an outage this may repeat
        # each probe, which is the intent: it stays visible in the log.
        logger.warning("Health check — database ping failed: %s", exc)

    body = {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "model": config.LLM_MODEL,
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)


if __name__ == "__main__":
    # Canonical entrypoint. The app holds rate-limit and session state in
    # process memory, so it must run as exactly one worker. We hardcode
    # workers=1 here so single-worker is the default without anyone having to
    # remember the --workers flag (and it overrides WEB_CONCURRENCY).
    import uvicorn

    # Configure our rotating-file + console logging before uvicorn boots, and
    # pass log_config=None so uvicorn doesn't install its own handlers on top
    # (its loggers propagate to our root config instead).
    setup_logging()

    uvicorn.run(
        "server.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        log_config=None,
    )
