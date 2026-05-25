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
    uvicorn server.main:app --host 0.0.0.0 --port 8000

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
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import config
from .rag_chain import build_chain, to_langchain_messages
from .rag_logger import RAGCallbackHandler, log_request
from .session import SessionStore
from .middleware import (
    QueryContext,
    classify_query,
    detect_pii,
    check_partisan_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(key: str | None = Security(_api_key_header)):
    if key is None or not hmac.compare_digest(key, config.VIOLETS_API_KEY):
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
            timestamps = [
                t for t in self._requests[key]
                if now - t < self._window
            ]
            if not timestamps:
                # Remove empty entries to prevent unbounded growth
                self._requests.pop(key, None)
                self._requests[key] = [now]
                return True
            if len(timestamps) >= self._max:
                self._requests[key] = timestamps
                return False
            timestamps.append(now)
            self._requests[key] = timestamps
            return True


_rate_limiter = _RateLimiter(max_requests=config.RATE_LIMIT_PER_MINUTE)


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


async def _periodic_session_cleanup():
    """Remove expired sessions every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        if store:
            store.cleanup_expired()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, store, _rag_callback, _pool

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info(
        "Starting VIOLETS server — model=%s  k=%d  session_ttl=%dm",
        config.LLM_MODEL,
        config.RETRIEVER_K,
        config.SESSION_TTL_MINUTES,
    )

    from psycopg_pool import ConnectionPool
    from pgvector.psycopg import register_vector

    _pool = ConnectionPool(
        conninfo=config.DATABASE_URL,
        min_size=4,
        max_size=25,
        configure=lambda conn: register_vector(conn),
        open=False,
    )
    _pool.open()

    store = SessionStore(
        ttl_minutes=config.SESSION_TTL_MINUTES,
        max_turns=config.MAX_HISTORY_TURNS,
    )

    chain = build_chain(_pool)
    _rag_callback = RAGCallbackHandler()

    cleanup_task = asyncio.create_task(_periodic_session_cleanup())

    logger.info("Server ready.")
    yield

    cleanup_task.cancel()
    _pool.close(timeout=30)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VIOLETS Election Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "https://umdsurvey.umd.edu"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    query: str = Field(min_length=1, max_length=2000)


class SourceReference(BaseModel):
    source_number: int
    source_url: str
    title: str
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceReference] = []


class ResetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
async def chat(req: ChatRequest):
    # Shared context object — travels through all guardrails
    ctx = QueryContext(user_id=req.user_id)

    # Rate limit check (before any LLM calls)
    if not _rate_limiter.check(req.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # ------------------------------------------------------------------
    # GUARDRAIL 1: PII detection (Presidio, zero tokens)
    # Runs first because it costs nothing. If PII is found we never
    # spend tokens on classification or the RAG chain.
    # ------------------------------------------------------------------
    pii_response = detect_pii(req.query, ctx)
    if pii_response:
        logger.warning(
            "Request blocked — PII detected [user=%s type=%s]",
            req.user_id,
            ctx.pii_type,
        )
        return ChatResponse(response=pii_response)

    # ------------------------------------------------------------------
    # GUARDRAIL 2: Query classification (~70 tokens)
    # Runs after PII check. Blocks out-of-scope and partisan queries
    # before the expensive RAG chain is invoked.
    # ------------------------------------------------------------------
    classification_response = await classify_query(req.query, ctx)
    if classification_response:
        logger.info(
            "Request blocked — query not in scope [user=%s category=%s]",
            req.user_id,
            ctx.query_category,
        )
        return ChatResponse(response=classification_response)

    # ------------------------------------------------------------------
    # Session created only after guardrails pass (avoids wasting memory
    # on blocked PII / out-of-scope / partisan queries).
    # ------------------------------------------------------------------
    store.get_or_create(req.user_id)
    chat_history = to_langchain_messages(store.get_history(req.user_id))

    # ------------------------------------------------------------------
    # MAIN RAG CHAIN
    # Only reached if both guardrails above passed.
    # ------------------------------------------------------------------
    start = time.time()
    try:
        result = await chain.ainvoke(
            {"input": req.query, "chat_history": chat_history,
             "query_category": ctx.query_category},
            config={"callbacks": [_rag_callback]},
        )
    except Exception as exc:
        logger.error("RAG chain error [user=%s]: %s", req.user_id, exc)
        raise HTTPException(status_code=502, detail="Failed to generate response.")

    answer = result["answer"]
    sources = [SourceReference(**s) for s in result.get("sources", [])]

    # ------------------------------------------------------------------
    # GUARDRAIL 3: Partisan response check (~100 tokens)
    # Runs after the chain so it can inspect the output.
    # Retries up to MAX_PARTISAN_RETRIES times with a stricter prompt
    # if partisan content is detected. If all retries fail, returns the
    # last generated response anyway (fail-open).
    # ------------------------------------------------------------------
    answer, new_sources = await check_partisan_response(
        query=req.query,
        response=answer,
        chat_history=chat_history,
        chain=chain,
        ctx=ctx,
        callbacks=[_rag_callback] if _rag_callback else None,
    )
    if new_sources is not None:
        sources = [SourceReference(**s) for s in new_sources]

    log_request(req.user_id, req.query, answer, elapsed=time.time() - start)
    store.add_exchange(req.user_id, req.query, answer)

    return ChatResponse(response=answer, sources=sources)


@app.post("/reset", dependencies=[Depends(_verify_api_key)])
async def reset_session(req: ResetRequest):
    store.reset(req.user_id)
    return {"status": "session cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
    }