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
    [5] detect_pii_in_response()    — Presidio, 0 tokens

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

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .rag_chain import build_chain, to_langchain_messages
from .rag_logger import RAGCallbackHandler, log_request, LOG_QUERIES
from .log_store import AsyncLogStore, ChatLogEntry
from .session import SessionStore
from .middleware import (
    QueryContext,
    classify_query,
    detect_pii,
    detect_pii_in_response,
    check_partisan_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized at startup)
# ---------------------------------------------------------------------------

store: SessionStore | None = None
chain = None
log_store: AsyncLogStore | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, store, log_store

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # API key validation is handled at import time by config._require_env().

    logger.info(
        "Starting VIOLETS server — model=%s  index=%s  k=%d  session_ttl=%dm",
        config.LLM_MODEL,
        config.PINECONE_INDEX_NAME,
        config.RETRIEVER_K,
        config.SESSION_TTL_MINUTES,
    )

    store = SessionStore(
        ttl_minutes=config.SESSION_TTL_MINUTES,
        max_turns=config.MAX_HISTORY_TURNS,
    )

    chain = build_chain()

    log_store = AsyncLogStore()
    log_store.start()

    logger.info("Server ready.")
    yield

    await log_store.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VIOLETS Election Chatbot", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    query: str


class SourceReference(BaseModel):
    source_number: int
    source_url: str
    title: str
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceReference] = []


class ResetRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    store.get_or_create(req.user_id)
    raw_history = store.get_history(req.user_id)
    session_turn = len(raw_history) // 2 + 1
    chat_history = to_langchain_messages(raw_history)

    # Shared context object — travels through all guardrails
    ctx = QueryContext(user_id=req.user_id)
    start = time.time()

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
        log_store.enqueue(ChatLogEntry(
            timestamp=start,
            user_id=req.user_id,
            session_turn=session_turn,
            query_len=len(req.query),
            query=None,  # never store a PII-containing query
            rephrased_query=None,
            query_category=None,
            guardrail_blocked_by="pii",
            pii_in_query=True,
            pii_type_in_query=ctx.pii_type,
            total_latency_ms=(time.time() - start) * 1000,
        ))
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
        log_store.enqueue(ChatLogEntry(
            timestamp=start,
            user_id=req.user_id,
            session_turn=session_turn,
            query_len=len(req.query),
            query=req.query if LOG_QUERIES else None,
            rephrased_query=None,
            query_category=ctx.query_category,
            guardrail_blocked_by=ctx.query_category,
            total_latency_ms=(time.time() - start) * 1000,
        ))
        return ChatResponse(response=classification_response)

    # ------------------------------------------------------------------
    # MAIN RAG CHAIN
    # Only reached if both guardrails above passed.
    # ------------------------------------------------------------------
    callback = RAGCallbackHandler()
    try:
        result = await chain.with_config({"callbacks": [callback]}).ainvoke({
            "input": req.query,
            "chat_history": chat_history,
        })
    except Exception as exc:
        logger.error("RAG chain error [user=%s]: %s", req.user_id, exc)
        log_store.enqueue(ChatLogEntry(
            timestamp=start,
            user_id=req.user_id,
            session_turn=session_turn,
            query_len=len(req.query),
            query=req.query if LOG_QUERIES else None,
            rephrased_query=None,
            query_category=ctx.query_category,
            error=str(exc),
            total_latency_ms=(time.time() - start) * 1000,
        ))
        raise HTTPException(status_code=502, detail="Failed to generate response.")

    answer = result["answer"]
    rephrased_query = result.get("rephrased_query")
    sources = [SourceReference(**s) for s in result.get("sources", [])]

    # ------------------------------------------------------------------
    # GUARDRAIL 3: Partisan response check (~100 tokens)
    # Runs after the chain so it can inspect the output.
    # Retries once with a stricter prompt if partisan content is found.
    # ------------------------------------------------------------------
    answer = await check_partisan_response(
        query=req.query,
        response=answer,
        chat_history=chat_history,
        chain=chain,
        ctx=ctx,
    )
    # ------------------------------------------------------------------
    # GUARDRAIL 4: PII in response (Presidio, zero tokens)
    # Runs last — catches PII that came from retrieved chunks or was
    # accidentally generated by the LLM. Returns safe fallback if found.
    # ------------------------------------------------------------------
    answer = detect_pii_in_response(answer, ctx)

    elapsed = time.time() - start
    log_request(req.user_id, req.query, answer, elapsed=elapsed)

    stats = callback.stats
    log_store.enqueue(ChatLogEntry(
        timestamp=start,
        user_id=req.user_id,
        session_turn=session_turn,
        query_len=len(req.query),
        query=req.query if LOG_QUERIES else None,
        rephrased_query=rephrased_query,
        query_category=ctx.query_category,
        guardrail_blocked_by=None,
        pii_in_query=False,
        retrieved_sources=result.get("sources", []),
        response=answer,
        response_len=len(answer),
        pii_in_response=ctx.pii_in_response,
        pii_type_in_response=ctx.pii_type_in_response,
        partisan_detected=ctx.partisan_detected,
        partisan_retried=ctx.partisan_retried,
        prompt_tokens=stats["prompt_tokens"],
        completion_tokens=stats["completion_tokens"],
        total_tokens=stats["total_tokens"],
        estimated_cost_usd=stats["estimated_cost_usd"],
        retrieval_latency_ms=stats["retrieval_latency_ms"],
        llm_latency_ms=stats["llm_latency_ms"],
        total_latency_ms=elapsed * 1000,
        model=stats["model"],
    ))

    store.add_exchange(req.user_id, req.query, answer)
    return ChatResponse(response=answer, sources=sources)


@app.post("/reset")
async def reset_session(req: ResetRequest):
    store.reset(req.user_id)
    return {"status": "session cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
        "index": config.PINECONE_INDEX_NAME,
    }