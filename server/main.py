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
from .rag_logger import RAGCallbackHandler, log_request
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


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, store

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set — check your .env file.")
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set — check your .env file.")

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

    logger.info("Server ready.")
    yield


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


class ChatResponse(BaseModel):
    response: str


class ResetRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    store.get_or_create(req.user_id)
    chat_history = to_langchain_messages(store.get_history(req.user_id))

    # Shared context object — travels through all guardrails
    ctx = QueryContext(user_id=req.user_id)

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
    classification_response = classify_query(req.query, ctx)
    if classification_response:
        logger.info(
            "Request blocked — query not in scope [user=%s category=%s]",
            req.user_id,
            ctx.query_category,
        )
        return ChatResponse(response=classification_response)

    # ------------------------------------------------------------------
    # MAIN RAG CHAIN
    # Only reached if both guardrails above passed.
    # ------------------------------------------------------------------
    start = time.time()
    try:
        result = chain.with_config({"callbacks": [RAGCallbackHandler()]}).invoke({
            "input": req.query,
            "chat_history": chat_history,
      })
    except Exception as exc:
        logger.error("RAG chain error [user=%s]: %s", req.user_id, exc)
        raise HTTPException(status_code=502, detail="Failed to generate response.")

    answer = str(result)

    # ------------------------------------------------------------------
    # GUARDRAIL 3: Partisan response check (~100 tokens)
    # Runs after the chain so it can inspect the output.
    # Retries once with a stricter prompt if partisan content is found.
    # ------------------------------------------------------------------
    answer = check_partisan_response(
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

    log_request(req.user_id, req.query, answer, elapsed=time.time() - start)
    store.add_exchange(req.user_id, req.query, answer)

    return ChatResponse(response=answer)


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