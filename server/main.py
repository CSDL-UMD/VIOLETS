"""
VIOLETS Election Chatbot — FastAPI server.

Endpoints:
    POST /chat          Send a message, get a response
    POST /reset         Clear conversation history for a user
    GET  /health        Health check

Run:
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .rag_chain import build_chain, to_langchain_messages
from .session import SessionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------

store: SessionStore | None = None
chain = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, store
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")

    logger.info(
        "Starting server — model=%s  index=%s  k=%d  session_ttl=%dm",
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
async def chat(req: ChatRequest):
    # Ensure session exists (creates if new, refreshes TTL if existing)
    store.get_or_create(req.user_id)
    chat_history = to_langchain_messages(store.get_history(req.user_id))

    try:
        result = await chain.ainvoke({
            "input": req.query,
            "chat_history": chat_history,
        })
    except Exception as exc:
        logger.error("Chain error for user %s: %s", req.user_id, exc)
        raise HTTPException(status_code=502, detail="Failed to generate response")

    # Chain returns a string directly (StrOutputParser)
    answer = result

    # Persist the exchange in session history
    store.add_exchange(req.user_id, req.query, answer)

    return ChatResponse(response=answer)


@app.post("/reset")
async def reset_session(req: ResetRequest):
    store.reset(req.user_id)
    return {"status": "session cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.LLM_MODEL, "index": config.PINECONE_INDEX_NAME}
