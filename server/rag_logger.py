"""
RAG logging — LLM prompts, responses, token usage, cost estimates,
per-step latency, and request-level history.

Changes made to main.py
-----------------------
1. Added `import time` at the top — needed to measure how long each
   request takes end-to-end.

2. Added to the imports:
       from .rag_logger import RAGCallbackHandler, log_request

3. Changed the chain construction in lifespan() from:
       chain = build_chain()
   to:
       chain = build_chain().with_config({"callbacks": [RAGCallbackHandler()]})

   build_chain() is unchanged. .with_config() attaches RAGCallbackHandler
   to the already-built chain so LangChain fires its logging events
   automatically on every call. RAGCallbackHandler is defined in this file.

4. In the /chat endpoint, added a start timestamp before the chain call
   and a log_request() call immediately after:
       start = time.time()
       result = await chain.ainvoke({...})
       answer = result
       log_request(req.user_id, req.query, answer, elapsed=time.time() - start)

   log_request() is defined in this file. It writes one log line per
   request capturing the user, their query, response size, and total time.
   This is separate from the callback handler because the handler only
   sees what happens inside the chain — it has no visibility into which
   user triggered the request or the total server-side time.

Coverage note
-------------
RAGCallbackHandler covers the final QA step fully: full prompt, full
response, token counts, cost, and latency.

The rephrase step and retriever call inside contextualize_and_retrieve()
in rag_chain.py are invoked via plain synchronous .invoke() calls with
no config passed through, so the callback system cannot reach them
automatically. Their logging is handled by the existing manual
logger.info() calls already in rag_chain.py. Full callback coverage of
those steps would require making contextualize_and_retrieve() async and
using .ainvoke() throughout.

TEST
--------------
To test this, create a virtual environment
    python -m venv venv (one time)
    venv\Scripts\activate (RUN THIS everytime you want to start the venv)
    pip install -r maryland_rag/requirements.txt (one time)
    pip install -r server/requirements.txt (one time)

Then start a local server
    uvicorn server.main:app --host 0.0.0.0 --port 8000

In Git Bash:
    curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"user_id": "test", "query": "When is the voter registration deadline?"}'

And check the uvicorn terminal to see the logs

"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging toggles
# Set to False before deploying to production to avoid writing PII to logs.
# Token counts, cost, and latency are always logged regardless of these flags.
# ---------------------------------------------------------------------------

LOG_PROMPTS: bool = True   # log full prompt sent to the LLM
LOG_RESPONSES: bool = True  # log full response from the LLM
LOG_QUERIES: bool = True    # log raw user query in log_request()

# ---------------------------------------------------------------------------
# Cost table (USD per million tokens)
# Update if OpenAI changes pricing or if you swap models.
# ---------------------------------------------------------------------------

_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":          (0.15,  0.60),
    "gpt-4o":               (5.00, 15.00),
    "gpt-4-turbo":          (10.0, 30.00),
    "gpt-3.5-turbo":        (0.50,  1.50),
}
_COST_FALLBACK = (0.0, 0.0)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Exact match first, then prefix match to handle versioned names like
    # "gpt-4o-mini-2024-07-18" mapping to "gpt-4o-mini".
    rates = _COST_TABLE.get(model)
    if rates is None:
        rates = next(
            (v for k, v in _COST_TABLE.items() if model.startswith(k)),
            _COST_FALLBACK,
        )
    input_rate, output_rate = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

class RAGCallbackHandler(BaseCallbackHandler):
    """
    Logs every LLM call the chain makes:
      - Full prompt sent to the model
      - Full model response
      - Token counts (prompt / completion / total)
      - Estimated cost in USD
      - Wall-clock latency

    Also logs retriever start/end when the retriever is invoked as a
    direct chain step (not from inside a RunnableLambda — see module
    docstring for the coverage limitation).
    """

    def __init__(self):
        self._llm_starts: dict[str, tuple[float, str]] = {}  # run_id → (start_time, model)
        self._ret_starts: dict[str, float] = {}               # run_id → start_time

    # ---- LLM events ----

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "unknown")
        )
        key = str(run_id)
        self._llm_starts[key] = (time.time(), model)

        if LOG_PROMPTS:
            for i, message_group in enumerate(messages):
                lines = []
                for msg in message_group:
                    role = msg.__class__.__name__.replace("Message", "").lower()
                    lines.append(f"[{role}]: {msg.content}")
                logger.info(
                    "LLM PROMPT [run=%s model=%s]:\n%s",
                    key[:8], model, "\n".join(lines),
                )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "unknown")
        )
        key = str(run_id)
        self._llm_starts[key] = (time.time(), model)

        if LOG_PROMPTS:
            for i, prompt in enumerate(prompts):
                logger.info(
                    "LLM PROMPT [run=%s model=%s]:\n%s",
                    key[:8], model, prompt,
                )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        start, model = self._llm_starts.pop(key, (time.time(), "unknown"))
        elapsed = time.time() - start

        if LOG_RESPONSES:
            for gen_list in response.generations:
                for gen in gen_list:
                    text = getattr(gen, "text", str(gen))
                    logger.info(
                        "LLM RESPONSE [run=%s] (%.2fs):\n%s",
                        key[:8], elapsed, text,
                    )

        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage", {})
        if usage:
            # Prefer the model name from the response — OpenAI always populates
            # llm_output["model_name"] with the exact model string used, which is
            # more reliable than what we extracted from the serialized dict at start.
            model = llm_output.get("model_name") or model
            prompt_tokens     = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
            cost              = _estimate_cost(model, prompt_tokens, completion_tokens)
            logger.info(
                "TOKENS [run=%s model=%s]: prompt=%d completion=%d total=%d cost=~$%.5f",
                key[:8], model, prompt_tokens, completion_tokens, total_tokens, cost,
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._llm_starts.pop(key, None)
        logger.error("LLM ERROR [run=%s]: %s", key[:8], error)

    # ---- Retriever events ----

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._ret_starts[key] = time.time()
        logger.info("RETRIEVER START [run=%s]: %r", key[:8], query)

    def on_retriever_end(
        self,
        documents: list[Document],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        elapsed = time.time() - self._ret_starts.pop(key, time.time())
        logger.info(
            "RETRIEVER END [run=%s] (%.2fs): %d docs",
            key[:8], elapsed, len(documents),
        )


# ---------------------------------------------------------------------------
# Request-level logging
# ---------------------------------------------------------------------------

def log_request(
    user_id: str,
    query: str,
    response: str,
    elapsed: float | None = None,
) -> None:
    """
    Log a completed /chat exchange at the request level.

    Captures: user_id, the query (truncated for log readability),
    response length in characters, and total wall-clock time if provided.

    Call this in main.py after chain.ainvoke() returns, e.g.:

        start = time.time()
        result = await chain.ainvoke({...})
        log_request(req.user_id, req.query, result, elapsed=time.time() - start)
    """
    timing = f" elapsed={elapsed:.2f}s" if elapsed is not None else ""
    query_field = repr(query[:120]) if LOG_QUERIES else f"len={len(query)}"
    logger.info(
        "REQUEST user=%s query=%s response_chars=%d%s",
        user_id,
        query_field,
        len(response),
        timing,
    )


