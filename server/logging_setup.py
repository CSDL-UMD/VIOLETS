"""
server/logging_setup.py
=======================
Centralized logging configuration for the VIOLETS server.

Installs a *persisted* rotating file handler plus a console handler on the root
logger, both at LOG_LEVEL (default INFO). Call ``setup_logging()`` once at
process start — it is idempotent, so calling it from both the module entrypoint
and the FastAPI lifespan is safe.

What lands in the file at the default INFO level:
    - startup / "Server ready" / heartbeat summaries
    - one line per request outcome (ok / blocked / error)
    - auth (401) and rate-limit (429) events
    - warnings and errors (chain failures carry a full traceback)

Routine per-call detail (token counts, retriever timings, prompts) is logged at
DEBUG and suppressed at INFO. Set ``LOG_LEVEL=DEBUG`` to surface it when
diagnosing a specific request.

Environment:
    LOG_LEVEL         DEBUG | INFO | WARNING | ...   (default INFO)
    SERVER_LOG_FILE   path to the log file           (default <repo>/logs/server.log)
"""

import contextvars
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False

# Per-request correlation id. Set once at the top of /chat; every log record
# emitted while handling that request (including middleware, rag_chain, and the
# callback handler) inherits it via the filter below, so an ERROR line can be
# traced back to its REQUEST summary. Defaults to "-" for non-request logs
# (startup, heartbeat). ContextVars are per-asyncio-Task and are copied into
# asyncio.to_thread workers, so the value follows the request across both.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    """Generate a short request id and bind it to the current context."""
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    return rid


class _RequestIdFilter(logging.Filter):
    """Inject the current request id onto every record so the formatter can show it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """Configure root logging with a rotating file + console handler. Idempotent."""
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = os.environ.get(
        "SERVER_LOG_FILE",
        str(Path(__file__).resolve().parent.parent / "logs" / "server.log"),
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
    )
    id_filter = _RequestIdFilter()

    # 10 MB per file, keep 5 rotations (~50 MB ceiling) so the log can't grow
    # unbounded on a long-running server.
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(id_filter)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(id_filter)

    root = logging.getLogger()
    root.setLevel(level)
    # Drop any handlers a prior basicConfig() may have installed so we don't
    # double-log every line to stdout.
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)

    # Uvicorn's access log emits one line per HTTP request, which duplicates our
    # own per-request REQUEST summary. Raise its threshold so that routine noise
    # stays out of the file; uvicorn.error (startup/shutdown/errors) still
    # propagates to root and is kept.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Third-party HTTP clients log one INFO line per outbound call. With ~4
    # OpenAI/embeddings calls per request that would bury our own logs — keep
    # them at WARNING so only real transport failures surface.
    for _noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging configured — level=%s file=%s", level_name, log_file
    )
