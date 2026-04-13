"""
Server configuration — loads from .env and environment variables.

Searches for .env in: server/, project root.
All settings are overridable via env vars.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------

_server_dir = Path(__file__).resolve().parent
_search_paths = [
    _server_dir / ".env",
    _server_dir.parent / ".env",
]

for _path in _search_paths:
    if _path.exists():
        load_dotenv(_path)
        break


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            "Add it to your .env file or set it in your shell."
        )
    return val


# ---------------------------------------------------------------------------
# Required keys — validated at import time so missing keys surface
# immediately, before middleware LLMs or the RAG chain are constructed.
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = _require_env("OPENAI_API_KEY")
DATABASE_URL: str = _require_env("DATABASE_URL")
VIOLETS_API_KEY: str = _require_env("VIOLETS_API_KEY")

OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ---------------------------------------------------------------------------
# Configurable settings
# ---------------------------------------------------------------------------

# LLM — change via LLM_MODEL env var to swap models without code changes
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Retrieval
RETRIEVER_K: int = int(os.environ.get("RETRIEVER_K", "5"))

# Session management
SESSION_TTL_MINUTES: int = int(os.environ.get("SESSION_TTL_MINUTES", "30"))
MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

# Rate limiting
RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
