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

# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")

# ---------------------------------------------------------------------------
# Configurable settings
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "maryland-elections")

# LLM — change via LLM_MODEL env var to swap models without code changes
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Retrieval
RETRIEVER_K: int = int(os.environ.get("RETRIEVER_K", "5"))

# Session management
SESSION_TTL_MINUTES: int = int(os.environ.get("SESSION_TTL_MINUTES", "30"))
MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "20"))
