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

OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1")

# ---------------------------------------------------------------------------
# Configurable settings
# ---------------------------------------------------------------------------

# LLM — change via LLM_MODEL env var to swap models without code changes.
# GPT-5 family does not accept `temperature` or other sampling params, so we
# no longer expose a temperature knob; if you swap to a sampling-class model
# (e.g. gpt-4o-mini), pass temperature explicitly at the call site.
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-5-nano")

# Reasoning effort for the RAG chain LLM (rephrase/QA/conversational/concerns).
# "low" cut QA generation from ~26s to ~6s vs "medium" with identical visible
# answers in a 45-call benchmark (2026-07-29). The classifier and partisan
# checker in middleware.py pin their own efforts and are not affected.
RAG_REASONING_EFFORT: str = os.environ.get("RAG_REASONING_EFFORT", "low")

# Election context — injected into the QA/concerns prompts so the model
# anchors deadlines and dates to the correct election.
ELECTION_NAME: str = os.environ.get(
    "ELECTION_NAME", "2026 Maryland Gubernatorial General Election"
)
ELECTION_DATE: str = os.environ.get("ELECTION_DATE", "November 3, 2026")

# Retrieval
RETRIEVER_K: int = int(os.environ.get("RETRIEVER_K", "5"))
# Minimum similarity score (1 - cosine distance) a retrieved chunk must meet.
# 0.0 (default) disables the floor and keeps every retrieved chunk.
SIMILARITY_FLOOR: float = float(os.environ.get("SIMILARITY_FLOOR", "0.0"))
# Down-weight chunks from past-election documents so they lose close calls to
# current-election ones but still surface when clearly the best match (e.g.
# "who ran in the primary?"). PAST_ELECTION_PENALTY is subtracted from the
# similarity score for ranking only — the floor and logged score stay raw.
# Patterns are comma-separated source_url substrings. 0.02 tuned on
# server/eval_retrieval.py (2026-10-06): fixes current-candidate questions with
# no general regressions; >=0.04 starts hiding primary answers entirely.
PAST_ELECTION_URL_PATTERNS: list[str] = [
    p.strip() for p in os.environ.get(
        "PAST_ELECTION_URL_PATTERNS", "/primary_candidates/"
    ).split(",") if p.strip()
]
PAST_ELECTION_PENALTY: float = float(os.environ.get("PAST_ELECTION_PENALTY", "0.02"))

# Hardcoded candidates redirect URL (middleware.py). Defaults to the 2026
# primary candidates page — the crawl contains no general-election candidates
# page yet, so set CANDIDATES_URL once the State Board publishes one.
CANDIDATES_URL: str = os.environ.get(
    "CANDIDATES_URL",
    "https://elections.maryland.gov/elections/2026/primary_candidates/index.html",
)

# Session management
SESSION_TTL_MINUTES: int = int(os.environ.get("SESSION_TTL_MINUTES", "30"))
MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

# Rate limiting
RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
# Global backstop across all users combined — the per-user limit is keyed on
# the client-supplied user_id and can be bypassed by rotating ids.
RATE_LIMIT_GLOBAL_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_GLOBAL_PER_MINUTE", "225"))
