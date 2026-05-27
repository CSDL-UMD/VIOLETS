"""Shared path constants for the box_ingest package."""
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent.parent
NEEDTOCHUNK_DIR = PROJECT_ROOT / "needtochunk"
MANIFEST_PATH   = NEEDTOCHUNK_DIR / "url_manifest.json"
REVIEW_LOG      = NEEDTOCHUNK_DIR / "review_files.txt"
STATE_PATH      = PROJECT_ROOT / "data" / "box_ingest.state.json"
DEFAULT_OUTPUT  = PROJECT_ROOT / "data" / "box_chunks.jsonl"
TOKEN_CACHE     = PROJECT_ROOT / ".box_token"
