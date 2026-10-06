"""
Configuration for the Maryland Elections RAG crawler (Pass 1).
"""
import os

# --- Seed URLs (entry points for the crawl) ---
# Derived from the allowlist's exact URLs so the two can't drift apart: to add
# a page to the crawl, add it to ALLOWED_EXACT_URLS in exclusions.py.
from .exclusions import ALLOWED_EXACT_URLS  # noqa: E402

SEED_URLS = list(ALLOWED_EXACT_URLS)

# --- Domains ---
DOMAINS = ['elections.maryland.gov', 'mcg.montgomerycountymd.gov']

# --- HTTP ---
# Honest project User-Agent sent on ALL outbound requests (page GETs,
# document HEADs, robots.txt). Both target sites' WAFs reject the default
# python-requests / Python-urllib user-agents with 403.
USER_AGENT = (
    "VIOLETS-crawler/1.0 "
    "(University of Maryland research; contact: violets-project@example.edu)"
)

# --- Crawl Limits ---
MAX_DEPTH = 6
RATE_LIMIT_SECONDS = 0.75
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
REQUESTS_PER_MINUTE_WARN = 80  # log warning if exceeded

# --- Paths ---
# Project root is two levels up from this file (pass1/ -> maryland_rag/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.db")
RAW_HTML_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "crawl.log")

# --- Feature Flags ---
SAVE_RAW_HTML = False
RESPECT_ROBOTS_TXT = True

# --- Content Detection ---
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xls', '.xlsx', '.csv'}

# --- Trafilatura Fallback ---
# If trafilatura returns fewer words than this for a 200-OK HTML page,
# re-extract with BeautifulSoup as a fallback.
TRAFILATURA_MIN_WORDS = 50

# --- PDF Probe ---
# Number of bytes to download from a PDF to check if it's text-extractable.
PDF_PROBE_BYTES = 4096

# --- Social / Skip Domains ---
SKIP_DOMAINS = [
    'facebook.com', 'twitter.com', 'x.com', 'youtube.com',
    'google.com', 'instagram.com', 'linkedin.com', 'tiktok.com',
]
