"""
Configuration for the Maryland Elections RAG crawler (Pass 1).
"""
import os

# --- Seed & Domain ---
SEED_URL = "https://montgomerycountymd.gov/elections"
DOMAIN = "montgomerycountymd.gov"

# --- Crawl Limits ---
MAX_DEPTH = 6
RATE_LIMIT_SECONDS = 0.75
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
REQUESTS_PER_MINUTE_WARN = 80  # log warning if exceeded

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "manifest.db")
RAW_HTML_DIR = os.path.join(BASE_DIR, "data", "raw")
LOG_DIR = os.path.join(BASE_DIR, "logs")
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
