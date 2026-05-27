"""
Configuration for the Maryland Elections RAG crawler (Pass 1).
"""
import os

# --- Seed URLs (entry points for the crawl) ---
SEED_URLS = [
    # State BoE — prefix-crawled paths
    "https://elections.maryland.gov/voting/index.html",
    "https://elections.maryland.gov/voter_registration/index.html",
    # State BoE — exact pages only
    "https://elections.maryland.gov/about/election_security.html",
    "https://elections.maryland.gov/press_room/index.html",
    "https://elections.maryland.gov/press_room/rumor_control.html",
    "https://elections.maryland.gov/elections/2026/index.html",
    # MoCo — exact pages only (child links not followed)
    "https://mcg.montgomerycountymd.gov/elections/dropbox.html",
    "https://mcg.montgomerycountymd.gov/elections/ElectionJudge/Overview.html",
    "https://mcg.montgomerycountymd.gov/Elections/ElectionJudge/ImportantDates.html",
    "https://mcg.montgomerycountymd.gov/Elections/FutureVote/school-poll-workers.html",
    "https://mcg.montgomerycountymd.gov/Elections/FrequentlyAskedQuestions/FAQsElectionWorker.html",
    "https://mcg.montgomerycountymd.gov/Elections/FrequentlyAskedQuestions/future-vote-faqs.html",
    "https://mcg.montgomerycountymd.gov/Elections/FrequentlyAskedQuestions/electionworker-faqs.html",
    "https://mcg.montgomerycountymd.gov/Elections/FrequentlyAskedQuestions/voter-registration-faqs.html",
    "https://mcg.montgomerycountymd.gov/elections/vote-by-mail.html",
    "https://mcg.montgomerycountymd.gov/Elections/Accessibility/voting-assistance.html",
    "https://mcg.montgomerycountymd.gov/Elections/EarlyVoting/EarlyVotingCenters.html",
]

# --- Domains ---
DOMAINS = ['elections.maryland.gov', 'mcg.montgomerycountymd.gov']

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
