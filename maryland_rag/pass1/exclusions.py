"""
Single source of truth for all URL exclusion logic.
No exclusion checks should exist in any other module.
"""
import re
from urllib.parse import urlparse

# ---- Path pattern exclusions (regex, reason) ----
EXCLUDED_PATH_PATTERNS = [
    (r'/elections/\d{4}/',                          "Past election results - year folder"),
    (r'/elections/\d{4}_special/',                   "Past special election results"),
    (r'/elections/special_elections\.html',           "Legacy special elections page"),
    (r'/elections/presidential',                     "Historical presidential data"),
    (r'/elections/baltimore/',                        "Legacy Baltimore city pages"),
    (r'/press_room/prior_releases',                  "Press releases older than 5 years"),
    (r'/petitions/',                                 "Petition procedures - out of scope"),
    (r'/election_data/',                             "Raw election results data"),
    (r'/elections/using_election_data',              "Election data usage docs"),
    (r'/campaign_finance/',                          "Campaign finance - out of scope"),
    (r'/voting_system/ballot_audit_plan_.*\.html',   "Past audit results"),
]

# ---- Exact URL exclusions ----
EXCLUDED_EXACT_URLS = {
    "https://elections.maryland.gov/press_room/index.html",
}

# ---- HTTP statuses that mean the page is not usable ----
EXCLUDED_HTTP_STATUSES = {404, 410, 403, 500, 502, 503}

# ---- File extensions to skip entirely (not documents, just junk) ----
SKIP_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.bmp', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.wmv',
    '.zip', '.tar', '.gz', '.rar',
    '.css', '.js', '.json', '.xml', '.rss',
    '.woff', '.woff2', '.ttf', '.eot',
}

# Pre-compile patterns for performance
_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in EXCLUDED_PATH_PATTERNS
]


def should_exclude(url: str) -> tuple[bool, str | None]:
    """
    Check whether a URL should be excluded from crawling.
    Returns (True, reason) if excluded, (False, None) if allowed.
    """
    if not url:
        return True, "Empty URL"

    if url in EXCLUDED_EXACT_URLS:
        return True, f"Exact exclusion: {url}"

    parsed = urlparse(url)
    path = parsed.path.lower()

    # Skip non-content file extensions (images, media, archives, assets)
    ext = ''
    if '.' in path.split('/')[-1]:
        ext = '.' + path.rsplit('.', 1)[-1]
    if ext in SKIP_EXTENSIONS:
        return True, f"Non-content file extension: {ext}"

    # Check path patterns
    for pattern, reason in _COMPILED_PATTERNS:
        if pattern.search(parsed.path):
            return True, reason

    # Skip mailto and tel links
    if parsed.scheme in ('mailto', 'tel', 'javascript'):
        return True, f"Non-HTTP scheme: {parsed.scheme}"

    return False, None


def is_excluded_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates an unusable page."""
    return status_code in EXCLUDED_HTTP_STATUSES
