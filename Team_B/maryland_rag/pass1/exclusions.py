"""
Single source of truth for all URL exclusion logic.
No exclusion checks should exist in any other module.

Patterns are specific to montgomerycountymd.gov/elections.
All regex patterns use (?i) for case-insensitive matching because
the site uses mixed-case paths (/Elections/ and /elections/ both exist).
"""
import re
from urllib.parse import urlparse

# ---- Path pattern exclusions (regex, reason) ----
EXCLUDED_PATH_PATTERNS = [
    # Boundary guard: only crawl /elections/ paths on the county site.
    # Without this, the crawler would follow links to parks, permits,
    # police, and every other county department.
    (r'(?i)^/(?!elections)',                                "Non-elections county path"),

    # Past election result archives — historical data with no current
    # relevance to voters. Includes year-based folders and the
    # dedicated past results section.
    (r'(?i)/elections/pastelections/',                      "Past election results section"),
    (r'(?i)/elections/\d{4}/',                              "Past election year folder"),

    (r'(?i)www3\.montgomerycountymd\.gov', "County 311 services - not elections"),
    

    # External state elections site — Montgomery County pages link to it
    # but it is out of scope for this crawl.
    (r'(?i)elections\.maryland\.gov',          "Maryland state elections site - out of scope"),
    (r'(?i)voterservices\.elections\.maryland', "Maryland state voter services - out of scope"),

    # Precinct map PDFs — large collection of boundary map files.
    # These are binary map documents with no extractable text value.
    (r'(?i)/elections/resources/files/pdfs/maps/',          "Precinct boundary map PDFs"),

    # Early voting PDFs from past elections — outdated location and
    # schedule info that no longer reflects current voting centers.
    (r'(?i)/elections/resources/files/pdfs/earlyvoting/20', "Past early voting PDFs"),

    # Images folder — photos of voting center buildings.
    # No text content, not useful for RAG ingestion.
    (r'(?i)/elections/resources/images/',                   "Site images - no text content"),

    # Election maps index pages — these only link to the precinct
    # map PDFs already excluded above, so the pages themselves
    # have no standalone value.
    (r'(?i)/elections/electionmaps/',                       "Election maps index pages"),
]

# ---- Exact URL exclusions ----
# Add specific one-off URLs here that don't fit a pattern rule.
EXCLUDED_EXACT_URLS = set()

# ---- HTTP statuses that mean the page is not usable ----
EXCLUDED_HTTP_STATUSES = {404, 410, 403, 500, 502, 503}

# ---- File extensions to skip entirely (images, media, assets) ----
# Note: .pdf, .docx, .xls etc. are NOT listed here — those are valid
# documents that Pass 2 will extract and chunk.
SKIP_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.bmp', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.wmv',
    '.zip', '.tar', '.gz', '.rar',
    '.css', '.js', '.json', '.xml', '.rss',
    '.woff', '.woff2', '.ttf', '.eot',
}

# Pre-compile all patterns once at import time for performance
_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in EXCLUDED_PATH_PATTERNS
]

def should_exclude(url: str) -> tuple[bool, str | None]:
    if not url:
        return True, "Empty URL"

    if url in EXCLUDED_EXACT_URLS:
        return True, f"Exact exclusion: {url}"

    parsed = urlparse(url)
    path = parsed.path

    # Block bare domain URLs with no path — these are never
    # elections-specific pages, just county homepages or subdomains
    if path == '' or path == '/':
        return True, "Bare domain URL - no elections path"

    # Skip non-content file extensions
    ext = ''
    if '.' in path.split('/')[-1]:
        ext = '.' + path.rsplit('.', 1)[-1]
    if ext in SKIP_EXTENSIONS:
        return True, f"Non-content file extension: {ext}"

    # Check patterns against FULL URL (not just path)
    # This is required for domain-level exclusions like elections.maryland.gov
    for pattern, reason in _COMPILED_PATTERNS:
        if pattern.search(url) or pattern.search(parsed.path):
            return True, reason

    if parsed.scheme in ('mailto', 'tel', 'javascript'):
        return True, f"Non-HTTP scheme: {parsed.scheme}"

    return False, None


def is_excluded_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates an unusable page."""
    return status_code in EXCLUDED_HTTP_STATUSES