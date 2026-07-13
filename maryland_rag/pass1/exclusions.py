"""
Single source of truth for all URL allowlist and exclusion logic.
No URL filtering checks should exist in any other module.

Two-layer filtering:
  1. Allowlist  — is this URL in scope at all?  (domain + path gates)
  2. Exclusions — within scope, is this specific pattern still junk?
Both layers must pass for a URL to be crawled.
"""
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Layer 1 — Allowlist
# ---------------------------------------------------------------------------

# Prefix rules: all child paths under these are in scope
ALLOWED_URL_PREFIXES = [
    "https://elections.maryland.gov/voting/",
    "https://elections.maryland.gov/voter_registration/",
    # 2026 press releases only (documents published under this path)
    "https://elections.maryland.gov/press_room/documents/2026/",
]

# Exact URLs that are in scope
ALLOWED_EXACT_URLS = {
    # State BoE exact pages
    "https://elections.maryland.gov/voting/index.html",
    "https://elections.maryland.gov/voter_registration/index.html",
    "https://elections.maryland.gov/about/election_security.html",
    # press_room/index.html: crawled for 2026 press releases + statistics
    "https://elections.maryland.gov/press_room/index.html",
    "https://elections.maryland.gov/press_room/rumor_control.html",
    "https://elections.maryland.gov/elections/2026/index.html",
    # MoCo exact pages (child links are not followed)
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
}

# Lowercase copies for case-insensitive matching
_ALLOWED_EXACT_LOWER = {u.lower() for u in ALLOWED_EXACT_URLS}
_ALLOWED_PREFIXES_LOWER = [p.lower() for p in ALLOWED_URL_PREFIXES]

# ---------------------------------------------------------------------------
# Layer 2 — Exclusions (applied within the allowlisted scope)
# ---------------------------------------------------------------------------

# Path pattern exclusions — catches junk pages that can be discovered as
# child links within an allowlisted prefix (e.g., /voting/* may link out to
# old election results or campaign finance pages).
EXCLUDED_PATH_PATTERNS = [
    # Past election year folders — exempt 2026 which is explicitly allowlisted
    (r'/elections/(?!2026)\d{4}/',          "Past election results folder"),
    (r'/elections/\d{4}_special/',           "Past special election results"),
    (r'/elections/special_elections\.html',  "Legacy special elections page"),
    (r'/elections/presidential',             "Historical presidential data"),
    (r'/elections/baltimore/',               "Legacy Baltimore city pages"),
    (r'/press_room/prior_releases',          "Press releases older than current cycle"),
    (r'/petitions/',                         "Petition procedures — out of scope"),
    (r'/election_data/',                     "Raw election results data"),
    (r'/elections/using_election_data',      "Election data usage docs"),
    (r'/campaign_finance/',                  "Campaign finance — out of scope"),
    (r'/voting_system/ballot_audit_plan_.*\.html', "Past audit plan archives"),
    # Translated forms (VRA-Amharic.pdf, Mail_In_Ballot_Application-Korean.pdf,
    # HowMDVotes_Spanish.pdf, ...): the chatbot serves English, the English
    # originals are indexed separately, and PDF extraction interleaves the
    # bilingual layouts into unreadable mixed-script text. pass2 additionally
    # drops any mixed-script chunk via langfilter as a content-based net.
    (r'[-_/](amharic|korean|vietnamese|spanish|french|russian|tagalog|urdu|'
     r'farsi|portuguese|haitian[-_]?creole|(simplified|traditional)[-_]?chinese'
     r'|chinese)(?![a-z])[^/]*\.(pdf|docx?|xlsx?)$',
     "non-English translated document — English original is indexed"),
]

# Pre-compile patterns for performance
_COMPILED_EXCLUSIONS = [
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in EXCLUDED_PATH_PATTERNS
]

# ---------------------------------------------------------------------------
# File extension and HTTP status helpers
# ---------------------------------------------------------------------------

EXCLUDED_HTTP_STATUSES = {404, 410, 403, 500, 502, 503}

# Statuses that are retryable and must never be persisted as page content or
# written to the Pass 2 cache (the response body is an error page, not the
# document). The crawler's retry loop and extractor's cache write both consult
# this set — keep them in sync through it. 5xx is handled separately via a
# `>= 500` check.
TRANSIENT_HTTP_STATUSES = {408, 429}

SKIP_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.bmp', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.wmv',
    '.zip', '.tar', '.gz', '.rar',
    '.css', '.js', '.json', '.xml', '.rss',
    '.woff', '.woff2', '.ttf', '.eot',
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_allowlisted(url: str) -> bool:
    """Return True if the URL is permitted by the allowlist."""
    url_lower = url.lower()
    if url_lower in _ALLOWED_EXACT_LOWER:
        return True
    return any(url_lower.startswith(p) for p in _ALLOWED_PREFIXES_LOWER)


def should_exclude(url: str) -> tuple[bool, str | None]:
    """
    Check whether a URL should be excluded from crawling.
    Returns (True, reason) if excluded, (False, None) if allowed.

    Order: allowlist gate → exclusion patterns → extension/scheme checks.
    """
    if not url:
        return True, "Empty URL"

    # Layer 1: allowlist gate
    if not is_allowlisted(url):
        return True, "Not in allowlist"

    parsed = urlparse(url)
    path = parsed.path

    # Layer 2: exclusion path patterns (junk within allowed scope)
    for pattern, reason in _COMPILED_EXCLUSIONS:
        if pattern.search(path):
            return True, reason

    # Skip non-content file extensions (images, media, archives, assets)
    path_lower = path.lower()
    ext = ''
    if '.' in path_lower.split('/')[-1]:
        ext = '.' + path_lower.rsplit('.', 1)[-1]
    if ext in SKIP_EXTENSIONS:
        return True, f"Non-content file extension: {ext}"

    # Legacy Word documents: python-docx can never parse pre-2007 OLE .doc
    # files, so they'd silently yield zero chunks every run. Skip them
    # visibly instead (reason surfaces in the audit report).
    if ext == '.doc':
        return True, "legacy .doc format unsupported — convert to PDF to ingest"

    # Skip non-HTTP schemes
    if parsed.scheme in ('mailto', 'tel', 'javascript'):
        return True, f"Non-HTTP scheme: {parsed.scheme}"

    return False, None


def is_excluded_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates an unusable page."""
    return status_code in EXCLUDED_HTTP_STATUSES
