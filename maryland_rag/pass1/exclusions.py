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
    # Current election cycle subtree — the past-years exclusion below
    # exempts 2026 via its (?!2026) lookahead, so the two rules compose
    "https://elections.maryland.gov/elections/2026/",
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
    # 2026 cycle bulk data published as the cycle progresses: hundreds of
    # per-district results detail pages and per-county sample-ballot books.
    # The /elections/2026/ allowlist keeps the cycle's informational pages;
    # these two subtrees are raw data, not chatbot content. 'general' is
    # included preemptively for the November publication wave.
    (r'/elections/2026/(primary|general)_results/',
     "2026 bulk results detail pages — raw data out of scope"),
    (r'/elections/2026/(primary|general)_ballots/',
     "2026 per-county sample ballot books — raw data out of scope"),
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
    # Language-suffix file naming for the same translated docs
    # (Maryland_Voting_Pocket_Guide-Es.pdf, English_Internet_VRA_KO.pdf,
    # ...-ES.docx). The token must sit between a -/_ separator and the
    # extension, so English filenames that merely end in these letters
    # (updates.pdf, notes.pdf, 02_Minutes.pdf) are never caught.
    # 'vi' and 'fr' are deliberately absent: 'VI' matches Roman-numeral
    # filenames (Title-VI.pdf) and 'FR' matches Final-Report-style
    # abbreviations; Vietnamese/French translations are still caught by the
    # full-language-name pattern above.
    (r'[-_](es|ko|zh|ru)\.(pdf|docx?|xlsx?)$',
     "non-English translated document (language-suffix filename) — "
     "English original is indexed"),
    # 2014–2016 wait-time / usability research reports: a decade old, ~28% of
    # the corpus as mostly context-free table fragments, diluting current
    # election info. Removed 2026-07-29. Separators match both literal
    # spaces (as stored in the manifest) and %20-encoded link variants;
    # "wait time" alone identifies the two wait-time PDFs (one filename
    # misspells "Observations").
    (r'/press_room/documents/[^/]*(schaefer(?:[ _+-]|%20)center'
     r'|wait(?:[ _+-]|%20)time'
     r'|onlineballot_usabilitytestresults)[^/]*\.pdf$',
     "2014-2016 wait-time/usability study — too old, out of scope"),
    (r'/voting_system/documents/expressvote[^/]*usability[^/]*\.pdf$',
     "2014-2016 wait-time/usability study — too old, out of scope"),
    # 2012 precinct register counts (PG12 = Presidential General, PP12 =
    # Presidential Primary): 52 per-county PDFs of 14-year-old raw
    # registration counts. Same class as the wait-time studies above —
    # stale raw data that answers "how many registered voters" questions
    # with 2012 numbers. Removed 2026-07-29.
    (r'/press_room/documents/p[gp]12/',
     "2012 precinct register counts — raw data, too old"),
    # The voter_registration copies of the same report family: bcp11 is the
    # 2011 Baltimore City primary, byprecinct.xls is the bulk spreadsheet
    # already dropped by the XLS row guardrail (excluding it here records
    # the decision instead of warning as zero-chunk every run).
    (r'/voter_registration/documents/precinctregistercounts_[^/]+$',
     "precinct register counts — raw data, too old"),
    # 2014 certification testing report for the prior ES&S EVS version:
    # superseded by the EVS 6.5.0.0 (2026) testing report in the Box corpus,
    # and its usability-survey tables extract as garbled header/fragment
    # text. Removed 2026-07-29.
    (r'/voting_system/documents/closed_certification(?:[ _+-]|%20)+testing'
     r'(?:[ _+-]|%20)+report[^/]*\.pdf$',
     "2014 ES&S EVS certification testing report — superseded by the "
     "EVS 6.5.0.0 (2026) report, garbled table extraction"),
    # Dead documents: the server soft-404s these (HTTP 200 + HTML site
    # template), so pass2's magic-byte guard refuses them and they warn as
    # zero-chunk every run. Confirmed removed from the site 2026-07-29.
    (r'/pdf/vrar/msr-\d{4}_\d{2}\.pdf$',
     "dead URL — monthly registration report removed from site (soft-404)"),
    (r'/get_involved/challenger(?:[ _+-]|%20)+watcher(?:[ _+-]|%20)summary'
     r'[^/]*\.pdf$',
     "dead URL — removed from site (soft-404)"),
    (r'/voter_registration/documents/english_internet_vra\.pdf$',
     "dead URL — removed from site (soft-404)"),
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
# `>= 500` check. 403 is here because both sites' WAFs intermittently
# challenge legitimate requests with a one-off 403: it gets the standard
# retry-with-backoff, and if it survives every retry the crawler's existing
# post-retry handling applies (preserve a previously-crawled row, else
# 'failed').
TRANSIENT_HTTP_STATUSES = {403, 408, 429}

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

    return matches_exclusion_rules(url)


def matches_exclusion_rules(url: str) -> tuple[bool, str | None]:
    """
    Layer 2 ONLY: exclusion path patterns + extension/scheme checks, without
    the Layer-1 allowlist gate.

    Used by the crawler's retroactive exclusion pass over EXISTING manifest
    rows: those rows already passed whatever scope decision was in force
    when they were crawled (including the keep_filter_2026 grandfathering of
    document rows that sit outside the link-follow allowlist), so re-running
    the allowlist gate against them would wrongly flip deliberately-kept
    content. New junk/translation rules land in this layer and DO need to
    apply retroactively.
    """
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
