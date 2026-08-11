"""
Shared classification rules for crawl-time (pass1/classifier.py) and
post-hoc DB reclassification (scripts/reclassify.py).

Single source of truth for signal lists, banner stripping, and the
HTML classification decision tree. Crawl-time also passes raw_html so
structural patterns (FAQ accordions, large tables) can serve as a
medium-confidence fallback when keyword/path signals miss.
"""
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup


# Site-wide announcement banners sometimes appear at the top of extracted
# text. Strip them so they don't pollute signal matching downstream.
BANNER_MARKERS = [
    "The Worcester County Board of Elections",
    "The Baltimore County Board of Elections",
    "The Montgomery County Board of Elections",
]

# FAQ keyword variants checked against url + title ONLY. Never matched
# against page text: the MoCo sidebar contains 'Frequently Asked Questions'
# on nearly every page, which misrouted unrelated pages to qa_pairs.
FAQ_SIGNALS = [
    'faq', 'frequently-asked', 'frequently asked',
    'q&a', 'q & a',
    'questions and answers', 'questions-and-answers',
]

# Known FAQ pages whose URL/title lack faq keywords. Checked against the URL
# path only so snippets that mention these topics in passing don't false-match.
FAQ_URL_PATHS = [
    'early_voting', 'absentee', 'election_day_questions',
    'learn_about_the_new_voting_system', 'redistricting',
    '/about/pia',
    # Lives under /press_room/ but is a Rumor/Fact accordion, not a news
    # item. PRESS_SIGNALS matched it first and routed it to simple_split,
    # which severed Rumor lines from the Fact answering them. The FAQ rule
    # runs before the press rule, so listing it here wins.
    'rumor_control',
    # A 350-word Q/A accordion ("What is a party primary election?"). Under
    # the word-count threshold for prose and listed in SHORT_STATIC_SIGNALS,
    # so it was collapsed into one blob before rule 9's structural FAQ
    # detection ever ran.
    '/voting/primary',
]

# Press-release keyword variants checked against url + title + text.
PRESS_SIGNALS = [
    'press_room', 'press-room',
    'press_release', 'press-release',
    'news-release', 'announcement', 'media-release',
    'rumor_control', 'dis-misinformation',
]

# URL path substrings that indicate an actual online form (not a page that
# happens to mention forms).
FORM_URL_PATHS = [
    '/forms/',
    'data_form',
    'schedule_appointment',
    'purchase_lists',
]

# URL path substrings for table/data pages. Long word counts here are tabular,
# not prose, so this check runs before the prose word-count fallback.
TABLE_SIGNALS = [
    'municipal_results', 'election_results', 'results_archive',
    '/elections/districts', '/elections/archive', '/elections/printed_copies',
    'voter_registration/archive', 'voter_registration/stats',
    '/voting/recount',
]

# URL path prefixes for known short navigational/static pages.
SHORT_STATIC_SIGNALS = [
    '/about/contact', '/about/directions', '/about/social_media',
    '/about/county_boards', '/about/state-links', '/about/federal-links',
    '/about/board', '/about/feedback',
    '/voting_system/voting_equipment', '/voting_system/how_to_vote',
    '/voting_system/ballot_audit_plan',
    '/laws_and_regs/sbe_policy', '/laws_and_regs/index',
    '/candidacy/qualifications', '/candidacy/ballot', '/candidacy/candidate_filing',
    '/get_involved/election_judges', '/get_involved/students', '/get_involved/index',
    '/get_involved/dis-misinformation',
    '/overseas_voters/other_information', '/overseas_voters/index',
    '/press_room/dis-misinformation', '/press_room/dis',
    '/voter_registration/nvra', '/voter_registration/data_form',
    '/voter_registration/archive_bydistricts',
    '/accessibility', '/privacy',
    '/voting/address', '/voting/primary',
    '/voter_services/',
    '/voting_system/procurement',
    '/elections/special_elections_past', '/elections/electoral_college',
]


def strip_banner(text: str | None) -> str:
    if not text:
        return ""
    for marker in BANNER_MARKERS:
        if text.startswith(marker):
            newline_pos = text.find("\n", len(marker))
            if newline_pos != -1:
                return text[newline_pos:].lstrip()
    return text


def classify_html(
    url: str,
    title: str | None,
    text: str | None,
    word_count: int,
    raw_html: str | None = None,
) -> tuple[str, str, str]:
    """
    Classify an HTML page.

    Inputs:
        url: page URL
        title: page title (may be None)
        text: extracted text or DB snippet (banner stripped automatically)
        word_count: page word count
        raw_html: optional raw HTML, enables structural-pattern fallback

    Returns:
        (page_classification, chunking_strategy, classification_confidence)
    """
    url_lower = (url or '').lower()
    title_lower = (title or '').lower()
    clean_text = strip_banner(text)
    text_head = clean_text.lower()[:2000]
    url_title = f"{url_lower} {title_lower}"
    combined = f"{url_title} {text_head}"
    wc = word_count or 0
    parsed = urlparse(url_lower)
    path = parsed.path
    domain = parsed.netloc

    page_class, strategy, confidence = _classify(
        url_lower, url_title, combined, path, domain, wc, raw_html
    )

    # Empty extraction => no useful chunks regardless of class.
    if wc == 0 and page_class != 'junk':
        strategy = 'skip'

    return page_class, strategy, confidence


def _classify(
    url_lower: str,
    url_title: str,
    combined: str,
    path: str,
    domain: str,
    wc: int,
    raw_html: str | None,
) -> tuple[str, str, str]:
    # 1. Junk (Cloudflare stubs)
    if 'cdn-cgi' in url_lower:
        return 'junk', 'skip', 'high'

    # 2. FAQ — keyword in url/title OR known FAQ path (never page text,
    # which false-matches on sidebar boilerplate)
    if any(s in url_title for s in FAQ_SIGNALS) or any(s in path for s in FAQ_URL_PATHS):
        return 'faq', 'qa_pairs', 'high'

    # 3. Press release / news
    if any(s in combined for s in PRESS_SIGNALS):
        strategy = 'simple_split' if wc >= 150 else 'ingest_as_single'
        return 'press_release', strategy, 'high'

    # 4. MoCo location pages (drop boxes, early voting sites)
    if domain == 'mcg.montgomerycountymd.gov' and ('earlyvotin' in path or 'dropbox' in path):
        return 'location_list', 'ingest_as_single', 'high'

    # 5. Table / data pages — path-based, takes priority over prose word-count
    if any(s in path for s in TABLE_SIGNALS):
        return 'table_data', 'table_rows', 'high'

    # 6. Forms — URL path or voterservices subdomain
    if any(p in path for p in FORM_URL_PATHS) or domain.startswith('voterservices'):
        strategy = 'simple_split' if wc >= 150 else 'ingest_as_single'
        return 'form', strategy, 'high'

    # 7. Prose — long-form informational content
    if wc >= 500:
        return 'prose', 'semantic_with_overlap', 'medium'

    # 8. Known short-static pages by URL path
    if any(s in path for s in SHORT_STATIC_SIGNALS):
        return 'short_static', 'ingest_as_single', 'high'

    # 9. Structural HTML detection — only when raw_html is available
    if raw_html:
        structural = _detect_structural_patterns(raw_html)
        if structural == 'faq':
            return 'faq', 'qa_pairs', 'medium'
        if structural == 'table_data':
            return 'table_data', 'table_rows', 'medium'

    # 10. Word-count fallbacks
    if wc >= 150:
        return 'nav_hub', 'ingest_as_single', 'medium'

    return 'short_static', 'ingest_as_single', 'low'


def _detect_structural_patterns(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return None

    if _has_faq_structure(soup):
        return 'faq'

    tables = soup.find_all('table')
    if tables:
        total_rows = sum(len(t.find_all('tr')) for t in tables)
        if total_rows >= 5:
            return 'table_data'

    return None


def _has_faq_structure(soup: BeautifulSoup) -> bool:
    dls = soup.find_all('dl')
    for dl in dls:
        dts = dl.find_all('dt')
        if len(dts) >= 3:
            return True

    accordions = soup.find_all('details')
    if len(accordions) >= 3:
        return True

    accordion_classes = soup.find_all(
        class_=re.compile(r'accordion|collapsible|faq-item|toggle', re.I)
    )
    if len(accordion_classes) >= 3:
        return True

    question_headings = soup.find_all(
        ['h2', 'h3', 'h4', 'strong'],
        string=re.compile(r'\?\s*$')
    )
    if len(question_headings) >= 3:
        return True

    return False
