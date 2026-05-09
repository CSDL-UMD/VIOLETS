"""
Rule-based content classification and chunking strategy assignment.

Improvements over original design:
- Removes over-broad 'register'/'registration' from form signals (matched nearly
  every page on an elections site, causing 258 pages to be classified as 'form').
- Adds 'nav_hub' class for link-heavy hub pages (150–499 words).
- Activates 'semantic_with_overlap' strategy for prose pages (was dead code before).
- URL-path-based form detection instead of keyword matching.
- press_room/* treated as press_release, not form.
- classification_confidence: 'high' for URL/structural matches, 'medium' for
  word-count-based fallbacks, 'low' for final default.
- TABLE_SIGNALS, FORM_URL_PATHS, SHORT_STATIC_SIGNALS applied to all domains.
- Adds MoCo location_list class for drop box and early voting pages.
"""
import re

from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Signal keywords — checked against url + title + text head
# ---------------------------------------------------------------------------

FAQ_SIGNALS = [
    'faq', 'frequently-asked', 'frequently asked',
    'q&a', 'q & a', 'questions and answers',
]

# URL path substrings for known FAQ pages on this site that lack faq keywords
FAQ_URL_PATHS = [
    'early_voting', 'absentee', 'election_day_questions',
    'learn_about_the_new_voting_system', 'redistricting',
    '/about/pia',
]

PRESS_SIGNALS = [
    'press_room', 'press-room',
    'press-release', 'press_release',
    'news-release', 'announcement', 'media-release',
    'rumor_control', 'dis-misinformation',
]

# URL path substrings that indicate actual online forms (not pages that mention forms)
FORM_URL_PATHS = [
    '/forms/',
    'data_form',
    'schedule_appointment',
    'purchase_lists',
]

TABLE_SIGNALS = [
    'municipal_results', 'election_results', 'results_archive',
    '/elections/districts', '/elections/archive', '/elections/printed_copies',
    'voter_registration/archive', 'voter_registration/stats',
    '/voting/recount',
]

SHORT_STATIC_SIGNALS = [
    'contact', 'directions', 'social_media', 'county_boards',
    'state-links', 'federal-links', 'feedback',
    'voting_equipment', 'sbe_policy',
    'qualifications',
]


def classify_page(result: dict) -> dict:
    """
    Classify an extracted page and assign a chunking strategy.

    Returns dict with keys:
        content_type, page_classification, chunking_strategy, classification_confidence
    """
    url = (result.get('url') or '').lower()
    title = (result.get('title') or '').lower()
    text_head = (result.get('text') or '').lower()[:2000]
    word_count = result.get('word_count') or 0
    ctype = result.get('content_type', 'html')
    raw_html = result.get('raw_html', '')

    # --- Documents: always route to document_extraction ---
    if ctype in ('pdf', 'docx', 'doc', 'xls', 'xlsx', 'csv'):
        return {
            'content_type': ctype,
            'page_classification': 'document',
            'chunking_strategy': 'document_extraction',
            'classification_confidence': 'high',
        }

    path = urlparse(url).path.lower()
    domain = urlparse(url).netloc.lower()
    combined = f"{url} {title} {text_head}"
    page_class = None
    confidence = 'low'

    # --- 1. Junk pages (Cloudflare stubs, empty external subdomains) ---
    if 'cdn-cgi' in url:
        page_class = 'junk'
        confidence = 'high'

    # --- 2. FAQ: keyword signals ---
    elif any(s in combined for s in FAQ_SIGNALS) or any(s in path for s in FAQ_URL_PATHS):
        page_class = 'faq'
        confidence = 'high'

    # --- 3. Press release / news ---
    elif any(s in combined for s in PRESS_SIGNALS):
        page_class = 'press_release'
        confidence = 'high'

    # --- 4. MoCo location pages (path-based, high confidence) ---
    # Only the specific location-list paths are caught here; other MoCo pages
    # (vote-by-mail, accessibility, etc.) fall through to prose/nav_hub below.
    elif domain == 'mcg.montgomerycountymd.gov' and ('earlyvotin' in path or 'dropbox' in path):
        page_class = 'location_list'
        confidence = 'high'

    # --- 5. Table data signals (URL-path based, takes priority over prose) ---
    elif any(s in combined for s in TABLE_SIGNALS):
        page_class = 'table_data'
        confidence = 'high'

    # --- 6. Form signals (URL-path based) ---
    elif any(p in path for p in FORM_URL_PATHS) or domain.startswith('voterservices'):
        page_class = 'form'
        confidence = 'high'

    # --- 7. Prose: long-form informational pages (any domain) ---
    elif word_count >= 500:
        page_class = 'prose'
        confidence = 'medium'

    # --- 8. Short static signals ---
    elif any(s in combined for s in SHORT_STATIC_SIGNALS):
        page_class = 'short_static'
        confidence = 'high'

    # --- 9. Structural HTML detection (medium confidence) ---
    # Separate if — not elif — so it runs for any domain when signals above didn't fire.
    if page_class is None and raw_html:
        structural = _detect_structural_patterns(raw_html)
        if structural:
            page_class = structural
            confidence = 'medium'

    # --- 10. Word-count fallbacks ---
    if page_class is None:
        if word_count >= 150:
            page_class = 'nav_hub'
            confidence = 'medium'
        else:
            page_class = 'short_static'
            confidence = 'low'

    # --- Assign chunking strategy ---
    strategy = _assign_strategy(page_class, word_count)

    return {
        'content_type': 'html',
        'page_classification': page_class,
        'chunking_strategy': strategy,
        'classification_confidence': confidence,
    }


def _detect_structural_patterns(html: str) -> str | None:
    """
    Scan raw HTML for structural patterns that indicate content type,
    even when URL/title signals are absent.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return None

    # FAQ patterns: definition lists, accordion components, Q/A headings
    if _has_faq_structure(soup):
        return 'faq'

    # Table-heavy pages
    tables = soup.find_all('table')
    if tables:
        total_rows = sum(len(t.find_all('tr')) for t in tables)
        if total_rows >= 5:
            return 'table_data'

    return None


def _has_faq_structure(soup: BeautifulSoup) -> bool:
    """Detect FAQ-like structural patterns in HTML."""

    # Pattern 1: Definition lists with multiple dt/dd pairs
    dls = soup.find_all('dl')
    for dl in dls:
        dts = dl.find_all('dt')
        if len(dts) >= 3:
            return True

    # Pattern 2: Accordion / collapsible components
    accordions = soup.find_all('details')
    if len(accordions) >= 3:
        return True

    accordion_classes = soup.find_all(
        class_=re.compile(r'accordion|collapsible|faq-item|toggle', re.I)
    )
    if len(accordion_classes) >= 3:
        return True

    # Pattern 3: Repeated heading + paragraph pattern suggesting Q&A
    question_headings = soup.find_all(
        ['h2', 'h3', 'h4', 'strong'],
        string=re.compile(r'\?\s*$')
    )
    if len(question_headings) >= 3:
        return True

    return False


def _assign_strategy(page_class: str, word_count: int) -> str:
    """Map classification + word count to a chunking strategy."""
    if word_count == 0:
        return 'skip'

    if page_class == 'junk':
        return 'skip'

    if page_class == 'faq':
        return 'qa_pairs'

    if page_class == 'table_data':
        return 'table_rows'

    if page_class == 'prose':
        return 'semantic_with_overlap'
    
    if page_class == 'location_list':
        return 'ingest_as_single'

    if page_class in ('nav_hub', 'short_static'):
        return 'ingest_as_single'

    # press_release, form
    if word_count < 150:
        return 'ingest_as_single'

    return 'simple_split'
