"""
Rule-based content classification and chunking strategy assignment.

Improvements over original design:
- Structural HTML detection for FAQs (dl/dt/dd, accordion markup, Q/A heading patterns).
- classification_confidence field: 'high' for strong signal matches, 'medium' for
  structural HTML detection, 'low' for default prose fallback.
"""
import re

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Signal keywords (checked against combined URL + title + text head)
# ---------------------------------------------------------------------------

FAQ_SIGNALS = [
    'faq', 'frequently-asked', 'frequently asked',
    'q&a', 'q & a', 'questions and answers',
]
PRESS_SIGNALS = [
    'press-release', 'press_release', 'news',
    'announcement', 'media-release',
]
FORM_SIGNALS = [
    'form', 'application', 'register', 'registration',
    'submit', 'apply',
]
TABLE_SIGNALS = [
    'results', 'candidates', 'districts', 'precincts',
    'statistics', 'lookup', 'search',
]
SHORT_STATIC_SIGNALS = [
    'contact', 'about', 'staff', 'office',
    'hours', 'location', 'directions',
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

    combined = f"{url} {title} {text_head}"
    page_class = None
    confidence = 'low'

    # --- 1. Keyword-signal matching (high confidence) ---
    if any(s in combined for s in FAQ_SIGNALS):
        page_class = 'faq'
        confidence = 'high'
    elif any(s in combined for s in PRESS_SIGNALS):
        page_class = 'press_release'
        confidence = 'high'
    elif any(s in combined for s in FORM_SIGNALS):
        page_class = 'form'
        confidence = 'high'
    elif any(s in combined for s in TABLE_SIGNALS):
        page_class = 'table_data'
        confidence = 'medium'
    elif any(s in combined for s in SHORT_STATIC_SIGNALS):
        page_class = 'short_static'
        confidence = 'medium'

    # --- 2. Structural HTML detection (medium confidence) ---
    # Only run if we haven't already found a high-confidence classification
    if page_class is None and raw_html:
        structural = _detect_structural_patterns(raw_html)
        if structural:
            page_class = structural
            confidence = 'medium'

    # --- 3. Default fallback ---
    if page_class is None:
        page_class = 'prose'
        confidence = 'low'

    # --- Assign chunking strategy based on classification + size ---
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
    # Common patterns: <details>/<summary>, .accordion, .collapsible, data-toggle
    accordions = soup.find_all('details')
    if len(accordions) >= 3:
        return True

    accordion_classes = soup.find_all(
        class_=re.compile(r'accordion|collapsible|faq-item|toggle', re.I)
    )
    if len(accordion_classes) >= 3:
        return True

    # Pattern 3: Repeated heading + paragraph pattern suggesting Q&A
    # Look for 3+ consecutive heading-then-paragraph blocks where headings end with '?'
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

    if page_class == 'faq':
        return 'qa_pairs'

    if page_class == 'table_data':
        return 'table_rows'

    if word_count < 150:
        return 'ingest_as_single'

    if page_class in ('press_release', 'form', 'short_static') or word_count < 800:
        return 'simple_split'

    return 'semantic_with_overlap'
