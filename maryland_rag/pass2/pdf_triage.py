"""
PDF triage classifier.

Classifies each crawled PDF into one of three buckets before Pass 2 extraction:
  process - fully extract, chunk, and embed
  skip    - store metadata only, do not extract or embed
  review  - uncertain; run preview extraction on first few pages then reassess
"""
import logging
import os
import re
import tempfile
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# --- Path signals ---

HIGH_RELEVANCE_PATHS = [
    '/elections/results/',
    '/elections/municipal_results/',
    '/voting/documents/',
]

LOW_RELEVANCE_PATHS = [
    '/about/meeting_materials/',
    '/about/documents/',
    '/press_room/documents/',
]

# --- Filename keyword signals ---

BOOST_KEYWORDS = [
    'results', 'election', 'ballot', 'polling',
    'precinct', 'voter', 'waiver',
]

LOWER_KEYWORDS = [
    'agenda', 'minutes', 'packet', 'meeting',
    'audit', 'report',
]

LARGE_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
PREVIEW_PAGES = 5

# Thresholds for text quality check
MIN_READABLE_WORDS = 20
MAX_SYMBOL_RATIO = 0.3


def triage_pdf(url: str, file_size_bytes: int, needs_ocr: int) -> dict:
    """
    Classify a PDF into process / skip / review.

    Returns a dict with keys:
        triage_bucket : 'process', 'skip', or 'review'
        reason        : short explanation
    """
    path = urlparse(url).path.lower()
    filename = os.path.basename(path).lower()
    filename_stem = re.sub(r'[_\-]', ' ', os.path.splitext(filename)[0])

    score = _score(path, filename_stem)
    is_large = file_size_bytes is not None and file_size_bytes > LARGE_FILE_BYTES
    is_ocr = bool(needs_ocr)

    # --- Initial bucket assignment ---
    if score >= 1:
        bucket = 'process'
        reason = f'high relevance score ({score})'
    elif score <= -1:
        bucket = 'skip'
        reason = f'low relevance score ({score})'
    else:
        bucket = 'review'
        reason = 'neutral score, needs preview'

    # Cost check: large OCR files default to review unless clearly relevant
    if is_large and is_ocr and bucket == 'process':
        bucket = 'review'
        reason = f'high relevance but large OCR file ({file_size_bytes // (1024*1024)}MB), running preview first'

    if bucket == 'skip':
        return {'triage_bucket': 'skip', 'reason': reason}

    if bucket == 'process' and not is_ocr:
        return {'triage_bucket': 'process', 'reason': reason}

    if bucket == 'process' and is_ocr:
        return {'triage_bucket': 'process', 'reason': reason + ', full OCR required'}

    # bucket == 'review': run preview extraction to decide
    return _preview_and_reassess(url, is_ocr, reason)


def _score(path: str, filename_stem: str) -> int:
    score = 0

    for p in HIGH_RELEVANCE_PATHS:
        if p in path:
            score += 2
            break

    for p in LOW_RELEVANCE_PATHS:
        if p in path:
            score -= 2
            break

    for kw in BOOST_KEYWORDS:
        if kw in filename_stem:
            score += 1
            break

    for kw in LOWER_KEYWORDS:
        if kw in filename_stem:
            score -= 1
            break

    return score


def _preview_and_reassess(url: str, is_ocr: bool, base_reason: str) -> dict:
    """Extract the first few pages and reassess relevance."""
    from .cache import get_bytes

    pdf_bytes = get_bytes(url)
    if not pdf_bytes:
        return {'triage_bucket': 'review', 'reason': base_reason + ', could not fetch for preview'}

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        if is_ocr:
            text = _ocr_preview(tmp.name)
        else:
            text = _text_preview(tmp.name)

    if not text:
        return {'triage_bucket': 'review', 'reason': base_reason + ', preview returned no text'}

    quality = _text_quality(text)
    if quality == 'unreadable':
        return {'triage_bucket': 'review', 'reason': base_reason + ', text quality too poor to assess'}

    relevance = _assess_relevance(text)
    if relevance == 'relevant':
        return {'triage_bucket': 'process', 'reason': base_reason + ', upgraded after preview'}
    if relevance == 'irrelevant':
        return {'triage_bucket': 'skip', 'reason': base_reason + ', downgraded after preview'}

    return {'triage_bucket': 'review', 'reason': base_reason + ', still unclear after preview'}


def _text_preview(path: str) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:PREVIEW_PAGES]:
                text = page.extract_text() or ''
                parts.append(text)
        text = '\n'.join(parts).strip()
        if text:
            return text
    except Exception as exc:
        logger.debug("pdfplumber preview failed: %s", exc)

    try:
        import fitz
        parts = []
        doc = fitz.open(path)
        for page in list(doc)[:PREVIEW_PAGES]:
            parts.append(page.get_text())
        doc.close()
        return '\n'.join(parts).strip()
    except Exception as exc:
        logger.debug("pymupdf preview failed: %s", exc)

    return ''


def _ocr_preview(path: str) -> str:
    try:
        import fitz
        from PIL import Image
        import pytesseract
        import io

        parts = []
        doc = fitz.open(path)
        for page in list(doc)[:PREVIEW_PAGES]:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(pytesseract.image_to_string(img))
        doc.close()
        return '\n'.join(parts).strip()
    except Exception as exc:
        logger.debug("OCR preview failed: %s", exc)
        return ''


def _text_quality(text: str) -> str:
    words = text.split()
    if len(words) < MIN_READABLE_WORDS:
        return 'unreadable'
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 0 and non_alnum / len(text) > MAX_SYMBOL_RATIO:
        return 'unreadable'
    return 'ok'


def _assess_relevance(text: str) -> str:
    text_lower = text.lower()
    boost_hits = sum(1 for kw in BOOST_KEYWORDS if kw in text_lower)
    lower_hits = sum(1 for kw in LOWER_KEYWORDS if kw in text_lower)
    if boost_hits >= 2:
        return 'relevant'
    if lower_hits >= 2 and boost_hits == 0:
        return 'irrelevant'
    return 'unclear'


def run_triage(db_path: str | None = None):
    """
    Run triage on all crawled PDFs in the manifest and write results back to the DB.
    Skips PDFs that already have a triage_bucket set.
    """
    from ..pass1.db import DB
    db = DB(db_path)
    pages = db.get_pdf_pages()

    already_done = 0
    processed = 0

    for page in pages:
        if page['triage_bucket']:
            already_done += 1
            continue

        result = triage_pdf(
            url=page['url'],
            file_size_bytes=page['file_size_bytes'],
            needs_ocr=page['needs_ocr'] or 0,
        )
        db.update_triage(page['url'], result['triage_bucket'], result['reason'])
        logger.info("[%s] %s — %s", result['triage_bucket'], page['url'], result['reason'])
        processed += 1

    logger.info("Triage complete. processed=%d skipped_already_done=%d", processed, already_done)
    db.close()
    return processed
