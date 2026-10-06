"""
PDF document extraction strategy.

Primary: pdfplumber for clean digital PDFs (tables, structured text).
Fallback: pymupdf (fitz) for complex layouts.
OCR path: pytesseract via pymupdf when digital extraction finds no text.

Digital extraction is always attempted first: the manifest's needs_ocr
column holds stale flags from a removed Pass 1 probe and is not trusted.

After text extraction, content is routed to the appropriate text chunking
strategy based on detected structure (FAQ, table, prose).
"""
import logging
import re
import tempfile


logger = logging.getLogger(__name__)


_EMPTY_RESULT = {'text': '', 'pages': [], 'tables': [], 'structure_type': 'empty'}

# Digital extraction counts as "effectively empty" below this many word
# characters: image-only PDFs often carry a few stray glyphs of embedded
# text (page numbers, watermarks) that shouldn't block the OCR fallback.
MIN_DIGITAL_TEXT_CHARS = 20


def extract_pdf_from_path(path: str, needs_ocr: bool = False, ocr_fallback: bool = False) -> dict:
    """
    Extract text and structure from a PDF already on local disk.

    Args:
        path: Local file path to the PDF.
        needs_ocr: Deprecated and ignored. The manifest permanently retains
                   stale needs_ocr=1 flags from a removed Pass 1 probe, so
                   digital extraction is always tried first (OCR is slower
                   and lossier).
        ocr_fallback: If True and digital extraction yields effectively no
                      text, fall back to OCR.

    Returns:
        Dict with 'text', 'pages', 'tables', 'structure_type' keys.
    """
    digital = _extract_with_pdfplumber(path)
    if digital and not _effectively_empty(digital.get('text', '')):
        return digital

    fallback = _extract_with_pymupdf(path)
    if fallback and not _effectively_empty(fallback.get('text', '')):
        return fallback

    if ocr_fallback:
        logger.info("No embedded text in %s; falling back to OCR", path)
        ocr = _extract_with_ocr(path)
        if not _effectively_empty(ocr.get('text', '')):
            return ocr

    # OCR unavailable or empty too: return whatever scraps digital
    # extraction found rather than dropping them.
    for result in (digital, fallback):
        if result and result.get('text', '').strip():
            return result
    return dict(_EMPTY_RESULT)


def extract_pdf(url: str, needs_ocr: bool = False, ocr_fallback: bool = True,
                data: bytes | None = None) -> dict:
    """
    Extract text and structure from a PDF, fetched via the Pass 2 disk cache.

    Args:
        url: PDF URL.
        needs_ocr: Deprecated and ignored (see extract_pdf_from_path).
        ocr_fallback: If True (default) and digital extraction yields
                      effectively no text, fall back to OCR.
        data: Pre-fetched (and magic-validated) PDF bytes; fetched via the
              cache when None.

    Returns:
        Dict with 'text', 'pages', 'tables', 'structure_type' keys.
    """
    pdf_bytes = data
    if pdf_bytes is None:
        from ..cache import get_bytes
        pdf_bytes = get_bytes(url, expect='pdf')
    if not pdf_bytes:
        return dict(_EMPTY_RESULT)

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        return extract_pdf_from_path(tmp.name, ocr_fallback=ocr_fallback)


def _effectively_empty(text: str) -> bool:
    """True when extracted text has too few word characters to be usable."""
    return len(re.findall(r'\w', text or '')) < MIN_DIGITAL_TEXT_CHARS


def _extract_with_pdfplumber(path: str) -> dict | None:
    """Extract text and tables using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping")
        return None

    try:
        pages = []
        tables = []
        full_text_parts = []

        column_pages = _two_column_page_texts(path)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text. pdfplumber reads line-by-line across the
                # full page width, which interleaves side-by-side columns
                # (one county's early-voting center next to another
                # county's address); two-column pages use the
                # column-ordered text instead.
                text = column_pages.get(i) or page.extract_text() or ''
                pages.append({'page_num': i + 1, 'text': text})
                full_text_parts.append(text)

                # Extract tables
                page_tables = page.extract_tables()
                for t_idx, table_data in enumerate(page_tables):
                    # A "table" with at most one non-empty cell per row is
                    # prose inside ruled borders/boxes (charter amendment
                    # texts, letterhead press releases), not tabular data.
                    # Keeping it routed the PDF to table_heavy and emitted
                    # every text line as its own row chunk (the Howard
                    # County charter text produced 448 ~36-word fragments).
                    if table_data and len(table_data) > 1 and _is_multi_column(table_data):
                        tables.append({
                            'page_num': i + 1,
                            'table_index': t_idx,
                            'headers': table_data[0] if table_data else [],
                            'rows': table_data[1:] if len(table_data) > 1 else [],
                        })

        full_text = '\n\n'.join(full_text_parts)
        structure_type = _detect_pdf_structure(full_text, tables)

        return {
            'text': full_text,
            'pages': pages,
            'tables': tables,
            'structure_type': structure_type,
        }
    except Exception as exc:
        logger.warning("pdfplumber extraction failed: %s", exc)
        return None


# Two-column page detection (fractions of page width). Text blocks starting
# left of _COL_SPLIT are the left column; a block at least _FULL_WIDTH wide
# spans both columns (title, intro paragraph) and separates column bands.
_COL_SPLIT = 0.40
_FULL_WIDTH = 0.55


def _is_two_column_page(blocks: list, width: float) -> bool:
    """True when the page body is two side-by-side text columns: at least
    three narrow blocks start in the left third and at least three start
    just right of center, vertically overlapping. Blocks starting anywhere
    else mean a multi-column table, whose row-wise reading order must be
    kept, so those pages are left alone."""
    narrow = [b for b in blocks if (b[2] - b[0]) < _FULL_WIDTH * width]
    left = [b for b in narrow if b[0] < 0.30 * width]
    right = [b for b in narrow if _COL_SPLIT * width <= b[0] < 0.65 * width]
    other = [b for b in narrow
             if 0.30 * width <= b[0] < _COL_SPLIT * width or b[0] >= 0.65 * width]
    if len(left) < 3 or len(right) < 3 or len(other) > max(2, 0.2 * len(narrow)):
        return False
    overlapping = sum(
        1 for r in right if any(l[1] < r[3] and r[1] < l[3] for l in left)
    )
    return overlapping >= 3


def _column_ordered_text(blocks: list, width: float) -> str:
    """Page text with each column read top-to-bottom, left column first.
    Full-width blocks are emitted in place and split the page into bands,
    so a heading above the columns stays above them."""
    out, band = [], []

    def flush():
        left = [b for b in band if b[0] < _COL_SPLIT * width]
        right = [b for b in band if b[0] >= _COL_SPLIT * width]
        for col in (left, right):
            out.extend(b[4].strip() for b in sorted(col, key=lambda b: b[1]))
        band.clear()

    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        if (b[2] - b[0]) >= _FULL_WIDTH * width:
            flush()
            out.append(b[4].strip())
        else:
            band.append(b)
    flush()
    return '\n'.join(out)


def _two_column_page_texts(path: str) -> dict[int, str]:
    """{page_index: column-ordered text} for the PDF's two-column pages.
    Uses pymupdf text blocks (one block = one paragraph/address within a
    column). Empty when pymupdf is unavailable or the PDF can't be read."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return {}
    texts = {}
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                blocks = [b for b in page.get_text("blocks")
                          if b[6] == 0 and b[4].strip()]
                if _is_two_column_page(blocks, page.rect.width):
                    texts[i] = _column_ordered_text(blocks, page.rect.width)
    except Exception as exc:
        logger.warning("two-column detection failed for %s: %s", path, exc)
        return {}
    return texts


def _extract_with_pymupdf(path: str) -> dict | None:
    """Extract text using pymupdf (fitz) as fallback."""
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("pymupdf not installed, skipping")
        return None

    try:
        pages = []
        full_text_parts = []

        doc = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({'page_num': i + 1, 'text': text})
            full_text_parts.append(text)
        doc.close()

        full_text = '\n\n'.join(full_text_parts)
        structure_type = _detect_pdf_structure(full_text, [])

        return {
            'text': full_text,
            'pages': pages,
            'tables': [],
            'structure_type': structure_type,
        }
    except Exception as exc:
        logger.warning("pymupdf extraction failed: %s", exc)
        return None


def _extract_with_ocr(path: str) -> dict:
    """
    OCR extraction for image-only PDFs using pymupdf + pytesseract.
    """
    try:
        import fitz  # pymupdf
        from PIL import Image
        import pytesseract
        import io
    except ImportError as exc:
        logger.warning("OCR dependencies not available: %s", exc)
        return {'text': '', 'pages': [], 'tables': [], 'structure_type': 'ocr_failed'}

    try:
        pages = []
        full_text_parts = []

        doc = fitz.open(path)
        for i, page in enumerate(doc):
            # Render page as image
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # OCR the image
            text = pytesseract.image_to_string(img)
            pages.append({'page_num': i + 1, 'text': text})
            full_text_parts.append(text)
        doc.close()

        full_text = '\n\n'.join(full_text_parts)

        return {
            'text': full_text,
            'pages': pages,
            'tables': [],
            'structure_type': 'ocr',
        }
    except Exception as exc:
        logger.error("OCR extraction failed: %s", exc)
        return {'text': '', 'pages': [], 'tables': [], 'structure_type': 'ocr_failed'}


def _is_multi_column(table_data: list) -> bool:
    """True when some row of the table has at least two non-empty cells."""
    return any(
        sum(1 for cell in row if cell and str(cell).strip()) >= 2
        for row in table_data
    )


def _detect_pdf_structure(text: str, tables: list) -> str:
    """
    Detect the dominant structure type of extracted PDF content
    to route to the correct text chunking strategy.
    """
    if not text.strip():
        return 'empty'

    if tables and len(tables) >= 2:
        return 'table_heavy'

    # Check for FAQ patterns
    question_marks = text.count('?')
    lines = text.split('\n')
    if question_marks >= 5 and question_marks / max(len(lines), 1) > 0.05:
        return 'faq'

    word_count = len(text.split())
    if word_count < 150:
        return 'short'

    return 'prose'
