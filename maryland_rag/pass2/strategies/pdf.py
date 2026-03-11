"""
PDF document extraction strategy.

Primary: pdfplumber for clean digital PDFs (tables, structured text).
Fallback: pymupdf (fitz) for complex layouts.
OCR path: pytesseract via pymupdf for image-only PDFs (flagged by needs_ocr in Pass 1).

After text extraction, content is routed to the appropriate text chunking
strategy based on detected structure (FAQ, table, prose).
"""
import logging
import re
import tempfile


logger = logging.getLogger(__name__)


def extract_pdf(url: str, needs_ocr: bool = False) -> dict:
    """
    Extract text and structure from a PDF.

    Args:
        url: PDF URL.
        needs_ocr: If True, attempt OCR extraction.

    Returns:
        Dict with 'text', 'pages', 'tables', 'structure_type' keys.
    """
    from ..cache import get_bytes
    pdf_bytes = get_bytes(url)
    if not pdf_bytes:
        return {'text': '', 'pages': [], 'tables': [], 'structure_type': 'empty'}

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        if needs_ocr:
            return _extract_with_ocr(tmp.name)

        # Try pdfplumber first
        result = _extract_with_pdfplumber(tmp.name)
        if result and result.get('text', '').strip():
            return result

        # Fallback to pymupdf
        result = _extract_with_pymupdf(tmp.name)
        if result and result.get('text', '').strip():
            return result

        return {'text': '', 'pages': [], 'tables': [], 'structure_type': 'empty'}


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

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ''
                pages.append({'page_num': i + 1, 'text': text})
                full_text_parts.append(text)

                # Extract tables
                page_tables = page.extract_tables()
                for t_idx, table_data in enumerate(page_tables):
                    if table_data and len(table_data) > 1:
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
