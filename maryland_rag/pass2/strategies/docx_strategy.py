"""
DOCX document extraction strategy.

Uses python-docx to walk the document's heading hierarchy and split
at H1/H2 boundaries. Each section preserves its heading chain in metadata
so retrieval can reconstruct the document's structure.
"""
import logging
import tempfile


logger = logging.getLogger(__name__)


_EMPTY_DOCX = {'sections': [], 'tables': [], 'full_text': ''}


def extract_docx_from_path(path: str) -> dict:
    """
    Extract text from a DOCX file already on local disk, split by heading
    hierarchy.

    Returns:
        Dict with 'sections', 'tables', and 'full_text' keys.
    """
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed")
        return dict(_EMPTY_DOCX)

    try:
        doc = docx.Document(path)
    except Exception as exc:
        logger.error("Failed to parse DOCX %s: %s", path, exc)
        return dict(_EMPTY_DOCX)

    return _walk_headings(doc)


def extract_docx(url: str) -> dict:
    """
    Extract text from a DOCX file fetched via the Pass 2 disk cache.

    Returns:
        Dict with 'sections', 'tables', and 'full_text' keys.
    """
    from ..cache import get_bytes
    docx_bytes = get_bytes(url)
    if not docx_bytes:
        return dict(_EMPTY_DOCX)

    with tempfile.NamedTemporaryFile(suffix='.docx', delete=True) as tmp:
        tmp.write(docx_bytes)
        tmp.flush()
        return extract_docx_from_path(tmp.name)


def _walk_headings(doc) -> dict:
    """
    Walk paragraphs and split into sections at heading boundaries.

    Each section carries its heading chain (e.g., ["Chapter 1", "Section 1.2"])
    for use as section_hierarchy metadata.
    """
    sections = []
    heading_stack = []  # (level, text) stack for current position
    current_text_parts = []
    full_text_parts = []

    # Heading style name → level mapping
    heading_levels = {
        'Heading 1': 1, 'Heading 2': 2, 'Heading 3': 3,
        'Heading 4': 4, 'Heading 5': 5, 'Heading 6': 6,
        'Title': 0,
    }

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else ''
        text = para.text.strip()

        if not text:
            continue

        full_text_parts.append(text)
        level = heading_levels.get(style_name)

        if level is not None:
            # Emit previous section
            if current_text_parts:
                section_text = '\n\n'.join(current_text_parts)
                sections.append({
                    'heading_chain': [h[1] for h in heading_stack],
                    'text': section_text,
                    'word_count': len(section_text.split()),
                })

            # Update heading stack: pop everything at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            current_text_parts = []
        else:
            current_text_parts.append(text)

    # Emit final section
    if current_text_parts:
        section_text = '\n\n'.join(current_text_parts)
        sections.append({
            'heading_chain': [h[1] for h in heading_stack],
            'text': section_text,
            'word_count': len(section_text.split()),
        })

    # Also extract tables
    tables = _extract_docx_tables(doc)

    return {
        'sections': sections,
        'tables': tables,
        'full_text': '\n\n'.join(full_text_parts),
    }


def _extract_docx_tables(doc) -> list[dict]:
    """Extract tables from a DOCX document."""
    tables = []
    for t_idx, table in enumerate(doc.tables):
        rows = []
        headers = []
        for r_idx, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            if r_idx == 0:
                headers = cells
            else:
                rows.append(cells)

        if rows:
            tables.append({
                'table_index': t_idx,
                'headers': headers,
                'rows': rows,
            })
    return tables
