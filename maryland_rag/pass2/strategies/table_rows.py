"""
Table-rows chunking strategy for pages classified as 'table_data'.

Each table row (or logical row group) becomes one chunk, with
column headers prepended for context so each chunk is self-contained.
"""
import re

from bs4 import BeautifulSoup

from .semantic import enforce_chunk_caps


def extract_table_chunks(url: str, raw_html: str | None = None) -> list[dict]:
    """
    Extract table data as row-level chunks from an HTML page.

    Args:
        url: Page URL.
        raw_html: Pre-fetched HTML (optional).

    Returns:
        List of dicts with 'text', 'table_index', 'row_index' keys.
    """
    if not raw_html:
        from ...pass2.cache import get_html
        raw_html = get_html(url)
        if not raw_html:
            return []

    soup = BeautifulSoup(raw_html, 'html.parser')

    # Strip script/style so embedded JS/CSS never leaks into cell text.
    for junk in soup(['script', 'style', 'noscript']):
        junk.decompose()

    tables = soup.find_all('table')

    if not tables:
        return []

    all_chunks = []

    for table_idx, table in enumerate(tables):
        # Extract caption if present
        caption = ''
        cap_tag = table.find('caption')
        if cap_tag:
            caption = _text(cap_tag)

        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [
                    _text(th)
                    for th in header_row.find_all(['th', 'td'])
                ]

        # If no thead, try first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                ths = first_row.find_all('th')
                if ths:
                    headers = [_text(th) for th in ths]

        # Extract body rows
        rows = []
        tbody = table.find('tbody')
        row_source = tbody if tbody else table
        for tr in row_source.find_all('tr'):
            cells = [_text(td) for td in tr.find_all(['td', 'th'])]
            if cells and any(c for c in cells):  # skip empty rows
                rows.append(cells)

        # Skip header-only tables
        if not rows:
            continue

        # Build chunks: each row gets headers prepended
        for row_idx, row in enumerate(rows):
            if headers and len(row) == len(headers):
                # Format as "Header: Value" pairs (bare value if the
                # header cell is empty, never an empty ': value' prefix)
                pairs = [f"{h}: {v}" if h else v for h, v in zip(headers, row) if v]
                text = ' | '.join(pairs)
            else:
                text = ' | '.join(cell for cell in row if cell)

            if caption:
                text = f"[{caption}] {text}"

            # enforce_chunk_caps splits the (rare) giant row so no chunk
            # exceeds the embedding-safe caps
            for part in enforce_chunk_caps([text]):
                all_chunks.append({
                    'text': part,
                    'table_index': table_idx,
                    'row_index': row_idx,
                })

    return all_chunks


def _text(el) -> str:
    """Whitespace-normalized cell text: space-separated so words never
    glue across inline tags."""
    return re.sub(r'\s+', ' ', el.get_text(separator=' ', strip=True)).strip()
