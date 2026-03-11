"""
Table-rows chunking strategy for pages classified as 'table_data'.

Each table row (or logical row group) becomes one chunk, with
column headers prepended for context so each chunk is self-contained.
"""
from bs4 import BeautifulSoup


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
    tables = soup.find_all('table')

    if not tables:
        return []

    all_chunks = []

    for table_idx, table in enumerate(tables):
        # Extract caption if present
        caption = ''
        cap_tag = table.find('caption')
        if cap_tag:
            caption = cap_tag.get_text(strip=True)

        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [
                    th.get_text(strip=True)
                    for th in header_row.find_all(['th', 'td'])
                ]

        # If no thead, try first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                ths = first_row.find_all('th')
                if ths:
                    headers = [th.get_text(strip=True) for th in ths]

        # Extract body rows
        rows = []
        tbody = table.find('tbody')
        row_source = tbody if tbody else table
        for tr in row_source.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cells and any(c for c in cells):  # skip empty rows
                rows.append(cells)

        # Skip header-only tables
        if not rows:
            continue

        # Build chunks: each row gets headers prepended
        for row_idx, row in enumerate(rows):
            if headers and len(row) == len(headers):
                # Format as "Header: Value" pairs
                pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
                text = ' | '.join(pairs)
            else:
                text = ' | '.join(cell for cell in row if cell)

            if caption:
                text = f"[{caption}] {text}"

            if text.strip():
                all_chunks.append({
                    'text': text.strip(),
                    'table_index': table_idx,
                    'row_index': row_idx,
                })

    return all_chunks
