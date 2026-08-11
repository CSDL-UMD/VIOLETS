"""
Table-rows chunking strategy for pages classified as 'table_data'.

Consecutive rows of a table are batched into grouped chunks (roughly
MIN_GROUP_CHARS–MAX_GROUP_CHARS of text each) with column headers inlined
per row and the table caption prepended per batch, so each chunk is
self-contained without emitting hundreds of sub-50-char row stubs.
"""
import re

from bs4 import BeautifulSoup

from .semantic import enforce_chunk_caps

# Row-grouping bounds: batch consecutive rows until a chunk reaches at least
# MIN_GROUP_CHARS, without growing past MAX_GROUP_CHARS (a single oversized
# row is still emitted alone — enforce_chunk_caps handles the pathological
# giant row). Tiny single-row chunks embed poorly and drown retrieval.
MIN_GROUP_CHARS = 200
MAX_GROUP_CHARS = 1000


def extract_table_chunks(url: str, raw_html: str | None = None) -> list[dict]:
    """
    Extract table data as grouped row chunks from an HTML page.

    Args:
        url: Page URL.
        raw_html: Pre-fetched HTML (optional).

    Returns:
        List of dicts with 'text', 'table_index', 'row_start', 'row_end' keys.
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

        # Extract headers. header_tr records WHICH <tr> supplied them so the
        # body scan below can skip it — html.parser doesn't synthesize
        # <tbody>, so a header taken from a first-row <th> scan (or a thead
        # without tbody) would otherwise be re-emitted as a data chunk.
        headers = []
        header_tr = None
        thead = table.find('thead')
        if thead:
            header_tr = thead.find('tr')
            if header_tr:
                headers = [
                    _text(th)
                    for th in header_tr.find_all(['th', 'td'])
                ]

        # If no thead, try first row — but only when it holds NO <td>
        # cells: a <th scope=row> label cell alongside <td> data cells is a
        # DATA row (row-label tables), and consuming it as a header would
        # silently drop the table's first row.
        if not headers:
            first_row = table.find('tr')
            if first_row:
                ths = first_row.find_all('th')
                if ths and not first_row.find_all('td'):
                    headers = [_text(th) for th in ths]
                    header_tr = first_row

        # Extract body rows: skip the recorded header row, and when a thead
        # exists exclude ALL its rows from the scan (multi-row theads).
        rows = []
        tbody = table.find('tbody')
        row_source = tbody if tbody else table
        for tr in row_source.find_all('tr'):
            if tr is header_tr:
                continue
            if thead is not None and thead in tr.parents:
                continue
            cells = [_text(td) for td in tr.find_all(['td', 'th'])]
            if cells and any(c for c in cells):  # skip empty rows
                rows.append(cells)

        # Skip header-only tables
        if not rows:
            continue

        # Format each row as "Header: Value" pairs (bare value if the
        # header cell is empty, never an empty ': value' prefix)
        row_texts = []
        for row in rows:
            if headers and len(row) == len(headers):
                pairs = [f"{h}: {v}" if h else v for h, v in zip(headers, row) if v]
                row_texts.append(' | '.join(pairs))
            else:
                row_texts.append(' | '.join(cell for cell in row if cell))

        # Batch consecutive rows into grouped chunks; the caption is
        # prepended once per batch and the row range recorded in metadata.
        for row_start, row_end, batch in group_row_texts(row_texts):
            text = f"[{caption}]\n{batch}" if caption else batch

            # enforce_chunk_caps splits the (rare) giant batch so no chunk
            # exceeds the embedding-safe caps
            for part in enforce_chunk_caps([text]):
                all_chunks.append({
                    'text': part,
                    'table_index': table_idx,
                    'row_start': row_start,
                    'row_end': row_end,
                })

    return all_chunks


def group_row_texts(row_texts: list[str],
                    min_chars: int = MIN_GROUP_CHARS,
                    max_chars: int = MAX_GROUP_CHARS) -> list[tuple[int, int, str]]:
    """
    Batch consecutive row strings into newline-joined groups of roughly
    min_chars–max_chars each. A batch is flushed before adding a row that
    would push it past max_chars, but only once it has reached min_chars —
    so a batch never ends undersized just because the next row is large.
    Also used by the XLS strategy so spreadsheet rows group identically.

    Returns [(row_start, row_end, joined_text), ...] with inclusive indices.
    """
    groups = []
    current: list[str] = []
    start = 0
    chars = 0
    for i, rt in enumerate(row_texts):
        if current and chars + len(rt) > max_chars and chars >= min_chars:
            groups.append((start, i - 1, '\n'.join(current)))
            current = []
            chars = 0
        if not current:
            start = i
        current.append(rt)
        chars += len(rt) + 1  # +1 for the joining newline
    if current:
        groups.append((start, len(row_texts) - 1, '\n'.join(current)))
    return groups


def _text(el) -> str:
    """Whitespace-normalized cell text: space-separated so words never
    glue across inline tags."""
    return re.sub(r'\s+', ' ', el.get_text(separator=' ', strip=True)).strip()
