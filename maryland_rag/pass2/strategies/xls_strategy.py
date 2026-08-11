"""
XLS / XLSX / CSV document extraction strategy.

Rows are formatted as pipe-delimited "Header: Value | ..." strings,
mirroring how table_rows handles HTML tables, then batched into grouped
chunks (see table_rows.group_row_texts) with the sheet name prepended once
per batch. Cells with no value are dropped.

xlrd (legacy .xls) and openpyxl (.xlsx) read spreadsheets; the stdlib csv
module reads CSVs. The manifest content_type decides the format when
available; the file/URL suffix is the fallback.
"""
import csv
import io
import logging
import tempfile

from .table_rows import group_row_texts

logger = logging.getLogger(__name__)

# Guardrail: a spreadsheet is emitted as grouped row chunks. Bulk statistical
# tables (precinct register counts, "Eligible Active Voters by ...") run to
# hundreds or tens of thousands of rows of raw numbers that are useless for
# semantic retrieval and drown out real content in the vector store. Any
# spreadsheet that exceeds this RAW (pre-grouping) row count is treated as
# bulk tabular data and skipped entirely, with a loud warning so a genuinely
# useful file caught here gets noticed and the threshold revisited.
# Narrative/reference spreadsheets are far smaller than this. The threshold
# was tuned on one-row-per-chunk emission, so it must keep counting raw rows:
# counting grouped chunks (~20-30 rows each) would loosen it ~20x.
MAX_XLS_ROW_CHUNKS = 200


def extract_xls_from_path(path: str, label: str | None = None) -> dict:
    """Return {'rows': [str, ...], 'sheets': [...]} for a spreadsheet on disk.

    Format (.xls vs .xlsx/.xlsm) is detected from the path suffix. Spreadsheets
    that exceed MAX_XLS_ROW_CHUNKS rows are skipped (see guardrail note above).
    """
    suffix = path.lower()
    if suffix.endswith('.xlsx') or suffix.endswith('.xlsm'):
        result = _extract_xlsx(path)
    else:
        result = _extract_xls(path)
    return _apply_row_guardrail(result, label or path)


def _apply_row_guardrail(result: dict, label: str) -> dict:
    """Drop bulk spreadsheets whose RAW row count exceeds the guardrail.

    The readers record the pre-grouping row count under 'raw_row_count'
    (popped here — it is internal); the grouped-chunk count is only a
    fallback for results that never carried it."""
    n = result.pop('raw_row_count', None)
    if n is None:
        n = len(result.get('rows', []))
    if n > MAX_XLS_ROW_CHUNKS:
        logger.warning(
            "Skipping bulk spreadsheet %s: %d rows exceeds guardrail of %d "
            "(treated as raw tabular data, not chunked). Raise "
            "MAX_XLS_ROW_CHUNKS if this file is actually useful.",
            label, n, MAX_XLS_ROW_CHUNKS,
        )
        return {'rows': [], 'sheets': result.get('sheets', [])}
    return result


def extract_xls(url: str, content_type: str | None = None,
                data: bytes | None = None) -> dict:
    """Return {'rows': [str, ...], 'sheets': [...]} for the spreadsheet.

    The manifest content_type ('xls', 'xlsx', 'xlsm', 'csv') picks the
    reader; the URL suffix is only consulted when it is absent. `data` is
    used when the caller already fetched (and validated) the bytes;
    otherwise they come via the Pass 2 disk cache.
    """
    if data is None:
        from ..cache import get_bytes, kind_for_content_type
        data = get_bytes(url, expect=kind_for_content_type(content_type))
    if not data:
        return {'rows': [], 'sheets': []}

    fmt = content_type or _format_from_suffix(url)
    if fmt == 'csv':
        return extract_csv_from_bytes(data, label=url)

    suffix = '.xlsx' if fmt in ('xlsx', 'xlsm') else '.xls'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return extract_xls_from_path(tmp.name, label=url)


def extract_csv_from_bytes(data: bytes, label: str) -> dict:
    """Return {'rows': [str, ...], 'sheets': [...]} for a CSV payload,
    emitting the same grouped row-format chunks as the XLS readers.
    utf-8 (with BOM tolerance) is tried first, latin-1 as fallback."""
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = data.decode('latin-1')

    sheet_name = label.rstrip('/').rsplit('/', 1)[-1] or 'CSV'
    try:
        parsed = [
            [_cell_to_str(cell) for cell in row]
            for row in csv.reader(io.StringIO(text))
        ]
    except csv.Error as exc:
        logger.error("Failed to parse CSV %s: %s", label, exc)
        return {'rows': [], 'sheets': []}

    grouped, n_raw = _format_rows(sheet_name, parsed)
    result = {'rows': grouped, 'sheets': [sheet_name], 'raw_row_count': n_raw}
    return _apply_row_guardrail(result, label)


def _format_from_suffix(url: str) -> str:
    """Fallback format sniff from the URL suffix when the manifest has no
    content_type (e.g. Box files on disk)."""
    path = url.split('?', 1)[0].split('#', 1)[0].lower()
    for fmt in ('xlsx', 'xlsm', 'csv'):
        if path.endswith('.' + fmt):
            return fmt
    return 'xls'


def _extract_xls(path: str) -> dict:
    try:
        import xlrd
    except ImportError:
        logger.warning("xlrd not installed; cannot read .xls")
        return {'rows': [], 'sheets': []}

    rows_out = []
    sheets_out = []
    raw_total = 0
    try:
        book = xlrd.open_workbook(path)
    except Exception as exc:
        logger.error("Failed to open xls %s: %s", path, exc)
        return {'rows': [], 'sheets': []}

    for sheet in book.sheets():
        sheet_rows = [
            [_cell_to_str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            for r in range(sheet.nrows)
        ]
        sheets_out.append(sheet.name)
        grouped, n_raw = _format_rows(sheet.name, sheet_rows)
        rows_out.extend(grouped)
        raw_total += n_raw
    return {'rows': rows_out, 'sheets': sheets_out, 'raw_row_count': raw_total}


def _extract_xlsx(path: str) -> dict:
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed; cannot read .xlsx")
        return {'rows': [], 'sheets': []}

    rows_out = []
    sheets_out = []
    raw_total = 0
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:
        logger.error("Failed to open xlsx %s: %s", path, exc)
        return {'rows': [], 'sheets': []}

    for sheet in wb.worksheets:
        sheet_rows = [
            [_cell_to_str(c) for c in row]
            for row in sheet.iter_rows(values_only=True)
        ]
        sheets_out.append(sheet.title)
        grouped, n_raw = _format_rows(sheet.title, sheet_rows)
        rows_out.extend(grouped)
        raw_total += n_raw
    wb.close()
    return {'rows': rows_out, 'sheets': sheets_out, 'raw_row_count': raw_total}


def _cell_to_str(v) -> str:
    if v is None:
        return ''
    return str(v).strip()


def _format_rows(sheet_name: str, rows: list[list[str]]) -> tuple[list[str], int]:
    """Convert a sheet's rows into pipe-delimited row strings, then batch
    consecutive rows into grouped chunks with the sheet name prepended once
    per batch (same grouping bounds as HTML table_rows).

    Returns (grouped_chunks, raw_row_count) — the guardrail must count RAW
    rows, not the ~20-30x denser grouped chunks."""
    non_empty = [r for r in rows if any(cell for cell in r)]
    if not non_empty:
        return [], 0

    headers = non_empty[0]
    body = non_empty[1:]
    has_header = any(headers) and all(isinstance(h, str) and h for h in headers[:2])

    row_texts = []
    for row in (body if has_header else non_empty):
        if has_header:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
        else:
            pairs = [str(v) for v in row if v]
        if not pairs:
            continue
        row_texts.append(' | '.join(pairs))

    grouped = [
        f"[{sheet_name}]\n{batch}"
        for _start, _end, batch in group_row_texts(row_texts)
    ]
    return grouped, len(row_texts)
