"""
XLS / XLSX document extraction strategy.

Each sheet is emitted row-by-row as a pipe-delimited string using
"Header: Value | ..." pairs, mirroring how table_rows handles HTML tables.
Cells with no value are dropped. Sheet name prepended as a one-line header.

xlrd (legacy .xls) and openpyxl (.xlsx) are both used; extension decides which.
"""
import logging
import tempfile

logger = logging.getLogger(__name__)


def extract_xls(url: str) -> dict:
    """Return {'rows': [str, ...], 'sheets': [...]} for the spreadsheet."""
    from ..cache import get_bytes
    data = get_bytes(url)
    if not data:
        return {'rows': [], 'sheets': []}

    is_xlsx = url.lower().endswith('.xlsx') or url.lower().endswith('.xlsm')
    suffix = '.xlsx' if is_xlsx else '.xls'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        if is_xlsx:
            return _extract_xlsx(tmp.name)
        return _extract_xls(tmp.name)


def _extract_xls(path: str) -> dict:
    try:
        import xlrd
    except ImportError:
        logger.warning("xlrd not installed; cannot read .xls")
        return {'rows': [], 'sheets': []}

    rows_out = []
    sheets_out = []
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
        rows_out.extend(_format_rows(sheet.name, sheet_rows))
    return {'rows': rows_out, 'sheets': sheets_out}


def _extract_xlsx(path: str) -> dict:
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed; cannot read .xlsx")
        return {'rows': [], 'sheets': []}

    rows_out = []
    sheets_out = []
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
        rows_out.extend(_format_rows(sheet.title, sheet_rows))
    wb.close()
    return {'rows': rows_out, 'sheets': sheets_out}


def _cell_to_str(v) -> str:
    if v is None:
        return ''
    return str(v).strip()


def _format_rows(sheet_name: str, rows: list[list[str]]) -> list[str]:
    """Convert a sheet's rows into pipe-delimited chunk strings with a shared header."""
    non_empty = [r for r in rows if any(cell for cell in r)]
    if not non_empty:
        return []

    headers = non_empty[0]
    body = non_empty[1:]
    has_header = any(headers) and all(isinstance(h, str) and h for h in headers[:2])

    chunks = []
    for row in (body if has_header else non_empty):
        if has_header:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
        else:
            pairs = [str(v) for v in row if v]
        if not pairs:
            continue
        chunks.append(f"[{sheet_name}] " + ' | '.join(pairs))
    return chunks
