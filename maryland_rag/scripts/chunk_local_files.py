"""
Chunk local PDF/DOCX files from the needtochunk/ folder.

Walks the folder recursively, extracts text from each document, and appends
chunks to data/chunks.jsonl. Bypasses the manifest DB and HTTP cache entirely
so it works on arbitrary local files.

Usage:
    python -m maryland_rag.scripts.chunk_local_files
    python -m maryland_rag.scripts.chunk_local_files --input needtochunk --output data/chunks.jsonl
    python -m maryland_rag.scripts.chunk_local_files --overwrite
"""
import argparse
import io
import json
import logging
import re
import sys
from pathlib import Path

CID_PATTERN = re.compile(r'\(cid:\d+\)')

from ..pass2.metadata import build_chunk_metadata
from ..pass2.strategies.semantic import semantic_chunk
from ..pass2.strategies.single import ingest_as_single

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "needtochunk"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "chunks.jsonl"

SHORT_DOC_WORDS = 150
OCR_DPI = 250


def ocr_pdf_local(path: Path) -> dict:
    """Rasterize each page via pymupdf and OCR with tesseract."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        logger.error("OCR dependencies missing (%s); skipping OCR for %s", exc, path.name)
        return {'text': '', 'pages': [], 'tables': []}

    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.error("OCR open failed on %s: %s", path.name, exc)
        return {'text': '', 'pages': [], 'tables': []}

    pages, parts = [], []
    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes('png')))
            text = pytesseract.image_to_string(img) or ''
        except Exception as exc:
            logger.warning("OCR failed on %s page %d: %s", path.name, i + 1, exc)
            text = ''
        pages.append({'page_num': i + 1, 'text': text})
        parts.append(text)
    doc.close()
    return {'text': '\n\n'.join(parts), 'pages': pages, 'tables': []}


def extract_pdf_local(path: Path) -> dict:
    """Extract text and tables from a local PDF via pdfplumber, fallback to pymupdf."""
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages, tables, parts = [], [], []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ''
                pages.append({'page_num': i + 1, 'text': text})
                parts.append(text)
                for t_idx, table_data in enumerate(page.extract_tables() or []):
                    if table_data and len(table_data) > 1:
                        tables.append({
                            'page_num': i + 1,
                            'table_index': t_idx,
                            'headers': table_data[0],
                            'rows': table_data[1:],
                        })
            full_text = '\n\n'.join(parts)
            if full_text.strip():
                return {'text': full_text, 'pages': pages, 'tables': tables}
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", path.name, exc)

    # Fallback: pymupdf
    try:
        import fitz
        doc = fitz.open(path)
        parts = [page.get_text() for page in doc]
        doc.close()
        full_text = '\n\n'.join(parts)
        if full_text.strip():
            return {'text': full_text, 'pages': [], 'tables': []}
    except Exception as exc:
        logger.warning("pymupdf failed on %s: %s", path.name, exc)

    # Final fallback: OCR (scanned PDFs)
    logger.info("No embedded text in %s; falling back to OCR", path.name)
    return ocr_pdf_local(path)


def extract_docx_local(path: Path) -> dict:
    """Extract heading-aware sections from a local DOCX."""
    try:
        import docx
    except ImportError:
        logger.error("python-docx not installed")
        return {'sections': [], 'full_text': ''}

    try:
        doc = docx.Document(path)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        return {'sections': [], 'full_text': ''}

    heading_levels = {f'Heading {i}': i for i in range(1, 7)}
    heading_levels['Title'] = 0

    sections, heading_stack, current, all_text = [], [], [], []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        all_text.append(text)
        style = para.style.name if para.style else ''
        level = heading_levels.get(style)
        if level is not None:
            if current:
                body = '\n\n'.join(current)
                sections.append({
                    'heading_chain': [h[1] for h in heading_stack],
                    'text': body,
                })
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            current = []
        else:
            current.append(text)

    if current:
        body = '\n\n'.join(current)
        sections.append({
            'heading_chain': [h[1] for h in heading_stack],
            'text': body,
        })

    return {'sections': sections, 'full_text': '\n\n'.join(all_text)}


def chunk_text(text: str) -> list[str]:
    """Route text to single or semantic chunking based on length."""
    if not text or not text.strip():
        return []
    if len(text.split()) <= SHORT_DOC_WORDS:
        return ingest_as_single(text)
    return semantic_chunk(text)


def chunk_file(path: Path, input_root: Path) -> list[dict]:
    """Extract, chunk, and build metadata for a single file."""
    rel_path = path.relative_to(input_root)
    folder_chain = list(rel_path.parts[:-1])
    title = path.stem
    source_url = f"local://{rel_path.as_posix()}"
    suffix = path.suffix.lower()

    chunks_out = []

    if suffix == '.pdf':
        result = extract_pdf_local(path)
        text = CID_PATTERN.sub('', result.get('text', ''))
        chunks = chunk_text(text)
        section_hierarchy = folder_chain
        for i, c in enumerate(chunks):
            chunks_out.append(build_chunk_metadata(
                source_url=source_url,
                title=title,
                section_hierarchy=section_hierarchy,
                page_classification='local_document',
                chunking_strategy='local_pdf',
                chunk_index=i,
                chunk_total=len(chunks),
                text=c,
            ))

    elif suffix in ('.docx', '.doc'):
        result = extract_docx_local(path)
        sections = result.get('sections') or []
        if not sections:
            chunks = chunk_text(result.get('full_text', ''))
            for i, c in enumerate(chunks):
                chunks_out.append(build_chunk_metadata(
                    source_url=source_url,
                    title=title,
                    section_hierarchy=folder_chain,
                    page_classification='local_document',
                    chunking_strategy='local_docx',
                    chunk_index=i,
                    chunk_total=len(chunks),
                    text=c,
                ))
        else:
            section_chunks = []
            for section in sections:
                s_text = section.get('text', '').strip()
                if not s_text:
                    continue
                heading_chain = section.get('heading_chain', [])
                hierarchy = folder_chain + heading_chain
                sub = chunk_text(s_text)
                for c in sub:
                    section_chunks.append((hierarchy, c))
            for i, (hierarchy, c) in enumerate(section_chunks):
                chunks_out.append(build_chunk_metadata(
                    source_url=source_url,
                    title=title,
                    section_hierarchy=hierarchy,
                    page_classification='local_document',
                    chunking_strategy='local_docx',
                    chunk_index=i,
                    chunk_total=len(section_chunks),
                    text=c,
                ))
    elif suffix in ('.txt', '.md'):
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = path.read_text(encoding='latin-1')
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            chunks_out.append(build_chunk_metadata(
                source_url=source_url,
                title=title,
                section_hierarchy=folder_chain,
                page_classification='local_document',
                chunking_strategy='local_text',
                chunk_index=i,
                chunk_total=len(chunks),
                text=c,
            ))
    else:
        logger.debug("Skipping unsupported file type: %s", path)
        return []

    return chunks_out


def walk_input(input_root: Path) -> list[Path]:
    """Collect all PDFs, DOCXs, and plain text files under input_root."""
    files = []
    for ext in ('*.pdf', '*.docx', '*.doc', '*.txt', '*.md',
                '*.PDF', '*.DOCX', '*.DOC', '*.TXT', '*.MD'):
        files.extend(input_root.rglob(ext))
    return sorted(f for f in files if f.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Root folder to walk (default: needtochunk/)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='JSONL output path (default: data/chunks.jsonl)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output instead of appending')
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input folder not found: %s", args.input)
        sys.exit(1)

    files = walk_input(args.input)
    logger.info("Found %d files under %s", len(files), args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if args.overwrite else 'a'

    total_chunks, total_files_ok, total_files_empty = 0, 0, 0
    with open(args.output, mode, encoding='utf-8') as f:
        for path in files:
            logger.info("Chunking: %s", path.relative_to(args.input))
            try:
                chunks = chunk_file(path, args.input)
            except Exception as exc:
                logger.error("Failed on %s: %s", path.name, exc, exc_info=True)
                continue
            if not chunks:
                total_files_empty += 1
                logger.warning("No chunks produced for %s", path.name)
                continue
            total_files_ok += 1
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            total_chunks += len(chunks)

    logger.info("Done. Wrote %d chunks from %d files (%d empty) to %s",
                total_chunks, total_files_ok, total_files_empty, args.output)


if __name__ == '__main__':
    main()
