"""
Box materials ingestion pipeline.

Everything in needtochunk/ is assumed pre-selected. This script:
  1. Walks all files in needtochunk/ (recursively)
  2. Looks up the Box URL from url_manifest.json
  3. Extracts text (PDF / DOCX / TXT)
  4. Chunks the text
  5. Writes data/box_chunks.jsonl in pass2-compatible format for pass3

Usage:
    python -m box_ingest.ingest
    python -m box_ingest.ingest --dry-run
    python -m box_ingest.ingest --output data/box_chunks.jsonl
"""
import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NEEDTOCHUNK_DIR = PROJECT_ROOT / 'needtochunk'
MANIFEST_PATH = NEEDTOCHUNK_DIR / 'url_manifest.json'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'box_chunks.jsonl'


def load_manifest() -> dict[str, str]:
    """Load relative path → Box URL mapping from url_manifest.json."""
    if not MANIFEST_PATH.exists():
        logger.warning("url_manifest.json not found at %s", MANIFEST_PATH)
        return {}
    with open(MANIFEST_PATH, encoding='utf-8') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith('_')}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                text = '\n\n'.join(p.extract_text() or '' for p in pdf.pages).strip()
            if len(text.split()) >= 30:
                return text
        except Exception as e:
            logger.debug("pdfplumber failed for %s: %s", path.name, e)
        try:
            import fitz
            doc = fitz.open(str(path))
            text = '\n\n'.join(page.get_text() for page in doc).strip()
            doc.close()
            return text
        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", path.name, e)
            return ''

    if suffix in ('.docx', '.doc'):
        try:
            from docx import Document
            doc = Document(str(path))
            return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.warning("DOCX extraction failed for %s: %s", path.name, e)
            return ''

    if suffix == '.txt':
        return path.read_text(encoding='utf-8', errors='replace')

    logger.warning("Unsupported file type: %s", path.name)
    return ''


def chunk_text(text: str, max_words: int = 400) -> list[str]:
    if not text.strip():
        return []
    try:
        from maryland_rag.pass2.strategies.semantic import semantic_chunk
        chunks = semantic_chunk(text)
        if chunks:
            return chunks
    except Exception:
        pass
    # Paragraph-boundary fallback
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current, current_words = [], [], 0
    for para in paragraphs:
        words = len(para.split())
        if current_words + words > max_words and current:
            chunks.append('\n\n'.join(current))
            current, current_words = [], 0
        current.append(para)
        current_words += words
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


def build_chunk(text: str, source_url: str, title: str,
                chunk_index: int, chunk_total: int) -> dict:
    return {
        'chunk_id': str(uuid.uuid4()),
        'source_url': source_url,
        'title': title,
        'section_hierarchy': [],
        'page_classification': 'document',
        'chunking_strategy': 'box_ingest',
        'chunk_index': chunk_index,
        'chunk_total': chunk_total,
        'word_count': len(text.split()),
        'text': text,
        'date_extracted': datetime.now(timezone.utc).isoformat(),
    }


def run_ingest(dry_run: bool = False, output_path: Path | None = None) -> list[dict]:
    manifest = load_manifest()
    all_chunks: list[dict] = []
    missing_urls: list[str] = []

    for file_path in sorted(NEEDTOCHUNK_DIR.rglob('*')):
        if not file_path.is_file():
            continue
        if file_path.name == 'url_manifest.json':
            continue

        rel_key = str(file_path.relative_to(NEEDTOCHUNK_DIR))
        box_url = manifest.get(rel_key, '').strip()

        if not box_url:
            missing_urls.append(rel_key)
            logger.warning("No Box URL in manifest: %s", rel_key)
            continue

        if dry_run:
            print(f"  {rel_key}\n    → {box_url}")
            continue

        logger.info("Processing: %s", rel_key)
        text = extract_text(file_path)
        if not text.strip():
            logger.warning("No text extracted from %s", rel_key)
            continue

        chunks = chunk_text(text)
        for i, chunk_text_str in enumerate(chunks):
            all_chunks.append(build_chunk(
                text=chunk_text_str,
                source_url=box_url,
                title=file_path.stem,
                chunk_index=i,
                chunk_total=len(chunks),
            ))

    if missing_urls:
        logger.info("%d file(s) skipped — add Box URLs to url_manifest.json:", len(missing_urls))
        for p in missing_urls:
            logger.info("  %s", p)

    if not dry_run:
        logger.info("Total chunks produced: %d", len(all_chunks))
        if all_chunks:
            out = output_path or DEFAULT_OUTPUT
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w', encoding='utf-8') as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            logger.info("Written to %s", out)

    return all_chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    run_ingest(dry_run=args.dry_run, output_path=args.output)
