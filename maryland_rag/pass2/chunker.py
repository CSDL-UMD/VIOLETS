"""
Pass 2 orchestration: reads the manifest from Pass 1 and routes each page
to its assigned chunking strategy, producing chunk records with full metadata.

Handles deduplication: pages with identical content_hash are extracted once
but all source URLs are preserved in chunk metadata.
"""
import json
import logging
import os
from collections import defaultdict

from ..pass1.config import DB_PATH, REQUEST_TIMEOUT
from ..pass1.db import DB
from .metadata import build_chunk_metadata, build_chunk_metadata_multi_source
from .strategies.faq import extract_qa_pairs
from .strategies.semantic import semantic_chunk
from .strategies.single import ingest_as_single
from .strategies.simple_split import simple_split
from .strategies.table_rows import extract_table_chunks
from .strategies.pdf import extract_pdf
from .strategies.docx_strategy import extract_docx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def run_pass2(
    db_path: str | None = None,
    only_changed: bool = False,
    output_path: str | None = None,
) -> list[dict]:
    """
    Execute Pass 2: extract and chunk all crawled pages.

    Args:
        db_path: Path to manifest DB (defaults to config).
        only_changed: If True, only process pages whose content changed since last crawl.
        output_path: If provided, write chunks as JSONL to this path.

    Returns:
        List of chunk dicts ready for vector store ingestion.
    """
    db = DB(db_path)

    # --- Deduplication: group pages by content_hash ---
    duplicates = _build_dedup_map(db)

    # --- Get pages to process ---
    if only_changed:
        pages = db.get_changed_pages()
        logger.info("Processing %d changed pages", len(pages))
    else:
        pages = db.get_crawled_pages()
        logger.info("Processing %d crawled pages", len(pages))

    # Track which content_hashes we've already processed
    processed_hashes = set()
    all_chunks = []

    for page in pages:
        url = page['url']
        content_hash = page['content_hash']
        strategy = page['chunking_strategy']

        # Skip duplicates we've already processed
        if content_hash and content_hash in processed_hashes:
            logger.debug("Skipping duplicate: %s", url)
            continue

        if strategy == 'skip':
            logger.debug("Skipping (strategy=skip): %s", url)
            continue

        if page['content_type'] != 'html':
            logger.debug("Skipping non-HTML (%s): %s", page['content_type'], url)
            continue

        logger.info("Chunking [%s]: %s", strategy, url)

        try:
            chunks = _route_to_strategy(page)
        except Exception as exc:
            logger.error("Failed to chunk %s: %s", url, exc, exc_info=True)
            continue

        if not chunks:
            logger.warning("No chunks produced for %s", url)
            continue

        # Determine all source URLs for deduplicated content
        source_urls = duplicates.get(content_hash, [url]) if content_hash else [url]

        # Build metadata-enriched chunks
        section_hierarchy = page['section_hierarchy'] or '[]'
        for i, chunk in enumerate(chunks):
            text = chunk if isinstance(chunk, str) else chunk.get('text', '')
            extra = chunk if isinstance(chunk, dict) else None

            if len(source_urls) > 1:
                meta_chunk = build_chunk_metadata_multi_source(
                    source_urls=source_urls,
                    title=page['title'],
                    section_hierarchy=json.loads(section_hierarchy) if isinstance(section_hierarchy, str) else section_hierarchy,
                    page_classification=page['page_classification'],
                    chunking_strategy=strategy,
                    chunk_index=i,
                    chunk_total=len(chunks),
                    text=text,
                )
            else:
                meta_chunk = build_chunk_metadata(
                    source_url=url,
                    title=page['title'],
                    section_hierarchy=section_hierarchy,
                    page_classification=page['page_classification'],
                    chunking_strategy=strategy,
                    chunk_index=i,
                    chunk_total=len(chunks),
                    text=text,
                    extra=extra if isinstance(extra, dict) else None,
                )

            all_chunks.append(meta_chunk)

        if content_hash:
            processed_hashes.add(content_hash)

    logger.info("Pass 2 complete. Total chunks: %d", len(all_chunks))

    # Optionally write to JSONL
    if output_path:
        _write_jsonl(all_chunks, output_path)
        logger.info("Chunks written to %s", output_path)

    db.close()
    return all_chunks


def _route_to_strategy(page) -> list:
    """Route a page to its assigned chunking strategy."""
    strategy = page['chunking_strategy']
    url = page['url']
    content_type = page['content_type']

    if strategy == 'qa_pairs':
        pairs = extract_qa_pairs(url)
        if pairs:
            return pairs
        # Fallback: if FAQ extraction fails, use simple split
        text = _fetch_text(url)
        return [{'text': t} for t in simple_split(text)] if text else []

    if strategy == 'ingest_as_single':
        text = _fetch_text(url)
        chunks = ingest_as_single(text)
        return [{'text': t} for t in chunks]

    if strategy == 'simple_split':
        text = _fetch_text(url)
        chunks = simple_split(text)
        return [{'text': t} for t in chunks]

    if strategy == 'semantic_with_overlap':
        text = _fetch_text(url)
        chunks = semantic_chunk(text)
        return [{'text': t} for t in chunks]

    if strategy == 'table_rows':
        return extract_table_chunks(url)

    if strategy == 'document_extraction':
        return _extract_document(url, content_type, page['needs_ocr'] or 0)

    logger.warning("Unknown strategy '%s' for %s, skipping", strategy, url)
    return []


def _extract_document(url: str, content_type: str, needs_ocr: int) -> list:
    """Extract and chunk a document (PDF or DOCX)."""
    if content_type == 'pdf':
        result = extract_pdf(url, needs_ocr=bool(needs_ocr))
        text = result.get('text', '')
        if not text.strip():
            return []

        structure = result.get('structure_type', 'prose')

        # Route extracted PDF text through appropriate text strategy
        if structure == 'faq':
            # Wrap text in minimal HTML for FAQ extractor
            # Fall through to semantic chunking since we have plain text
            chunks = semantic_chunk(text)
        elif structure == 'table_heavy' and result.get('tables'):
            # Format tables as chunks
            chunks_list = []
            for table in result['tables']:
                headers = table.get('headers', [])
                for row in table.get('rows', []):
                    if headers and len(row) == len(headers):
                        pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
                        chunks_list.append(' | '.join(pairs))
                    else:
                        chunks_list.append(' | '.join(str(c) for c in row if c))
            return [{'text': t} for t in chunks_list if t.strip()]
        elif structure == 'short':
            chunks = ingest_as_single(text)
        else:
            chunks = semantic_chunk(text)

        return [{'text': t} for t in chunks]

    elif content_type in ('docx', 'doc'):
        result = extract_docx(url)
        sections = result.get('sections', [])
        if not sections:
            text = result.get('full_text', '')
            if text:
                chunks = semantic_chunk(text)
                return [{'text': t} for t in chunks]
            return []

        # Each DOCX section becomes one or more chunks
        all_chunks = []
        for section in sections:
            text = section.get('text', '')
            if not text.strip():
                continue
            word_count = section.get('word_count', 0)
            heading_chain = section.get('heading_chain', [])

            if word_count <= 300:
                all_chunks.append({
                    'text': text,
                    'heading_chain': heading_chain,
                })
            else:
                sub_chunks = semantic_chunk(text)
                for sc in sub_chunks:
                    all_chunks.append({
                        'text': sc,
                        'heading_chain': heading_chain,
                    })
        return all_chunks

    else:
        logger.warning("Unsupported document type: %s", content_type)
        return []


def _fetch_text(url: str) -> str:
    """Fetch a page via the disk cache and extract text with trafilatura."""
    try:
        import trafilatura
        from .cache import get_html
        html = get_html(url)
        if not html:
            return ''
        text = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_images=False,
        )
        return text or ''
    except Exception as exc:
        logger.warning("Failed to fetch text for %s: %s", url, exc)
        return ''


def _build_dedup_map(db: DB) -> dict[str, list[str]]:
    """Build a mapping of content_hash → list of URLs for deduplication."""
    dupes = db.get_duplicate_hashes()
    dedup_map = {}
    for row in dupes:
        urls = row['urls'].split(',')
        dedup_map[row['content_hash']] = urls
    return dedup_map


def _write_jsonl(chunks: list[dict], path: str):
    """Write chunks as newline-delimited JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    chunks = run_pass2(output_path='data/chunks.jsonl')
    print(f"Produced {len(chunks)} chunks")
