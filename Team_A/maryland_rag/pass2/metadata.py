"""
Chunk metadata assembly for vector store ingestion.
Every chunk carries full provenance regardless of strategy.
"""
import json
import uuid
from datetime import datetime, timezone


def build_chunk_metadata(
    source_url: str,
    title: str | None,
    section_hierarchy: list[str] | str,
    page_classification: str,
    chunking_strategy: str,
    chunk_index: int,
    chunk_total: int,
    text: str,
    extra: dict | None = None,
) -> dict:
    """
    Assemble a complete chunk metadata record.

    Args:
        source_url: Original page/document URL.
        title: Page or document title.
        section_hierarchy: Breadcrumb path as list or JSON string.
        page_classification: From the classifier (faq, prose, etc.).
        chunking_strategy: Strategy that produced this chunk.
        chunk_index: 0-based position of this chunk within the source.
        chunk_total: Total chunks produced from this source.
        text: The chunk text content.
        extra: Optional dict of additional metadata (e.g., question/answer for FAQ).

    Returns:
        Complete chunk dict ready for vector store ingestion.
    """
    if isinstance(section_hierarchy, str):
        try:
            section_hierarchy = json.loads(section_hierarchy)
        except (json.JSONDecodeError, TypeError):
            section_hierarchy = []

    chunk = {
        'chunk_id': str(uuid.uuid4()),
        'source_url': source_url,
        'title': title,
        'section_hierarchy': section_hierarchy,
        'page_classification': page_classification,
        'chunking_strategy': chunking_strategy,
        'chunk_index': chunk_index,
        'chunk_total': chunk_total,
        'word_count': len(text.split()) if text else 0,
        'text': text,
        'date_extracted': datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        chunk.update(extra)

    return chunk


def build_chunk_metadata_multi_source(
    source_urls: list[str],
    title: str | None,
    section_hierarchy: list[str],
    page_classification: str,
    chunking_strategy: str,
    chunk_index: int,
    chunk_total: int,
    text: str,
) -> dict:
    """
    Build chunk metadata for deduplicated content that exists at multiple URLs.
    The source_url field becomes a JSON array of all source URLs.
    """
    chunk = build_chunk_metadata(
        source_url=source_urls[0],
        title=title,
        section_hierarchy=section_hierarchy,
        page_classification=page_classification,
        chunking_strategy=chunking_strategy,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
    )
    chunk['source_urls'] = source_urls
    return chunk
