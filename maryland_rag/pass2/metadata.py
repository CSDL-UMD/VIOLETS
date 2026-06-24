"""
Chunk metadata assembly for vector store ingestion.
Every chunk carries full provenance regardless of strategy.
"""
import hashlib
import json
import re
from datetime import datetime, timezone


def _normalize_text(text: str) -> str:
    """Collapse whitespace so trivially-different encodings of the same
    content hash to the same value (stable across re-ingests)."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _content_chunk_id(
    text: str,
    section_hierarchy: list[str] | None,
    chunk_index: int,
) -> str:
    """Compute a stable chunk_id from the chunk's CONTENT.

    The id depends only on the normalized chunk text plus a stable
    doc/section disambiguator (the section hierarchy + chunk position).
    It deliberately does NOT depend on the set of source URLs, so the
    same content keeps the same id across re-ingests even when the number
    or order of URLs referencing it changes. This lets pass3's
    ON CONFLICT (chunk_id) upsert replace the existing vector instead of
    orphaning it.
    """
    section_key = "/".join(section_hierarchy) if section_hierarchy else ""
    raw = f"{_normalize_text(text)}\x00{section_key}\x00{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


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

    chunk_id = _content_chunk_id(text, section_hierarchy, chunk_index)

    chunk = {
        'chunk_id': chunk_id,
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

    # Strategies emit chunk dicts that may carry strategy-specific fields
    # (e.g. FAQ pairs include 'question'/'answer'; DOCX sections include
    # 'heading_chain'). Restrict the merge to an allowlist so a strategy
    # can never overwrite the canonical chunk_id / chunk_index / text /
    # word_count values computed above.
    if extra:
        for key in ("question", "answer", "heading_chain", "table_index", "row_index"):
            if key in extra:
                chunk[key] = extra[key]

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
    The primary source_url is the first URL; the full list is preserved in
    source_urls (a JSON array surfaced to the retriever for multi-citation).

    chunk_id is computed by build_chunk_metadata from the chunk CONTENT, NOT
    from the source-URL set, so the id stays stable across re-ingests even if
    the number or order of URLs referencing this content changes. (Previously
    the id was derived from the sorted URL set, which caused the same content
    to be re-embedded under a new id — orphaning the old vector — whenever its
    duplicate set shifted.)
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
    # Preserve the full, deduped source list (order-stable) for citations.
    seen = set()
    deduped = []
    for u in source_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    chunk['source_urls'] = deduped
    return chunk
