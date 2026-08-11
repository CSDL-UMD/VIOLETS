"""
Pass 2 orchestration: reads the manifest from Pass 1 and routes each page
to its assigned chunking strategy, producing chunk records with full metadata.

Handles deduplication: pages with identical content_hash are extracted once
but all source URLs are preserved in chunk metadata.
"""
import glob
import hashlib
import json
import logging
import os
import re
from collections import defaultdict

from ..pass1.config import DB_PATH, REQUEST_TIMEOUT
from ..pass1.db import DB
from .cache import get_bytes, kind_for_content_type, purge_poisoned
from .langfilter import is_non_english
from .metadata import build_chunk_metadata, build_chunk_metadata_multi_source
from .strategies.faq import extract_qa_pairs
from .strategies.semantic import enforce_chunk_caps, semantic_chunk
from .strategies.single import ingest_as_single
from .strategies.simple_split import simple_split
from .strategies.table_rows import extract_table_chunks
from .strategies.pdf import extract_pdf
from .strategies.docx_strategy import extract_docx
from .strategies.xls_strategy import extract_xls

logger = logging.getLogger(__name__)

# Minimum chunk BODY length (chars, excluding any leading "[caption]" context
# line — the table strategies' captions would otherwise push every fragment
# past any floor). Below 50 chars the corpus holds only extraction debris:
# bare page numbers, form-field labels ("For Board Use Only"), and glued
# header fragments. Real content — FAQ answers, table rows with inlined
# headers — starts well above it, so this floor removes junk without
# touching anything retrievable.
MIN_CHUNK_BODY_CHARS = 50


def _chunk_body(text: str) -> str:
    """Return the chunk text minus a leading '[...]' context line, if any."""
    if text.startswith('[') and '\n' in text:
        first, rest = text.split('\n', 1)
        if first.rstrip().endswith(']'):
            return rest
    return text


class _FetchFailure(Exception):
    """A page's bytes/HTML could not be obtained or validated this run
    (cache miss + live fetch failed, or poisoned payload). Distinct from a
    page that legitimately yields zero chunks: the page's existing chunks
    must be KEPT in a --changed merge, and it must stay out of the
    covered/snapshot set so it is retried."""


def run_pass2(
    db_path: str | None = None,
    only_changed: bool = False,
    output_path: str | None = None,
) -> list[dict]:
    """
    Execute Pass 2: extract and chunk all crawled pages.

    Args:
        db_path: Path to manifest DB (defaults to config).
        only_changed: If True, only re-chunk pages whose content changed
            since it was last reflected in the output file. Documents are
            always cheaply revalidated (conditional GET + hash compare)
            and re-chunked only when genuinely changed. After a successful
            merge+write, covered pages are snapshotted so the next
            --changed run skips them; failed pages keep their existing
            chunks and are retried.
        output_path: If provided, write chunks as JSONL to this path.

    Returns:
        List of chunk dicts ready for vector store ingestion. For a
        --changed run with an existing output file this is the full MERGED
        corpus (kept + regenerated chunks), i.e. exactly what was written.
    """
    db = DB(db_path)

    # Purge poisoned cache entries up front (HTML nav pages cached under
    # document URLs) so every document fetch this run starts from clean
    # bytes instead of re-extracting junk.
    purge_poisoned(
        (p['url'], kind_for_content_type(p['content_type']))
        for p in db.get_crawled_pages()
        if p['chunking_strategy'] == 'document_extraction'
    )

    # --- Deduplication: group pages by content_hash ---
    duplicates = _build_dedup_map(db)

    # --- Get pages to process ---
    forced_partner_urls: set[str] = set()
    if only_changed:
        pages = db.get_changed_pages()
        logger.info("Processing %d changed pages", len(pages))
        if output_path:
            pages, forced_partner_urls = _expand_with_dedup_partners(
                pages, db, output_path,
            )
    else:
        pages = db.get_crawled_pages()
        logger.info("Processing %d crawled pages", len(pages))

    # Track which content_hashes we've already processed
    processed_hashes = set()
    all_chunks = []
    # Per-page outcome tracking. Each processed page lands in one bucket:
    #   - success: >= 1 chunk regenerated;
    #   - deliberate zero: legitimately no chunks (skip strategy, legacy
    #     .doc, langfilter dropped everything, genuinely empty document) —
    #     the zero-chunk state IS the page's reflected state;
    #   - unchanged (documents, --changed only): revalidated bytes still
    #     hash to previous_content_hash, existing rows kept as-is;
    #   - failed: bytes/HTML unobtainable or extraction crashed.
    # Success and deliberate-zero go into reprocessed_urls so the --changed
    # merge drops their old rows; failed pages stay OUT of it (their
    # existing chunks are kept) and out of covered_urls (so they are not
    # snapshotted and get retried). Duplicate partners share their
    # primary's outcome.
    reprocessed_urls = set()
    deliberate_zero_urls = set()
    covered_urls = set()
    failed_urls = set()

    for page in pages:
        url = page['url']
        content_hash = page['content_hash']
        content_type = page['content_type']
        strategy = page['chunking_strategy']

        # All URLs sharing this page's content (dedup partners) — they
        # share its outcome below.
        source_urls = duplicates.get(content_hash, [url]) if content_hash else [url]

        # Skip duplicates we've already processed
        if content_hash and content_hash in processed_hashes:
            logger.debug("Skipping duplicate: %s", url)
            continue

        if strategy == 'skip':
            logger.debug("Skipping (strategy=skip): %s", url)
            deliberate_zero_urls.add(url)
            covered_urls.add(url)
            reprocessed_urls.add(url)
            continue

        # --changed document revalidation: get_changed_pages() returns
        # EVERY document row because the manifest cannot see server-side
        # document changes on its own. The cache's ETag/Last-Modified
        # revalidation makes an unchanged document a cheap 304, and when
        # its bytes still hash to previous_content_hash (i.e. this exact
        # content is already reflected in chunks.jsonl) re-chunking is
        # skipped: existing rows are kept in the merge (NOT reprocessed)
        # but the page still counts as covered. Forced dedup partners must
        # re-chunk regardless — their old shared rows are being dropped.
        if (only_changed and strategy == 'document_extraction'
                and content_type not in (None, 'html', 'doc')
                and url not in forced_partner_urls):
            data = get_bytes(url, expect=kind_for_content_type(content_type))
            if not data:
                logger.warning(
                    "Could not fetch document %s — keeping its existing "
                    "chunks and retrying next --changed run", url,
                )
                failed_urls.add(url)
                continue
            digest = hashlib.sha256(data).hexdigest()
            db.update_document_hash(url, digest)
            if digest == page['previous_content_hash']:
                logger.debug("Document unchanged, keeping chunks: %s", url)
                covered_urls.add(url)
                continue

        logger.info("Chunking [%s %s]: %s", content_type, strategy, url)

        try:
            chunks, effective_strategy = _route_to_strategy(page, db)
        except _FetchFailure as exc:
            logger.warning(
                "Fetch failed for %s — keeping its existing chunks: %s",
                url, exc,
            )
            failed_urls.add(url)
            continue
        except Exception as exc:
            logger.error("Failed to chunk %s: %s", url, exc, exc_info=True)
            failed_urls.add(url)
            continue

        # Freshly-discovered documents enter the loop with content_hash
        # NULL (pass1 only HEADs them); _route_to_strategy just fetched
        # their bytes and stored the real hash. Re-read it so a second URL
        # serving identical bytes this run (e.g. a renamed or case-variant
        # path) dedups instead of chunking twice — the run-start dedup map
        # cannot see these hashes.
        if not content_hash:
            content_hash = db.get_content_hash(url)
            if content_hash and content_hash in processed_hashes:
                logger.info(
                    "Skipping duplicate (hash computed mid-run): %s", url)
                covered_urls.add(url)
                reprocessed_urls.add(url)
                continue

        # Content-based net for mixed-script extraction salad — translated
        # documents themselves are already excluded by URL in pass1.
        kept = []
        for chunk in chunks:
            text = chunk if isinstance(chunk, str) else chunk.get('text', '')
            if is_non_english(text):
                logger.info("Dropped non-English chunk from %s: %.60r", url, text)
            elif len(_chunk_body(text).strip()) < MIN_CHUNK_BODY_CHARS:
                logger.info("Dropped under-length chunk from %s: %.60r", url, text)
            else:
                kept.append(chunk)
        chunks = kept

        if not chunks:
            # Deliberate zero (fetch failures raised _FetchFailure above):
            # "no chunks" is this page's reflected state, so its old rows
            # are dropped from the merge and it IS snapshotted — otherwise
            # it would be refetched and re-processed on every --changed run.
            logger.warning("No chunks produced for %s", url)
            reprocessed_urls.update(source_urls)
            deliberate_zero_urls.update(source_urls)
            covered_urls.update(source_urls)
            if content_hash:
                processed_hashes.add(content_hash)
            continue

        # Build metadata-enriched chunks
        section_hierarchy = page['section_hierarchy'] or '[]'
        if isinstance(section_hierarchy, str):
            try:
                hierarchy = json.loads(section_hierarchy)
            except (json.JSONDecodeError, TypeError):
                hierarchy = []
        else:
            hierarchy = section_hierarchy

        for i, chunk in enumerate(chunks):
            text = chunk if isinstance(chunk, str) else chunk.get('text', '')
            extra = chunk if isinstance(chunk, dict) else None

            # Context header: prose/FAQ/document chunks embed bare text with
            # no page context. Prepend a "[<title> — <section>]" line unless
            # the chunk already carries a bracketed caption/sheet line (the
            # table strategies' convention) or starts with the title itself.
            text = _prepend_context(
                text, page['title'], hierarchy,
                extra.get('heading_chain') if isinstance(extra, dict) else None,
            )

            if len(source_urls) > 1:
                meta_chunk = build_chunk_metadata_multi_source(
                    source_urls=source_urls,
                    title=page['title'],
                    section_hierarchy=hierarchy,
                    page_classification=page['page_classification'],
                    chunking_strategy=effective_strategy,
                    chunk_index=i,
                    chunk_total=len(chunks),
                    text=text,
                    extra=extra if isinstance(extra, dict) else None,
                )
            else:
                meta_chunk = build_chunk_metadata(
                    source_url=url,
                    title=page['title'],
                    section_hierarchy=hierarchy,
                    page_classification=page['page_classification'],
                    chunking_strategy=effective_strategy,
                    chunk_index=i,
                    chunk_total=len(chunks),
                    text=text,
                    extra=extra if isinstance(extra, dict) else None,
                )

            all_chunks.append(meta_chunk)

        reprocessed_urls.update(source_urls)
        covered_urls.update(source_urls)
        if content_hash:
            processed_hashes.add(content_hash)

    logger.info("Pass 2 complete. Total chunks: %d", len(all_chunks))

    # Optionally write to JSONL. Incremental (--changed) runs MERGE into the
    # existing corpus file instead of truncating it to just the changed
    # pages' chunks; full runs keep the replace-everything behavior.
    if output_path:
        if only_changed:
            all_chunks = _merge_into_existing(
                all_chunks, output_path, reprocessed_urls,
                db.get_expected_chunk_urls(),
            )
        _write_jsonl(all_chunks, output_path)
        logger.info("Chunks written to %s", output_path)

        if only_changed:
            # The merged file now reflects every covered page (success,
            # deliberate-zero, unchanged documents) — snapshot them so the
            # next --changed run stops returning HTML pages whose content
            # is unchanged and stops re-chunking revalidated documents.
            db.snapshot_hashes_for_recrawl(urls=covered_urls)
            if failed_urls:
                # Failed pages must be retried: NULL their snapshot AFTER
                # the covered snapshot so even an unchanged forced partner
                # that failed re-enters the --changed set.
                db.clear_snapshot(failed_urls)
                logger.warning(
                    "%d page(s) failed reprocessing — existing chunks kept, "
                    "will retry next --changed run", len(failed_urls),
                )
        elif deliberate_zero_urls:
            # Full run: successful pages are snapshotted by the caller
            # after its coverage gate (__main__), but deliberate-zero
            # pages produce no chunks so that snapshot never sees them.
            # Their zero-chunk state IS reflected in the file just written,
            # so snapshot them here — otherwise every --changed run
            # refetches and re-processes them forever.
            db.snapshot_hashes_for_recrawl(urls=deliberate_zero_urls)

    db.close()
    return all_chunks


def _route_to_strategy(page, db: DB) -> tuple[list, str]:
    """
    Route a page to its assigned chunking strategy.

    Returns (chunks, effective_strategy). The second element is what
    ACTUALLY produced the chunks, which is not always the assigned
    strategy: qa_pairs and table_rows both fall back to simple_split when
    their structure is absent. Recording the assigned strategy in that
    case made silent fallbacks invisible in the corpus metadata — pages
    were labelled 'qa_pairs' while holding blind text splits.
    """
    strategy = page['chunking_strategy']
    url = page['url']
    content_type = page['content_type']

    if strategy == 'qa_pairs':
        pairs = extract_qa_pairs(url)
        if pairs:
            return pairs, strategy
        # Fallback: if FAQ extraction fails, use simple split
        logger.warning(
            "FAQ extraction found no Q/A structure in %s — "
            "falling back to simple_split", url,
        )
        text = _fetch_text(url)
        chunks = [{'text': t} for t in simple_split(text)] if text else []
        return chunks, 'simple_split'

    if strategy == 'ingest_as_single':
        text = _fetch_text(url)
        chunks = ingest_as_single(text)
        return [{'text': t} for t in chunks], strategy

    if strategy == 'simple_split':
        text = _fetch_text(url)
        chunks = simple_split(text)
        return [{'text': t} for t in chunks], strategy

    if strategy == 'semantic_with_overlap':
        text = _fetch_text(url)
        chunks = semantic_chunk(text)
        return [{'text': t} for t in chunks], strategy

    if strategy == 'table_rows':
        chunks = extract_table_chunks(url)
        if chunks:
            return chunks, strategy
        # Fallback: misrouted page with no <table>, use simple split
        logger.warning(
            "No tables found in %s — falling back to simple_split", url,
        )
        text = _fetch_text(url)
        chunks = [{'text': t} for t in simple_split(text)] if text else []
        return chunks, 'simple_split'

    if strategy == 'document_extraction':
        return _extract_document(url, content_type, db), strategy

    logger.warning("Unknown strategy '%s' for %s, skipping", strategy, url)
    return [], strategy


def _extract_document(url: str, content_type: str, db: DB) -> list:
    """Extract and chunk a document (PDF, DOCX, or spreadsheet/CSV)."""
    if content_type == 'doc':
        # python-docx cannot parse the legacy binary .doc container; a junk
        # parse would poison the corpus, so skip loudly instead.
        logger.warning(
            "Skipping legacy .doc document (no supported extractor — "
            "convert to .docx or PDF to ingest): %s", url,
        )
        return []
    if content_type not in ('pdf', 'docx', 'xls', 'xlsx', 'xlsm', 'csv'):
        logger.warning("Unsupported document type: %s", content_type)
        return []

    # Fetch once, centrally: get_bytes magic-validates the payload against
    # the manifest content_type (deleting + refetching a poisoned cache
    # entry), so extraction never sees an HTML nav page served in place of
    # the document. The validated bytes' sha256 is persisted as the page's
    # content_hash so `pass2 --changed` and content-hash dedup are
    # meaningful for documents (pass1 only HEAD-probes them).
    data = get_bytes(url, expect=kind_for_content_type(content_type))
    if not data:
        raise _FetchFailure(
            f"no valid {content_type} bytes for {url} "
            "(cache miss + fetch failed, or payload failed magic check)"
        )
    db.update_document_hash(url, hashlib.sha256(data).hexdigest())

    if content_type == 'pdf':
        # Digital extraction first, OCR fallback for image-only PDFs.
        # The manifest's stale needs_ocr flag is deliberately ignored.
        result = extract_pdf(url, ocr_fallback=True, data=data)
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
            # Emit table-row chunks AND the narrative prose: report PDFs
            # often mix a couple of summary tables with pages of text, and
            # the prose would otherwise be dropped entirely.
            row_chunks = []
            for table in result['tables']:
                headers = table.get('headers') or []
                for row in table.get('rows', []):
                    row_text = _format_table_row(headers, row)
                    if row_text.strip():
                        row_chunks.append(row_text)
            row_chunks = enforce_chunk_caps(row_chunks)
            prose_chunks = _dedupe_against_rows(semantic_chunk(text), row_chunks)
            return [{'text': t} for t in row_chunks + prose_chunks]
        elif structure == 'short':
            chunks = ingest_as_single(text)
        else:
            chunks = semantic_chunk(text)

        return [{'text': t} for t in chunks]

    elif content_type == 'docx':
        result = extract_docx(url, data=data)
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
                for part in enforce_chunk_caps([text]):
                    all_chunks.append({
                        'text': part,
                        'heading_chain': heading_chain,
                    })
            else:
                sub_chunks = semantic_chunk(text)
                for sc in sub_chunks:
                    all_chunks.append({
                        'text': sc,
                        'heading_chain': heading_chain,
                    })

        # Also chunk any tables extracted from the DOCX
        for table in result.get('tables', []):
            headers = table.get('headers') or []
            for row in table.get('rows', []):
                row_text = _format_table_row(headers, row)
                if row_text.strip():
                    all_chunks.append({'text': row_text})

        return all_chunks

    else:  # xls / xlsx / xlsm / csv — validated against the allowlist above
        result = extract_xls(url, content_type=content_type, data=data)
        rows = [r for r in result.get('rows', []) if r.strip()]
        return [{'text': t} for t in enforce_chunk_caps(rows)]


def _format_table_row(headers: list, row: list) -> str:
    """
    Format one extracted table row as 'Header: value | ...' text.

    pdfplumber/docx header cells can be None or hold embedded newlines, so
    headers are coerced to stripped single-line strings and a 'Header:'
    prefix is only emitted when the header is truthy — values under a
    missing header are kept bare instead of prefixed with literal 'None:'.
    """
    if headers and len(row) == len(headers):
        pairs = []
        for h, v in zip(headers, row):
            h_txt = ' '.join(str(h).split()) if h else ''
            v_txt = str(v).strip() if v else ''
            if not v_txt:
                continue
            pairs.append(f"{h_txt}: {v_txt}" if h_txt else v_txt)
        return ' | '.join(pairs)
    return ' | '.join(str(c).strip() for c in row if c)


def _dedupe_against_rows(prose_chunks: list[str], row_chunks: list[str]) -> list[str]:
    """
    Cheap dedupe for table_heavy PDFs: drop prose chunks whose tokens are
    >80% contained in the emitted table rows. pdfplumber's page text
    includes the table cells, so a pure-table page would otherwise be
    emitted twice (once as rows, once as 'prose').
    """
    row_tokens = set()
    for r in row_chunks:
        for tok in r.lower().split():
            tok = re.sub(r'\W+', '', tok)
            if tok:
                row_tokens.add(tok)

    kept = []
    for chunk in prose_chunks:
        tokens = [t for t in (re.sub(r'\W+', '', w) for w in chunk.lower().split()) if t]
        if tokens:
            contained = sum(1 for t in tokens if t in row_tokens)
            if contained / len(tokens) > 0.8:
                continue
        kept.append(chunk)
    return kept


def _fetch_text(url: str) -> str:
    """Fetch a page via the disk cache and extract text with trafilatura.

    Raises _FetchFailure when the HTML cannot be obtained (cache miss and
    the live fetch failed) so the caller treats the page as FAILED rather
    than deliberately empty; returns '' only when the page fetched fine but
    yielded no extractable text."""
    import trafilatura
    from .cache import get_html
    html = get_html(url)
    if not html:
        raise _FetchFailure(f"no HTML available for {url}")
    try:
        text = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_images=False,
        )
    except Exception as exc:
        raise _FetchFailure(f"text extraction failed for {url}: {exc}") from exc
    return text or ''


def _build_dedup_map(db: DB) -> dict[str, list[str]]:
    """Build a mapping of content_hash → list of URLs for deduplication."""
    dupes = db.get_duplicate_hashes()
    dedup_map = {}
    for row in dupes:
        urls = row['urls'].split(',')
        # SQLite GROUP_CONCAT is unbounded by default (only capped by
        # SQLITE_MAX_LENGTH, ~1GB), so truncation is not a real risk for the
        # small duplicate sets seen here. Defensively flag any mismatch between
        # the COUNT(*) and the number of concatenated URLs, which would signal
        # a dropped/empty URL or (theoretically) a truncated field.
        expected = row['dupes']
        if expected is not None and len(urls) != expected:
            logger.warning(
                "Dedup URL count mismatch for hash %s: expected %d, parsed %d "
                "(possible GROUP_CONCAT truncation or empty URL)",
                row['content_hash'], expected, len(urls),
            )
        dedup_map[row['content_hash']] = urls
    return dedup_map


def _prepend_context(text: str, title: str | None,
                     section_hierarchy: list | None,
                     heading_chain: list | None = None) -> str:
    """
    Prepend a one-line "[<title> — <section>]" context header so bare
    prose/FAQ/document chunks carry their page context into the embedding,
    formatted like the caption/sheet line the table strategies already emit.
    Skipped when the chunk already opens with a bracketed context line or
    with the page title itself.
    """
    if not text or not title or not str(title).strip():
        return text
    title_txt = ' '.join(str(title).split())
    stripped = text.lstrip()
    if stripped.startswith('['):
        return text  # already carries a caption/sheet/context line
    # Word-boundary match: a chunk beginning with the page title itself
    # suppresses the header, but a title that is merely a PREFIX of the
    # first word must not (title "Vote" vs text "Voters must...").
    if re.match(re.escape(title_txt) + r'(?!\w)', stripped, re.IGNORECASE):
        return text  # chunk already begins with the page title

    # Section: innermost DOCX heading when available, else the breadcrumb
    # leaf — but never just the title repeated, and never a filename-like
    # leaf ("Stats.Html", "Index.Html" from URL-derived breadcrumbs), which
    # would pollute every chunk's header with junk.
    section = None
    if heading_chain:
        section = ' '.join(str(heading_chain[-1]).split())
    elif section_hierarchy:
        section = ' '.join(str(section_hierarchy[-1]).split())
    if section and re.search(r'\.\w{1,5}$', section):
        section = None  # filename-like leaf: fall back to title-only
    if section and section.lower() != title_txt.lower():
        return f"[{title_txt} — {section}]\n{text}"
    return f"[{title_txt}]\n{text}"


def _expand_with_dedup_partners(pages: list, db: DB,
                                output_path: str) -> tuple[list, set[str]]:
    """
    --changed only: a reprocessed page drops ALL its existing corpus rows,
    including multi-source (dedup) rows that also cite unchanged partner
    pages — the partners would silently lose their coverage and, being
    snapshotted, never be revisited. Scan the existing corpus for
    multi-source rows sharing a URL with a page being reprocessed
    (transitively) and add the missing partner pages to this run so they
    are re-chunked alongside it.

    Returns (pages + partner rows, set of partner URLs added). Partner URLs
    no longer crawled in the manifest are not added — their stale rows are
    dropped by the merge's expected-set filter anyway.
    """
    if not os.path.exists(output_path):
        return pages, set()

    groups = []
    with open(output_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            urls = json.loads(line).get('source_urls') or []
            if len(urls) > 1:
                groups.append(set(urls))
    if not groups:
        return pages, set()

    selected = {p['url'] for p in pages}
    closure = set(selected)
    grew = True
    while grew:  # transitive closure over shared multi-source rows
        grew = False
        for group in groups:
            if group & closure and not group <= closure:
                closure |= group
                grew = True

    partners = db.get_pages_by_urls(closure - selected)
    if partners:
        logger.info(
            "--changed: pulling in %d dedup partner page(s) whose shared "
            "rows are being reprocessed", len(partners),
        )
    return list(pages) + partners, {p['url'] for p in partners}


def _merge_into_existing(new_chunks: list[dict], output_path: str,
                         reprocessed_urls: set[str],
                         expected_urls: set[str]) -> list[dict]:
    """
    Merge an incremental (--changed) run into the existing corpus file:
    keep every existing row EXCEPT those citing a reprocessed page (their
    chunks were just regenerated — possibly as zero chunks) or citing no
    page still in the manifest's chunkable set (stale), then append the new
    chunks. Returns the full merged corpus for the atomic write.
    """
    if not os.path.exists(output_path):
        return new_chunks

    existing = []
    with open(output_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))

    kept = []
    for chunk in existing:
        urls = set(chunk.get('source_urls') or [])
        if chunk.get('source_url'):
            urls.add(chunk['source_url'])
        if urls & reprocessed_urls:
            continue  # regenerated (or intentionally dropped) this run
        if not urls & expected_urls:
            continue  # no longer in the manifest's chunkable set
        kept.append(chunk)

    logger.info(
        "--changed merge: kept %d existing + %d regenerated chunks "
        "(dropped %d stale/replaced)",
        len(kept), len(new_chunks), len(existing) - len(kept),
    )
    return kept + new_chunks


def _write_jsonl(chunks: list[dict], path: str):
    """Write chunks as newline-delimited JSON, atomically: write a temp
    file in the same directory then os.replace (mirroring cache._write) so
    a crash mid-write never leaves a truncated corpus file. The temp name
    is pid-suffixed so concurrent processes never collide, and any stale
    temp left by a crashed earlier run is cleaned up first."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    for stale in glob.glob(glob.escape(path) + '.tmp*'):
        if stale != tmp:
            try:
                os.remove(stale)
            except OSError:
                pass
    with open(tmp, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    chunks = run_pass2(output_path='data/chunks.jsonl')
    print(f"Produced {len(chunks)} chunks")
