"""
Box materials ingestion — Step 2 of 2.

Reads every file in needtochunk/, extracts the text using the shared Pass 2
extractors, chunks it, and writes data/box_chunks.jsonl in the schema Pass 3
(embedding) expects.

Two efficiency features:

  - Skip-if-unchanged. A per-file state cache (data/box_ingest.state.json)
    records each file's size+mtime fingerprint, its produced chunks, and a
    status ('ok' | 'empty' | 'failed'). On re-run, unchanged 'ok' files reuse
    their cached chunks instead of being re-extracted (OCR is the expensive
    path we most want to avoid); 'empty'/'failed' files are retried every run
    and reported in a warning summary so they never vanish silently.

  - Parallel extraction. Files are extracted in a ProcessPoolExecutor so
    PDF text extraction and OCR run across CPU cores.

The consolidated JSONL is the union of all cached chunks, so downstream
Pass 3 always sees the full corpus regardless of which files were
re-extracted this run.

Run:
    python -m box_ingest.automate     # download files from Box (Step 1)
    python -m box_ingest.ingest       # extract + chunk (Step 2)

Called automatically by `python -m maryland_rag all`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from box_ingest.paths import (
    NEEDTOCHUNK_DIR,
    MANIFEST_PATH,
    STATE_PATH,
    DEFAULT_OUTPUT,
)
from maryland_rag.pass2.chunker import MIN_CHUNK_BODY_CHARS
from maryland_rag.pass2.metadata import build_chunk_metadata
from maryland_rag.pass2.strategies.docx_strategy import extract_docx_from_path
from maryland_rag.pass2.strategies.pdf import extract_pdf_from_path
from maryland_rag.pass2.strategies.semantic import semantic_chunk
from maryland_rag.pass2.strategies.single import ingest_as_single
from maryland_rag.pass2.strategies.xls_strategy import extract_xls_from_path

logger = logging.getLogger(__name__)

SHORT_DOC_WORDS      = 150
FALLBACK_CHUNK_WORDS = 400
SKIP_NAMES           = {"url_manifest.json", "review_files.txt", "README.md", ".DS_Store"}


# ---------------------------------------------------------------------------
# Manifest + state
# ---------------------------------------------------------------------------

def _load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        logger.warning("url_manifest.json not found at %s", MANIFEST_PATH)
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read state file (%s); rebuilding from scratch", exc)
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    tmp.replace(STATE_PATH)


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _record(state: dict, rel_key: str, fp: Path, box_url: str,
            chunks: list[dict], status: str, reason: str = "") -> None:
    """Record a file's outcome in the state cache.

    status is 'ok' | 'empty' | 'failed'. Non-ok entries carry an empty chunk
    list plus a reason, and are re-attempted on every run — the fingerprint
    check only trusts 'ok' entries, so a failure can never be cached as
    success. (Pre-status entries in old state files lack the field and are
    treated as 'ok'.)
    """
    try:
        fingerprint = _fingerprint(fp)
    except OSError as exc:
        # A plausible cause of a 'failed' status is the file vanishing
        # mid-run — fingerprinting it again inside the failure handler must
        # not raise. An empty fingerprint never matches a real one, so the
        # entry is retried (or dropped as stale) on the next run.
        logger.warning("Could not fingerprint %s while recording %s status: %s",
                       fp, status, exc)
        fingerprint = ""
    entry = {
        "fingerprint": fingerprint,
        "box_url": box_url,
        "status": status,
        "chunks": chunks,
    }
    if reason:
        entry["reason"] = reason
    state[rel_key] = entry


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Route text to single/semantic/paragraph-fallback chunking by length.

    Chunks under MIN_CHUNK_BODY_CHARS (same floor as web pass 2) are dropped:
    they are section stubs like "No updates.", not retrievable content."""
    chunks = _route_chunks(text)
    kept = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_BODY_CHARS]
    for c in chunks:
        if len(c.strip()) < MIN_CHUNK_BODY_CHARS:
            logger.info("Dropped under-length chunk: %.60r", c)
    return kept


def _route_chunks(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    if len(text.split()) <= SHORT_DOC_WORDS:
        return ingest_as_single(text)
    try:
        chunks = semantic_chunk(text)
        if chunks:
            return chunks
    except Exception as exc:
        logger.debug("semantic_chunk failed (%s); using paragraph fallback", exc)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for para in paragraphs:
        w = len(para.split())
        if cur_words + w > FALLBACK_CHUNK_WORDS and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
        cur.append(para)
        cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _build(text: str, source_url: str, title: str,
           chunk_index: int, chunk_total: int,
           section_hierarchy: list[str]) -> dict:
    return build_chunk_metadata(
        source_url=source_url,
        title=title,
        section_hierarchy=section_hierarchy,
        page_classification="document",
        chunking_strategy="box_ingest",
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
    )


# ---------------------------------------------------------------------------
# Per-file worker (must be top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _process_file(rel_key: str, abs_path_str: str, box_url: str) -> tuple[list[dict], str]:
    """Extract + chunk one file.

    Returns (chunks, empty_reason). empty_reason is set only when chunks is
    empty and explains why — it is surfaced in the end-of-run warning summary
    so zero-chunk files never disappear silently.
    """
    abs_path = Path(abs_path_str)
    folder_chain = list(abs_path.relative_to(NEEDTOCHUNK_DIR).parts[:-1])
    title = abs_path.stem
    suffix = abs_path.suffix.lower()

    if suffix == ".pdf":
        result = extract_pdf_from_path(str(abs_path), ocr_fallback=True)
        text = result.get("text", "")
        if not text.strip():
            return [], "no extractable text in PDF (even with OCR fallback)"
        pieces = _chunk_text(text)
        return [_build(c, box_url, title, i, len(pieces), folder_chain)
                for i, c in enumerate(pieces)], ""

    if suffix in (".docx", ".doc"):
        result = extract_docx_from_path(str(abs_path))
        sections = result.get("sections", [])
        if sections:
            records: list[tuple[list[str], str]] = []
            for sec in sections:
                for c in _chunk_text(sec.get("text", "")):
                    records.append((folder_chain + sec.get("heading_chain", []), c))
            return [_build(c, box_url, title, i, len(records), chain)
                    for i, (chain, c) in enumerate(records)], \
                   "" if records else "DOCX sections contained no chunkable text"
        text = result.get("full_text", "")
        if not text.strip():
            return [], "no extractable text in DOCX/DOC"
        pieces = _chunk_text(text)
        return [_build(c, box_url, title, i, len(pieces), folder_chain)
                for i, c in enumerate(pieces)], ""

    if suffix in (".xlsx", ".xlsm"):
        rows = extract_xls_from_path(str(abs_path)).get("rows", [])
        if not rows:
            return [], "no rows extracted (empty workbook, or openpyxl missing)"
        return [_build(r, box_url, title, i, len(rows), folder_chain)
                for i, r in enumerate(rows)], ""

    if suffix == ".txt":
        text = abs_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return [], "file contains no text"
        pieces = _chunk_text(text)
        return [_build(c, box_url, title, i, len(pieces), folder_chain)
                for i, c in enumerate(pieces)], ""

    logger.warning("Unsupported file type: %s", abs_path.name)
    return [], f"unsupported file type: {suffix or abs_path.name}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_ingest(
    dry_run: bool = False,
    output_path: Path | None = None,
    workers: int | None = None,
) -> list[dict]:
    if workers is None:
        workers = max(2, os.cpu_count() or 4)

    manifest = _load_manifest()
    state = _load_state() if not dry_run else {}

    on_disk: list[tuple[str, Path, str]] = []
    missing_url: list[str] = []
    for fp in sorted(NEEDTOCHUNK_DIR.rglob("*")):
        if not fp.is_file():
            continue
        if fp.name in SKIP_NAMES:
            continue
        rel_key = str(fp.relative_to(NEEDTOCHUNK_DIR))
        box_url = (manifest.get(rel_key) or "").strip()
        if not box_url:
            missing_url.append(rel_key)
            continue
        on_disk.append((rel_key, fp, box_url))

    on_disk_keys = {k for k, _, _ in on_disk}
    stale_keys = [k for k in state if k not in on_disk_keys]
    if stale_keys:
        logger.info("Dropping %d state entries for files no longer on disk", len(stale_keys))
        for k in stale_keys:
            state.pop(k, None)

    todo: list[tuple[str, Path, str]] = []
    reused = 0
    for rel_key, fp, box_url in on_disk:
        fp_print = _fingerprint(fp)
        cached = state.get(rel_key)
        # Only status 'ok' satisfies the cache: 'failed'/'empty' files are
        # cheap to retry at this corpus size and must stay visible until
        # they produce chunks. Entries written before status tracking lack
        # the field — treat missing status as 'ok'.
        if (
            cached
            and cached.get("fingerprint") == fp_print
            and cached.get("box_url") == box_url
            and cached.get("status", "ok") == "ok"
        ):
            reused += 1
            continue
        todo.append((rel_key, fp, box_url))

    logger.info(
        "Box ingest plan: %d on disk, %d cached, %d to (re)extract, %d missing URL",
        len(on_disk), reused, len(todo), len(missing_url),
    )
    for p in missing_url:
        logger.warning("No Box URL in manifest: %s", p)

    if dry_run:
        for rel_key, _, box_url in todo:
            print(f"  [extract] {rel_key} -> {box_url}")
        todo_keys = {k for k, _, _ in todo}
        for rel_key, _, _ in on_disk:
            if rel_key not in todo_keys:
                print(f"  [cached]  {rel_key}")
        return []

    if todo:
        effective_workers = min(workers, len(todo))
        logger.info("Extracting %d file(s) with %d worker(s)", len(todo), effective_workers)
        if effective_workers <= 1:
            for rel_key, fp, box_url in todo:
                try:
                    chunks, empty_reason = _process_file(rel_key, str(fp), box_url)
                except Exception as exc:
                    logger.error("Failed to process %s: %s", rel_key, exc, exc_info=True)
                    _record(state, rel_key, fp, box_url, [], "failed", str(exc))
                    continue
                if chunks:
                    _record(state, rel_key, fp, box_url, chunks, "ok")
                else:
                    _record(state, rel_key, fp, box_url, [], "empty",
                            empty_reason or "extractor produced 0 chunks")
                logger.info("Processed %s (%d chunks)", rel_key, len(chunks))
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(_process_file, rel_key, str(fp), box_url): (rel_key, fp, box_url)
                    for rel_key, fp, box_url in todo
                }
                for fut in as_completed(futures):
                    rel_key, fp, box_url = futures[fut]
                    try:
                        chunks, empty_reason = fut.result()
                    except Exception as exc:
                        logger.error("Failed to process %s: %s", rel_key, exc, exc_info=True)
                        _record(state, rel_key, fp, box_url, [], "failed", str(exc))
                        continue
                    if chunks:
                        _record(state, rel_key, fp, box_url, chunks, "ok")
                    else:
                        _record(state, rel_key, fp, box_url, [], "empty",
                                empty_reason or "extractor produced 0 chunks")
                    logger.info("Processed %s (%d chunks)", rel_key, len(chunks))

    all_chunks: list[dict] = []
    for rel_key in sorted(state):
        all_chunks.extend(state[rel_key].get("chunks", []))

    # Write the JSONL output FIRST and durably (atomic tmp+rename, fsync'd) so
    # the success state is never recorded while the chunks file is missing or
    # truncated. If the disk fills here, state is not updated and the next run
    # re-extracts and recovers instead of trusting a stale cache.
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(out.suffix + ".tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_out, out)
    logger.info("Wrote %d chunks -> %s", len(all_chunks), out)

    _save_state(state)

    # Loud end-of-run summary: every file currently contributing zero chunks.
    # These entries are re-attempted on every run, so this warning repeats
    # until the file extracts cleanly or is removed.
    problems = [
        (rel_key, entry.get("status"), entry.get("reason", ""))
        for rel_key, entry in sorted(state.items())
        if entry.get("status", "ok") != "ok"
    ]
    if problems:
        logger.warning(
            "Box ingest: %d file(s) contributed NO chunks to the corpus:",
            len(problems),
        )
        for rel_key, status, reason in problems:
            logger.warning("  [%s] %s: %s", status, rel_key, reason or "unknown reason")

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Chunk files in needtochunk/ for Pass 3")
    parser.add_argument("--dry-run", action="store_true", help="List planned actions, write nothing")
    parser.add_argument("--output", type=Path, default=None, help="Override output JSONL path")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count (default: CPU count)")
    args = parser.parse_args()
    run_ingest(dry_run=args.dry_run, output_path=args.output, workers=args.workers)
