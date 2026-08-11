"""
Post-ingest verification gate — check chunk coverage, junk heuristics, and
(optionally) JSONL-vs-pgvector consistency after a pipeline run.

Checks:
  1. Coverage  — every manifest.db page with crawl_status='crawled' (except
                 chunking_strategy='skip', which is deliberately unchunked)
                 must have at least one chunk in data/chunks.jsonl.
  2. Junk      — chunks.jsonl + box_chunks.jsonl are scanned for literal
                 'None:' prefixes, texts exceeding the Pass-2 size caps,
                 empty/whitespace-only texts, nav-menu boilerplate ("Skip to
                 Content" / mostly menu-link lines — the poisoned-cache
                 signature), non-Latin-script text, and Spanish leakage
                 (langfilter.is_spanish).
  3. Dup ids   — duplicate chunk_ids within and across the two JSONL files.
                 All collisions are reported; only collisions whose
                 source_urls point at DIFFERENT documents fail (URL-variant
                 / same-document collisions are accepted dedup).
  4. Lengths   — chunks under MIN_CHUNK_CHARS_WARN or over
                 MAX_CHUNK_CHARS_WARN chars are counted and listed.
                 WARN-only, never fails.
  5. Box cov   — every url_manifest.json entry must have >= 1 chunk in
                 box_chunks.jsonl. SKIPPED when the box chunks file or the
                 manifest is absent.
  6. DB parity — if DATABASE_URL is set and Postgres is reachable, the JSONL
                 chunk_id set is compared against the pgvector chunks table
                 in both directions. SKIPPED (not a failure) when the DB is
                 unavailable.

Exits non-zero if any check fails, so it can gate a pipeline run.

Usage:
    python -m maryland_rag.scripts.verify_chunks
    python -m maryland_rag.scripts.verify_chunks --chunks data/chunks.jsonl --box-chunks data/box_chunks.jsonl
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from maryland_rag.pass1.config import DB_PATH as _DB_PATH, PROJECT_ROOT
from maryland_rag.pass2.langfilter import (
    MAX_NON_LATIN_RATIO, is_non_english, is_spanish,
)
from maryland_rag.pass2.strategies.semantic import MAX_CHUNK_CHARS, MAX_CHUNK_WORDS

DB_PATH = Path(_DB_PATH)
ROOT = Path(PROJECT_ROOT)

# How many offenders to print per failing check.
SAMPLE = 10

# Length bands (check 4, warn-only): sub-50-char chunks are usually table-row
# stubs that embed poorly; >8000 chars usually signals extraction salad.
MIN_CHUNK_CHARS_WARN = 50
MAX_CHUNK_CHARS_WARN = 8000

# Boilerplate detection (check 2): the poisoned-cache signature is the site's
# HTML nav page extracted as text — it starts with the skip link, and its menu
# items render as literal '•'-bulleted link lines. Only bullet lines count as
# nav-menu-like: PDF forms, instruction sheets, and trafilatura's '- ' lists
# also produce many short lines and must never fail this gate.
BOILERPLATE_PREFIX = "Skip to Content"
NAV_LINE_MAX_WORDS = 4   # a menu-item line has at most this many words
NAV_LINE_RATIO = 0.5     # >50% menu-like lines => boilerplate
NAV_MIN_LINES = 10       # don't judge short chunks on line shape alone


def _load_jsonl(path: Path) -> list[dict]:
    """Load a chunks JSONL file.

    A malformed line is itself a verification failure: report it as a
    diagnostic (file, line number, parse error) and exit non-zero instead
    of crashing with a raw traceback.
    """
    chunks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"ERROR: malformed JSONL at {path}:{lineno}: {exc}\n"
                    f"  line starts with: {line[:120]!r}",
                    file=sys.stderr,
                )
                print("\nRESULT: FAIL")
                sys.exit(1)
    return chunks


def _print_sample(items: list[str], indent: str = "    ") -> None:
    for item in items[:SAMPLE]:
        print(f"{indent}{item}")
    if len(items) > SAMPLE:
        print(f"{indent}... and {len(items) - SAMPLE} more")


# ---------------------------------------------------------------------------
# Check 1: per-page coverage
# ---------------------------------------------------------------------------

def check_coverage(web_chunks: list[dict]) -> list[str]:
    """Return crawled, non-skip page URLs with zero chunks in chunks.jsonl."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT url, chunking_strategy FROM pages
        WHERE crawl_status = 'crawled'
        ORDER BY url
    """).fetchall()
    conn.close()

    # Deduplicated content carries every contributing URL in source_urls,
    # so a page is covered if ANY chunk cites it as primary or secondary.
    covered: set[str] = set()
    for c in web_chunks:
        if c.get("source_url"):
            covered.add(c["source_url"])
        for u in c.get("source_urls") or []:
            covered.add(u)

    offenders = []
    skip_pages = 0
    for url, strategy in rows:
        if strategy == "skip":
            skip_pages += 1  # deliberately unchunked; not a coverage failure
            continue
        if url not in covered:
            offenders.append(url)

    print(f"  Crawled pages:            {len(rows)}")
    print(f"  Strategy 'skip' (exempt): {skip_pages}")
    print(f"  Pages with ZERO chunks:   {len(offenders)}")
    if offenders:
        _print_sample(offenders)
    return offenders


# ---------------------------------------------------------------------------
# Check 2: junk heuristics
# ---------------------------------------------------------------------------

def _is_nav_boilerplate(text: str) -> bool:
    """Nav-menu boilerplate: the skip link, or a chunk whose lines are
    mostly short '•'-bulleted menu links (the extracted-nav signature)."""
    if text.lstrip().startswith(BOILERPLATE_PREFIX):
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < NAV_MIN_LINES:
        return False
    nav_like = sum(
        1 for ln in lines
        if ln.startswith("•") and len(ln.lstrip("• ").split()) <= NAV_LINE_MAX_WORDS
    )
    return nav_like / len(lines) > NAV_LINE_RATIO


def check_junk(chunks: list[dict]) -> dict[str, list[str]]:
    """Scan all chunks for junk. Returns {problem: [chunk_ids]} for failures."""
    none_prefix: list[str] = []
    oversize_chars: list[str] = []
    oversize_words: list[str] = []
    empty: list[str] = []
    non_english: list[str] = []
    spanish: list[str] = []
    boilerplate: list[str] = []

    for c in chunks:
        cid = c.get("chunk_id", "?")
        text = c.get("text") or ""
        if not text.strip():
            empty.append(cid)
            continue
        if "None:" in text:
            none_prefix.append(cid)
        if len(text) > MAX_CHUNK_CHARS:
            oversize_chars.append(cid)
        if len(text.split()) > MAX_CHUNK_WORDS:
            oversize_words.append(cid)
        # Spanish leakage is reported separately from script-based
        # non-English so the operator can tell the two failure modes apart.
        if is_spanish(text):
            spanish.append(cid)
        elif is_non_english(text):
            non_english.append(cid)
        if _is_nav_boilerplate(text):
            boilerplate.append(cid)

    print(f"  Chunks scanned:                      {len(chunks)}")
    print(f"  Empty/whitespace-only text:          {len(empty)}")
    print(f"  Containing literal 'None:':          {len(none_prefix)}")
    print(f"  Over char cap (>{MAX_CHUNK_CHARS}):          {len(oversize_chars)}")
    print(f"  Over word cap (>{MAX_CHUNK_WORDS}, warn only): {len(oversize_words)}")
    print(f"  Non-English (>{MAX_NON_LATIN_RATIO:.0%} non-Latin letters):  {len(non_english)}")
    print(f"  Spanish leakage (is_spanish):        {len(spanish)}")
    print(f"  Nav-menu boilerplate:                {len(boilerplate)}")
    for label, ids in (("empty", empty), ("'None:'", none_prefix),
                       ("over char cap", oversize_chars),
                       ("non-English", non_english),
                       ("Spanish", spanish),
                       ("boilerplate", boilerplate)):
        if ids:
            print(f"  Sample chunk_ids [{label}]:")
            _print_sample(ids)

    # Word-cap overshoot is WARN-only: semantic chunking legitimately emits
    # chunks up to target+overlap+one capped sentence past MAX_CHUNK_WORDS.
    # The char cap is the hard embedding-safety limit.
    return {
        "empty texts": empty,
        "'None:' prefixes": none_prefix,
        "over char cap": oversize_chars,
        "non-English text": non_english,
        "Spanish text": spanish,
        "nav-menu boilerplate": boilerplate,
    }


# ---------------------------------------------------------------------------
# Check 3: duplicate chunk_ids
# ---------------------------------------------------------------------------

def _norm_doc_url(url: str) -> str:
    """Collapse URL variants of the same document: drop query/fragment and
    the trailing slash, percent-decode, and lowercase — the manifest holds
    space-vs-%20 and path-case variants of identical documents (accepted
    dedup). Genuinely different paths remain different documents."""
    from urllib.parse import unquote
    u = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    return unquote(u).lower()


def check_duplicate_ids(chunks: list[dict]) -> list[str]:
    """Report chunk_id collisions within/across the JSONL files.

    Every collision is printed (warning-level detail with its sources), but
    only ids whose colliding rows cite DIFFERENT documents are returned as
    failures — same-document / URL-variant collisions are accepted dedup
    per operator decision.
    """
    occurrences: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            occurrences[cid].append(c.get("source_url") or "?")

    collisions = {cid: urls for cid, urls in occurrences.items() if len(urls) > 1}
    cross_doc = [
        cid for cid, urls in collisions.items()
        if len({_norm_doc_url(u) for u in urls}) > 1
    ]

    print(f"  Distinct chunk_ids:                  {len(occurrences)}")
    print(f"  Colliding ids (any source):          {len(collisions)}")
    print(f"  Colliding ids across documents:      {len(cross_doc)}")
    if collisions:
        print("  WARNING: collision detail (id: sources):")
        _print_sample([
            f"{cid}: {' | '.join(sorted(set(urls)))}"
            for cid, urls in sorted(collisions.items())
        ])
    return cross_doc


# ---------------------------------------------------------------------------
# Check 4: length bands (warn-only)
# ---------------------------------------------------------------------------

def check_length_bands(chunks: list[dict]) -> None:
    """Count/list chunks outside the useful length band. Never fails."""
    tiny = [c.get("chunk_id", "?") for c in chunks
            if len(c.get("text") or "") < MIN_CHUNK_CHARS_WARN]
    huge = [c.get("chunk_id", "?") for c in chunks
            if len(c.get("text") or "") > MAX_CHUNK_CHARS_WARN]
    print(f"  Under {MIN_CHUNK_CHARS_WARN} chars (warn only):          {len(tiny)}")
    if tiny:
        _print_sample(tiny)
    print(f"  Over {MAX_CHUNK_CHARS_WARN} chars (warn only):         {len(huge)}")
    if huge:
        _print_sample(huge)


# ---------------------------------------------------------------------------
# Check 5: box manifest coverage
# ---------------------------------------------------------------------------

def check_box_coverage(box_chunks: list[dict]) -> tuple[str, list[str]]:
    """Every url_manifest.json entry must have >= 1 chunk in box_chunks.

    Returns (status, offenders) with status 'ok' | 'fail' | 'skipped'.
    """
    try:
        from box_ingest.paths import MANIFEST_PATH
    except ImportError:
        print("  SKIPPED: box_ingest package not importable.")
        return "skipped", []
    if not MANIFEST_PATH.exists():
        print(f"  SKIPPED: {MANIFEST_PATH} not found.")
        return "skipped", []

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    entries = {k: v for k, v in manifest.items() if not k.startswith("_")}

    covered = {c.get("source_url") for c in box_chunks}
    offenders = [f"{key} -> {url}" for key, url in sorted(entries.items())
                 if url not in covered]

    print(f"  Manifest entries:                    {len(entries)}")
    print(f"  Entries with ZERO box chunks:        {len(offenders)}")
    if offenders:
        _print_sample(offenders)
    return ("fail" if offenders else "ok"), offenders


# ---------------------------------------------------------------------------
# Check 6: JSONL vs pgvector parity
# ---------------------------------------------------------------------------

def check_db_parity(jsonl_ids: set[str]) -> tuple[str, list[str], list[str]]:
    """Compare JSONL chunk_ids against the pgvector chunks table.

    Returns (status, missing, orphans) where status is 'ok' | 'fail' |
    'skipped'. Degrades to 'skipped' when the DB is unreachable.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("  SKIPPED: DATABASE_URL not set — cannot compare against pgvector.")
        return "skipped", [], []
    try:
        import psycopg
    except ImportError:
        print("  SKIPPED: psycopg not installed — cannot compare against pgvector.")
        return "skipped", [], []
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            db_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall()}
    except Exception as exc:
        print(f"  SKIPPED: Postgres unavailable ({exc})")
        return "skipped", [], []

    missing = sorted(jsonl_ids - db_ids)  # in JSONL, never embedded
    orphans = sorted(db_ids - jsonl_ids)  # in DB, not in current JSONL (stale)
    print(f"  JSONL chunk_ids: {len(jsonl_ids)}")
    print(f"  DB chunk_ids:    {len(db_ids)}")
    print(f"  MISSING (in JSONL, not in DB): {len(missing)}")
    if missing:
        _print_sample(missing)
    print(f"  ORPHANS (in DB, not in JSONL): {len(orphans)}")
    if orphans:
        _print_sample(orphans)
    return ("fail" if missing or orphans else "ok"), missing, orphans


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_verify(chunks_path: str, box_chunks_path: str) -> int:
    load_dotenv()

    if not DB_PATH.exists():
        print(f"ERROR: manifest DB not found at {DB_PATH}", file=sys.stderr)
        return 1

    web_path = ROOT / chunks_path
    box_path = ROOT / box_chunks_path

    if not web_path.exists():
        print(f"ERROR: chunks file not found at {web_path}", file=sys.stderr)
        return 1
    web_chunks = _load_jsonl(web_path)

    box_chunks: list[dict] = []
    if box_path.exists():
        box_chunks = _load_jsonl(box_path)
    else:
        print(f"WARNING: box chunks file not found at {box_path}; "
              "box checks limited to web chunks.\n")

    failures: list[str] = []

    print("== 1. Page coverage (manifest.db vs chunks.jsonl) ==")
    offenders = check_coverage(web_chunks)
    if offenders:
        failures.append(f"coverage: {len(offenders)} crawled page(s) with zero chunks")

    print("\n== 2. Junk heuristics (chunks.jsonl + box_chunks.jsonl) ==")
    junk = check_junk(web_chunks + box_chunks)
    for problem, ids in junk.items():
        if ids:
            failures.append(f"junk: {len(ids)} chunk(s) with {problem}")

    print("\n== 3. Duplicate chunk_ids (within + across files) ==")
    cross_doc = check_duplicate_ids(web_chunks + box_chunks)
    if cross_doc:
        failures.append(
            f"dup ids: {len(cross_doc)} chunk_id(s) shared across different documents"
        )

    print("\n== 4. Length bands (warn only) ==")
    check_length_bands(web_chunks + box_chunks)

    print("\n== 5. Box manifest coverage ==")
    if box_chunks:
        box_status, box_offenders = check_box_coverage(box_chunks)
    else:
        print("  SKIPPED: no box chunks loaded.")
        box_status, box_offenders = "skipped", []
    if box_status == "fail":
        failures.append(
            f"box coverage: {len(box_offenders)} manifest entr(ies) with zero chunks"
        )

    print("\n== 6. JSONL vs pgvector parity ==")
    jsonl_ids = {c.get("chunk_id", "") for c in web_chunks + box_chunks}
    jsonl_ids.discard("")
    db_status, missing, orphans = check_db_parity(jsonl_ids)
    if db_status == "fail":
        failures.append(
            f"db parity: {len(missing)} missing from DB, {len(orphans)} orphaned in DB"
        )

    print("\n== SUMMARY ==")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
    if box_status == "skipped":
        print("  SKIP  box coverage: box chunks/manifest unavailable")
    if db_status == "skipped":
        print("  SKIP  db parity: Postgres unavailable")
    if not failures:
        print("  All checks passed."
              if db_status != "skipped"
              else "  All runnable checks passed (db parity skipped).")

    print(f"\nRESULT: {'FAIL' if failures else 'OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify chunk coverage, junk, and pgvector parity")
    parser.add_argument("--chunks", default="data/chunks.jsonl",
                        help="Web chunks JSONL path (relative to project root)")
    parser.add_argument("--box-chunks", default="data/box_chunks.jsonl",
                        help="Box chunks JSONL path (relative to project root)")
    args = parser.parse_args()
    sys.exit(run_verify(chunks_path=args.chunks, box_chunks_path=args.box_chunks))
