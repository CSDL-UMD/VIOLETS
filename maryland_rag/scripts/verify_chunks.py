"""
Post-ingest verification gate — check chunk coverage, junk heuristics, and
(optionally) JSONL-vs-pgvector consistency after a pipeline run.

Checks:
  1. Coverage  — every manifest.db page with crawl_status='crawled' (except
                 chunking_strategy='skip', which is deliberately unchunked)
                 must have at least one chunk in data/chunks.jsonl.
  2. Junk      — chunks.jsonl + box_chunks.jsonl are scanned for literal
                 'None:' prefixes, texts exceeding the Pass-2 size caps, and
                 empty/whitespace-only texts.
  3. DB parity — if DATABASE_URL is set and Postgres is reachable, the JSONL
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
from pathlib import Path

from dotenv import load_dotenv

from maryland_rag.pass1.config import DB_PATH as _DB_PATH, PROJECT_ROOT
from maryland_rag.pass2.langfilter import MAX_NON_LATIN_RATIO, is_non_english
from maryland_rag.pass2.strategies.semantic import MAX_CHUNK_CHARS, MAX_CHUNK_WORDS

DB_PATH = Path(_DB_PATH)
ROOT = Path(PROJECT_ROOT)

# How many offenders to print per failing check.
SAMPLE = 10


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

def check_junk(chunks: list[dict]) -> dict[str, list[str]]:
    """Scan all chunks for junk. Returns {problem: [chunk_ids]} for failures."""
    none_prefix: list[str] = []
    oversize_chars: list[str] = []
    oversize_words: list[str] = []
    empty: list[str] = []
    non_english: list[str] = []

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
        if is_non_english(text):
            non_english.append(cid)

    print(f"  Chunks scanned:                      {len(chunks)}")
    print(f"  Empty/whitespace-only text:          {len(empty)}")
    print(f"  Containing literal 'None:':          {len(none_prefix)}")
    print(f"  Over char cap (>{MAX_CHUNK_CHARS}):          {len(oversize_chars)}")
    print(f"  Over word cap (>{MAX_CHUNK_WORDS}, warn only): {len(oversize_words)}")
    print(f"  Non-English (>{MAX_NON_LATIN_RATIO:.0%} non-Latin letters):  {len(non_english)}")
    for label, ids in (("empty", empty), ("'None:'", none_prefix),
                       ("over char cap", oversize_chars),
                       ("non-English", non_english)):
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
    }


# ---------------------------------------------------------------------------
# Check 3: JSONL vs pgvector parity
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

    print("\n== 3. JSONL vs pgvector parity ==")
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
