"""
Mark rows in manifest.db as 'excluded' unless they match the keep rules
defined below. Rows are never deleted — Pass 2 reads crawl_status='crawled'
only, so excluded rows are automatically skipped downstream.

Rules come in three shapes:
  - year_prefix: URL starts with a prefix AND contains 2025 or 2026 anywhere
  - prefix:     URL starts with a prefix (no year filter)
  - exact:      URL matches exactly (or matches any of several variants,
                e.g. space-encoded and %20-encoded forms)

Usage:
    python -m maryland_rag.scripts.apply_keep_filter --dry-run
    python -m maryland_rag.scripts.apply_keep_filter --apply
    python -m maryland_rag.scripts.apply_keep_filter --revert
"""
import argparse
import sqlite3
import sys
from collections import Counter

from ..pass1.config import DB_PATH

EXCLUSION_REASON = "keep_filter_2026"

BASE = "https://elections.maryland.gov"

YEAR_PREFIX_RULES = [
    f"{BASE}/about/documents/",
    f"{BASE}/about/meeting_materials/",
]

PREFIX_RULES = [
    f"{BASE}/pdf/vrar/MSR-2025",
    f"{BASE}/pdf/vrar/MSR-2026",
    f"{BASE}/press_room/documents/",
    f"{BASE}/voting/documents/",
    f"{BASE}/pdf/summary_guide/",
    f"{BASE}/voter_registration/documents/",
    f"{BASE}/laws_and_regs/documents/",
    f"{BASE}/elections/documents/",
    f"{BASE}/voting_system/documents/",
    f"{BASE}/overseas_voters/documents/",
]

EXACT_RULES = [
    f"{BASE}/pdf/Challenger_and_Watchers_Manual.pdf",
    f"{BASE}/get_involved/Challenger  Watcher Summary - Opening the Polls.pdf",
    f"{BASE}/get_involved/Challenger%20%20Watcher%20Summary%20-%20Opening%20the%20Polls.pdf",
    f"{BASE}/get_involved/Challenger  Watcher Summary - Closing the Polls.pdf",
    f"{BASE}/get_involved/Challenger%20%20Watcher%20Summary%20-%20Closing%20the%20Polls.pdf",
    f"{BASE}/get_involved/Election_Judge_Application.pdf",
    f"{BASE}/pdf/Request_for_Accessible_Polling_Place.pdf",
    f"{BASE}/press_room/Voter Registration Security Talking Points.pdf",
    f"{BASE}/press_room/Voter%20Registration%20Security%20Talking%20Points.pdf",
    f"{BASE}/about/Signed Board Bylaws.pdf",
    f"{BASE}/about/Signed%20Board%20Bylaws.pdf",
    f"{BASE}/documents/instructions_box.pdf",
]


def should_keep(url: str) -> bool:
    if url in EXACT_RULES:
        return True
    for prefix in PREFIX_RULES:
        if url.startswith(prefix):
            return True
    for prefix in YEAR_PREFIX_RULES:
        if url.startswith(prefix) and ("2025" in url or "2026" in url):
            return True
    return False


def summarize(rows: list) -> dict:
    keep = []
    drop = []
    for row in rows:
        (keep if should_keep(row["url"]) else drop).append(row)

    keep_by_type = Counter(r["content_type"] for r in keep)
    drop_by_type = Counter(r["content_type"] for r in drop)
    return {
        "keep": keep,
        "drop": drop,
        "keep_by_type": keep_by_type,
        "drop_by_type": drop_by_type,
    }


def print_summary(summary: dict):
    keep, drop = summary["keep"], summary["drop"]
    print(f"KEEP: {len(keep)} rows")
    for ct, n in sorted(summary["keep_by_type"].items()):
        print(f"  {ct}: {n}")
    print(f"DROP: {len(drop)} rows")
    for ct, n in sorted(summary["drop_by_type"].items()):
        print(f"  {ct}: {n}")


def apply_filter(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT url, content_type FROM pages WHERE crawl_status = 'crawled'"
    ).fetchall()
    summary = summarize(rows)
    print_summary(summary)

    drop_urls = [r["url"] for r in summary["drop"]]
    cur = conn.cursor()
    cur.executemany(
        "UPDATE pages SET crawl_status = 'excluded', exclusion_reason = ? "
        "WHERE url = ? AND crawl_status = 'crawled'",
        [(EXCLUSION_REASON, u) for u in drop_urls],
    )
    conn.commit()
    print(f"\nMarked {cur.rowcount} rows as excluded.")


def revert_filter(conn: sqlite3.Connection):
    cur = conn.execute(
        "UPDATE pages SET crawl_status = 'crawled', exclusion_reason = NULL "
        "WHERE crawl_status = 'excluded' AND exclusion_reason = ?",
        (EXCLUSION_REASON,),
    )
    conn.commit()
    print(f"Reverted {cur.rowcount} rows from excluded back to crawled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Show counts, no writes")
    group.add_argument("--apply", action="store_true", help="Mark non-matching rows excluded")
    group.add_argument("--revert", action="store_true", help="Restore rows previously excluded by this filter")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.dry_run:
        rows = conn.execute(
            "SELECT url, content_type FROM pages WHERE crawl_status = 'crawled'"
        ).fetchall()
        print_summary(summarize(rows))
    elif args.apply:
        apply_filter(conn)
    elif args.revert:
        revert_filter(conn)

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
