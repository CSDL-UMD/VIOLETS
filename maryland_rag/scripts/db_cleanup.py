"""
DB Cleanup Script — Pass 1 post-processing.

Removes duplicate URL variants and junk pages from manifest.db before Pass 2 chunking.

Deletions performed (in order):
  1. All http:// rows           — ~3,292 rows (same content as https://)
  2. All https://www. rows      — ~1,120 rows (same content as canonical, plus www-only docs)
  3. cdn-cgi Cloudflare stubs   — Cloudflare email-protection pages, no real content
  4. businessdisclosure domain  — External subdomain, 0 words
  5. All remaining failed rows  — Spaces-in-filename PDFs, mailto fragments, dead weight

A timestamped backup is created before any changes are made.

Usage:
    python -m maryland_rag.scripts.db_cleanup [--dry-run]
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "manifest.db"


def backup_db(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_suffix(f".db.bak.{ts}")
    shutil.copy2(db_path, backup)
    return backup


def run_cleanup(dry_run: bool = False):
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    # --- Backup ---
    if not dry_run:
        backup = backup_db(DB_PATH)
        print(f"Backup created: {backup}")
    else:
        print("[DRY RUN] No changes will be written.\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # --- Before counts ---
    total_before = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    crawled_before = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE crawl_status='crawled'"
    ).fetchone()[0]
    failed_before = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE crawl_status='failed'"
    ).fetchone()[0]

    print(f"\nBEFORE cleanup:")
    print(f"  Total rows:   {total_before:>6}")
    print(f"  Crawled:      {crawled_before:>6}")
    print(f"  Failed:       {failed_before:>6}")

    deletions = [
        (
            "http:// variants (all schemes)",
            "DELETE FROM pages WHERE url LIKE 'http://%'",
        ),
        (
            "https://www.elections.maryland.gov/ variants",
            "DELETE FROM pages WHERE url LIKE 'https://www.elections.maryland.gov/%'",
        ),
        (
            "Cloudflare cdn-cgi stubs",
            "DELETE FROM pages WHERE url LIKE '%cdn-cgi%'",
        ),
        (
            "businessdisclosure external subdomain",
            "DELETE FROM pages WHERE url LIKE '%businessdisclosure-elections.maryland.gov%'",
        ),
        (
            "Remaining failed rows (unresolvable)",
            "DELETE FROM pages WHERE crawl_status = 'failed'",
        ),
    ]

    print("\nDeletion plan:")
    total_deleted = 0
    for label, sql in deletions:
        # Count how many rows this would affect
        count_sql = sql.replace("DELETE FROM pages", "SELECT COUNT(*) FROM pages")
        count = conn.execute(count_sql).fetchone()[0]
        print(f"  {label}: {count:>5} rows")
        total_deleted += count

    print(f"\n  TOTAL to delete: {total_deleted}")
    print(f"  REMAINING after cleanup: {total_before - total_deleted}")

    if dry_run:
        conn.close()
        print("\n[DRY RUN] No changes made. Re-run without --dry-run to apply.")
        return

    # --- Execute deletions ---
    print("\nApplying deletions...")
    actual_deleted = 0
    for label, sql in deletions:
        cur = conn.execute(sql)
        n = cur.rowcount
        actual_deleted += n
        print(f"  ✓ {label}: deleted {n}")
    conn.commit()

    # --- Also clean up orphaned links rows ---
    links_before = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    conn.execute("""
        DELETE FROM links
        WHERE source_url NOT IN (SELECT url FROM pages)
           OR target_url NOT IN (SELECT url FROM pages)
    """)
    links_deleted = links_before - conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    conn.commit()

    # --- VACUUM to reclaim disk space ---
    print("\nVACUUMing database...")
    conn.execute("VACUUM")
    conn.commit()

    # --- After counts ---
    total_after = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    crawled_after = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE crawl_status='crawled'"
    ).fetchone()[0]

    print(f"\nAFTER cleanup:")
    print(f"  Total rows:   {total_after:>6}  (removed {total_before - total_after})")
    print(f"  Crawled:      {crawled_after:>6}")
    print(f"  Failed:       0")
    print(f"  Links cleaned: {links_deleted}")

    # --- Classification breakdown after ---
    print("\nClassification distribution (crawled pages):")
    rows = conn.execute("""
        SELECT page_classification, content_type, COUNT(*) as n
        FROM pages WHERE crawl_status='crawled'
        GROUP BY page_classification, content_type
        ORDER BY n DESC
    """).fetchall()
    for row in rows:
        print(f"  {row[0]:20} [{row[1]:5}] {row[2]:>5}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up manifest.db before Pass 2")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    run_cleanup(dry_run=args.dry_run)
