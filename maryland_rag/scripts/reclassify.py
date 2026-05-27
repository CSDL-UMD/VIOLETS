"""
Reclassification Script — re-assign page_classification and chunking_strategy
for all crawled HTML pages using stored metadata (url, title, extracted_snippet,
word_count) without re-fetching any pages.

Uses the shared rules engine in maryland_rag.pass1.rules so crawl-time and
reclassification stay in lockstep. raw_html is unavailable post-hoc, so
structural-pattern fallbacks don't fire here.

Usage:
    python -m maryland_rag.scripts.reclassify [--dry-run]
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from maryland_rag.pass1.config import DB_PATH as _DB_PATH
from maryland_rag.pass1.rules import classify_html

DB_PATH = Path(_DB_PATH)


def run_reclassify(dry_run: bool = False):
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT url, title, extracted_snippet, word_count,
               page_classification, chunking_strategy, classification_confidence
        FROM pages
        WHERE crawl_status = 'crawled' AND content_type = 'html'
        ORDER BY url
    """).fetchall()

    print(f"HTML pages to reclassify: {len(rows)}")
    if dry_run:
        print("[DRY RUN] No changes will be written.\n")

    changes = []
    unchanged = 0
    transition_counts = defaultdict(int)

    for row in rows:
        url = row["url"]
        old_class = row["page_classification"]
        old_strategy = row["chunking_strategy"]

        new_class, new_strategy, new_confidence = classify_html(
            url=url,
            title=row["title"],
            text=row["extracted_snippet"],
            word_count=row["word_count"] or 0,
        )

        if new_class == old_class and new_strategy == old_strategy:
            unchanged += 1
            continue

        changes.append({
            "url": url,
            "old_class": old_class,
            "new_class": new_class,
            "old_strategy": old_strategy,
            "new_strategy": new_strategy,
            "new_confidence": new_confidence,
            "word_count": row["word_count"],
        })
        transition_counts[(old_class, new_class)] += 1

    print(f"\nUnchanged: {unchanged}")
    print(f"Changed:   {len(changes)}")
    print()

    if changes:
        print("Transitions (old → new, count):")
        for (old, new), count in sorted(transition_counts.items(), key=lambda x: -x[1]):
            print(f"  {old or 'NULL':20} → {new:20}  {count:>4}")

        print("\nDetailed changes:")
        for c in sorted(changes, key=lambda x: (x["new_class"], -(x["word_count"] or 0))):
            shorturl = c["url"].replace("https://elections.maryland.gov/", "/")
            shorturl = shorturl.replace("https://voterservices.elections.maryland.gov/", "/vs/")
            print(
                f"  {c['old_class'] or 'NULL':20} → {c['new_class']:20}  "
                f"words={c['word_count'] or 0:5}  {shorturl}"
            )

    if not dry_run and changes:
        print("\nApplying changes...")
        for c in changes:
            conn.execute("""
                UPDATE pages
                SET page_classification       = ?,
                    chunking_strategy         = ?,
                    classification_confidence = ?
                WHERE url = ?
            """, (c["new_class"], c["new_strategy"], c["new_confidence"], c["url"]))
        conn.commit()
        print(f"  ✓ Updated {len(changes)} rows")

        print("\nFinal classification distribution (HTML pages):")
        dist = conn.execute("""
            SELECT page_classification, chunking_strategy, COUNT(*) as n
            FROM pages WHERE crawl_status='crawled' AND content_type='html'
            GROUP BY page_classification, chunking_strategy
            ORDER BY n DESC
        """).fetchall()
        for row in dist:
            print(f"  {row[0]:20}  [{row[1]:25}]  {row[2]:>4}")

    elif dry_run:
        print("\n[DRY RUN] Re-run without --dry-run to apply.")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reclassify HTML pages using stored metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    run_reclassify(dry_run=args.dry_run)
