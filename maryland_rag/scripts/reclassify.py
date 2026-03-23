"""
Reclassification Script — re-assign page_classification and chunking_strategy
for all crawled HTML pages using stored metadata (url, title, extracted_snippet,
word_count) without re-fetching any pages.

Key improvements over original classifier.py:
  - Removes over-broad 'register'/'registration' from form signals
  - Adds 'nav_hub' class for link-heavy hub pages (150–499 words)
  - Activates 'semantic_with_overlap' strategy for prose pages
  - Strips site-wide announcement banner from snippets before classification
  - URL-path-based form detection instead of keyword matching
  - press_room treated as press_release, not form

Usage:
    python -m maryland_rag.scripts.reclassify [--dry-run]
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

DB_PATH = Path(__file__).parent.parent.parent / "data" / "manifest.db"

# ---------------------------------------------------------------------------
# Banner stripping
# ---------------------------------------------------------------------------
# Many extracted_snippets begin with a site-wide announcement that was
# captured before the real content. Strip it so it doesn't pollute signals.
BANNER_MARKERS = [
    "The Worcester County Board of Elections",
    "The Baltimore County Board of Elections",
    "The Montgomery County Board of Elections",
]


def strip_banner(snippet: str | None) -> str:
    if not snippet:
        return ""
    for marker in BANNER_MARKERS:
        if snippet.startswith(marker):
            # Advance past the first newline following the banner sentence
            newline_pos = snippet.find("\n", len(marker))
            if newline_pos != -1:
                return snippet[newline_pos:].lstrip()
    return snippet


# ---------------------------------------------------------------------------
# Classification signals
# ---------------------------------------------------------------------------

# URL path substrings that indicate actual online forms (not just pages about forms)
FORM_URL_PATHS = [
    "/forms/",
    "data_form",
    "schedule_appointment",
    "purchase_lists",
]

# These are checked against url+title+snippet (broad match)
FAQ_SIGNALS = [
    "faq", "frequently-asked", "q&a", "questions-and-answers",
]

# These are checked against the URL path ONLY to avoid false positives
# when snippets mention these topics in passing (e.g., a contact page
# that says "questions about absentee ballots" would match FAQ_SIGNALS
# but should not be classified as FAQ).
FAQ_URL_PATHS = [
    "early_voting", "absentee", "election_day_questions",
    "learn_about_the_new_voting_system", "redistricting",
    "/about/pia",
]

PRESS_SIGNALS = [
    "press_room", "press-room", "press_release", "press-release",
    "rumor_control", "dis-misinformation",
]

# URL/title keywords that reliably indicate table/data pages
TABLE_SIGNALS = [
    "municipal_results", "election_results", "results_archive",
    "/elections/districts", "/elections/archive", "/elections/printed_copies",
    "voter_registration/archive", "voter_registration/stats",
    "/voting/recount",
]

# URL keywords for short navigational/static pages
SHORT_STATIC_SIGNALS = [
    "/about/contact", "/about/directions", "/about/social_media",
    "/about/county_boards", "/about/state-links", "/about/federal-links",
    "/about/board", "/about/feedback",
    "/voting_system/voting_equipment", "/voting_system/how_to_vote",
    "/voting_system/ballot_audit_plan",
    "/laws_and_regs/sbe_policy", "/laws_and_regs/index",
    "/candidacy/qualifications", "/candidacy/ballot", "/candidacy/candidate_filing",
    "/get_involved/election_judges", "/get_involved/students", "/get_involved/index",
    "/get_involved/dis-misinformation",
    "/overseas_voters/other_information", "/overseas_voters/index",
    "/press_room/dis-misinformation", "/press_room/dis",
    "/voter_registration/nvra", "/voter_registration/data_form",
    "/voter_registration/archive_bydistricts",
    "/accessibility", "/privacy",
    "/voting/address", "/voting/primary",
    "/voter_services/",
    "/voting_system/procurement",
    "/elections/special_elections_past", "/elections/electoral_college",
]


def classify_html(url: str, title: str | None, snippet: str | None, word_count: int) -> tuple[str, str, str]:
    """
    Returns (page_classification, chunking_strategy, classification_confidence).
    Works from stored metadata only — no raw HTML.
    """
    url_lower = url.lower()
    title_lower = (title or "").lower()
    clean_snippet = strip_banner(snippet)
    combined = f"{url_lower} {title_lower} {clean_snippet.lower()}"
    wc = word_count or 0
    path = urlparse(url).path.lower()
    domain = urlparse(url).netloc.lower()

    # --- 1. Junk (Cloudflare stubs, empty pages) ---
    if "cdn-cgi" in url_lower:
        return "junk", "skip", "high"

    # --- 2. FAQ ---
    # Broad signals checked against full combined string
    if any(s in combined for s in FAQ_SIGNALS):
        return "faq", "qa_pairs", "high"
    # Path-specific signals checked against URL only (avoids false positives
    # when snippets mention these topics in passing)
    if any(s in path for s in FAQ_URL_PATHS):
        return "faq", "qa_pairs", "high"

    # --- 3. Press release / news ---
    if any(s in combined for s in PRESS_SIGNALS):
        strategy = "simple_split" if wc >= 150 else "ingest_as_single"
        return "press_release", strategy, "high"

    # --- 4. Table / data pages (URL-path check BEFORE prose word-count check) ---
    # Stats, results, and archive pages can have high word counts but are
    # fundamentally tabular/list data — catch them before the prose fallback.
    if any(s in path for s in TABLE_SIGNALS):
        return "table_data", "table_rows", "high"

    # --- 5. Prose (long-form informational content) ---
    if wc >= 500:
        return "prose", "semantic_with_overlap", "medium"

    # --- 6. Actual online forms (URL-path-based only) ---
    if any(p in path for p in FORM_URL_PATHS) or domain.startswith("voterservices"):
        strategy = "simple_split" if wc >= 150 else "ingest_as_single"
        return "form", strategy, "high"

    # --- 7. Known short-static pages (by URL pattern) ---
    if any(s in path for s in SHORT_STATIC_SIGNALS):
        return "short_static", "ingest_as_single", "high"

    # --- 8. Nav hub (mid-length pages with no strong signal — mostly link lists) ---
    if wc >= 150:
        return "nav_hub", "ingest_as_single", "medium"

    # --- 9. Default: short static ---
    return "short_static", "ingest_as_single", "low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Track changes
    changes = []
    unchanged = 0
    transition_counts = defaultdict(int)  # (old, new) -> count

    for row in rows:
        url = row["url"]
        old_class = row["page_classification"]
        old_strategy = row["chunking_strategy"]

        new_class, new_strategy, new_confidence = classify_html(
            url=url,
            title=row["title"],
            snippet=row["extracted_snippet"],
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

    # --- Print change report ---
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

        # --- Final distribution ---
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
