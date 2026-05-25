"""
Full automation pipeline for Box SBE materials.

Steps:
  1. Crawl the Box hub → classify every file (include/exclude/review)
  2. Download INCLUDE files into needtochunk/<folder>/
  3. Update url_manifest.json with their Box URLs
  4. Run box_ingest.ingest to produce data/box_chunks.jsonl
  5. Print a report of REVIEW files that need manual triage

Usage:
    python -m box_ingest.automate              # full run
    python -m box_ingest.automate --dry-run    # show what would happen, no downloads
    python -m box_ingest.automate --no-ingest  # skip the final ingest step
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from box_ingest.crawler import crawl_hub, BoxFile, get_access_token
from box_ingest.filter  import FILTER_INCLUDE, FILTER_EXCLUDE, FILTER_REVIEW
from box_ingest.manifest import update_manifest

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).parent.parent
NEEDTOCHUNK_DIR = PROJECT_ROOT / "needtochunk"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(box_file: BoxFile, access_token: str, dry_run: bool = False) -> Path:
    """
    Download a single Box file into needtochunk/<folder_path>/<name>.
    Skips if the file already exists locally.
    Returns the local path.
    """
    dest_dir  = NEEDTOCHUNK_DIR / box_file.folder_path if box_file.folder_path else NEEDTOCHUNK_DIR
    dest_path = dest_dir / box_file.name

    if dest_path.exists():
        logger.debug("Already exists locally, skipping download: %s", dest_path)
        return dest_path

    if dry_run:
        logger.info("[dry-run] Would download: %s → %s", box_file.box_url, dest_path)
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    from box_ingest.crawler import SHARED_LINK
    download_url = f"https://api.box.com/2.0/files/{box_file.file_id}/content"
    req = urllib.request.Request(download_url)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("BoxApi", f"shared_link={SHARED_LINK}")

    logger.info("Downloading: %s", box_file.name)
    with urllib.request.urlopen(req) as resp:
        dest_path.write_bytes(resp.read())

    logger.info("Saved to: %s", dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, no_ingest: bool = False) -> None:
    # Step 1: crawl
    logger.info("=== Step 1: Crawling Box hub ===")
    access_token = get_access_token()
    files = crawl_hub(access_token)

    include = [f for f in files if f.decision == FILTER_INCLUDE]
    exclude = [f for f in files if f.decision == FILTER_EXCLUDE]
    review  = [f for f in files if f.decision == FILTER_REVIEW]

    logger.info("Found %d include / %d exclude / %d review", len(include), len(exclude), len(review))

    # Step 2: download INCLUDE files
    logger.info("=== Step 2: Downloading %d included files ===", len(include))
    for box_file in include:
        download_file(box_file, access_token, dry_run=dry_run)

    # Step 3: update manifest
    logger.info("=== Step 3: Updating url_manifest.json ===")
    summary = update_manifest(files, dry_run=dry_run)
    logger.info(
        "Manifest: %d added/filled, %d already present, %d non-include skipped",
        summary["added"], summary["already_present"], summary["skipped_non_include"],
    )

    # Step 4: run ingest
    if not no_ingest and not dry_run:
        logger.info("=== Step 4: Running box_ingest.ingest ===")
        from box_ingest.ingest import run_ingest
        chunks = run_ingest()
        logger.info("Ingest produced %d chunks", len(chunks))
    else:
        logger.info("=== Step 4: Skipped (--dry-run or --no-ingest) ===")

    # Step 5: report REVIEW files
    if review:
        print("\n" + "=" * 60)
        print(f"MANUAL REVIEW NEEDED ({len(review)} files)")
        print("These matched neither include nor exclude rules.")
        print("=" * 60)
        for f in review:
            print(f"  [{f.folder_path}] {f.name}")
            print(f"    {f.box_url}")
        print()
        print("For each file above, decide:")
        print("  - Add its name keyword to INCLUDE_TERMS in box_ingest/filter.py, OR")
        print("  - Add its name keyword to EXCLUDE_TERMS in box_ingest/filter.py, OR")
        print("  - Manually add it to url_manifest.json if it's a one-off")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Automate Box SBE materials ingestion")
    parser.add_argument("--dry-run",   action="store_true", help="Preview only, no downloads or file writes")
    parser.add_argument("--no-ingest", action="store_true", help="Skip running box_ingest.ingest at the end")
    args = parser.parse_args()
    run(dry_run=args.dry_run, no_ingest=args.no_ingest)
