"""
Crawl the Maryland SBE Box hub, download relevant files, and update the manifest.

This script is Step 1 of the Box ingestion pipeline. It does NOT chunk files —
chunking happens later when you run: python -m maryland_rag all

Steps:
  1. Scrape the Box Hub page to find folder IDs, then walk each folder via
     the Box API to get a full list of files (2026 folders only).
  2. Classify each file as include / exclude / review using keyword rules
     defined in box_ingest/filter.py.
  3. Download INCLUDE files into needtochunk/<folder>/<filename>.
     Files already present locally are skipped.
  4. Update needtochunk/url_manifest.json with the Box share URL for each
     downloaded file. The manifest is used later by the chunking pipeline
     to attach a source URL to every chunk.
  5. Print a list of REVIEW files — files that matched neither include nor
     exclude rules — so you can manually decide whether to add them.

Usage:
    python -m box_ingest.automate            # full run
    python -m box_ingest.automate --dry-run  # preview only, no downloads or file writes
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from box_ingest.crawler import crawl_hub, BoxFile, get_access_token
from box_ingest.filter import FILTER_INCLUDE, FILTER_EXCLUDE, FILTER_REVIEW
from box_ingest.manifest import update_manifest

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).parent.parent
NEEDTOCHUNK_DIR = PROJECT_ROOT / "needtochunk"
REVIEW_LOG      = NEEDTOCHUNK_DIR / "review_files.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_path(box_file: BoxFile) -> Path:
    base = NEEDTOCHUNK_DIR / box_file.folder_path if box_file.folder_path else NEEDTOCHUNK_DIR
    return base / box_file.name


def _already_downloaded(box_file: BoxFile) -> bool:
    return _local_path(box_file).exists()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(box_file: BoxFile, access_token: str, dry_run: bool = False) -> Path:
    """
    Download a single Box file into needtochunk/<folder_path>/<name>.
    Skips if the file already exists locally.
    Returns the local path.
    """
    dest_path = _local_path(box_file)

    if dest_path.exists():
        logger.debug("Already exists locally, skipping: %s", dest_path)
        return dest_path

    if dry_run:
        logger.info("[dry-run] Would download: %s → %s", box_file.box_url, dest_path)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

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
# Review log
# ---------------------------------------------------------------------------

def _write_review_log(review: list[BoxFile], dry_run: bool = False) -> None:
    """Write needtochunk/review_files.txt listing every REVIEW file."""
    lines = [
        "Files requiring manual review",
        "These matched neither INCLUDE_TERMS nor EXCLUDE_TERMS in box_ingest/filter.py.",
        "For each file, open it on Box and decide:",
        "  - Add a keyword to INCLUDE_TERMS  → file will be downloaded on next run",
        "  - Add a keyword to EXCLUDE_TERMS  → file will be ignored on next run",
        "  - Manually add to url_manifest.json if it's a one-off inclusion",
        "",
        f"Total: {len(review)} file(s)",
        "=" * 60,
        "",
    ]
    for f in review:
        lines.append(f"Name:     {f.name}")
        lines.append(f"Folder:   {f.folder_path or '(root)'}")
        lines.append(f"Box URL:  {f.box_url}")
        lines.append("")

    content = "\n".join(lines)

    if dry_run:
        logger.info("[dry-run] Would write review log with %d entries", len(review))
        return

    NEEDTOCHUNK_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_LOG.write_text(content, encoding="utf-8")
    logger.info("Review log written: %s (%d files)", REVIEW_LOG, len(review))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    # Step 1: crawl
    logger.info("=== Step 1: Crawling Box hub ===")
    access_token = get_access_token()
    files = crawl_hub(access_token)

    include = [f for f in files if f.decision == FILTER_INCLUDE]
    exclude = [f for f in files if f.decision == FILTER_EXCLUDE]
    review  = [f for f in files if f.decision == FILTER_REVIEW]

    logger.info("Found %d include / %d exclude / %d review", len(include), len(exclude), len(review))

    # Step 2: download INCLUDE files (skip already-downloaded ones)
    to_download = [f for f in include if not _already_downloaded(f)]
    logger.info("=== Step 2: Downloading %d new file(s) (%d already local) ===",
                len(to_download), len(include) - len(to_download))
    for box_file in to_download:
        download_file(box_file, access_token, dry_run=dry_run)

    # Step 3: update manifest
    logger.info("=== Step 3: Updating url_manifest.json ===")
    summary = update_manifest(files, dry_run=dry_run)
    logger.info(
        "Manifest: %d added/filled, %d already present, %d non-include skipped",
        summary["added"], summary["already_present"], summary["skipped_non_include"],
    )

    # Step 4: write review log
    logger.info("=== Step 4: Writing review log ===")
    _write_review_log(review, dry_run=dry_run)
    if review:
        print(f"\nReview log written to: {REVIEW_LOG}")
        print(f"{len(review)} file(s) need manual triage — open the log for details.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Automate Box SBE materials ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no downloads or file writes")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
