"""
Manifest updater for url_manifest.json.

Merges newly-crawled BoxFile entries into the existing manifest without
overwriting or removing any existing entries.

Key: "<folder_path>/<filename>"  e.g. "2026-03/State Administrator's Report- March 26, 2026.pdf"
Value: the Box share URL for that file
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from box_ingest.crawler import BoxFile
from box_ingest.filter import FILTER_INCLUDE

logger = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).parent.parent
MANIFEST_PATH  = PROJECT_ROOT / "needtochunk" / "url_manifest.json"


def _load_raw() -> dict:
    """Load manifest as-is (preserving comment keys)."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def update_manifest(files: list[BoxFile], dry_run: bool = False) -> dict[str, int]:
    """
    Add any INCLUDE-classified BoxFiles that are not yet in the manifest.

    Returns a summary dict: {"added": N, "already_present": M, "skipped_non_include": K}
    Does NOT remove or overwrite existing entries.
    """
    raw = _load_raw()
    include_files = [f for f in files if f.decision == FILTER_INCLUDE]

    added = 0
    already_present = 0

    for box_file in include_files:
        key = f"{box_file.folder_path}/{box_file.name}".lstrip("/")
        if key in raw:
            existing_url = raw[key]
            if not existing_url:
                # Key exists but URL was blank — fill it in
                if not dry_run:
                    raw[key] = box_file.box_url
                logger.info("Filled blank URL: %s", key)
                added += 1
            else:
                already_present += 1
                logger.debug("Already present: %s", key)
        else:
            if not dry_run:
                raw[key] = box_file.box_url
            logger.info("New entry: %s → %s", key, box_file.box_url)
            added += 1

    if not dry_run and added > 0:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        logger.info("Manifest written: %d new/updated entries", added)
    elif dry_run:
        logger.info("[dry-run] Would add/update %d entries", added)

    skipped = len(files) - len(include_files)
    return {"added": added, "already_present": already_present, "skipped_non_include": skipped}
