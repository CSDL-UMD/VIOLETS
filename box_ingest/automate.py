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
     A per-file download state (data/box_download.state.json) records each
     file's Box sha1, so files updated in Box are re-downloaded; unchanged
     files are skipped. Downloads are written to a .tmp file, verified
     against Box's size/sha1, then atomically renamed into place.
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
import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path

from box_ingest.crawler import crawl_hub, BoxFile, SHARED_LINK, get_access_token
from box_ingest.filter import FILTER_INCLUDE, FILTER_EXCLUDE, FILTER_REVIEW
from box_ingest.manifest import update_manifest
from box_ingest.paths import NEEDTOCHUNK_DIR, REVIEW_LOG, DOWNLOAD_STATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_path(box_file: BoxFile) -> Path:
    # Reject Box-supplied names that try to escape the download root.
    for component in (*Path(box_file.folder_path).parts, box_file.name):
        if component == ".." or "/" in component or "\\" in component:
            raise ValueError(f"Unsafe Box path component: {component!r}")
    base = NEEDTOCHUNK_DIR / box_file.folder_path if box_file.folder_path else NEEDTOCHUNK_DIR
    candidate = base / box_file.name
    root = NEEDTOCHUNK_DIR.resolve()
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes download root: {candidate}")
    return candidate


def _rel_key(box_file: BoxFile) -> str:
    """Manifest-style key: path relative to needtochunk/ (e.g. '2026-03/report.pdf')."""
    return str(_local_path(box_file).relative_to(NEEDTOCHUNK_DIR))


def _sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download state (data/box_download.state.json)
#
# Maps rel_key -> {"sha1": <Box content sha1>, "size": <bytes>} for every
# successfully downloaded file. Same load/save pattern as the Step 2 chunk
# cache (data/box_ingest.state.json): tolerate a corrupt file, write
# atomically via tmp + replace.
# ---------------------------------------------------------------------------

def _load_download_state() -> dict:
    if not DOWNLOAD_STATE.exists():
        return {}
    try:
        with open(DOWNLOAD_STATE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected dict, got {type(data).__name__}")
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception as exc:
        logger.warning("Failed to read download state (%s); rebuilding from scratch", exc)
        return {}


def _save_download_state(state: dict) -> None:
    DOWNLOAD_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DOWNLOAD_STATE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(DOWNLOAD_STATE)


def _plan_action(box_file: BoxFile, state: dict) -> str:
    """Decide what to do with one INCLUDE file: 'new' | 'updated' | 'unchanged'.

    - Missing locally                      → 'new' (download).
    - Recorded sha1 differs from Box's     → 'updated' (re-download).
    - No recorded sha1 (pre-tracking file) → verify by hashing the local
      bytes against Box's sha1: match → 'unchanged' (state backfilled),
      mismatch → 'updated'.
    - Box reported no sha1                 → 'unchanged' (nothing to compare).
    """
    dest = _local_path(box_file)
    if not dest.exists():
        return "new"
    if not box_file.sha1:
        return "unchanged"
    rel_key = _rel_key(box_file)
    recorded = (state.get(rel_key) or {}).get("sha1", "")
    if recorded:
        return "unchanged" if recorded == box_file.sha1 else "updated"
    if _sha1_of(dest) == box_file.sha1:
        state[rel_key] = {"sha1": box_file.sha1, "size": box_file.size}
        return "unchanged"
    return "updated"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(box_file: BoxFile, access_token: str, dry_run: bool = False) -> bool:
    """
    Download a single Box file into needtochunk/<folder_path>/<name>.

    Writes to <name>.tmp in the destination directory, verifies the byte
    count against Box's size field (and content sha1 when Box reports one),
    then atomically renames into place — a failed or truncated download
    never replaces an existing good copy or leaves a partial final file.
    Returns True on success, False on failure (already logged).
    """
    dest_path = _local_path(box_file)

    if dry_run:
        logger.info("[dry-run] Would download: %s → %s", box_file.box_url, dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")

    download_url = f"https://api.box.com/2.0/files/{box_file.file_id}/content"
    req = urllib.request.Request(download_url)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("BoxApi", f"shared_link={SHARED_LINK}")

    logger.info("Downloading: %s", box_file.name)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        tmp_path.write_bytes(data)
        if box_file.size >= 0 and len(data) != box_file.size:
            raise ValueError(f"size mismatch: got {len(data)} bytes, Box reports {box_file.size}")
        if box_file.sha1 and hashlib.sha1(data).hexdigest() != box_file.sha1:
            raise ValueError("sha1 mismatch against Box-reported content hash")
    except Exception as exc:
        logger.error("Download failed for %s: %s", box_file.name, exc)
        tmp_path.unlink(missing_ok=True)
        return False

    os.replace(tmp_path, dest_path)
    logger.info("Saved to: %s", dest_path)
    return True


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

# Pipeline-owned files under needtochunk/ that never correspond to a Box file.
DRIFT_SKIP_NAMES = {"url_manifest.json", "review_files.txt", "README.md", ".DS_Store"}


def _report_local_drift(files: list[BoxFile]) -> None:
    """INFO-log local files under needtochunk/ with no current Box counterpart.

    Report-only: the operator's workflow drops and re-ingests the database
    wholesale, so lingering local files are accepted — surface them so drift
    is visible, never delete them.
    """
    if not NEEDTOCHUNK_DIR.exists():
        return
    current: set[str] = set()
    for f in files:
        try:
            current.add(_rel_key(f))
        except ValueError:
            continue  # unsafe Box path — never downloaded locally anyway
    drifted = [
        str(fp.relative_to(NEEDTOCHUNK_DIR))
        for fp in sorted(NEEDTOCHUNK_DIR.rglob("*"))
        if fp.is_file()
        and fp.name not in DRIFT_SKIP_NAMES
        and str(fp.relative_to(NEEDTOCHUNK_DIR)) not in current
    ]
    if drifted:
        logger.info("%d local file(s) no longer correspond to any Box file (kept, not deleted):",
                    len(drifted))
        for rel in drifted:
            logger.info("  [drift] %s", rel)


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

    # Step 2: download new files and re-download files updated in Box.
    state = _load_download_state()
    plan: dict[str, list[BoxFile]] = {"new": [], "updated": [], "unchanged": []}
    for f in include:
        plan[_plan_action(f, state)].append(f)

    logger.info("=== Step 2: Downloading %d new + %d updated file(s) (%d unchanged) ===",
                len(plan["new"]), len(plan["updated"]), len(plan["unchanged"]))
    failed = 0
    for box_file in plan["new"] + plan["updated"]:
        if download_file(box_file, access_token, dry_run=dry_run):
            state[_rel_key(box_file)] = {"sha1": box_file.sha1, "size": box_file.size}
        else:
            failed += 1

    # Drop state entries for files no longer on disk so the state can't
    # grow stale entries forever (mirrors the Step 2 chunk cache).
    stale = [k for k in state if not (NEEDTOCHUNK_DIR / k).exists()]
    for k in stale:
        state.pop(k, None)

    if not dry_run:
        _save_download_state(state)
    logger.info("Download summary: %d new, %d updated, %d unchanged, %d failed",
                len(plan["new"]), len(plan["updated"]), len(plan["unchanged"]), failed)

    # Report-only: surface local files with no current Box counterpart.
    _report_local_drift(files)

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
