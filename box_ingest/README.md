# box_ingest/ — Curated Box Document Pipeline

Documents the State Board shares from a private Box Hub (worker manuals, monthly
admin reports, etc.) aren't crawlable, so they get their own pipeline that lands
in the **same** pgvector store with the **same** chunk schema as the web crawl.
See root [README](../README.md) Section 6 for the narrative.

Two steps, run independently:

```
Step 1  box_ingest.automate   Box Hub ──▶ needtochunk/ + url_manifest.json
Step 2  box_ingest.ingest     needtochunk/ ──▶ data/box_chunks.jsonl
```

`python -m maryland_rag all` runs **Step 2 only**. Step 1 (downloading from Box)
is always a manual command.

## Step 1 — auto-download (`automate.py`)

```bash
python -m box_ingest.automate            # crawl, download, update manifest
python -m box_ingest.automate --dry-run  # preview only
```

Requires `BOX_CLIENT_ID` / `BOX_CLIENT_SECRET` in `.env` and Playwright
(`pip install -r box_ingest/requirements.txt && playwright install chromium`).

Flow: OAuth (browser once, then `.box_token` refresh) → Playwright scrapes the
Hub for folder IDs → Box REST API lists each `2026-*` folder (with `sha1`/`size`
per file) → `filter.py` classifies each filename `include`/`exclude`/`review` →
`include` files downloaded into `needtochunk/<folder>/<name>` → `manifest.py`
merges new entries into `url_manifest.json` (never overwriting) → `review` files
written to `needtochunk/review_files.txt` for a human decision.

Update detection: `data/box_download.state.json` maps each downloaded file
(path relative to `needtochunk/`) to `{"sha1": …, "size": …}` from Box. On each
run a file is **new** (missing locally → download), **updated** (Box sha1
differs from the recorded one — or, when no sha1 was recorded, from a hash of
the local bytes → re-download), or **unchanged** (skip). Downloads are written
to `<name>.tmp`, verified against Box's size/sha1, then atomically renamed, so
a truncated download never replaces a good copy. Local files with no current
Box counterpart are logged (`[drift]`) but never deleted — the operator's
drop-and-reingest workflow accepts lingering files.

## Step 2 — extract & chunk (`ingest.py`)

```bash
python -m box_ingest.ingest              # → data/box_chunks.jsonl
python -m box_ingest.ingest --dry-run
python -m box_ingest.ingest --output PATH
python -m box_ingest.ingest --workers 8  # default: CPU count
```

- Reuses the `maryland_rag.pass2` extractors: `.pdf` (pdfplumber → pymupdf → OCR
  @ 300 dpi), `.docx`/`.doc` (heading hierarchy), `.xlsx`/`.xlsm` (one chunk per
  row), `.txt` (plain). Unknown types are skipped.
- Chunking (PDF/DOCX/TXT): ≤150 words → `ingest_as_single`, else `semantic_chunk`.
- `chunk_id` is content-derived (shared `_content_chunk_id`), **not** the URL.
- Parallel extraction (`ProcessPoolExecutor`) with a skip-if-unchanged cache in
  `data/box_ingest.state.json` (size+mtime fingerprint). Files with no
  `url_manifest.json` entry are skipped with a warning.
- Output rows use `page_classification='document'`,
  `chunking_strategy='box_ingest'` — Pass 3 ingests them identically to web chunks.

## Files

| File | Responsibility |
|---|---|
| `automate.py` | Step 1 orchestrator (crawl → download → manifest → review log). |
| `crawler.py` | Box Hub scraper (Playwright) + Box OAuth 2.0 / REST v2.0 client. A corrupt `.box_token` is dropped and triggers re-auth instead of crashing. `python -m box_ingest.crawler` prints a classified file listing. |
| `filter.py` | `INCLUDE_TERMS` / `EXCLUDE_TERMS` keyword rules → include/exclude/review. |
| `manifest.py` | Merge new `include` files into `url_manifest.json` (additive). |
| `ingest.py` | Step 2 extract + chunk. |
| `paths.py` | Shared path constants (`NEEDTOCHUNK_DIR`, `MANIFEST_PATH`, `STATE_PATH`, `DOWNLOAD_STATE`, `TOKEN_CACHE`, …). |
| `requirements.txt` | Playwright + document-extraction libs. |

## Auth & secrets

Box OAuth tokens are cached in `.box_token` at the repo root (mode `0600`,
gitignored). Rotate by deleting `.box_token` and re-authing, or by rotating the
app secret in the Box console.
