# needtochunk/ — Curated Box Document Drop

The staging area for Box-hosted documents that feed the Box ingest pipeline
([box_ingest/](../box_ingest/README.md)). Everything here is chunked by
`python -m box_ingest.ingest` into `data/box_chunks.jsonl` and embedded into the
same pgvector store as the web crawl.

## Contents

```
needtochunk/
├── url_manifest.json      ← relative path → Box share URL (required to be indexed)
├── review_files.txt       ← Box files awaiting a keep/drop decision (written by box_ingest.automate)
├── <year-folder>/         ← e.g. 2026-01 … 2026-06
│   └── <file>.pdf|docx|xlsx|txt
└── <loose file>.txt       ← files with no year folder are allowed too
```

## `url_manifest.json`

Maps each file's path **relative to `needtochunk/`** to its Box share URL, so a
retrieved chunk can cite a working link. Keys beginning with `_` (`_comment`,
`_format`) are documentation and ignored by the ingester.

```json
{
  "_comment": "Maps relative paths (from needtochunk/) to Box.com URLs.",
  "Montgomery County Election Day Plans.txt": "https://mdsbe.app.box.com/s/.../file/<id>",
  "2026-02/State Administrator's Report- February 19, 2026.pdf": "https://..."
}
```

A file whose mapping is **missing or empty is skipped with a warning** — it will
not be embedded. `url_manifest.json`, `review_files.txt`, and `.DS_Store` are
always skipped.

## How files get here

- **Automated:** `python -m box_ingest.automate` downloads `include`-classified
  files from the Box Hub and adds their manifest entries automatically.
- **Manual:** drop a file into the right folder and add its Box URL to
  `url_manifest.json`. Files listed in `review_files.txt` are ones the keyword
  filter couldn't auto-decide — add a keyword to `box_ingest/filter.py` or
  hand-add the file to the manifest.

Re-running ingest is cheap: a skip-if-unchanged cache
(`data/box_ingest.state.json`) only re-extracts files whose size/mtime changed,
and content-derived `chunk_id`s make Pass 3's upsert a no-op for unchanged docs.
