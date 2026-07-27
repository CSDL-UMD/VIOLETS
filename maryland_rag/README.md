# maryland_rag/ — Web Crawl → Chunk → Embed Pipeline

The three-pass web pipeline that crawls the allowlisted slice of
`elections.maryland.gov` + hand-picked Montgomery County pages, chunks the
content, and embeds it into pgvector. Full narrative in the root
[README](../README.md) (Sections 4–7, 10–14); this file is the package map.

## CLI

```bash
python -m maryland_rag pass1              # crawl        (--no-resume)
python -m maryland_rag pass2              # chunk        (--changed, --output)
python -m maryland_rag pass3              # embed        (--chunks, --resume)
python -m maryland_rag all                # full rebuild (--output)
python -m maryland_rag audit              # manifest report
```

`all` always chunks the **full corpus** (drop-and-reingest model) and aborts if
>20% of expected pages produce zero chunks. See root README Section 11.

## Layout

### `pass1/` — crawl & classify → `data/manifest.db`
| File | Responsibility |
|---|---|
| `config.py` | Seeds, `DOMAINS`, `MAX_DEPTH=6`, `MAX_RETRIES=3`, rate limit, paths. |
| `crawler.py` | BFS crawler; robots.txt per-domain (RFC 9309, fetched via `requests`); transient-failure retry that never demotes an already-`crawled` row. |
| `extractor.py` | trafilatura → BS4 fallback; links + breadcrumbs + `content_hash`; document HEAD probe (no PDF byte-probe — `needs_ocr` always `False`). |
| `classifier.py` → `rules.py` | Rule-based page classification (shared with `reclassify`). |
| `exclusions.py` | Allowlist + exclusion gate + status sets + translated-doc URL rule. Single source of truth. |
| `db.py` | All SQLite reads/writes (WAL). |
| `utils.py` | URL normalization. |

### `pass2/` — chunk → `data/chunks.jsonl`
| File | Responsibility |
|---|---|
| `chunker.py` | Orchestrates strategies; dedup by `content_hash`; runs the non-English filter. |
| `metadata.py` | Chunk metadata + content-derived `chunk_id` (`_content_chunk_id`). |
| `cache.py` | Disk cache (`data/cache/`); HTML written by Pass 1; `.bin` revalidated once/run via ETag/Last-Modified sidecars. No TTL. |
| `langfilter.py` | Drops chunks above `MAX_NON_LATIN_RATIO=0.10` non-Latin letters. |
| `strategies/` | `single`, `simple_split`, `semantic` (+ shared `enforce_chunk_caps`), `faq`, `table_rows`, `pdf` (digital-first, OCR fallback @ 300 dpi), `docx_strategy`, `xls_strategy` (skips >`MAX_XLS_ROW_CHUNKS=200`). |

### `pass3/` — embed → pgvector
`embed.py`: batches of 100 → `text-embedding-3-small` (1536-dim); linear
backoff + jitter; deterministic-4xx bisection; `ON CONFLICT DO UPDATE` upsert
with per-row savepoints.

### `scripts/` — maintenance & verification
| Script | Purpose |
|---|---|
| `db_cleanup.py` | Remove duplicate URL variants + junk (timestamped backup first). |
| `reclassify.py` | Re-run `rules.py` on stored metadata (no re-crawl). |
| `apply_keep_filter.py` | Mark out-of-scope rows `excluded` (`keep_filter_2026`). |
| `audit.py` | Manifest audit report (backs `maryland_rag audit`). |
| `verify_chunks.py` | Post-ingest gate: coverage / junk / pgvector parity. |
| `stress_test.py` | Server concurrency + edge-case harness. |

`db_cleanup` / `reclassify` default to **apply**; pass `--dry-run` to preview.
`apply_keep_filter` requires one of `--dry-run` / `--apply` / `--revert`.
