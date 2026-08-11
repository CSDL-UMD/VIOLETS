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
`pass2 --changed` re-chunks only changed pages and **merges** them into the
existing `chunks.jsonl` (unchanged pages' rows are kept; reprocessed and
no-longer-chunkable pages' rows are dropped). Documents are always cheaply
revalidated (conditional GET + sha256 vs the snapshot) and re-chunked only
when genuinely changed; dedup partners sharing a multi-source row with a
reprocessed page are re-chunked in the same run. Pages whose new state is
reflected in the merged file — chunk-producing AND deliberate-zero pages
(skip strategy, legacy `.doc`, langfilter-emptied) — are snapshotted;
pages whose fetch/extraction **failed** keep their existing rows and are
retried by the next `--changed` run.

Chunk text carries a one-line **context header** — `[<title> — <section>]` —
prepended by the chunker (except where the strategy already emits a
caption/sheet line or the chunk starts with the title), so bare prose/FAQ/
document chunks embed with their page context.

## Layout

### `pass1/` — crawl & classify → `data/manifest.db`
| File | Responsibility |
|---|---|
| `config.py` | Seeds, `DOMAINS`, `USER_AGENT` (sent on all requests), `MAX_DEPTH=6`, `MAX_RETRIES=3`, rate limit, paths. |
| `crawler.py` | BFS crawler; robots.txt per-domain (RFC 9309, fetched via `requests`; **aborts the run** if unreachable after retries instead of sticky-excluding the domain); transient-failure retry-with-backoff (403 WAF challenges included) that never demotes an already-`crawled` row; off-allowlist redirect targets are excluded only if never crawled (a challenge redirect preserves the row); startup retroactive pass flips existing rows matching newly-added exclusion rules; page + links + child rows commit in one transaction. |
| `extractor.py` | trafilatura → BS4 fallback; charset sniffed when the header omits it; links + breadcrumbs + `content_hash` + post-redirect `final_url`; document HEAD probe (no PDF byte-probe — `needs_ocr` always `False`). |
| `classifier.py` → `rules.py` | Rule-based page classification (shared with `reclassify`). |
| `exclusions.py` | Allowlist (incl. `/elections/2026/`) + exclusion gate + status sets + translated-doc URL rules (language names and `-ES`/`_KO`-style suffixes; `vi`/`fr` tokens omitted — Roman numerals / "FR" abbreviations). `matches_exclusion_rules()` is the allowlist-free layer used by the retroactive pass. Single source of truth. |
| `db.py` | All SQLite reads/writes (WAL). `update_document_hash()` lets Pass 2 store document hashes (preserved across recrawls via `COALESCE`); `get_changed_pages()` returns hash-changed HTML plus **all** document rows for cheap revalidation. |
| `utils.py` | URL normalization (upgrades `http://` → `https://` on target domains). |

### `pass2/` — chunk → `data/chunks.jsonl`
| File | Responsibility |
|---|---|
| `chunker.py` | Orchestrates strategies; dedup by `content_hash`; runs the non-English filter; prepends the context header (filename-like section leaves skipped; title-suppression is word-boundary-safe); persists document sha256 via `update_document_hash`; per-page outcome tracking (success / deliberate-zero / unchanged / failed) drives the `--changed` merge + snapshot; atomic pid-suffixed corpus write; skips legacy `.doc` loudly. |
| `metadata.py` | Chunk metadata + content-derived `chunk_id` (`_content_chunk_id`). |
| `cache.py` | Disk cache (`data/cache/`); HTML written by Pass 1; `.bin` revalidated once/run via ETag/Last-Modified sidecars. No TTL. Binary payloads magic-validated (`%PDF-`/`PK`/OLE2) — a poisoned entry (HTML nav page under a document URL) is purged + refetched once, else zero chunks; startup `purge_poisoned` scan. |
| `langfilter.py` | Drops chunks above `MAX_NON_LATIN_RATIO=0.10` non-Latin letters (with a `MIN_NON_LATIN_LETTERS=15` floor so language-access lines survive) and Spanish leakage via `is_spanish` function-word frequency. |
| `strategies/` | `single`, `simple_split`, `semantic` (+ shared `enforce_chunk_caps`), `faq`, `table_rows` (header row never re-emitted; consecutive rows grouped ~200–1000 chars with `row_start`/`row_end`), `pdf` (digital-first, OCR fallback @ 300 dpi), `docx_strategy`, `xls_strategy` (same row grouping; stdlib-csv reader; skips sheets over `MAX_XLS_ROW_CHUNKS=200` **raw** rows, counted pre-grouping). |

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
| `verify_chunks.py` | Post-ingest gate: coverage / junk (incl. nav boilerplate + Spanish leakage) / duplicate `chunk_id`s (fails only cross-document) / length bands (warn) / Box manifest coverage / pgvector parity. |
| `stress_test.py` | Server concurrency + edge-case harness. |

`db_cleanup` / `reclassify` default to **apply**; pass `--dry-run` to preview.
`apply_keep_filter` requires one of `--dry-run` / `--apply` / `--revert`.
