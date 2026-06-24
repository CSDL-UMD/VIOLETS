# VIOLETS Comprehensive Codebase Audit — Plan

## Context

The user (senior staff perspective) asked for a thorough audit of the VIOLETS RAG codebase (~6.5k LOC across `server/`, `maryland_rag/`, `box_ingest/`). The code was shipped fast; the audit is meant to surface what could bite in production. Scope per the user's clarification:

- **Dimensions covered:** 5 (Data Integrity), 6 (Architecture & Design), 7 (Best Practices & Code Quality) — *only* these three.
- **Deliverable:** a single `AUDIT.md` at the repo root with a top-section "Top 10 Fix-Now" triage table cutting across dimensions, then per-dimension finding lists grouped by severity.
- **Remediation roadmap:** *not* included. Each finding has a concrete fix; sequencing is left to whoever picks up the work.

Three parallel Explore passes have already scanned the codebase. The plan below records the deliverable's exact shape and the synthesized findings the document will contain.

---

## Deliverable

**File:** `/Users/ryan/Downloads/VIOLETS/AUDIT.md` (new, at repo root).

**Structure:**

1. Executive summary (one paragraph: scope, severity counts, headline themes).
2. **Top 10 Fix-Now** — cross-dimension triage table, ranked by blast radius.
3. **Data Integrity findings** — grouped Critical / High / Medium.
4. **Architecture & Design findings** — grouped Critical / High / Medium.
5. **Code Quality findings** — grouped High / Medium / Low.
6. Cross-cutting themes (one short list).
7. Methodology / coverage note.

Each finding follows the required schema: **file:line — severity — issue — why it matters — concrete fix**. Findings already provided in the three agent passes are deduplicated below (e.g., the dedup-map / `GROUP_CONCAT` issue showed up in both Data Integrity and Architecture; it's reported once in Data Integrity and cross-referenced from Architecture).

---

## Top 10 Fix-Now (preview of audit's lead section)

| # | Severity | File:line | Issue | Blast radius |
|---|---|---|---|---|
| 1 | Critical | `maryland_rag/pass1/db.py:77-80` | SQLite connection uses default autocommit (`isolation_level` not set); multi-statement updates have no transaction boundary even though WAL is enabled | Mid-crawl crash leaves DB partially updated; not recoverable |
| 2 | Critical | `maryland_rag/pass2/metadata.py:101-104` + `maryland_rag/pass1/db.py:239-246` | Dedupe `chunk_id` derives from `sorted(source_urls)` joined as a string; `GROUP_CONCAT(url)` truncates silently past the SQLite limit, and ordering shifts re-hash the chunk → duplicate vectors in pgvector under different IDs | Permanent index pollution, no detection path |
| 3 | Critical | `maryland_rag/pass2/metadata.py:105` ↔ `server/rag_chain.py:118-134` | Deduped chunks store `source_urls` (list) in JSONB but the server retriever only reads the scalar `source_url` column → all sources beyond the first are dropped from citations | User-facing citations are wrong / incomplete |
| 4 | Critical | `maryland_rag/pass3/embed.py:186-200` | `ON CONFLICT (chunk_id) DO UPDATE` silently replaces the embedding when dedup keys shift between runs; no audit trail, no orphan cleanup | Vector store drifts from manifest with no way to reconcile |
| 5 | Critical | `box_ingest/ingest.py:82-86` + `box_ingest/manifest.py:30-65` | Box state file and manifest are read-modify-write JSON with no file lock and (for `manifest.py`) no temp+rename; concurrent `automate.py` runs clobber each other; crash mid-write corrupts state | Cache wipeout → re-OCR everything; lost manifest entries → chunks never embedded |
| 6 | Critical | `server/middleware.py:179, 250-259, 384-392` + `server/config.py:23-46` | Presidio analyzer, both `ChatOpenAI` clients, and required env-var validation all execute at module import time | Importing `server.middleware` blocks ~3s and ~500MB; config rotation requires restart; tests can't mock |
| 7 | High | `server/main.py:109-119` | `store`, `chain`, `_rag_callback`, `_pool` are module-level mutables initialized inside `lifespan()` via `global`; endpoints access them with no guard | NoneType on early requests; non-testable; hidden init order |
| 8 | High | `box_ingest/ingest.py:37-48` | `box_ingest` reaches directly into `maryland_rag.pass2` strategy internals (`extract_pdf_from_path`, `semantic_chunk`, `build_chunk_metadata`) | Any pass2 refactor breaks ingest; no public seam |
| 9 | High | `maryland_rag/pass2/chunker.py:44, 51-54, 285-292` | Pass 2 directly opens the Pass 1 SQLite DB and parses `GROUP_CONCAT` strings; schema changes ripple across passes with no contract | Brittle pipeline; can't evolve pass 1 schema |
| 10 | High | `server/middleware.py:329-359` | `classify_query` sets `ctx.query_category` on success but the `except` path returns `None` *without* setting a default → downstream (`rag_chain.py:211`) sees `None` and silently misbehaves | Silent category fallthrough on LLM failure |

---

## Full finding list (one-liners, audit will expand each into the full schema)

### Data Integrity

**Critical**
- `pass1/db.py:77-80` — Default SQLite autocommit; no transaction boundaries (#1 in Top 10).
- `pass1/db.py:94-99` + `pass1/crawler.py:129-216` — Concurrent / restarted crawls double-insert URLs; `add_page` + link insert not in one tx.
- `pass2/metadata.py:101-104` — Dedupe `chunk_id` is unstable under URL order / count shifts (#2).
- `pass2/metadata.py:105` ↔ `server/rag_chain.py:118-134` — `source_urls` list lost on retrieval (#3).
- `pass3/embed.py:186-200` — UPSERT overwrites old embeddings without audit (#4).
- `box_ingest/ingest.py:82-86` — `_save_state` not atomic; corruption on mid-write crash (#5).

**High**
- `box_ingest/manifest.py:30-65` — JSON manifest read-modify-write race (part of #5).
- `pass2/metadata.py:38-42` — Malformed `section_hierarchy` silently falls back to `[]`, no log.
- `pass3/embed.py:87-125` — Failed embedding batches drop chunks with no failure record.
- `box_ingest/crawler.py:99-104` — Token refresh failure jumps straight to interactive re-auth → hangs in headless runs; no retry cap.
- `pass1/db.py:239-246` + `pass2/chunker.py:285-291` — `GROUP_CONCAT(url)` truncation silently drops source URLs.
- `scripts/db_cleanup.py:66-87` — Pattern-based DELETE with no `--confirm` gate, no audit log, non-idempotent.
- `pass1/extractor.py:217-258` — PDF text-extractability probe reads only first N bytes; sparse PDFs misclassified as OCR-needed.
- `pass1/db.py:73-80` — Per-instance PRAGMA setup with no enforced singleton; multiple `DB()` instances can race.
- `server/rag_chain.py:115-141` — Retriever doesn't validate `source_url` non-null; broken citations possible.
- `box_ingest/crawler.py:60-66, 112` — OAuth callback handler doesn't validate `state` param (CSRF / code-injection risk).
- `scripts/apply_keep_filter.py:108-115` — `executemany` UPDATE with no `BEGIN`; partial-state failures are permanent.

**Medium**
- `pass2/cache.py:28-77` — Cache directory has no TTL / eviction; unbounded growth, stale-cache hits possible.
- `server/main.py:85-110` — In-memory rate limiter resets on restart and isn't shared across workers.
- `box_ingest/ingest.py:89-91` — Fingerprint = `size:mtime_ns` is fragile under copy / NTP adjustments → false cache misses.
- `scripts/reclassify.py:54-58` — `extracted_snippet` passed to classifier without UTF-8 / emptiness validation.
- `scripts/audit.py:56-58` — Audit only counts DB duplicates, doesn't verify Pass 2 dedup map matches.
- `server/session.py:49-52, 63-70` — `_sessions` accessed across an async background task; lock coverage is incomplete.

### Architecture & Design

**Critical**
- `server/main.py:109-119` — Module-level mutable globals filled by `lifespan` (#7).
- `server/middleware.py:179` — `AnalyzerEngine()` at import time (#6).
- `server/middleware.py:250-259, 384-392` — `ChatOpenAI` clients instantiated at import time (#6).

**High**
- `server/main.py:85-110` — Custom in-process rate limiter; no Redis / multi-instance support (also DI risk).
- `box_ingest/ingest.py:37-48` — Direct import of `pass2` strategy internals (#8).
- `pass2/chunker.py:44, 51-54` — Pass 2 directly queries Pass 1 SQLite (#9); pipeline handoff via implicit shared schema.
- `pass2/chunker.py:136-172` — `_route_to_strategy` is a 10-branch `if/elif` switch with scattered fallback logic; no registry.
- `server/rag_chain.py:110-142` — `PgVectorRetriever` embeds raw SQL with column names; no vector-store abstraction.
- `server/config.py:23-46` — Config validation runs at import (`_require_env`); fails before `lifespan` can react.
- `server/middleware.py` (whole file, 508 lines) — Conflates PII detection, classification, partisan filtering, fallback URL routing, and shared context object.
- `maryland_rag/__main__.py:81-112` — Pass 1 → Pass 2 → Pass 3 contract is implicit (DB / JSONL / pgvector); no schemas, no version stamps.

**Medium**
- Cross-module — Inconsistent error handling: pass1/pass2 swallow + continue, pass3 retries with backoff, server raises HTTPException, box ingest does silent fallback. No shared error taxonomy.
- `server/rag_logger.py:39-41` — Global `LOG_PROMPTS` / `LOG_RESPONSES` / `LOG_QUERIES` booleans hard-coded; no env override, no per-component granularity.
- `pass2/chunker.py:285-292` — Dedup map built from string-split of `GROUP_CONCAT`; junction table would be sturdier (data-integrity counterpart #2).
- `server/session.py` — Sessions in-memory only; lost on restart, not shared across workers.
- `pass2/strategies/semantic.py:16-20` — `TARGET_CHUNK_WORDS`, `MAX_CHUNK_WORDS`, `OVERLAP_RATIO` are module constants; not configurable per doc type.
- `server/middleware.py:250-259` — Same `LLM_MODEL` used for classification, partisan check, and the main chain.
- `server/middleware.py:419-509` — `check_partisan_response` knows the chain's input keys and prompt-injection format; should take a callback.
- Repo-wide — `PROJECT_ROOT` computed three different ways (`pass1/config.py`, `pass2/cache.py`, `box_ingest/paths.py`).

### Best Practices & Code Quality

**High**
- `server/middleware.py:329-359` — `classify_query` `except` returns `None` without defaulting `ctx.query_category` (#10).
- `pass1/db.py:101-137` — `update_page` accepts an unvalidated `result` dict; missing keys become silent NULLs.
- `pass2/strategies/pdf.py:39-82` — `extract_pdf_from_path` cascades pdfplumber → pymupdf → OCR but doesn't log which strategy failed or won.
- `server/rag_chain.py:144-147` — TODO acknowledges `_get_relevant_documents` is sync and blocks worker threads under load; no async impl, no tracked issue.
- `server/main.py:280-286` — Bare `except Exception` returns generic 502; loses error class for ops.

**Medium**
- `pass2/chunker.py:136-172` — `_route_to_strategy` mixes return shapes (`[str]` wrapped vs `[dict]` direct) across branches.
- `box_ingest/ingest.py:51-54` — `SHORT_DOC_WORDS = 150`, `FALLBACK_CHUNK_WORDS = 400` with no comment / justification.
- `server/rag_logger.py:48-56` — Cost table lists `gpt-5-nano` / `gpt-5-mini` / `gpt-5`; pricing fictional → cost logs misleading.
- `pass1/crawler.py:87-221` — `run_crawl` is 135 lines, 5 nested gate checks, `finally` sleeps unconditionally; high cyclomatic complexity.
- `pass2/metadata.py:66-69` — Strategy-supplied metadata fields not in the allowlist are silently dropped.
- `box_ingest/ingest.py:257-291` — `ProcessPoolExecutor` results merged into `state` dict without locking — works today because it happens in main thread, but the pattern is one careless edit from breaking.
- `pass1/extractor.py:69-78` — Word-count threshold for trafilatura → BeautifulSoup fallback isn't logged; no metric on fallback rate.
- `server/config.py:44-68` — Env var names UPPER-cased while Python attrs match; no `pydantic-settings`; no grep target for "which env vars matter."
- `pass1/rules.py:99 → 140 → 197` — Classification dispatches through 3 layers of private helpers; entry point unclear.
- `pass1/db.py` (all methods) — No return type hints; callers don't know the dict shape (`get_pending` etc.).
- `server/rag_chain.py:105` — `pool: Any  # psycopg_pool.ConnectionPool` — explicit `Any` instead of importing the type.
- `pass2/strategies/*.py` — `simple_split`, `semantic`, `single` re-implement chunk-boundary / min-max-word logic; should share a primitive.
- `server/middleware.py:261+` ↔ `server/rag_chain.py:35, 81` — Three system-prompt blocks live in separate files; should be centralized.

**Low**
- `pass3/embed.py:214-225` — `RETRY_DELAY * (attempt + 1)` is linear back-off but variable name suggests fixed delay.
- `server/rag_logger.py:67-70` — Dense `next((... for k,v ...), fallback)` would read better with a name.
- `pass2/chunker.py:92-120` — Redundant `isinstance(section_hierarchy, str)` check on a value already coerced to str.
- `box_ingest/crawler.py:67` — `log_message` overridden to silent `pass`; hides OAuth flow debug info.
- `pass1/utils.py:51-57` — `is_internal` does `any(d in netloc for d in DOMAINS)`; `DOMAINS` could be a `set`.
- `server/session.py:28, 61` — `_cleanup_expired` invoked from both `get_or_create` and the periodic task; double work.
- `pass1/classifier.py` — 32-line wrapper around `rules.classify_html`; could be inlined.
- `box_ingest/paths.py` — `Path(__file__).parent.parent` fragile to file moves.
- `pass1/crawler.py:227-228` — `urlparse` re-imported inside `_save_raw_html` (already imported at module top).
- `pass1/crawler.py` — Log-level usage inconsistent (`info` / `warning` / `error`) for similar events; no shared convention.

---

## Cross-cutting themes (will close the audit with this short list)

1. **Import-time side effects everywhere** — Presidio, LLMs, env validation, file path computation. Causes slow startup, untestable modules, and config-rotation requires restart.
2. **Implicit pipeline contracts** — Pass 1 / Pass 2 / Pass 3 communicate via "whatever the SQLite schema happens to be" and "whatever keys end up in chunk dicts." No schemas, no versioning, no validation at boundaries.
3. **Silent failure paths** — JSON parse fallbacks, missing dict keys, embedding batch drops, classification exceptions, PDF strategy cascades. All log-and-continue with no metric.
4. **Concurrency story is unfinished** — SQLite autocommit, file-based caches with no locking, in-memory rate limiter, in-memory sessions. The system works because nobody has hit the second writer yet.
5. **Dedup is the load-bearing wall most likely to fall over** — Three independent issues (chunk_id stability, `GROUP_CONCAT` truncation, `source_urls` vs `source_url`) all converge on the same feature, all silent, all data-loss-shaped.

---

## Verification

This is a static audit deliverable; "verification" means the reader can trust each finding:

- Every citation includes `file:line(s)` from the actual codebase as of `4a3781c` (current `main`). A reviewer can spot-check any finding by opening the cited lines.
- Top 10 items can be reproduced experimentally: e.g., issue #1 by killing the crawler mid-run and inspecting `manifest.db` for partial rows; issue #3 by retrieving a known dedup-hit URL and checking the `source_urls` JSONB vs returned citation; issue #5 by `kill -9` during `box_ingest` and inspecting `box_ingest.state.json`.
- No code changes are part of this deliverable. The audit lists fixes but does not apply them.

## Risks / open items

- The Explore passes did not run the code or query the live `manifest.db` / pgvector store. Findings about runtime behavior (e.g., race conditions, `GROUP_CONCAT` truncation actually happening) are based on code inspection — the audit will note these as "latent" where there's no direct evidence of impact yet.
- `box_ingest/automate.py` was scanned but not deeply read; if any of its behaviors are surprising, a follow-up pass may add 1–2 findings.
- Tests directory was not found in the repo; the audit will note the absence as context but not file it as a finding (out of scope per the user's dimensions 5/6/7 selection).
