# VIOLETS — Maryland Elections RAG Pipeline

## Table of Contents

1. [Big Picture: How the Pipeline Works](#1-big-picture-how-the-pipeline-works)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Pass 1 — Crawling & Classification](#4-pass-1--crawling--classification)
5. [Pass 2 — Chunking](#5-pass-2--chunking)
6. [Box Ingest — Curated Document Pipeline](#6-box-ingest--curated-document-pipeline)
7. [Pass 3 — Embedding & pgvector Upload](#7-pass-3--embedding--pgvector-upload)
8. [Server — FastAPI Chatbot](#8-server--fastapi-chatbot)
9. [Operating the Server (Auth, CORS, Rate Limit)](#9-operating-the-server-auth-cors-rate-limit)
10. [Utility Scripts](#10-utility-scripts)
11. [Running the Full Pipeline](#11-running-the-full-pipeline)
12. [The Database (manifest.db)](#12-the-database-manifestdb)
13. [Chunking Strategies — Deep Dive](#13-chunking-strategies--deep-dive)
14. [Configuration Reference](#14-configuration-reference)
15. [Security Warning](#15-security-warning)
16. [Troubleshooting](#16-troubleshooting)
17. [Glossary](#17-glossary)

---

## 1. Big Picture: How the Pipeline Works

```
   ┌───────────────────────────┐   ┌─────────────────────────────┐
   │  elections.maryland.gov   │   │  mcg.montgomerycountymd.gov │
   │  (allowlisted paths)      │   │  (allowlisted pages only)   │
   └────────────┬──────────────┘   └──────────────┬──────────────┘
                │                                  │
                └──────────────┬───────────────────┘
                               │  PASS 1: Crawl (BFS w/ allowlist)
                               ▼
                  ┌────────────────────────────┐
                  │  data/manifest.db (SQLite) │
                  │  one row per URL,          │
                  │  classified + tagged       │
                  └────────────┬───────────────┘
                               │  PASS 2: Chunk web content
                               ▼
                  ┌────────────────────────────┐    ┌────────────────────────┐
                  │  data/chunks.jsonl         │    │  needtochunk/  (Box)   │
                  │  (one chunk per record)    │    │  + url_manifest.json   │
                  └────────────┬───────────────┘    └───────────┬────────────┘
                               │                                │  BOX INGEST
                               │                                ▼
                               │                    ┌────────────────────────┐
                               │                    │  data/box_chunks.jsonl │
                               │                    └───────────┬────────────┘
                               │                                │
                               └────────┬───────────────────────┘
                                        │  PASS 3: Embed + upsert (twice — once per JSONL)
                                        ▼
                          ┌─────────────────────────────────┐
                          │  PostgreSQL + pgvector          │
                          │  ready for semantic search      │
                          └────────────────┬────────────────┘
                                           │  SERVER: /chat
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │  FastAPI chatbot (X-API-Key required)    │
                          │  Guardrails: PII → classify (8 paths) →  │
                          │  RAG (or hardcoded URL / blocked) →      │
                          │  partisan check w/ retry                 │
                          └──────────────────────────────────────────┘
```

Each pass reads only from the previous stage's output. Re-running a single pass is safe: Pass 1 resumes from `pending` rows, Pass 2 has `--changed` mode, Pass 3 has `--resume` and uses `ON CONFLICT … DO UPDATE` upserts.

Embedding model: `text-embedding-3-large` (3,072-dim, OpenAI). Set in two places that must match: `maryland_rag/pass3/embed.py` `EMBED_MODEL` and `server/rag_chain.py` `EMBED_MODEL`.
Default LLM: `gpt-5-nano` (overridable via `LLM_MODEL`).

---

## 2. Repository Structure

Each top-level code package (`maryland_rag/`, `server/`, `box_ingest/`) and the
`needtochunk/` document drop also carry their own `README.md` describing that
folder's files in detail; this root README is the end-to-end reference.

```
VIOLETS/
├── .env                              ← API keys + Box OAuth creds (NEVER commit to GitHub)
├── .box_token                        ← Cached Box OAuth tokens (gitignored, mode 0600)
├── README.md                         ← This file
│
├── data/                             ← All pipeline artifacts (gitignored)
│   ├── manifest.db                   ← SQLite crawl database (Pass 1 output)
│   ├── chunks.jsonl                  ← Web chunks (Pass 2 output)
│   ├── box_chunks.jsonl              ← Box document chunks (Box Ingest output)
│   ├── box_ingest.state.json         ← Box ingest skip-if-unchanged cache
│   └── cache/                        ← Disk cache of fetched pages / binaries (+ .bin.meta sidecars)
│
├── logs/                             ← Runtime logs (gitignored)
│   ├── crawl.log                     ← Crawl activity log
│   └── server.log                    ← Server log (rotating, 10 MB × 5)
│
├── needtochunk/                      ← Curated documents from Box
│   ├── README.md                     ← How the document drop + manifest work
│   ├── <year-folder>/<file>.pdf      ← PDFs, DOCX, XLSX, TXT, etc.
│   ├── url_manifest.json             ← Maps each file's relative path → Box share URL
│   └── review_files.txt              ← Triage log of Box files needing a keep/drop decision
│
├── maryland_rag/                     ← Web pipeline Python package
│   ├── README.md                     ← Package-level guide
│   ├── __main__.py                   ← CLI entry point (pass1 / pass2 / pass3 / all / audit)
│   ├── requirements.txt              ← Pipeline Python dependencies
│   │
│   ├── pass1/                        ← Phase 1: Crawl the allowlisted sites
│   │   ├── config.py                 ← Seeds, domains, rate limit, retries, paths
│   │   ├── crawler.py                ← BFS web crawler (robots.txt, transient-failure retry)
│   │   ├── extractor.py              ← Text/link/metadata extraction (HTML + doc HEAD probe)
│   │   ├── classifier.py             ← Crawl-time wrapper over rules.py
│   │   ├── rules.py                  ← Shared classification rules (used by reclassify too)
│   │   ├── exclusions.py             ← Allowlist + exclusion gate (single source of truth)
│   │   ├── db.py                     ← All SQLite read/write operations
│   │   └── utils.py                  ← URL normalization helpers
│   │
│   ├── pass2/                        ← Phase 2: Break pages into chunks
│   │   ├── chunker.py                ← Orchestrates all chunking
│   │   ├── metadata.py               ← Builds chunk metadata (content-derived chunk_id, dedup)
│   │   ├── cache.py                  ← Disk cache of HTTP fetches (freshness-aware)
│   │   ├── langfilter.py             ← Non-English chunk filter (non-Latin ratio)
│   │   └── strategies/               ← One file per chunking approach
│   │       ├── single.py             ← Entire page as one chunk
│   │       ├── simple_split.py       ← Split at paragraph boundaries
│   │       ├── semantic.py           ← Sentence splits with overlap + shared chunk-cap enforcement
│   │       ├── faq.py                ← Extract Q&A pairs
│   │       ├── table_rows.py         ← One chunk per HTML table row
│   │       ├── pdf.py                ← Extract text from PDFs (pdfplumber → pymupdf → OCR)
│   │       ├── docx_strategy.py      ← Extract DOCX by heading hierarchy
│   │       └── xls_strategy.py       ← Extract XLS/XLSX rows
│   │
│   ├── pass3/                        ← Phase 3: Embed and upsert
│   │   └── embed.py                  ← OpenAI embeddings + pgvector upsert
│   │
│   └── scripts/                      ← Maintenance & verification utilities
│       ├── db_cleanup.py             ← Remove duplicate URL variants from manifest.db
│       ├── reclassify.py             ← Re-classify pages using stored metadata
│       ├── audit.py                  ← Manifest audit report (backs `maryland_rag audit`)
│       ├── verify_chunks.py          ← Post-ingest gate: coverage / junk / pgvector parity
│       └── stress_test.py            ← Server-side concurrency + edge-case test harness
│
├── box_ingest/                       ← Curated Box document pipeline (download + chunk)
│   ├── README.md                     ← Box subsystem guide
│   ├── automate.py                   ← Step 1: crawl Box hub → download → update manifest
│   ├── crawler.py                    ← Box Hub scraper (Playwright) + Box OAuth/API client
│   ├── filter.py                     ← Keyword include/exclude/review rules for filenames
│   ├── manifest.py                   ← Merge new files into url_manifest.json
│   ├── paths.py                      ← Shared filesystem path constants
│   ├── ingest.py                     ← Step 2: extract needtochunk/ → data/box_chunks.jsonl
│   └── requirements.txt              ← Box subsystem dependencies (incl. playwright)
│
└── server/                           ← FastAPI chatbot server
    ├── README.md                     ← Server guide
    ├── main.py                       ← App, lifespan, /chat, /reset, /health, auth, rate limit
    ├── config.py                     ← Loads .env, exposes settings
    ├── rag_chain.py                  ← LangChain RAG chain + async PgVectorRetriever
    ├── middleware.py                 ← PII, classification, partisan-check guardrails
    ├── rag_logger.py                 ← Callback handler for token/cost/timing logging
    ├── logging_setup.py              ← Central logging config + per-request IDs
    ├── metrics.py                    ← In-process counters feeding the heartbeat log
    ├── session.py                    ← Thread-safe in-memory conversation store
    ├── eval_guardrails.py            ← Offline eval: reasoning-effort regression check
    ├── eval_retrieval.py             ← Offline eval: retrieval ranking vs labeled benchmark
    ├── retrieval_benchmark.json      ← Labeled questions + gold for eval_retrieval
    └── requirements.txt              ← Server Python dependencies
```

---

## 3. Prerequisites & Setup

### 3.1 Software Requirements

- **Python 3.11+**
  ```bash
  python --version
  ```
- **PostgreSQL 15+** with the **pgvector** extension
  - macOS: `brew install postgresql@15 pgvector`
  - Ubuntu: `sudo apt-get install postgresql-15 postgresql-15-pgvector`
  - Create a database: `createdb violets`
- **Tesseract OCR** *(optional — only needed for image-only PDFs)*
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`

### 3.2 Install Python Dependencies

From the project root:

```bash
# Pipeline dependencies
pip install -r maryland_rag/requirements.txt

# Server dependencies
pip install -r server/requirements.txt

# Box automation dependencies — only needed to auto-download from Box (Section 6.5)
pip install -r box_ingest/requirements.txt
playwright install chromium

# Download spaCy language model (required by server PII detection)
python -m spacy download en_core_web_lg
```

> The three `requirements.txt` files overlap (all pull the document-extraction
> libraries). `box_ingest/requirements.txt` is only required if you run the Box
> **automation** step (`box_ingest.automate`) — plain `box_ingest.ingest`
> reuses the extraction libraries already installed for the pipeline.

| Library | Used In | Purpose |
|---|---|---|
| `trafilatura` | Pass 1 | Strips nav/boilerplate from HTML, returns clean article text |
| `beautifulsoup4` | Pass 1, 2 | Parses HTML for links, tables, Q&A structure, FAQ accordions |
| `requests` | Pass 1, 2 | HTTP fetching (incl. Range probes for PDFs) |
| `pdfplumber` | Pass 2, Box | Extracts text from digital (non-scanned) PDFs |
| `PyMuPDF` (`fitz`) | Pass 2, Box | Fallback PDF extraction + renders pages as images for OCR |
| `python-docx` | Pass 2, Box | Reads `.docx` Word files, walks heading hierarchy |
| `openpyxl` | Pass 2 | Reads `.xlsx` spreadsheets |
| `xlrd` | Pass 2 | Reads legacy `.xls` spreadsheets |
| `pytesseract` | Pass 2, Box | OCR on scanned PDF page images |
| `Pillow` | Pass 2, Box | Image processing (used with pytesseract) |
| `openai` | Pass 3 | Generates embeddings |
| `psycopg[binary]` | Pass 3, Server | PostgreSQL adapter for vector storage |
| `psycopg-pool` | Server | Connection pool used by the FastAPI lifespan |
| `pgvector` | Pass 3, Server | pgvector extension support for psycopg |
| `fastapi` | Server | Web framework |
| `uvicorn[standard]` | Server | ASGI server to run FastAPI |
| `pydantic` | Server | Request/response validation (`ChatRequest`, structured guardrail outputs) |
| `langchain` / `langchain-openai` | Server | RAG chain orchestration + ChatOpenAI + OpenAI embeddings |
| `presidio-analyzer` | Server | PII detection (SSN, credit card, email, phone, passport, driver's license, IP) |
| `spacy` | Server | NLP backend for Presidio (requires `en_core_web_lg`) |
| `python-dotenv` | Pipeline, Server, Box | Loads `.env` at startup |
| `playwright` | Box automation | Headless Chromium to scrape the Box Hub folder list |

> The non-English chunk filter (`pass2/langfilter.py`) is pure-Python (Unicode
> codepoint ratio) — it needs no `langdetect`/`langid` dependency.

### 3.3 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/violets
VIOLETS_API_KEY=<a long random string — required by /chat and /reset>

# Only needed for the Box auto-download step (box_ingest.automate — Section 6.5):
BOX_CLIENT_ID=<Box developer app client id>
BOX_CLIENT_SECRET=<Box developer app client secret>
```

The server validates `OPENAI_API_KEY`, `DATABASE_URL`, and `VIOLETS_API_KEY` at import time; missing values raise immediately instead of failing later inside a request. `BOX_CLIENT_ID` / `BOX_CLIENT_SECRET` are read only when you run `box_ingest.automate`; the rest of the pipeline and the server never touch them.

Optional server overrides (`LLM_MODEL`, `RETRIEVER_K`, `LOG_LEVEL`, logging toggles, etc.) are listed in [Section 14](#14-configuration-reference).

Optional overrides are listed in [Section 14](#14-configuration-reference). See [Section 9](#9-operating-the-server-auth-cors-rate-limit) for how `VIOLETS_API_KEY` is enforced. See [Section 15](#15-security-warning) for key handling.

---

## 4. Pass 1 — Crawling & Classification

**Goal:** Crawl an allowlisted slice of `elections.maryland.gov` plus a hand-picked set of Montgomery County (`mcg.montgomerycountymd.gov`) pages, extract content, classify each page.

**Run it:**
```bash
python -m maryland_rag pass1
# Start fresh (ignore saved progress):
python -m maryland_rag pass1 --no-resume
```

**Output:** `data/manifest.db` — one row per discovered URL.

---

### 4.1 How the Crawler Works (`pass1/crawler.py`)

BFS from the seeds in `pass1/config.SEED_URLS`, up to 6 levels deep, scoped by the allowlist in `pass1/exclusions.py`. The seeds are derived from the allowlist's `ALLOWED_EXACT_URLS`, so a page is added to the crawl by adding it there.

**Seed set (current):**
- State BoE — prefix-crawled: `/voting/`, `/voter_registration/`, plus `/press_room/documents/2026/`
- State BoE — exact pages: `election_security.html`, `press_room/index.html`, `rumor_control.html`, `elections/2026/index.html`
- Montgomery County — exact pages only (child links discovered are not followed unless they re-match the allowlist): drop boxes, election judge pages, vote-by-mail, FAQs, early voting centers, accessibility

**Crawl sequence per page:**

```
1. Gate 1: Allowlist + exclusion patterns (no network call) → Mark 'excluded' if not in scope
2. Gate 2: Depth > MAX_DEPTH (6)                            → Mark 'skipped'
3. Gate 3: robots.txt (per-domain parser)                   → Mark 'excluded' if disallowed
4. Fetch the page (with transient-failure retry)           ← HTTP request(s)
5. Gate 4: Permanent 403 / 404 / 410?                       → Mark 'failed'
6. Extract content (text, links, metadata, breadcrumbs)
7. Classify the page (rules.py)
8. Persist row to manifest.db
9. Enqueue discovered internal links
```

**Single fetch per page:** No double-fetch (check then extract); everything happens in the one request. The freshly-fetched HTML is written into the Pass 2 disk cache so Pass 2 chunks exactly the bytes Pass 1 hashed (see [Section 5.2](#52-the-http-cache-pass2cachepy)).

**robots.txt (RFC 9309):** At crawl start, one parser is built per domain in `DOMAINS` (only when `RESPECT_ROBOTS_TXT`). The file is fetched with the **crawler's own `requests` client + User-Agent** — not `urllib`'s `RobotFileParser.read()`, whose default `Python-urllib` UA is 403'd by both sites' WAFs (a 403 would make robotparser disallow everything). The response text is then handed to a `urllib.robotparser.RobotFileParser` via `parse()`. Per RFC 9309 §2.3.1:
- **2xx** → parse and honor the rules.
- **4xx** → no restrictions (parser is `None`; Gate 3 is skipped).
- **5xx / unreachable** (after retries) → a synthetic **disallow-all** parser, logged at ERROR — every URL on that domain is excluded until the fetch succeeds.

The robots fetch itself retries up to `MAX_RETRIES` (3) times with `2 ** attempt` backoff on 5xx/network errors.

**Transient-failure retry (`_fetch_with_retry`):** Page fetches retry up to `MAX_RETRIES` (3) times (4 attempts total) with exponential backoff `RATE_LIMIT_SECONDS * 2^(n+1)` = 1.5 s → 3 s → 6 s. A failure is *transient* if the fetch returned `None` (network error/timeout), the HTTP status is ≥ 500, or it's in `TRANSIENT_HTTP_STATUSES` (`408`, `429`). Permanent `403` / `404` / `410` return immediately and hit Gate 4. After retries are exhausted, a still-failing URL is marked `failed` **only if it has never been crawled** — a row already `crawled` from a prior run is preserved (`_mark_failure_preserving_crawled`), so Pass 2 can still chunk the cached copy. This same helper catches unexpected processing exceptions in the loop.

**Resumability:** Every discovered URL is written to `manifest.db` as `pending` before fetching. On restart, the crawler seeds the queue from `pending` rows — no progress lost.

**Rate limiting:** `RATE_LIMIT_SECONDS = 0.75` between requests. A sliding-window `RateMonitor` logs a warning if request rate exceeds `REQUESTS_PER_MINUTE_WARN` (80) in the last 60 s.

---

### 4.2 The Allowlist + Exclusions (`pass1/exclusions.py`)

`exclusions.py` is the single source of truth for both layers — referenced by the crawler and `db_cleanup`.

**Layer 1 — Allowlist** (`should_exclude` returns "Not in allowlist" otherwise):
- `ALLOWED_URL_PREFIXES` — full subtrees in scope (e.g., `…/voting/`)
- `ALLOWED_EXACT_URLS` — specific pages added one-by-one (MoCo pages, top-level State BoE pages)

**Layer 2 — Exclusion patterns** (regex on path; applied within allowed scope):

| Category | Example pattern | Why |
|---|---|---|
| Past election year folders (≠ 2026) | `/elections/(?!2026)\d{4}/` | Historical |
| Special election archives | `/elections/\d{4}_special/` | Historical |
| Presidential / Baltimore archives | `/elections/presidential`, `/elections/baltimore/` | Historical |
| Prior press releases | `/press_room/prior_releases` | Out of cycle |
| Petitions / election data / campaign finance | `/petitions/`, `/election_data/`, `/campaign_finance/` | Out of scope |
| Past audit plans | `/voting_system/ballot_audit_plan_.*\.html` | Historical |

Non-content file extensions (`.jpg`, `.css`, `.js`, `.json`, fonts, archives, media) and non-HTTP schemes (`mailto:`, `tel:`, `javascript:`) are also skipped, as are domains in `SKIP_DOMAINS` (Facebook, Twitter, YouTube, etc.).

`EXCLUDED_HTTP_STATUSES` still lists `{404, 410, 403, 500, 502, 503}`, but in practice the 5xx codes are intercepted earlier by the transient-retry path (Section 4.1); only permanent `403` / `404` / `410` reach Gate 4 and flip a never-crawled URL to `failed`.

**Non-English translated documents** are excluded here too: filenames matching a language suffix (`amharic`, `korean`, `vietnamese`, `spanish`, `french`, `russian`, `tagalog`, `urdu`, `farsi`, `portuguese`, `haitian-creole`, `chinese`, …) on a `.pdf`/`.docx`/`.xls` are dropped with reason `"non-English translated document — English original is indexed"`. A second, content-based net runs in Pass 2 ([Section 5.5](#55-non-english-filter-pass2langfilterpy)).

---

### 4.3 Content Extraction (`pass1/extractor.py`)

For each **HTML page**:
1. Fetch with `requests`
2. Extract clean text with `trafilatura` (strips nav, footers, sidebars)
3. Fall back to BeautifulSoup if trafilatura returns fewer than `TRAFILATURA_MIN_WORDS` (50)
4. Extract outbound links with anchor text + up to 200 chars of surrounding context
5. Extract breadcrumbs (`Home > Voter Registration > Deadlines`) — tries standard nav patterns, falls back to URL-path segments
6. Compute SHA256 `content_hash` of the extracted text (used for dedup and change detection)

For **documents (PDF, DOCX, XLS)**:
- HEAD request only — get file size (`Content-Length`) without downloading the body.
- `needs_ocr` is **no longer probed at crawl time** and is always stored as `False`. The old 4 KB `Range` byte-probe was removed because it mis-flagged compressed-but-digital PDFs; the OCR decision now happens in Pass 2 after a real extraction attempt ([Section 13, PDF extraction](#pdf-extraction-pass2strategiespdfpy)). The `PDF_PROBE_BYTES` constant is retained in `config.py` but is now dead.

Full document extraction is deferred to Pass 2 (discovery vs. extraction stay separate).

---

### 4.4 Classification (`pass1/classifier.py` → `pass1/rules.py`)

`pass1/classifier.py` is a thin wrapper: documents (`pdf`/`docx`/`xls`/`xlsx`/`csv`/`doc`) get `page_classification='document'`, `chunking_strategy='document_extraction'` immediately. Everything else goes through `rules.classify_html`, which is shared with `scripts/reclassify.py` so crawl-time and post-hoc classification stay in lockstep.

**Why rule-based, not ML?** The elections websites have consistent, predictable structure. Rules are transparent and auditable, and easy to fix without retraining.

**Decision tree (checked in order, first match wins):**

```
URL contains 'cdn-cgi'                       → junk            (strategy: skip)

FAQ signals matched (in url/title/text body OR known FAQ path)
  → faq                                      strategy: qa_pairs

Press / news signals (press_room, press_release, news-release,
                     announcement, rumor_control, dis-misinformation)
  → press_release                            strategy: simple_split (≥150w) | ingest_as_single

MoCo location page (early voting centers / drop boxes)
  → location_list                            strategy: ingest_as_single

Table/data path signals (results, archives, stats, recount)
  → table_data                               strategy: table_rows

Form path signal OR voterservices subdomain
  → form                                     strategy: simple_split (≥150w) | ingest_as_single

word_count ≥ 500
  → prose                                    strategy: semantic_with_overlap

Known short-static path signal
  → short_static                             strategy: ingest_as_single

Raw HTML available + structural pattern matches (≥3 dt/dl, details, accordion
   classes, or question-style headings; OR ≥5 table rows)
  → faq OR table_data                        (medium confidence)

word_count ≥ 150  (no other signal)
  → nav_hub                                  strategy: ingest_as_single

Fallback
  → short_static                             strategy: ingest_as_single

If word_count == 0 (no useful extraction)
  → strategy forced to 'skip'  regardless of class (except 'junk')
```

Each classification carries a confidence level: `high` (URL/structural match), `medium` (structural heuristic or word-count fallback), `low` (no signal).

`scripts/reclassify.py` re-runs this engine against stored metadata (url, title, snippet, word_count) so the rules can be refined without re-crawling. Structural-pattern fallbacks don't fire in reclassify mode because `raw_html` isn't persisted.

---

## 5. Pass 2 — Chunking

**Goal:** Re-fetch each crawled page from cache, extract fully, and split into chunks sized for embedding.

**Run it:**
```bash
python -m maryland_rag pass2
# Only re-process pages whose content changed since last snapshot:
python -m maryland_rag pass2 --changed
# Custom output path (default is data/chunks.jsonl):
python -m maryland_rag pass2 --output data/chunks.jsonl
```

**Output:** `data/chunks.jsonl` — one JSON record per chunk.

---

### 5.1 Chunk Sizing Rationale

Different page types warrant different chunking approaches:
- A FAQ answer is already a natural retrieval unit — each Q&A as its own chunk means a query retrieves the right answer, not a mixed page of many answers.
- A long policy document needs splitting, but cutting at fixed word counts splits sentences mid-stream. Sentence-aware splitting with overlap preserves coherence.
- A table is best as one-row-per-chunk — each row is a self-contained fact that can be retrieved independently.

---

### 5.2 The HTTP Cache (`pass2/cache.py`)

Pass 2 needs full content (Pass 1 only stored a 500-char snippet for HTML). All fetches go through a disk cache at `data/cache/`, keyed by `SHA256(url)`, stored as `.html` (text) or `.bin` (bytes). Writes are atomic (temp file + `os.replace`). Cache misses sleep `RATE_LIMIT_SECONDS` before fetching so re-runs after a crash don't hammer the server. Disk (not in-memory) because Pass 2 runs can be long and interrupted — a persistent cache survives restarts.

**Freshness (there is no time-based TTL):**
- **HTML** — Pass 1 writes every page it fetches straight into this cache (`put_html`), so a Pass 2 HTML cache hit returns exactly the bytes Pass 1 hashed. On a genuine miss, `get_html` fetches live (rate-limited).
- **Binaries (`.bin`)** — each blob has a `.bin.meta` JSON sidecar holding its `ETag` / `Last-Modified`. On a cache hit, the URL is revalidated **at most once per process run** (tracked in an in-memory set) with a conditional GET (`If-None-Match` / `If-Modified-Since`): **304** keeps the cached blob, **200** replaces blob + sidecar, a network error falls back to the cached copy. Blobs with no validators are trusted; legacy blobs with no sidecar get one plain refresh GET.

---

### 5.3 Deduplication & chunk_id (`pass2/chunker.py`, `pass2/metadata.py`)

Pages with identical `content_hash` (same content under different URLs) are extracted once. Resulting chunks carry `source_urls` (plural array) listing every URL pointing at that content. This avoids inserting duplicate vectors while preserving full provenance.

The `chunk_id` is **derived from the chunk's content**, not from its URL set:
`chunk_id = SHA256( normalize(text) + "\x00" + section_key + "\x00" + chunk_index )[:32]`.
It was deliberately changed away from the old sorted-URL-set id: when a duplicate URL appeared or disappeared between runs the id would shift and Pass 3's upsert would orphan the old vector. A content-derived id is stable across re-ingests regardless of which URLs currently point at the content. (`box_ingest` uses the same `_content_chunk_id` helper.)

---

### 5.4 Chunk Metadata

Every record in `chunks.jsonl`:

| Field | Description |
|---|---|
| `chunk_id` | SHA256-derived id, stable across re-runs |
| `source_url` | URL this chunk came from |
| `source_urls` | (only for deduplicated content) all URLs pointing at this content |
| `title` | Page title |
| `section_hierarchy` | Breadcrumb array (e.g., `["Home", "Voter Registration"]`) |
| `page_classification` | Page type (`faq`, `prose`, `table_data`, `document`, etc.) |
| `chunking_strategy` | How it was split |
| `chunk_index` | 0-based position within the page |
| `chunk_total` | Total chunks from this page |
| `word_count` | Word count of this chunk |
| `text` | The chunk text |
| `date_extracted` | ISO timestamp |

Strategy-specific fields are merged in via an allowlist (`question`, `answer`, `heading_chain`, `table_index`, `row_index`) so a strategy can never overwrite canonical fields like `chunk_id` or `text`.

The full chunk dict (minus the columns `chunk_id`/`text`/`source_url`/`title`) is stored as JSONB `metadata` in PostgreSQL, so retrieved chunks can be cited with source, title, and page location.

---

### 5.5 Non-English Filter (`pass2/langfilter.py`)

Bilingual PDFs and mixed-script pages can produce "extraction salad" (English interleaved with another script). A per-chunk content net drops those: `non_latin_ratio(text)` is the fraction of alphabetic characters whose codepoint is above `0x024F` (end of Latin Extended-B), and a chunk is dropped when that ratio exceeds `MAX_NON_LATIN_RATIO` (`0.10`). Accented European letters (é, ñ, ő) count as Latin and pass; Vietnamese diacritics and CJK/Cyrillic/Arabic scripts count as non-Latin. It runs in `chunker.run_pass2` after each page is chunked. This is the content-side companion to the URL-based exclusion of translated documents in Pass 1 ([Section 4.2](#42-the-allowlist--exclusions-pass1exclusionspy)); the threshold has wide margin (measured English chunks sit below 0.02).

### 5.6 Chunk Size Caps

Every strategy routes its output through `enforce_chunk_caps` (`pass2/strategies/semantic.py`): no chunk may exceed `MAX_CHUNK_WORDS` (500) or `MAX_CHUNK_CHARS` (20000, the hard embedding-safety limit). Oversized chunks are re-split semantically, and pathological single "sentences" (OCR / CID-stream garbage with no punctuation) are hard word/char-split. These are per-chunk **size** caps — there is no cap on the number of chunks per HTML page or PDF. The one per-*document* count cap is spreadsheets: a workbook producing more than `MAX_XLS_ROW_CHUNKS` (200) rows is treated as bulk tabular data and skipped entirely (see [XLS extraction](#xls--xlsx-extraction-pass2strategiesxls_strategypy)).

---

## 6. Box Ingest — Curated Document Pipeline

**Goal:** Ingest a hand-curated set of Box-hosted documents (election worker manuals, monthly admin reports, etc.) into the same pgvector store as the web crawl, using the same chunk schema.

Why a separate pipeline: Box files aren't crawlable — the State Board uploads them to a private Box folder and shares them by URL. The workflow is:

1. Download or mirror the file into `needtochunk/<year-folder>/<file>`.
2. Add a line to `needtochunk/url_manifest.json` mapping the relative path → Box share URL (so retrieved chunks can cite a working link).
3. Run `python -m box_ingest.ingest` (or include it via `python -m maryland_rag all`).

`needtochunk/url_manifest.json` format:

```json
{
  "_comment": "Maps relative paths (from needtochunk/) to Box.com URLs.",
  "Montgomery County Election Day Plans.txt": "https://mdsbe.app.box.com/s/.../file/<id>",
  "2026-02/State Administrator's Report- February 19, 2026.pdf": "https://..."
}
```

Files whose mapping is empty or missing are **skipped with a warning** — they won't be embedded. `url_manifest.json`, `review_files.txt`, and `.DS_Store` are always skipped (`SKIP_NAMES`).

**Run it (Step 2 — extract & chunk):**
```bash
python -m box_ingest.ingest                       # write data/box_chunks.jsonl
python -m box_ingest.ingest --dry-run             # list planned actions, write nothing
python -m box_ingest.ingest --output path/to.jsonl
python -m box_ingest.ingest --workers 8           # parallel extraction (default: CPU count)
```

Extraction runs in a `ProcessPoolExecutor`. A skip-if-unchanged cache in `data/box_ingest.state.json` (size + mtime fingerprint per file) means unchanged files aren't re-extracted; `box_chunks.jsonl` is rewritten each run as the union of all cached chunks. Deleted files are dropped from the state and every zero-chunk file is logged in a warning summary.

**Extraction (all via the shared `maryland_rag.pass2` extractors):**
- `.pdf` → pdfplumber → pymupdf → OCR (tesseract @ **300 dpi** for scanned PDFs — the same path Pass 2 uses)
- `.docx` / `.doc` → python-docx walks heading hierarchy; each section keeps its `heading_chain`
- `.xlsx` / `.xlsm` → one chunk per spreadsheet row (bypasses the word-count routing below)
- `.txt` → plain read
- Other extensions → warned and skipped

**Chunking (PDF / DOCX / TXT):**
- DOCX with headings → each section chunked independently
- ≤ `SHORT_DOC_WORDS` (150) → `ingest_as_single`
- > 150 words → `semantic_chunk` (with a `FALLBACK_CHUNK_WORDS` = 400 paragraph-boundary fallback)
- Section hierarchy = folder chain (relative to `needtochunk/`) + DOCX heading chain

**Stability:** the `chunk_id` is content-derived (`_content_chunk_id`: normalized text + section hierarchy + chunk index — **not** the source URL; see [Section 5.3](#53-deduplication--chunk_id-pass2chunkerpy-pass2metadatapy)). Re-running on unchanged files produces the same IDs, so Pass 3's upsert is a no-op for unchanged content.

Each record in `data/box_chunks.jsonl` uses the same shape as `chunks.jsonl` (`page_classification='document'`, `chunking_strategy='box_ingest'`), so Pass 3 ingests it identically.

---

### 6.5 Box Auto-Download (`box_ingest.automate`)

Step 2 above chunks whatever is already sitting in `needtochunk/`. **Step 1** (`box_ingest.automate`) is what fills `needtochunk/` from the State Board's public Box Hub in the first place. It is a **manual/separate** command — `python -m maryland_rag all` runs Step 2 (`ingest`) but never Step 1.

```bash
python -m box_ingest.automate                     # crawl Box hub, download, update manifest
python -m box_ingest.automate --dry-run           # preview only, no downloads or writes
```

Flow:
1. **Auth** — Box OAuth 2.0 (authorization-code grant) using `BOX_CLIENT_ID` / `BOX_CLIENT_SECRET`. First run opens a browser and captures the redirect at `http://localhost:8080`; tokens are cached in `.box_token` (mode `0600`) and refreshed silently thereafter.
2. **Crawl** (`crawler.py`) — Playwright (headless Chromium) scrapes the public Hub page for folder IDs, then the Box REST API v2.0 lists each `2026-*` folder recursively (carrying the Hub's shared link on every call).
3. **Filter** (`filter.py`) — each filename is classified `include` / `exclude` / `review` by keyword lists (`INCLUDE_TERMS` / `EXCLUDE_TERMS`; anything matching neither → `review`).
4. **Download** — `include` files are downloaded into `needtochunk/<year-folder>/<name>` (already-present files skipped).
5. **Manifest** (`manifest.py`) — new `include` files are merged into `url_manifest.json` (existing/populated entries are never overwritten).
6. **Review log** — `review` files are written to `needtochunk/review_files.txt` with their Box URLs so an operator can add a keyword to `filter.py` or hand-add the file to the manifest.

See [box_ingest/README.md](box_ingest/README.md) for the full module breakdown.

---

## 7. Pass 3 — Embedding & pgvector Upload

**Goal:** Embed every chunk with OpenAI and upsert into PostgreSQL with pgvector.

**Run it:**
```bash
python -m maryland_rag pass3
# Resume an interrupted upload (skips chunk_ids already present):
python -m maryland_rag pass3 --resume
# Use a different input file (e.g. the Box pipeline output):
python -m maryland_rag pass3 --chunks data/box_chunks.jsonl
```

---

### 7.1 How It Works (`pass3/embed.py`)

1. Read chunks from the JSONL file.
2. Create the `chunks` table if it doesn't exist (`CREATE EXTENSION IF NOT EXISTS vector` first).
3. If `--resume`: query existing `chunk_id`s and skip them.
4. Embed in batches of `EMBED_BATCH_SIZE` (100) → `text-embedding-3-large` → 3,072-dim vectors. Up to 3 attempts with **linear backoff + jitter** (`RETRY_DELAY * attempt + random(0, RETRY_DELAY)`) on transient errors (incl. 408/429). On a deterministic 4xx the batch is recursively **bisected** to isolate the offending input; a chunk that still fails deterministically raises `RuntimeError` and aborts the run (non-zero exit) rather than silently dropping content.
5. Upsert into PostgreSQL with `ON CONFLICT (chunk_id) DO UPDATE`. Each row is wrapped in a savepoint so a single failure doesn't roll back the rest of the batch.

---

### 7.2 pgvector Table Schema

```sql
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    embedding  vector(3072),
    text       TEXT,
    source_url TEXT,
    title      TEXT,
    metadata   JSONB DEFAULT '{}'::jsonb
);
```

| Setting | Value | Reason |
|---|---|---|
| Dimension | 3,072 | Matches `text-embedding-3-large` |
| Metric | Cosine | Standard for normalized text embeddings (`embedding <=> %s::vector`) |

> ⚠️ The table is created with no ANN index. For the current corpus size this is fine and queries do a sequential scan. If the corpus grows enough that retrieval latency matters, add an `hnsw` index manually. pgvector can't index a plain `vector` column above 2,000 dimensions, so with 3,072-dim embeddings use a `halfvec` expression index (`CREATE INDEX ON chunks USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);`) and cast the query the same way. At ~2.5k chunks the exact scan takes ~23 ms locally, small next to the ~200 ms query-embedding call, so no index is needed yet.

---

### 7.3 Resume Support

`--resume` queries existing `chunk_id`s and skips them. Without `--resume`, the upsert path safely updates existing rows (matters when `python -m maryland_rag all` re-embeds changed pages without `--resume`).

---

## 8. Server — FastAPI Chatbot

**Goal:** Serve a conversational RAG chatbot over HTTP, backed by the pgvector table populated by Pass 3 (web + Box). Includes guardrail middleware for PII protection, query classification with 8 categories, and partisan-response prevention with retry.

**Run it:**
```bash
python -m server.main            # host/port via HOST/PORT env (default 0.0.0.0:8000)
```

> The server keeps rate-limit and session state in process memory, so it **must
> run as a single worker**. `python -m server.main` pins `workers=1` for you
> (overriding `WEB_CONCURRENCY`). If you invoke uvicorn directly
> (`uvicorn server.main:app --host 0.0.0.0 --port 8000`), do **not** pass
> `--workers >1` or set `WEB_CONCURRENCY`.

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | `X-API-Key` required | Send a message, get a response (+ source list) |
| `POST` | `/reset` | `X-API-Key` required | Clear conversation history for a user |
| `GET` | `/health` | Public | Health check — pings the DB; returns `{status, database, model}` |

`/health` runs `SELECT 1` against the pool: healthy → **200** `{"status":"ok","database":"ok","model":…}`; DB/pool down → **503** `{"status":"degraded","database":"unreachable",…}`. Before the pool/chain finish initializing, `/chat` and `/reset` return **503** `"Service starting, retry shortly"`. There is **no** `/metrics` HTTP endpoint — metrics surface only in the periodic heartbeat log line.

### Request / Response shape

```jsonc
// POST /chat
{ "user_id": "abc-123", "query": "When is the voter registration deadline?" }

// 200 OK
{
  "response": "...with inline [Source N] markers replaced by markdown links...",
  "sources": [
    {
      "source_number": 1,
      "source_url": "https://elections.maryland.gov/voter_registration/...",
      "title": "Voter Registration",
      "score": 0.8472
    }
  ]
}
```

`user_id` must match `^[a-zA-Z0-9_-]+$` and be 1–128 chars. `query` is 1–2000 chars. Both are enforced by Pydantic; violations return 422.

### How It Works

```
User query
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  AUTH: X-API-Key header (hmac.compare_digest)         │
│  → 401 if missing/wrong                                │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  RATE LIMIT: 20/min per user_id + 225/min global      │
│  → 429 if exceeded                                     │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  Guard 1: Input PII detection (Presidio, 0 tokens)    │
│  → Block with canned PII message if found             │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  Guard 2: Query classification (gpt-5-nano, ~70tok)   │
│  Categories: normal, conversational, concerns,        │
│              polling_location, voter_lookup,          │
│              voter_update, candidates, partisan,      │
│              out_of_scope                             │
│  → polling_location / voter_lookup / voter_update /   │
│    candidates  → return hardcoded URL, exit           │
│  → partisan / out_of_scope   → return fallback, exit  │
│  → normal / conversational / concerns → continue      │
│  (Survey system tag "__User concerns:__" short-       │
│   circuits LLM, jumps straight to 'concerns')         │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  Session created / fetched (in-memory, per user_id)   │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  RAG CHAIN                                            │
│   conversational  → answer from chat history only,    │
│                     skip retrieval                    │
│   normal / concerns →                                 │
│      1. Rephrase follow-ups into a standalone Q       │
│         (skipped when chat_history is empty)          │
│      2. Embed Q, query pgvector top-k                 │
│      3. concerns → CONCERNS_PROMPT (Rumor Control)    │
│         normal   → QA_PROMPT                          │
│      4. Replace [Source N] markers with markdown      │
│         links built from the retrieved source list    │
└───────────────────────┬───────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  Guard 3: Partisan-response check (gpt-5-nano)        │
│  If flagged: re-invoke the chain with a stricter      │
│  nonpartisan retry prompt appended to the user msg.   │
│  Up to MAX_PARTISAN_RETRIES (2) retries; checks each  │
│  retry. FAILS CLOSED — if still partisan after        │
│  retries (or on error) it discards the answer and     │
│  returns a canned refusal, never the unvetted text.   │
└───────────────────────┬───────────────────────────────┘
                        ▼
   log_request() + store.add_exchange() + return ChatResponse
```

> ℹ️ There is **no output-side PII scrub** in the current code — only input PII is blocked. The partisan check is the only post-generation guard.
>
> ⚠️ **The three guardrails fail *closed*, not open.** `detect_pii` (on analyzer error), `classify_query` (on classifier error → canned error reply), and `check_partisan_response` (still-partisan after retries → `partisan_persist`; on exception → `partisan`) all suppress the response rather than let an unvetted answer through. This is the opposite of a fail-open design and is intentional for an elections chatbot.

### Server Modules

**`main.py`** — FastAPI app with async lifespan startup: `setup_logging()`, an **`AsyncConnectionPool`** (`min_size=4, max_size=25, timeout=10`, opened with `wait=True`; if the DB is unreachable at startup the server refuses to start), `SessionStore`, and `build_chain(pool)`. Hosts `_RateLimiter` (which also evicts stale per-user windows each call), the `X-API-Key` dependency, CORS middleware, and a background `_periodic_maintenance()` task (every `HEARTBEAT_INTERVAL` = 300 s) that runs `store.cleanup_expired()` **and** emits a `HEARTBEAT` log line (requests / errors / blocked / auth failures / cost + pool gauge). Failed-auth warnings are throttled to one per 60 s (with a suppressed count) so unauthenticated floods can't rotate real entries out of the log. The RAG call is wrapped in `asyncio.wait_for(..., RAG_CHAIN_TIMEOUT=60)`; timeout or chain error → HTTP 502. Each `/chat` gets an 8-char request id (`new_request_id()`) that tags every log line for that request. Pool is closed with a 30 s grace on shutdown; the process pins `workers=1`.

**`rag_chain.py`** — Built with `langchain_core` runnables. `full_pipeline` is a `RunnableLambda` that branches on `query_category`: `conversational` → answer from history, no retrieval; `concerns` → Rumor Control prompt; else standard QA. A custom `PgVectorRetriever(BaseRetriever)` queries PostgreSQL via pgvector directly (`embedding <=> %s::vector`, with `SET LOCAL statement_timeout = '30s'`). Ranking adds `PAST_ELECTION_PENALTY` to the cosine distance of chunks whose `source_url` matches `PAST_ELECTION_URL_PATTERNS`, so stale primary-election lists lose close calls to current ones. Four prompts:
  - `_CONTEXTUALIZE_PROMPT` — rephrases follow-ups into standalone questions
  - `_QA_PROMPT` — main answer prompt with `[Source N]` citation contract
  - `_CONVERSATIONAL_PROMPT` — answers from chat history only, no retrieval
  - `_CONCERNS_PROMPT` — Rumor Control prompt; always starts the response by linking https://elections.maryland.gov/press_room/rumor_control.html

After answer generation, `_replace_source_refs` substitutes inline `[Source N]` markers with markdown links built from the retrieved source list (multi-source chunks render their extra URLs as numbered links).

Retrieval is **fully async**: `_aget_relevant_documents` uses `aembed_query` + an async pool connection. The sync `_get_relevant_documents` raises `NotImplementedError` (async-only), so retrieval no longer runs in the anyio threadpool.

**`middleware.py`** — All guardrail logic. Shared `QueryContext` dataclass travels through the request. All guardrail LLM calls are wrapped in `asyncio.wait_for` (`GUARDRAIL_LLM_TIMEOUT` = 30 s; partisan retry `PARTISAN_RETRY_TIMEOUT` = 60 s).

| Function | Purpose | Failure mode |
|---|---|---|
| `detect_pii(query, ctx)` | Presidio scan for `US_SSN`, `CREDIT_CARD`, `EMAIL_ADDRESS`, `IP_ADDRESS`, `PHONE_NUMBER`, `US_PASSPORT`, `US_DRIVER_LICENSE` at score ≥ 0.5 | **Fail closed** — blocks (canned PII fallback) on any analyzer error |
| `classify_query(query, ctx, history)` | LLM (`LLM_MODEL`, structured `ClassificationResult`, `reasoning_effort="medium"`) → 9 categories. Sees the latest message plus the last 2 exchanges (each truncated to 600 chars) so follow-ups like "where is it?" resolve against the conversation. Short-circuits `__User concerns:__` to skip the LLM | **Fail closed** — returns canned `error` reply on classifier failure |
| `check_partisan_response(...)` | Structured `PartisanCheckResult` (`reasoning_effort="minimal"`); on `is_partisan=True`, re-invokes chain with stricter retry prompt up to `MAX_PARTISAN_RETRIES` (2) | **Fail closed** — still-partisan → `partisan_persist` refusal; exception → `partisan` refusal |

Categories `normal` / `conversational` / `concerns` reach the RAG chain. The four "hardcoded URL" categories (`polling_location`, `voter_lookup`, `voter_update`, `candidates`) return a static URL from `FALLBACK_RESPONSES` without ever calling the LLM. `partisan` returns a refusal; `out_of_scope` returns a redirect toward what the assistant can help with (registration, polling locations, ballot procedures, candidates) rather than a flat decline. `FALLBACK_RESPONSES` also carries `pii`, `partisan_persist`, and `error` messages.

**`rag_logger.py`** — LangChain `BaseCallbackHandler` that captures LLM prompts/responses (toggled by env vars `LOG_PROMPTS`, `LOG_RESPONSES`, `LOG_QUERIES` — **all default `False`** / production-safe; set to `1`/`true` to opt in for debugging), token usage, estimated cost from a built-in `_COST_TABLE`, and retriever timing. Per-call token/cost and retriever lines log at `DEBUG` (suppressed at default `INFO`); `on_llm_end` also feeds `metrics.METRICS.record_llm(...)`. `log_request(user_id, query, response, elapsed, outcome)` writes one summary line per `/chat` exchange (query text logged only when `LOG_QUERIES`). The lock around the run-tracking dicts is required because callbacks fire from worker threads.

**`logging_setup.py`** — Central logging config (imported by `main.py`). `setup_logging()` (idempotent) installs a `RotatingFileHandler` (`SERVER_LOG_FILE`, default `logs/server.log`, 10 MB × 5 backups) + console handler at `LOG_LEVEL` (default `INFO`), silences noisy third-party loggers (uvicorn.access, httpx, openai, urllib3), and injects a per-request id into every record. `new_request_id()` mints the id and binds it to a `contextvars.ContextVar`.

**`metrics.py`** — Thread-safe in-process counters (`METRICS` singleton). `record_request(outcome)` and `record_llm(tokens, cost)` accumulate both cumulative and rolling totals; `drain_rolling()` snapshots-and-resets the rolling window for the heartbeat line. No HTTP surface, no new dependencies.

**`session.py`** — Thread-safe in-memory per-`user_id` conversation store with TTL expiration (`SESSION_TTL_MINUTES`) and max-turn cap (`MAX_HISTORY_TURNS`). Designed for pilot-scale (tens of concurrent users) — swap to Redis or a database for production scale.

**`eval_guardrails.py`** — Offline eval (not part of the server runtime). Runs the classifier and partisan checker at both `reasoning_effort="minimal"` and `"medium"` over labeled fixtures and reports whether `minimal` disagrees with the expected labels or with `medium`. Makes ~26 real OpenAI calls; exit code = number of `minimal` mislabels. `python -m server.eval_guardrails`.

**`eval_retrieval.py`** — Offline retrieval eval. Runs the 49 questions in `retrieval_benchmark.json` through the production `PgVectorRetriever` and scores hit@1, hit@K, MRR@10 and P@K. Gold chunks are matched by URL substring + text regex (not chunk_id), so labels survive a drop-and-reingest. Questions tagged `candidates` are reported separately because the classifier usually routes them to the canned `CANDIDATES_URL` reply. Rerun after any change to chunking, embedding or ranking. `python -m server.eval_retrieval [-v]`.

---

## 9. Operating the Server (Auth, CORS, Rate Limit)

### 9.1 API Key Authentication

Every `/chat` and `/reset` request must include:

```
X-API-Key: <value of VIOLETS_API_KEY>
```

The key is compared with `hmac.compare_digest` (constant-time). Missing or wrong → **HTTP 401** `{"detail":"Missing or invalid API key"}`. `/health` is intentionally unauthenticated for load-balancer probes.

**Generating a key:**
```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

**Calling /chat with auth:**
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $VIOLETS_API_KEY" \
     -d '{"user_id":"abc","query":"When is the voter registration deadline?"}'
```

### 9.2 CORS

Configured at startup from the `CORS_ORIGINS` env var (comma-separated). Default: `https://umdsurvey.umd.edu`. Allowed methods: `GET`, `POST`. Allowed headers: `Content-Type`, `X-API-Key`.

For local browser testing, override:
```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:5173 uvicorn server.main:app --port 8000
```

### 9.3 Rate Limiting

Two in-memory sliding-window limiters (`_RateLimiter` in `main.py`, 60 s window), both checked **before** PII/classification/RAG so a runaway client cannot drain OpenAI credits:

- **Per-user** — keyed by `user_id`, `RATE_LIMIT_PER_MINUTE = 20`. Checked first; its rejections don't consume global slots.
- **Global** — one shared bucket for all `/chat` + `/reset` traffic, `RATE_LIMIT_GLOBAL_PER_MINUTE = 225`. Backstop for clients rotating `user_id`s past the per-user limit.

Exceeding either → **HTTP 429** `{"detail":"Rate limit exceeded"}`.

At ~6.5k OpenAI tokens per `/chat` (4–5 calls), a saturated 225/min global cap is ~1.5M tokens/min — above OpenAI's Build-tier TPM, within Launch. Check your tier before raising it further.

The limiter is per-process. Behind multiple replicas you would either pin users to a replica or move the counter into Redis.

### 9.4 Stress Testing

`maryland_rag/scripts/stress_test.py` exercises the running server with concurrent RAG queries, same-user races, malformed inputs, mixed guardrail paths, session reset under load, health responsiveness during load, and the exact rate-limit boundaries. It exits non-zero on any failure.

```bash
python -m server.main &                                     # start server first
python -m maryland_rag.scripts.stress_test                  # full suite (real LLM spend)
python -m maryland_rag.scripts.stress_test --limits-only    # rate limits only (free, ~2 min)
```

The rate-limit phase reads both limits from `server.config`, so run it with the same `.env`/environment as the server. It waits out one window, then asserts that exactly `RATE_LIMIT_PER_MINUTE` requests from one user pass and the next 429, that distinct users fill exactly `RATE_LIMIT_GLOBAL_PER_MINUTE` before everything (including `/reset`) 429s, and that both recover after the window. It uses PII-blocked queries (an email address), which count against the limiters but never reach OpenAI; it aborts before the global burst if that stops being true. Counts are exact, so the server must have no other traffic during the run.

---

## 10. Utility Scripts

### 10.1 Database Cleanup (`scripts/db_cleanup.py`)

Removes duplicate URL variants and pure junk from `manifest.db`. Always creates a timestamped backup (`data/manifest.db.bak.<ts>`) before modifying. Deletions, in order:

1. All `http://` rows (redirects to https://, pure duplicates)
2. All `https://www.elections.maryland.gov/` rows (canonical domain omits `www.`)
3. Cloudflare `cdn-cgi` stubs
4. `businessdisclosure-elections.maryland.gov` subdomain (external)
5. All remaining `failed` rows

Then orphaned `links` rows (where either endpoint no longer exists in `pages`) are cleaned up and the database is `VACUUM`ed.

```bash
python -m maryland_rag.scripts.db_cleanup --dry-run   # preview
python -m maryland_rag.scripts.db_cleanup              # apply
```

### 10.2 Re-classification (`scripts/reclassify.py`)

Re-applies the current `pass1/rules.py` classifier to all crawled HTML rows using stored metadata (no re-fetching). Use this whenever you change `rules.py`. Crawl-time vs. reclassify-time differ only in that crawl-time has `raw_html` for the structural-pattern fallback; reclassify does not.

```bash
python -m maryland_rag.scripts.reclassify --dry-run   # preview transitions
python -m maryland_rag.scripts.reclassify              # apply
```

### 10.3 Verify Chunks (`scripts/verify_chunks.py`)

Post-ingest verification gate. Exits non-zero on any failure so it can gate a pipeline run. Three checks:
1. **Coverage** — every `crawl_status='crawled'` page (except `chunking_strategy='skip'`) must have at least one chunk in `chunks.jsonl` (counting both `source_url` and dedup `source_urls`).
2. **Junk** — scans `chunks.jsonl` + `box_chunks.jsonl` for empty/whitespace text, literal `None:` prefixes, chunks over the char cap (`MAX_CHUNK_CHARS`), over the word cap (warn-only), and non-English text (`langfilter`).
3. **DB parity** — if `DATABASE_URL` is set and Postgres is reachable, compares the JSONL `chunk_id` set against the pgvector `chunks` table both ways (missing / orphaned). **Skipped** (not failed) when the DB is unavailable.

```bash
python -m maryland_rag.scripts.verify_chunks
python -m maryland_rag.scripts.verify_chunks --chunks data/chunks.jsonl --box-chunks data/box_chunks.jsonl
```

### 10.4 Audit (`scripts/audit.py`)

Backs the `python -m maryland_rag audit` command — see [Section 11](#audit-the-database). Prints the classification/strategy breakdown, exclusion reasons, depth distribution, failed pages, duplicate content, top documents by inbound links, an exclusion-leak check, and the largest `semantic_with_overlap` candidates.

### 10.5 Stress Test (`scripts/stress_test.py`)

See [Section 9.4](#94-stress-testing).

---

## 11. Running the Full Pipeline

### First-time setup

```bash
# 1. Install dependencies
pip install -r maryland_rag/requirements.txt
pip install -r server/requirements.txt
python -m spacy download en_core_web_lg

# 2. Create .env with all three required keys (see Section 3.3)

# 3. Crawl
python -m maryland_rag pass1

# 4. Clean up duplicates (recommended after a fresh crawl)
python -m maryland_rag.scripts.db_cleanup

# 5. Chunk the web pages
python -m maryland_rag pass2

# 6. (Optional) auto-download curated Box documents into needtochunk/
#    Requires BOX_CLIENT_ID / BOX_CLIENT_SECRET in .env + playwright (Section 6.5)
python -m box_ingest.automate

# 7. Chunk Box documents (skips any file missing a URL in url_manifest.json)
python -m box_ingest.ingest

# 8. Verify chunk coverage / junk / DB parity before embedding
python -m maryland_rag.scripts.verify_chunks

# 9. Embed and upload — web chunks
python -m maryland_rag pass3 --chunks data/chunks.jsonl

# 10. Embed and upload — Box chunks (same table, different input)
python -m maryland_rag pass3 --chunks data/box_chunks.jsonl

# 11. Verify vectors are in PostgreSQL:
#     psql $DATABASE_URL -c "SELECT count(*) FROM chunks;"

# 12. Start the server
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### One-shot: `python -m maryland_rag all`

`all` runs the full pipeline end-to-end and is safe to re-run. The operating model is **drop-and-reingest**: the operator rebuilds the pgvector table from `chunks.jsonl` each run, and Pass 2 truncates that file — so `all` must always emit the full corpus.

```bash
python -m maryland_rag all
```

Internally it:
1. Runs Pass 1 with `resume=False` (full re-discovery).
2. Runs Pass 2 with `only_changed=False` — **always the full corpus.** An incremental pass here would silently drop every unchanged page from the rebuilt vector store (`manifest.db` persists, so change detection would find nothing changed). Incremental chunking stays available via the standalone `pass2 --changed`.
3. **Coverage gate** (`_report_chunk_coverage`): prints per-page chunk coverage and **aborts with a non-zero exit if more than 20% of expected pages produced zero chunks**, so pgvector is never rebuilt from a badly incomplete `chunks.jsonl`.
4. Snapshots `content_hash → previous_content_hash` **after** a successful full chunk pass (so `previous_content_hash` means "reflected in `chunks.jsonl`", the baseline `pass2 --changed` diffs against).
5. Runs Box ingest (`box_ingest.ingest`) — Step 2 only; it does **not** auto-download from Box (run `box_ingest.automate` separately for that, Section 6.5).
6. Runs Pass 3 on web chunks with `resume=False`.
7. Runs Pass 3 on Box chunks (only if any were produced).

> `all` accepts `--output` (default `data/chunks.jsonl`) to relocate the web-chunk file.

### Audit the database

```bash
python -m maryland_rag audit
```

Prints: classification/strategy breakdown, exclusion reasons, depth distribution, failed pages, duplicate content, top documents by inbound links, an exclusion-leak check, and the largest `semantic_with_overlap` candidates.

### Re-running after website updates

```bash
python -m maryland_rag pass1                              # resumes, detects content-hash changes
python -m maryland_rag pass2 --changed                    # only changed pages
python -m maryland_rag pass3 --resume                     # skip already-upserted chunk_ids
```

---

## 12. The Database (manifest.db)

Inspect directly with the SQLite CLI:

```bash
sqlite3 data/manifest.db
.tables
.schema pages
SELECT crawl_status, COUNT(*) FROM pages GROUP BY crawl_status;
SELECT page_classification, COUNT(*) FROM pages
  WHERE crawl_status='crawled' GROUP BY page_classification;
.quit
```

For a richer breakdown use `python -m maryland_rag audit` — it always reflects the current state.

### Tables

**`pages`** — one row per discovered URL:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `url` | TEXT | Full URL (unique) |
| `parent_url` | TEXT | Which page linked here |
| `title` | TEXT | `<title>` tag content |
| `section_hierarchy` | TEXT | JSON breadcrumb array |
| `content_type` | TEXT | `html`, `pdf`, `docx`, `xls`, `csv` |
| `page_classification` | TEXT | Classification label |
| `chunking_strategy` | TEXT | Which Pass 2 strategy to use |
| `classification_confidence` | TEXT | `high`, `medium`, or `low` |
| `word_count` | INTEGER | Word count of extracted text |
| `depth` | INTEGER | Crawl depth from seed URL |
| `crawl_status` | TEXT | `pending`, `crawled`, `failed`, `skipped`, `excluded` |
| `exclusion_reason` | TEXT | Why excluded |
| `http_status` | INTEGER | HTTP response code |
| `content_hash` | TEXT | SHA256 of extracted text |
| `previous_content_hash` | TEXT | Snapshotted hash from prior run (powers `--changed`) |
| `file_size_bytes` | INTEGER | File size (documents) |
| `needs_ocr` | INTEGER | Legacy/deprecated — now always `0`; OCR is decided in Pass 2, not at crawl time |
| `extracted_snippet` | TEXT | First ~500 chars of content |
| `links_out_count` | INTEGER | Outbound link count |
| `discovered_at` | TIMESTAMP | When URL was first found |
| `crawled_at` | TIMESTAMP | When it was fetched and processed |

**`links`** — one row per hyperlink (UNIQUE on `(source_url, target_url)`):

| Column | Description |
|---|---|
| `source_url` | Page containing the link |
| `target_url` | Link destination |
| `link_text` | Anchor text |
| `link_context` | Surrounding text (up to 200 chars) |
| `is_internal` | 1 if on a crawl-target domain |
| `is_document` | 1 if target is PDF/DOCX/etc. |

**`crawl_runs`** — one row per Pass 1 run (seed URL, started/completed, counts, notes).

WAL journal mode (`PRAGMA journal_mode=WAL`) is set so reads don't block writes during long crawls.

---

## 13. Chunking Strategies — Deep Dive

### `ingest_as_single` ([pass2/strategies/single.py](maryland_rag/pass2/strategies/single.py))
**Used for:** `short_static`, `nav_hub`, `location_list`, small forms, small press releases

Entire page as a single chunk. Used when the page is short enough that splitting only fragments information, or when the page's value is the full list of links/locations.

---

### `simple_split` ([pass2/strategies/simple_split.py](maryland_rag/pass2/strategies/simple_split.py))
**Used for:** `press_release` (≥150w), `form` (≥150w)

- Target: ~250 words/chunk
- Splits at double-newline paragraph boundaries
- Final chunk smaller than `MIN_CHUNK_WORDS` (30) is merged into the previous chunk
- No overlap

Press releases and form descriptions are linear prose — paragraph-boundary splitting respects natural structure without sentence analysis overhead.

---

### `semantic_with_overlap` ([pass2/strategies/semantic.py](maryland_rag/pass2/strategies/semantic.py))
**Used for:** `prose` pages ≥ 500 words

- Target: ~300 words/chunk, max 500, hard character cap 20000 (≈ 5k tokens, safe under the 8192-token embedding cap)
- Splits at sentence boundaries (`(?<=[.!?])\s+(?=[A-Z])`)
- 20% overlap between consecutive chunks
- Sentences exceeding the word or char cap are hard-split (handles OCR / CID-stream garbage that lacks proper punctuation)

**Why overlap?** Long prose carries context across sentence boundaries. Without overlap, a chunk might start mid-explanation. 20% overlap keeps the previous chunk's closing sentences at the start of the next one.

---

### `qa_pairs` ([pass2/strategies/faq.py](maryland_rag/pass2/strategies/faq.py))
**Used for:** `faq` pages

Parses HTML for Q&A pairs, in order of reliability:
1. `<dl><dt>Question</dt><dd>Answer</dd></dl>`
2. `<details><summary>Question</summary>Answer</details>`
3. Heading patterns — `<h2>Question?</h2>` + following content until next same/higher heading
4. Bold/strong patterns — `<strong>Question?</strong>` + following text until the next strong/bold

Each Q&A pair becomes its own chunk; `question` and `answer` are preserved in chunk metadata.

---

### `table_rows` ([pass2/strategies/table_rows.py](maryland_rag/pass2/strategies/table_rows.py))
**Used for:** `table_data` pages

- Parses all `<table>` elements
- Extracts `<thead>` headers (falls back to first row if it's all `<th>`)
- Each data row → one chunk: `"Column1: Value1 | Column2: Value2 | ..."`
- Prepends the `<caption>` if present

Embedding a whole results table as one vector makes every row equally retrievable, which is too coarse. One row per chunk means a query for a specific county or district retrieves exactly that row.

---

### PDF extraction ([pass2/strategies/pdf.py](maryland_rag/pass2/strategies/pdf.py))
**Used for:** all `.pdf` files

Three-tier extraction, **digital-first**:
1. **pdfplumber** — primary, handles clean digital PDFs well; extracts tables too
2. **PyMuPDF (fitz)** — fallback for complex layouts or mixed-column formats
3. **OCR via pytesseract** — last resort, triggered **only when both digital extractors return effectively empty text** (fewer than `MIN_DIGITAL_TEXT_CHARS` = 20 word-characters) and `ocr_fallback=True`. PyMuPDF renders each page at 300 dpi (`get_pixmap(dpi=300)`, no image preprocessing) and Tesseract reads it.

This is the "OCR rework": the Pass 1 `needs_ocr` manifest flag is **deprecated and ignored** — every PDF is tried digitally first, and OCR is decided from the real extraction result rather than a crawl-time byte-probe. If OCR dependencies are missing, `structure_type='ocr_failed'` and whatever digital scraps exist are kept rather than dropped.

After extraction, `_detect_pdf_structure` classifies the dominant shape (`table_heavy` → tables become row chunks, plus deduped narrative prose; `faq` → semantic chunk, since there's no HTML to parse into Q&A; `short` → `ingest_as_single`; default `prose` → `semantic_chunk`). Every branch's output passes through `enforce_chunk_caps`.

---

### DOCX extraction ([pass2/strategies/docx_strategy.py](maryland_rag/pass2/strategies/docx_strategy.py))
**Used for:** `.docx` Word files

- Walks heading hierarchy (`Title`, `Heading 1…6`)
- Each section (heading chain + body content) is a candidate chunk
- Sections ≤ 300 words: one chunk
- Sections > 300 words: re-split with `semantic_chunk`
- Tables within the DOCX are also extracted and emitted as row chunks

Heading chain is preserved in each chunk so a retrieved chunk always carries its section context (e.g., `["Chapter 2", "Mail-in Voting"]`), making it self-contained.

---

### XLS / XLSX extraction ([pass2/strategies/xls_strategy.py](maryland_rag/pass2/strategies/xls_strategy.py))
**Used for:** `.xls`, `.xlsx` spreadsheets

- `.xlsx` / `.xlsm` via `openpyxl` (read-only, data-only); `.xls` via `xlrd`
- Each sheet processed independently
- First row is treated as headers if the first two cells are non-empty strings
- Each non-empty data row → `"[SheetName] Header: Value | Header: Value | ..."` (or pipe-joined values if no headers)
- **Guardrail:** a workbook producing more than `MAX_XLS_ROW_CHUNKS` (200) row-chunks is treated as bulk tabular data and **skipped entirely** (logged loudly) rather than flooding the vector store with thousands of near-identical rows.

Same rationale as `table_rows`: per-row chunks make spreadsheet data independently retrievable.

---

## 14. Configuration Reference

### Pipeline (`maryland_rag/pass1/config.py`)

| Constant | Default | Description |
|---|---|---|
| `SEED_URLS` | `ALLOWED_EXACT_URLS` | Seed URLs for the BFS crawl (State BoE + MoCo), derived from the allowlist in `exclusions.py` |
| `DOMAINS` | `elections.maryland.gov`, `mcg.montgomerycountymd.gov` | Domains considered "internal" for link queuing |
| `MAX_DEPTH` | `6` | Max BFS depth |
| `MAX_RETRIES` | `3` | Retries for transient page fetches and the robots.txt fetch |
| `RATE_LIMIT_SECONDS` | `0.75` | Delay between crawl requests (also base for retry backoff) |
| `REQUEST_TIMEOUT` | `15` | Per-request HTTP timeout |
| `REQUESTS_PER_MINUTE_WARN` | `80` | Log a warning if exceeded in last 60 s |
| `TRAFILATURA_MIN_WORDS` | `50` | Min words for trafilatura to be trusted (else BS4 fallback) |
| `DOCUMENT_EXTENSIONS` | `.pdf .docx .doc .xls .xlsx .csv` | Treated as documents, not HTML |
| `SKIP_DOMAINS` | Facebook, Twitter, YouTube, etc. | External domains never queued |
| `RESPECT_ROBOTS_TXT` | `True` | Honor robots.txt (per-domain, RFC 9309) |
| `SAVE_RAW_HTML` | `False` | Save raw HTML to `data/raw/` (debug) |
| `LOG_DIR` / `LOG_FILE` | `logs/`, `logs/crawl.log` | Where the crawler's file handler writes |
| `PDF_PROBE_BYTES` | `4096` | **Dead** — the crawl-time PDF byte-probe was removed; OCR is decided in Pass 2 |

`TRANSIENT_HTTP_STATUSES` (`{408, 429}`) and `EXCLUDED_HTTP_STATUSES` (`{404, 410, 403, 500, 502, 503}`) live in `pass1/exclusions.py`.

### Pass 2 caps & filters

| Constant | Location | Default | Description |
|---|---|---|---|
| `MAX_CHUNK_WORDS` | `pass2/strategies/semantic.py` | `500` | Per-chunk word cap (enforced across all strategies) |
| `MAX_CHUNK_CHARS` | `pass2/strategies/semantic.py` | `20000` | Per-chunk char cap (hard embedding-safety limit) |
| `TARGET_CHUNK_WORDS` | `semantic.py` / `simple_split.py` | `300` / `250` | Target chunk size |
| `OVERLAP_RATIO` | `pass2/strategies/semantic.py` | `0.20` | Overlap between consecutive semantic chunks |
| `MAX_NON_LATIN_RATIO` | `pass2/langfilter.py` | `0.10` | Drop a chunk above this non-Latin-letter ratio |
| `MAX_XLS_ROW_CHUNKS` | `pass2/strategies/xls_strategy.py` | `200` | Skip a spreadsheet producing more rows than this |

### Pass 3 (`maryland_rag/pass3/embed.py`)

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-large` | OpenAI embedding model (also `server/rag_chain.py`) |
| `EMBED_DIM` | `3072` | Vector dimension (must match model) |
| `EMBED_BATCH_SIZE` | `100` | Texts per OpenAI call |
| `INSERT_BATCH_SIZE` | `100` | Rows per commit (each wrapped in a savepoint) |
| `RETRY_DELAY` | `5` | Base seconds for embedding retry backoff — linear + jitter, 3 attempts |

### Server — env vars

Validated in `server/config.py` (the first three raise at import time if missing):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | OpenAI API key |
| `DATABASE_URL` | **required** | PostgreSQL connection string |
| `VIOLETS_API_KEY` | **required** | Server API key — clients must send as `X-API-Key` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `gpt-5-nano` | Chat model used by RAG, classifier, and partisan checker (GPT-5 family ignores `temperature`) |
| `ELECTION_NAME` | `2026 Maryland Gubernatorial General Election` | Election named in the QA/concerns system prompts (with today's date) so deadlines are anchored to the right election |
| `ELECTION_DATE` | `November 3, 2026` | Date of that election, injected alongside `ELECTION_NAME` |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_FLOOR` | `0.0` | Drop retrieved chunks whose similarity score (1 − cosine distance) is below this value; `0.0` disables the filter |
| `PAST_ELECTION_URL_PATTERNS` | `/primary_candidates/` | Comma-separated `source_url` substrings marking past-election documents, which get down-weighted at ranking time. Empty disables. |
| `PAST_ELECTION_PENALTY` | `0.02` | Amount subtracted from a past-election chunk's similarity for ranking only (the floor and logged score stay raw). 0.02 tuned with `eval_retrieval`; ≥0.04 starts hiding answers to questions about the primary. `0` disables. |
| `CANDIDATES_URL` | 2026 primary candidates page | URL returned verbatim for `candidates`-classified queries — repoint at the general-election page once the State Board publishes it |
| `SESSION_TTL_MINUTES` | `30` | Session expiration |
| `MAX_HISTORY_TURNS` | `20` | Max conversation turns kept per user |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-`user_id` sliding-window request limit |
| `RATE_LIMIT_GLOBAL_PER_MINUTE` | `225` | Request cap per minute across **all** users combined (`/chat` + `/reset`) — backstop for `RATE_LIMIT_PER_MINUTE`, which rotating `user_id`s can bypass |

Read elsewhere at runtime (not in `config.py`):

| Variable | Default | Read in | Description |
|---|---|---|---|
| `CORS_ORIGINS` | `https://umdsurvey.umd.edu` | `main.py` | Comma-separated allowed origins |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | `main.py` | Bind address when run via `python -m server.main` |
| `LOG_LEVEL` | `INFO` | `logging_setup.py` | Root log level |
| `SERVER_LOG_FILE` | `logs/server.log` | `logging_setup.py` | Rotating server log path |
| `LOG_PROMPTS` / `LOG_RESPONSES` / `LOG_QUERIES` | `False` | `rag_logger.py` | Opt-in verbose logging of prompts / responses / query text (leave off in prod) |

### Box automation — env vars (only for `box_ingest.automate`)

| Variable | Default | Description |
|---|---|---|
| `BOX_CLIENT_ID` | **required for automate** | Box developer app client id |
| `BOX_CLIENT_SECRET` | **required for automate** | Box developer app client secret |

Tokens are cached in `.box_token` (repo root, mode `0600`) after the first browser auth.

---

## 15. Security Warning

**The `.env` file contains API keys for paid services and the server's auth token; `.box_token` caches Box OAuth tokens.**

- **Never commit `.env` or `.box_token` to Git.** Verify `.gitignore` excludes both before pushing.
- Rotate immediately if exposed:
  - OpenAI: https://platform.openai.com/api-keys
  - PostgreSQL: rotate the database password and update `DATABASE_URL`
  - `VIOLETS_API_KEY`: regenerate (`python -c 'import secrets; print(secrets.token_urlsafe(48))'`) and redeploy
  - Box: rotate the developer-app secret in the Box console and delete `.box_token`
- Prompt/response logging in `server/rag_logger.py` (`LOG_PROMPTS`, `LOG_RESPONSES`, `LOG_QUERIES`) now **defaults to `False`** (production-safe) — it is opt-in via env var. Leave these unset in production so user PII is not written to logs.

---

## 16. Troubleshooting

**Pass 1 stops unexpectedly**
Fully resumable — just rerun `python -m maryland_rag pass1`. It picks up from `pending` rows.

**Pass 2 fails on a specific PDF**
All three extraction tiers are tried before failing. If all fail, the error is logged and the batch continues. Check the URL manually — the PDF may be password-protected or corrupted.

**Pass 3: "Dimension mismatch" from pgvector**
The `chunks` table was created with a different vector dimension than `EMBED_DIM` (3,072). Pass 3 checks this at startup and stops with a `RuntimeError` naming both dimensions. Back up the table (`pg_dump -Fc -t chunks`), `DROP TABLE chunks;`, and re-run Pass 3 for both `chunks.jsonl` and `box_chunks.jsonl`. Then run `python -m server.eval_retrieval` and restart the server so query embeddings use the same model.

**Pass 3: persistent rate limit errors from OpenAI**
The code batches and retries. If limits persist, reduce `EMBED_BATCH_SIZE` in `embed.py`.

**Box ingest: "No Box URL in manifest"**
The file has no entry in `needtochunk/url_manifest.json` (or its value is `""`). Add the Box share URL and re-run; existing chunks are upserted, missing ones are added.

**Wrong classifications in audit output**
Run `python -m maryland_rag.scripts.reclassify --dry-run` to preview current rules. After updating `pass1/rules.py`, run `reclassify.py` to apply.

**Database locked error**
A previous run didn't exit cleanly. Kill any running `python -m maryland_rag` processes and retry.

**Server returns 401 on every request**
Missing or wrong `X-API-Key` header. Confirm `VIOLETS_API_KEY` is set in `.env` and that your client is sending it as `X-API-Key`.

**Server returns 429**
A single `user_id` exceeded `RATE_LIMIT_PER_MINUTE`, or all traffic combined exceeded `RATE_LIMIT_GLOBAL_PER_MINUTE` (the log line says which: `Rate limit exceeded` vs `Global rate limit exceeded`). Either back off or raise the limit.

**Browser CORS error**
Your origin isn't in `CORS_ORIGINS`. Set it explicitly at startup (comma-separated for multiple).

**Server won't start**
Check that `OPENAI_API_KEY`, `DATABASE_URL`, and `VIOLETS_API_KEY` are set in `.env` — `server/config.py` raises at import time if any are missing. The lifespan also opens the Postgres pool eagerly and **refuses to start if the database is unreachable** — confirm `DATABASE_URL` points at a running pgvector instance. Also verify the spaCy model is installed (`python -m spacy download en_core_web_lg`) — Presidio's `AnalyzerEngine` loads it at import time.

**`/health` returns 503 / `/chat` returns 503 "Service starting"**
`/health` pings the DB and returns 503 when the pool can't reach Postgres — check the database. A 503 `"Service starting, retry shortly"` on `/chat` or `/reset` means the pool/chain haven't finished initializing yet; retry after startup completes.

**Box automate: `EnvironmentError` / browser auth loop**
`box_ingest.automate` needs `BOX_CLIENT_ID` and `BOX_CLIENT_SECRET` in `.env`. First run opens a browser for OAuth and captures the redirect on `http://localhost:8080` — allow that port. If refresh keeps failing, delete `.box_token` to force a fresh browser login. Missing Playwright → `pip install -r box_ingest/requirements.txt && playwright install chromium`.

---

## 17. Glossary

Project-specific terms and non-obvious library names only.

| Term | Definition |
|---|---|
| **manifest.db** | The SQLite database produced by Pass 1. One row per discovered URL with all classification and metadata. |
| **chunks.jsonl** | JSONL file (one JSON object per line) produced by Pass 2. Each line is one chunk with text and full metadata. |
| **box_chunks.jsonl** | Same shape as `chunks.jsonl`, produced by `box_ingest` from `needtochunk/`. |
| **chunks (table)** | PostgreSQL table storing vectors and metadata. `--resume` queries existing `chunk_id`s to skip them. |
| **content_hash** | SHA256 of a page's extracted text. Two pages with the same hash have identical content and are deduplicated in Pass 2. |
| **previous_content_hash** | Snapshot of `content_hash` taken before a re-crawl; powers `pass2 --changed` and the `all` command's first-run logic. |
| **needs_ocr** | Legacy manifest flag. **Deprecated** — Pass 1 no longer probes PDFs and always stores `False`; Pass 2 decides OCR from a real digital-extraction attempt instead. |
| **chunk_id** | Stable 32-char id derived from a chunk's *content* (`sha256(normalized_text + section_hierarchy + chunk_index)[:32]`), not from its source URL — so re-ingests don't orphan pgvector rows. |
| **langfilter** | `pass2/langfilter.py` — drops chunks whose non-Latin-letter ratio exceeds `MAX_NON_LATIN_RATIO` (0.10). Content-side companion to the Pass-1 URL exclusion of translated documents. |
| **heartbeat** | Periodic server log line (every 300 s) emitted by `_periodic_maintenance`, summarizing rolling requests / errors / blocked / cost and the pool gauge (fed by `metrics.py`). |
| **Box automate** | `box_ingest.automate` — Step 1 of the Box pipeline: OAuth into Box, scrape the Hub (Playwright), download `include` files into `needtochunk/`, and update `url_manifest.json`. Separate from `box_ingest.ingest` (Step 2, chunking). |
| **verify_chunks** | Post-ingest gate (`scripts/verify_chunks.py`) checking per-page coverage, junk heuristics, and JSONL-vs-pgvector parity; exits non-zero on failure. |
| **allowlist** | Two-layer URL gate in `pass1/exclusions.py`: prefix list + exact list. Crawler will not enqueue anything failing the allowlist. |
| **trafilatura** | Library that extracts clean article text from HTML, removing nav, footers, boilerplate. Primary HTML extractor in Pass 1. |
| **pdfplumber** | Primary digital-PDF extractor in Pass 2 (text + tables). |
| **nav_hub** | Pages 150–499 words that are primarily lists of links — treated as navigation, kept as a single chunk so the link set stays together. |
| **location_list** | MoCo pages whose value is a list of polling/drop-box locations — kept whole. |
| **concerns** | Query category for queries about election integrity, rumors, or misinformation. RAG chain runs with the Rumor Control prompt that always links Maryland's official rumor-control page first. |
| **Pass** | One stage of the web pipeline. Pass 1 = crawl, Pass 2 = chunk, Pass 3 = embed and upload. |
| **Box ingest** | Parallel pipeline for documents shared via Box that the crawler can't reach. Writes the same chunk schema. |
| **RAG chain** | The LangChain runnable pipeline in the server: contextualize → retrieve → answer. Implemented as a `RunnableLambda` that branches on `query_category`. |
| **PgVectorRetriever** | Custom `BaseRetriever` subclass in `rag_chain.py` that embeds queries with OpenAI and queries PostgreSQL via pgvector for cosine similarity search. |
| **SessionStore** | Thread-safe in-memory conversation store in `session.py`. Tracks chat history per user with TTL expiration and max-turn limits. |
| **Presidio** | Microsoft's PII detection engine, used in `middleware.detect_pii` to scan user input. |
| **QueryContext** | Dataclass in `middleware.py` that travels through guardrails: classification result, PII detection flags, query category. |
| **RAGCallbackHandler** | LangChain callback handler in `rag_logger.py` that captures token usage, estimates cost, and logs retriever performance per request. |
| **Structured output** | LangChain/OpenAI feature used by the guardrail LLMs — returns Pydantic models (`ClassificationResult`, `PartisanCheckResult`) instead of free-form text. |
| **Fail closed** | Guardrail error-handling strategy used by all three server guardrails: if PII detection, classification, or the partisan check errors or can't clear the response, the answer is **suppressed** (canned refusal / error message) rather than passed through. Chosen deliberately for an elections chatbot, where shipping an unvetted answer is worse than a graceful refusal. |
