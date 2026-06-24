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

Embedding model: `text-embedding-3-small` (1,536-dim, OpenAI).
Default LLM: `gpt-5-nano` (overridable via `LLM_MODEL`).

---

## 2. Repository Structure

```
VIOLETS/
├── .env                              ← API keys (NEVER commit to GitHub)
├── README.md                         ← This file
│
├── data/                             ← All pipeline artifacts (gitignored)
│   ├── manifest.db                   ← SQLite crawl database (Pass 1 output)
│   ├── chunks.jsonl                  ← Web chunks (Pass 2 output)
│   ├── box_chunks.jsonl              ← Box document chunks (Box Ingest output)
│   └── cache/                        ← Disk cache of fetched pages / binaries
│
├── logs/                             ← Runtime logs (gitignored)
│   └── crawl.log                     ← Crawl activity log
│
├── needtochunk/                      ← Curated documents from Box (pre-selected)
│   ├── <year-folder>/<file>.pdf      ← PDFs, DOCX, TXT, etc.
│   └── url_manifest.json             ← Maps each file's relative path → Box share URL
│
├── maryland_rag/                     ← Web pipeline Python package
│   ├── __main__.py                   ← CLI entry point (pass1 / pass2 / pass3 / all / audit)
│   ├── requirements.txt              ← Pipeline Python dependencies
│   │
│   ├── pass1/                        ← Phase 1: Crawl the allowlisted sites
│   │   ├── config.py                 ← Seeds, domains, rate limit, paths
│   │   ├── crawler.py                ← BFS web crawler (multi-seed, multi-domain)
│   │   ├── extractor.py              ← Text/link/metadata extraction (HTML + doc HEAD probes)
│   │   ├── classifier.py             ← Crawl-time wrapper over rules.py
│   │   ├── rules.py                  ← Shared classification rules (used by reclassify too)
│   │   ├── exclusions.py             ← Allowlist + exclusion gate (single source of truth)
│   │   ├── db.py                     ← All SQLite read/write operations
│   │   └── utils.py                  ← URL normalization helpers
│   │
│   ├── pass2/                        ← Phase 2: Break pages into chunks
│   │   ├── chunker.py                ← Orchestrates all chunking
│   │   ├── metadata.py               ← Builds chunk metadata (incl. multi-source dedup)
│   │   ├── cache.py                  ← Caches HTTP fetches to disk
│   │   └── strategies/               ← One file per chunking approach
│   │       ├── single.py             ← Entire page as one chunk
│   │       ├── simple_split.py       ← Split at paragraph boundaries
│   │       ├── semantic.py           ← Sentence splits with overlap
│   │       ├── faq.py                ← Extract Q&A pairs
│   │       ├── table_rows.py         ← One chunk per HTML table row
│   │       ├── pdf.py                ← Extract text from PDFs (pdfplumber → pymupdf → OCR)
│   │       ├── docx_strategy.py      ← Extract DOCX by heading hierarchy
│   │       └── xls_strategy.py       ← Extract XLS/XLSX rows
│   │
│   ├── pass3/                        ← Phase 3: Embed and upsert
│   │   └── embed.py                  ← OpenAI embeddings + pgvector upsert
│   │
│   └── scripts/                      ← Maintenance utilities
│       ├── db_cleanup.py             ← Remove duplicate URL variants from manifest.db
│       ├── reclassify.py             ← Re-classify pages using stored metadata
│       ├── apply_keep_filter.py      ← Mark out-of-scope rows as excluded
│       └── stress_test.py            ← Server-side concurrency + edge-case test harness
│
├── box_ingest/                       ← Parallel pipeline for curated Box documents
│   └── ingest.py                     ← Reads needtochunk/, writes data/box_chunks.jsonl
│
└── server/                           ← FastAPI chatbot server
    ├── main.py                       ← App, lifespan, /chat, /reset, /health, auth, rate limit
    ├── config.py                     ← Loads .env, exposes settings
    ├── rag_chain.py                  ← LangChain RAG chain + PgVectorRetriever
    ├── middleware.py                 ← PII, classification, partisan-check guardrails
    ├── rag_logger.py                 ← Callback handler for token/cost/timing logging
    ├── session.py                    ← Thread-safe in-memory conversation store
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

# Download spaCy language model (required by server PII detection)
python -m spacy download en_core_web_lg
```

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
| `langchain` / `langchain-openai` | Server | RAG chain orchestration + ChatOpenAI + OpenAI embeddings |
| `presidio-analyzer` | Server | PII detection (SSN, credit card, email, phone, passport, driver's license, IP) |
| `spacy` | Server | NLP backend for Presidio (requires `en_core_web_lg`) |

### 3.3 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/violets
VIOLETS_API_KEY=<a long random string — required by /chat and /reset>
```

The server validates `OPENAI_API_KEY`, `DATABASE_URL`, and `VIOLETS_API_KEY` at import time; missing values raise immediately instead of failing later inside a request.

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

BFS from the seeds defined in `pass1/config.SEED_URLS`, up to 6 levels deep, scoped by the allowlist in `pass1/exclusions.py`.

**Seed set (current):**
- State BoE — prefix-crawled: `/voting/`, `/voter_registration/`, plus `/press_room/documents/2026/`
- State BoE — exact pages: `election_security.html`, `press_room/index.html`, `rumor_control.html`, `elections/2026/index.html`
- Montgomery County — exact pages only (child links discovered are not followed unless they re-match the allowlist): drop boxes, election judge pages, vote-by-mail, FAQs, early voting centers, accessibility

**Crawl sequence per page:**

```
1. Gate 1: Allowlist + exclusion patterns (no network call) → Skip if not in scope
2. Gate 2: Depth > MAX_DEPTH (6)                            → Mark 'skipped'
3. Gate 3: robots.txt (per-domain parser)                   → Mark 'excluded'
4. Fetch the page                                           ← Single HTTP request
5. Gate 4: Bad HTTP status (404, 410, 403, 500, 502, 503)?  → Mark 'failed'
6. Extract content (text, links, metadata, breadcrumbs)
7. Classify the page (rules.py)
8. Persist row to manifest.db
9. Enqueue discovered internal links
```

**Single fetch per page:** No double-fetch (check then extract); everything happens in the one request.

**Resumability:** Every discovered URL is written to `manifest.db` as `pending` before fetching. On restart, the crawler seeds the queue from `pending` rows — no progress lost.

**Rate limiting:** `RATE_LIMIT_SECONDS = 0.75` between requests. A sliding-window `RateMonitor` logs a warning if request rate exceeds `REQUESTS_PER_MINUTE_WARN` (80) in the last 60 s.

---

### 4.2 The Allowlist + Exclusions (`pass1/exclusions.py`)

`exclusions.py` is the single source of truth for both layers — referenced by the crawler, `db_cleanup`, and `apply_keep_filter`.

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

`is_excluded_status({404, 410, 403, 500, 502, 503})` is what flips an HTTP response to `failed` in step 5 above.

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
- HEAD request only — get file size without downloading
- PDFs additionally: download the first `PDF_PROBE_BYTES` (4096) via HTTP `Range` and look for text markers (`/Font`, `/Text`, `Tj`, `TJ`, `/ToUnicode`). If none found, flag `needs_ocr = 1`. If the server ignores `Range`, the probe reads just one chunk and aborts cleanly so we never download the full PDF.

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

Pass 2 re-fetches HTML and documents to get full content (Pass 1 only stored a 500-char snippet for HTML). All fetches go through a disk cache at `data/cache/`, keyed by `SHA256(url)`, stored as `.html` (text) or `.bin` (bytes). Cache hits skip the network entirely. Cache misses sleep `RATE_LIMIT_SECONDS` before fetching so re-runs after a crash don't hammer the server.

Disk cache (not in-memory) because Pass 2 runs can be long and interrupted — a persistent cache survives restarts.

---

### 5.3 Deduplication (`pass2/chunker.py`)

Pages with identical `content_hash` (same content under different URLs) are extracted once. Resulting chunks carry `source_urls` (plural array) listing every URL pointing at that content. The chunk's `chunk_id` is derived from the sorted URL set so it remains stable even if a secondary URL disappears between runs. This avoids inserting duplicate vectors while preserving full provenance.

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

Files whose mapping is empty or missing are **skipped with a warning** — they won't be embedded.

**Run it:**
```bash
python -m box_ingest.ingest                       # write data/box_chunks.jsonl
python -m box_ingest.ingest --dry-run             # show manifest mappings, no writes
python -m box_ingest.ingest --output path/to.jsonl
```

**Extraction:**
- `.pdf` → pdfplumber → pymupdf → OCR (tesseract @ 250 dpi for scanned PDFs)
- `.docx` / `.doc` → python-docx walks heading hierarchy; each section keeps its `heading_chain`
- `.txt` → plain read
- Other extensions → warned and skipped

**Chunking:**
- DOCX with headings → each section chunked independently
- ≤ 150 words → `ingest_as_single`
- > 150 words → `semantic_chunk` (with paragraph-boundary fallback)
- Section hierarchy = folder chain (relative to `needtochunk/`) + DOCX heading chain

**Stability:** `chunk_id = sha256(source_url + ":" + chunk_index)[:32]` — deterministic. Re-running on unchanged files produces the same IDs, so Pass 3's upsert is a no-op for unchanged content.

Each record in `data/box_chunks.jsonl` uses the same shape as `chunks.jsonl` (`page_classification='document'`, `chunking_strategy='box_ingest'`), so Pass 3 ingests it identically.

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
4. Embed in batches of `EMBED_BATCH_SIZE` (100) → `text-embedding-3-small` → 1,536-dim vectors. 3 retries with exponential backoff.
5. Upsert into PostgreSQL with `ON CONFLICT (chunk_id) DO UPDATE`. Each row is wrapped in a savepoint so a single failure doesn't roll back the rest of the batch.

---

### 7.2 pgvector Table Schema

```sql
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    embedding  vector(1536),
    text       TEXT,
    source_url TEXT,
    title      TEXT,
    metadata   JSONB DEFAULT '{}'::jsonb
);
```

| Setting | Value | Reason |
|---|---|---|
| Dimension | 1,536 | Matches `text-embedding-3-small` |
| Metric | Cosine | Standard for normalized text embeddings (`embedding <=> %s::vector`) |

> ⚠️ The table is created with no ANN index. For the current corpus size this is fine and queries do a sequential scan. If the corpus grows enough that retrieval latency matters, add an `ivfflat` or `hnsw` index manually on `embedding`.

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
| `GET` | `/health` | Public | Health check (returns `{status, model}`) |

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
│  RATE LIMIT: sliding-window per user_id (20/min)      │
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
│              voter_update, candidates, partisan       │
│  → polling_location / voter_lookup / voter_update /   │
│    candidates  → return hardcoded URL, exit           │
│  → partisan                  → return fallback, exit  │
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
│  retry. Fails open after exhaustion — returns last    │
│  generated response.                                   │
└───────────────────────┬───────────────────────────────┘
                        ▼
   log_request() + store.add_exchange() + return ChatResponse
```

> ℹ️ There is **no output-side PII scrub** in the current code — only input PII is blocked. The partisan check is the only post-generation guard.

### Server Modules

**`main.py`** — FastAPI app with async lifespan startup (logging, `ConnectionPool`, `SessionStore`, `build_chain()`). Hosts `_RateLimiter`, the `X-API-Key` dependency, CORS middleware, and a background task that calls `store.cleanup_expired()` every 5 minutes. On RAG failure → HTTP 502. Pool is closed with a 30 s grace on shutdown.

**`rag_chain.py`** — Built with `langchain_core` runnables. A custom `PgVectorRetriever(BaseRetriever)` queries PostgreSQL via pgvector directly (`embedding <=> %s::vector`) and emits debug logs per retrieved chunk. Four prompts:
  - `_CONTEXTUALIZE_PROMPT` — rephrases follow-ups into standalone questions
  - `_QA_PROMPT` — main answer prompt with `[Source N]` citation contract
  - `_CONVERSATIONAL_PROMPT` — answers from chat history only, no retrieval
  - `_CONCERNS_PROMPT` — Rumor Control prompt; always starts the response by linking https://elections.maryland.gov/press_room/rumor_control.html

After answer generation, `_replace_source_refs` substitutes inline `[Source N]` markers with markdown links built from the retrieved source list, so the front-end gets clickable citations.

`PgVectorRetriever._aget_relevant_documents` is currently a TODO — `BaseRetriever.ainvoke` runs the sync method in a threadpool, which limits per-worker throughput.

**`middleware.py`** — All guardrail logic. Shared `QueryContext` dataclass travels through the request.

| Function | Purpose | Failure mode |
|---|---|---|
| `detect_pii(query, ctx)` | Presidio scan for `US_SSN`, `CREDIT_CARD`, `EMAIL_ADDRESS`, `IP_ADDRESS`, `PHONE_NUMBER`, `US_PASSPORT`, `US_DRIVER_LICENSE` at score ≥ 0.5 | Hard block — canned PII fallback |
| `classify_query(query, ctx)` | LLM (`LLM_MODEL`, structured `ClassificationResult`) → 8 categories. Short-circuits `__User concerns:__` to skip the LLM | Fail open — allows query through |
| `check_partisan_response(...)` | Structured `PartisanCheckResult`; on `is_partisan=True`, re-invokes chain with stricter retry prompt up to `MAX_PARTISAN_RETRIES` (2) | Fail open — returns last response |

Categories `normal` / `conversational` / `concerns` reach the RAG chain. The four "hardcoded URL" categories (`polling_location`, `voter_lookup`, `voter_update`, `candidates`) return a static URL from `FALLBACK_RESPONSES` without ever calling the LLM. `partisan` returns a refusal message.

**`rag_logger.py`** — LangChain `BaseCallbackHandler` that captures LLM prompts/responses (toggleable via `LOG_PROMPTS`, `LOG_RESPONSES`, `LOG_QUERIES` module flags — currently all `True`; set to `False` before deploying so PII is not written to logs), token usage, estimated cost per request from a built-in `_COST_TABLE`, and retriever start/end timing. `log_request()` writes one summary line per `/chat` exchange. The lock around the run-tracking dicts is required because the retriever fires callbacks from worker threads.

**`session.py`** — Thread-safe in-memory per-`user_id` conversation store with TTL expiration (`SESSION_TTL_MINUTES`) and max-turn cap (`MAX_HISTORY_TURNS`). Designed for pilot-scale (tens of concurrent users) — swap to Redis or a database for production scale.

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

In-memory sliding-window limiter (`_RateLimiter` in `main.py`), keyed by `user_id`, defaulting to `RATE_LIMIT_PER_MINUTE = 20` requests per 60 s. Exceeded → **HTTP 429** `{"detail":"Rate limit exceeded"}`. Runs **before** PII/classification/RAG, so a runaway client cannot drain OpenAI credits.

The limiter is per-process. Behind multiple replicas you would either pin users to a replica or move the counter into Redis.

### 9.4 Stress Testing

`maryland_rag/scripts/stress_test.py` exercises the running server with concurrent RAG queries, same-user races, malformed inputs, mixed guardrail paths, session reset under load, and health responsiveness during load.

```bash
export VIOLETS_API_KEY=...
uvicorn server.main:app --host 0.0.0.0 --port 8000 &   # start server first
python -m maryland_rag.scripts.stress_test
```

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

### 10.2 Apply Keep Filter (`scripts/apply_keep_filter.py`)

Narrows the crawled set down to a curated keep list for 2025–2026, marking everything else as `excluded` with reason `keep_filter_2026`. Rows are **never deleted** — Pass 2 reads `crawl_status='crawled'` only, so excluded rows are skipped automatically. This is how the corpus is restricted to current-cycle PDFs and key handbooks even when the crawler discovered older material.

Rule shapes:
- `year_prefix` — URL starts with prefix AND contains `2025` or `2026`
- `prefix` — URL starts with prefix (no year filter)
- `exact` — exact URL match (including alternative encodings, e.g. spaces vs `%20`)

```bash
python -m maryland_rag.scripts.apply_keep_filter --dry-run   # show counts
python -m maryland_rag.scripts.apply_keep_filter --apply     # mark excluded
python -m maryland_rag.scripts.apply_keep_filter --revert    # undo
```

### 10.3 Re-classification (`scripts/reclassify.py`)

Re-applies the current `pass1/rules.py` classifier to all crawled HTML rows using stored metadata (no re-fetching). Use this whenever you change `rules.py`. Crawl-time vs. reclassify-time differ only in that crawl-time has `raw_html` for the structural-pattern fallback; reclassify does not.

```bash
python -m maryland_rag.scripts.reclassify --dry-run   # preview transitions
python -m maryland_rag.scripts.reclassify              # apply
```

### 10.4 Stress Test (`scripts/stress_test.py`)

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

# 5. (Optional) narrow corpus to the 2025/2026 keep list
python -m maryland_rag.scripts.apply_keep_filter --apply

# 6. Chunk the web pages
python -m maryland_rag pass2

# 7. Chunk Box documents (skips any file missing a URL in url_manifest.json)
python -m box_ingest.ingest

# 8. Embed and upload — web chunks
python -m maryland_rag pass3 --chunks data/chunks.jsonl

# 9. Embed and upload — Box chunks (same table, different input)
python -m maryland_rag pass3 --chunks data/box_chunks.jsonl

# 10. Verify vectors are in PostgreSQL:
#     psql $DATABASE_URL -c "SELECT count(*) FROM chunks;"

# 11. Start the server
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### One-shot: `python -m maryland_rag all`

`all` runs the full pipeline end-to-end and is safe to re-run:

```bash
python -m maryland_rag all
```

Internally it:
1. Captures `is_first_run()` **before** snapshotting hashes (the order matters — once `snapshot_hashes_for_recrawl()` runs, `previous_content_hash` is populated and the question becomes meaningless).
2. Snapshots current `content_hash → previous_content_hash`.
3. Runs Pass 1 with `resume=False` (full re-discovery).
4. Runs Pass 2 with `only_changed=not first_run` — first ever run chunks everything; subsequent runs re-chunk only pages whose hash actually changed.
5. Runs Box ingest.
6. Runs Pass 3 on web chunks with `resume=False` (so changed content always upserts).
7. Runs Pass 3 on Box chunks (only if any were produced).

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
| `needs_ocr` | INTEGER | 1 if PDF has no text layer in first 4KB |
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

Three-tier extraction:
1. **pdfplumber** — primary, handles clean digital PDFs well; extracts tables too
2. **PyMuPDF (fitz)** — fallback for complex layouts or mixed-column formats
3. **OCR via pytesseract** — for `needs_ocr=1` PDFs flagged in Pass 1; PyMuPDF renders pages at 300 dpi, Tesseract reads the text

After extraction, `_detect_pdf_structure` classifies the dominant shape (`table_heavy` → tables become row chunks; `faq` → semantic chunk (no HTML to parse); `short` → `ingest_as_single`; default `prose` → `semantic_chunk`).

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

- `.xlsx` via `openpyxl` (read-only, data-only); `.xls` via `xlrd`
- Each sheet processed independently
- First row is treated as headers if all the first two cells are non-empty strings
- Each non-empty data row → `"[SheetName] Header: Value | Header: Value | ..."` (or pipe-joined values if no headers)

Same rationale as `table_rows`: per-row chunks make spreadsheet data independently retrievable.

---

## 14. Configuration Reference

### Pipeline (`maryland_rag/pass1/config.py`)

| Constant | Default | Description |
|---|---|---|
| `SEED_URLS` | curated list | Seed URLs for the BFS crawl (State BoE + MoCo) |
| `DOMAINS` | `elections.maryland.gov`, `mcg.montgomerycountymd.gov` | Domains considered "internal" for link queuing |
| `MAX_DEPTH` | `6` | Max BFS depth |
| `RATE_LIMIT_SECONDS` | `0.75` | Delay between crawl requests |
| `REQUEST_TIMEOUT` | `15` | Per-request HTTP timeout |
| `REQUESTS_PER_MINUTE_WARN` | `80` | Log a warning if exceeded in last 60 s |
| `PDF_PROBE_BYTES` | `4096` | Bytes checked for PDF text markers |
| `TRAFILATURA_MIN_WORDS` | `50` | Min words for trafilatura to be trusted (else BS4 fallback) |
| `DOCUMENT_EXTENSIONS` | `.pdf .docx .doc .xls .xlsx .csv` | Treated as documents, not HTML |
| `SKIP_DOMAINS` | Facebook, Twitter, YouTube, etc. | External domains never queued |
| `RESPECT_ROBOTS_TXT` | `True` | Honor robots.txt |
| `SAVE_RAW_HTML` | `False` | Save raw HTML to `data/raw/` (debug) |
| `LOG_DIR` / `LOG_FILE` | `logs/`, `logs/crawl.log` | Where the crawler's file handler writes |

### Pass 3 (`maryland_rag/pass3/embed.py`)

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBED_DIM` | `1536` | Vector dimension (must match model) |
| `EMBED_BATCH_SIZE` | `100` | Texts per OpenAI call |
| `INSERT_BATCH_SIZE` | `100` | Rows per commit (each wrapped in a savepoint) |
| `RETRY_DELAY` | `5` | Seconds between embedding retries (3 attempts) |

### Server (`server/config.py`) — env vars

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | OpenAI API key |
| `DATABASE_URL` | **required** | PostgreSQL connection string |
| `VIOLETS_API_KEY` | **required** | Server API key — clients must send as `X-API-Key` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `gpt-5-nano` | Chat model used by RAG, classifier, and partisan checker (GPT-5 family ignores `temperature`) |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `SESSION_TTL_MINUTES` | `30` | Session expiration |
| `MAX_HISTORY_TURNS` | `20` | Max conversation turns kept per user |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-`user_id` sliding-window request limit |
| `CORS_ORIGINS` | `https://umdsurvey.umd.edu` | Comma-separated allowed origins |

---

## 15. Security Warning

**The `.env` file contains API keys for paid services and the server's auth token.**

- **Never commit `.env` to Git.** Verify `.gitignore` excludes it before pushing.
- Rotate immediately if exposed:
  - OpenAI: https://platform.openai.com/api-keys
  - PostgreSQL: rotate the database password and update `DATABASE_URL`
  - `VIOLETS_API_KEY`: regenerate (`python -c 'import secrets; print(secrets.token_urlsafe(48))'`) and redeploy
- Prompt/response logging in `server/rag_logger.py` (`LOG_PROMPTS`, `LOG_RESPONSES`, `LOG_QUERIES`) defaults to `True`. Set these to `False` before deploying to production so user PII is not written to logs.

---

## 16. Troubleshooting

**Pass 1 stops unexpectedly**
Fully resumable — just rerun `python -m maryland_rag pass1`. It picks up from `pending` rows.

**Pass 2 fails on a specific PDF**
All three extraction tiers are tried before failing. If all fail, the error is logged and the batch continues. Check the URL manually — the PDF may be password-protected or corrupted.

**Pass 3: "Dimension mismatch" from pgvector**
The `chunks` table was created with a different vector dimension than 1,536. Drop and recreate the table (`DROP TABLE chunks;`) and re-run Pass 3, or adjust `EMBED_DIM` in `embed.py`.

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
A single `user_id` exceeded `RATE_LIMIT_PER_MINUTE`. Either back off or raise the limit.

**Browser CORS error**
Your origin isn't in `CORS_ORIGINS`. Set it explicitly at startup (comma-separated for multiple).

**Server won't start**
Check that `OPENAI_API_KEY`, `DATABASE_URL`, and `VIOLETS_API_KEY` are set in `.env` — `server/config.py` raises at import time if any are missing. Also verify the spaCy model is installed (`python -m spacy download en_core_web_lg`) — Presidio's `AnalyzerEngine` loads it at import time.

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
| **needs_ocr** | Flag on PDFs where no text layer was detected in the first 4KB. |
| **allowlist** | Two-layer URL gate in `pass1/exclusions.py`: prefix list + exact list. Crawler will not enqueue anything failing the allowlist. |
| **keep filter** | Curated 2025–2026 keep list applied via `apply_keep_filter` — narrows the crawl down to current-cycle materials by marking everything else `excluded`. |
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
| **Fail open** | Guardrail error-handling strategy: if the classifier or partisan checker LLM call fails, the query is allowed through rather than blocked. Prevents guardrail outages from taking down the chatbot. |
