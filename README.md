# VIOLETS — Maryland Elections RAG Pipeline

## Table of Contents

1. [Big Picture: How the Pipeline Works](#1-big-picture-how-the-pipeline-works)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Pass 1 — Crawling & Classification](#4-pass-1--crawling--classification)
5. [Pass 2 — Chunking](#5-pass-2--chunking)
6. [Pass 3 — Embedding & Pinecone Upload](#6-pass-3--embedding--pinecone-upload)
7. [Server — FastAPI Chatbot](#7-server--fastapi-chatbot)
8. [Utility Scripts](#8-utility-scripts)
9. [Running the Full Pipeline](#9-running-the-full-pipeline)
10. [The Database (manifest.db)](#10-the-database-manifestdb)
11. [Chunking Strategies — Deep Dive](#11-chunking-strategies--deep-dive)
12. [Configuration Reference](#12-configuration-reference)
13. [Security Warning](#13-security-warning)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)

---

## 1. Big Picture: How the Pipeline Works

```
                           ┌──────────────────────────────────────┐
                           │  elections.maryland.gov (website)     │
                           └─────────────────┬────────────────────┘
                                             │
                                    PASS 1: Crawl
                                             │
                           ┌─────────────────▼────────────────────┐
                           │  data/manifest.db  (SQLite database)  │
                           │  1,347 rows — every URL discovered,   │
                           │  classified, and metadata-tagged       │
                           └─────────────────┬────────────────────┘
                                             │
                                   PASS 2: Chunk
                                             │
                           ┌─────────────────▼────────────────────┐
                           │  data/chunks.jsonl                     │
                           │  One record per chunk, with full       │
                           │  metadata (source URL, title, etc.)    │
                           └─────────────────┬────────────────────┘
                                             │
                                PASS 3: Embed & Upload
                                             │
                           ┌─────────────────▼────────────────────┐
                           │  Pinecone — index: maryland-elections  │
                           │  Ready for semantic search             │
                           └─────────────────┬────────────────────┘
                                             │
                                    SERVER: Query
                                             │
                           ┌─────────────────▼────────────────────┐
                           │  FastAPI chatbot — /chat endpoint      │
                           │  Guardrails (PII, classification,      │
                           │  partisan check) + RAG chain with      │
                           │  conversation history                  │
                           └──────────────────────────────────────┘
```

Each pass reads from the previous stage's output. This makes it easy to re-run any single stage independently — e.g., if you change chunking logic, you only need to re-run Pass 2 and 3, not the full crawl.

Embedding model: `text-embedding-3-small` (1,536-dim, OpenAI).

---

## 2. Repository Structure

```
VIOLETS/
├── .env                              ← API keys (NEVER commit this to GitHub)
├── README.md                         ← This file
│
├── data/                             ← All pipeline artifacts (gitignored)
│   ├── manifest.db                   ← SQLite crawl database (Pass 1 output)
│   ├── chunks.jsonl                  ← Chunked text records (Pass 2 output)
│   ├── chunks.checkpoint             ← Tracks what has been uploaded (Pass 3 resume)
│   └── cache/                        ← Disk cache of fetched pages (used by Pass 2)
│
├── logs/                             ← Runtime logs (gitignored)
│   └── crawl.log                     ← Crawl activity log
│
├── maryland_rag/                     ← RAG pipeline Python package
│   ├── __main__.py                   ← CLI entry point (run all passes from here)
│   ├── requirements.txt              ← Pipeline Python dependencies
│   │
│   ├── pass1/                        ← Phase 1: Crawl the website
│   │   ├── config.py                 ← Configuration (seed URL, rate limit, paths)
│   │   ├── crawler.py                ← BFS web crawler
│   │   ├── classifier.py             ← Assigns page_classification to each page
│   │   ├── extractor.py              ← Extracts text, links, metadata from pages
│   │   ├── db.py                     ← All database read/write operations
│   │   ├── utils.py                  ← URL normalization helpers
│   │   └── exclusions.py             ← Rules for skipping certain URLs
│   │
│   ├── pass2/                        ← Phase 2: Break pages into chunks
│   │   ├── chunker.py                ← Orchestrates all chunking
│   │   ├── metadata.py               ← Builds metadata records for each chunk
│   │   ├── cache.py                  ← Caches HTTP fetches to disk
│   │   └── strategies/               ← One file per chunking approach
│   │       ├── single.py             ← Entire page as one chunk
│   │       ├── simple_split.py       ← Split at paragraph boundaries
│   │       ├── semantic.py           ← Split by sentence with overlap
│   │       ├── faq.py                ← Extract Q&A pairs
│   │       ├── table_rows.py         ← One chunk per table row
│   │       ├── pdf.py                ← Extract text from PDF files
│   │       └── docx_strategy.py      ← Extract DOCX by heading hierarchy
│   │
│   ├── pass3/                        ← Phase 3: Embed and upload
│   │   └── embed.py                  ← OpenAI embeddings + Pinecone upsert
│   │
│   └── scripts/                      ← Maintenance utilities
│       ├── db_cleanup.py             ← Remove duplicates and junk from manifest.db
│       └── reclassify.py             ← Re-classify pages without re-crawling
│
└── server/                           ← FastAPI chatbot server
    ├── main.py                       ← App, endpoints (/chat, /reset, /health)
    ├── config.py                     ← Loads .env, configurable settings
    ├── rag_chain.py                  ← LangChain RAG chain with Pinecone retriever
    ├── middleware.py                  ← Guardrails: PII detection, query classification, partisan check
    ├── rag_logger.py                 ← Callback handler for token/cost logging per request
    ├── session.py                    ← In-memory conversation session store
    └── requirements.txt              ← Server Python dependencies
```

---

## 3. Prerequisites & Setup

### 3.1 Software Requirements

- **Python 3.11+**
  ```bash
  python --version
  ```
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
| `beautifulsoup4` | Pass 1, 2 | Parses HTML to find links, tables, Q&A structure |
| `requests` | Pass 1, 2 | HTTP fetching |
| `pdfplumber` | Pass 2 | Extracts text from digital (non-scanned) PDFs |
| `PyMuPDF` | Pass 2 | Fallback PDF extraction + renders pages as images for OCR |
| `python-docx` | Pass 2 | Reads `.docx` Word files |
| `pytesseract` | Pass 2 | OCR on scanned PDF page images |
| `Pillow` | Pass 2 | Image processing (used with pytesseract) |
| `openai` | Pass 3 | Generates embeddings |
| `pinecone` | Pass 3, Server | Upserts vectors to Pinecone / vector store client |
| `fastapi` | Server | Web framework |
| `uvicorn` | Server | ASGI server to run FastAPI |
| `langchain` | Server | RAG chain orchestration (core runnables, prompts, output parsers) |
| `langchain-openai` | Server | ChatOpenAI LLM + OpenAI embeddings integration |
| `presidio-analyzer` | Server | PII detection in user queries and LLM responses (SSN, credit card, phone, etc.) |
| `spacy` | Server | NLP backend for Presidio PII entity recognition (requires `en_core_web_lg` model) |

### 3.3 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=maryland-elections
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

> **See [Section 13](#13-security-warning) for API key security.**

---

## 4. Pass 1 — Crawling & Classification

**Goal:** BFS crawl of `elections.maryland.gov`, extract content, classify each page.

**Run it:**
```bash
python -m maryland_rag pass1
# Start fresh (ignore saved progress):
python -m maryland_rag pass1 --no-resume
```

**Output:** `data/manifest.db` — one row per discovered URL.

---

### 4.1 How the Crawler Works (`pass1/crawler.py`)

BFS starting at `https://elections.maryland.gov`, up to 6 levels deep.

**Crawl sequence per page:**

```
1. Gate 1: Is this URL in the exclusion list?  → Skip (no network call)
2. Gate 2: Is depth > 6?                       → Skip
3. Gate 3: Blocked by robots.txt?              → Skip
4. Fetch the page                              ← Single HTTP request
5. Gate 4: Bad HTTP status (404, 500, etc.)?   → Mark excluded, move on
6. Extract content (text, links, metadata)
7. Classify the page
8. Save to manifest.db
9. Enqueue discovered links
```

**Single fetch per page:** We do not double-fetch (once to check, once to extract). All extraction happens in the same request. This halves load on Maryland's servers.

**Resumability:** Every discovered URL is written to `manifest.db` as `pending` before it is fetched. On restart, the crawler seeds from `pending` rows — no progress is lost.

**Rate limiting:** 0.75s between requests. A sliding-window `RateMonitor` logs a warning if the rate exceeds 80 req/min.

---

### 4.2 What Gets Excluded and Why (`pass1/exclusions.py`)

`exclusions.py` is the single source of truth for all skip rules — both the crawler and `db_cleanup.py` reference it.

| Category | Why Excluded |
|---|---|
| Past election results by year (`/elections/2014/` etc.) | Historical, not relevant to current election procedures |
| Old press releases (`/press_room/prior_releases`) | Outdated |
| Campaign finance pages | Different regulatory domain |
| Image/media/asset files (.jpg, .css, .js, etc.) | No text content |
| Social media domains (Facebook, Twitter, YouTube, etc.) | External links only |
| 404/410/403/500/502/503 responses | Broken or unavailable |

If you need to add an exclusion (e.g., skip a newly added section), add it in `exclusions.py` and it will take effect in both the crawler and the cleanup script.

---

### 4.3 Content Extraction (`pass1/extractor.py`)

For each **HTML page**:
1. Fetch with `requests`
2. Extract clean text with `trafilatura` (strips nav, footers, sidebars)
3. Fall back to BeautifulSoup if trafilatura returns < 50 words (some pages have structure trafilatura misses)
4. Extract all outbound links with anchor text and up to 200 chars of surrounding context
5. Extract breadcrumbs (`Home > Voter Registration > Deadlines`) — tries standard nav patterns, falls back to URL path segments
6. Compute SHA256 `content_hash` of the extracted text (used for deduplication and change detection)

For **documents (PDF, DOCX, XLS)**:
- HEAD request only — get file size without downloading
- PDFs additionally: download the first 4,096 bytes and check for text stream markers (`/Font`, `/Text`, `Tj`, `TJ`). If none found, flag `needs_ocr = true`

**Why defer full document extraction to Pass 2?** Pass 1 is a discovery pass. Downloading and processing hundreds of potentially large PDFs during the crawl would slow discovery significantly and conflate two concerns. Pass 2 handles extraction on demand.

---

### 4.4 Classification (`pass1/classifier.py`)

Each page gets a `page_classification` label and a `chunking_strategy`. Pass 2 uses these to decide how to split the page.

**Why rule-based, not ML?** The elections website has consistent, predictable structure. Rules are transparent and auditable — you can see exactly why a page was classified a certain way — and easy to fix without retraining anything.

**Classification hierarchy (checked in order):**

```
Is the URL a document file (.pdf, .docx, .xls)?
  → class = document, strategy = document_extraction

Is it a Cloudflare cdn-cgi stub?
  → class = skip

Does the URL or content signal FAQ?
  → class = faq, strategy = qa_pairs

Does the URL or content signal a press release?
  → class = press_release
         ≥150 words → strategy = simple_split
         <150 words → strategy = ingest_as_single

Is the page ≥500 words with no strong other signal?
  → class = prose, strategy = semantic_with_overlap

Does the URL or structure signal tabular data (election results)?
  → class = table_data, strategy = table_rows

Does the URL signal an online form?
  → class = form
         ≥150 words → strategy = simple_split
         <150 words → strategy = ingest_as_single

Does the URL signal a contact/short info page?
  → class = short_static, strategy = ingest_as_single

Is the page 150–499 words and link-heavy (hub page)?
  → class = nav_hub, strategy = ingest_as_single

Fallback:
  → class = short_static, strategy = ingest_as_single
```

Each classification gets a **confidence level**: `high` (URL/structural match), `medium` (structural heuristic), `low` (word-count fallback).

**Why is there a `reclassify.py` script?** As we refined the rules post-crawl (e.g., the `register`/`registration` signal over-classified pages as `form` in the elections domain), we needed to re-apply the improved classifier to all 1,347 rows without re-crawling. `reclassify.py` re-runs classification against stored metadata.

---

## 5. Pass 2 — Chunking

**Goal:** Re-fetch and fully extract each page, then split into chunks sized for embedding.

**Run it:**
```bash
python -m maryland_rag pass2
# Only re-process pages whose content changed:
python -m maryland_rag pass2 --changed
# Specify output path:
python -m maryland_rag pass2 --output data/chunks.jsonl
```

**Output:** `data/chunks.jsonl` — one JSON record per chunk.

---

### 5.1 Chunk Sizing Rationale

Different page types warrant different chunking approaches:
- A FAQ answer is already a natural retrieval unit — each Q&A pair as its own chunk means a query retrieves exactly the right answer, not a mixed page of many answers.
- A long policy document needs splitting, but cutting at fixed word counts splits sentences and loses context at boundaries. Sentence-aware splitting with overlap preserves coherence.
- A table of election results is best as one-row-per-chunk — each row is a self-contained fact that can be retrieved independently.

---

### 5.2 The HTTP Cache (`pass2/cache.py`)

Pass 2 re-fetches HTML to get full content (Pass 1 saved only a 500-char snippet). To avoid re-hitting Maryland's servers on every development run, all fetched pages are cached to `data/cache/` keyed by SHA256(url), stored as `.html` or `.bin` files. Cache hits skip the network entirely.

Disk cache (not in-memory) because Pass 2 runs can be long and interrupted — a persistent cache survives restarts.

---

### 5.3 Deduplication (`pass2/chunker.py`)

Pages with identical `content_hash` values (same content under different URLs) are extracted once. The resulting chunks carry `source_urls` (plural) listing all URLs for that content, rather than a single `source_url`. This avoids uploading duplicate vectors to Pinecone while preserving full provenance.

---

### 5.4 Chunk Metadata

Every record in `chunks.jsonl`:

| Field | Description |
|---|---|
| `chunk_id` | UUID — unique identifier for this chunk |
| `source_url` | URL this chunk came from |
| `title` | Page title |
| `section_hierarchy` | Breadcrumb array (e.g., `["Home", "Voter Registration"]`) |
| `page_classification` | Page type (`faq`, `prose`, `table_data`, etc.) |
| `chunking_strategy` | How it was split |
| `chunk_index` | 0-based position within the page |
| `chunk_total` | Total chunks from this page |
| `word_count` | Word count of this chunk |
| `text` | The chunk text |
| `date_extracted` | ISO timestamp of when this was processed |

This metadata is stored alongside the vector in Pinecone so retrieved chunks can be cited with source, title, and page location.

---

## 6. Pass 3 — Embedding & Pinecone Upload

**Goal:** Embed every chunk with OpenAI and upsert to Pinecone.

**Run it:**
```bash
python -m maryland_rag pass3
# Resume an interrupted upload:
python -m maryland_rag pass3 --resume
# Use a different input file:
python -m maryland_rag pass3 --chunks data/my_chunks.jsonl
```

---

### 6.1 How It Works (`pass3/embed.py`)

1. Read chunks from `chunks.jsonl`
2. If `--resume`: skip chunk IDs already listed in `data/chunks.checkpoint`
3. Send texts to OpenAI in batches of 100 → `text-embedding-3-small` → 1,536-dim vectors
4. Sanitize metadata for Pinecone (only accepts `str`, `int`, `float`, `bool`, `list[str]` — nested dicts are JSON-encoded to string)
5. Upsert vectors to Pinecone in batches of 100
6. Append successfully upserted chunk IDs to `chunks.checkpoint`

Retry logic: 3 attempts with 5s exponential backoff on API failures.

---

### 6.2 Pinecone Index Configuration

| Setting | Value | Reason |
|---|---|---|
| Dimension | 1,536 | Matches `text-embedding-3-small` output |
| Metric | Cosine | Standard for normalized text embeddings |
| Cloud | AWS | Default; set via `PINECONE_CLOUD` |
| Region | us-east-1 | Default; set via `PINECONE_REGION` |

---

### 6.3 The Checkpoint System

`data/chunks.checkpoint` — one `chunk_id` per line for every chunk successfully upserted.

If Pass 3 is interrupted, `--resume` reads the checkpoint and skips already-uploaded chunks. Without it, a failed run would either re-upload duplicates or require starting entirely from scratch.

---

## 7. Server — FastAPI Chatbot

**Goal:** Serve a conversational RAG chatbot over HTTP, backed by the Pinecone index populated by Pass 3. Includes guardrail middleware for PII protection, query classification, and partisan-response prevention.

**Run it:**
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a response |
| `POST` | `/reset` | Clear conversation history for a user |
| `GET` | `/health` | Health check (returns model + index name) |

### How It Works

```
User query
        │
        ▼
┌───────────────────────────────┐
│  Guard 1: Input PII Detection │
│  Presidio scans for SSN,      │
│  credit card, phone, etc.     │
│  → Block if PII found         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Guard 2: Query Classification│
│  LLM classifies as normal,    │
│  out_of_scope, or partisan    │
│  → Block if not normal        │
└───────────────┬───────────────┘
                │  normal query + chat history
                ▼
┌───────────────────────────────┐
│  Stage 1: Contextualize       │
│  If history exists, LLM       │
│  rephrases the follow-up      │
│  into a standalone question   │
└───────────────┬───────────────┘
                │  standalone question
                ▼
┌───────────────────────────────┐
│  Stage 2: Retrieve            │
│  Embed question → query       │
│  Pinecone → top-k chunks      │
└───────────────┬───────────────┘
                │  context + history + query
                ▼
┌───────────────────────────────┐
│  Stage 3: Answer              │
│  LLM generates a grounded     │
│  response using context        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Guard 3: Partisan Check      │
│  LLM verifies the response    │
│  is nonpartisan. If not,       │
│  retries with stricter prompt  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Guard 4: Output PII Scrub    │
│  Presidio re-scans the LLM    │
│  response before returning     │
└───────────────┬───────────────┘
                │
                ▼
        Response + session update + request log
```

### Server Modules

**`main.py`** — FastAPI app with async lifespan startup (logging, `SessionStore`, `build_chain()`). The `/chat` handler orchestrates the full guardrail + RAG pipeline. On RAG failure, returns HTTP 502.

**`rag_chain.py`** — Built with `langchain_core` runnables — no `langchain-pinecone` dependency. A custom `PineconeRetriever(BaseRetriever)` queries the Pinecone SDK directly and logs retrieval scores. The chain is `RunnableLambda(contextualize_and_retrieve) | qa_prompt | llm | StrOutputParser()`. The contextualization prompt instructs the LLM to reformulate follow-ups into standalone questions without answering them. The QA prompt grounds the LLM in the retrieved context and tells it to say when it doesn't have enough information rather than guessing.

**`middleware.py`** — All guardrail logic:

| Function | Purpose | Failure mode |
|---|---|---|
| `detect_pii(query, ctx)` | Scans user input for PII (SSN, credit card, email, phone, passport, driver's license) via Presidio | Hard block — returns canned response |
| `classify_query(query, ctx)` | LLM-based structured classification (`normal` / `out_of_scope` / `partisan`) using `gpt-4o-mini` with structured output | Fail open — on error, allows query through |
| `check_partisan_response(query, response, ...)` | LLM-based structured check for partisan bias in the generated answer; if flagged, retries the RAG chain with a stricter nonpartisan prompt appended | Fail open — on error, returns original response |
| `detect_pii_in_response(response, ctx)` | Re-scans LLM output for PII before returning to user | Hard block — replaces response with fallback |

The classifier and partisan checker use `gpt-4o-mini` with Pydantic structured output (`ClassificationResult`, `PartisanCheckResult`), independent of the main `LLM_MODEL` setting.

**`rag_logger.py`** — LangChain `BaseCallbackHandler` that tracks LLM token usage, estimates cost per request (using a built-in cost table for OpenAI models), and logs retriever timing/doc counts. Optional prompt/response/query logging controlled by module-level flags (`LOG_PROMPTS`, `LOG_RESPONSES`, `LOG_QUERIES` — all `False` by default). Also provides `log_request()` for per-request summary logging.

**`session.py`** — In-memory per-`user_id` conversation store with TTL expiration and max turn limit. Thread-safe via `threading.Lock`. Designed for pilot-scale (tens of concurrent users) — swap to Redis or a database for production scale.

### Configuration

All settings via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `PINECONE_API_KEY` | (required) | Pinecone API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `PINECONE_INDEX_NAME` | `maryland-elections` | Pinecone index name |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for RAG answers |
| `LLM_TEMPERATURE` | `0.2` | Model temperature |
| `RETRIEVER_K` | `5` | Number of chunks to retrieve |
| `SESSION_TTL_MINUTES` | `30` | Session expiration |
| `MAX_HISTORY_TURNS` | `20` | Max conversation turns kept |

---

## 8. Utility Scripts

### 8.1 Database Cleanup (`scripts/db_cleanup.py`)

After the initial crawl, `manifest.db` had ~5,700 rows. This script reduced it to the canonical 1,347 by removing (in order):

1. All `http://` URLs — redirects to `https://`, pure duplicates
2. All `https://www.elections.maryland.gov/` URLs — the canonical domain omits `www.`
3. Cloudflare `cdn-cgi` stub pages — auto-generated email-obfuscation stubs, no real content
4. `businessdisclosure` subdomain pages — out of scope
5. All remaining `failed` rows — spaces-in-filename PDFs, mailto fragments, dead weight

After deletions, orphaned `links` rows (where both source and target no longer exist in `pages`) are cleaned up and the database is VACUUMed. Always creates a timestamped backup before modifying the database.

```bash
python -m maryland_rag.scripts.db_cleanup --dry-run   # preview
python -m maryland_rag.scripts.db_cleanup              # apply
```

---

### 8.2 Re-classification (`scripts/reclassify.py`)

Re-applies updated classification rules to all rows in `manifest.db` without re-crawling. Use this whenever you change `classifier.py`.

Key changes made vs. the original Pass 1 classification:
- Dropped `register`/`registration` as form signals (too broad for the elections domain — flagged too many non-form pages)
- Added `nav_hub` class for link-heavy pages (150–499 words) that are navigation, not content
- Activated `semantic_with_overlap` for prose (was dead code in the original classifier)
- Strips site-wide announcement banners (e.g., *"The Worcester County Board of Elections announces..."*) from snippets before classification — these were shifting word counts and confusing signal detection

```bash
python -m maryland_rag.scripts.reclassify --dry-run   # preview
python -m maryland_rag.scripts.reclassify              # apply
```

---

## 9. Running the Full Pipeline

### First-time setup

```bash
# 1. Install dependencies
pip install -r maryland_rag/requirements.txt
pip install -r server/requirements.txt
python -m spacy download en_core_web_lg

# 2. Create .env with API keys (see Section 3.3)

# 3. Crawl (~30-60 min for full site)
python -m maryland_rag pass1

# 4. Clean up duplicates (recommended after a fresh crawl)
python -m maryland_rag.scripts.db_cleanup

# 5. Chunk (~10-20 min depending on PDF count)
python -m maryland_rag pass2 --output data/chunks.jsonl

# 6. Embed and upload (~5-15 min depending on API rate limits)
python -m maryland_rag pass3 --chunks data/chunks.jsonl

# 7. Verify at console.pinecone.io → maryland-elections index

# 8. Start the server
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Audit the database

```bash
python -m maryland_rag audit
```

Prints: classification breakdown, exclusion reasons, depth distribution, failed pages, duplicate content, top pages by inbound links, exclusion leak check.

### Re-running after website updates

```bash
# Re-crawl (resumes, detects changed content hashes)
python -m maryland_rag pass1

# Re-chunk only changed pages
python -m maryland_rag pass2 --changed --output data/chunks.jsonl

# Resume upload (skip already-upserted chunks)
python -m maryland_rag pass3 --resume
```

---

## 10. The Database (manifest.db)

Inspect directly with the SQLite CLI:

```bash
sqlite3 data/manifest.db
.tables
.schema pages
SELECT COUNT(*) FROM pages;
SELECT page_classification, COUNT(*) FROM pages GROUP BY page_classification;
.quit
```

### Tables

**`pages`** — one row per discovered URL:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `url` | TEXT | Full URL (unique) |
| `parent_url` | TEXT | Which page linked here |
| `title` | TEXT | `<title>` tag content |
| `section_hierarchy` | JSON | Breadcrumb trail as array |
| `content_type` | TEXT | `html`, `pdf`, `docx`, `xls`, `csv` |
| `page_classification` | TEXT | Classification label |
| `chunking_strategy` | TEXT | Which Pass 2 strategy to use |
| `classification_confidence` | TEXT | `high`, `medium`, or `low` |
| `word_count` | INTEGER | Word count of extracted text |
| `depth` | INTEGER | Crawl depth from seed URL |
| `crawl_status` | TEXT | `pending`, `crawled`, `failed`, `skipped`, `excluded` |
| `exclusion_reason` | TEXT | Why excluded (if applicable) |
| `http_status` | INTEGER | HTTP response code |
| `content_hash` | TEXT | SHA256 of extracted text |
| `file_size_bytes` | INTEGER | File size (documents) |
| `needs_ocr` | BOOLEAN | True if PDF has no text layer |
| `extracted_snippet` | TEXT | First ~500 chars of content |
| `links_out_count` | INTEGER | Outbound link count |
| `discovered_at` | TIMESTAMP | When URL was first found |
| `crawled_at` | TIMESTAMP | When it was fetched and processed |

**`links`** — one row per hyperlink:

| Column | Description |
|---|---|
| `source_url` | Page containing the link |
| `target_url` | Link destination |
| `link_text` | Anchor text |
| `link_context` | Surrounding text (up to 200 chars) |
| `is_internal` | True if on elections.maryland.gov |
| `is_document` | True if target is PDF, DOCX, etc. |

**`crawl_runs`** — one row per Pass 1 run:

| Column | Description |
|---|---|
| `started_at`, `completed_at` | Timing |
| `total_discovered`, `total_crawled` | Run statistics |
| `notes` | Manual notes |

---

## 11. Chunking Strategies — Deep Dive

### `ingest_as_single` ([pass2/strategies/single.py](maryland_rag/pass2/strategies/single.py))
**Used for:** `short_static`, `nav_hub`, small forms, small press releases

Returns the entire page as a single chunk. Used when the page is short enough (< ~150 words) that splitting would only fragment information, or when the page is a nav hub whose value is the full list of links.

---

### `simple_split` ([pass2/strategies/simple_split.py](maryland_rag/pass2/strategies/simple_split.py))
**Used for:** `press_release` (≥150w), `form` (≥150w)

- Target: ~250 words/chunk
- Splits at double newlines (paragraph boundaries)
- Merges trailing fragments < 30 words into the previous chunk
- No overlap

Press releases and form descriptions are linear prose — paragraph-boundary splitting respects the natural structure without the overhead of sentence analysis.

---

### `semantic_with_overlap` ([pass2/strategies/semantic.py](maryland_rag/pass2/strategies/semantic.py))
**Used for:** `prose` pages ≥ 500 words

- Target: ~300 words/chunk, max 500
- Splits at sentence boundaries (`(?<=[.!?])\s+(?=[A-Z])`)
- 20% overlap between consecutive chunks

**Why overlap?** Long prose explanations often carry context across sentence boundaries. Without overlap, a chunk might start mid-explanation. The 20% overlap ensures each chunk includes the closing sentences of the previous one, so context is never entirely cut off at a boundary.

**Why sentence boundaries?** Fixed word-count splits cut sentences mid-stream, degrading embedding quality and producing poor search snippets.

---

### `qa_pairs` ([pass2/strategies/faq.py](maryland_rag/pass2/strategies/faq.py))
**Used for:** `faq` pages

Parses HTML for Q&A pairs. Detection methods tried in order:
1. `<dl><dt>Question</dt><dd>Answer</dd></dl>`
2. `<details><summary>Question</summary>Answer</details>`
3. Heading patterns — `<h2>Question?</h2>` + following paragraph content
4. Bold/strong patterns — `<strong>Question?</strong>` + following text

Each Q&A pair is its own chunk. The `question` and `answer` fields are preserved in chunk metadata.

FAQ pages are the highest-value content for RAG — they're already question-answer structured. Arbitrary splitting would likely separate a question from its answer.

---

### `table_rows` ([pass2/strategies/table_rows.py](maryland_rag/pass2/strategies/table_rows.py))
**Used for:** `table_data` pages (election results, district lists)

- Parses all `<table>` elements
- Extracts `<th>` headers
- Each data row → one chunk: `"Column1: Value1 | Column2: Value2 | ..."`
- Prepends table caption if present

Embedding an entire results table as one vector makes every row equally retrievable — which is too coarse. One-row-per-chunk means a query for a specific county or district retrieves exactly that row.

---

### PDF extraction ([pass2/strategies/pdf.py](maryland_rag/pass2/strategies/pdf.py))
**Used for:** all `.pdf` files

Three-tier extraction:
1. **pdfplumber** — primary, handles clean digital PDFs well
2. **PyMuPDF (fitz)** — fallback for complex layouts or mixed-column formats pdfplumber struggles with
3. **OCR via pytesseract** — for `needs_ocr = true` PDFs flagged in Pass 1; PyMuPDF renders pages as images, Tesseract reads the text

After extraction, the PDF text is analyzed for structure and internally routed to `semantic_with_overlap`, `qa_pairs`, or `ingest_as_single`.

---

### DOCX extraction ([pass2/strategies/docx_strategy.py](maryland_rag/pass2/strategies/docx_strategy.py))
**Used for:** `.docx` Word files

- Walks heading hierarchy (Heading 1 → 2 → 3...)
- Each section (heading chain + body content) is a candidate chunk
- Sections ≤ 300 words: one chunk
- Sections > 300 words: further split with the semantic strategy

Heading hierarchy is preserved in each chunk so a retrieved chunk always carries its section context (e.g., `"2024 Results > Carroll County"`), making it self-contained without needing to know which document it came from.

---

## 12. Configuration Reference

`pass1/config.py`:

| Constant | Default | Description |
|---|---|---|
| `SEED_URL` | `https://elections.maryland.gov` | Crawl starting point |
| `MAX_DEPTH` | `6` | Max BFS depth |
| `RATE_LIMIT_SECONDS` | `0.75` | Delay between requests |
| `PDF_PROBE_BYTES` | `4096` | Bytes checked for PDF text markers |
| `TRAFILATURA_MIN_WORDS` | `50` | Min words for trafilatura to be trusted |
| `DOCUMENT_EXTENSIONS` | `.pdf .docx .doc .xls .xlsx .csv` | Treated as documents, not HTML |
| `SKIP_DOMAINS` | Facebook, Twitter, YouTube, etc. | External domains to skip |

---

## 13. Security Warning

**The `.env` file contains API keys for paid services.**

- **Never commit `.env` to Git.**
- Verify `.gitignore` excludes it before pushing.
- If keys have been exposed, rotate immediately:
  - OpenAI: https://platform.openai.com/api-keys
  - Pinecone: https://console.pinecone.io → API Keys

---

## 14. Troubleshooting

**Pass 1 stops unexpectedly**
Fully resumable — just rerun `python -m maryland_rag pass1`. It picks up from `pending` rows.

**Pass 2 fails on a specific PDF**
All three extraction tiers are tried before failing. If all fail, the error is logged and the batch continues. Check the URL manually — the PDF may be password-protected or corrupted.

**Pass 3: "Dimension mismatch" from Pinecone**
The Pinecone index was created with a different vector dimension than 1,536. Delete the index in the console and let Pass 3 recreate it, or adjust `EMBEDDING_DIM` in `embed.py`.

**Pass 3: Persistent rate limit errors from OpenAI**
The code already batches and retries. If limits persist, reduce `EMBED_BATCH_SIZE` in `embed.py`.

**Wrong classifications in audit output**
Run `python -m maryland_rag.scripts.reclassify --dry-run` to preview current rules. After updating `classifier.py`, run `reclassify.py` to apply.

**Database locked error**
A previous run didn't exit cleanly. Kill any running `python -m maryland_rag` processes and retry.

**Server won't start**
Check that `OPENAI_API_KEY` and `PINECONE_API_KEY` are set in `.env` at the project root. Also verify the spaCy model is installed (`python -m spacy download en_core_web_lg`) — Presidio's `AnalyzerEngine` loads it at import time.

---

## 15. Glossary

Project-specific terms and non-obvious library names only.

| Term | Definition |
|---|---|
| **manifest.db** | The SQLite database produced by Pass 1. One row per discovered URL with all classification and metadata. |
| **chunks.jsonl** | JSONL file (one JSON object per line) produced by Pass 2. Each line is one chunk with text and full metadata. |
| **chunks.checkpoint** | Plain text file listing `chunk_id`s that have been successfully upserted to Pinecone. Enables `--resume`. |
| **content_hash** | SHA256 of a page's extracted text. Two pages with the same hash have identical content and are deduplicated in Pass 2. |
| `needs_ocr` | Flag on PDFs where no text layer was detected in the first 4KB — meaning the PDF is a scanned image. |
| **trafilatura** | Library that extracts clean article text from HTML, removing nav, footers, and boilerplate. Primary extractor in Pass 1. |
| **pdfplumber** | Library for extracting text and tables from digital (non-scanned) PDFs. Primary PDF extractor in Pass 2. |
| **nav_hub** | Our classification for pages that are 150-499 words and primarily consist of links — they're navigation, not content. |
| **Pass** | One stage of the pipeline. Pass 1 = crawl, Pass 2 = chunk, Pass 3 = embed and upload. |
| **RAG chain** | The LangChain runnable pipeline in the server: rephrase → retrieve → answer. Built from `langchain_core` primitives. |
| **PineconeRetriever** | Custom `BaseRetriever` subclass in `rag_chain.py` that embeds queries with OpenAI and queries Pinecone directly (no `langchain-pinecone`). |
| **SessionStore** | In-memory conversation store in `session.py`. Tracks chat history per user with TTL expiration and max turn limits. |
| **Presidio** | Microsoft's PII detection engine, used in `middleware.py` to scan both user input and LLM output for sensitive data (SSN, credit card, phone, etc.). |
| **QueryContext** | Dataclass in `middleware.py` that tracks per-request guardrail state: classification result, PII detection flags, and safety status. |
| **RAGCallbackHandler** | LangChain callback handler in `rag_logger.py` that captures token usage, estimates cost, and logs retriever performance per request. |
| **Structured output** | LangChain/OpenAI feature used by the guardrail LLMs — returns Pydantic models (`ClassificationResult`, `PartisanCheckResult`) instead of free-form text. |
| **Fail open** | Guardrail error-handling strategy: if the classifier or partisan checker LLM call fails, the query is allowed through rather than blocked. Prevents guardrail outages from taking down the chatbot. |
