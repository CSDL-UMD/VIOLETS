# VIOLETS Codebase Audit

## Context

The VIOLETS chatbot has been built rapidly: a FastAPI `/chat` endpoint over a four-stage guardrail pipeline (PII → classify → RAG over pgvector → partisan check) plus two ingestion pipelines (`maryland_rag` for crawled HTML/PDF/XLS sources and `box_ingest` for Box-hosted SBE materials). Total surface area is ~6,300 lines of Python across three subsystems. The user — preparing for production — asked for a senior-level audit that surfaces correctness, security, performance, and reliability risks before the system is exposed to real participants.

This document is the audit deliverable. Findings are organized by severity, each anchored to a specific file/line with a one-line rationale and one-line fix. Cross-cutting themes are summarized at the end with a recommended remediation order. No code changes are proposed in this plan — apply fixes in a follow-up once the user prioritizes.

---

## Audit Scope

| Subsystem | Files | Lines |
|---|---|---|
| FastAPI server | [server/](server/) — `main.py`, `middleware.py`, `rag_chain.py`, `rag_logger.py`, `session.py`, `config.py` | 1,520 |
| RAG pipeline | [maryland_rag/](maryland_rag/) — `pass1/`, `pass2/`, `pass3/`, `scripts/` | 3,100 |
| Box ingestion | [box_ingest/](box_ingest/) — `crawler.py`, `automate.py`, `ingest.py`, `manifest.py`, `filter.py` | 880 |

Audit was read-only: every listed file was read in full and findings were spot-verified against the source.

---

## CRITICAL

These can hang the server under load, lose data, or expose unauthenticated write paths.

1. **Sync Presidio call blocks the event loop** — [server/middleware.py:182-217](server/middleware.py#L182-L217). `detect_pii()` is invoked from the async `/chat` handler ([server/main.py:243](server/main.py#L243)) but calls `_analyzer.analyze()` synchronously; Presidio analysis of a 2 k-char query can take 100–400 ms. While it runs, the FastAPI event loop is frozen and no other request makes progress. **Fix:** wrap in `await asyncio.to_thread(_analyzer.analyze, ...)`.

2. **Sync embedding + pgvector query block the event loop** — [server/rag_chain.py:113-125](server/rag_chain.py#L113-L125). `embed_query()` is a sync HTTP call to OpenAI, and the `with self.pool.connection()` block is sync. The code comment at lines 144-147 already flags this but it has not been fixed. Under load, every retrieval ties up one threadpool worker; the default starlette threadpool is 40, so >40 concurrent users will queue. **Fix:** implement `_aget_relevant_documents` with `AsyncOpenAI` + `psycopg_pool.AsyncConnectionPool`.

3. **No timeouts on any external LLM/DB call** — [server/main.py:279](server/main.py#L279) (`chain.ainvoke`), [server/middleware.py:329](server/middleware.py#L329) (classifier), [server/middleware.py:452](server/middleware.py#L452) and [475](server/middleware.py#L475) (partisan checker + retry), [server/rag_chain.py:115-125](server/rag_chain.py#L115-L125) (pgvector), and every `urllib.request.urlopen` in `box_ingest/crawler.py` ([74](box_ingest/crawler.py#L74), [84](box_ingest/crawler.py#L84), [130](box_ingest/crawler.py#L130)) and [automate.py:80](box_ingest/automate.py#L80). A hung OpenAI or Box socket will deadlock the request indefinitely. **Fix:** wrap async LLM calls in `asyncio.wait_for(..., timeout=N)`; pass `timeout=` to every `urllib.request.urlopen`; `SET LOCAL statement_timeout = '30s'` on pgvector queries.

4. **Path traversal in Box ingestion** — [box_ingest/automate.py:43-45](box_ingest/automate.py#L43-L45) and [box_ingest/crawler.py:212](box_ingest/crawler.py#L212). `_local_path` builds the on-disk filename from `box_file.folder_path / box_file.name` with no sanitization, and `_walk_folder` builds folder paths the same way. A Box folder or filename containing `..` or starting with `/` writes outside `NEEDTOCHUNK_DIR`. The Box account is shared and externally writable. **Fix:** after joining, resolve and assert the result is inside `NEEDTOCHUNK_DIR.resolve()`; reject names containing path separators or `..`.

5. **Unbounded in-memory rate limiter** — [server/main.py:85-106](server/main.py#L85-L106). `_RateLimiter._requests` is a `defaultdict(list)` that never evicts user keys; over weeks of operation the dict grows monotonically. Also pure-Python, so it doesn't share state across worker processes if uvicorn is run with `--workers >1`. **Fix:** prune entries whose newest timestamp is older than `_window` on each `check()`; if scaling out, move to Redis.

6. **In-memory session store, same problems** — [server/session.py](server/session.py) (whole file). The session store is a process-local `dict`; multi-worker uvicorn loses sessions on a different worker, and a process restart wipes all history. The TTL cleanup only runs on the lucky-request path or the 5-minute periodic task ([server/main.py:127-132](server/main.py#L127-L132)). **Fix:** pin uvicorn to `--workers 1` or move sessions into Postgres/Redis; document the constraint.

7. **Pagination off-by-one in Box folder listing** — [box_ingest/crawler.py:142](box_ingest/crawler.py#L142). The loop breaks when `offset + limit >= total_count`. When `total_count == limit` (e.g., 1000 items, limit 1000), it correctly exits on the first page. But when `total_count == 2*limit`, the second iteration sets `offset=1000`, checks `1000+1000 >= 2000` → True, and exits before any check on the *returned* `entries` length. If Box ever returns fewer than `limit` entries before `total_count` is reached (race with concurrent edits), files are silently dropped. **Fix:** loop while `len(items) < total_count and last_page_was_nonempty`; or paginate by marker.

8. **Manifest written non-atomically** — [box_ingest/manifest.py:63-64](box_ingest/manifest.py#L63-L64). `json.dump(MANIFEST_PATH, ...)` writes in place; a Ctrl-C mid-write truncates the manifest. (Note `ingest.py:_save_state` does it correctly with tmp+rename — divergent.) **Fix:** write to `MANIFEST_PATH.with_suffix(".json.tmp")` then `Path.replace()`.

9. **OAuth tokens stored world-readable** — [box_ingest/crawler.py:52-53](box_ingest/crawler.py#L52-L53). `_save_tokens` writes Box access + refresh tokens to `.box_token` without restricting mode bits. On a shared host any user can lift the refresh token and impersonate the app indefinitely. **Fix:** `os.chmod(TOKEN_CACHE, 0o600)` after every write; create the file with `os.open(..., O_CREAT|O_WRONLY|O_TRUNC, 0o600)`.

10. **Globals dereferenced without nil-check** — [server/main.py:270-271, 317](server/main.py#L270-L271). `store.get_or_create` and `store.reset` are called on globals that are `None` until lifespan startup completes. If a request arrives during the startup race window (or after a failed lifespan that didn't propagate), it crashes with `AttributeError`. **Fix:** either remove the `| None` type and assign at module load, or guard each handler with an explicit check + 503.

---

## HIGH

### Correctness

11. **Manifest update reverses success ordering** — [box_ingest/ingest.py:293-303](box_ingest/ingest.py#L293-L303). State is saved *before* the JSONL is written. If disk fills between line 293 and line 303, the state claims success but the chunks file is missing/partial; the next run treats every file as cached and never recovers. **Fix:** write JSONL first, fsync, then save state.

12. **Failed extractions force re-OCR every run** — [box_ingest/ingest.py:262-291](box_ingest/ingest.py#L262-L291). When `_process_file` raises, the file is skipped without updating state. Next run hits the same fingerprint mismatch and re-runs OCR — expensive and often deterministic. **Fix:** record failures in state with `{"chunks": [], "error": str(exc)}` and a fingerprint, so retries only happen when the input changes.

13. **Partisan checker re-invokes the chain without partisan-category guard** — [server/middleware.py:475-488](server/middleware.py#L475-L488). The retry calls `chain.ainvoke` passing whatever `ctx.query_category` was set to. If classification said `conversational`, the retry runs the conversational prompt (no retrieval) but with the "strict nonpartisan" append. The strict instructions land in a chat-history-only flow which has no factual context to back up the answer. **Fix:** force `query_category="normal"` (or a dedicated `"strict_nonpartisan"`) on retries.

14. **Strict nonpartisan retry pollutes chat history on next turn** — [server/main.py:310](server/main.py#L310). `store.add_exchange` saves the *user input* `req.query` and the final `answer`. But the retry chain saw a `query + STRICT_PROMPT` augmented input. The history persisted to the session is `(query, answer)` where `answer` is from the augmented prompt — fine. But because the retry re-rephrased, future turns may treat the prior turn's answer as something that responded to the *literal* query. Minor; document or pass the original `input` shape. **Fix:** add a comment, or store the augmented form for fidelity.

15. **CORS origins not whitespace-stripped** — [server/main.py:192-194](server/main.py#L192-L194). `os.environ.get("CORS_ORIGINS", "...").split(",")` leaves spaces around each origin. Anyone setting `CORS_ORIGINS="https://a.com, https://b.com"` silently breaks CORS for `b.com`. **Fix:** `[o.strip() for o in raw.split(",") if o.strip()]`.

16. **Box token refresh swallows all errors and re-launches a browser** — [box_ingest/crawler.py:99-105](box_ingest/crawler.py#L99-L105). A transient 5xx or network blip during refresh discards the cached refresh token and opens a browser. Running on a headless box, this hangs waiting for a redirect that never comes. **Fix:** distinguish transient (`URLError`, 5xx) and retry with backoff vs. permanent (401) and re-auth.

17. **Robots-parser keyed by netloc with port** — [maryland_rag/pass1/crawler.py:150-155](maryland_rag/pass1/crawler.py#L150-L155). A site reached on multiple ports gets multiple cached parsers and bypasses checks if any one fails to load. **Fix:** key by `urlparse(url).hostname` (no port).

18. **Semantic chunker overlap math drops sentences** — [maryland_rag/pass2/strategies/semantic.py:77-80](maryland_rag/pass2/strategies/semantic.py#L77-L80). The overlap loop breaks on `overlap_words + s_words > overlap_target`; when the first candidate sentence already exceeds the target, overlap is 0. With 60-word target and a 60-word first sentence, you get zero overlap — boundary-context loss for short chunks. **Fix:** use `>=` and allow the inclusive sentence, or accept the first sentence regardless.

19. **FAQ strategy double-fetches** — [maryland_rag/pass2/chunker.py:142-148](maryland_rag/pass2/chunker.py#L142-L148). On `extract_qa_pairs([])` fallback, `_fetch_text(url)` re-downloads the same URL. Doubles crawl latency for every FAQ that doesn't match the QA heuristic. **Fix:** cache the fetched HTML at the call site and pass into both functions.

20. **Hard-coded EMBED_DIM in DDL** — [maryland_rag/pass3/embed.py:150](maryland_rag/pass3/embed.py#L150). `vector(1536)` is baked into table creation. If you ever switch to `text-embedding-3-large` (3072-dim) the create-if-not-exists silently keeps the wrong column type. **Fix:** at startup, query `information_schema` for the existing dim and `ASSERT` it matches `EMBED_DIM`, or fail loudly.

21. **Sync embedding loop in pass3 not parallel** — [maryland_rag/pass3/embed.py:87-125](maryland_rag/pass3/embed.py#L87-L125). One OpenAI call at a time, one DB insert at a time. Embedding 1,347 documents serially adds minutes. **Fix:** `concurrent.futures.ThreadPoolExecutor` over `EMBED_BATCH_SIZE` batches; the DB insert can stay sequential.

22. **Pass3 connection has no statement timeout** — [maryland_rag/pass3/embed.py:146](maryland_rag/pass3/embed.py#L146). A stuck index build or hung session can wedge the pipeline forever. **Fix:** `conn.execute("SET statement_timeout = '60s'")` right after `register_vector`.

23. **Connection-pool open can fail silently** — [server/main.py:158-165](server/main.py#L158-L165). `_pool.open()` without `wait=True` doesn't block until a connection is verified. The first request can hit a not-ready pool. **Fix:** `_pool.open(wait=True, timeout=10)`.

24. **`add_exchange` failure silently loses history** — [server/main.py:310](server/main.py#L310). Wrapped in nothing; if the session store raises, the response is already sent so the error vanishes. Low-impact alone, but combined with the in-memory store there is no second source of truth. **Fix:** `try/except` with `logger.exception(...)`; consider a write-ahead in Postgres if persistence matters.

### Security

25. **Hardcoded Box shared-link token in source** — [box_ingest/crawler.py:42-44](box_ingest/crawler.py#L42-L44). `SHARED_LINK_TOKEN` is committed to the repo. It's a public link by design, but committing it conflates "secret-bearing token" with "public URL component" and makes rotation a code change. **Fix:** move to `.env`; document its public/non-secret nature in a comment.

26. **`classify_query` fails open** — [server/middleware.py:353-359](server/middleware.py#L353-L359). On any LLM exception, the function returns `None` and the query proceeds to the RAG chain unfiltered. That includes partisan queries that should be blocked. **Fix:** fail closed — return a generic "couldn't process your question, please rephrase" message when classification errors.

27. **`check_partisan_response` fails open** — [server/middleware.py:500-508](server/middleware.py#L500-L508). On checker error, the unchecked LLM output is returned. **Fix:** on exception, return a safer canned fallback (`FALLBACK_RESPONSES["partisan"]`) for the user rather than the raw model output.

28. **No `LOG_PROMPTS`/`LOG_RESPONSES` startup warning** — [server/rag_logger.py:39-41](server/rag_logger.py#L39-L41). Flags default to log-everything. Without an explicit warning, you ship to prod with PII-bearing queries in logs. **Fix:** at lifespan startup, `if LOG_PROMPTS or LOG_RESPONSES: logger.warning("Verbose logging enabled — disable for production.")`.

29. **Box token leak risk in `auth_url` printed to stdout** — [box_ingest/crawler.py:106-108](box_ingest/crawler.py#L106-L108). The `client_id` (not secret) is fine, but the printed URL ends up in shell history and any tee'd log. Low risk, but worth scrubbing. **Fix:** print only the scheme + host; the user follows the browser, they don't need to see the full URL.

30. **No status-code check before parsing response bodies** — [box_ingest/crawler.py:77, 87, 131](box_ingest/crawler.py#L77). `urlopen` raises on HTTP 4xx/5xx by default, so parsing happens only on 2xx. But 429s and 503s pass straight up as `HTTPError` with no retry/backoff. **Fix:** explicit try/except `HTTPError`, branch on `code` for 429 (retry with `Retry-After`) vs 401 (re-auth) vs other.

31. **Path traversal in pass1 raw-HTML save** — [maryland_rag/pass1/crawler.py:224-234](maryland_rag/pass1/crawler.py#L224-L234). Filename is derived from URL path. Practically guarded by `urlparse` normalization, but the code does not assert containment. Belt-and-braces: use a content hash for the filename instead. **Fix:** filename = `hashlib.sha256(url.encode()).hexdigest()[:16] + ".html"`.

### Reliability

32. **fitz.Document not closed on exception** — [maryland_rag/pass2/strategies/pdf.py:133-138, 171-182](maryland_rag/pass2/strategies/pdf.py#L133-L138). `fitz.open()` followed by `doc.close()` without `try/finally`; OCR path holds large image memory. Long-running pass2 leaks file handles and RAM. **Fix:** `with closing(fitz.open(...))` or `try/finally`.

33. **Generic `except Exception` in pass1 crawler swallows OOM and DB errors** — [maryland_rag/pass1/crawler.py:212-214](maryland_rag/pass1/crawler.py#L212-L214). A SQLite "database is locked" error skips one page and logs at WARN; you only notice when the manifest is short. **Fix:** narrow the catch to `(requests.RequestException, ParseError)`; let SQLite/OOM propagate.

34. **Bare-except in `_fallback_extract`** — [maryland_rag/pass1/extractor.py:104-118](maryland_rag/pass1/extractor.py#L104-L118), and [_probe_pdf_text_extractable](maryland_rag/pass1/extractor.py#L256-L258). Hides every error including KeyboardInterrupt. **Fix:** `except Exception as exc: logger.exception(...)`; never bare `except:`.

35. **In-memory `chunks.jsonl` load** — [maryland_rag/pass3/embed.py:233-240](maryland_rag/pass3/embed.py#L233-L240). Whole file is read into a Python list before iteration. At current 1,347 docs this is fine; at 100k+ it OOMs. **Fix:** stream-iterate the file; only buffer one batch in memory.

36. **No retry/backoff on Box API 429s** — [box_ingest/crawler.py:124-131](box_ingest/crawler.py#L124-L131). A burst against `_api_get` will surface 429s as uncaught `HTTPError`. **Fix:** wrap with retry on 429/5xx using `Retry-After` header.

37. **Embed retry retries permanent errors** — [maryland_rag/pass3/embed.py:214-225](maryland_rag/pass3/embed.py#L214-L225). Catches all `Exception` and retries, wasting time on `AuthenticationError`/`InvalidRequestError`. **Fix:** only retry on `RateLimitError` and `APIConnectionError`.

38. **Playwright handle_request blocks forever if user closes browser** — [box_ingest/crawler.py:111](box_ingest/crawler.py#L111). `server.handle_request()` waits for a request that may never come. **Fix:** set `server.timeout = 300` and break with a clear error.

---

## MEDIUM

39. **TOCTOU in SessionStore.get_or_create** — [server/session.py:28](server/session.py#L28). `_cleanup_expired()` runs inside `get_or_create`; in concurrent access the just-cleaned session may be re-created twice. **Fix:** drop the cleanup from `get_or_create`; rely on the periodic task.

40. **Logging duplicated on hot reload** — [server/main.py:143-146](server/main.py#L143-L146). `basicConfig` is idempotent only because root has no handlers on first run; uvicorn `--reload` can leave duplicated handlers. **Fix:** clear handlers before `basicConfig`, or skip if already configured.

41. **Generic 502 hides errors** — [server/main.py:284-286](server/main.py#L284-L286). `logger.error("RAG chain error %s", exc)` — no traceback, no exception type. Debugging prod is painful. **Fix:** `logger.exception(...)`.

42. **Env vars not stripped** — [server/config.py:29-36](server/config.py#L29-L36). Trailing spaces in `.env` lines silently propagate into API keys and break `compare_digest`. **Fix:** `val.strip()` in `_require_env`.

43. **`SourceReference(**s)` raises on non-dict** — [server/main.py:289](server/main.py#L289). A chain that returns sources as something other than `list[dict]` 500s the request instead of returning a partial. **Fix:** `if not isinstance(s, dict): continue`.

44. **Cache `get_html()` returns `""` indistinguishable from real empty body** — [maryland_rag/pass2/cache.py:28-51](maryland_rag/pass2/cache.py#L28-L51). Callers can't tell "fetch failed" from "empty page". **Fix:** return `None` on failure.

45. **`simple_split` fallback can produce oversized final chunk** — [maryland_rag/pass2/strategies/simple_split.py:58-61](maryland_rag/pass2/strategies/simple_split.py#L58-L61). Final-merge logic exceeds `TARGET_CHUNK_WORDS`. **Fix:** re-split after merging.

46. **`detect_pii` empty-input not guarded** — [server/middleware.py:182-217](server/middleware.py#L182-L217). `str(query)` is fine, but a 2000-char malformed UTF-8 input (rare since Pydantic validates) would still run analyzer. Low risk; the bigger issue is the sync-call CRITICAL #1.

47. **Survey-tag match is case-sensitive and starts-with** — [server/middleware.py:322](server/middleware.py#L322). A copy-edit on the survey side ("__user concerns:__") slips past the guardrail. **Fix:** lowercased compare, document the contract in a comment.

48. **`metadata.py` chunk_id depends on URL string** — [maryland_rag/pass2/metadata.py:44-45](maryland_rag/pass2/metadata.py#L44-L45). URL canonicalization changes orphan old chunk_ids in pgvector. **Fix:** include a stable content hash; or run a re-index whenever URLs change.

49. **`user_id` regex too restrictive** — [server/main.py:205](server/main.py#L205). `^[a-zA-Z0-9_-]+$` blocks dots, common in Qualtrics PIDs. **Fix:** add `.` if the survey vendor uses it; otherwise document.

50. **Periodic cleanup task swallows its own exceptions** — [server/main.py:127-132](server/main.py#L127-L132). The task body has no `try/except`; one error stops it forever and there's no log. **Fix:** wrap body in try/except + `logger.exception`; loop continues.

51. **Box manifest load lacks JSONDecodeError handling** — [box_ingest/manifest.py:26-27](box_ingest/manifest.py#L26-L27). A corrupt manifest crashes ingest with a stack trace. **Fix:** catch `JSONDecodeError`, log, return `{}` so the next run can rebuild.

52. **Pass1 extractor doesn't `raise_for_status`** — [maryland_rag/pass1/extractor.py:43-101](maryland_rag/pass1/extractor.py#L43-L101). 5xx pages get parsed as documents. **Fix:** `resp.raise_for_status()` or explicit `if resp.status_code != 200`.

53. **Crawler "complete" log doesn't acknowledge failures** — [box_ingest/crawler.py:239-244](box_ingest/crawler.py#L239-L244). "Crawl complete: X files found" prints even when half the folders 401'd. **Fix:** track `errors` count and include it in the log.

54. **Box file download buffers full file in memory** — [box_ingest/automate.py:80-81](box_ingest/automate.py#L80-L81). `resp.read()` then `write_bytes` doubles memory for large PDFs. **Fix:** `shutil.copyfileobj(resp, fh)` with a 64 KiB buffer.

55. **`_CallbackHandler.auth_code` is a class attribute** — [box_ingest/crawler.py:60-67](box_ingest/crawler.py#L60-L67). Subsequent runs would read a stale code; not an issue today since the server is created fresh per run, but fragile. **Fix:** instance attribute on a server-bound handler.

56. **Logger emits raw user input** — [server/rag_chain.py:233](server/rag_chain.py#L233). `Rephrased: '%s' → '%s'` ignores `LOG_QUERIES`. **Fix:** guard with `if LOG_QUERIES`.

---

## LOW (selected; full list is mostly cosmetic)

- **Dead field `safety_flag`** — [server/middleware.py:73-87, 335](server/middleware.py#L73-L87) — set but never read. Remove or wire up.
- **Heading-level int cast not bounds-checked** — [maryland_rag/pass2/strategies/docx_strategy.py:79-102](maryland_rag/pass2/strategies/docx_strategy.py#L79-L102) — `int(heading.name[1])` crashes on a malformed `h` tag. Use `re.match(r"h(\d)", ...)`.
- **First-row-as-header heuristic** — [maryland_rag/pass2/strategies/xls_strategy.py:103-120](maryland_rag/pass2/strategies/xls_strategy.py#L103-L120) — undocumented assumption; add a comment.
- **`add_page` silently dedups across URL paths** — [maryland_rag/pass1/db.py:94-99](maryland_rag/pass1/db.py#L94-L99) — log skipped dupes if you care for audit.
- **Pass1 SQLite has no `check_same_thread` guard** — [maryland_rag/pass1/db.py:77](maryland_rag/pass1/db.py#L77) — fine while crawler is sync; document.
- **Hardcoded Playwright timeout** — [box_ingest/crawler.py:170](box_ingest/crawler.py#L170) — promote to config.

---

## Cross-Cutting Themes

1. **No timeouts anywhere.** Every external boundary (LLMs, embeddings, pgvector, Box, OpenAI, requests, urllib) is unbounded. This is the single biggest reliability gap; a one-pass sweep can fix it across the codebase.

2. **Sync code on async hot path.** [main.py](server/main.py) → [middleware.detect_pii](server/middleware.py#L182-L217) → [rag_chain.PgVectorRetriever](server/rag_chain.py#L101-L142). Every concurrent user blocks one of starlette's threadpool slots. A handful of slow users exhausts capacity. The fix is mechanical: `asyncio.to_thread` for Presidio, `AsyncOpenAI` + `AsyncConnectionPool` for retrieval.

3. **Fail-open guardrails.** Classification, partisan-check, and PII all return "allow" on exception. For a chatbot that explicitly aims to suppress partisan output, the safer default is fail-closed with a canned message.

4. **In-memory state with no eviction.** Rate limiter, session store, periodic cleanup. All single-process, all unbounded. Either keep the process pinned to one worker and ship the eviction fixes, or move to Redis/Postgres.

5. **Divergent crawlers.** [box_ingest/crawler.py](box_ingest/crawler.py) and [maryland_rag/pass1/crawler.py](maryland_rag/pass1/crawler.py) reinvent overlapping concerns (HTTP fetch with retry/timeout, manifest writing, exception handling). The Box crawler uses `urllib`; the pass1 crawler uses `requests`. Consolidating on one HTTP wrapper with shared retry/timeout would close several of the High/Medium findings at once.

6. **No atomic writes for shared files.** `ingest.py` uses tmp+rename; `manifest.py` does not. Same data-loss risk on Ctrl-C or OOM. Pick one helper, use it everywhere.

7. **Bare/over-broad `except`.** Multiple sites swallow KeyboardInterrupt and SystemExit alongside real errors. Mechanical sweep: replace `except:` and `except Exception` (without re-raise) with narrow exception types or `logger.exception` + re-raise.

---

## Recommended Remediation Order

If the goal is "production-safe for the survey deployment in a few weeks," tackle in this order:

1. **All Criticals 1-3 (timeouts + async hot path)** — single largest reliability win, no schema changes.
2. **Criticals 4-6 (path traversal, rate limiter eviction, session-store guardrails)** — security and memory.
3. **Highs 26-28, 33-34 (fail-closed guardrails + bare-except cleanup)** — safety posture.
4. **Atomic manifest write (#8) + state-ordering (#11)** — data integrity.
5. **Token file permissions (#9) + Box token leakage (#29)** — credential hygiene.
6. **Everything Medium can be batched into a one-day cleanup pass.**

Lows are cosmetic; address opportunistically.

---

## Verification

After fixes land, verify end-to-end with the existing tooling plus a few new spot checks:

- **Smoke tests** in [README.md](README.md) (the `curl` examples at [server/main.py:13-38](server/main.py#L13-L38)) — registration deadline, PII test, partisan test — should still pass.
- **Stress test** — [maryland_rag/scripts/stress_test.py](maryland_rag/scripts/stress_test.py) hit at 30+ concurrent users; latency variance should drop sharply after the async fixes. Capture p50/p95/p99 before and after.
- **Timeout test** — point `OPENAI_BASE_URL` at a slow proxy (toxiproxy or a `nc -l` that never responds); verify `/chat` returns 502 within the configured timeout rather than hanging.
- **Path-traversal test** — manually upload a Box file named `../../escape.txt` and confirm `automate.py` refuses to write it.
- **Rate-limiter eviction** — run the smoke test in a loop with N>RATE_LIMIT users; verify dict size stays bounded via `len(_rate_limiter._requests)`.
- **Database migration safety** — run [maryland_rag/scripts/db_cleanup.py](maryland_rag/scripts/db_cleanup.py) on the existing manifest backup (`data/manifest.db.bak.20260225_155124`) and confirm pre/post row counts match expectations.
- **Pass3 idempotency** — re-run `python -m maryland_rag pass3 --resume` and confirm "0 inserted" on a clean DB.

No deliverable is implementation in this turn; the user requested an audit. Fixes should be staged in separate PRs by section, starting with the timeout sweep.
