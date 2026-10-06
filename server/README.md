# server/ — FastAPI RAG Chatbot

The conversational RAG server for the Maryland Elections chatbot. It answers
questions over the pgvector store built by the pipeline (see the root
[README](../README.md), Section 8), behind API-key auth, per-user rate
limiting, and PII / classification / partisan guardrails.

## Run

```bash
python -m server.main            # pins workers=1; HOST/PORT via env (default 0.0.0.0:8000)
# or:
uvicorn server.main:app --host 0.0.0.0 --port 8000   # do NOT pass --workers >1
```

The server keeps rate-limit and session state in process memory, so it **must
run as a single worker**.

## Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| `POST` | `/chat` | `X-API-Key` | Message → answer + source list |
| `POST` | `/reset` | `X-API-Key` | Clear a user's conversation |
| `GET`  | `/health` | public | Pings the DB → `{status, database, model}`; 503 if DB down |

## Files

| File | Responsibility |
|---|---|
| `main.py` | App + async lifespan (logging, `AsyncConnectionPool` `min=4/max=25`, `SessionStore`, `build_chain`), `X-API-Key` dependency, `_RateLimiter` (per-user + global backstop), CORS, `_periodic_maintenance` heartbeat (300 s), `RAG_CHAIN_TIMEOUT=60`. |
| `config.py` | Loads `.env`; validates `OPENAI_API_KEY` / `DATABASE_URL` / `VIOLETS_API_KEY` at import; exposes `LLM_MODEL`, `RETRIEVER_K`, session/rate-limit settings. |
| `rag_chain.py` | `RunnableLambda` pipeline branching on `query_category`; async `PgVectorRetriever` (`embedding <=> %s::vector`, 30 s statement timeout); four prompts; `_replace_source_refs`. |
| `middleware.py` | Guardrails — `detect_pii` (Presidio), `classify_query` (9 categories, incl. `out_of_scope`), `check_partisan_response`. All **fail closed**. Guardrail LLM timeouts 30 s / 60 s. |
| `rag_logger.py` | LangChain callback handler: token/cost estimate (`_COST_TABLE`), retriever timing, `log_request()`. `LOG_PROMPTS/RESPONSES/QUERIES` default **False** (opt-in). |
| `logging_setup.py` | Central logging: rotating `logs/server.log` (10 MB × 5) + console at `LOG_LEVEL`; per-request id via `contextvars`. |
| `metrics.py` | Thread-safe in-process counters (`METRICS`) feeding the heartbeat line. No HTTP surface. |
| `session.py` | Thread-safe in-memory per-`user_id` history with TTL + max-turn cap, plus a cache of the latest retrieval turn's sources (for link follow-ups). |
| `eval_guardrails.py` | Offline eval: does `reasoning_effort="minimal"` match `"medium"` on labeled fixtures? `python -m server.eval_guardrails` (makes real OpenAI calls). |
| `eval_retrieval.py` | Offline eval: scores the production retriever against `retrieval_benchmark.json` (49 hand-labeled questions; gold = URL substring + text regex, so it survives re-ingest). Reports hit@1 / hit@K / MRR@10 / P@K. `python -m server.eval_retrieval [-v]`. |
| `requirements.txt` | Server dependencies. |

## Guardrails fail *closed*

`detect_pii`, `classify_query`, and `check_partisan_response` all suppress the
response (canned refusal / error) on error or an unresolvable partisan verdict,
rather than passing an unvetted answer through. This is intentional for an
elections chatbot. There is **no output-side PII scrub** — only input PII is
blocked.

## Config

See root [README Section 14](../README.md#14-configuration-reference) for the
full env-var table (`LLM_MODEL`, `RETRIEVER_K`, `CORS_ORIGINS`, `LOG_LEVEL`,
`SERVER_LOG_FILE`, `LOG_*`, `HOST`/`PORT`, …). Server-specific additions:

| Env var | Default | Purpose |
|---|---|---|
| `RATE_LIMIT_GLOBAL_PER_MINUTE` | `90` | Request cap per minute across **all** users combined (`/chat` + `/reset`). Backstop for `RATE_LIMIT_PER_MINUTE`, which is keyed on the client-supplied `user_id` and can be bypassed by rotating ids. |
| `ELECTION_NAME` | `2026 Maryland Gubernatorial General Election` | Election named in the QA/concerns system prompts (with today's date) so deadlines are anchored to the right election. |
| `ELECTION_DATE` | `November 3, 2026` | Date of that election, injected alongside `ELECTION_NAME`. |
| `SIMILARITY_FLOOR` | `0.0` | Drop retrieved chunks whose similarity score (1 − cosine distance) is below this value. `0.0` disables the filter. |
| `PAST_ELECTION_URL_PATTERNS` | `/primary_candidates/` | Comma-separated `source_url` substrings marking past-election documents, which get down-weighted at ranking time. Empty disables. |
| `PAST_ELECTION_PENALTY` | `0.02` | Amount subtracted from a past-election chunk's similarity for ranking only (the floor and logged score stay raw). 0.02 tuned with `eval_retrieval`; ≥0.04 starts hiding answers to questions about the primary. `0` disables. |
| `CANDIDATES_URL` | 2026 primary candidates page | URL returned verbatim for `candidates`-classified queries. **Set this to the general-election candidates page once the State Board publishes it** — the crawl has no such page yet, so the default still points at the primary. |
