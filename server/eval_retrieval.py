"""
Retrieval ranking eval.

Runs every question in retrieval_benchmark.json through the production
PgVectorRetriever (same SQL, same similarity floor) and scores the ranking
against hand-labeled gold chunks.

Gold is matched by URL substring + optional text regex rather than chunk_id,
so labels survive the drop-and-reingest workflow (chunk IDs change, content
doesn't). A chunk is relevant if it matches ANY gold spec for the question.

Metrics (per question, averaged):
  hit@1   — top result is relevant
  hit@K   — any of the top RETRIEVER_K (what the LLM actually sees) is relevant
  MRR@10  — 1 / rank of the first relevant chunk
  P@K     — share of the top K that is relevant (context quality)

Questions tagged "candidates" are reported separately: in production the
classifier routes most of them to a canned CANDIDATES_URL reply, so they only
reach retrieval when the classifier misses or on follow-ups.

Run from the repo root with the server venv and a populated .env:

    python -m server.eval_retrieval            # summary + misses
    python -m server.eval_retrieval -v         # also print top-5 per question

Makes one embedding call per question (~50, fractions of a cent). Exit code
is the number of non-candidate questions with no relevant chunk in the top K.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing config — it reads required env vars at import time.
load_dotenv()

from langchain_openai import OpenAIEmbeddings  # noqa: E402
from pgvector.psycopg import register_vector_async  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from . import config  # noqa: E402
from .rag_chain import EMBED_MODEL, PgVectorRetriever  # noqa: E402

BENCHMARK = Path(__file__).with_name("retrieval_benchmark.json")
DEPTH = 10  # retrieve this deep so MRR@10 is meaningful


def _is_relevant(doc, gold: list[dict]) -> bool:
    urls = [u.lower() for u in doc.metadata.get("source_urls") or [doc.metadata["source_url"]]]
    for g in gold:
        if "url" in g and not any(g["url"].lower() in u for u in urls):
            continue
        if "text" in g and not re.search(g["text"], doc.page_content, re.I | re.S):
            continue
        return True
    return False


def _summary(label: str, rows: list[dict], k: int) -> str:
    n = len(rows) or 1
    hit1 = sum(r["first"] == 0 for r in rows) / n
    hitk = sum(r["first"] is not None and r["first"] < k for r in rows) / n
    mrr = sum(1 / (r["first"] + 1) for r in rows if r["first"] is not None) / n
    pk = sum(r["rel_topk"] for r in rows) / (n * k)
    return (f"{label:<11} n={len(rows):<3} hit@1 {hit1:.2f}  hit@{k} {hitk:.2f}  "
            f"MRR@10 {mrr:.3f}  P@{k} {pk:.2f}")


async def main() -> int:
    verbose = "-v" in sys.argv
    bench = json.loads(BENCHMARK.read_text())
    k = config.RETRIEVER_K

    pool = AsyncConnectionPool(
        conninfo=config.DATABASE_URL, min_size=1, max_size=4,
        configure=register_vector_async, open=False,
    )
    await pool.open(wait=True, timeout=10)
    retriever = PgVectorRetriever(
        embeddings=OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        ),
        pool=pool,
        k=DEPTH,
    )

    rows = []
    try:
        for item in bench:
            docs = await retriever.ainvoke(item["q"])
            flags = [_is_relevant(d, item["gold"]) for d in docs]
            first = flags.index(True) if True in flags else None
            rows.append({**item, "first": first, "rel_topk": sum(flags[:k])})
            if verbose:
                print(f"\n[{item['id']}] {item['q']}  (first relevant: {first})")
                for d, rel in zip(docs[:5], flags):
                    print(f"  {'*' if rel else ' '} {d.metadata['score']:.3f} "
                          f"{d.metadata['source_url'][-70:]}")
    finally:
        await pool.close()

    cand = [r for r in rows if "candidates" in r["tags"]]
    general = [r for r in rows if "candidates" not in r["tags"]]
    print(f"\nembedding={EMBED_MODEL}  K={k}  floor={config.SIMILARITY_FLOOR}")
    print(_summary("general", general, k))
    print(_summary("candidates", cand, k))
    print(_summary("all", rows, k))

    misses = [r for r in rows if r["first"] is None or r["first"] >= k]
    if misses:
        print(f"\nNot in top {k}:")
        for r in misses:
            print(f"  {r['id']:<24} first relevant rank: {r['first']}  — {r['q']}")
    return sum(1 for r in misses if "candidates" not in r["tags"])


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
