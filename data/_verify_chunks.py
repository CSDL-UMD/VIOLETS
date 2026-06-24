"""Throwaway verification of the pgvector `chunks` table vs the source JSONL."""
import json
import os
import re
from pathlib import Path

# Load DATABASE_URL from .env if not already in environment.
if not os.environ.get("DATABASE_URL"):
    envp = Path(__file__).resolve().parent.parent / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL=") and "=" in line:
                os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip('"').strip("'")

import psycopg

ROOT = Path(__file__).resolve().parent.parent
files = ["data/chunks.jsonl", "data/box_chunks.jsonl"]

# Collect chunk_ids from JSONL, detect dupes within files.
jsonl_ids = {}
multi_source = 0
bad_id_fmt = 0
empty_text = 0
for f in files:
    p = ROOT / f
    if not p.exists():
        print(f"  MISSING file: {f}")
        continue
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        n += 1
        cid = c.get("chunk_id", "")
        jsonl_ids.setdefault(cid, []).append(f)
        if not re.fullmatch(r"[0-9a-f]{32}", cid or ""):
            bad_id_fmt += 1
        if not (c.get("text") or "").strip():
            empty_text += 1
        su = c.get("source_urls")
        if isinstance(su, list) and len(su) > 1:
            multi_source += 1
    print(f"  {f}: {n} chunk lines")

dupe_ids = {k: v for k, v in jsonl_ids.items() if len(v) > 1}
print(f"\nJSONL total unique chunk_ids: {len(jsonl_ids)}")
print(f"JSONL duplicate chunk_ids (same id >1 line): {len(dupe_ids)}")
print(f"JSONL chunk_ids NOT matching new 32-hex scheme: {bad_id_fmt}")
print(f"JSONL chunks with empty text: {empty_text}")
print(f"JSONL chunks carrying multi-source source_urls (>1): {multi_source}")

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    null_url = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_url IS NULL").fetchone()[0]
    dim = conn.execute("SELECT vector_dims(embedding) FROM chunks LIMIT 1").fetchone()
    db_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall()}
    has_su = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE metadata ? 'source_urls'"
    ).fetchone()[0]

print(f"\nDB rows in chunks: {total}")
print(f"DB embedding dim: {dim[0] if dim else 'n/a'}")
print(f"DB rows with NULL embedding: {nulls}")
print(f"DB rows with NULL source_url: {null_url}")
print(f"DB rows whose metadata has source_urls key: {has_su}")

js = set(jsonl_ids)
orphans = db_ids - js          # in DB but not in current JSONL -> stale/old-scheme
missing = js - db_ids          # in JSONL but never embedded
print(f"\nORPHANS (in DB, not in JSONL — stale rows): {len(orphans)}")
for o in list(orphans)[:5]:
    print(f"    {o}")
print(f"MISSING (in JSONL, not embedded): {len(missing)}")
for m in list(missing)[:5]:
    print(f"    {m}")

print("\nRESULT:", "OK" if not orphans and not missing and nulls == 0 else "NEEDS ATTENTION")
