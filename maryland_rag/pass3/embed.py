"""
Pass 3: Embed chunks and insert into PostgreSQL with pgvector.

Reads data/chunks.jsonl (output from Pass 2), embeds each chunk's text
using OpenAI text-embedding-3-small (1536-dim), and inserts vectors +
metadata into a PostgreSQL table with the pgvector extension.

Env vars (set in .env or shell):
    OPENAI_API_KEY
    DATABASE_URL          (e.g. postgresql://user:pass@localhost:5432/violets)

Usage:
    python -m maryland_rag pass3
    python -m maryland_rag pass3 --chunks data/chunks.jsonl
    python -m maryland_rag pass3 --resume        # skip already-inserted chunk IDs
"""
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# OpenAI allows up to 2048 inputs per request; 100 is conservative and
# keeps individual request payloads small.
EMBED_BATCH_SIZE = 100

# Number of rows to insert per transaction.
INSERT_BATCH_SIZE = 100

# Seconds to wait after a retryable API error before re-attempting.
RETRY_DELAY = 5


def run_embed(
    chunks_path: str = "data/chunks.jsonl",
    resume: bool = False,
) -> int:
    """
    Embed all chunks and insert into PostgreSQL.

    Args:
        chunks_path: Path to the JSONL file produced by Pass 2.
        resume: If True, skip chunk IDs already present in the database.

    Returns:
        Number of vectors inserted.
    """
    _setup_logging()
    load_dotenv()

    openai_api_key = _require_env("OPENAI_API_KEY")
    database_url = _require_env("DATABASE_URL")

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    conn = _setup_pgvector(database_url)

    if resume:
        done_ids = _get_existing_ids(conn)
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        logger.info(
            "Resume mode: skipping %d already-inserted, %d remaining",
            before - len(chunks),
            len(chunks),
        )

    if not chunks:
        logger.info("Nothing to insert.")
        conn.close()
        return 0

    from openai import OpenAI
    oai = OpenAI(api_key=openai_api_key)

    total = len(chunks)
    inserted = 0

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # --- Embed ---
        embeddings = _embed_with_retry(oai, texts)
        if embeddings is None:
            logger.error(
                "Skipping batch %d-%d after embedding failure",
                batch_start,
                batch_start + len(batch),
            )
            continue

        # --- Insert into PostgreSQL ---
        rows = []
        for chunk, embedding in zip(batch, embeddings):
            # Separate known columns from extra metadata
            chunk_id = chunk["chunk_id"]
            text = chunk.get("text", "")
            source_url = chunk.get("source_url", "")
            title = chunk.get("title", "")
            meta = {
                k: v for k, v in chunk.items()
                if k not in ("chunk_id", "text", "source_url", "title")
            }
            rows.append((chunk_id, embedding, text, source_url, title, json.dumps(meta)))

        count = _insert_batch(conn, rows)
        inserted += count

        pct = 100.0 * (batch_start + len(batch)) / total
        logger.info(
            "Progress: %d/%d (%.1f%%) — %d inserted",
            batch_start + len(batch),
            total,
            pct,
            inserted,
        )

    conn.close()
    logger.info("Pass 3 complete. %d vectors inserted into pgvector.", inserted)
    return inserted


# ---------------------------------------------------------------------------
# PostgreSQL / pgvector setup
# ---------------------------------------------------------------------------


def _setup_pgvector(database_url: str):
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise ImportError(
            "psycopg or pgvector not installed. Run: pip install 'psycopg[binary]' pgvector"
        )

    conn = psycopg.connect(database_url)
    register_vector(conn)

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            embedding  vector({EMBED_DIM}),
            text       TEXT,
            source_url TEXT,
            title      TEXT,
            metadata   JSONB DEFAULT '{{}}'::jsonb
        )
    """)
    # Index for fast approximate nearest-neighbor search
    conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
    conn.commit()
    logger.info("pgvector table 'chunks' is ready (dim=%d).", EMBED_DIM)
    return conn


def _get_existing_ids(conn) -> set[str]:
    """Return the set of chunk_ids already in the database."""
    rows = conn.execute("SELECT chunk_id FROM chunks").fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Batch insert
# ---------------------------------------------------------------------------


def _insert_batch(conn, rows: list[tuple]) -> int:
    """Insert a batch of rows, using ON CONFLICT to upsert.

    Each row is wrapped in a savepoint so a single failure doesn't
    roll back previously-committed rows in the same batch.
    """
    count = 0
    for row in rows:
        try:
            conn.execute("SAVEPOINT insert_row")
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, embedding, text, source_url, title, metadata)
                VALUES (%s, %s::vector, %s, %s, %s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    text       = EXCLUDED.text,
                    source_url = EXCLUDED.source_url,
                    title      = EXCLUDED.title,
                    metadata   = EXCLUDED.metadata
                """,
                row,
            )
            conn.execute("RELEASE SAVEPOINT insert_row")
            count += 1
        except Exception as exc:
            logger.warning("Insert failed for chunk %s: %s", row[0], exc)
            conn.execute("ROLLBACK TO SAVEPOINT insert_row")
            continue
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Embedding with retry
# ---------------------------------------------------------------------------


def _embed_with_retry(oai, texts: list[str], retries: int = 3) -> list | None:
    for attempt in range(retries):
        try:
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.warning(
                "Embedding attempt %d/%d failed: %s", attempt + 1, retries, exc
            )
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def _load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Add it to your .env file or shell environment."
        )
    return val


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
