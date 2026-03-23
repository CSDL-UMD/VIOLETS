"""
Pass 3: Embed chunks and upsert to Pinecone.

Reads data/chunks.jsonl (output from Pass 2), embeds each chunk's text
using OpenAI text-embedding-3-small (1536-dim), and upserts vectors +
metadata to a Pinecone serverless index.

Env vars (set in .env or shell):
    OPENAI_API_KEY
    PINECONE_API_KEY
    PINECONE_INDEX_NAME   (default: maryland-elections)
    PINECONE_CLOUD        (default: aws)
    PINECONE_REGION       (default: us-east-1)

Usage:
    python -m maryland_rag pass3
    python -m maryland_rag pass3 --chunks data/chunks.jsonl
    python -m maryland_rag pass3 --resume        # skip already-upserted chunk IDs
"""
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
METRIC = "cosine"

# OpenAI allows up to 2048 inputs per request; 100 is conservative and
# keeps individual request payloads small.
EMBED_BATCH_SIZE = 100

# Pinecone serverless supports large upsert batches but 100 vectors is a
# safe default that avoids payload-size errors.
UPSERT_BATCH_SIZE = 100

# Seconds to wait after a retryable API error before re-attempting.
RETRY_DELAY = 5


def run_embed(
    chunks_path: str = "data/chunks.jsonl",
    resume: bool = False,
) -> int:
    """
    Embed all chunks and upsert to Pinecone.

    Args:
        chunks_path: Path to the JSONL file produced by Pass 2.
        resume: If True, skip chunk IDs already recorded in the checkpoint file.

    Returns:
        Number of vectors upserted.
    """
    _setup_logging()
    _load_dotenv()

    openai_api_key = _require_env("OPENAI_API_KEY")
    pinecone_api_key = _require_env("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "maryland-elections")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    if resume:
        done_ids = _load_checkpoint(chunks_path)
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        logger.info(
            "Resume mode: skipping %d already-upserted, %d remaining",
            before - len(chunks),
            len(chunks),
        )
    else:
        done_ids = set()

    if not chunks:
        logger.info("Nothing to upsert.")
        return 0

    index = _setup_pinecone(pinecone_api_key, index_name, cloud, region)

    from openai import OpenAI
    oai = OpenAI(api_key=openai_api_key)

    total = len(chunks)
    upserted = 0

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

        # --- Build Pinecone vectors ---
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            meta = _sanitize_metadata(
                {k: v for k, v in chunk.items() if k != "chunk_id"}
            )
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": meta,
                }
            )

        # --- Upsert in sub-batches ---
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            sub = vectors[i : i + UPSERT_BATCH_SIZE]
            success = _upsert_with_retry(index, sub)
            if success:
                upserted += len(sub)

        # --- Checkpoint ---
        for c in batch:
            done_ids.add(c["chunk_id"])
        _save_checkpoint(chunks_path, done_ids)

        pct = 100.0 * (batch_start + len(batch)) / total
        logger.info(
            "Progress: %d/%d (%.1f%%) — %d upserted",
            batch_start + len(batch),
            total,
            pct,
            upserted,
        )

    logger.info(
        "Pass 3 complete. %d vectors in Pinecone index '%s'.", upserted, index_name
    )
    return upserted


# ---------------------------------------------------------------------------
# Pinecone setup
# ---------------------------------------------------------------------------


def _setup_pinecone(api_key: str, index_name: str, cloud: str, region: str):
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        raise ImportError(
            "pinecone package not installed. Run: pip install pinecone-client"
        )

    pc = Pinecone(api_key=api_key)
    existing_names = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_names:
        logger.info(
            "Creating Pinecone serverless index '%s' (dim=%d, metric=%s, %s/%s)...",
            index_name,
            EMBED_DIM,
            METRIC,
            cloud,
            region,
        )
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Poll until ready
        while True:
            status = pc.describe_index(index_name).status
            if status.get("ready", False):
                break
            logger.info("Waiting for index to become ready...")
            time.sleep(3)
        logger.info("Index '%s' is ready.", index_name)
    else:
        logger.info("Using existing Pinecone index '%s'.", index_name)

    return pc.Index(index_name)


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
# Upsert with retry
# ---------------------------------------------------------------------------


def _upsert_with_retry(index, vectors: list, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            index.upsert(vectors=vectors)
            return True
        except Exception as exc:
            logger.warning(
                "Upsert attempt %d/%d failed: %s", attempt + 1, retries, exc
            )
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return False


# ---------------------------------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------------------------------


def _sanitize_metadata(meta: dict) -> dict:
    """
    Convert a chunk metadata dict to Pinecone-compatible types.

    Pinecone metadata values must be: str, int, float, bool, or list[str].
    None values, nested dicts, and mixed-type lists are all rejected.
    """
    result = {}
    for k, v in meta.items():
        if v is None:
            result[k] = ""
        elif isinstance(v, bool):
            result[k] = v
        elif isinstance(v, (int, float)):
            result[k] = v
        elif isinstance(v, str):
            result[k] = v
        elif isinstance(v, list):
            # Pinecone requires list[str] — coerce all items
            result[k] = [str(item) for item in v if item is not None]
        elif isinstance(v, dict):
            # Flatten nested dicts to a JSON string so they're retrievable
            result[k] = json.dumps(v, ensure_ascii=False)
        else:
            result[k] = str(v)
    return result


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
# Checkpoint (resume support)
# ---------------------------------------------------------------------------


def _checkpoint_path(chunks_path: str) -> str:
    return str(Path(chunks_path).with_suffix(".checkpoint"))


def _load_checkpoint(chunks_path: str) -> set[str]:
    path = _checkpoint_path(chunks_path)
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_checkpoint(chunks_path: str, done_ids: set[str]):
    path = _checkpoint_path(chunks_path)
    with open(path, "w", encoding="utf-8") as f:
        for chunk_id in done_ids:
            f.write(chunk_id + "\n")


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


def _load_dotenv():
    """Load key=value pairs from a .env file in the current directory."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Strip inline comments and surrounding quotes
            value = value.split("#")[0].strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
