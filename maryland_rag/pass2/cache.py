"""
Simple on-disk HTTP cache for Pass 2 fetches.

Every URL is keyed by SHA256(url). HTML responses are stored as UTF-8 text
files (.html), binary downloads (PDF, DOCX) as raw bytes (.bin).

On a cache hit the file is read from disk — no network request is made.
On a cache miss the URL is fetched, stored, and returned. A polite delay
(RATE_LIMIT_SECONDS from config) is applied on every live fetch so
re-runs after a crash don't hammer the server.

Cache directory: data/cache/  (alongside manifest.db)
"""
import hashlib
import logging
import os
import time

import requests

from ..pass1.config import PROJECT_ROOT, RATE_LIMIT_SECONDS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")


def get_html(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Return the raw HTML for a URL, using the disk cache when available.

    Returns empty string on fetch failure.
    """
    path = _cache_path(url, ".html")
    if os.path.exists(path):
        logger.debug("Cache hit (html): %s", url)
        with open(path, encoding="utf-8") as f:
            return f.read()

    logger.debug("Cache miss (html): %s", url)
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""

    _write(path, html.encode("utf-8"))
    return html


def get_bytes(url: str, timeout: int = REQUEST_TIMEOUT * 2) -> bytes:
    """
    Return the raw bytes for a URL (PDF, DOCX, etc.), using the disk cache.

    Returns empty bytes on fetch failure.
    """
    path = _cache_path(url, ".bin")
    if os.path.exists(path):
        logger.debug("Cache hit (bytes): %s", url)
        with open(path, "rb") as f:
            return f.read()

    logger.debug("Cache miss (bytes): %s", url)
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.content
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return b""

    _write(path, data)
    return data


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _cache_path(url: str, ext: str) -> str:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ext)


def _write(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
