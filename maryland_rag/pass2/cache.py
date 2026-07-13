"""
Simple on-disk HTTP cache for Pass 2 fetches.

Every URL is keyed by SHA256(url). HTML responses are stored as UTF-8 text
files (.html), binary downloads (PDF, DOCX) as raw bytes (.bin).

HTML freshness: Pass 1 pushes every successfully-fetched page into this
cache via put_html() at crawl time, so get_html() always sees exactly the
bytes Pass 1 hashed. Binary freshness: each .bin has a .bin.meta sidecar
holding the server's ETag / Last-Modified; get_bytes() revalidates the
cached blob once per run with a conditional GET (304 keeps the cache,
200 replaces it). On any network error the cached copy is used.

On a cache miss the URL is fetched, stored, and returned. A polite delay
(RATE_LIMIT_SECONDS from config) is applied on every live request so
re-runs after a crash don't hammer the server.

Cache directory: data/cache/  (alongside manifest.db)
"""
import hashlib
import json
import logging
import os
import time

import requests

from ..pass1.config import PROJECT_ROOT, RATE_LIMIT_SECONDS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# URLs whose cached blob has already been revalidated (or freshly fetched)
# during this process run — no point re-probing the server within a run.
_REVALIDATED_URLS: set[str] = set()


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


def put_html(url: str, html: str):
    """
    Store freshly-fetched HTML for a URL, overwriting any cached copy.

    Called by Pass 1 at crawl time so the cache always holds exactly the
    HTML the crawler hashed — otherwise a changed page would be re-chunked
    from its stale cached copy forever. Write failures are logged, never
    raised, so a cache problem can't fail the crawl.
    """
    try:
        _write(_cache_path(url, ".html"), html.encode("utf-8"))
    except OSError as exc:
        logger.warning("Failed to cache HTML for %s: %s", url, exc)


def get_bytes(url: str, timeout: int = REQUEST_TIMEOUT * 2) -> bytes:
    """
    Return the raw bytes for a URL (PDF, DOCX, etc.), using the disk cache.

    A cached blob is revalidated against the server once per run using the
    validators stored in its sidecar (ETag / Last-Modified): 304 keeps the
    cached copy, 200 replaces blob + sidecar. If the server offered no
    validators the cache is trusted as-is; if revalidation errors out the
    cached copy is used with a warning.

    Returns empty bytes on fetch failure with no cached copy to fall back on.
    """
    path = _cache_path(url, ".bin")
    meta_path = path + ".meta"

    if os.path.exists(path):
        if url in _REVALIDATED_URLS:
            logger.debug("Cache hit (bytes, revalidated): %s", url)
            return _read_bytes(path)

        meta = _read_meta(meta_path)
        if meta is not None and not meta.get("etag") and not meta.get("last_modified"):
            # Server sent no validators on download — nothing to revalidate
            # against, so trust the cache without re-probing.
            logger.debug("Cache hit (bytes, no validators): %s", url)
            _REVALIDATED_URLS.add(url)
            return _read_bytes(path)

        # Conditional GET when we have validators; a legacy blob with no
        # sidecar gets a plain GET once, which refreshes it and writes one.
        headers = {}
        if meta:
            if meta.get("etag"):
                headers["If-None-Match"] = meta["etag"]
            if meta.get("last_modified"):
                headers["If-Modified-Since"] = meta["last_modified"]

        logger.debug("Revalidating (bytes): %s", url)
        time.sleep(RATE_LIMIT_SECONDS)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 304:
                _REVALIDATED_URLS.add(url)
                return _read_bytes(path)
            resp.raise_for_status()
            data = resp.content
        except Exception as exc:
            logger.warning(
                "Revalidation failed for %s, using cached copy: %s", url, exc
            )
            return _read_bytes(path)

        _write(path, data)
        _write_meta(meta_path, resp)
        _REVALIDATED_URLS.add(url)
        return data

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
    _write_meta(meta_path, resp)
    _REVALIDATED_URLS.add(url)
    return data


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _cache_path(url: str, ext: str) -> str:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ext)


def _write(path: str, data: bytes):
    # Atomic: write to a temp file in the same directory, then rename over
    # the target, so a crash mid-write never leaves a truncated cache entry.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_meta(meta_path: str) -> dict | None:
    """Load a validator sidecar; None if missing or unreadable."""
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable cache sidecar %s: %s", meta_path, exc)
        return None


def _write_meta(meta_path: str, resp: requests.Response):
    """Persist the response's cache validators next to the blob."""
    meta = {
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
    }
    try:
        _write(meta_path, json.dumps(meta).encode("utf-8"))
    except OSError as exc:
        logger.warning("Failed to write cache sidecar %s: %s", meta_path, exc)
