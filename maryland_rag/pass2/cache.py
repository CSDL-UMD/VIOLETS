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

Binary payloads are additionally validated against the expected type's
magic bytes (%PDF-, PK, OLE2) before being returned: a poisoned entry
(e.g. an HTML nav page cached under a .pdf URL) is deleted and refetched
once, and empty bytes are returned if the refetch is still invalid. See
get_bytes() / purge_poisoned().

On a cache miss the URL is fetched, stored, and returned. A polite delay
(RATE_LIMIT_SECONDS from config) is applied on every live request so
re-runs after a crash don't hammer the server.

Cache directory: data/cache/  (alongside manifest.db)
"""
import hashlib
import json
import logging
import os
import re
import time

import requests

from ..pass1.config import PROJECT_ROOT, RATE_LIMIT_SECONDS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# URLs whose cached blob has already been revalidated (or freshly fetched)
# during this process run — no point re-probing the server within a run.
_REVALIDATED_URLS: set[str] = set()

# --- Binary payload validation ---------------------------------------------
# Magic-byte prefixes per expected payload kind. Guards against the live
# failure mode where a document URL is answered with the site's HTML nav
# page (e.g. the /pdf/vrar/MSR-*.pdf reports): the HTML got cached as .bin
# and "extracted" into junk chunks. 'cfb' is the OLE2/Compound File Binary
# container used by legacy .xls/.doc; 'ooxml' (docx/xlsx/xlsm) is a ZIP.
# 'csv' has no magic — it is only rejected when the payload looks like HTML.
_MAGIC_BYTES = {
    "pdf": b"%PDF-",
    "ooxml": b"PK",
    "cfb": b"\xd0\xcf\x11\xe0",
}

_KIND_BY_CONTENT_TYPE = {
    "pdf": "pdf",
    "docx": "ooxml",
    "xlsx": "ooxml",
    "xlsm": "ooxml",
    "xls": "cfb",
    "doc": "cfb",
    "csv": "csv",
}

_KIND_BY_EXTENSION = {
    ".pdf": "pdf",
    ".docx": "ooxml",
    ".xlsx": "ooxml",
    ".xlsm": "ooxml",
    ".xls": "cfb",
    ".doc": "cfb",
    ".csv": "csv",
}


def kind_for_content_type(content_type: str | None) -> str | None:
    """Map a manifest content_type to a payload-validation kind."""
    return _KIND_BY_CONTENT_TYPE.get(content_type or "")


def kind_for_url(url: str) -> str | None:
    """Infer the payload-validation kind from a URL's file extension."""
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    for ext, kind in _KIND_BY_EXTENSION.items():
        if path.endswith(ext):
            return kind
    return None


def looks_like_html(data: bytes) -> bool:
    """True when the payload starts like an HTML document ('<', '<!DOCTYPE',
    '<html', case-insensitive, ignoring leading whitespace/BOM)."""
    head = data.lstrip(b"\xef\xbb\xbf \t\r\n\x0c")[:16].lower()
    return head.startswith(b"<")


def payload_matches(kind: str, data: bytes) -> bool:
    """True when the payload's magic bytes match the expected kind. Only
    the first KB is examined, so callers may pass a head-only read."""
    if not data:
        return False
    if kind == "csv":
        # No magic for CSV — only reject the known HTML-poisoning mode.
        return not looks_like_html(data)
    magic = _MAGIC_BYTES.get(kind)
    if magic is None:
        return True  # unknown kind: nothing to validate against
    if kind == "pdf":
        # Some generators emit a BOM, whitespace, or other junk before the
        # %PDF- header — accept it anywhere in the first 1024 bytes.
        return magic in data[:1024]
    return data.startswith(magic)


def drop_cache_entry(url: str):
    """Delete a URL's cached binary blob and its validator sidecar."""
    path = _cache_path(url, ".bin")
    for p in (path, path + ".meta"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to delete cache entry %s: %s", p, exc)
    _REVALIDATED_URLS.discard(url)


def purge_poisoned(urls_with_kinds) -> int:
    """
    Scan cached binaries and delete every entry whose bytes fail the magic
    check for its expected kind (the HTML-nav-page poisoning above), so the
    next fetch starts clean. Takes an iterable of (url, kind) pairs; entries
    with kind=None or no cached blob are skipped. Returns the purge count.
    """
    purged = 0
    for url, kind in urls_with_kinds:
        if not kind:
            continue
        path = _cache_path(url, ".bin")
        if not os.path.exists(path):
            continue
        # Magic check only needs the head — don't read whole blobs.
        data = _read_head(path)
        if payload_matches(kind, data):
            continue
        logger.error(
            "Purging poisoned cache entry for %s: expected %s, got %s",
            url, kind, "HTML" if looks_like_html(data) else f"{data[:8]!r}",
        )
        drop_cache_entry(url)
        purged += 1
    if purged:
        logger.warning("Purged %d poisoned cache entr%s",
                       purged, "y" if purged == 1 else "ies")
    return purged


# Cloudflare email obfuscation: the real address is XOR-encoded in
# data-cfemail and the visible text is the literal "[email protected]".
# Pass 1 decodes before caching, but get_html() decodes again so chunking
# from a cache written before that fix (or fetched here on a miss) can't
# leak the placeholder into chunks.
_CFEMAIL_RE = re.compile(
    r'<(a|span)\b[^>]*\bdata-cfemail="([0-9a-fA-F]+)"[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)


def _decode_cfemail(hex_str: str) -> str:
    data = bytes.fromhex(hex_str)
    key = data[0]
    return bytes(b ^ key for b in data[1:]).decode('utf-8')


def deobfuscate_cloudflare_emails(html: str) -> str:
    def _repl(m: re.Match) -> str:
        try:
            return _decode_cfemail(m.group(2))
        except (ValueError, UnicodeDecodeError):
            return m.group(0)
    return _CFEMAIL_RE.sub(_repl, html)


def get_html(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Return the raw HTML for a URL, using the disk cache when available.

    Returns empty string on fetch failure.
    """
    path = _cache_path(url, ".html")
    if os.path.exists(path):
        logger.debug("Cache hit (html): %s", url)
        with open(path, encoding="utf-8") as f:
            return deobfuscate_cloudflare_emails(f.read())

    logger.debug("Cache miss (html): %s", url)
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        html = deobfuscate_cloudflare_emails(resp.text)
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


def get_bytes(url: str, timeout: int = REQUEST_TIMEOUT * 2,
              expect: str | None = None) -> bytes:
    """
    Return the raw bytes for a URL (PDF, DOCX, etc.), using the disk cache,
    validated against the expected payload kind's magic bytes.

    `expect` is a kind from kind_for_content_type() ('pdf', 'ooxml', 'cfb',
    'csv'); when None it is inferred from the URL extension. If the payload
    fails the magic check (typically the site's HTML nav page served in
    place of the document), the poisoned cache entry is deleted and ONE
    refetch is attempted; if the refetched payload is still invalid, an
    ERROR is logged and empty bytes are returned so no junk is extracted.

    Returns empty bytes on fetch failure with no cached copy to fall back on.
    """
    if expect is None:
        expect = kind_for_url(url)

    data = _get_bytes_uncheck(url, timeout)
    if not data or expect is None or payload_matches(expect, data):
        return data

    logger.warning(
        "Payload for %s failed %s magic check (%s) — purging cache entry "
        "and refetching once",
        url, expect, "HTML" if looks_like_html(data) else f"{data[:8]!r}",
    )
    drop_cache_entry(url)
    data = _get_bytes_uncheck(url, timeout)
    if data and payload_matches(expect, data):
        return data

    logger.error(
        "Refetched payload for %s still fails %s magic check — refusing to "
        "extract (zero chunks for this page)", url, expect,
    )
    drop_cache_entry(url)  # don't leave the bad refetch cached
    return b""


def _get_bytes_uncheck(url: str, timeout: int = REQUEST_TIMEOUT * 2) -> bytes:
    """
    Fetch/return raw bytes with cache + revalidation but NO payload
    validation (get_bytes wraps this with the magic check).

    A cached blob is revalidated against the server once per run using the
    validators stored in its sidecar (ETag / Last-Modified): 304 keeps the
    cached copy, 200 replaces blob + sidecar. If the server offered no
    validators the cache is trusted as-is; if revalidation errors out the
    cached copy is used with a warning.
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


def _read_head(path: str, n: int = 1024) -> bytes:
    """First n bytes of a file — enough for every magic/HTML check."""
    with open(path, "rb") as f:
        return f.read(n)


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
