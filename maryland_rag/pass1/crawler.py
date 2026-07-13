"""
BFS web crawler for Maryland State Board of Elections.

Pass 1: Discover all URLs, classify pages, assign chunking strategies,
and persist everything to SQLite. No full document extraction.

Key improvements over original design:
- Fully resumable: on restart, seeds queue from pending rows, visited from non-pending.
- Single HTTP fetch per page (no double-fetch).
- robots.txt compliance.
- Sliding-window rate monitoring with warnings.
- Proper base URL for link normalization (current page, not domain root).
- Transient failures (timeouts, connection errors, 5xx) are retried with
  exponential backoff. Neither they nor unexpected processing exceptions
  ever demote a previously-crawled row to 'failed'.
"""
import logging
import os
import time
from collections import deque
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from .config import (
    SEED_URLS,
    DOMAINS,
    RATE_LIMIT_SECONDS,
    MAX_DEPTH,
    MAX_RETRIES,
    LOG_DIR,
    LOG_FILE,
    REQUEST_TIMEOUT,
    REQUESTS_PER_MINUTE_WARN,
    RESPECT_ROBOTS_TXT,
    SAVE_RAW_HTML,
    RAW_HTML_DIR,
)
from .db import DB
from .extractor import extract_page
from .classifier import classify_page
from .exclusions import should_exclude, is_excluded_status, TRANSIENT_HTTP_STATUSES
from .utils import normalize_url, is_internal, get_content_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

def _load_robots(domain: str) -> RobotFileParser | None:
    """Fetch and parse robots.txt using the same HTTP client (and therefore
    the same User-Agent) as every other crawler request.

    RobotFileParser.read() is deliberately NOT used: it fetches with
    urllib's default 'Python-urllib' UA, which both target sites' WAFs
    reject with 403 — and robotparser treats 401/403 as disallow-ALL,
    silently excluding entire domains whose robots.txt actually allows '*'.

    Response handling follows RFC 9309 §2.3.1:
      - 2xx: parse the rules.
      - 4xx ("unavailable"): robots.txt imposes no restrictions -> None.
      - 5xx / network failure after retries ("unreachable"): must assume
        complete disallow -> disallow-all parser, logged as an error so the
        run visibly explains why everything is being excluded.
    """
    robots_url = f"https://{domain}/robots.txt"
    last_err = ""
    for attempt in range(MAX_RETRIES):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            resp = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_err = str(exc)
            continue
        if resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            logger.warning(
                "robots.txt unavailable for %s (HTTP %d) — no crawl "
                "restrictions per RFC 9309", domain, resp.status_code,
            )
            return None
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(resp.text.splitlines())
        return rp

    logger.error(
        "robots.txt UNREACHABLE for %s after %d attempts (%s) — treating the "
        "entire domain as disallowed per RFC 9309; every %s URL will be "
        "excluded until the fetch succeeds",
        domain, MAX_RETRIES, last_err, domain,
    )
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.parse(["User-agent: *", "Disallow: /"])
    return rp


# ---------------------------------------------------------------------------
# Rate monitoring
# ---------------------------------------------------------------------------

class RateMonitor:
    """Track requests per minute with a sliding window."""

    def __init__(self, warn_threshold: int = 80):
        self.timestamps: deque[float] = deque()
        self.warn_threshold = warn_threshold

    def record(self):
        now = time.time()
        self.timestamps.append(now)
        # Evict entries older than 60s
        cutoff = now - 60.0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) > self.warn_threshold:
            logger.warning(
                "Rate limit warning: %d requests in the last 60s (threshold: %d)",
                len(self.timestamps), self.warn_threshold,
            )


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------

def _is_transient_failure(result: dict | None) -> bool:
    """True for failures worth retrying: network errors/timeouts (extract_page
    returns None), 5xx server errors, and 429/408 rate-limit/timeout
    responses."""
    if result is None:
        return True
    status = result.get('http_status') or 0
    return status >= 500 or status in TRANSIENT_HTTP_STATUSES


def _fetch_with_retry(url: str, rate_monitor: RateMonitor) -> dict | None:
    """
    Fetch a page, retrying transient failures up to MAX_RETRIES times with
    exponential backoff. Permanent outcomes (success, 4xx) return immediately.

    Returns the final extract_page() result — None or a 5xx result means the
    failure persisted through every attempt.
    """
    result = None
    for attempt in range(MAX_RETRIES + 1):
        result = extract_page(url)
        rate_monitor.record()
        if not _is_transient_failure(result):
            return result
        if attempt < MAX_RETRIES:
            # Modest exponential backoff, never faster than the polite rate:
            # 2x, 4x, 8x... RATE_LIMIT_SECONDS.
            backoff = RATE_LIMIT_SECONDS * (2 ** (attempt + 1))
            status = result.get('http_status') if result else 'network error'
            logger.warning(
                "Transient failure (%s) on %s — retry %d/%d in %.1fs",
                status, url, attempt + 1, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
    return result


def _mark_failure_preserving_crawled(db: DB, url: str, why: str):
    """
    Record a failure without demoting a previously-crawled page.

    If a prior run already crawled this page, keep its 'crawled' row instead
    of demoting it to 'failed': manifest.db persists across runs, so Pass 2
    can still chunk the page (from the disk cache when present). Only pages
    that have never been crawled successfully are marked 'failed'.

    Used for transient fetch failures that survived all retries AND for
    unexpected processing exceptions (classifier/DB/link-parsing bugs) — in
    both cases the previously-captured content is still perfectly good.
    """
    if db.get_page_status(url) == 'crawled':
        logger.warning(
            "%s on previously-crawled page %s — keeping prior "
            "'crawled' row so Pass 2 still chunks it", why, url,
        )
        return
    db.update_status(url, 'failed')


# ---------------------------------------------------------------------------
# Main crawl loop
# ---------------------------------------------------------------------------

def run_crawl(resume: bool = True):
    """
    Execute the BFS crawl.

    Args:
        resume: If True and there are pending rows in the DB, resume from them
                instead of re-seeding from SEED_URLS.
    """
    db = DB()
    db.init_schema()

    # --- robots.txt (one parser per domain) ---
    robots_map: dict[str, RobotFileParser | None] = {}
    if RESPECT_ROBOTS_TXT:
        for d in DOMAINS:
            robots_map[d] = _load_robots(d)

    # --- Resume or fresh start ---
    pending = db.get_pending() if resume else []

    if pending:
        logger.info("Resuming crawl with %d pending URLs", len(pending))
        queue = deque((row['url'], row['parent_url'], row['depth']) for row in pending)
        visited = db.get_all_visited_urls()
        run_id = db.start_run(f"multi-domain resumed ({len(SEED_URLS)} seeds)")
    else:
        logger.info("Starting fresh crawl from %d seed URLs", len(SEED_URLS))
        run_id = db.start_run(f"multi-domain ({len(SEED_URLS)} seeds)")
        queue = deque()
        visited = set()

        for seed_url in SEED_URLS:
            excluded, reason = should_exclude(seed_url)
            if excluded:
                logger.error("Seed URL excluded: %s — %s", seed_url, reason)
                continue
            db.add_page(url=seed_url, parent_url=None, depth=0)
            queue.append((seed_url, None, 0))

    rate_monitor = RateMonitor(REQUESTS_PER_MINUTE_WARN)
    pages_crawled = 0

    while queue:
        url, parent_url, depth = queue.popleft()

        if url in visited:
            continue

        # Gate 1: exclusion check (no network call)
        excluded, reason = should_exclude(url)
        if excluded:
            logger.info("EXCLUDED [%s]: %s", reason, url)
            db.update_status(url, 'excluded', reason=reason)
            visited.add(url)
            continue

        # Gate 2: depth limit
        if depth > MAX_DEPTH:
            db.update_status(url, 'skipped', reason='max_depth')
            visited.add(url)
            continue

        # Gate 3: robots.txt — look up parser by domain
        _robots = robots_map.get(urlparse(url).netloc)
        if _robots and not _robots.can_fetch('*', url):
            logger.info("ROBOTS.TXT blocked: %s", url)
            db.update_status(url, 'excluded', reason='robots.txt')
            visited.add(url)
            continue

        visited.add(url)
        logger.info("[depth=%d] Crawling: %s", depth, url)

        try:
            result = _fetch_with_retry(url, rate_monitor)

            if result is None:
                # Network-level failure that survived all retries
                _mark_failure_preserving_crawled(db, url, "Network failure after retries")
                continue

            http_status = result.get('http_status', 200)

            # 5xx / 429 / 408 that survived all retries is still transient —
            # never demote a previously-crawled row over it, and never persist
            # the error body as page content
            if http_status >= 500 or http_status in TRANSIENT_HTTP_STATUSES:
                logger.info("HTTP %d after retries: %s", http_status, url)
                _mark_failure_preserving_crawled(db, url, f"HTTP {http_status} after retries")
                continue

            # Gate 4: HTTP status exclusion (permanent errors, e.g. 403/404/410)
            if is_excluded_status(http_status):
                logger.info("HTTP %d — marking failed: %s", http_status, url)
                db.update_status(url, 'failed')
                continue

            # Classify and persist
            classification = classify_page(result)
            db.update_page(url, result, classification)
            pages_crawled += 1

            # Optionally save raw HTML
            if SAVE_RAW_HTML and result.get('raw_html'):
                _save_raw_html(url, result['raw_html'])

            # Process outbound links — use CURRENT page as base URL
            for link in result.get('links', []):
                norm = normalize_url(link['href'], base=url)
                if not norm:
                    continue

                # Exclude links before recording
                link_excluded, link_reason = should_exclude(norm)
                if link_excluded:
                    continue

                is_int = is_internal(norm)
                is_doc = get_content_type(norm) != 'html'

                db.add_link(
                    source=url,
                    target=norm,
                    text=link.get('text', ''),
                    context=link.get('context', ''),
                    is_internal=int(is_int),
                    is_document=int(is_doc),
                )

                # Only queue internal links we haven't visited
                if is_int and norm not in visited:
                    db.add_page(url=norm, parent_url=url, depth=depth + 1)
                    queue.append((norm, url, depth + 1))

        except Exception as exc:
            logger.error("Exception on %s: %s", url, exc, exc_info=True)
            # Same no-demotion rule as fetch failures: an unexpected bug
            # (classifier, sqlite write, link parsing) while re-processing a
            # previously-crawled page must not flip its row to 'failed' —
            # that would silently drop the page from Pass 2 even though its
            # cached content is still perfectly chunkable.
            _mark_failure_preserving_crawled(
                db, url, f"Unexpected exception ({exc.__class__.__name__})"
            )
        finally:
            time.sleep(RATE_LIMIT_SECONDS)

    # --- Finalize ---
    db.finalize_run(run_id)
    logger.info("Pass 1 complete. Pages crawled: %d", pages_crawled)
    db.close()


def _save_raw_html(url: str, html: str):
    """Save raw HTML to disk for debugging / reprocessing."""
    os.makedirs(RAW_HTML_DIR, exist_ok=True)
    from urllib.parse import urlparse
    safe_name = urlparse(url).path.strip('/').replace('/', '_') or 'index'
    path = os.path.join(RAW_HTML_DIR, f"{safe_name}.html")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception as exc:
        logger.warning("Failed to save raw HTML for %s: %s", url, exc)


if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    run_crawl()
