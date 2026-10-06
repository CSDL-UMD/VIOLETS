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
    USER_AGENT,
)
from .db import DB
from .extractor import extract_page
from .classifier import classify_page
from .exclusions import (
    should_exclude,
    is_excluded_status,
    TRANSIENT_HTTP_STATUSES,
)
from .utils import normalize_url, is_internal, get_content_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

class RobotsUnreachableError(RuntimeError):
    """robots.txt could not be fetched after retries — the crawl must abort
    rather than run under the RFC 9309 disallow-all assumption."""


def _load_robots(domain: str) -> RobotFileParser | None:
    """Fetch and parse robots.txt with the project User-Agent — the same
    one sent on every other crawler request.

    RobotFileParser.read() is deliberately NOT used: it fetches with
    urllib's default 'Python-urllib' UA, which both target sites' WAFs
    reject with 403 — and robotparser treats 401/403 as disallow-ALL,
    silently excluding entire domains whose robots.txt actually allows '*'.

    Response handling follows RFC 9309 §2.3.1:
      - 2xx: parse the rules.
      - 4xx ("unavailable"): robots.txt imposes no restrictions -> None.
      - 5xx / network failure after retries ("unreachable"): RFC 9309 says
        assume complete disallow — but writing 'excluded' rows for a whole
        domain would sticky-poison the persistent manifest, so for this
        occasional two-domain crawl we ABORT the run instead (raise) and
        let the operator retry once the outage clears.
    """
    robots_url = f"https://{domain}/robots.txt"
    last_err = ""
    for attempt in range(MAX_RETRIES):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            resp = requests.get(robots_url, timeout=REQUEST_TIMEOUT,
                                headers={'User-Agent': USER_AGENT})
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

    msg = (
        f"robots.txt UNREACHABLE for {domain} after {MAX_RETRIES} attempts "
        f"({last_err}) — RFC 9309 would require disallowing the entire "
        f"domain, which would write sticky 'excluded' rows across the "
        f"manifest. Aborting the crawl instead; re-run once {domain} is "
        f"reachable."
    )
    logger.error(msg)
    raise RobotsUnreachableError(msg)


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


def apply_retroactive_exclusions(db: DB) -> int:
    """
    Re-apply the exclusion RULES to rows that already exist in the manifest.

    Exclusion patterns normally only run at link-discovery time, so a rule
    added AFTER a URL was crawled leaves its row 'crawled'/'failed' forever
    — a permanent zero-chunk coverage failure in Pass 2. Flip newly-matching
    rows to 'excluded' with the rule's reason. Idempotent: already-excluded
    rows are not re-examined, and a row is only ever updated once per run.

    Uses the full should_exclude() check, allowlist included, so the
    allowlist is the single authority over what stays in the corpus: removing
    a URL from it (or from the exclusion rules) also drops an already-crawled
    row. Until 2026-10-05 this pass skipped the allowlist to preserve 35
    documents kept from the Feb 2026 whole-site crawl; those were reviewed,
    and the 8 worth keeping were added to ALLOWED_EXACT_URLS.
    """
    rows = db.conn.execute(
        "SELECT url FROM pages WHERE crawl_status IN ('crawled', 'failed')"
    ).fetchall()
    flipped = 0
    for row in rows:
        excluded, reason = should_exclude(row['url'])
        if excluded:
            logger.info("RETRO-EXCLUDED [%s]: %s", reason, row['url'])
            db.update_status(row['url'], 'excluded', reason=reason)
            flipped += 1
    if flipped:
        logger.warning(
            "Retroactive exclusion pass flipped %d existing row(s) to "
            "'excluded'", flipped,
        )
    return flipped


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

    # Exclusion rules added since the last crawl must also apply to rows
    # already in the persistent manifest, not just newly-discovered links.
    apply_retroactive_exclusions(db)

    # --- robots.txt (one parser per domain) ---
    robots_map: dict[str, RobotFileParser | None] = {}
    if RESPECT_ROBOTS_TXT:
        for d in DOMAINS:
            robots_map[d] = _load_robots(d)

    # --- Resume or fresh start ---
    pending = db.get_pending() if resume else []

    # Both target sites are served case-insensitively (IIS), and pages link
    # to the same file under multiple path casings. All dedup sets therefore
    # hold casefolded URLs; the first-seen casing is what gets crawled and
    # stored.
    if pending:
        logger.info("Resuming crawl with %d pending URLs", len(pending))
        queue = deque((row['url'], row['parent_url'], row['depth']) for row in pending)
        visited = {u.lower() for u in db.get_all_visited_urls()}
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

    # Everything ever enqueued (casefolded) — prevents a second case-variant
    # of a URL from getting its own pending row and queue entry.
    enqueued = {u.lower() for u, _, _ in queue} | visited

    rate_monitor = RateMonitor(REQUESTS_PER_MINUTE_WARN)
    pages_crawled = 0
    # Exact (case-preserved) URLs fetched this run, and how many fetches
    # failed in a way that may have hidden child links — both feed the
    # end-of-run sweep of rows this crawl never reached.
    fetched_exact: set[str] = set()
    fetch_failures = 0

    while queue:
        url, parent_url, depth = queue.popleft()

        if url.lower() in visited:
            continue

        # Gate 1: exclusion check (no network call)
        excluded, reason = should_exclude(url)
        if excluded:
            logger.info("EXCLUDED [%s]: %s", reason, url)
            db.update_status(url, 'excluded', reason=reason)
            visited.add(url.lower())
            continue

        # Gate 2: depth limit
        if depth > MAX_DEPTH:
            db.update_status(url, 'skipped', reason='max_depth')
            visited.add(url.lower())
            continue

        # Gate 3: robots.txt — look up parser by domain
        _robots = robots_map.get(urlparse(url).netloc)
        if _robots and not _robots.can_fetch('*', url):
            logger.info("ROBOTS.TXT blocked: %s", url)
            db.update_status(url, 'excluded', reason='robots.txt')
            visited.add(url.lower())
            continue

        visited.add(url.lower())
        fetched_exact.add(url)
        logger.info("[depth=%d] Crawling: %s", depth, url)

        try:
            result = _fetch_with_retry(url, rate_monitor)

            if result is None:
                # Network-level failure that survived all retries
                fetch_failures += 1
                _mark_failure_preserving_crawled(db, url, "Network failure after retries")
                continue

            http_status = result.get('http_status', 200)

            # 5xx / 429 / 408 / 403 that survived all retries is still
            # transient — never demote a previously-crawled row over it, and
            # never persist the error body as page content
            if http_status >= 500 or http_status in TRANSIENT_HTTP_STATUSES:
                logger.info("HTTP %d after retries: %s", http_status, url)
                fetch_failures += 1
                _mark_failure_preserving_crawled(db, url, f"HTTP {http_status} after retries")
                continue

            # Gate 4: HTTP status exclusion (permanent errors, e.g. 404/410).
            # 403 normally never reaches here — it is in
            # TRANSIENT_HTTP_STATUSES (WAF challenges) and handled above —
            # but keep the no-demotion rule as defense in depth.
            if is_excluded_status(http_status):
                logger.info("HTTP %d — marking failed: %s", http_status, url)
                if http_status == 403:
                    _mark_failure_preserving_crawled(db, url, "HTTP 403 (WAF challenge?)")
                else:
                    db.update_status(url, 'failed')
                continue

            # Gate 5: redirect target must still be in scope. The manifest
            # row stays keyed by the originally-requested URL; the final URL
            # is only used for the scope check and as the link-resolution
            # base below.
            final_url = result.get('final_url') or url
            link_base = url
            if final_url != url:
                resolved_final = normalize_url(final_url, base=url) or final_url
                redirected_excluded, redirect_reason = should_exclude(resolved_final)
                if redirected_excluded:
                    # A WAF challenge served as a 302 to an off-allowlist
                    # challenge page must not demote a previously-crawled
                    # row — same no-demotion rule as transient failures.
                    # Only rows that have never been crawled are excluded.
                    if db.get_page_status(url) == 'crawled':
                        logger.warning(
                            "REDIRECT off-allowlist on previously-crawled "
                            "page %s -> %s (%s) — keeping prior 'crawled' "
                            "row (possible WAF challenge redirect)",
                            url, final_url, redirect_reason,
                        )
                    else:
                        logger.info(
                            "REDIRECT off-allowlist: %s -> %s (%s)",
                            url, final_url, redirect_reason,
                        )
                        db.update_status(
                            url, 'excluded',
                            reason=f"redirect target excluded: {redirect_reason}",
                        )
                    continue
                link_base = resolved_final

            # Classify, then collect outbound links BEFORE persisting: the
            # child 'pending' rows must land in the same transaction as this
            # page's 'crawled' update — committing the parent first would
            # orphan the children if the run crashed in between (parent
            # never re-crawled on resume, children never enqueued).
            classification = classify_page(result)

            link_rows = []      # rows for the links table
            child_pages = []    # (url, parent_url, depth) pending rows
            # Resolve against the FINAL (post-redirect) URL of this page
            for link in result.get('links', []):
                norm = normalize_url(link['href'], base=link_base)
                if not norm:
                    continue

                # Exclude links before recording
                link_excluded, link_reason = should_exclude(norm)
                if link_excluded:
                    continue

                is_int = is_internal(norm)
                is_doc = get_content_type(norm) != 'html'

                link_rows.append((
                    url, norm, link.get('text', ''), link.get('context', ''),
                    int(is_int), int(is_doc),
                ))

                # Only queue internal links we haven't visited or already
                # enqueued (casefolded, so a case-variant of a known URL
                # doesn't get crawled as a separate page)
                if is_int and norm.lower() not in enqueued:
                    enqueued.add(norm.lower())
                    child_pages.append((norm, url, depth + 1))

            # One atomic transaction: crawled status + links + children
            db.update_page_with_children(
                url, result, classification, link_rows, child_pages,
            )
            pages_crawled += 1

            # Optionally save raw HTML
            if SAVE_RAW_HTML and result.get('raw_html'):
                _save_raw_html(url, result['raw_html'])

            for child_url, child_parent, child_depth in child_pages:
                queue.append((child_url, child_parent, child_depth))

        except Exception as exc:
            logger.error("Exception on %s: %s", url, exc, exc_info=True)
            fetch_failures += 1
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

    # A fresh crawl is the authority on what the site currently links to.
    # (A resumed crawl only covers the pending tail, so it must not sweep.)
    if not pending:
        _sweep_unreached_rows(db, fetched_exact, visited, fetch_failures)

    # --- Finalize ---
    db.finalize_run(run_id)
    logger.info("Pass 1 complete. Pages crawled: %d", pages_crawled)
    db.close()


def _sweep_unreached_rows(db: DB, fetched_exact: set[str],
                          visited: set[str], fetch_failures: int) -> None:
    """
    Exclude 'crawled' rows that a completed fresh crawl did not fetch, so
    Pass 2 never chunks stale cached content for them:

      - case-variant duplicates: a different casing of the same URL was
        fetched this run (both sites serve paths case-insensitively, but
        manifest rows are keyed case-sensitively, so a seed or link that
        changed casing leaves the old row behind with old content);
      - orphans: no in-scope page links to the URL any more, i.e. the site
        stopped publishing it.

    Exclusion is not sticky: if a later crawl discovers a link to the URL
    again, it is re-queued and its row returns to 'crawled'. Orphans are
    only swept when every fetch succeeded — a hub page that failed
    transiently would otherwise orphan all of its children for a run.
    """
    rows = db.conn.execute(
        "SELECT url FROM pages WHERE crawl_status = 'crawled'"
    ).fetchall()
    dupes, orphans = [], []
    for row in rows:
        url = row['url']
        if url in fetched_exact:
            continue
        (dupes if url.lower() in visited else orphans).append(url)

    for url in dupes:
        logger.warning("CASE-DUPLICATE row excluded (another casing was crawled): %s", url)
        db.update_status(url, 'excluded',
                         reason="case-variant duplicate of a page crawled this run")

    if orphans and fetch_failures:
        logger.warning(
            "%d crawled row(s) were not reached this run, but %d fetch(es) "
            "failed — keeping them rather than risk excluding children of a "
            "page that failed transiently:", len(orphans), fetch_failures,
        )
        for url in orphans:
            logger.warning("  unreached (kept): %s", url)
        return
    for url in orphans:
        logger.warning("ORPHANED row excluded (no longer linked from the site): %s", url)
        db.update_status(url, 'excluded',
                         reason="not linked from the site in the latest crawl")


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
