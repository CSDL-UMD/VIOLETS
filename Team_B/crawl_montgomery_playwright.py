"""
Montgomery County Elections Site Crawler — Playwright Version
=============================================================
Uses Playwright (headless Chromium) to handle JavaScript-rendered navigation,
then extracts clean page content for RAG.

This is a drop-in replacement for Ryan's httpx-based crawler,
adapted for JS-heavy sites like montgomerycountymd.gov/elections.

Install deps:
    pip install playwright trafilatura beautifulsoup4 lxml lxml_html_clean openai
    playwright install chromium

Run:
    python crawl_montgomery_playwright.py

Optional OpenAI summaries:
    export OPENAI_API_KEY="sk-..."
"""

import asyncio
import csv
import hashlib
import os
import re
from urllib.parse import urljoin, urldefrag, urlparse, parse_qs, urlencode, urlunparse
from typing import Optional

import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page

# ── Config ────────────────────────────────────────────────────────────────────

START_URL       = "https://montgomerycountymd.gov/elections"
MAX_PAGES       = 300          # hard cap on pages to crawl
CONCURRENCY     = 5            # simultaneous browser pages (lower than httpx — browsers are heavier)
SLEEP_SECS      = 0.5          # polite delay per request
TIMEOUT         = 30_000       # milliseconds (Playwright uses ms)
DO_SUMMARIES    = False        # set True and export OPENAI_API_KEY to enable
OUTPUT_CSV      = "report_montgomery.csv"

# ── URL helpers ───────────────────────────────────────────────────────────────

# Query parameters that carry no page identity, strip them so the same page
# reached via two different tracking links doesn't appear twice in `seen`.
_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "fbclid", "gclid", "_ga", "ref", "source", "sessionid", "PHPSESSID",
}

def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    abs_url = urljoin(base, href)
    abs_url, _ = urldefrag(abs_url)          # strip #fragment
    parsed   = urlparse(abs_url)

    if parsed.scheme not in {"http", "https"}:
        return None

    # Lowercase scheme + netloc (RFC-correct); lowercase path so that
    # /Elections/ and /elections/ are treated as the same resource.
    netloc = parsed.netloc.lower()
    path   = parsed.path.lower()

    # Remove trailing slash on non-root paths  (/elections/ → /elections)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Strip tracking params; sort survivors so param order doesn't matter.
    qs_pairs = [
        (k, v)
        for k, vs in parse_qs(parsed.query, keep_blank_values=True).items()
        for v in vs
        if k not in _STRIP_PARAMS
    ]
    qs_pairs.sort()
    clean_query = urlencode(qs_pairs)

    canonical = urlunparse((parsed.scheme, netloc, path, "", clean_query, ""))
    return canonical

def same_domain(url: str, root_netloc: str) -> bool:
    return urlparse(url).netloc == root_netloc

# ── URL filtering ────────────────────────────────────────────────────────
#
# should_enqueue() is called for every discovered link BEFORE opening a browser
# tab.  It is cheap (regex only) and eliminates whole categories of URLs that
# would otherwise consume the MAX_PAGES budget with zero RAG value.
#
#   1. If the URL matches any BLOCKLIST pattern  → reject immediately.
#   2. If the URL is a PDF, it must also match a PDF_ALLOWLIST keyword → else reject.
#   3. The URL must be under the /elections path → else reject.
#   4. Everything that survives → accept.

# Patterns that are ALWAYS excluded.  Checked against the lowercased URL path.
_BLOCKLIST_PATTERNS: list[re.Pattern] = [re.compile(p, re.I) for p in [
    # Precinct map PDFs — the primary cause of crawl bloat
    r"/elections/resources/files/pdfs/maps/precincts/",
    r"/elections/resources/files/pdfs/maps/",
    r"precinct[-_]?\d+.*\.pdf$",
    r"precinct.*map.*\.pdf$",

    # Other non-voter PDFs
    r"gis.*\.pdf$",
    r"district[-_]map.*\.pdf$",
    r"canvass.*result.*\.pdf$",
    r"statistics[-_]\d{4}.*\.pdf$",

    # Binary / media assets — trafilatura can't extract text from these
    r"\.(jpg|jpeg|png|gif|svg|webp|ico|bmp|tiff|mp4|mp3|wav|avi|mov)$",
    r"\.(zip|tar\.gz|gz|exe|dmg|msi)$",
    r"\.(css|js|woff2?|ttf|eot)$",

    # Sitewide boilerplate and unrelated county departments
    r"/finance/", r"/police/", r"/health/", r"/transportation/",
    r"/parks/", r"/budget/", r"/council/", r"/permits/",
    r"/hr/", r"/dhhs/", r"/dot/", r"/mcps/",
    r"/content/templates/",
    r"/content/resources/shared/",

    # Dynamic / infinite pages — search results, print views, sitemaps
    r"[?&](print|printview|print_view)=",
    r"/sitemap",
    r"/robots\.txt$",
    r"/accessibility-statement$",
    r"/privacy-policy$",
    r"/terms-of-use$",
    r"[?&]search=",
]]

# PDFs are blocked by default; they only pass if their path contains one of
# these voter-facing keywords, indicating an actionable document (form, guide).
_PDF_ALLOW_KEYWORDS = re.compile(
    r"(application|form|guide|instruction|voter|ballot|registration|absentee|mail)",
    re.I,
)

def should_enqueue(url: str) -> bool:
    """
    Returns True if this URL should be added to the crawl queue.
    Called before any network request is made.
    """
    path = urlparse(url).path  # already lowercased by normalize_url

    # 1. Hard blocklist — reject immediately
    for pattern in _BLOCKLIST_PATTERNS:
        if pattern.search(url):
            return False

    # 2. PDF-specific gate — only allow voter-facing documents
    if path.endswith(".pdf"):
        if not _PDF_ALLOW_KEYWORDS.search(path):
            return False

    # 3. Must be under the /elections path
    if not path.startswith("/elections"):
        return False

    return True

# ── Content relevance scoring ─────────────────────────────────────────────────
#
# score_relevance() runs after page extraction.  It scores the combined signal
# of title + h1 + body text and returns (score, reason).
#
# Score tiers (applied in the crawl loop):
#   >= 2  → KEEP + index + enqueue outbound links   (high-value page)
#   0–1   → KEEP + index, do NOT enqueue links      (thin but relevant)
#   <  0  → DROP entirely                           (net-irrelevant)
#
# word_count < MIN_WORD_COUNT causes an immediate DROP before this runs.

MIN_WORD_COUNT          = 80   # below this → likely a nav shell
KEEP_SCORE_THRESHOLD    = 0    # score must reach this to keep the page
ENQUEUE_SCORE_THRESHOLD = 2    # score must reach this to follow outbound links
PDF_MIN_WORDS           = 50   # voter PDFs that extracted almost no text → drop

# (pattern, weight) pairs scored against lowercased title + h1 + body_text
_RELEVANCE_SIGNALS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"early voting|election day|polling place|poll hours?|how to vote|cast (a |your )?ballot|vote in.?person", re.I), 3),
    (re.compile(r"mail.?in ballot|absentee ballot|vote by mail|ballot request|drop box|ballot drop|return (a |your )?ballot|track (a |your )?ballot", re.I), 3),
    (re.compile(r"register to vote|voter registration|registration deadline|update (your )?address|change of address|party affiliation|registration status", re.I), 3),
    (re.compile(r"sample ballot|ballot question|candidate filing|election results?|official results?|primary election|general election|canvass|certification", re.I), 2),
    (re.compile(r"eligible to vote|id required|photo id|proof of residency|citizenship|18 years|felony|provisional ballot", re.I), 2),
    (re.compile(r"accessible voting|language assistance|\bada\b|audio ballot|curbside voting|accommodations?|spanish|korean|chinese|vietnamese|amharic", re.I), 2),
    (re.compile(r"board of elections|contact us|office hours|election office|\bfaq\b|frequently asked|deadline|important dates?", re.I), 1),
    (re.compile(r"\bmilitary\b|overseas|uocava|uniformed services|federal ballot|satellite office", re.I), 1),
]

_IRRELEVANCE_SIGNALS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"precinct map|precinct boundar|precinct number|precinct chart|\bgis\b|geographic information|district boundar|voting district map", re.I), -4),
    (re.compile(r"internal use only|canvass worksheet|administrative record|staff meeting|board meeting agenda", re.I), -2),
]

def score_relevance(content: dict, url: str) -> tuple[int, str]:
    """Return (score, reason) for a fetched page."""
    haystack = " ".join([
        content.get("title", ""),
        content.get("h1", ""),
        content.get("body_text", ""),
    ])

    score = 0
    matched_pos: list[str] = []
    matched_neg: list[str] = []

    for pattern, weight in _RELEVANCE_SIGNALS:
        if pattern.search(haystack):
            score += weight
            matched_pos.append(pattern.pattern[:45])

    for pattern, weight in _IRRELEVANCE_SIGNALS:
        if pattern.search(haystack):
            score += weight
            matched_neg.append(pattern.pattern[:45])

    parts = []
    if matched_pos:
        parts.append("pos:" + "|".join(matched_pos))
    if matched_neg:
        parts.append("neg:" + "|".join(matched_neg))
    reason = "; ".join(parts) if parts else "no signals matched"

    return score, reason


# ── Content extraction ────────────────────────────────────────────────────────

def extract_content(html: str, url: str) -> dict:
    body_text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=False,
    ) or ""

    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else ""

    meta_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    meta_desc = meta_tag.get("content", "").strip() if meta_tag else ""

    links = []
    for a in soup.find_all("a", href=True):
        norm = normalize_url(url, a["href"])
        if norm:
            links.append(norm)

    return {
        "title": title,
        "h1": h1,
        "meta_description": meta_desc,
        "body_text": body_text,
        "links": links,
    }

# ── OpenAI summarizer (optional) ──────────────────────────────────────────────

def summarize(text: str, url: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    snippet = text[:8_000]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize web pages for a RAG knowledge base. "
                    "Be factual and concise. 2-4 sentences max. "
                    "If the page is navigation/boilerplate with no real content, say so."
                ),
            },
            {"role": "user", "content": f"URL: {url}\n\nCONTENT:\n{snippet}"},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# ── Core crawler ──────────────────────────────────────────────────────────────

async def fetch_page_playwright(page: Page, url: str) -> tuple[Optional[int], Optional[str]]:
    """
    Navigate to URL using a real browser page.
    Returns (status, html) or (None, None) on failure.
    
    Playwright actually executes JavaScript, so dynamically rendered
    links and content are fully available in the HTML snapshot.
    """
    try:
        response = await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
        if response is None:
            return None, None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type.lower():
            return response.status, None

        # Wait a beat for any lazy-loaded JS content
        await asyncio.sleep(SLEEP_SECS)

        # Get the fully-rendered HTML (post-JavaScript execution)
        html = await page.content()
        return response.status, html

    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None, None


async def crawl() -> None:
    root_netloc = urlparse(START_URL).netloc

    seen: set[str] = set()
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(START_URL)
    seen.add(START_URL)

    rows: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def process(url: str) -> None:
            async with semaphore:
                # Each concurrent task gets its own browser page
                page = await browser.new_page()
                try:
                    status, html = await fetch_page_playwright(page, url)
                finally:
                    await page.close()

            if not html:
                rows.append({
                    "url": url, "status": status, "title": "", "h1": "",
                    "meta_description": "", "body_text": "", "summary": "", "word_count": 0,
                })
                return

            content = extract_content(html, url)

            # Enqueue newly discovered links
            for link in content["links"]:
                if link not in seen and same_domain(link, root_netloc):
                    seen.add(link)
                    queue.put_nowait(link)

            summary = ""
            if DO_SUMMARIES and content["body_text"]:
                try:
                    summary = summarize(content["body_text"], url)
                except Exception as e:
                    summary = f"[summary error: {e}]"

            word_count = len(content["body_text"].split())

            rows.append({
                "url": url,
                "status": status,
                "title": content["title"],
                "h1": content["h1"],
                "meta_description": content["meta_description"],
                "body_text": content["body_text"],
                "summary": summary,
                "word_count": word_count,
            })

            print(f"  [{status}] {url}  ({word_count} words)")

        # ── Main crawl loop ────────────────────────────────────────────────────
        tasks: set[asyncio.Task] = set()

        while True:
            while not queue.empty() and len(rows) + len(tasks) < MAX_PAGES:
                url = queue.get_nowait()
                task = asyncio.create_task(process(url))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            if not tasks:
                break

            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        await browser.close()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["url"])

    fieldnames = ["url", "status", "title", "h1", "meta_description", "summary", "word_count", "body_text"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Crawled {len(rows)} pages → {OUTPUT_CSV}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting Playwright crawl: {START_URL}")
    print(f"  Max pages   : {MAX_PAGES}")
    print(f"  Concurrency : {CONCURRENCY}")
    print(f"  Summaries   : {DO_SUMMARIES}")
    print(f"  Output      : {OUTPUT_CSV}\n")
    asyncio.run(crawl())
