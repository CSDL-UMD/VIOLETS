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
import os
import re
from urllib.parse import urljoin, urldefrag, urlparse
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

def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None
    abs_url = urljoin(base, href)
    abs_url, _ = urldefrag(abs_url)
    scheme = urlparse(abs_url).scheme
    if scheme not in {"http", "https"}:
        return None
    return abs_url

def same_domain(url: str, root_netloc: str) -> bool:
    return urlparse(url).netloc == root_netloc

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
