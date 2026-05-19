"""
Content extraction for HTML pages and document metadata.

Key improvements over the original design:
- Single HTTP fetch (requests) passed to trafilatura — no double-fetch.
- BeautifulSoup fallback when trafilatura returns suspiciously few words.
- Lightweight PDF probe to detect image-only PDFs in Pass 1.
"""
import hashlib
import logging
import re

import requests
import trafilatura
from bs4 import BeautifulSoup

from .config import (
    REQUEST_TIMEOUT,
    TRAFILATURA_MIN_WORDS,
    PDF_PROBE_BYTES,
)
from .utils import get_content_type

logger = logging.getLogger(__name__)


def extract_page(url: str) -> dict | None:
    """
    Main entry point. Routes to HTML or document-metadata extraction
    based on URL file extension.
    """
    ctype = get_content_type(url)
    if ctype != 'html':
        return _extract_document_metadata(url, ctype)
    return _extract_html(url)


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def _extract_html(url: str) -> dict | None:
    """Fetch an HTML page, extract text + metadata + links."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        http_status = resp.status_code
        raw_html = resp.text
    except requests.RequestException as exc:
        logger.warning("HTTP error fetching %s: %s", url, exc)
        return None

    if not raw_html:
        return None

    # --- Primary extraction via trafilatura ---
    metadata = trafilatura.extract_metadata(raw_html)
    text = trafilatura.extract(
        raw_html,
        include_links=True,
        include_tables=True,
        include_images=False,
        output_format='txt',  # plain text for word counting / snippets
    )

    word_count = len(text.split()) if text else 0

    # --- Fallback: if trafilatura returned too little, try BeautifulSoup ---
    if word_count < TRAFILATURA_MIN_WORDS:
        fallback_text = _fallback_extract(raw_html)
        fallback_wc = len(fallback_text.split()) if fallback_text else 0
        if fallback_wc > word_count:
            logger.info(
                "Trafilatura returned %d words for %s; BS4 fallback got %d",
                word_count, url, fallback_wc,
            )
            text = fallback_text
            word_count = fallback_wc

    soup = BeautifulSoup(raw_html, 'html.parser')
    links = _extract_links_with_context(soup)
    breadcrumb = _extract_breadcrumb(soup, url)

    content_hash = hashlib.sha256((text or '').encode()).hexdigest()

    return {
        'url': url,
        'title': metadata.title if metadata else _title_from_soup(soup),
        'date': metadata.date if metadata else None,
        'section_hierarchy': breadcrumb,
        'raw_html': raw_html,
        'text': text,
        'word_count': word_count,
        'links': links,
        'http_status': http_status,
        'content_hash': content_hash,
        'snippet': (text or '')[:500] or None,
        'content_type': 'html',
        'file_size_bytes': None,
        'needs_ocr': False,
    }


def _fallback_extract(html: str) -> str | None:
    """BeautifulSoup plain-text fallback for pages trafilatura mishandles."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script/style/nav/footer noise
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        # Try to grab the main content area
        main = soup.find('main') or soup.find('article') or soup.find(id='content')
        target = main if main else soup.body
        if target:
            return target.get_text(separator=' ', strip=True)
    except Exception:
        pass
    return None


def _title_from_soup(soup: BeautifulSoup) -> str | None:
    tag = soup.find('title')
    return tag.get_text(strip=True) if tag else None


# ---------------------------------------------------------------------------
# Link & breadcrumb extraction
# ---------------------------------------------------------------------------

def _extract_links_with_context(soup: BeautifulSoup) -> list[dict]:
    links = []
    for a in soup.find_all('a', href=True):
        parent_text = ''
        if a.parent:
            parent_text = a.parent.get_text(strip=True)[:200]
        links.append({
            'href': a['href'],
            'text': a.get_text(strip=True),
            'context': parent_text,
        })
    return links


def _extract_breadcrumb(soup: BeautifulSoup, url: str) -> list[str]:
    """Try several common breadcrumb selectors, then fall back to URL path."""
    selectors = [
        'nav.breadcrumb', '.breadcrumbs', '[aria-label="breadcrumb"]',
        'ol.breadcrumb', '.breadcrumb-trail', '#breadcrumb',
    ]
    for selector in selectors:
        bc = soup.select(selector)
        if bc:
            items = bc[0].find_all(['li', 'a', 'span'])
            crumbs = [i.get_text(strip=True) for i in items if i.get_text(strip=True)]
            if crumbs:
                return crumbs

    # Fallback: derive from URL path segments
    from urllib.parse import urlparse
    path = urlparse(url).path
    parts = [
        p.replace('-', ' ').replace('_', ' ').title()
        for p in path.strip('/').split('/') if p
    ]
    return parts


# ---------------------------------------------------------------------------
# Document metadata (PDFs, DOCX, etc.) — no full extraction in Pass 1
# ---------------------------------------------------------------------------

def _extract_document_metadata(url: str, ctype: str) -> dict | None:
    """
    For documents, we only collect metadata in Pass 1:
    file size from HEAD, and for PDFs a lightweight text-extractability probe.
    """
    try:
        resp = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        http_status = resp.status_code
        file_size = int(resp.headers.get('Content-Length', 0) or 0)
    except requests.RequestException as exc:
        logger.warning("HEAD request failed for %s: %s", url, exc)
        return None

    needs_ocr = False
    if ctype == 'pdf':
        needs_ocr = _probe_pdf_text_extractable(url)

    # Derive a human-readable title from the filename
    filename = url.split('/')[-1].split('?')[0]
    title = filename.replace('_', ' ').replace('-', ' ')
    if '.' in title:
        title = title.rsplit('.', 1)[0]

    # Derive section hierarchy from URL path (excluding filename)
    from urllib.parse import urlparse
    path = urlparse(url).path
    parts = path.strip('/').split('/')
    hierarchy = [
        p.replace('-', ' ').replace('_', ' ').title()
        for p in parts[:-1] if p
    ]

    return {
        'url': url,
        'title': title,
        'section_hierarchy': hierarchy,
        'content_type': ctype,
        'http_status': http_status,
        'file_size_bytes': file_size,
        'word_count': None,
        'links': [],
        'content_hash': None,
        'snippet': None,
        'needs_ocr': needs_ocr,
    }


def _probe_pdf_text_extractable(url: str) -> bool:
    """
    Download the first few KB of a PDF and check for text stream markers.
    Returns True if the PDF appears to be image-only (needs OCR).
    Returns False if text content is detected.
    """
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={'Range': f'bytes=0-{PDF_PROBE_BYTES}'},
            stream=True,
        )
        try:
            # If the server honored the Range request we'll see 206 Partial
            # Content; anything else (200 OK from a CDN that ignores Range,
            # 4xx/5xx errors) means resp.content would download the full
            # PDF, so read just one chunk and discard the rest.
            if resp.status_code == 206:
                chunk = resp.content
            else:
                chunk = next(
                    resp.iter_content(chunk_size=PDF_PROBE_BYTES),
                    b'',
                )
        finally:
            resp.close()

        # Look for text stream markers in PDF binary
        # /Type /Page + stream content with text operators (Tj, TJ, Tf)
        # or /Font references indicate text-extractable content
        has_text_markers = (
            b'/Font' in chunk
            or b'/Text' in chunk
            or b'Tj' in chunk
            or b'TJ' in chunk
            or b'/ToUnicode' in chunk
        )
        return not has_text_markers
    except Exception:
        # If probe fails, assume text-extractable (optimistic default)
        return False
