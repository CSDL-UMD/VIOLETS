"""
URL normalization, domain checks, and file type detection utilities.
"""
import os
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode

from .config import DOMAINS, DOCUMENT_EXTENSIONS, SKIP_DOMAINS


def normalize_url(href: str, base: str) -> str | None:
    """
    Resolve a potentially-relative href against a base URL.
    Strips fragments, normalizes trailing slashes, filters junk schemes.
    Returns None if the URL should be discarded.
    """
    if not href or not href.strip():
        return None

    href = href.strip()

    # Skip non-HTTP schemes early
    if href.startswith(('mailto:', 'tel:', 'javascript:', '#', 'data:')):
        return None

    try:
        full = urljoin(base, href)
        parsed = urlparse(full)

        # Only HTTP(S)
        if parsed.scheme not in ('http', 'https'):
            return None

        # Skip social / external skip-domains
        if any(skip in parsed.netloc for skip in SKIP_DOMAINS):
            return None

        # Drop fragment
        clean_parsed = parsed._replace(fragment='')

        # Normalize: sort query params for consistency
        if clean_parsed.query:
            qs = parse_qs(clean_parsed.query, keep_blank_values=True)
            sorted_qs = urlencode(sorted(qs.items()), doseq=True)
            clean_parsed = clean_parsed._replace(query=sorted_qs)

        return urlunparse(clean_parsed)
    except Exception:
        return None


def is_internal(url: str) -> bool:
    """Check if a URL belongs to one of the target domains."""
    try:
        netloc = urlparse(url).netloc
        return any(d in netloc for d in DOMAINS)
    except Exception:
        return False


def get_content_type(url: str) -> str:
    """Determine content type from URL file extension."""
    try:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        if ext == '.docx':
            return 'docx'
        if ext == '.doc':
            # Legacy OLE format — python-docx cannot parse it, so it must
            # never be labeled 'docx'. Excluded from crawling in exclusions.py.
            return 'doc'
        if ext in ('.xls', '.xlsx'):
            return 'xls'
        if ext == '.csv':
            return 'csv'
    except Exception:
        pass
    return 'html'


def is_document_url(url: str) -> bool:
    """Check if a URL points to a downloadable document."""
    try:
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        return ext in DOCUMENT_EXTENSIONS
    except Exception:
        return False
