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

        # Upgrade http -> https on the target domains: both sites serve
        # https, and the allowlist prefixes are https:// strings, so an
        # absolute http:// internal link would otherwise be dropped. The
        # netloc is lowercased in the same step — an uppercase-host http://
        # link would otherwise pass the allowlist as a case-variant URL and
        # create duplicate manifest rows.
        if parsed.scheme == 'http' and parsed.netloc.lower() in DOMAINS:
            parsed = parsed._replace(scheme='https', netloc=parsed.netloc.lower())

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
        # .xlsx is an OOXML zip, not the legacy OLE .xls container: pass2
        # picks the payload magic check and the reader from this label, so
        # lumping them together rejects every valid .xlsx as corrupt.
        if ext == '.xls':
            return 'xls'
        if ext == '.xlsx':
            return 'xlsx'
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
