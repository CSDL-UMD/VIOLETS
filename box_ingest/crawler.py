"""
Box crawler for State Board of Elections materials.

Strategy:
  1. Playwright scrapes the public Box Hub page to extract folder IDs from
     the folder URLs visible in the page (no auth needed — the hub is public).
  2. For each folder ID, the Box API lists its contents using the top-level
     shared link as context (the single valid shared link for all calls).
  3. Each file is classified by filter.py rules.

Requires environment variables:
    BOX_CLIENT_ID      — from your Box developer app
    BOX_CLIENT_SECRET  — from your Box developer app

The first run opens a browser for OAuth login and saves a token to
.box_token (gitignored). Subsequent runs reuse the saved token.

Usage (standalone, for testing):
    python -m box_ingest.crawler
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import webbrowser
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv

from box_ingest.filter import classify_filename, FILTER_INCLUDE, FILTER_EXCLUDE, FILTER_REVIEW
from box_ingest.paths import NEEDTOCHUNK_DIR, TOKEN_CACHE

load_dotenv()
logger = logging.getLogger(__name__)

HUB_URL           = "https://mdsbe.app.box.com/hubs/263564910?s=ly67mqf875239kr4otek9phuueqzxw3r"
SHARED_LINK_TOKEN = "ly67mqf875239kr4otek9phuueqzxw3r"
# This is the ONE valid shared link used as context for all Box API calls
SHARED_LINK       = f"https://mdsbe.app.box.com/s/{SHARED_LINK_TOKEN}"
OAUTH_REDIRECT    = "http://localhost:8080"


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

def _save_tokens(access: str, refresh: str) -> None:
    # Tokens include a long-lived refresh token; create the file 0o600 so it is
    # never briefly world-readable on a shared host.
    data = json.dumps({"access_token": access, "refresh_token": refresh})
    fd = os.open(TOKEN_CACHE, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)


def _load_tokens() -> dict | None:
    """Return cached tokens, or None when the cache is missing or unusable.

    A corrupt/truncated .box_token must never crash the run — any load
    failure (bad JSON, missing keys, unreadable file) is treated as "no
    cached token": the bad file is dropped and the normal re-auth flow
    takes over.
    """
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("refresh_token"):
            raise ValueError("token cache has no refresh_token")
        return data
    except (OSError, ValueError) as exc:  # JSONDecodeError subclasses ValueError
        logger.warning("Ignoring unusable token cache %s (%s); re-authorizing", TOKEN_CACHE, exc)
        try:
            TOKEN_CACHE.unlink()
        except OSError:
            pass
        return None


class _CallbackHandler(BaseHTTPRequestHandler):
    auth_code: str | None = None
    def do_GET(self):
        _CallbackHandler.auth_code = parse_qs(urlparse(self.path).query).get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h1>Authorized. You can close this tab.</h1>")
    def log_message(self, *_): pass


def _exchange_code(code: str, client_id: str, client_secret: str) -> tuple[str, str]:
    body = urlencode({"grant_type": "authorization_code", "code": code,
                      "client_id": client_id, "client_secret": client_secret,
                      "redirect_uri": OAUTH_REDIRECT}).encode()
    with urllib.request.urlopen(urllib.request.Request(
        "https://api.box.com/oauth2/token", data=body, method="POST"
    ), timeout=30) as r:
        d = json.loads(r.read())
    return d["access_token"], d["refresh_token"]


def _refresh(client_id: str, client_secret: str, refresh_token: str) -> tuple[str, str]:
    body = urlencode({"grant_type": "refresh_token", "refresh_token": refresh_token,
                      "client_id": client_id, "client_secret": client_secret}).encode()
    with urllib.request.urlopen(urllib.request.Request(
        "https://api.box.com/oauth2/token", data=body, method="POST"
    ), timeout=30) as r:
        d = json.loads(r.read())
    return d["access_token"], d["refresh_token"]


def get_access_token() -> str:
    client_id     = os.environ.get("BOX_CLIENT_ID", "")
    client_secret = os.environ.get("BOX_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError("BOX_CLIENT_ID and BOX_CLIENT_SECRET must be set in .env")

    cached = _load_tokens()
    if cached:
        try:
            access, refresh = _refresh(client_id, client_secret, cached["refresh_token"])
            _save_tokens(access, refresh)
            return access
        except Exception as exc:
            logger.warning("Token refresh failed (%s); re-authorizing", exc)

    auth_url = f"https://account.box.com/api/oauth2/authorize?{urlencode({'response_type': 'code', 'client_id': client_id, 'redirect_uri': OAUTH_REDIRECT})}"
    print(f"\nOpening browser for Box authorization:\n  {auth_url}\n")
    webbrowser.open(auth_url)
    server = HTTPServer(("localhost", 8080), _CallbackHandler)
    print("Waiting for Box to redirect back to localhost:8080 ...")
    server.handle_request()
    code = _CallbackHandler.auth_code
    if not code:
        raise RuntimeError("OAuth flow failed — no code received")
    access, refresh = _exchange_code(code, client_id, client_secret)
    _save_tokens(access, refresh)
    return access


# ---------------------------------------------------------------------------
# Box API
# ---------------------------------------------------------------------------

def _api_get(path: str, access_token: str) -> dict:
    """GET /2.0/<path> with the shared link context header on every call."""
    url = f"https://api.box.com/2.0/{path.lstrip('/')}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("BoxApi", f"shared_link={SHARED_LINK}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _list_folder(folder_id: str, access_token: str) -> list[dict]:
    items, offset, limit = [], 0, 1000
    while True:
        # sha1 + size let automate.py detect files updated in Box and verify
        # downloads; both come back only for file entries (folders lack them).
        data = _api_get(
            f"folders/{folder_id}/items?limit={limit}&offset={offset}&fields=id,name,type,sha1,size",
            access_token,
        )
        items.extend(data.get("entries", []))
        if offset + limit >= data.get("total_count", 0):
            break
        offset += limit
    return items


# ---------------------------------------------------------------------------
# Playwright hub scraper — extracts folder IDs from the hub page
# ---------------------------------------------------------------------------

def _scrape_hub_folder_ids() -> list[str]:
    """
    Load the Box Hub page with Playwright and extract all folder IDs from
    URLs matching /s/<token>/folder/<id>.
    Returns a deduplicated list of folder ID strings.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError("Run: pip install playwright && playwright install chromium")

    folder_ids: list[str] = []
    pattern = re.compile(r"/folder/(\d+)")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        logger.info("Loading Box Hub page ...")
        page.goto(HUB_URL, wait_until="networkidle", timeout=30000)

        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        for href in hrefs:
            m = pattern.search(href)
            if m:
                fid = m.group(1)
                if fid not in folder_ids:
                    folder_ids.append(fid)
                    logger.debug("Found folder ID: %s  (%s)", fid, href)

        browser.close()

    logger.info("Found %d top-level folder(s) on hub", len(folder_ids))
    return folder_ids


# ---------------------------------------------------------------------------
# Public dataclass + crawl entry point
# ---------------------------------------------------------------------------

@dataclass
class BoxFile:
    file_id:     str
    name:        str
    folder_path: str   # e.g. "2026-03" or "2026-03/subfolder"
    box_url:     str   # direct share URL for the file
    decision:    str   # FILTER_INCLUDE | FILTER_EXCLUDE | FILTER_REVIEW
    sha1:        str = ""   # Box-reported content sha1 ("" if not returned)
    size:        int = -1   # Box-reported byte size (-1 if not returned)


def _is_2026_folder(name: str, depth: int) -> bool:
    """At the top level (depth=0), only enter folders whose name starts with '2026-'."""
    return depth != 0 or name.startswith("2026-")


def _walk_folder(folder_id: str, folder_path: str, access_token: str, results: list[BoxFile], depth: int = 0) -> None:
    for item in _list_folder(folder_id, access_token):
        name = item["name"]
        # Box names are externally writable; reject ones that could escape the
        # download root once joined onto NEEDTOCHUNK_DIR downstream.
        if name == ".." or "/" in name or "\\" in name:
            logger.warning("Skipping Box item with unsafe name: %r", name)
            continue
        if item["type"] == "folder":
            if not _is_2026_folder(name, depth):
                logger.debug("Skipping non-2026 folder: %s", name)
                continue
            sub = f"{folder_path}/{name}".lstrip("/")
            candidate = (NEEDTOCHUNK_DIR / sub).resolve()
            if not candidate.is_relative_to(NEEDTOCHUNK_DIR.resolve()):
                logger.warning("Skipping folder whose path escapes download root: %s", sub)
                continue
            _walk_folder(item["id"], sub, access_token, results, depth + 1)
        elif item["type"] == "file":
            results.append(BoxFile(
                file_id=item["id"],
                name=name,
                folder_path=folder_path,
                box_url=f"{SHARED_LINK}/file/{item['id']}",
                decision=classify_filename(name),
                sha1=item.get("sha1") or "",
                size=int(item["size"]) if item.get("size") is not None else -1,
            ))


def crawl_hub(access_token: str | None = None) -> list[BoxFile]:
    """
    Walk the entire Box Hub and return a BoxFile for every file found.
    Scrapes folder IDs from the hub page, then uses the Box API to list contents.
    """
    if access_token is None:
        access_token = get_access_token()

    folder_ids = _scrape_hub_folder_ids()
    if not folder_ids:
        raise RuntimeError("No folder IDs found on the Box Hub page")

    results: list[BoxFile] = []
    for fid in folder_ids:
        logger.info("Walking folder %s ...", fid)
        try:
            _walk_folder(fid, "", access_token, results)
        except Exception as exc:
            logger.warning("Could not walk folder %s: %s", fid, exc)

    logger.info("Crawl complete: %d files found", len(results))
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = crawl_hub()
    for decision, label in [(FILTER_INCLUDE, "INCLUDE"), (FILTER_REVIEW, "REVIEW"), (FILTER_EXCLUDE, "EXCLUDE")]:
        group = [f for f in files if f.decision == decision]
        print(f"\n--- {label} ({len(group)}) ---")
        for f in group:
            print(f"  {f.folder_path}/{f.name}")
