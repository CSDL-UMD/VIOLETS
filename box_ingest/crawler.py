"""
Box API crawler for State Board of Elections materials.

Connects to the Box shared hub, lists all files, classifies each filename
using filter.py rules, and returns structured results for the orchestrator.

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
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv

from box_ingest.filter import classify_filename, FILTER_INCLUDE, FILTER_EXCLUDE, FILTER_REVIEW

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).parent.parent
TOKEN_CACHE     = PROJECT_ROOT / ".box_token"

# The public shared hub URL
HUB_SHARED_LINK = "https://mdsbe.app.box.com/hubs/263564910?s=ly67mqf875239kr4otek9phuueqzxw3r"
# Shared link token extracted from the URL above (the `s=` param)
SHARED_LINK_TOKEN = "ly67mqf875239kr4otek9phuueqzxw3r"

OAUTH_REDIRECT_URI = "http://localhost:8080"


# ---------------------------------------------------------------------------
# OAuth token cache helpers
# ---------------------------------------------------------------------------

def _save_tokens(access_token: str, refresh_token: str) -> None:
    TOKEN_CACHE.write_text(
        json.dumps({"access_token": access_token, "refresh_token": refresh_token}),
        encoding="utf-8",
    )


def _load_tokens() -> dict | None:
    if TOKEN_CACHE.exists():
        return json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# One-shot local HTTP server to capture the OAuth redirect code
# ---------------------------------------------------------------------------

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code: str | None = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        _OAuthCallbackHandler.auth_code = params.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h1>Authorized. You can close this tab.</h1>")

    def log_message(self, *_):  # silence request logs
        pass


def _run_oauth_flow(client_id: str, client_secret: str) -> tuple[str, str]:
    """Open browser for Box OAuth, capture code, exchange for tokens."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": OAUTH_REDIRECT_URI,
    }
    auth_url = f"https://account.box.com/api/oauth2/authorize?{urlencode(params)}"
    print(f"\nOpening browser for Box authorization:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8080), _OAuthCallbackHandler)
    print("Waiting for Box to redirect back to localhost:8080 ...")
    server.handle_request()  # handles exactly one request then returns

    code = _OAuthCallbackHandler.auth_code
    if not code:
        raise RuntimeError("OAuth flow failed — no code received from Box")

    # Exchange code for tokens
    import urllib.request
    body = urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": OAUTH_REDIRECT_URI,
    }).encode()
    req = urllib.request.Request(
        "https://api.box.com/oauth2/token",
        data=body,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["access_token"], data["refresh_token"]


def _refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> tuple[str, str]:
    import urllib.request
    body = urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        "https://api.box.com/oauth2/token",
        data=body,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["access_token"], data["refresh_token"]


def get_access_token() -> str:
    """Return a valid Box access token, refreshing or re-authorizing as needed."""
    client_id     = os.environ.get("BOX_CLIENT_ID", "")
    client_secret = os.environ.get("BOX_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "BOX_CLIENT_ID and BOX_CLIENT_SECRET must be set in .env"
        )

    cached = _load_tokens()
    if cached:
        try:
            access, refresh = _refresh_access_token(
                client_id, client_secret, cached["refresh_token"]
            )
            _save_tokens(access, refresh)
            logger.debug("Box token refreshed successfully")
            return access
        except Exception as exc:
            logger.warning("Token refresh failed (%s); re-authorizing", exc)

    access, refresh = _run_oauth_flow(client_id, client_secret)
    _save_tokens(access, refresh)
    return access


# ---------------------------------------------------------------------------
# Box API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, access_token: str, shared_link: str | None = None) -> dict:
    """GET https://api.box.com/2.0/<path> and return parsed JSON."""
    import urllib.request
    url = f"https://api.box.com/2.0/{path.lstrip('/')}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {access_token}")
    if shared_link:
        req.add_header("BoxApi", f"shared_link={shared_link}&shared_link_password=")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _list_folder(folder_id: str, access_token: str, shared_link: str) -> list[dict]:
    """Return all items (files + subfolders) in a Box folder, handling pagination."""
    items, offset, limit = [], 0, 1000
    while True:
        data = _api_get(
            f"folders/{folder_id}/items?limit={limit}&offset={offset}&fields=id,name,type,shared_link",
            access_token,
            shared_link,
        )
        items.extend(data.get("entries", []))
        total = data.get("total_count", 0)
        offset += limit
        if offset >= total:
            break
    return items


def _get_shared_item(access_token: str, shared_link: str) -> dict:
    """Resolve the Box item (folder, file, or hub) from a shared link."""
    return _api_get("shared_items", access_token, shared_link)


def _get_hub_folder_ids(hub_id: str, access_token: str) -> list[str]:
    """
    A Box Hub is a collection of sections, each section points to a folder.
    Returns the list of folder IDs that make up the hub's content.
    """
    data = _api_get(f"hubs/{hub_id}/sections?limit=1000&fields=id,name,folder", access_token)
    folder_ids = []
    for section in data.get("entries", []):
        folder = section.get("folder")
        if folder and folder.get("id"):
            logger.debug("Hub section '%s' → folder %s", section.get("name"), folder["id"])
            folder_ids.append(folder["id"])
    return folder_ids


def _make_file_url(file_id: str) -> str:
    """Construct a direct Box share URL for a file given its ID."""
    return f"https://mdsbe.app.box.com/s/{SHARED_LINK_TOKEN}/file/{file_id}"


# ---------------------------------------------------------------------------
# Public dataclass + main crawl function
# ---------------------------------------------------------------------------

@dataclass
class BoxFile:
    file_id:    str
    name:       str
    folder_path: str   # e.g. "2026-01" or "2026-02/subfolder"
    box_url:    str
    decision:   str    # FILTER_INCLUDE | FILTER_EXCLUDE | FILTER_REVIEW


def crawl_hub(access_token: str | None = None) -> list[BoxFile]:
    """
    Walk the entire shared Box hub and return a BoxFile for every file found,
    classified by filter rules.

    If access_token is None, one is obtained automatically via get_access_token().
    """
    if access_token is None:
        access_token = get_access_token()

    shared_link = HUB_SHARED_LINK
    logger.info("Resolving shared hub ...")
    item = _get_shared_item(access_token, shared_link)
    item_type = item.get("type")
    item_id   = item["id"]
    logger.info("Shared item type=%s id=%s", item_type, item_id)

    results: list[BoxFile] = []

    if item_type == "folder":
        # Simple case: shared link points directly to a folder
        _walk_folder(item_id, "", access_token, shared_link, results)
    else:
        # Hub: resolve sections → folders, then walk each folder
        folder_ids = _get_hub_folder_ids(item_id, access_token)
        logger.info("Hub has %d section folder(s)", len(folder_ids))
        for folder_id in folder_ids:
            _walk_folder(folder_id, "", access_token, shared_link, results)

    logger.info("Crawl complete: %d files found", len(results))
    return results


def _walk_folder(
    folder_id: str,
    folder_path: str,
    access_token: str,
    shared_link: str,
    results: list[BoxFile],
) -> None:
    items = _list_folder(folder_id, access_token, shared_link)
    for item in items:
        name = item["name"]
        if item["type"] == "folder":
            sub_path = f"{folder_path}/{name}".lstrip("/")
            logger.debug("Entering subfolder: %s", sub_path)
            _walk_folder(item["id"], sub_path, access_token, shared_link, results)
        elif item["type"] == "file":
            decision = classify_filename(name)
            results.append(BoxFile(
                file_id=item["id"],
                name=name,
                folder_path=folder_path,
                box_url=_make_file_url(item["id"]),
                decision=decision,
            ))
            logger.debug("[%s] %s/%s", decision.upper(), folder_path, name)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = crawl_hub()
    include = [f for f in files if f.decision == FILTER_INCLUDE]
    exclude = [f for f in files if f.decision == FILTER_EXCLUDE]
    review  = [f for f in files if f.decision == FILTER_REVIEW]
    print(f"\n--- INCLUDE ({len(include)}) ---")
    for f in include:
        print(f"  {f.folder_path}/{f.name}")
    print(f"\n--- REVIEW ({len(review)}) ---")
    for f in review:
        print(f"  {f.folder_path}/{f.name}")
    print(f"\n--- EXCLUDE ({len(exclude)}) ---")
    for f in exclude:
        print(f"  {f.folder_path}/{f.name}")
