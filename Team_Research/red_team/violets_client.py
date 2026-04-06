"""
violets_client.py
=================
HTTP client for VIOLETS's custom API.
 
Expected request:
  POST <VIOLETS_ENDPOINT>
  { "user_id": "<stable uuid>", "query": "<latest user message>" }
 
 
Expected response (handles multiple formats):
  { "response": "..."  }              ← most likely
  { "answer":   "..."  }              ← alternate field name
  { "choices": [{"message": {"content": "..."}}] }  ← unlikely but handled
"""

import logging
import httpx
from config import RedTeamConfig
import uuid

logger = logging.getLogger("VIOLETSClient")


class VIOLETSClient:
    def __init__(self, cfg: RedTeamConfig):
        self.endpoint = cfg.violets_endpoint
        self.timeout = cfg.violets_timeout
        self.headers = {"Content-Type": "application/json"}
        if cfg.violets_api_key:
            self.headers["Authorization"] = f"Bearer {cfg.violets_api_key}"

    def new_session(self) -> "VIOLETSSession":
        """
        Create a new session with a fresh stable user_id.
        Call once per conversation; reuse the session for all turns.
        """
        return VIOLETSSession(
            endpoint=self.endpoint,
            headers=self.headers,
            timeout=self.timeout,
        )
 
 
class VIOLETSSession:
    """
    Represents one conversation with VIOLETS.
    Holds the stable user_id for the lifetime of the conversation.
    Each call to chat() sends only the latest user turn — VIOLETS
    manages the full history on the server side.
    """
 
    def __init__(self, endpoint: str, headers: dict, timeout: float):
        self.endpoint = endpoint
        self.headers = headers
        self.timeout = timeout
        # One stable UUID per conversation — this is the server-side session key
        self.user_id: str = str(uuid.uuid4())
        logger.debug(f"New VIOLETS session | user_id={self.user_id}")
 
    async def chat(self, latest_user_message: str) -> str:
        """
        Send the latest attacker message to VIOLETS and return its reply.
        Only sends user_id + query — history is server-managed.
        """
        payload = {
            "user_id": self.user_id,
            "query": latest_user_message,
        }
 
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())
 
        except httpx.HTTPStatusError as e:
            logger.error(
                f"VIOLETS HTTP {e.response.status_code} "
                f"[user_id={self.user_id}]: {e.response.text[:200]}"
            )
            return f"[VIOLETS error: HTTP {e.response.status_code}]"
        except httpx.RequestError as e:
            logger.error(f"VIOLETS connection error [user_id={self.user_id}]: {e}")
            return "[VIOLETS error: connection failed]"
        except Exception as e:
            logger.error(f"VIOLETS unexpected error [user_id={self.user_id}]: {e}")
            return f"[VIOLETS error: {e}]"
 
    @staticmethod
    def _parse_response(data: dict) -> str:
        """Parse VIOLETS's response. Edit this if the field name changes."""
        # Most likely VIOLETS formats
        for key in ("response", "answer", "reply", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]
        # Last resort
        logger.warning(f"Unrecognised VIOLETS response shape: {list(data.keys())}")
        return str(data)