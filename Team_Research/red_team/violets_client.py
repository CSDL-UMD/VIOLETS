"""
violets_client.py
=================
HTTP client for VIOLETS. Sends the full conversation history each turn.

Expected request (OpenAI-compatible):
  POST <VIOLETS_ENDPOINT>
  { "messages": [{"role": "user"|"assistant", "content": "..."}] }

Expected response (handles multiple formats):
  {"choices": [{"message": {"content": "..."}}]}   ← OpenAI-compatible
  {"response": "..."}                               ← simple custom format

Edit _parse_response() if your endpoint uses a different schema.
"""

import logging
import httpx
from config import RedTeamConfig

logger = logging.getLogger("VIOLETSClient")


class VIOLETSClient:
    def __init__(self, cfg: RedTeamConfig):
        self.endpoint = cfg.violets_endpoint
        self.timeout = cfg.violets_timeout
        self.headers = {"Content-Type": "application/json"}
        if cfg.violets_api_key:
            self.headers["Authorization"] = f"Bearer {cfg.violets_api_key}"

    async def chat(self, messages: list[dict]) -> str:
        """Send conversation history to VIOLETS and return its reply."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.endpoint,
                    json={"messages": messages},
                    headers=self.headers,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())

        except httpx.HTTPStatusError as e:
            logger.error(f"VIOLETS HTTP {e.response.status_code}: {e.response.text[:200]}")
            return f"[VIOLETS error: HTTP {e.response.status_code}]"
        except httpx.RequestError as e:
            logger.error(f"VIOLETS connection error: {e}")
            return "[VIOLETS error: connection failed]"
        except Exception as e:
            logger.error(f"VIOLETS unexpected error: {e}")
            return f"[VIOLETS error: {e}]"

    @staticmethod
    def _parse_response(data: dict) -> str:
        # OpenAI-compatible
        if "choices" in data:
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
        # Single-field formats
        for key in ("response", "reply", "message", "content", "text", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]
        logger.warning(f"Unrecognised VIOLETS response shape: {list(data.keys())}")
        return str(data)
