"""
baseline_client.py
==================
Client for the baseline model (e.g. GPT-4.1-nano / GPT-4o-mini).
 
The client must:
  1. Maintain the full conversation history locally.
  2. Send the complete messages array every turn.
  3. Append each new (user, assistant) pair to its local history.
 
This produces outputs in the same JSONL schema as VIOLETS so both agents
can be compared directly for RQ1/RQ2.
"""
 
import logging
from openai import AsyncOpenAI
from config import RedTeamConfig
import uuid
 
logger = logging.getLogger("BaselineClient")
 
 
class BaselineClient:
    def __init__(self, cfg: RedTeamConfig, client: AsyncOpenAI):
        self.cfg = cfg
        self.client = client
 
    def new_session(self) -> "BaselineSession":
        """
        Create a new conversation session with a fresh message history.
        Call once per conversation; reuse the session for all turns.
        """
        return BaselineSession(
            client=self.client,
            model=self.cfg.baseline_model,
            system_prompt=self.cfg.baseline_system_prompt,
        )
 
 
class BaselineSession:
    """
    Represents one conversation with the baseline model.
    Maintains client-side message history across turns — the baseline has
    no server-side memory, so we must send the full history every call.
    """
 
    def __init__(self, client: AsyncOpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        # Build the initial messages array with the system prompt
        self.history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        logger.debug(f"New baseline session | model={model}")
 
    async def chat(self, latest_user_message: str) -> str:
        """
        Append the latest user message to history, call the baseline model
        with the full history, append the assistant reply, and return it.
        """
        self.history.append({"role": "user", "content": latest_user_message})
 
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                max_tokens=512,
                temperature=0.7,
            )
            reply = resp.choices[0].message.content.strip()
            # Persist the assistant's reply so it's included in the next turn
            self.history.append({"role": "assistant", "content": reply})
            return reply
 
        except Exception as e:
            logger.error(f"Baseline model error: {e}")
            # Still append a placeholder so history stays consistent
            error_msg = f"[baseline error: {e}]"
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
 