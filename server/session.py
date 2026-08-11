"""
In-memory conversation session store.

Keeps chat history per user_id with automatic TTL expiration.
Designed for pilot-scale usage (tens of concurrent users).
"""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Session:
    messages: list = field(default_factory=list)
    # Sources (URLs + titles) from the most recent retrieval turn, overwritten
    # each turn so link follow-ups can be answered without inventing URLs.
    last_sources: list = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_minutes: int = 30, max_turns: int = 20):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self._ttl = ttl_minutes * 60
        self._max_turns = max_turns

    def get_or_create(self, user_id: str) -> Session:
        with self._lock:
            self._cleanup_expired()
            if user_id not in self._sessions:
                self._sessions[user_id] = Session()
            session = self._sessions[user_id]
            session.last_active = time.time()
            return session

    def add_exchange(self, user_id: str, user_msg: str, assistant_msg: str):
        """Record a full user/assistant turn."""
        with self._lock:
            session = self._sessions.get(user_id)
            if not session:
                return
            session.messages.append({"role": "user", "content": user_msg})
            session.messages.append({"role": "assistant", "content": assistant_msg})
            # Trim to max turns (1 turn = 2 messages)
            max_msgs = self._max_turns * 2
            if len(session.messages) > max_msgs:
                session.messages = session.messages[-max_msgs:]
            session.last_active = time.time()

    def set_last_sources(self, user_id: str, sources: list):
        """Overwrite the cached sources with the latest retrieval turn's."""
        with self._lock:
            session = self._sessions.get(user_id)
            if not session:
                return
            session.last_sources = list(sources)

    def get_last_sources(self, user_id: str) -> list:
        with self._lock:
            session = self._sessions.get(user_id)
            return list(session.last_sources) if session else []

    def get_history(self, user_id: str) -> list[dict]:
        with self._lock:
            session = self._sessions.get(user_id)
            return list(session.messages) if session else []

    def reset(self, user_id: str):
        with self._lock:
            self._sessions.pop(user_id, None)

    def cleanup_expired(self):
        """Remove expired sessions. Called by background task."""
        with self._lock:
            self._cleanup_expired()

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            uid for uid, s in self._sessions.items()
            if now - s.last_active > self._ttl
        ]
        for uid in expired:
            del self._sessions[uid]
