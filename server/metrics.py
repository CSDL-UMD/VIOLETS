"""
server/metrics.py
=================
Lightweight, thread-safe in-process counters that feed the periodic heartbeat
log line. This is NOT the error channel — errors are logged the instant they
happen. These counters exist only so the heartbeat can answer "how's the server
doing?" at a glance (volume, cost).

Two horizons are tracked:
    - rolling:    reset each heartbeat  → "last 5m: ..."
    - cumulative: since process start   → "since boot: ..."

Token/cost totals are fed by the RAG callback handler (server/rag_logger.py);
request outcomes are recorded by the /chat endpoint (server/main.py).
"""

import threading


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Cumulative — since process start.
        self.total_requests = 0
        self.total_errors = 0
        self.total_blocked = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        # Rolling — reset on each drain_rolling() (i.e. each heartbeat).
        self._roll_requests = 0
        self._roll_errors = 0
        self._roll_blocked = 0
        self._roll_tokens = 0
        self._roll_cost = 0.0

    def record_request(self, outcome: str) -> None:
        """Count a completed request. ``outcome`` is "ok", "blocked:*" or "error:*"."""
        with self._lock:
            self.total_requests += 1
            self._roll_requests += 1
            if outcome.startswith("error"):
                self.total_errors += 1
                self._roll_errors += 1
            elif outcome.startswith("blocked"):
                self.total_blocked += 1
                self._roll_blocked += 1

    def record_llm(self, tokens: int, cost: float) -> None:
        """Accumulate token + cost totals from a single LLM call."""
        with self._lock:
            self.total_tokens += tokens
            self.total_cost += cost
            self._roll_tokens += tokens
            self._roll_cost += cost

    def drain_rolling(self) -> dict:
        """Return a snapshot of the rolling window and reset it. Cumulative totals kept."""
        with self._lock:
            snap = {
                "requests": self._roll_requests,
                "errors": self._roll_errors,
                "blocked": self._roll_blocked,
                "tokens": self._roll_tokens,
                "cost": self._roll_cost,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "total_cost": self.total_cost,
            }
            self._roll_requests = 0
            self._roll_errors = 0
            self._roll_blocked = 0
            self._roll_tokens = 0
            self._roll_cost = 0.0
            return snap


# Module-level singleton shared across the server process.
METRICS = Metrics()
