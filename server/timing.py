"""
server/timing.py
================
Per-request, per-stage wall-clock timing.

A request binds a fresh dict with ``begin_request()`` (contextvar, so it
follows the request across awaits, child tasks, and asyncio.to_thread — the
same mechanism as the request-id log filter). Any code on that request's path
then wraps work in ``with stage("name"):`` and the elapsed seconds accumulate
into the bound dict. Repeated stages (e.g. a partisan retry re-running
"generate") sum, which is intentional: the dict answers "where did this
request's wall-clock go", not "how fast is one call".

Zero-cost when no dict is bound (non-request code paths).
"""

import contextvars
import time
from contextlib import contextmanager

# Canonical display order for the TIMINGS log line.
STAGE_ORDER = [
    "pii",        # Presidio scan (thread offload included)
    "classify",   # classifier LLM call
    "rephrase",   # follow-up → standalone question LLM call
    "embed",      # OpenAI embedding of the retrieval query
    "retrieve",   # pgvector similarity query (incl. pool checkout)
    "generate",   # QA / conversational / concerns LLM call
    "partisan",   # post-chain checker (incl. any retry chain re-invocations)
]

_timings: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "stage_timings", default=None
)


def begin_request() -> dict:
    """Bind and return a fresh timing dict for the current request context."""
    d: dict[str, float] = {}
    _timings.set(d)
    return d


@contextmanager
def stage(name: str):
    """Accumulate the wall-clock of the wrapped block into the bound dict."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        d = _timings.get()
        if d is not None:
            d[name] = d.get(name, 0.0) + (time.perf_counter() - t0)


def format_line(timings: dict, total_seconds: float) -> str:
    """Render 'pii=52ms classify=1830ms ... total=6234ms' in canonical order.

    Stages that never ran on this request are omitted rather than shown as 0,
    so the line doubles as a record of which path the request took.
    """
    parts = [
        f"{name}={timings[name] * 1000:.0f}ms"
        for name in STAGE_ORDER
        if name in timings
    ]
    # Anything recorded outside the canonical list still gets shown.
    parts += [
        f"{name}={secs * 1000:.0f}ms"
        for name, secs in timings.items()
        if name not in STAGE_ORDER
    ]
    parts.append(f"total={total_seconds * 1000:.0f}ms")
    return " ".join(parts)
