"""
Production-readiness stress test for VIOLETS server.
Tests auth, concurrency, session races, bad inputs, the Qualtrics usage
pattern (multi-turn sequential conversations), and the exact rate-limit
boundaries.

Usage:
    python -m maryland_rag.scripts.stress_test                # full suite
    python -m maryland_rag.scripts.stress_test --limits-only  # rate limits only

Rate limits are read from server.config (RATE_LIMIT_PER_MINUTE,
RATE_LIMIT_GLOBAL_PER_MINUTE), so run this with the same .env / environment
as the server under test. The rate-limit phase asserts the boundaries
exactly: every request up to a limit must succeed and every request past it
must get 429. It therefore needs the server to itself — any other traffic in
the same 60 s window shifts the counts and fails the run.

The rate-limit phase uses PII-blocked queries: they count against both
limiters (which run first) but Presidio blocks them locally before any LLM
call, so `--limits-only` spends no OpenAI tokens. It waits out one full
window before starting and another to check recovery, so it takes ~2 min.

Refuses to run against a non-localhost target (real LLM spend + junk
sessions) unless STRESS_CONFIRM=yes is set. Exits non-zero on any failure.
"""
import argparse
import asyncio
import os
import sys
import time
import uuid
from collections import Counter

import httpx

# Also loads .env, so VIOLETS_API_KEY and the limits match the server's.
from server import config

BASE = os.environ.get("VIOLETS_BASE_URL", "http://localhost:8000")
TIMEOUT = 60.0
API_KEY = config.VIOLETS_API_KEY
AUTH_HEADERS = {"X-API-Key": API_KEY}

PER_USER_LIMIT = config.RATE_LIMIT_PER_MINUTE
GLOBAL_LIMIT = config.RATE_LIMIT_GLOBAL_PER_MINUTE
# Must match _RateLimiter's default window_seconds in server/main.py.
LIMIT_WINDOW_SECONDS = 60
# Slack on top of the window for request latency and clock granularity.
WINDOW_SLACK_SECONDS = 3
# Requests sent past each limit; every one of them must be rejected.
OVERFLOW_REQUESTS = 5
RATE_LIMIT_DETAIL = "Rate limit exceeded"
# Blocked locally by Presidio, so limit tests make no LLM calls. An email is
# used because Presidio matches it deterministically; SSN detection skips
# known dummy numbers (e.g. 123-45-6789), which would reach the classifier.
FREE_QUERY = "My email is jane.doe@example.com, am I registered?"
# Distinguishing phrase from FALLBACK_RESPONSES["pii"] in server/middleware.py.
PII_MARKER = "may contain personal information"
# Fresh user_ids per run so a re-run never inherits an earlier run's windows.
RUN_ID = uuid.uuid4().hex[:8]

results = {"pass": 0, "fail": 0, "errors": []}


def record(name, passed, detail=""):
    if passed:
        results["pass"] += 1
        print(f"  PASS  {name}")
    else:
        results["fail"] += 1
        results["errors"].append(f"{name}: {detail}")
        print(f"  FAIL  {name} — {detail}")


async def chat(client, user_id, query):
    resp = await client.post(
        f"{BASE}/chat",
        json={"user_id": user_id, "query": query},
        headers=AUTH_HEADERS,
        timeout=TIMEOUT,
    )
    return resp.status_code, resp.json()


async def test_auth():
    """Missing/wrong API key must 401 on both authed endpoints; /health stays open."""
    print("\n[0] Authentication")
    async with httpx.AsyncClient() as client:
        payload = {"user_id": "auth-test", "query": "When is early voting?"}
        resp = await client.post(f"{BASE}/chat", json=payload, timeout=TIMEOUT)
        record("Missing API key returns 401", resp.status_code == 401, f"got {resp.status_code}")
        resp = await client.post(
            f"{BASE}/chat", json=payload,
            headers={"X-API-Key": "wrong-key-000"}, timeout=TIMEOUT,
        )
        record("Wrong API key returns 401", resp.status_code == 401, f"got {resp.status_code}")
        resp = await client.post(
            f"{BASE}/reset", json={"user_id": "auth-test"}, timeout=TIMEOUT,
        )
        record("/reset without key returns 401", resp.status_code == 401, f"got {resp.status_code}")
        resp = await client.get(f"{BASE}/health", timeout=5.0)
        record("/health needs no key", resp.status_code == 200, f"got {resp.status_code}")


async def test_qualtrics_conversations():
    """3 users in parallel, each holding a sequential 7-turn conversation —
    the real Qualtrics usage pattern (5-8 turns/user, history growing each turn)."""
    print("\n[7] Qualtrics-style multi-turn conversations (3 users x 7 turns)")
    turns = [
        "How do I register to vote in Maryland?",
        "What is the registration deadline?",
        "Can I register on election day?",
        "What ID do I need to bring?",
        "Can I vote by mail instead?",
        "How do I request a mail-in ballot?",
        "What did I ask you about first?",
    ]
    async def one_conversation(user_id):
        outcomes = []
        async with httpx.AsyncClient() as client:
            for q in turns:
                try:
                    s, body = await chat(client, user_id, q)
                    outcomes.append(s == 200 and "response" in body)
                except Exception:
                    outcomes.append(False)
        return outcomes
    start = time.time()
    all_outcomes = await asyncio.gather(
        *[one_conversation(f"survey-user-{i}") for i in range(3)]
    )
    elapsed = time.time() - start
    total = sum(len(o) for o in all_outcomes)
    ok = sum(sum(o) for o in all_outcomes)
    record(
        f"21 sequential turns across 3 users in {elapsed:.1f}s",
        ok == total,
        f"{ok}/{total} turns succeeded",
    )


async def test_concurrent_rag_queries():
    """25 different users, same question, all at once — saturates the connection pool (max_size=25)."""
    print("\n[1] Concurrent RAG queries (25 users, same question)")
    async with httpx.AsyncClient() as client:
        tasks = [
            chat(client, f"concurrent-{i}", "How do I register to vote in Maryland?")
            for i in range(25)
        ]
        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
    errors = [r for r in responses if isinstance(r, Exception)]
    successes = [r for r in responses if not isinstance(r, Exception)]
    ok_count = sum(1 for s, body in successes if s == 200 and "response" in body)
    record(
        f"25 concurrent RAG queries in {elapsed:.1f}s",
        ok_count == 25 and len(errors) == 0,
        f"{ok_count}/25 succeeded, {len(errors)} exceptions"
        + (f": {errors[0]}" if errors else ""),
    )
    # Verify source_urls field is present and non-empty in sources (new in rag_chain refactor)
    bad_sources = []
    for s, body in successes:
        if s == 200:
            for src in body.get("sources", []):
                if not src.get("source_urls"):
                    bad_sources.append(src.get("source_url", "?"))
    record(
        "source_urls populated in all source references",
        len(bad_sources) == 0,
        f"missing source_urls in: {bad_sources[:3]}",
    )
    # Verify no response is the silent error fallback (would mean fail-closed triggered on a normal query)
    error_fallback = "couldn't process your question"
    fallback_count = sum(
        1 for s, body in successes
        if s == 200 and error_fallback in body.get("response", "")
    )
    record(
        "No silent error fallbacks on normal RAG queries",
        fallback_count == 0,
        f"{fallback_count}/25 responses returned the error fallback",
    )


async def test_same_user_concurrent():
    """Same user_id, 5 concurrent requests — tests session store race."""
    print("\n[2] Same user_id under concurrent load (session race)")
    async with httpx.AsyncClient() as client:
        tasks = [
            chat(client, "race-user", f"Question number {i} about Maryland voting")
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in responses if isinstance(r, Exception)]
    successes = [r for r in responses if not isinstance(r, Exception)]
    ok_count = sum(1 for s, body in successes if s == 200)
    record(
        "5 concurrent requests, same user_id",
        ok_count == 5 and len(errors) == 0,
        f"{ok_count}/5 succeeded, {len(errors)} exceptions"
        + (f": {errors[0]}" if errors else ""),
    )
    # Verify session has accumulated history without corruption
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE}/chat",
            json={"user_id": "race-user", "query": "What did I just ask you about?"},
            headers=AUTH_HEADERS,
            timeout=TIMEOUT,
        )
        record(
            "Session history intact after concurrent writes",
            resp.status_code == 200 and "response" in resp.json(),
            f"status={resp.status_code}",
        )


async def test_malformed_inputs():
    """Bad inputs that shouldn't crash the server."""
    print("\n[3] Malformed / edge-case inputs")
    async with httpx.AsyncClient() as client:
        # Empty query (rejected by input validation)
        s, body = await chat(client, "edge-1", "")
        record("Empty string query returns 422", s == 422, f"status={s}")
        # Whitespace-only query (single space passes min_length but is just whitespace)
        s, body = await chat(client, "edge-2", "   ")
        record("Whitespace-only query handled", s in (200, 422), f"status={s}")
        # Very long query (over 2000 char limit)
        long_q = "What is voter registration? " * 200
        s, body = await chat(client, "edge-3", long_q)
        record(f"Over-limit query ({len(long_q)} chars) returns 422", s == 422, f"status={s}")
        # Unicode / emoji query
        s, body = await chat(client, "edge-4", "How do I vote? 🗳️ 投票はどうすればいいですか？")
        record("Unicode/emoji query", s == 200, f"status={s}")
        # Missing user_id field
        resp = await client.post(
            f"{BASE}/chat",
            json={"query": "test"},
            headers=AUTH_HEADERS,
            timeout=TIMEOUT,
        )
        record(
            "Missing user_id returns 422",
            resp.status_code == 422,
            f"got {resp.status_code}",
        )
        # Missing query field
        resp = await client.post(
            f"{BASE}/chat",
            json={"user_id": "test"},
            headers=AUTH_HEADERS,
            timeout=TIMEOUT,
        )
        record(
            "Missing query returns 422",
            resp.status_code == 422,
            f"got {resp.status_code}",
        )
        # Completely empty body
        resp = await client.post(
            f"{BASE}/chat",
            content=b"",
            headers={"Content-Type": "application/json", **AUTH_HEADERS},
            timeout=TIMEOUT,
        )
        record(
            "Empty body returns 422",
            resp.status_code == 422,
            f"got {resp.status_code}",
        )
        # Not JSON at all
        resp = await client.post(
            f"{BASE}/chat",
            content=b"this is not json",
            headers={"Content-Type": "application/json", **AUTH_HEADERS},
            timeout=TIMEOUT,
        )
        record(
            "Non-JSON body returns 422",
            resp.status_code == 422,
            f"got {resp.status_code}",
        )


async def test_mixed_guardrail_paths():
    """Fire PII, partisan, out-of-scope, and normal queries concurrently."""
    print("\n[4] Mixed guardrail paths concurrently")
    queries = [
        ("pii-user", "My credit card is 4111-1111-1111-1111", "pii/classifier"),
        ("partisan-user", "Which party should I vote for?", "partisan block"),
        ("oos-user", "What's the weather in California?", "out-of-scope block"),
        ("normal-user-1", "Where is my polling place in Montgomery County?", "normal RAG"),
        ("normal-user-2", "Can I vote by mail in Maryland?", "normal RAG"),
    ]
    async with httpx.AsyncClient() as client:
        tasks = [chat(client, uid, q) for uid, q, _ in queries]
        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
    error_fallback = "couldn't process your question"
    # Distinguishing phrase from FALLBACK_RESPONSES["out_of_scope"] in
    # server/middleware.py — a real weather answer would never contain this,
    # so this is what actually proves the block fired (200 + "response"
    # alone would pass even if the query got a genuine weather answer).
    oos_marker = "Maryland elections information assistant"
    for (uid, q, label), resp in zip(queries, responses):
        if isinstance(resp, Exception):
            record(f"{label} ({uid})", False, str(resp))
        else:
            s, body = resp
            has_response = isinstance(body, dict) and "response" in body
            record(f"{label} ({uid})", s == 200 and has_response, f"status={s}")
            # Normal RAG paths should never hit the silent error fallback
            if label == "normal RAG" and has_response:
                record(
                    f"No silent error fallback ({uid})",
                    error_fallback not in body.get("response", ""),
                    "got error fallback on a normal RAG query",
                )
            # Out-of-scope path must actually redirect, not answer the question
            if label == "out-of-scope block" and has_response:
                record(
                    f"Redirected, not answered ({uid})",
                    oos_marker in body.get("response", ""),
                    f"expected out_of_scope fallback, got: {body.get('response', '')[:120]!r}",
                )
    print(f"  All 5 mixed queries completed in {elapsed:.1f}s")


async def test_session_reset_under_load():
    """Reset a session while another request is in-flight for that user."""
    print("\n[5] Session reset during active request")
    async with httpx.AsyncClient() as client:
        # Start a slow RAG query
        rag_task = asyncio.create_task(
            chat(client, "reset-race", "Tell me about absentee voting in Maryland")
        )
        # Immediately reset the session
        await asyncio.sleep(0.1)
        reset_resp = await client.post(
            f"{BASE}/reset",
            json={"user_id": "reset-race"},
            headers=AUTH_HEADERS,
            timeout=TIMEOUT,
        )
        record("Reset during in-flight request", reset_resp.status_code == 200)
        # Wait for the RAG query to finish
        try:
            s, body = await asyncio.wait_for(rag_task, timeout=TIMEOUT)
            record(
                "In-flight request completes after reset",
                s == 200,
                f"status={s}",
            )
        except Exception as e:
            record("In-flight request completes after reset", False, str(e))


async def test_health_during_load():
    """Health endpoint should respond quickly even when RAG is busy."""
    print("\n[6] Health endpoint responsiveness under load")
    async with httpx.AsyncClient() as client:
        # Fire 20 RAG queries to put real pressure on the threadpool
        rag_tasks = [
            asyncio.create_task(
                chat(client, f"health-load-{i}", "How do I vote early in Maryland?")
            )
            for i in range(20)
        ]
        await asyncio.sleep(0.5)
        # Health should respond fast even with 5 RAG queries in-flight
        start = time.time()
        health = await client.get(f"{BASE}/health", timeout=5.0)
        health_time = time.time() - start
        record(
            f"Health responds in {health_time:.2f}s during load",
            health.status_code == 200 and health_time < 2.0,
            f"status={health.status_code}, took {health_time:.2f}s",
        )
        # Let RAG queries finish so we don't leave dangling connections
        await asyncio.gather(*rag_tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Rate-limit boundaries
# ---------------------------------------------------------------------------

async def wait_out_window(reason):
    seconds = LIMIT_WINDOW_SECONDS + WINDOW_SLACK_SECONDS
    print(f"  ... waiting {seconds}s for the rate-limit window to clear ({reason})")
    await asyncio.sleep(seconds)


async def burst(client, user_ids):
    """Send one free (PII-blocked) /chat per user_id, all concurrently."""
    return await asyncio.gather(
        *(chat(client, uid, FREE_QUERY) for uid in user_ids),
        return_exceptions=True,
    )


def summarize(responses):
    counts = Counter(
        type(r).__name__ if isinstance(r, Exception) else r[0] for r in responses
    )
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=str))


def all_accepted(responses):
    return all(not isinstance(r, Exception) and r[0] == 200 for r in responses)


def all_pii_blocked(responses):
    return all(
        not isinstance(r, Exception) and PII_MARKER in r[1].get("response", "")
        for r in responses
    )


def all_rate_limited(responses):
    return all(
        not isinstance(r, Exception)
        and r[0] == 429
        and r[1].get("detail") == RATE_LIMIT_DETAIL
        for r in responses
    )


async def test_per_user_limit(client):
    """Exactly PER_USER_LIMIT requests from one user_id pass; the rest 429.

    Returns the number of global slots consumed, or None if the query was
    not PII-blocked (the later bursts would then make real LLM calls). Per-user
    rejections are checked before the global limiter and must not consume
    global slots — test_global_limit's exact count proves that.
    """
    print(f"\n[8] Per-user rate limit (exactly {PER_USER_LIMIT}/min, one user_id)")
    uid = f"limit-user-{RUN_ID}"

    within = await burst(client, [uid] * PER_USER_LIMIT)
    record(
        f"{PER_USER_LIMIT} requests at the per-user limit all succeed",
        all_accepted(within),
        summarize(within),
    )
    accepted = [r for r in within if not isinstance(r, Exception) and r[0] == 200]
    # Needs at least one 200 to prove anything — an all-failed burst must not
    # pass this vacuously and green-light the 200+ request global burst.
    free = bool(accepted) and all_pii_blocked(accepted)
    record(
        "Limit-test query is PII-blocked (no LLM spend)",
        free,
        "query reached the LLM — aborting before the global burst",
    )
    if not free:
        return None

    over = await burst(client, [uid] * OVERFLOW_REQUESTS)
    record(
        f"{OVERFLOW_REQUESTS} requests over the per-user limit all get 429",
        all_rate_limited(over),
        summarize(over),
    )
    return PER_USER_LIMIT


async def test_global_limit(client, used):
    """Distinct user_ids fill the global cap exactly; anything past it 429s,
    including /reset, which shares the same global budget."""
    remaining = GLOBAL_LIMIT - used
    print(
        f"\n[9] Global rate limit (exactly {GLOBAL_LIMIT}/min across all "
        f"user_ids, {used} already used)"
    )

    within = await burst(client, [f"global-{RUN_ID}-{i}" for i in range(remaining)])
    record(
        f"{remaining} requests from distinct users fill the global cap",
        all_accepted(within),
        summarize(within),
    )

    over = await burst(
        client, [f"global-over-{RUN_ID}-{i}" for i in range(OVERFLOW_REQUESTS)]
    )
    record(
        f"{OVERFLOW_REQUESTS} fresh user_ids over the global cap all get 429",
        all_rate_limited(over),
        summarize(over),
    )

    resp = await client.post(
        f"{BASE}/reset",
        json={"user_id": f"global-over-{RUN_ID}-0"},
        headers=AUTH_HEADERS,
        timeout=TIMEOUT,
    )
    record(
        "/reset over the global cap gets 429",
        resp.status_code == 429,
        f"got {resp.status_code}",
    )


async def test_limit_recovery(client):
    """Once the window slides past, previously rejected callers are served."""
    print("\n[10] Rate-limit recovery after the window slides")
    await wait_out_window("recovery check")

    s, _ = await chat(client, f"limit-user-{RUN_ID}", FREE_QUERY)
    record("Previously limited user_id is accepted again", s == 200, f"got {s}")

    resp = await client.post(
        f"{BASE}/reset",
        json={"user_id": f"limit-user-{RUN_ID}"},
        headers=AUTH_HEADERS,
        timeout=TIMEOUT,
    )
    record("/reset is accepted again", resp.status_code == 200, f"got {resp.status_code}")


async def run_rate_limit_tests():
    if PER_USER_LIMIT > GLOBAL_LIMIT:
        record(
            "Rate-limit config is testable",
            False,
            f"RATE_LIMIT_PER_MINUTE ({PER_USER_LIMIT}) exceeds "
            f"RATE_LIMIT_GLOBAL_PER_MINUTE ({GLOBAL_LIMIT})",
        )
        return
    print(f"\nRate limits under test: {PER_USER_LIMIT}/min per user, {GLOBAL_LIMIT}/min global")
    await wait_out_window("start from an empty window")
    async with httpx.AsyncClient() as client:
        used = await test_per_user_limit(client)
        if used is None:
            return
        await test_global_limit(client, used)
        await test_limit_recovery(client)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(limits_only):
    print("=" * 60)
    print("VIOLETS Production Stress Test")
    print("=" * 60)
    local = any(h in BASE for h in ("localhost", "127.0.0.1", "[::1]"))
    if not local and os.environ.get("STRESS_CONFIRM") != "yes":
        print(f"Refusing to stress-test non-local target {BASE!r}: this spends")
        print("real LLM tokens and writes junk sessions. Set STRESS_CONFIRM=yes to override.")
        return False
    # Quick sanity check
    async with httpx.AsyncClient() as client:
        try:
            h = await client.get(f"{BASE}/health", timeout=5.0)
        except httpx.HTTPError:
            h = None
        if h is None or h.status_code != 200:
            print("Server not reachable. Aborting.")
            return False
    if not limits_only:
        await test_auth()
        await test_concurrent_rag_queries()
        await test_same_user_concurrent()
        await test_malformed_inputs()
        await test_mixed_guardrail_paths()
        await test_session_reset_under_load()
        await test_health_during_load()
        await test_qualtrics_conversations()
    # Always last: the phase deliberately exhausts both limiters.
    await run_rate_limit_tests()
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['pass']} passed, {results['fail']} failed")
    if results["errors"]:
        print("\nFAILURES:")
        for e in results["errors"]:
            print(f"  - {e}")
    else:
        print("All tests passed.")
    print("=" * 60)
    return results["fail"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--limits-only",
        action="store_true",
        help="run only the rate-limit boundary tests (no LLM spend, ~2 min)",
    )
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(main(args.limits_only)) else 1)
