"""
Production-readiness stress test for VIOLETS server.
Tests concurrency, session races, bad inputs, and edge cases.

Requires VIOLETS_API_KEY env var (or set API_KEY below).
"""
import asyncio
import json
import os
import time
import httpx
BASE = "http://localhost:8000"
TIMEOUT = 30.0
API_KEY = os.environ.get("VIOLETS_API_KEY", "")
AUTH_HEADERS = {"X-API-Key": API_KEY}
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
async def test_concurrent_rag_queries():
    """10 different users, same question, all at once."""
    print("\n[1] Concurrent RAG queries (10 users, same question)")
    async with httpx.AsyncClient() as client:
        tasks = [
            chat(client, f"concurrent-{i}", "How do I register to vote in Maryland?")
            for i in range(10)
        ]
        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
    errors = [r for r in responses if isinstance(r, Exception)]
    successes = [r for r in responses if not isinstance(r, Exception)]
    ok_count = sum(1 for s, body in successes if s == 200 and "response" in body)
    record(
        f"10 concurrent RAG queries in {elapsed:.1f}s",
        ok_count == 10 and len(errors) == 0,
        f"{ok_count}/10 succeeded, {len(errors)} exceptions"
        + (f": {errors[0]}" if errors else ""),
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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
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
    for (uid, q, label), resp in zip(queries, responses):
        if isinstance(resp, Exception):
            record(f"{label} ({uid})", False, str(resp))
        else:
            s, body = resp
            has_response = isinstance(body, dict) and "response" in body
            record(f"{label} ({uid})", s == 200 and has_response, f"status={s}")
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
        # Fire 5 RAG queries
        rag_tasks = [
            asyncio.create_task(
                chat(client, f"health-load-{i}", "How do I vote early in Maryland?")
            )
            for i in range(5)
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
async def main():
    print("=" * 60)
    print("VIOLETS Production Stress Test")
    print("=" * 60)
    if not API_KEY:
        print("WARNING: VIOLETS_API_KEY not set. Auth-protected endpoints will fail.")
    # Quick sanity check
    async with httpx.AsyncClient() as client:
        h = await client.get(f"{BASE}/health", timeout=5.0)
        if h.status_code != 200:
            print("Server not reachable. Aborting.")
            return
    await test_concurrent_rag_queries()
    await test_same_user_concurrent()
    await test_malformed_inputs()
    await test_mixed_guardrail_paths()
    await test_session_reset_under_load()
    await test_health_during_load()
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['pass']} passed, {results['fail']} failed")
    if results["errors"]:
        print("\nFAILURES:")
        for e in results["errors"]:
            print(f"  - {e}")
    else:
        print("All tests passed.")
    print("=" * 60)
if __name__ == "__main__":
    asyncio.run(main())