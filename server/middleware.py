"""
server/middleware.py
====================
Middleware layer for the VIOLETS Election Chatbot.

This module contains all guardrails that run BEFORE and AFTER the main
RAG chain. The goal is to block bad queries early (before spending tokens)
and catch bad responses late (before sending to the user).

PIPELINE ORDER (called inside /chat in main.py):
-------------------------------------------------
    [1] detect_pii()              — regex, 0 tokens
    [2] classify_query()          — small LLM, ~70 tokens
    [3] Main RAG chain            — only if [1] and [2] pass
    [4] check_partisan_response() — small LLM, ~100 tokens

WHY THIS ORDER:
    - Cheapest check runs first (regex is free).
    - If PII is found we never spend tokens on classification.
    - If the query is out-of-scope we never hit the expensive RAG chain.
    - Partisan response check must run last because it needs the LLM output.
"""

import re
import logging
from dataclasses import dataclass
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1: Shared context
# ---------------------------------------------------------------------------
# QueryContext travels through the entire middleware pipeline.
# Each guardrail reads from it and may write to it.
# This avoids re-classifying the same query multiple times and gives every
# part of the pipeline visibility into what earlier parts decided.
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """
    Shared state that travels through the middleware pipeline.

    Attributes
    ----------
    user_id : str
        Anonymized session identifier. Never the real participant PID.

    query_category : str or None
        Set by classify_query(). None means it hasn't run yet.
        Possible values: "normal", "out_of_scope", "partisan"

    safety_flag : bool
        Set to True by classify_query() when the query is not "normal".
        When True, the RAG chain is skipped entirely.

    pii_detected : bool
        Set to True by detect_pii() when a PII pattern is found.

    pii_type : str or None
        Which type of PII was detected ("email", "credit_card", "ip", "ssn").
        Useful for logging and auditing.
    """
    user_id: str
    query_category: str | None = None
    safety_flag: bool = False
    pii_detected: bool = False
    pii_type: str | None = None


# ---------------------------------------------------------------------------
# SECTION 2: Fallback responses
# ---------------------------------------------------------------------------
# Defined here — not scattered through main.py — so the team can update
# user-facing messaging in one place without touching endpoint logic.
# ---------------------------------------------------------------------------

FALLBACK_RESPONSES = {
    "out_of_scope": (
        "I can only answer questions about voting in Maryland for the 2026 "
        "elections. For information about other states or federal races, "
        "please visit vote.gov."
    ),
    "partisan": (
        "I'm not able to provide candidate endorsements or partisan political "
        "opinions. I'm here to help with factual voting information for "
        "Maryland — such as registration deadlines, polling locations, and "
        "ballot procedures."
    ),
    "pii": (
        "I noticed your message may contain personal information (such as an "
        "email address, credit card number, IP address, or Social Security "
        "Number). Please don't share sensitive personal data. I can answer "
        "your question without it — what would you like to know about "
        "Maryland elections?"
    ),
}


# ---------------------------------------------------------------------------
# SECTION 3: PII detection (node-style, zero LLM cost)
# ---------------------------------------------------------------------------
# We use regex here instead of an LLM because:
#   1. These patterns are well-defined — no language understanding needed.
#   2. It costs 0 tokens.
#   3. It's deterministic — same input always gives same result.
#
# Trade-off: regex can have false positives (e.g. a number that looks like
# an SSN but isn't). For an election chatbot this is acceptable — we err on
# the side of caution.
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ \-]?){13,19}\b"
    ),
    "ip": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"
    ),
}


def detect_pii(query: str, ctx: QueryContext) -> str | None:
    """
    Scan the user query for PII using regex.

    This is a node-style guardrail — it runs BEFORE the RAG chain and
    costs zero tokens. Returns a fallback message string if PII is found,
    or None if the query is clean.

    The caller (main.py) checks the return value: if it's not None, it
    returns the fallback immediately and skips the RAG chain entirely.
    """
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(query):
            # Never log the actual query — that would defeat the purpose.
            logger.warning(
                "PII detected in query [user=%s type=%s] — query blocked.",
                ctx.user_id,
                pii_type,
            )
            ctx.pii_detected = True
            ctx.pii_type = pii_type
            return FALLBACK_RESPONSES["pii"]

    return None  # clean — proceed


# ---------------------------------------------------------------------------
# SECTION 4: Query classification (node-style, small LLM)
# ---------------------------------------------------------------------------
# We use an LLM here instead of regex because partisan/out-of-scope intent
# is expressed in natural language and can't be caught with keywords.
# Example: "What do you think about the candidates?" — no keywords, but
# clearly partisan.
#
# We use a separate small model (not the full RAG model) because the
# classifier only needs to output one of three labels. Using the full
# RAG model with retrieval for this would be wasteful (~70 tokens vs
# hundreds).
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """
    Structured output from the classifier LLM.

    Using with_structured_output() guarantees the LLM returns exactly
    the fields we expect — no free-form text parsing needed.
    """
    category: Literal["normal", "out_of_scope", "partisan"]
    reason: str  # used for logging only, never shown to the user


# Build the classifier once at module load time, not on every request.
_classifier_llm = (
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # deterministic output
        openai_api_key=config.OPENAI_API_KEY,
    )
    .with_structured_output(ClassificationResult)
)

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query classifier for VIOLETS, a voter information chatbot
for Maryland (2026 elections only).

Classify the user query into exactly one of the following categories:

- normal       : the query is about Maryland voting in 2026
                 (registration, polling locations, mail-in ballots,
                  ID requirements, deadlines, absentee voting, etc.)

- out_of_scope : the query targets federal races, other states,
                 other countries, or elections before 2026.
                 Also use this for completely off-topic questions.

- partisan     : the query requests candidate endorsements, asks
                 which party is better, or asks for partisan political
                 judgments about candidates or parties.

Return your classification and a brief reason (1 sentence).
Be decisive — every query must map to exactly one category.
"""


def classify_query(query: str, ctx: QueryContext) -> str | None:
    """
    Classify the user query using a lightweight LLM.

    Node-style guardrail — runs BEFORE the RAG chain.
    Returns a fallback message if the query should be blocked, or None
    if it's normal and should proceed.

    Fails open: if the classifier LLM errors, we log it and return None
    so the user still gets an answer. A broken classifier should degrade
    gracefully, not take down the chatbot.
    """
    try:
        result: ClassificationResult = _classifier_llm.invoke([
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ])

        ctx.query_category = result.category
        ctx.safety_flag = result.category != "normal"

        logger.info(
            "Query classified [user=%s category=%s reason=%s]",
            ctx.user_id,
            result.category,
            result.reason,
        )

        if ctx.safety_flag:
            return FALLBACK_RESPONSES[result.category]

        return None  # normal — proceed to RAG chain

    except Exception as exc:
        logger.error(
            "classify_query failed [user=%s error=%s] — allowing query through.",
            ctx.user_id,
            exc,
        )
        return None  # fail open


# ---------------------------------------------------------------------------
# SECTION 5: Partisan response guardrail (wrap-style, small LLM)
# ---------------------------------------------------------------------------
# This runs AFTER the RAG chain. It catches cases where a normal-looking
# query accidentally gets a partisan answer from the LLM — for example,
# "Who are the candidates for County Executive?" is a legitimate question
# but the LLM might unexpectedly add "Candidate X has a stronger record."
#
# Why wrap-style? Because we can only detect a partisan *response* after
# the LLM has spoken. There's no way to know what it will say beforehand.
#
# Why only one retry? classify_query() already blocks obvious partisan
# queries upstream, so this guardrail rarely fires. Two consecutive
# partisan responses on a normal query would be extremely unusual.
# ---------------------------------------------------------------------------

class PartisanCheckResult(BaseModel):
    """Structured output from the partisan response checker."""
    is_partisan: bool
    reason: str  # logging only


_partisan_checker_llm = (
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )
    .with_structured_output(PartisanCheckResult)
)

_PARTISAN_CHECKER_SYSTEM_PROMPT = """\
You are a partisan content checker for VIOLETS, a voter information
chatbot for Maryland elections.

Check whether the response below contains ANY of the following:
- Mentions of specific candidate names in a favorable or unfavorable way
- Endorsements of a political party or candidate
- Party comparisons that imply one is better than another
- Language encouraging voting for or against a specific candidate or party

Return is_partisan=True if any of the above are present, False otherwise.
Return a brief reason for your decision (1 sentence).
"""

_STRICT_NONPARTISAN_RETRY_PROMPT = (
    "IMPORTANT: Your previous response contained partisan content. "
    "Rewrite it following these strict rules:\n"
    "- Do NOT mention any candidate names.\n"
    "- Do NOT compare political parties.\n"
    "- Do NOT express any preference for any candidate or party.\n"
    "- Provide only neutral, factual voting information."
)


def check_partisan_response(
    query: str,
    response: str,
    chat_history: list,
    chain,
    ctx: QueryContext,
) -> str:
    """
    Check the RAG chain's response for partisan content.

    Wrap-style guardrail — runs AFTER the RAG chain.
    If partisan content is detected, retries the chain once with a
    stricter system prompt prepended.

    Fails open: if the checker errors, returns the original response
    rather than crashing the request.
    """
    try:
        result: PartisanCheckResult = _partisan_checker_llm.invoke([
            {"role": "system", "content": _PARTISAN_CHECKER_SYSTEM_PROMPT},
            {"role": "user",   "content": response},
        ])

        logger.info(
            "Partisan response check [user=%s is_partisan=%s reason=%s]",
            ctx.user_id,
            result.is_partisan,
            result.reason,
        )

        if not result.is_partisan:
            return response  # clean — return as-is

        # Partisan detected — retry with stricter prompt
        logger.warning(
            "Partisan response detected [user=%s] — retrying with strict prompt.",
            ctx.user_id,
        )

        retry_history = [
            {"role": "system", "content": _STRICT_NONPARTISAN_RETRY_PROMPT},
            *chat_history,
        ]

        retry_response = chain.invoke({
            "input": query,
            "chat_history": retry_history,
        })

        return retry_response

    except Exception as exc:
        logger.error(
            "check_partisan_response failed [user=%s error=%s] — returning original.",
            ctx.user_id,
            exc,
        )
        return response  # fail open