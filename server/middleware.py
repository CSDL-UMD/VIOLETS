"""
server/middleware.py
====================
Middleware layer for the VIOLETS Election Chatbot.

This module contains all guardrails that run BEFORE and AFTER the main
RAG chain. The goal is to block bad queries early (before spending tokens)
and catch bad responses late (before sending to the user).

PIPELINE ORDER (called inside /chat in main.py):
-------------------------------------------------
    [1] detect_pii()                — Presidio, 0 tokens
    [2] classify_query()            — small LLM, ~70 tokens
    [3] Main RAG chain              — only if [1] and [2] pass
    [4] check_partisan_response()   — small LLM, ~100 tokens

WHY THIS ORDER:
    - Cheapest checks run first (Presidio is free, regex was free).
    - If PII is found we never spend tokens on classification.
    - If the query is out-of-scope we never hit the expensive RAG chain.
    - Partisan response check must run after chain — needs LLM output.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine 

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
# SECTION 3: PII detection (Presidio)
# ---------------------------------------------------------------------------
# We use Presidio instead of regex because:
#   1. Uses ML + pattern matching together — understands context
#   2. Handles format variations automatically (cards with/without dashes,
#      SSNs with spaces, phone numbers in any format, etc.)
#   3. Built-in validation — Luhn check for cards, range check for IPs
#   4. Works identically for both input AND output checking
#   5. Actively maintained — improvements come for free
#
# Trade-off: ~0.1-0.3s per check vs ~0.001s for regex.
# Acceptable at pilot scale (tens of users).
#
# WHY PERSON and LOCATION are excluded:
#   Users legitimately mention candidate names and Maryland counties.
#   Detecting those as PII would block valid election questions.
#   Example: "Is John Smith running in Montgomery County?" should not
#   be blocked — it's a legitimate voter question.
# ---------------------------------------------------------------------------

_PII_ENTITIES = [
    "US_SSN",
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "PHONE_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
]

# Minimum Presidio confidence score to treat a detection as real PII.
# 0.5 balances catching real PII vs blocking legitimate messages.
# Lower = more sensitive, more false positives
# Higher = less sensitive, might miss real PII
_PII_SCORE_THRESHOLD = 0.5

# Initialize once at module load time.
# Loading the spaCy NLP model takes 2-3 seconds and ~500MB memory.
# Never initialize inside a function that runs per request.
_analyzer = AnalyzerEngine()


def detect_pii(query: str, ctx : QueryContext) -> str | None:
    """
    Scan the user query for PII using Presidio.

    Node-style guardrail — runs BEFORE the RAG chain at zero token cost.
    Returns a fallback message string if PII is found, None if clean.

    The caller (main.py) checks the return value: if not None, it
    returns the fallback immediately and skips the RAG chain entirely.
    """

    query = str(query)
    results = _analyzer.analyze(
        text=query,
        language="en",
        entities=_PII_ENTITIES,
    )

    # Filter to detections above confidence threshold
    hits = [r for r in results if r.score >= _PII_SCORE_THRESHOLD]

    if not hits:
        return None  # clean — proceed to classify_query

    # Use highest confidence hit for logging
    # Never log the actual query text — that would store the PII
    # we're supposed to be protecting
    top_hit = max(hits, key=lambda r: r.score)

    logger.warning(
        "PII detected in query [user=%s type=%s score=%.2f] — blocking.",
        ctx.user_id,
        top_hit.entity_type,
        top_hit.score,
    )

    ctx.pii_detected = True
    ctx.pii_type = top_hit.entity_type
    return FALLBACK_RESPONSES["pii"]

#---------------------------------------------------------------------------
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
        base_url=config.OPENAI_BASE_URL,
    )
    .with_structured_output(ClassificationResult)
)

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query classifier for VIOLETS, a voter information chatbot
for Maryland elections.

Classify the user query into exactly one of the following categories:

- normal       : the query is about Maryland voting or elections
                (registration, polling locations, mail-in ballots,
                  ID requirements, deadlines, absentee voting,
                  election procedures, or any Maryland election topic).

- out_of_scope : the query is about other states, federal races,
                other countries, or topics completely unrelated
                to Maryland voting and elections.

- partisan     : the query requests candidate endorsements, asks
                which party is better, or asks for partisan political
                judgments about candidates or parties.

Return your classification and a brief reason (1 sentence).
Be decisive — every query must map to exactly one category.
"""

async def classify_query(query: str, ctx: QueryContext) -> str | None:
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
        result: ClassificationResult = await _classifier_llm.ainvoke([
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
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
# Retry logic: up to MAX_PARTISAN_RETRIES (2) additional attempts. The
# checker re-runs after each retry so we never return partisan content
# blindly. If all retries are exhausted, the last generated response is
# returned anyway (fail-open) rather than leaving the user with no answer.
# ---------------------------------------------------------------------------
MAX_PARTISAN_RETRIES = 2

class PartisanCheckResult(BaseModel):
    """Structured output from the partisan response checker."""
    is_partisan: bool
    reason: str  # logging only


_partisan_checker_llm = (
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
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


async def check_partisan_response(
    query: str,
    response: str,
    chat_history: list,
    chain,
    ctx: QueryContext,
) -> tuple[str, list | None]:
    """
    Check the RAG chain's response for partisan content, retrying up to
    MAX_PARTISAN_RETRIES times if partisan content is detected.

    The strict retry prompt is appended to the user message so history stays clean for the rephrase step.

    After each retry the checker runs again — we never return a retry
    response blindly without verifying it first.

    If all retries are exhausted and the response is still partisan, a
    warning is logged and the last generated response is returned anyway
    (fail-open) rather than leaving the user with no answer.

    Returns a tuple of (answer, sources). Sources is None if no retry
    fired (main.py keeps the original sources), or the retry's sources
    if a retry produced a clean response.
 
    Also fails open on exceptions: if the checker itself errors, the
    current response is returned as-is.
    """
    current_response = response
    current_sources = None

    for attempt in range(MAX_PARTISAN_RETRIES + 1):  # 0 = original, 1-2 = retries
        try:
            result = await _partisan_checker_llm.ainvoke([
                SystemMessage(content=_PARTISAN_CHECKER_SYSTEM_PROMPT),
                HumanMessage(content=current_response),
            ])

            logger.info(
                "Partisan check [user=%s attempt=%d is_partisan=%s reason=%s]",
                ctx.user_id,
                attempt,
                result.is_partisan,
                result.reason,
            )

            if not result.is_partisan:
                return current_response, current_sources  # clean — return as-is

            if attempt < MAX_PARTISAN_RETRIES:
                # Retries remaining — re-invoke the chain with stricter prompt
                logger.warning(
                    "Partisan response detected [user=%s attempt=%d] — retrying.",
                    ctx.user_id,
                    attempt,
                )
                retry_result = await chain.ainvoke({
                    "input": query + "\n\n" + _STRICT_NONPARTISAN_RETRY_PROMPT,
                    "chat_history": chat_history,
                })
                if isinstance(retry_result, dict):
                    current_response = str(retry_result.get("answer", retry_result))
                    current_sources = retry_result.get("sources")
                else:
                    current_response = str(retry_result)
                    current_sources = None

            else:
                # All retries exhausted — return last generated response anyway
                logger.warning(
                    "Partisan content persists after %d retries [user=%s] — "
                    "returning last generated response.",
                    MAX_PARTISAN_RETRIES,
                    ctx.user_id,
                )
                return current_response, current_sources

        except Exception as exc:
            logger.error(
                "check_partisan_response failed [user=%s attempt=%d error=%s] "
                "— returning current response.",
                ctx.user_id,
                attempt,
                exc,
            )
            return current_response, current_sources  # fail open
