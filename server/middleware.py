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
    - Partisan response check must run after chain — needs LLM output.

QUERY CATEGORIES AND WHAT HAPPENS TO THEM:
-------------------------------------------
    - normal          → RAG chain runs normally
    - conversational  → RAG chain runs using chat history only
    - concerns        → RAG chain runs with Rumor Control system prompt
    - polling_location → hardcoded URL returned, chain never runs
    - voter_lookup     → hardcoded URL returned, chain never runs
    - voter_update     → hardcoded URL returned, chain never runs
    - candidates       → hardcoded URL returned, chain never runs
    - partisan         → blocked entirely, fallback message returned
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine 

from . import config

logger = logging.getLogger(__name__)

# Hard ceiling on the small guardrail LLM calls so a hung upstream can't
# stall the request pipeline. A timeout is treated exactly like any other
# exception by the existing fail-open except blocks.
GUARDRAIL_LLM_TIMEOUT = 30  # seconds

# The partisan retry re-invokes the full RAG chain, so it gets the larger
# chain-sized ceiling rather than the small-LLM one.
PARTISAN_RETRY_TIMEOUT = 60  # seconds


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
        Possible values: "normal", "conversational", "concerns",
        "polling_location", "voter_lookup", "voter_update",
        "candidates", "partisan"

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
#
# WHY HARDCODED URLS FOR SOME CATEGORIES:
#   For polling location, voter lookup, voter update, and candidates,
#   we return a direct URL instead of running the RAG chain because:
#   1. The answer is always the same URL — no retrieval needed
#   2. These are official Maryland election tools — better to send users
#      there directly than have the LLM describe them
#   3. Zero token cost — no LLM call needed at all
# ---------------------------------------------------------------------------

FALLBACK_RESPONSES = {
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
    "polling_location": (
        "You can find your polling place using the official Maryland Polling "
        "Place Search tool here: "
        "https://voterservices.elections.maryland.gov/PollingPlaceSearch"
    ),
    "voter_lookup": (
        "You can look up your voter registration information using the "
        "official Maryland Voter Search tool here: "
        "https://voterservices.elections.maryland.gov/VoterSearch"
    ),
    "voter_update": (
        "You can update your voter registration information online here: "
        "https://voterservices.elections.maryland.gov/OnlineVoterUpdate/InstructionsStep1"
    ),
    "candidates": (
        "You can find the up-to-date list of candidates for the 2026 "
        "Maryland Primary Election here: "
        "https://elections.maryland.gov/elections/2026/primary_candidates/index.html"
    ),
    "error": (
        "Sorry, I couldn't process your question right now. Please try "
        "rephrasing it, and I'll do my best to help with Maryland election "
        "information."
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
    """

    query = str(query)
    try:
        results = _analyzer.analyze(
            text=query,
            language="en",
            entities=_PII_ENTITIES,
        )
    except Exception as exc:
        # Fail closed: if the analyzer errors we can't confirm the input is
        # clean, so treat it as if PII were detected and block the request
        # rather than letting unscanned input through.
        logger.error(
            "detect_pii failed [user=%s error=%s] — blocking (fail closed).",
            ctx.user_id,
            exc,
        )
        ctx.pii_detected = True
        ctx.pii_type = "unknown"
        return FALLBACK_RESPONSES["pii"]

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
# We use an LLM here instead of regex because intent is expressed in
# natural language and can't be caught with keywords alone.
#
# Each category maps to a specific behavior:
#   - normal/conversational/concerns → pass through to RAG chain
#   - polling_location/voter_lookup/voter_update/candidates → hardcoded URL
#   - partisan → blocked entirely
#
# WHY SEPARATE CATEGORIES FOR EACH URL:
#   Each URL serves a different user need. The category name is used as
#   the key to look up the correct hardcoded response in FALLBACK_RESPONSES.
#   If they shared a category we wouldn't know which URL to return.
# ---------------------------------------------------------------------------

# Categories that pass through to the RAG chain (with or without
# system prompt modification). Everything else gets a hardcoded response
# or is blocked entirely.

_PASSTHROUGH_CATEGORIES = {"normal", "conversational", "concerns"}
class ClassificationResult(BaseModel):
    """
    Structured output from the classifier LLM.
    """
    category: Literal["normal", "conversational", "concerns", "polling_location", "voter_lookup", "voter_update", "candidates", "partisan"]
    reason: str  # used for logging only, never shown to the user


# Build the classifier once at module load time, not on every request.
_classifier_llm = (
    ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        reasoning_effort="medium",
        verbosity="low",
    )
    .with_structured_output(ClassificationResult)
)

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query classifier for VIOLETS, a voter information chatbot
for Maryland elections.

Classify the user query into exactly one of the following categories:

- normal          : any query about Maryland voting, elections, or civic
                    topics — OR any query that doesn't clearly fit the
                    other categories. When in doubt, classify as normal.

- conversational  : the query is about the conversation itself — e.g.
                  summarizing what was discussed, asking what was said
                  earlier, requesting clarification of a previous answer,
                  saying thanks, or other meta/social messages that do
                  not require external knowledge.

- concerns        : the query expresses concerns, rumors, conspiracy
                    theories, or misinformation about elections — e.g.
                    "I heard the election is rigged", "is mail-in voting
                    fraudulent?", "I don't trust the voting machines".
                    Also applies when the query is prefixed with
                    "__User concerns:__" from the survey system.                  
- polling_location : the user is asking where to vote, where their
                    polling place is, or what their polling location is.
 
- voter_lookup    : the user wants to look up or check their voter
                    registration status or information.
 
- voter_update    : the user wants to update, change, or correct their
                    voter registration information.
 
- candidates      : the user is asking about who is running, candidate
                    lists, or who is on the ballot.

- partisan        : the query requests candidate endorsements, asks
                  which party is better, or asks for partisan political
                  judgments about candidates or parties.

Return your classification and a brief reason (1 sentence).
Be decisive — every query must map to exactly one category.
"""

async def classify_query(query: str, ctx: QueryContext) -> str | None:
    """
    Classify the user query using a lightweight LLM.
 
    Returns None for categories that should reach the RAG chain
    (normal, conversational, concerns). Returns a hardcoded response
    string for categories that should be short-circuited (polling_location,
    voter_lookup, voter_update, candidates, partisan).
 
    The query_category is always set on ctx so downstream components
    (e.g. rag_chain.py) can adjust their behavior — for example,
    the concerns category causes the system prompt to be overwritten
    with a Rumor Control directive.
 
    Fails closed: if the classifier LLM errors, logs and returns a safe
    canned fallback asking the user to rephrase, so an unclassified query
    never reaches the RAG chain unfiltered.
    """
    # Check for hardcoded survey tag first — no LLM call needed
    # The survey system prefixes concern queries with "__User concerns:__"
    if query.strip().startswith("__User concerns:__"):
        ctx.query_category = "concerns"
        ctx.safety_flag = False
        logger.info("Query tagged as concerns by survey system [user=%s]", ctx.user_id)
        return None  # pass through to RAG chain with concerns prompt
 
    try:
        result: ClassificationResult = await asyncio.wait_for(
            _classifier_llm.ainvoke([
                SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]),
            timeout=GUARDRAIL_LLM_TIMEOUT,
        )
 
        ctx.query_category = result.category
        ctx.safety_flag = result.category == "partisan"
 
        logger.info(
            "Query classified [user=%s category=%s reason=%s]",
            ctx.user_id,
            result.category,
            result.reason,
        )
 
        # Passthrough categories go to the RAG chain
        if result.category in _PASSTHROUGH_CATEGORIES:
            return None
 
        # All other categories return their specific hardcoded response
        # Each category has its own key in FALLBACK_RESPONSES so we
        # return exactly the right URL or message for that category
        return FALLBACK_RESPONSES[result.category]
 
    except Exception as exc:
        logger.error(
            "classify_query failed [user=%s error=%s] — blocking query (fail closed).",
            ctx.user_id,
            exc,
        )
        # Fail closed: do NOT let an unclassified query reach the RAG chain
        # unfiltered. Return a safe canned fallback so the user is asked to
        # rephrase instead of receiving an unvetted answer.
        ctx.safety_flag = True
        return FALLBACK_RESPONSES["error"]  # fail closed
 


# ---------------------------------------------------------------------------
# SECTION 5: Partisan response guardrail (wrap-style, small LLM)
# ---------------------------------------------------------------------------
# This runs AFTER the RAG chain. It catches cases where a normal-looking
# query accidentally gets a partisan answer from the LLM — for example,
# "Who are the candidates for County Executive?" is a legitimate question
# but the LLM might unexpectedly add "Candidate X has a stronger record."
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
        model=config.LLM_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        reasoning_effort="medium",
        verbosity="low",
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
    callbacks: list | None = None,
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
 
    Fails closed on exceptions: if the checker itself errors, the unchecked
    response is discarded and the safe canned nonpartisan fallback is
    returned instead.
    """
    current_response = response
    current_sources = None

    for attempt in range(MAX_PARTISAN_RETRIES + 1):  # 0 = original, 1-2 = retries
        try:
            result = await asyncio.wait_for(
                _partisan_checker_llm.ainvoke([
                    SystemMessage(content=_PARTISAN_CHECKER_SYSTEM_PROMPT),
                    HumanMessage(content=current_response),
                ]),
                timeout=GUARDRAIL_LLM_TIMEOUT,
            )

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
                retry_result = await asyncio.wait_for(
                    chain.ainvoke(
                        {
                            "input": query + "\n\n" + _STRICT_NONPARTISAN_RETRY_PROMPT,
                            "chat_history": chat_history,
                            # Force the standard RAG-with-retrieval path so the
                            # strict-nonpartisan rewrite always runs grounded in
                            # retrieved context, never the conversational
                            # (no-retrieval) branch.
                            "query_category": "normal",
                        },
                        config={"callbacks": callbacks} if callbacks else None,
                    ),
                    timeout=PARTISAN_RETRY_TIMEOUT,
                )
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
                "— returning safe nonpartisan fallback (fail closed).",
                ctx.user_id,
                attempt,
                exc,
            )
            # Fail closed: the response was never verified, so never show the
            # unchecked LLM output. Return the canned nonpartisan fallback and
            # drop any retry sources tied to the unverified answer.
            return FALLBACK_RESPONSES["partisan"], None  # fail closed
