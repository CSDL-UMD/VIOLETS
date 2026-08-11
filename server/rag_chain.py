"""
LangChain RAG chain with conversation-aware retrieval.

Flow per query:
  1. Rephrase the user's question using chat history (handles follow-ups)
  2. Embed the rephrased question and retrieve top-k chunks from PostgreSQL (pgvector)
  3. Generate an answer grounded in retrieved context + chat history

Built with langchain_core runnables (no langchain.chains).
"""

import logging
import re
from datetime import datetime
from typing import Any

from pydantic import ConfigDict

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config
from .rag_logger import LOG_QUERIES
from .timing import stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, reformulate the "
     "question as a standalone question that can be understood without the "
     "chat history. Do NOT answer the question. If the question is already "
     "standalone, return it as-is. Return ONLY the reformulated question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def _escape_braces(value: str) -> str:
    """Escape { } so env-provided values can't break the f-string templates."""
    return value.replace("{", "{{").replace("}", "}}")


# Shared date/election context for the retrieval prompts. {{current_date}} is
# left as a template variable, filled per-request in full_pipeline. The env
# values are brace-escaped — a stray { or } in ELECTION_NAME/ELECTION_DATE
# would otherwise make every prompt format raise KeyError.
_ELECTION_CONTEXT = (
    f"Today's date is {{current_date}}. The upcoming election is the "
    f"{_escape_braces(config.ELECTION_NAME)}, held on "
    f"{_escape_braces(config.ELECTION_DATE)}. When stating "
    "deadlines or dates, always say which election they apply to, and never "
    "present information from a past election as if it were current.\n\n"
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Maryland elections information. "
     + _ELECTION_CONTEXT +
     "Answer questions as fully as possible using the provided context. "
     "Even if the context only partially covers the question, share what you know — "
     "do not refuse to answer just because the context is incomplete. "
     "Only say you don't have information if the context contains nothing relevant whatsoever.\n\n"
     "Each context chunk is labeled [Source N] with a title and URL. Cite your "
     "sources by naming them IN WORDS, not only with the tag:\n"
     "1. REQUIRED: your FIRST sentence must name the publishing organization "
     "or website in plain prose — e.g. \"According to the Maryland State Board "
     "of Elections, ...\" (infer the name from the source's title and URL). If "
     "the answer is a list, that naming sentence is the lead-in before the "
     "list. The [Source N] tags may be HIDDEN from the user, so an answer "
     "whose prose never names a source shows the user no citation at all — "
     "such an answer is incomplete.\n"
     "2. Immediately after each attributed statement, also add its [Source N] "
     "tag (the system uses these to resolve links and may hide them).\n"
     "3. Never write raw URLs or the source title in parentheses yourself — "
     "use only the [Source N] tag for the machine-readable citation.\n"
     "4. Do NOT append a separate \"Sources:\" or references list at the end. "
     "Cite inline only.\n"
     "5. Only attribute a statement to a source when the provided context "
     "actually supports it. If the context does not cover the question, say so "
     "plainly — never fabricate an attribution. Never write a URL that does "
     "not appear in the provided context or the conversation history.\n\n"
     "Be concise, accurate, and conversational.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Maryland elections information. "
     "The user is asking about the conversation itself — for example, "
     "summarizing what was discussed, clarifying a previous answer, or "
     "just being conversational. Answer based on the chat history. "
     "Never write a URL that does not appear in the conversation history. "
     "Be concise and friendly."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

#SEPARATE USER CONCERN PROMPTS: 
# When a user expresses a rumor, concern, or conspiracy theory about elections,
# we don't want to simply answer with retrieved chunks — that might inadvertently
# validate or engage with misinformation. Instead we overwrite the system prompt
# to direct the user to Maryland's official Rumor Control page first, then
# supplement with any relevant factual context from the RAG chain.
# This prompt is used when query_category == "concerns" OR when the survey
# system tags the query with "__User concerns:__".

_CONCERNS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Maryland elections information. "
     + _ELECTION_CONTEXT +
     "The user has expressed a concern, rumor, or question about election "
     "integrity or misinformation.\n\n"
     "IMPORTANT: Always start your response by directing the user to Maryland's "
     "official Rumor Control page for verified information: "
     "https://elections.maryland.gov/press_room/rumor_control.html\n\n"
     "After referencing Rumor Control, you may use the provided context to "
     "give additional factual information if relevant. Be empathetic, calm, "
     "and factual. Do not dismiss the user's concern — acknowledge it and "
     "redirect to official sources.\n\n"
     "When you use information from a source, attribute it in natural language "
     "within your sentence by naming the publishing organization or website — "
     "for example, \"According to the Maryland State Board of Elections...\". "
     "Infer the name from the source's title and URL. Immediately after the "
     "attributed statement, also add its [Source N] tag so the citation can be "
     "resolved (these tags are handled by the system and may be hidden from "
     "the user).\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
# ---------------------------------------------------------------------------
# Custom pgvector retriever
# ---------------------------------------------------------------------------

class PgVectorRetriever(BaseRetriever):
    """Retriever that queries PostgreSQL with pgvector using OpenAI embeddings."""

    embeddings: OpenAIEmbeddings
    pool: Any  # psycopg_pool.ConnectionPool
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_docs(self, rows) -> list[Document]:
        """Turn raw pgvector rows into Documents (shared by sync/async paths)."""
        # Optional relevance floor — drop chunks whose similarity score
        # (1 - cosine distance) is below SIMILARITY_FLOOR. The default 0.0
        # disables the filter, so nothing changes unless it's configured.
        if config.SIMILARITY_FLOOR > 0:
            kept = [row for row in rows if row[5] >= config.SIMILARITY_FLOOR]
            if len(kept) < len(rows):
                logger.info(
                    "Similarity floor %.2f dropped %d/%d chunks",
                    config.SIMILARITY_FLOOR, len(rows) - len(kept), len(rows),
                )
            rows = kept
        if not rows:
            logger.warning("pgvector returned no matches for query")

        docs = []
        for row in rows:
            chunk_id, text, source_url, title, metadata, score = row
            meta = dict(metadata) if metadata else {}
            meta["source_url"] = source_url or "unknown"
            meta["title"] = title or ""
            meta["score"] = round(score, 4)
            # Deduplicated chunks carry a `source_urls` list in the JSONB
            # metadata (all URLs the content appears at). Surface the full,
            # deduped list so citations can include every source. Fall back to
            # the scalar source_url for chunks that only have one.
            raw_urls = meta.get("source_urls") or [meta["source_url"]]
            seen, urls = set(), []
            for u in raw_urls:
                if u and u != "unknown" and u not in seen:
                    seen.add(u)
                    urls.append(u)
            meta["source_urls"] = urls or [meta["source_url"]]
            logger.debug(
                "  Retrieved [%.4f] %s — %s",
                score, source_url, (text or "")[:80].replace("\n", " ")
            )
            docs.append(Document(page_content=text or "", metadata=meta))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Both the embedding call and the pgvector query are genuinely async
        # (AsyncOpenAI client + AsyncConnectionPool), so a concurrent retrieval
        # no longer pins an anyio threadpool worker for its full duration; it
        # only holds a pool connection for the (indexed) vector query itself.
        with stage("embed"):
            query_embedding = await self.embeddings.aembed_query(query)

        with stage("retrieve"):
            async with self.pool.connection() as conn:
                # Bound query time so a slow/hung pgvector scan can't hold a pool
                # connection indefinitely. SET LOCAL applies within the implicit
                # transaction of this (non-autocommit) connection.
                await conn.execute("SET LOCAL statement_timeout = '30s'")
                cur = await conn.execute(
                    """
                    SELECT chunk_id, text, source_url, title, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, self.k),
                )
                rows = await cur.fetchall()

        return self._build_docs(rows)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Async-only: this retriever holds an AsyncConnectionPool and is always
        # driven via `ainvoke`. A sync call would need a separate sync pool, so
        # fail loudly rather than silently degrade.
        raise NotImplementedError(
            "PgVectorRetriever is async-only; use `ainvoke` / "
            "`aget_relevant_documents`."
        )


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        primary = doc.metadata.get("source_url", "unknown")
        urls = doc.metadata.get("source_urls") or [primary]
        title = doc.metadata.get("title", "")
        header = f"[Source {i}]: {title}" if title else f"[Source {i}]"
        url_line = "URL: " + " ; ".join(urls)
        parts.append(f"{header}\n{url_line}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# Matches a user asking for the underlying links/sources/citations. When this
# fires we render the [Source N] markers as markdown links and return the
# `sources` array; otherwise the markers are stripped and only the natural-
# language attribution the model wrote in prose is shown.
_SOURCE_LINK_REQUEST_RE = re.compile(
    r"\b(link|links|url|urls|source|sources|cite|cited|citation|citations|"
    r"reference|references|hyperlink|hyperlinks)\b",
    re.IGNORECASE,
)


def _wants_source_links(query: str) -> bool:
    """True if the user's query explicitly asks for source links/citations."""
    return bool(_SOURCE_LINK_REQUEST_RE.search(query or ""))


# Strict intent match used ONLY for the conversational-path short-circuit that
# replaces the LLM reply with the cached link list. Unlike the broad
# mention-based regex above (kept for its link-rendering role), this requires
# an explicit request shape — "can you send me the links?", "what's the URL?",
# "sources please" — and must NOT fire on negations ("thanks, I don't need the
# links") or mere mentions ("what did you mean by that reference?"), which
# fall through to the conversational LLM instead.
_NEGATION_WORDS = r"(?:no|not|never|don'?t|won'?t|wouldn'?t|without)"
_EXPLICIT_LINK_REQUEST_RE = re.compile(
    # "(can|could|will|would) you ... <send-verb> ... <link-noun>", with no
    # negation word anywhere between "you" and the noun.
    r"\b(?:can|could|will|would)\s+you\b"
    rf"(?:(?!\b{_NEGATION_WORDS}\b).)*?"
    r"\b(?:give|send|share|show|provide|resend)\b"
    rf"(?:(?!\b{_NEGATION_WORDS}\b).)*?"
    r"\b(?:links?|sources?|urls?|citations?|references?)\b"
    # "what (is|are|'s) the (link|links|url|urls|source|sources)"
    r"|\bwhat(?:'s|\s+(?:is|are))\s+the\s+(?:links?|urls?|sources?)\b"
    # bare "(the) links/sources/urls please" — not preceded by a negation
    r"|(?<!no )(?<!not )\b(?:the\s+)?(?:links?|sources?|urls?),?\s+please\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_explicit_link_request(query: str) -> bool:
    """True only for an explicit ask to (re)send links — gates the
    conversational-path cached-links short-circuit."""
    return bool(_EXPLICIT_LINK_REQUEST_RE.search(query or ""))


def _format_cached_source_links(sources: list[dict]) -> str:
    """Render the previous turn's cached sources as a markdown link list.

    Used for link follow-ups on the conversational path, where no retrieval
    runs and the model would otherwise have to write URLs from memory.
    """
    lines, seen = [], set()
    for s in sources:
        title = s.get("title") or ""
        urls = s.get("source_urls") or ([s["source_url"]] if s.get("source_url") else [])
        for u in urls:
            if not u or u == "unknown" or u in seen:
                continue
            seen.add(u)
            lines.append(f"- [{title or u}]({u})")
    if not lines:
        return (
            "I don't have any source links from my previous answer to share. "
            "Ask me a Maryland elections question and I can cite sources for it."
        )
    return "Here are the sources from my previous answer:\n" + "\n".join(lines)


# Matches [Source N] citation markers, including combined comma-separated forms
# the model sometimes emits ("[Source 2, Source 1, 3]"). Any leading whitespace
# is captured so stripping a marker doesn't leave a dangling space before
# punctuation. The inner group holds the digit/comma run for number extraction.
_SOURCE_MARKER_RE = re.compile(
    r"\s*\[\s*source[s]?\s+([\d]+(?:\s*,\s*(?:source\s+)?\d+)*)\s*\]",
    re.IGNORECASE,
)

# Catch-all for any leftover "[Source ...]" bracket the model invents in a form
# we can't resolve to a numbered citation — e.g. "[Source: Challenger Manual]".
# The negative lookahead `(?!\()` skips real markdown links "[text](url)" so we
# never strip a rendered citation, only stray literal tags. Applied last in both
# modes so no citation junk ever reaches the user.
_LEFTOVER_SOURCE_RE = re.compile(r"\s*\[\s*sources?\b[^\]]*\](?!\()", re.IGNORECASE)


def _strip_source_refs(answer: str) -> str:
    """Remove internal [Source N] markers, leaving the prose attribution intact.

    The model is prompted to name the source organization in natural language
    (e.g. "According to the Maryland State Board of Elections...") AND to tag it
    with [Source N]. By default we hide the tags so the citation stays embedded
    in the message rather than surfaced as a link.
    """
    # Drop any whitespace immediately before the marker so "text [Source 1]."
    # collapses cleanly to "text.". Handles single markers and comma-separated
    # combined forms the model sometimes emits, e.g. "[Source 2, Source 1, 3]".
    cleaned = _SOURCE_MARKER_RE.sub("", answer)
    # Sweep any non-numbered leftover tags the model invented.
    cleaned = _LEFTOVER_SOURCE_RE.sub("", cleaned)
    # Tidy any doubled spaces left mid-sentence.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _replace_source_refs(answer: str, sources: list[dict]) -> str:
    """Replace [Source N] markers in the answer with markdown links."""
    source_map = {
        s["source_number"]: s for s in sources
    }

    def _links_for(num: int) -> list[str]:
        src = source_map.get(num)
        if not src or src["source_url"] == "unknown":
            return []
        urls = src.get("source_urls") or [src["source_url"]]
        title = src.get("title") or src["source_url"]
        if len(urls) == 1:
            return [f"[{title}]({urls[0]})"]
        # Multi-source chunk: render the title link to the primary URL plus
        # the remaining URLs as numbered links so all citations are surfaced.
        links = [f"[{title}]({urls[0]})"]
        links += [f"[{j}]({u})" for j, u in enumerate(urls[1:], 2)]
        return links

    def _sub(m: re.Match) -> str:
        # The marker may combine several numbers ("[Source 2, Source 1, 3]");
        # render each to a link, de-duplicating so a repeated source appears once.
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        links, seen = [], set()
        for num in nums:
            for link in _links_for(num):
                if link not in seen:
                    seen.add(link)
                    links.append(link)
        # No resolvable link (all "unknown"/missing) — drop the raw marker
        # rather than leaking "[Source N]" literal text to the user.
        return " " + " ".join(links) if links else ""

    rendered = _SOURCE_MARKER_RE.sub(_sub, answer)
    # Strip any leftover unresolved "[Source: ...]" tags the model invented that
    # aren't numbered markers (and so weren't turned into links above).
    return _LEFTOVER_SOURCE_RE.sub("", rendered)


def build_chain(pool):
    """Build and return the full RAG chain. Called once at server startup."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    retriever = PgVectorRetriever(
        embeddings=embeddings,
        pool=pool,
        k=config.RETRIEVER_K,
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        reasoning_effort=config.RAG_REASONING_EFFORT,
        verbosity="low",
    )

    rephrase_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
    qa_chain = _QA_PROMPT | llm | StrOutputParser()
    conversational_chain = _CONVERSATIONAL_PROMPT | llm | StrOutputParser()
    concerns_chain = _CONCERNS_PROMPT | llm | StrOutputParser()

    async def full_pipeline(inputs: dict, run_config: RunnableConfig | None = None) -> dict:
        chat_history = inputs.get("chat_history", [])
        user_input = inputs["input"]
        query_category = inputs.get("query_category")

        if query_category == "conversational":
            # Link follow-ups ("can you give me the links?") reference the
            # previous answer, whose sources the model can't see — answer from
            # the cached sources for that turn instead of letting the model
            # invent URLs. Honest empty message when nothing is cached.
            # skip_partisan tells main.py this answer is a deterministic
            # template over already-vetted titles/URLs, so the partisan
            # checker (an LLM call) is skipped for it.
            if _is_explicit_link_request(user_input):
                last_sources = inputs.get("last_sources") or []
                logger.debug("Conversational link request — using cached sources")
                return {
                    "answer": _format_cached_source_links(last_sources),
                    "sources": last_sources,
                    "skip_partisan": True,
                }
            logger.debug("Conversational query — answering without retrieval")
            with stage("generate"):
                answer = await conversational_chain.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history,
                }, run_config)
            return {"answer": answer, "sources": []}

        # Strip the survey tag from concerns queries so the LLM doesn't get
        # confused by the "__User concerns:__" prefix. Only used by the concerns
        # branch below; the rephrase step still sees the raw user input so it
        # has the full original phrasing available.
        clean_input = user_input.replace("__User concerns:__", "").strip()

        # Step 1: rephrase follow-ups into standalone questions
        if chat_history:
            with stage("rephrase"):
                standalone_q = await rephrase_chain.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history,
                }, run_config)
            if LOG_QUERIES:
                logger.info("Rephrased: '%s' → '%s'", user_input, standalone_q)
        else:
            standalone_q = user_input
            if LOG_QUERIES:
                logger.info("Query: '%s'", standalone_q)

        # Step 2: retrieve from pgvector
        logger.debug("Retrieving top-%d from pgvector...", retriever.k)
        docs = await retriever.ainvoke(standalone_q, run_config)

        # Step 3: extract source metadata for the response
        sources = []
        for i, doc in enumerate(docs, 1):
            primary = doc.metadata.get("source_url", "unknown")
            source_urls = doc.metadata.get("source_urls") or [primary]
            sources.append({
                "source_number": i,
                "source_url": primary,
                "source_urls": source_urls,
                "title": doc.metadata.get("title", ""),
                "score": round(doc.metadata.get("score", 0), 4),
            })

        # Step 4: generate answer with citations. Concerns queries use the
        # Rumor Control system prompt; everything else uses the standard QA prompt.
        # Both prompts carry the election/date context, so fill in today's date.
        now = datetime.now()
        current_date = f"{now:%B} {now.day}, {now.year}"
        if query_category == "concerns":
            logger.debug("Concerns query — using Rumor Control system prompt")
            with stage("generate"):
                answer = await concerns_chain.ainvoke({
                    "context": _format_docs(docs),
                    "input": clean_input,
                    "chat_history": chat_history,
                    "current_date": current_date,
                }, run_config)
        else:
            with stage("generate"):
                answer = await qa_chain.ainvoke({
                    "context": _format_docs(docs),
                    "input": user_input,
                    "chat_history": chat_history,
                    "current_date": current_date,
                }, run_config)

        # By default keep citations embedded in the prose (the model names the
        # source organization inline) and hide the [Source N] markers + link
        # list. Only when the user explicitly asks for links/sources do we
        # render the markdown links and surface the sources array.
        # retrieved_sources is always the full list (even when hidden from the
        # response) so main.py can cache it for link follow-ups.
        retrieved_sources = list(sources)
        if _wants_source_links(user_input):
            answer = _replace_source_refs(answer, sources)
        else:
            answer = _strip_source_refs(answer)
            sources = []

        return {"answer": answer, "sources": sources,
                "retrieved_sources": retrieved_sources}

    return RunnableLambda(full_pipeline)


# ---------------------------------------------------------------------------
# History conversion
# ---------------------------------------------------------------------------

def to_langchain_messages(messages: list[dict]) -> list:
    """Convert session message dicts to LangChain message objects."""
    out = []
    for msg in messages:
        if msg["role"] == "user":
            out.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            out.append(AIMessage(content=msg["content"]))
    return out
