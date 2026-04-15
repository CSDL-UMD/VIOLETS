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
from typing import Any

from pydantic import ConfigDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config

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

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Maryland elections information. "
     "Answer questions using the provided context. If the context doesn't "
     "contain enough information to answer, say you don't have that "
     "information and suggest what the user could try instead.\n\n"
     "Each context chunk is labeled [Source N] with a URL. When you use "
     "information from a source, cite it inline using [Source N] notation. "
     "Always cite your sources so users can verify the information.\n\n"
     "Be concise, accurate, and conversational.\n\n"
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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)

        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, text, source_url, title, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, self.k),
            ).fetchall()

        if not rows:
            logger.warning("pgvector returned no matches for query")

        docs = []
        for row in rows:
            chunk_id, text, source_url, title, metadata, score = row
            meta = dict(metadata) if metadata else {}
            meta["source_url"] = source_url or "unknown"
            meta["title"] = title or ""
            meta["score"] = round(score, 4)
            logger.info(
                "  Retrieved [%.4f] %s — %s",
                score, source_url, (text or "")[:80].replace("\n", " ")
            )
            docs.append(Document(page_content=text or "", metadata=meta))
        return docs


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_url", "unknown")
        title = doc.metadata.get("title", "")
        header = f"[Source {i}]: {title}" if title else f"[Source {i}]"
        parts.append(f"{header}\nURL: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _replace_source_refs(answer: str, sources: list[dict]) -> str:
    """Replace [Source N] markers in the answer with markdown links."""
    source_map = {
        s["source_number"]: s for s in sources
    }

    def _sub(m: re.Match) -> str:
        num = int(m.group(1))
        src = source_map.get(num)
        if not src or src["source_url"] == "unknown":
            return m.group(0)
        title = src.get("title") or src["source_url"]
        return f"[{title}]({src['source_url']})"

    return re.sub(r"\[Source\s+(\d+)\]", _sub, answer, flags=re.IGNORECASE)


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
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    # Step 1: Rephrase chain — converts follow-ups into standalone questions
    rephrase_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    qa_chain = _QA_PROMPT | llm | StrOutputParser()

    async def full_pipeline(inputs: dict, run_config: RunnableConfig | None = None) -> dict:
        chat_history = inputs.get("chat_history", [])
        user_input = inputs["input"]

        # Step 1: rephrase follow-ups into standalone questions
        if chat_history:
            standalone_q = await rephrase_chain.ainvoke({
                "input": user_input,
                "chat_history": chat_history,
            }, run_config)
            logger.info("Rephrased: '%s' → '%s'", user_input, standalone_q)
        else:
            standalone_q = user_input
            logger.info("Query: '%s'", standalone_q)

        # Step 2: retrieve from pgvector
        logger.info("Retrieving top-%d from pgvector...", retriever.k)
        docs = await retriever.ainvoke(standalone_q, run_config)

        # Step 3: extract source metadata for the response
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "source_number": i,
                "source_url": doc.metadata.get("source_url", "unknown"),
                "title": doc.metadata.get("title", ""),
                "score": round(doc.metadata.get("score", 0), 4),
            })

        # Step 4: generate answer with citations
        answer = await qa_chain.ainvoke({
            "context": _format_docs(docs),
            "input": user_input,
            "chat_history": chat_history,
        }, run_config)

        answer = _replace_source_refs(answer, sources)

        return {"answer": answer, "sources": sources}

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
