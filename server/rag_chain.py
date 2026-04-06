"""
LangChain RAG chain with conversation-aware retrieval.

Flow per query:
  1. Rephrase the user's question using chat history (handles follow-ups)
  2. Embed the rephrased question and retrieve top-k chunks from Pinecone
  3. Generate an answer grounded in retrieved context + chat history

Built with langchain_core runnables (no langchain.chains, no langchain-pinecone).
"""

import logging

from pydantic import ConfigDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

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
     "Be concise, accurate, and conversational.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ---------------------------------------------------------------------------
# Custom Pinecone retriever
# ---------------------------------------------------------------------------

class PineconeRetriever(BaseRetriever):
    """Retriever that queries Pinecone directly using OpenAI embeddings."""

    embeddings: OpenAIEmbeddings
    index: object  # Pinecone Index
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=self.k,
            include_metadata=True,
        )

        if isinstance(results, dict):
            matches = results.get("matches", [])
        else:
            matches = getattr(results, "matches", []) or []

        if not matches:
            logger.warning("Pinecone returned no matches for query")

        docs = []
        for match in matches:
            if isinstance(match, dict):
                meta = dict(match.get("metadata", {}))
                score = match.get("score", 0)
            else:
                meta = dict(getattr(match, "metadata", {}) or {})
                score = getattr(match, "score", 0) or 0
            text = meta.pop("text", "")
            source = meta.get("source_url", "unknown")
            logger.info(
                "  Retrieved [%.4f] %s — %s",
                score, source, text[:80].replace("\n", " ")
            )
            meta["score"] = score
            docs.append(Document(page_content=text, metadata=meta))
        return docs


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain():
    """Build and return the full RAG chain. Called once at server startup."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_BASE_URL,
    )

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    retriever = PineconeRetriever(
        embeddings=embeddings,
        index=index,
        k=config.RETRIEVER_K,
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_BASE_URL,
    )

    # Step 1: Rephrase chain — converts follow-ups into standalone questions
    rephrase_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    async def contextualize_and_retrieve(inputs: dict, config: RunnableConfig) -> dict:
        chat_history = inputs.get("chat_history", [])
        user_input = inputs["input"]

        # If there's history, rephrase; otherwise use as-is
        if chat_history:
            standalone_q = await rephrase_chain.ainvoke({
                "input": user_input,
                "chat_history": chat_history,
            }, config)
            logger.info("Rephrased: '%s' → '%s'", user_input, standalone_q)
        else:
            standalone_q = user_input
            logger.info("Query: '%s'", standalone_q)

        # Retrieve using the standalone question
        logger.info("Retrieving top-%d from Pinecone...", retriever.k)
        docs = await retriever.ainvoke(standalone_q, config)
        return {
            "context": _format_docs(docs),
            "input": user_input,
            "chat_history": chat_history,
        }

    # Step 2: QA chain — generates answer from context + history
    qa_chain = _QA_PROMPT | llm | StrOutputParser()

    # Combined chain: rephrase -> retrieve -> answer
    full_chain = RunnableLambda(contextualize_and_retrieve) | qa_chain

    return full_chain


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
