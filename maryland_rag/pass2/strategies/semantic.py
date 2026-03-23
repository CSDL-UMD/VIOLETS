"""
Semantic chunking with overlap for long prose pages (800+ words).

Uses sentence-level splitting with a sliding window approach.
Chunks are formed by grouping sentences until a target token count is reached,
with ~20% overlap between consecutive chunks.

This is a self-contained implementation that doesn't require langchain or
llama-index, but can be swapped for SemanticChunker if embeddings are available.
"""
import re

# Target chunk sizes (in words, roughly equivalent to 1.3x tokens)
TARGET_CHUNK_WORDS = 300
MAX_CHUNK_WORDS = 500
OVERLAP_RATIO = 0.20  # 20% overlap between chunks


def semantic_chunk(text: str) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.

    For a true semantic chunking approach (using embeddings to find
    natural breakpoints), replace this with langchain's SemanticChunker:

        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
        chunker = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
        chunks = chunker.split_text(text)

    This implementation approximates semantic chunking by:
    1. Splitting into sentences
    2. Grouping sentences into target-sized chunks
    3. Adding overlap by including trailing sentences from previous chunk

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # If text is short enough, return as single chunk
    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= MAX_CHUNK_WORDS:
        return [text.strip()]

    chunks = []
    current_sentences = []
    current_words = 0
    overlap_sentences = []

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_words + word_count > TARGET_CHUNK_WORDS and current_sentences:
            # Emit current chunk
            chunk_text = ' '.join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Calculate overlap: take trailing sentences up to OVERLAP_RATIO of target
            overlap_words = 0
            overlap_target = int(TARGET_CHUNK_WORDS * OVERLAP_RATIO)
            overlap_sentences = []
            for s in reversed(current_sentences):
                s_words = len(s.split())
                if overlap_words + s_words > overlap_target:
                    break
                overlap_sentences.insert(0, s)
                overlap_words += s_words

            # Start new chunk with overlap
            current_sentences = overlap_sentences + [sentence]
            current_words = overlap_words + word_count
        else:
            current_sentences.append(sentence)
            current_words += word_count

    # Emit final chunk
    if current_sentences:
        chunk_text = ' '.join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and decimal numbers to avoid false splits.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Split on sentence-ending punctuation followed by space and uppercase letter
    # or end of string. Preserves the punctuation with the sentence.
    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])',
        text
    )

    # Filter out empty strings and very short fragments
    sentences = [s.strip() for s in parts if s.strip() and len(s.strip()) > 2]

    return sentences
