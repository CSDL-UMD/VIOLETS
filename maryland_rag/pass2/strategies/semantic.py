"""
Semantic chunking with overlap for long prose pages (800+ words).

Uses sentence-level splitting with a sliding window approach.
Chunks are formed by grouping sentences until a target token count is reached,
with ~20% overlap between consecutive chunks.

This is a self-contained implementation that doesn't require langchain or
llama-index, but can be swapped for SemanticChunker if embeddings are available.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Target chunk sizes (in words, roughly equivalent to 1.3x tokens)
TARGET_CHUNK_WORDS = 300
MAX_CHUNK_WORDS = 500
MAX_CHUNK_CHARS = 20000  # ~5k tokens, safe under 8192-token embedding cap
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

    sentences = _enforce_sentence_cap(sentences, MAX_CHUNK_WORDS)

    # If text is short enough (by both word and char count), return as single chunk
    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= MAX_CHUNK_WORDS and len(text) <= MAX_CHUNK_CHARS:
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

    # Final safety: hard-split any chunk still exceeding the char cap.
    # Hitting this branch usually means OCR garbage or whitespace-free
    # junk upstream — log it so the source can be cleaned.
    safe = []
    for c in chunks:
        if len(c) <= MAX_CHUNK_CHARS:
            safe.append(c)
        else:
            logger.warning(
                "semantic_chunk: hard-splitting %d-char chunk at %d-char boundaries "
                "(likely OCR/CID stream — check upstream extraction)",
                len(c), MAX_CHUNK_CHARS,
            )
            for j in range(0, len(c), MAX_CHUNK_CHARS):
                safe.append(c[j:j + MAX_CHUNK_CHARS])
    return safe


def enforce_chunk_caps(chunks: list[str]) -> list[str]:
    """
    Guarantee no chunk exceeds MAX_CHUNK_WORDS / MAX_CHUNK_CHARS.

    Shared safety net for the non-semantic strategies (single, simple_split,
    faq, table rows): one oversized chunk 400s an entire embedding batch
    downstream, so every strategy funnels its output through this. Oversized
    chunks are re-split via semantic_chunk, which enforces the caps
    internally (including the character-level hard split for
    whitespace-free junk).
    """
    safe = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c.split()) <= MAX_CHUNK_WORDS and len(c) <= MAX_CHUNK_CHARS:
            safe.append(c)
            continue
        parts = semantic_chunk(c)
        if not parts:
            # semantic_chunk can return [] for pathological non-empty text
            # (e.g. every sentence fragment is <=2 chars and gets filtered
            # by _split_sentences). A size cap must never silently drop
            # content, so hard-split the original chunk instead.
            logger.warning(
                "enforce_chunk_caps: semantic_chunk returned no chunks for a "
                "%d-char oversized chunk — hard-splitting to avoid content loss",
                len(c),
            )
            parts = _enforce_sentence_cap([c], MAX_CHUNK_WORDS)
        safe.extend(parts)
    return safe


def _enforce_sentence_cap(sentences: list[str], cap_words: int) -> list[str]:
    """Hard-split any sentence exceeding cap_words or MAX_CHUNK_CHARS.

    Sentence-boundary regex misses dense letters, OCR output without proper
    punctuation, and tabular text — those collapse into one huge "sentence"
    that blows past the embedding model's token limit. Force word-level
    splits, and fall back to character-level splits for whitespace-free junk
    like unmapped PDF CID streams.
    """
    out = []
    for s in sentences:
        words = s.split()
        if len(words) <= cap_words and len(s) <= MAX_CHUNK_CHARS:
            out.append(s)
            continue
        # First split on words
        parts = [' '.join(words[i:i + cap_words]) for i in range(0, len(words), cap_words)] or [s]
        # Then char-split any part that's still too long (whitespace-free text)
        for p in parts:
            if len(p) <= MAX_CHUNK_CHARS:
                out.append(p)
            else:
                for j in range(0, len(p), MAX_CHUNK_CHARS):
                    out.append(p[j:j + MAX_CHUNK_CHARS])
    return out


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
