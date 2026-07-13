"""
Simple split strategy for medium-length pages (150-800 words).
Splits on paragraph boundaries without overlap.
Used for press releases, forms, and other mid-length content.
"""
import re

from .semantic import enforce_chunk_caps

TARGET_CHUNK_WORDS = 250
MIN_CHUNK_WORDS = 30


def simple_split(text: str) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.

    Args:
        text: Full page text.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    # Split on newlines. trafilatura (v2) separates blocks with a SINGLE
    # \n, so splitting on blank lines only would leave whole pages as one
    # chunk. Each newline-delimited block is treated as a paragraph.
    paragraphs = re.split(r'\n+', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # If total text is short, return as single chunk
    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words <= TARGET_CHUNK_WORDS:
        return enforce_chunk_caps([text.strip()])

    # Group paragraphs into chunks up to target word count
    chunks = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > TARGET_CHUNK_WORDS and current_parts:
            chunk_text = '\n\n'.join(current_parts)
            chunks.append(chunk_text)
            current_parts = [para]
            current_words = para_words
        else:
            current_parts.append(para)
            current_words += para_words

    # Emit final chunk
    if current_parts:
        chunk_text = '\n\n'.join(current_parts)
        # If final chunk is very small, merge with previous
        if chunks and len(chunk_text.split()) < MIN_CHUNK_WORDS:
            chunks[-1] = chunks[-1] + '\n\n' + chunk_text
        else:
            chunks.append(chunk_text)

    # Safety net: any chunk still over the embedding caps (e.g. one giant
    # newline-free block) is re-split via semantic chunking.
    return enforce_chunk_caps(chunks)
