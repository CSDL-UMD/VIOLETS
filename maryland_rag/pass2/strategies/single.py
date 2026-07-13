"""
Single-chunk strategy for short pages (<150 words).
The entire page content becomes one chunk with no splitting.
"""
from .semantic import enforce_chunk_caps


def ingest_as_single(text: str) -> list[str]:
    """
    Return the text as a single chunk.

    Args:
        text: Full page text.

    Returns:
        List containing a single chunk string, or empty list if no content.
        Pages misclassified as short (or containing OCR junk) that exceed
        the embedding-safe caps are split rather than emitted oversized.
    """
    if not text or not text.strip():
        return []
    return enforce_chunk_caps([text.strip()])
