"""
Single-chunk strategy for short pages (<150 words).
The entire page content becomes one chunk with no splitting.
"""


def ingest_as_single(text: str) -> list[str]:
    """
    Return the text as a single chunk.

    Args:
        text: Full page text.

    Returns:
        List containing a single chunk string, or empty list if no content.
    """
    if not text or not text.strip():
        return []
    return [text.strip()]
