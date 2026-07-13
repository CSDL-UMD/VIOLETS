"""
Content-based non-English chunk filter.

The corpus serves an English-language chatbot. Translated documents are
excluded by URL pattern in pass1 (see exclusions.EXCLUDED_PATH_PATTERNS),
but extraction can still emit mixed-script "salad" whenever a bilingual
layout is interleaved by the PDF text extractor (e.g. 'የአያLትas ስt ምna me').
A chunk whose letters are substantially non-Latin can neither be read in a
citation nor retrieved usefully by an English query, so the chunker drops
it at the funnel and verify_chunks fails if one ever ships.

The Latin cutoff is the end of Latin Extended-B: accented European letters
(é, ñ, ő) count as Latin, so ordinary names and loanwords never trip the
filter. Vietnamese diacritics (Latin Extended Additional) intentionally
count as non-Latin. Measured on the 2026-07 corpus, every chunk above a
0.02 ratio came from a translated form and every English chunk sat below
it, so the 0.10 threshold has a wide margin on both sides.
"""

MAX_NON_LATIN_RATIO = 0.10
_LATIN_MAX_CODEPOINT = 0x024F  # end of Latin Extended-B


def non_latin_ratio(text: str) -> float:
    """Fraction of alphabetic characters outside the Latin blocks."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for ch in letters if ord(ch) > _LATIN_MAX_CODEPOINT)
    return non_latin / len(letters)


def is_non_english(text: str) -> bool:
    """True when the text is dominated by non-Latin script."""
    return non_latin_ratio(text) > MAX_NON_LATIN_RATIO
