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

Two guards on top of the ratio test:

  - Absolute floor: a short English chunk that merely NAMES other languages
    ("Interpretation: 中文 | Korean: 한국어") can exceed the ratio with only a
    handful of non-Latin letters. The ratio only trips when at least
    MIN_NON_LATIN_LETTERS non-Latin letters are present.
  - Spanish detector: Spanish is pure Latin script, so leaked -ES press
    releases sail through the ratio test. is_spanish() catches them via
    function-word frequency (see below), tuned for zero false positives on
    English chunks.
"""
import re

MAX_NON_LATIN_RATIO = 0.10
MIN_NON_LATIN_LETTERS = 15  # absolute floor: below this, never drop on ratio
_LATIN_MAX_CODEPOINT = 0x024F  # end of Latin Extended-B

# --- Spanish (Latin-script) leakage detection ------------------------------
# Conservative frequency test: a genuinely Spanish chunk is saturated with
# Spanish function words and nearly empty of English ones. All three
# thresholds must hold, so short chunks and bilingual boilerplate that mixes
# both languages are never dropped. Priority is ZERO false positives on
# English chunks; false negatives just mean a Spanish chunk survives to be
# flagged by verify_chunks.
SPANISH_MIN_WORDS = 30      # don't judge short chunks at all
SPANISH_MIN_RATIO = 0.20    # >= 20% of words are Spanish function words
SPANISH_MAX_ENGLISH_RATIO = 0.05  # < 5% of words are English function words

# Measured on the 2026-07 corpus: every pure-Spanish chunk (the leaked -ES
# press releases + Pocket Guide) scores 0.24-0.46 on the Spanish ratio with
# ~0.00 English; every English chunk scores <= 0.03, and the worst
# bilingual English/Spanish chunk (which must be KEPT — it carries the
# English text) scores 0.16. The 0.20 threshold splits those bands with
# margin on both sides.

_SPANISH_FUNCTION_WORDS = frozenset({
    'el', 'la', 'los', 'las', 'de', 'del', 'para', 'por', 'con', 'una',
    'uno', 'que', 'es', 'en', 'su', 'se', 'como', 'más', 'este', 'esta',
    'votación', 'elección', 'boleta', 'votante', 'condado', 'estado',
    'fecha', 'correo', 'electoral',
})

_ENGLISH_FUNCTION_WORDS = frozenset({
    'the', 'of', 'to', 'and', 'in', 'is', 'for', 'you', 'your', 'that',
    'are', 'be', 'will', 'on', 'with', 'must', 'may', 'by', 'or', 'if',
})

_WORD_RE = re.compile(r"[a-záéíóúüñ]+")


def non_latin_ratio(text: str) -> float:
    """Fraction of alphabetic characters outside the Latin blocks."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for ch in letters if ord(ch) > _LATIN_MAX_CODEPOINT)
    return non_latin / len(letters)


def is_spanish(text: str) -> bool:
    """True when the text reads as Spanish by function-word frequency."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < SPANISH_MIN_WORDS:
        return False
    spanish = sum(1 for w in words if w in _SPANISH_FUNCTION_WORDS)
    english = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    return (spanish / len(words) >= SPANISH_MIN_RATIO
            and english / len(words) < SPANISH_MAX_ENGLISH_RATIO)


def is_non_english(text: str) -> bool:
    """True when the text is dominated by non-Latin script or is Spanish."""
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        non_latin = sum(1 for ch in letters if ord(ch) > _LATIN_MAX_CODEPOINT)
        if (non_latin >= MIN_NON_LATIN_LETTERS
                and non_latin / len(letters) > MAX_NON_LATIN_RATIO):
            return True
    return is_spanish(text)
