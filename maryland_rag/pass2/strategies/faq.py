"""
FAQ chunking strategy: extract Q+A pairs from HTML content.

Detects question patterns via:
- Heading tags (h2, h3, h4) ending with '?'
- Bold/strong text ending with '?'
- Definition lists (dt/dd pairs)
- Accordion/details-summary patterns
- Numbered or bulleted Q/A patterns

Each Q+A pair becomes a single chunk. Questions are never split from answers.

Navigation chrome is aggressively excluded: both state and MoCo templates
build sidebar menus out of <dl>/<details>, so nav/aside/header/footer are
stripped before extraction and every dt/summary must actually look like a
question. An acceptance gate rejects extractions that only captured menu
scraps, so the chunker falls back to plain text chunking instead.
"""
import re

from bs4 import BeautifulSoup, Tag

from .semantic import MAX_CHUNK_CHARS, MAX_CHUNK_WORDS, enforce_chunk_caps

# Acceptance gate: a Q/A extraction is only trusted when it found at least
# this many pairs AND covered at least this fraction of the page's own
# main-content words. Nav-menu false positives extract a handful of tiny
# "pairs" covering a few percent of the page; genuine FAQ pages are
# dominated by their Q/A content, so 0.2 leaves generous margin for intro
# paragraphs and other non-Q/A prose.
MIN_QA_PAIRS = 2
MIN_QA_COVERAGE = 0.2


def extract_qa_pairs(url: str, raw_html: str | None = None) -> list[dict]:
    """
    Extract question-answer pairs from a FAQ page.

    Args:
        url: Page URL (used to fetch if raw_html not provided).
        raw_html: Pre-fetched HTML content (optional).

    Returns:
        List of dicts with 'question', 'answer', and 'text' keys.
        Returns [] when extraction fails the acceptance gate, so the
        caller can fall back to plain text chunking.
    """
    if not raw_html:
        from ...pass2.cache import get_html
        raw_html = get_html(url)
        if not raw_html:
            return []

    soup = BeautifulSoup(raw_html, 'html.parser')

    # Strip non-content elements before any extraction: script/style would
    # leak JS/CSS into answers, and nav/aside/header/footer hold the
    # sidebar menus that masquerade as <dl>/<details> FAQ structures.
    for junk in soup(['script', 'style', 'noscript', 'nav', 'aside', 'header', 'footer']):
        junk.decompose()

    # Page word count for the coverage gate, computed from the page's own
    # remaining (main-content) text. Must be measured before extraction:
    # _extract_from_details_summary mutates the soup.
    page_words = len(_text(soup).split())

    # Try each detection method in order of reliability
    pairs = _extract_from_definition_lists(soup)
    if not pairs:
        pairs = _extract_from_details_summary(soup)
    if not pairs:
        pairs = _extract_from_heading_patterns(soup)
    if not pairs:
        pairs = _extract_from_bold_patterns(soup)

    # Acceptance gate: reject sparse extractions (e.g. residual menu
    # scraps) so the chunker falls through to text chunking.
    if len(pairs) < MIN_QA_PAIRS:
        return []
    extracted_words = sum(
        len(p['question'].split()) + len(p['answer'].split()) for p in pairs
    )
    if extracted_words / max(page_words, 1) < MIN_QA_COVERAGE:
        return []

    return _cap_long_answers(pairs)


def _extract_from_definition_lists(soup: BeautifulSoup) -> list[dict]:
    """Extract Q+A from <dl><dt>...<dd>... structures."""
    pairs = []
    for dl in soup.find_all('dl'):
        dts = dl.find_all('dt')
        for dt in dts:
            question = _text(dt)
            # Only accept dt entries that actually read as questions:
            # both site templates also use <dl> for navigation menus.
            if not _looks_like_question(question):
                continue
            # Collect all dd siblings until next dt
            answer_parts = []
            sibling = dt.find_next_sibling()
            while sibling and sibling.name == 'dd':
                answer_parts.append(_text(sibling))
                sibling = sibling.find_next_sibling()
            if question and answer_parts:
                answer = ' '.join(answer_parts)
                pairs.append(_make_pair(question, answer))
    return pairs


def _extract_from_details_summary(soup: BeautifulSoup) -> list[dict]:
    """Extract Q+A from <details><summary>Q</summary>A</details> patterns."""
    pairs = []
    for details in soup.find_all('details'):
        summary = details.find('summary')
        if not summary:
            continue
        question = _text(summary)
        # Same nav-menu guard as definition lists: accordions are also
        # used for collapsible navigation, so require a question.
        if not _looks_like_question(question):
            continue
        # Answer is everything in details except the summary
        summary.decompose()
        answer = _text(details)
        if question and answer:
            pairs.append(_make_pair(question, answer))
    return pairs


def _extract_from_heading_patterns(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A where questions are headings (h2-h4) and answers are
    the following paragraph/content blocks until the next heading.
    """
    pairs = []
    headings = soup.find_all(['h2', 'h3', 'h4'])

    for heading in headings:
        question = _text(heading)
        # Only treat as FAQ if the heading looks like a question
        if not _looks_like_question(question):
            continue

        # Collect content until next sibling heading of same or higher level
        answer_parts = []
        sibling = heading.find_next_sibling()
        heading_level = int(heading.name[1])

        while sibling:
            if isinstance(sibling, Tag) and sibling.name in ('h1', 'h2', 'h3', 'h4'):
                sib_level = int(sibling.name[1])
                if sib_level <= heading_level:
                    break
            if isinstance(sibling, Tag):
                text = _text(sibling)
                if text:
                    answer_parts.append(text)
            sibling = sibling.find_next_sibling()

        if question and answer_parts:
            answer = ' '.join(answer_parts)
            pairs.append(_make_pair(question, answer))

    return pairs


def _extract_from_bold_patterns(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A where questions are bold/strong text ending with '?'
    followed by answer text in the same or next paragraph.
    """
    pairs = []
    strongs = soup.find_all(['strong', 'b'])

    for strong in strongs:
        question = _text(strong)
        if not _looks_like_question(question):
            continue

        # Answer: rest of parent paragraph + following paragraphs until next strong/bold
        parent = strong.parent
        if not parent:
            continue

        # Get text after the strong tag within the same parent
        answer_parts = []
        for sibling in strong.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in ('strong', 'b'):
                break
            text = _text(sibling) if isinstance(sibling, Tag) else _normalize_ws(str(sibling))
            if text:
                answer_parts.append(text)

        # Also check next paragraph siblings
        next_el = parent.find_next_sibling()
        while next_el and isinstance(next_el, Tag):
            if next_el.find(['strong', 'b']):
                break
            text = _text(next_el)
            if text:
                answer_parts.append(text)
            next_el = next_el.find_next_sibling()

        if question and answer_parts:
            answer = ' '.join(answer_parts)
            pairs.append(_make_pair(question, answer))

    return pairs


def _make_pair(question: str, answer: str) -> dict:
    return {
        'question': question,
        'answer': answer,
        'text': f"Q: {question}\nA: {answer}",
    }


def _cap_long_answers(pairs: list[dict]) -> list[dict]:
    """
    Enforce the embedding-safe chunk caps on Q/A pairs.

    A pair with a very long answer is split into multiple pairs that each
    repeat the question, so every chunk stays self-contained and under cap.
    """
    capped = []
    for pair in pairs:
        text = pair['text']
        if len(text.split()) <= MAX_CHUNK_WORDS and len(text) <= MAX_CHUNK_CHARS:
            capped.append(pair)
            continue
        for part in enforce_chunk_caps([pair['answer']]):
            capped.append(_make_pair(pair['question'], part))
    return capped


def _text(el) -> str:
    """Whitespace-normalized text: space-separated so words never glue
    across inline tags ('register online or by mail', not
    'registeronlineor by mail')."""
    return _normalize_ws(el.get_text(separator=' ', strip=True))


def _normalize_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _looks_like_question(text: str) -> bool:
    """Heuristic: does this text look like a FAQ question?"""
    if not text:
        return False
    text = text.strip()
    if text.endswith('?'):
        return True
    # Common question starters
    lower = text.lower()
    q_starters = [
        'how ', 'what ', 'when ', 'where ', 'who ', 'why ',
        'can i', 'do i', 'does ', 'is ', 'are ', 'will ',
        'am i', 'should ', 'may i', 'must i',
    ]
    return any(lower.startswith(s) for s in q_starters)
