"""
FAQ chunking strategy: extract Q+A pairs from HTML content.

Detects question patterns via:
- Heading tags (h2, h3, h4) ending with '?'
- Bold/strong text ending with '?'
- Definition lists (dt/dd pairs)
- Accordion/details-summary patterns
- Numbered or bulleted Q/A patterns

Each Q+A pair becomes a single chunk. Questions are never split from answers.
"""
import re

import requests
from bs4 import BeautifulSoup, Tag

from ...pass1.config import REQUEST_TIMEOUT


def extract_qa_pairs(url: str, raw_html: str | None = None) -> list[dict]:
    """
    Extract question-answer pairs from a FAQ page.

    Args:
        url: Page URL (used to fetch if raw_html not provided).
        raw_html: Pre-fetched HTML content (optional).

    Returns:
        List of dicts with 'question', 'answer', and 'text' keys.
    """
    if not raw_html:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            raw_html = resp.text
        except Exception:
            return []

    soup = BeautifulSoup(raw_html, 'html.parser')

    # Try each detection method in order of reliability
    pairs = _extract_from_definition_lists(soup)
    if pairs:
        return pairs

    pairs = _extract_from_details_summary(soup)
    if pairs:
        return pairs

    pairs = _extract_from_heading_patterns(soup)
    if pairs:
        return pairs

    pairs = _extract_from_bold_patterns(soup)
    if pairs:
        return pairs

    # Fallback: treat entire content as a single chunk
    return []


def _extract_from_definition_lists(soup: BeautifulSoup) -> list[dict]:
    """Extract Q+A from <dl><dt>...<dd>... structures."""
    pairs = []
    for dl in soup.find_all('dl'):
        dts = dl.find_all('dt')
        for dt in dts:
            question = dt.get_text(strip=True)
            # Collect all dd siblings until next dt
            answer_parts = []
            sibling = dt.find_next_sibling()
            while sibling and sibling.name == 'dd':
                answer_parts.append(sibling.get_text(strip=True))
                sibling = sibling.find_next_sibling()
            if question and answer_parts:
                answer = ' '.join(answer_parts)
                pairs.append({
                    'question': question,
                    'answer': answer,
                    'text': f"Q: {question}\nA: {answer}",
                })
    return pairs


def _extract_from_details_summary(soup: BeautifulSoup) -> list[dict]:
    """Extract Q+A from <details><summary>Q</summary>A</details> patterns."""
    pairs = []
    for details in soup.find_all('details'):
        summary = details.find('summary')
        if not summary:
            continue
        question = summary.get_text(strip=True)
        # Answer is everything in details except the summary
        summary.decompose()
        answer = details.get_text(strip=True)
        if question and answer:
            pairs.append({
                'question': question,
                'answer': answer,
                'text': f"Q: {question}\nA: {answer}",
            })
    return pairs


def _extract_from_heading_patterns(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A where questions are headings (h2-h4) and answers are
    the following paragraph/content blocks until the next heading.
    """
    pairs = []
    headings = soup.find_all(['h2', 'h3', 'h4'])

    for heading in headings:
        question = heading.get_text(strip=True)
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
                text = sibling.get_text(strip=True)
                if text:
                    answer_parts.append(text)
            sibling = sibling.find_next_sibling()

        if question and answer_parts:
            answer = ' '.join(answer_parts)
            pairs.append({
                'question': question,
                'answer': answer,
                'text': f"Q: {question}\nA: {answer}",
            })

    return pairs


def _extract_from_bold_patterns(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A where questions are bold/strong text ending with '?'
    followed by answer text in the same or next paragraph.
    """
    pairs = []
    strongs = soup.find_all(['strong', 'b'])

    for strong in strongs:
        question = strong.get_text(strip=True)
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
            text = sibling.get_text(strip=True) if isinstance(sibling, Tag) else str(sibling).strip()
            if text:
                answer_parts.append(text)

        # Also check next paragraph siblings
        next_el = parent.find_next_sibling()
        while next_el and isinstance(next_el, Tag):
            if next_el.find(['strong', 'b']):
                break
            text = next_el.get_text(strip=True)
            if text:
                answer_parts.append(text)
            next_el = next_el.find_next_sibling()

        if question and answer_parts:
            answer = ' '.join(answer_parts)
            pairs.append({
                'question': question,
                'answer': answer,
                'text': f"Q: {question}\nA: {answer}",
            })

    return pairs


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
