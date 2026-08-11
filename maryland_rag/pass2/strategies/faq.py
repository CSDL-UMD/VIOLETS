"""
FAQ chunking strategy: extract Q+A pairs from HTML content.

Detects question patterns via:
- Definition lists (dt/dd pairs)
- Accordion/details-summary patterns
- Bootstrap collapse panels (panel-heading + panel-collapse)
- Heading tags (h2, h3, h4) ending with '?'
- Bold/strong text ending with '?'

Each Q+A pair becomes a single chunk. Questions are never split from answers.

Accordion panels are not always headed by a literal question. Both site
templates also use them as *topic labels* over Q/A-shaped content —
rumor_control.html files its Rumor/Fact pairs under "Ballot Drop Boxes",
election_day_questions.html files numbered voter rules under "While
Voting". Requiring a '?' on the summary dropped those pages to blind text
splitting, which severed Rumor lines from the Fact that answers them.
Label-headed panels are therefore accepted as sections, and a section
whose body carries internal Rumor:/Fact: markers is split into one chunk
per pair so a claim always travels with its refutation.

Navigation chrome is aggressively excluded: both templates build sidebar
menus out of <dl>/<details>, so nav/aside/header/footer are stripped
before extraction, and every label-headed panel must survive _is_nav_block
(enough prose, not mostly link text). An acceptance gate rejects
extractions that only captured menu scraps, so the chunker falls back to
plain text chunking instead.
"""
import re

from bs4 import BeautifulSoup, Tag

from .semantic import (
    MAX_CHUNK_CHARS, MAX_CHUNK_WORDS, enforce_chunk_caps, semantic_chunk,
)

# Acceptance gate: a Q/A extraction is only trusted when it found at least
# this many pairs AND covered at least this fraction of the page's own
# main-content words. Nav-menu false positives extract a handful of tiny
# "pairs" covering a few percent of the page; genuine FAQ pages are
# dominated by their Q/A content, so 0.2 leaves generous margin for intro
# paragraphs and other non-Q/A prose.
MIN_QA_PAIRS = 2
MIN_QA_COVERAGE = 0.2

# Guards for accepting a *label*-headed accordion panel (no '?' on the
# summary). A real content section runs to at least this many words and is
# mostly prose; a collapsible nav menu is short and almost entirely anchor
# text. Both site templates ship such menus inside <details>, so without
# these the sidebar would be extracted as "Q/A".
MIN_SECTION_WORDS = 20
MAX_LINK_RATIO = 0.6

# A FAQ page is rarely nothing but Q/A. early_voting.html carries the
# election's dates and the statutory early-voting-centre-per-county table
# as plain prose *outside* its accordions — 65% of the page. Returning
# only the pairs silently dropped all of it, so whatever prose is left
# after the accepted pairs are lifted out is chunked and appended too.
# Below this many words the residue is just headings and breadcrumbs.
MIN_RESIDUE_WORDS = 30

# Rumor/Fact debunk pairs (press_room/rumor_control.html). Splitting a
# section on these keeps each false claim glued to its correction — the
# retrieval-critical property, since users query in the rumor's wording.
_RUMOR_SPLIT_RE = re.compile(r'(?=\bRumor:)')
_RUMOR_FACT_RE = re.compile(r'^\s*Rumor:\s*(.+?)\s*\bFact:\s*(.+)$', re.S)


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

    # Page word count for the coverage gate, computed from the page's own
    # main-content text.
    page_words = len(_text(_clean_soup(raw_html)).split())

    # Try EVERY detection method and keep the best extraction that clears
    # the gate. Short-circuiting on the first method that returned anything
    # used to strand pages whose earlier method found one junk pair: the
    # chain stopped, MIN_QA_PAIRS rejected the single pair, and the real
    # structure further down the list was never tried. Each method gets a
    # fresh soup because the accordion extractors mutate the tree.
    best: list[dict] = []
    best_coverage = 0.0
    best_soup = None
    best_method = None
    for method in (
        _extract_from_definition_lists,
        _extract_from_details_summary,
        _extract_from_bootstrap_panels,
        _extract_from_heading_patterns,
        _extract_from_bold_patterns,
        _extract_from_anchor_patterns,
    ):
        soup = _clean_soup(raw_html)
        pairs = method(soup)
        if len(pairs) < MIN_QA_PAIRS:
            continue
        extracted_words = sum(
            len(p['question'].split()) + len(p['answer'].split()) for p in pairs
        )
        coverage = extracted_words / max(page_words, 1)
        if coverage < MIN_QA_COVERAGE:
            continue
        if coverage > best_coverage:
            best, best_coverage = pairs, coverage
            best_soup, best_method = soup, method

    if not best:
        return []

    chunks = _cap_long_answers(best)
    # Every extractor below deletes what it consumed, so the text left in
    # its soup is genuine non-Q/A page prose — emit it too rather than
    # dropping it. _extract_from_heading_patterns is excluded: its answer
    # walk is bounded by heading level, not by nodes it can safely detach.
    if best_method in _RESIDUE_SAFE_METHODS:
        chunks.extend(_residue_chunks(best_soup))
    return chunks


def _residue_chunks(soup: BeautifulSoup) -> list[dict]:
    """Chunk the page prose left over after the Q/A pairs were lifted out."""
    residue = _text(soup)
    if len(residue.split()) < MIN_RESIDUE_WORDS:
        return []
    return [{'text': t} for t in semantic_chunk(residue) if t.strip()]


def _clean_soup(raw_html: str) -> BeautifulSoup:
    """Parse and strip non-content elements: script/style would leak
    JS/CSS into answers, and nav/aside/header/footer hold the sidebar
    menus that masquerade as <dl>/<details> FAQ structures."""
    soup = BeautifulSoup(raw_html, 'html.parser')
    for junk in soup(['script', 'style', 'noscript', 'nav', 'aside', 'header', 'footer']):
        junk.decompose()
    return soup


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
            consumed = []
            sibling = dt.find_next_sibling()
            while sibling and sibling.name == 'dd':
                answer_parts.append(_text(sibling))
                consumed.append(sibling)
                sibling = sibling.find_next_sibling()
            if question and answer_parts:
                answer = ' '.join(answer_parts)
                pairs.append(_make_pair(question, answer))
                dt.decompose()
                for el in consumed:
                    el.decompose()
    return pairs


def _extract_from_details_summary(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A from <details><summary>...</summary>...</details> panels.

    A summary that reads as a question yields a plain Q/A pair. A summary
    that is a topic *label* ("Ballot Drop Boxes", "While Voting") yields a
    labelled section instead — split on internal Rumor:/Fact: markers when
    present. Nav menus built out of <details> are rejected by
    _is_nav_block rather than by demanding a '?'.
    """
    pairs = []
    for details in soup.find_all('details'):
        # Only outermost panels: a nested <details> would otherwise have
        # its text counted twice, once on its own and once inside its
        # parent's body, inflating the coverage gate.
        if details.find_parent('details') is not None:
            continue
        summary = details.find('summary')
        if not summary:
            continue
        label = _text(summary)
        if not label:
            continue
        # extract() (not decompose()) so the summary is detached but the
        # remaining subtree stays intact for the nav/link measurements.
        summary.extract()
        body = _text(details)
        if not body:
            continue

        if _looks_like_question(label):
            pairs.append(_make_pair(label, body))
        elif _is_nav_block(details, body):
            # A collapsible menu: drop it so it cannot resurface as
            # residue prose.
            details.decompose()
            continue
        else:
            pairs.extend(_split_labelled_section(label, body))
        details.decompose()
    return pairs


def _extract_from_bootstrap_panels(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A from Bootstrap collapse accordions:

        <div class="panel">
          <div class="panel-heading"><a class="collapsed">Question?</a></div>
          <div class="panel-collapse"><div class="panel-body">Answer</div></div>
        </div>

    The MoCo FAQ pages are built entirely this way — no <details>, no
    <dl>, and questions in <a>, not <strong> — so none of the other
    extractors could see them.
    """
    pairs = []
    for heading in soup.find_all(class_=re.compile(r'\bpanel-heading\b')):
        question = _text(heading)
        if not question:
            continue
        body = next(
            (
                sib for sib in heading.find_next_siblings()
                if isinstance(sib, Tag)
                and re.search(r'panel-collapse|panel-body|\bcollapse\b',
                              ' '.join(sib.get('class') or []))
            ),
            None,
        )
        if body is None:
            continue
        answer = _text(body)
        if not answer:
            continue

        if _looks_like_question(question):
            pairs.append(_make_pair(question, answer))
        elif _is_nav_block(body, answer):
            continue
        else:
            pairs.extend(_split_labelled_section(question, answer))
        heading.decompose()
        body.decompose()
    return pairs


def _extract_from_anchor_patterns(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Q+A from flat markup where each question is a bare <a> and the
    answer is the loose text that follows it, up to the next question:

        <a href="...">What is an Election Worker?</a><br/><br/>
        An Election Worker is a registered Maryland voter who ...<br/><br/>
        <a href="...">What are the hours?</a><br/><br/>
        ...

    There is no wrapper element per pair, so the walk is over the parent's
    flat child sequence rather than over a container. Only anchors that
    read as questions start a pair, which keeps ordinary inline links
    inside an answer from splitting it.
    """
    pairs = []
    anchors = [a for a in soup.find_all('a') if _looks_like_question(_text(a))]
    question_ids = {id(a) for a in anchors}
    consumed = []

    for anchor in anchors:
        question = _text(anchor)
        answer_parts = []
        taken = [anchor]
        for node in anchor.next_siblings:
            if isinstance(node, Tag):
                # Stop at the next question anchor, or at a block that
                # contains one (the following pair's wrapper).
                if id(node) in question_ids:
                    break
                if any(id(a) in question_ids for a in node.find_all('a')):
                    break
                text = _text(node)
            else:
                text = _normalize_ws(str(node))
            taken.append(node)
            if text:
                answer_parts.append(text)

        answer = ' '.join(answer_parts).strip()
        if question and len(answer.split()) >= MIN_SECTION_WORDS:
            pairs.append(_make_pair(question, answer))
            consumed.extend(taken)

    _drop_all(consumed)
    return pairs


def _split_labelled_section(label: str, body: str) -> list[dict]:
    """
    Turn one label-headed panel into chunks.

    If the body carries Rumor:/Fact: debunk pairs, emit one chunk per
    pair so the false claim and its correction can never be separated by
    a chunk boundary. Otherwise the panel stays whole, tagged with its
    section label for context.
    """
    units = [u.strip() for u in _RUMOR_SPLIT_RE.split(body) if u.strip()]
    rumor_units = [u for u in units if _RUMOR_FACT_RE.match(u)]
    if len(rumor_units) < 2:
        return [_make_pair(label, body, q_tag=None)]

    pairs = []
    for unit in rumor_units:
        match = _RUMOR_FACT_RE.match(unit)
        rumor, fact = match.group(1).strip(), match.group(2).strip()
        pairs.append(
            _make_pair(rumor, fact, section=label, q_tag='Rumor', a_tag='Fact')
        )
    return pairs


def _is_nav_block(el: Tag, body: str) -> bool:
    """
    Is this panel a collapsible navigation menu rather than content?

    Menus are short and made almost entirely of anchor text; real
    sections run to real prose. Applied only to label-headed panels —
    a panel whose summary is a literal question is trusted on that alone.
    """
    words = len(body.split())
    if words < MIN_SECTION_WORDS:
        return True
    link_words = sum(len(_text(a).split()) for a in el.find_all('a'))
    return link_words / max(words, 1) > MAX_LINK_RATIO


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
    consumed = []

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
        taken = [strong]
        for sibling in strong.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in ('strong', 'b'):
                break
            text = _text(sibling) if isinstance(sibling, Tag) else _normalize_ws(str(sibling))
            taken.append(sibling)
            if text:
                answer_parts.append(text)

        # Also check next paragraph siblings
        next_el = parent.find_next_sibling()
        while next_el and isinstance(next_el, Tag):
            if next_el.find(['strong', 'b']):
                break
            text = _text(next_el)
            taken.append(next_el)
            if text:
                answer_parts.append(text)
            next_el = next_el.find_next_sibling()

        if question and answer_parts:
            answer = ' '.join(answer_parts)
            pairs.append(_make_pair(question, answer))
            consumed.extend(taken)

    _drop_all(consumed)
    return pairs


def _drop_all(nodes: list) -> None:
    """
    Detach every node an extractor consumed, so the text left in the soup
    is exactly the page's non-Q/A prose.

    Deferred to the end of extraction rather than done inline: the flat
    extractors walk sibling chains that overlap between pairs, and
    mutating mid-walk would cut the traversal short. Nodes already
    detached (as a descendant of an earlier one) are skipped.
    """
    for node in nodes:
        try:
            node.extract()
        except (AttributeError, ValueError):
            continue


def _make_pair(
    question: str,
    answer: str,
    *,
    section: str | None = None,
    q_tag: str | None = 'Q',
    a_tag: str = 'A',
) -> dict:
    """
    Build a Q/A chunk record.

    q_tag/a_tag name the two halves so a debunk pair renders as
    'Rumor:'/'Fact:' rather than being flattened to 'Q:'/'A:', which would
    strip the signal that the claim is false. q_tag=None renders a
    label-headed section as its heading followed by its body. section
    prefixes the retrieval context the accordion label carries.
    """
    return {
        'question': question,
        'answer': answer,
        'section': section,
        'q_tag': q_tag,
        'a_tag': a_tag,
        'text': _render(question, answer, section, q_tag, a_tag),
    }


def _render(
    question: str, answer: str, section: str | None, q_tag: str | None, a_tag: str
) -> str:
    head = f"[{section}]\n" if section else ""
    if q_tag is None:
        return f"{head}{question}\n{answer}".strip()
    return f"{head}{q_tag}: {question}\n{a_tag}: {answer}"


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
            capped.append(_make_pair(
                pair['question'], part,
                section=pair.get('section'),
                q_tag=pair.get('q_tag', 'Q'),
                a_tag=pair.get('a_tag', 'A'),
            ))
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


# Extractors that delete each accepted pair from the soup, so the text
# remaining afterwards is genuine non-Q/A prose rather than a duplicate of
# what they already emitted. Only these are safe to harvest residue from.
_RESIDUE_SAFE_METHODS = frozenset({
    _extract_from_definition_lists,
    _extract_from_details_summary,
    _extract_from_bootstrap_panels,
    _extract_from_bold_patterns,
    _extract_from_anchor_patterns,
})
