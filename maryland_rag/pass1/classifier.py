"""
Crawl-time content classification. Wraps the shared rules engine in
maryland_rag.pass1.rules with a document-extension fast path, since the
crawler sees content types (pdf/docx/xls/...) that the post-hoc
reclassifier doesn't.
"""
from .rules import classify_html

_DOCUMENT_TYPES = {'pdf', 'docx', 'doc', 'xls', 'xlsx', 'csv'}


def classify_page(result: dict) -> dict:
    """
    Classify an extracted page and assign a chunking strategy.

    Returns dict with keys:
        content_type, page_classification, chunking_strategy, classification_confidence
    """
    ctype = result.get('content_type', 'html')

    if ctype in _DOCUMENT_TYPES:
        return {
            'content_type': ctype,
            'page_classification': 'document',
            'chunking_strategy': 'document_extraction',
            'classification_confidence': 'high',
        }

    page_class, strategy, confidence = classify_html(
        url=result.get('url') or '',
        title=result.get('title') or '',
        text=result.get('text') or '',
        word_count=result.get('word_count') or 0,
        raw_html=result.get('raw_html') or '',
    )

    return {
        'content_type': 'html',
        'page_classification': page_class,
        'chunking_strategy': strategy,
        'classification_confidence': confidence,
    }
