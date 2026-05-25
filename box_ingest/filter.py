"""
Include/exclude rules for State Board of Elections Box materials.

Rules (case-insensitive, applied to filename stem):
  INCLUDE  — filename contains any INCLUDE_TERMS keyword
  EXCLUDE  — filename contains any EXCLUDE_TERMS keyword (checked second)
  REVIEW   — neither matched; needs manual triage

INCLUDE takes priority: a file matching both include and exclude terms is
included (e.g. "State Administrator's Election Plan Report" would be kept).
"""
from __future__ import annotations

INCLUDE_TERMS = [
    "aag",
    "assistant attorney general",
    "state administrator",
    "administrator's report",
]

EXCLUDE_TERMS = [
    "agenda",
    "minutes",
    "memo",
    "waiver",
    "comments",
    "title",
    "memorandum",
    "election plan",
]

FILTER_INCLUDE = "include"
FILTER_EXCLUDE = "exclude"
FILTER_REVIEW  = "review"


def classify_filename(filename: str) -> str:
    """
    Returns FILTER_INCLUDE, FILTER_EXCLUDE, or FILTER_REVIEW.
    Matching is case-insensitive against the full filename (including extension).
    """
    lower = filename.lower()

    for term in INCLUDE_TERMS:
        if term in lower:
            return FILTER_INCLUDE

    for term in EXCLUDE_TERMS:
        if term in lower:
            return FILTER_EXCLUDE

    return FILTER_REVIEW
