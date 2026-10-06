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
    "election law",
    "security",
    "integrity",
    "certification",
    "accessibility",
    "provisional",
    "poll worker",
    "absentee",
    "mail-in",
    # Added from manual triage of REVIEW files (2026-06-24):
    "voter guide",        # MoCo Voter Guide for the 2026 Primary
    "canvass",            # canvass overview / instructions / precanvass / scenario chart
    "final regulations",  # Final Regulations COMAR (not the superseded "Proposed Regs")
    "usability",          # ExpressVote 3 Usability Report
    "testing report",     # ES&S EVS 6.5.0.0 Testing Report
    "election judge",     # Election Judge Parity Reporting policy
    # Mail-in-ballot legislative Q&A batch (2026-06/25), kept per request:
    "question and answer",  # Delegate/Senator Q&A files (Fisher, Korman, Long, Miller, Szeliga, Hester)
    "delegate",             # delegate Q&As + Mail Message to Delegate Mangione
    "caucus",               # House Republican Caucus Question/Response/Attachment
    "lwv",                  # Letter From/To LWV (League of Women Voters)
    "del long",             # Del Long response pt 2
    "house_admin",          # House_Admin_Response_06122026
    # Added from manual triage of REVIEW files (2026-07-29):
    "polling place",        # official polling place change notices (Cecil GP26, notification letters)
    "request policy",       # Policy 2025-03 PIA/PSR voter-data request policy (official SBE policy)
    "lvw",                  # "LVW Questions Reply" — typo'd LWV; SBE's official answers on ballot curing
    # Content-verified additions (2026-07-29, opened every REVIEW/EXCLUDE file):
    "ppp",                  # "2026 PPP Baltimore city" — full Baltimore City 2026 Polling Place Plan
    "baltimorecity",        # "BaltimoreCity.pdf" — official Waxter Center polling place change notice (image-only, needs OCR)
    "potomac heights",      # Washington Co. precincts 18002/18004 polling place move (filename says "Letter")
    "doj_response",         # SBE's official NVRA/HAVA list-maintenance compliance response to DOJ
    # Manual triage of REVIEW files (2026-10-05):
    "drop box",             # county drop-box location changes (Carroll GG26)
    "prediction markets",   # Policy 2026-02: staff barred from election prediction markets
    "board by-laws",        # signed SBE bylaws (Feb 2025, adds §4.7) — supersedes the site's 2024 PDF.
                            # Hyphenated on purpose: "bylaws" stays in EXCLUDE_TERMS for the Howard
                            # County board's bylaws-amendment correspondence.
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
    "budget",
    "financial",
    "contract",
    "bylaws",
    "transmittal",
    "appointment",
    # Added from manual triage of REVIEW files (2026-07-29):
    "letter",          # advocacy/vendor/meeting correspondence (MAEO, ES&S, Potomac Heights)
    "presentation",    # public testimony presentations to the Board
    "testimony",
    "comment",         # public comments (subsumes "comments" above)
    "proposed regs",   # superseded by Final Regulations (kept via INCLUDE)
    "ccf",             # campaign-finance administrivia (CCF regs / closure requests)
    "cost estimate",
    "dashboard",       # NVS demo dashboard screenshots
    "moco request",    # MoCo language-assistance request correspondence
    "mdgop",           # partisan-org letters
    "fatal flaws",     # third-party RLA-audit critique (advocacy)
    "thompsom",        # one-off individual submission
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
