"""Guideline labels, in three tiers of descending trustworthiness.

The stored `guidelines` column matches what the judge actually ruled only 6.7%
of the time. Its two defects, both in one line of the source parser:

    'A': r'Guideline\\s*A|Allegiance|AG\\s*\\u00b6\\s*2'

  * `AG para 2` has no digit boundary, so it also matches paras 20/22/24-28 --
    which belong to Guidelines F, G, H and I. Worse, para 2 is the *general
    adjudicative policy* paragraph present in nearly every decision. Result:
    Guideline A flagged on 20,662 cases where the true count is 3.
  * `Guideline C` has no word boundary, so it matches inside "Guideline can".

Rather than patch that, we read what the judge wrote. Tier 1a finds the formal
ruling; tier 1b inherits it for appeals; tier 2 is a flagged last resort.
"""
from __future__ import annotations

import re

from src import config

# --- tier 1a: the formal ruling ---------------------------------------------
# "Paragraph 1, Guideline B: AGAINST APPLICANT" is the syntax of a judge's
# formal finding and appears nowhere else, so it needs no section detection.
#
# NOTE ON VALIDATION: measuring this against the stored `formal_findings`
# column would be circular -- that column was produced by the same regex in
# scraper.py:1230. Against an independent signal (guideline sub-headings in the
# ANALYSIS section) recall is 98.0%; precision is pending hand-verification of a
# 50-row sample. Do not quote a precision figure until that sample exists.
_FORMAL_RULING = re.compile(
    r"Paragraph\s+(\d+)[.,]?\s*\(?Guideline\s+([A-M])\b",
    re.IGNORECASE,
)
_FORMAL_ALT = re.compile(
    r"GUIDELINE\s+([A-M])\s*\([^)]{3,60}\)\s*:\s*(FOR|AGAINST)\s*APPLICANT",
    re.IGNORECASE,
)
# The verdict routinely sits on its own line below the heading:
#     Paragraph 1, Guideline B:
#
#     AGAINST APPLICANT
# so it is read from a forward window rather than the same line. The window
# stops at the next ruling so one guideline cannot borrow another's verdict.
_VERDICT_AFTER = re.compile(r"\b(FOR|AGAINST)\s+APPLICANT", re.IGNORECASE)

# Subparagraph references look like "1.a", "1.a-1.c", "2.a through 2.d". The
# leading digit-dot-letter shape is required: without it this pattern happily
# matched "Paragraph 1, Guideline B" and recorded the guideline heading as a
# subparagraph.
_SUBPARAGRAPH = re.compile(
    r"^[ \t]*(?:Sub)?paragraphs?\s+(\d+\.[a-z][^\n:]{0,28}?)\s*:?\s*\n?\s*"
    r"(For|Against)\s+(?:the\s+)?Applicant",
    re.IGNORECASE | re.MULTILINE,
)


def extract_formal_rulings(text: str) -> tuple[list[str], dict[str, str], list[dict]]:
    """Returns (codes, {code: FOR|AGAINST}, subparagraph findings with offsets)."""
    findings: dict[str, str] = {}
    order: list[str] = []

    anchors = list(_FORMAL_RULING.finditer(text))
    for index, match in enumerate(anchors):
        code = match.group(2).upper()
        window_end = anchors[index + 1].start() if index + 1 < len(anchors) else match.end() + 200
        verdict_match = _VERDICT_AFTER.search(text, match.end(), min(window_end, match.end() + 200))
        verdict = verdict_match.group(1).upper() if verdict_match else None
        if code not in findings:
            order.append(code)
            findings[code] = verdict
        elif verdict and not findings[code]:
            findings[code] = verdict

    for match in _FORMAL_ALT.finditer(text):
        code = match.group(1).upper()
        if code not in findings:
            order.append(code)
        findings[code] = match.group(2).upper()

    subparagraphs = [
        {
            "para": " ".join(match.group(1).split()),
            "finding": match.group(2).title(),
            "offset": match.start(),
        }
        for match in _SUBPARAGRAPH.finditer(text)
    ]
    return order, findings, subparagraphs


# --- tier 1b: appeal inheritance ---------------------------------------------
# Appeals carry no formal findings (the tier-1a pattern fires on 0.1% of them),
# but 95.4% cite an ISCR case number in their opening and 95.8% of those cite
# their OWN -- confirming the citation identifies the same underlying case. An
# appeal reviews the guidelines the hearing judge ruled on, so inheriting them
# is sound. It is still inference rather than observation, hence its own
# label_source value.
_ISCR_CITE = re.compile(r"ISCR\s+Case\s+No\.?\s*:?\s*(\d{2}-\d{4,6})", re.IGNORECASE)


def cited_case_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(_ISCR_CITE.findall(text[:4000])))


# --- tier 2: regex fallback --------------------------------------------------
# Even corrected this tops out at ~60% precision with ~1 spurious label per
# case, so it is a candidate generator, never a label. Recall is 100% -- the
# match set is always a superset of the truth -- which is what makes it usable
# for narrowing later.
#
# Every pattern carries \b word boundaries and (?!\d) after paragraph numbers.
# Both were missing from the source and both caused real damage.
_FALLBACK = {
    "A": r"\bGuideline\s+A\b|\bAllegiance\b|AG\s*¶\s*2(?!\d)",
    "B": r"\bGuideline\s+B\b|\bForeign\s+Influence\b|AG\s*¶\s*[67](?!\d)",
    "C": r"\bGuideline\s+C\b|\bForeign\s+Preference\b|AG\s*¶\s*(?:9|10)(?!\d)",
    "D": r"\bGuideline\s+D\b|\bSexual\s+Behavior\b|AG\s*¶\s*1[23](?!\d)",
    "E": r"\bGuideline\s+E\b|\bPersonal\s+Conduct\b|AG\s*¶\s*1[56](?!\d)",
    "F": r"\bGuideline\s+F\b|\bFinancial\s+Considerations\b|AG\s*¶\s*(?:18|19|20)(?!\d)",
    "G": r"\bGuideline\s+G\b|\bAlcohol\s+Consumption\b|AG\s*¶\s*2[12](?!\d)",
    "H": r"\bGuideline\s+H\b|\bDrug\s+Involvement\b|AG\s*¶\s*2[456](?!\d)",
    "I": r"\bGuideline\s+I\b|\bPsychological\s+Conditions\b|AG\s*¶\s*2[78](?!\d)",
    "J": r"\bGuideline\s+J\b|\bCriminal\s+Conduct\b|AG\s*¶\s*3[012](?!\d)",
    "K": r"\bGuideline\s+K\b|\bHandling\s+Protected\s+Information\b|AG\s*¶\s*3[34](?!\d)",
    "L": r"\bGuideline\s+L\b|\bOutside\s+Activities\b|AG\s*¶\s*3[67](?!\d)",
    "M": r"\bGuideline\s+M\b|\bUse\s+of\s+Information\s+Technology\b|AG\s*¶\s*(?:39|40)(?!\d)",
}
_FALLBACK_COMPILED = {code: re.compile(p, re.IGNORECASE) for code, p in _FALLBACK.items()}


def fallback_labels(text: str, policies_span: tuple[int, int] | None) -> list[str]:
    """Regex match with the POLICIES boilerplate excised.

    POLICIES recites the guidelines generically in 95.9% of documents; leaving
    it in is worth ~5 points of precision on its own.
    """
    if policies_span:
        text = text[: policies_span[0]] + text[policies_span[1]:]
    return [code for code in config.GUIDELINE_CODES if _FALLBACK_COMPILED[code].search(text)]


# --- orchestration -----------------------------------------------------------

SOURCE_FORMAL = "formal_ruling"
SOURCE_INHERITED = "inherited_from_hearing"
SOURCE_FALLBACK = "regex_fallback"
SOURCE_NONE = "none"


def label_row(text: str, policies_span: tuple[int, int] | None):
    """Tier 1a then tier 2. Tier 1b is applied afterwards, across rows."""
    codes, findings, subparagraphs = extract_formal_rulings(text)
    if codes:
        ordered = [c for c in config.GUIDELINE_CODES if c in codes]
        return ordered, findings, subparagraphs, SOURCE_FORMAL
    fallback = fallback_labels(text, policies_span)
    if fallback:
        return fallback, {}, [], SOURCE_FALLBACK
    return [], {}, [], SOURCE_NONE
