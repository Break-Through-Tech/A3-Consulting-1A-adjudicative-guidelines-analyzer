"""Re-derive the scalar fields, each from the section that legitimately holds it.

The source parser read whole documents for every field, which is why outcomes
were ambiguous, dates landed in year 9201 and the judge column captured sentence
fragments like 'Board reverses the'. Everything here is scoped to a span.

Originals are never overwritten: each cleaned value ships alongside the stored
one (`outcome_stored`, `date_stored`, `judge_raw`, `case_type_stored`) so every
correction is inspectable.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src import config

# --- outcome -----------------------------------------------------------------
# Read from the verdict span only (segment.verdict_span), never the whole
# document. Negations are listed before their positive forms because
# "not clearly consistent" contains "clearly consistent".

_OUTCOME_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("DENIED", re.compile(r"\b(?:is|are)\s+denied\b", re.I)),
    ("DENIED", re.compile(r"\bnot\s+clearly\s+consistent\b", re.I)),
    ("DENIED", re.compile(r"\bclearly\s+not\s+consistent\b", re.I)),
    ("DENIED", re.compile(r"\bunfavorable\s+(?:determination|decision)\b", re.I)),
    ("DENIED", re.compile(r"\badverse\s+decision\s+(?:is\s+)?affirmed\b", re.I)),
    ("DENIED", re.compile(r"\bfavorable\s+decision\s+(?:is\s+)?reversed\b", re.I)),
    ("REVOKED", re.compile(r"\b(?:is|are)\s+revoked\b", re.I)),
    ("REMANDED", re.compile(r"\b(?:is|are)\s+remanded\b|\bremanded\s+(?:to|for)\b", re.I)),
    ("GRANTED", re.compile(r"\b(?:is|are)\s+granted\b", re.I)),
    ("GRANTED", re.compile(r"\bclearly\s+consistent\b", re.I)),
    ("GRANTED", re.compile(r"\bfavorable\s+(?:determination|decision)\b", re.I)),
    ("GRANTED", re.compile(r"\badverse\s+decision\s+(?:is\s+)?reversed\b", re.I)),
]


# An appeal outcome is a COMPOSITION, not a phrase: the Administrative Judge's
# underlying decision combined with what the Appeal Board did to it.
#
#   "The Judge's decision denying Applicant a clearance is AFFIRMED"  -> DENIED
#   "The decision ... granting Applicant a clearance is AFFIRMED"     -> GRANTED
#   "The Judge's adverse security clearance decision is REVERSED"     -> GRANTED
#
# Flat phrase matching cannot express this, which is why 2,252 of a 4,000-row
# unresolved sample were appeals: the span says "denying ... AFFIRMED" and
# contains neither "is denied" nor "is granted".
_UNDERLYING_ADVERSE = re.compile(r"\b(?:denying|denied|adverse|unfavorable)\b", re.I)
_UNDERLYING_FAVORABLE = re.compile(r"\b(?:granting|granted|favorable)\b", re.I)
_ACTION_AFFIRM = re.compile(r"\b(?:affirmed|sustained)\b", re.I)
_ACTION_REVERSE = re.compile(r"\breversed\b", re.I)
_ACTION_REMAND = re.compile(r"\bremand(?:ed)?\b", re.I)

_COMPOSITION = {
    ("adverse", "affirm"): "DENIED",
    ("adverse", "reverse"): "GRANTED",
    ("favorable", "affirm"): "GRANTED",
    ("favorable", "reverse"): "DENIED",
}


def _compose_appeal_outcome(body: str) -> str | None:
    """Resolve underlying decision x board action. None if either is absent."""
    if _ACTION_REMAND.search(body):
        return "REMANDED"
    action = None
    if _ACTION_AFFIRM.search(body):
        action = "affirm"
    elif _ACTION_REVERSE.search(body):
        action = "reverse"
    if action is None:
        return None

    adverse = _UNDERLYING_ADVERSE.search(body)
    favorable = _UNDERLYING_FAVORABLE.search(body)
    if adverse and favorable:
        # Both words present: the one nearer the action verb is the one being
        # acted on, e.g. "reversed the favorable decision, finding adverse ...".
        underlying = "adverse" if adverse.start() > favorable.start() else "favorable"
    elif adverse:
        underlying = "adverse"
    elif favorable:
        underlying = "favorable"
    else:
        return None
    return _COMPOSITION[(underlying, action)]


# Where the Order states an action but not a direction -- "The decision of the
# Judge is AFFIRMED" -- the underlying grant/denial sits earlier in the opinion.
# These patterns look for it in the body preceding the Order.
_UNDERLYING_IN_BODY = [
    ("adverse", re.compile(r"\bjudge[^.]{0,60}?\b(?:denied|denying)\b", re.I)),
    ("favorable", re.compile(r"\bjudge[^.]{0,60}?\b(?:granted|granting)\b", re.I)),
    ("adverse", re.compile(r"\b(?:adverse|unfavorable)\s+(?:security\s+)?(?:clearance\s+)?(?:decision|determination)\b", re.I)),
    ("favorable", re.compile(r"\bfavorable\s+(?:security\s+)?(?:clearance\s+)?(?:decision|determination)\b", re.I)),
]


def _underlying_from_body(body_before: str) -> str | None:
    """Latest directional statement before the Order wins."""
    best, best_pos = None, -1
    for direction, pattern in _UNDERLYING_IN_BODY:
        for match in pattern.finditer(body_before):
            if match.start() > best_pos:
                best, best_pos = direction, match.start()
    return best


def extract_outcome(text: str, span: tuple[int, int] | None) -> tuple[object, str]:
    """Returns (outcome, source). Null rather than a guess when unresolvable."""
    fallback = False
    if not span:
        # Some appeals carry no ORDER header at all. The ruling still sits in
        # the closing paragraphs, so fall back to the tail rather than give up
        # -- but record that we did, so the row is distinguishable.
        if len(text) < 200:
            return pd.NA, "no_verdict_span"
        span, fallback = (max(0, len(text) - 1500), len(text)), True
    body = text[span[0]:span[1]]
    if not body.strip():
        return pd.NA, "empty_verdict_span"

    # Direct phrasing. Two rules, and the second one matters more than it looks:
    #
    #   1. Latest match wins -- a verdict span often restates the concern
    #      before stating the ruling.
    #   2. Ties on END position are broken by LONGER match.
    #
    # Rule 2 is what makes negation work. In "not clearly consistent", both
    # `not clearly consistent` (DENIED) and `clearly consistent` (GRANTED)
    # match and END at the same character. Comparing start positions instead
    # hands the win to the shorter, inner match -- which silently flipped
    # 2,170 rows from DENIED to GRANTED before this was caught.
    candidates = [
        (match.end(), match.end() - match.start(), outcome)
        for outcome, pattern in _OUTCOME_PATTERNS
        for match in pattern.finditer(body)
    ]
    if candidates:
        return max(candidates)[2], "verdict_span_tail" if fallback else "verdict_span"

    composed = _compose_appeal_outcome(body)
    if composed is not None:
        return composed, "appeal_composition"

    # Action stated without direction. Recover the direction from the opinion
    # body, which is where "the Judge denied Applicant a clearance" appears.
    if _ACTION_REMAND.search(body):
        return "REMANDED", "appeal_composition"
    action = "affirm" if _ACTION_AFFIRM.search(body) else ("reverse" if _ACTION_REVERSE.search(body) else None)
    if action:
        underlying = _underlying_from_body(text[: span[0]])
        if underlying:
            return _COMPOSITION[(underlying, action)], "appeal_composition_body"

    return pd.NA, "no_pattern_match"


# --- guideline citation regime ----------------------------------------------
# Pre-2006 decisions cite Directive 5220.6 (E2.An.n.n); SEAD-4 uses AG para n.
# The two are not interchangeable and tier-1a label coverage differs sharply
# between them (90.9% vs 61.7%), so the regime is recorded on every row.
_RE_SEAD4 = re.compile(r"AG\s*¶\s*\d", re.I)
_RE_DIRECTIVE = re.compile(r"E2\.A\d{1,2}\.\d", re.I)


def extract_regime(text: str) -> str:
    if _RE_SEAD4.search(text):
        return config.REGIME_SEAD4
    if _RE_DIRECTIVE.search(text):
        return config.REGIME_DIRECTIVE
    return config.REGIME_UNKNOWN


# --- case type ---------------------------------------------------------------
# 2,246 rows stored as `hearing` carry an APPEAL BOARD DECISION title (98.3%)
# and no administrative-judge title (0.6%). Structure decides, not the label.
_RE_APPEAL_TITLE = re.compile(
    r"appeal\s+board\s+(?:decision|determination|summary\s+disposition)", re.I
)
_RE_AJ_TITLE = re.compile(r"decision\s+of\s+administrative\s+judge", re.I)


def extract_case_type(text: str, stored: str) -> tuple[str, bool]:
    """Returns (case_type, corrected)."""
    head = text[:4000]
    if _RE_APPEAL_TITLE.search(head) and not _RE_AJ_TITLE.search(head):
        return "appeal", stored != "appeal"
    if _RE_AJ_TITLE.search(head):
        return "hearing", stored != "hearing"
    return stored, False


# --- date --------------------------------------------------------------------
# Runs on de-prefixed text so the browser print timestamp cannot be picked up.
# The docket year from case_number is a DIFFERENT fact -- decisions land one to
# two years after docketing and the two agree exactly only 11% of the time -- so
# it is used to sanity-check a parsed date, never to substitute for one.
_DATE_PATTERNS = [
    re.compile(r"(?:Date|Dated)\s*[:\s]\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", re.I),
    re.compile(r"(?:Date|Dated)\s*[:\s]\s*(\d{1,2}/\d{1,2}/\d{4})", re.I),
    re.compile(r"\b([A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
]


def docket_year(case_number: str) -> object:
    match = re.match(r"^(\d{2})-", str(case_number))
    if not match:
        return pd.NA
    two = int(match.group(1))
    return two + 2000 if two < 50 else two + 1900


def extract_date(text: str, case_number: str) -> tuple[object, str]:
    """Returns (timestamp, source). Implausible values become NaT, not guesses."""
    head = text[:3000]
    year = docket_year(case_number)
    for index, pattern in enumerate(_DATE_PATTERNS):
        for match in pattern.finditer(head):
            parsed = pd.to_datetime(match.group(1), errors="coerce")
            if pd.isna(parsed):
                continue
            if not (config.MIN_YEAR <= parsed.year <= config.MAX_YEAR):
                continue
            # A decision predating its own docket year, or trailing it by more
            # than a decade, means we matched some other date in the caption.
            if year is not pd.NA and not (year - 1 <= parsed.year <= year + 10):
                continue
            return parsed, "labelled" if index < 2 else "positional"
    return pd.NaT, "unresolved"


# --- judge -------------------------------------------------------------------
# Read from the signature block only. The source extractor scanned loosely and
# produced 2,772 values containing newlines, plus fragments such as
# 'Board reverses the' and 'evidence before the'.
_NAME_LINE = re.compile(
    r"^[ \t]*([A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]*){1,3})[ \t]*$",
    re.MULTILINE,
)


def _looks_like_name(candidate: str) -> bool:
    if not candidate or len(candidate) > config.JUDGE_MAX_LEN:
        return False
    if "\n" in candidate or any(ch.isdigit() for ch in candidate):
        return False
    words = [w.strip(".,").lower() for w in candidate.split()]
    if len(words) < 2:
        return False
    return not any(word in config.JUDGE_STOPWORDS for word in words)


def normalize_name(name: str) -> str:
    """Fold the typographic apostrophe so Ra'anan and Ra`anan are one person."""
    return unicodedata.normalize("NFC", str(name).replace("’", "'")).strip()


def name_key(name: str) -> tuple[str, str]:
    """(first, last) identity key used to group spelling variants for review."""
    cleaned = re.sub(r"[^A-Za-z ]", " ", normalize_name(name).lower())
    parts = [p for p in cleaned.split() if len(p) > 1]
    return (parts[0], parts[-1]) if len(parts) >= 2 else ("", "")


def extract_judge(text: str) -> tuple[object, str]:
    """Returns (judge, status)."""
    tail = text[-1800:]
    anchor = re.search(r"Administrative\s+Judge", tail, re.I)
    window = tail[: anchor.start()] if anchor else tail
    for match in reversed(list(_NAME_LINE.finditer(window))):
        cleaned = normalize_name(match.group(1))
        if _looks_like_name(cleaned):
            return cleaned, "signature_block"
    return pd.NA, "unresolved"
