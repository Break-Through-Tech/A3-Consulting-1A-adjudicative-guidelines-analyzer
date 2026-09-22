"""Locate document sections and return character offsets.

This is the backbone: almost every other repair depends on knowing which region
of the document a field legitimately lives in. The source parser's central
mistake was scanning whole documents for everything, which is why it read the
POLICIES boilerplate as guideline labels and the SYNOPSIS as a verdict.

Measured structure across 2,500 hearings (position as a fraction of length):

    0.03  SYNOPSIS                leaks the outcome
    0.05  STATEMENT OF THE CASE   feature
    0.13  FINDINGS OF FACT        feature
    0.34  POLICIES                boilerplate reciting every guideline
    0.80  CONCLUSIONS             reasoning, partially leaks
    0.95  FORMAL FINDINGS         authoritative per-guideline labels
    0.98  DECISION                authoritative outcome

Header casing varies (`Formal Findings` 66.5% / `FORMAL FINDINGS` 24.8%), so
matching is case-insensitive and anchored to a line of its own. Where a name
occurs more than once the LAST occurrence wins: the synopsis and table of
contents name-drop later sections before they actually begin.
"""
from __future__ import annotations

import difflib
import re

from src import config

# A header sits alone on its line, optionally followed by a colon. Requiring
# end-of-line is what stops "Guideline C" matching inside "Guideline can".
_HEADER_TEMPLATE = r"^[ \t]*{name}[ \t]*:?[ \t]*(?=\n|\Z)"

_ALL_SECTIONS = list(dict.fromkeys(config.HEARING_SECTIONS + config.APPEAL_SECTIONS))

_PATTERNS = {
    name: re.compile(
        _HEADER_TEMPLATE.format(name=config.SECTION_ALIASES.get(name, re.escape(name))),
        re.MULTILINE | re.IGNORECASE,
    )
    for name in _ALL_SECTIONS
}

_FUZZY_THRESHOLD = 0.88
_CANDIDATE_LINE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z' ]{4,40})[ \t]*:?[ \t]*$", re.MULTILINE)


def column_name(section: str) -> str:
    """'FORMAL FINDINGS' -> 'sec_formal_findings'"""
    slug = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")
    return f"sec_{slug}"


def _exact_hits(text: str) -> dict[str, int]:
    """Last exact header position per section."""
    hits = {}
    for name, pattern in _PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            hits[name] = matches[-1].start()
    return hits


def _fuzzy_hits(text: str, missing: list[str]) -> dict[str, int]:
    """Levenshtein-ish rescue for headers damaged by PDF extraction.

    Catches things like `FORMAL FINDlNGS` (lowercase L for I) that exact
    matching drops. Only consulted for sections the exact pass missed.
    """
    if not missing:
        return {}
    found = {}
    targets = {name: name.lower() for name in missing}
    for match in _CANDIDATE_LINE.finditer(text):
        candidate = " ".join(match.group(1).split()).lower()
        for name, target in targets.items():
            if name in found:
                continue
            if abs(len(candidate) - len(target)) > 3:
                continue
            if difflib.SequenceMatcher(None, candidate, target).ratio() >= _FUZZY_THRESHOLD:
                found[name] = match.start()
    return found


def segment(text: str) -> tuple[dict[str, tuple[int, int]], bool, bool]:
    """Map section name -> (start, end) offsets into `text`.

    Returns (spans, used_fuzzy, partial). A section absent from `spans` simply
    was not found; callers store None rather than guessing. Never raises.
    """
    if not text:
        return {}, False, True

    hits = _exact_hits(text)
    missing = [s for s in _ALL_SECTIONS if s not in hits]
    fuzzy = _fuzzy_hits(text, missing)
    used_fuzzy = bool(fuzzy)
    hits.update(fuzzy)

    if not hits:
        return {}, used_fuzzy, True

    # Each section runs from the end of its own header line to the start of
    # whichever section header comes next by position -- not by canonical
    # order, so documents that omit sections still segment correctly.
    ordered = sorted(hits.items(), key=lambda kv: kv[1])
    spans: dict[str, tuple[int, int]] = {}
    for index, (name, start) in enumerate(ordered):
        line_end = text.find("\n", start)
        body_start = line_end + 1 if line_end != -1 else len(text)
        body_end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        if body_end > body_start:
            spans[name] = (body_start, body_end)

    # "Partial" means we could not locate the sections that actually matter:
    # the feature region and the two authoritative label regions.
    critical = set(config.FEATURE_SECTIONS) | {"FORMAL FINDINGS", "DECISION", "ORDER"}
    partial = not (critical & set(spans))
    return spans, used_fuzzy, partial


def shift(spans: dict[str, tuple[int, int]], offset: int) -> dict[str, tuple[int, int]]:
    """Re-base spans computed on de-prefixed text back onto original full_text."""
    if not offset:
        return spans
    return {name: (s + offset, e + offset) for name, (s, e) in spans.items()}


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def paragraph_offsets(text: str, span: tuple[int, int] | None) -> list[tuple[int, int]]:
    """Paragraph (start, end) offsets inside a span, for later chunking."""
    if not span:
        return []
    start, end = span
    body = text[start:end]
    out, cursor = [], 0
    for match in _PARAGRAPH_BREAK.finditer(body):
        if match.start() > cursor:
            out.append((start + cursor, start + match.start()))
        cursor = match.end()
    if cursor < len(body):
        out.append((start + cursor, end))
    return [(s, e) for s, e in out if e - s > 40]


# --- verdict region ----------------------------------------------------------
# Judges do not agree on a heading for the section that states the outcome.
# In 11-00068 the verdict sits under "Conclusion" while "Decision" appears at
# 0.015 as a caption label in the title block -- so selecting by name alone
# picks the wrong region. Measured medians: CONCLUSIONS 0.98, FORMAL FINDINGS
# 0.96, ORDER 0.95, but DECISION 0.03 because the caption dominates.
#
# Taking whichever verdict-ish section starts LAST is robust to all of these.
_VERDICT_SECTIONS = ("DECISION", "CONCLUSIONS", "ORDER")


def verdict_span(spans: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
    """The span the outcome should be read from: the latest verdict section."""
    candidates = [spans[name] for name in _VERDICT_SECTIONS if name in spans]
    if not candidates:
        return None
    return max(candidates, key=lambda span: span[0])


def feature_spans(spans: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    """Leak-free spans, in document order.

    Excludes SYNOPSIS and everything from ANALYSIS onward (config.FEATURE_SECTIONS),
    then drops any span that reaches into the verdict region. In ~0.17% of
    documents the headers appear out of the usual order and a fact section
    extends past the ruling; keeping it would put the verdict in the model's
    input. Losing features on 41 rows is cheap, leaking on 41 rows is not.
    """
    out = [spans[name] for name in config.FEATURE_SECTIONS if name in spans]
    verdict = verdict_span(spans)
    if verdict:
        out = [span for span in out if span[1] <= verdict[0]]
    return sorted(out, key=lambda span: span[0])
