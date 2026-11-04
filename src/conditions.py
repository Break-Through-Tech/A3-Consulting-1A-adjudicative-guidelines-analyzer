"""Guideline F conditions the judge actually cited, with character offsets.

The subparagraph is the condition. `AG ¶ 19` names the whole disqualifying
paragraph and is not a citation; `AG ¶ 19(a)` is. Letters are accepted only
when they are on the closed list in src.sead4_reference, so a pattern cannot
invent `19(z)`.

Three forms show up in the decisions:

* SEAD-4: `AG ¶ 19(a)`, `¶ 20(b)`, `AG & 19(a)` (the pilcrow often survives
  HTML as an ampersand), and lists such as `20(b)-(d)` or `19(a) and (c)`.
* Directive 5220.6: `E2.A6.1.2.n` (disqualifying) and `E2.A6.1.3.n`
  (mitigating). The heading paragraphs `E2.A6.1.2` and `E2.A6.1.3` are not
  conditions, and neither is a number the Directive does not have.
* Short forms that still name the paragraph: `MC 20(a)`, `FC DC 19(c)`.
  `FC MC 1` is the Directive's own first mitigator, so it becomes
  `E2.A6.1.3.1`. A bare `DC 3` does not name Guideline F and is left alone.
"""
from __future__ import annotations

import re

from src.sead4_reference import condition

# Directive 5220.6 Guideline F stops at these subparagraphs. A higher number
# is not a real condition.
_DIRECTIVE_LAST = {2: 5, 3: 6}
_DIRECTIVE_KIND = {2: "disqualifier", 3: "mitigator"}

# Same condition, different rulebook. Only where the Directive sentence is one
# SEAD-4 code. E2.A6.1.2.3 ("inability or unwillingness") was later split into
# 19(a) and 19(b), so it has no single code and is not in this table.
_DIRECTIVE_SEAD4 = {
    (2, 1): "AG ¶ 19(c)",  # history of not meeting financial obligations
    (2, 2): "AG ¶ 19(d)",  # deceptive or illegal financial practices
    (2, 4): "AG ¶ 19(g)",  # unexplained affluence
    (3, 1): "AG ¶ 20(a)",  # not recent
    (3, 3): "AG ¶ 20(b)",  # largely beyond the person's control
    (3, 4): "AG ¶ 20(c)",  # counseling, problem under control
    (3, 5): "AG ¶ 20(f)",  # affluence from a legal source
    (3, 6): "AG ¶ 20(d)",  # good-faith effort to repay
}

# Longer prefixes first. `19(a)` with no prefix is the bare short form.
_CITE = re.compile(
    r"(?P<prefix>\b(?:FC\s*)?(?:MCs?|DCs?)\b"
    r"|\b(?:Mitigating|Disqualifying)\s+Conditions?"
    r"|AG\s*(?:¶{1,2}|&)"
    r"|\bpara(?:graph)?\.?"
    r"|¶{1,2})?"
    r"\s*(?<!\d)(?P<num>19|20)\s*\(\s*(?P<letter>[a-z])\s*\)",
    re.IGNORECASE,
)
# `20(a) through (d)` and `20(b)-(d)` fill the letters in between.
# `19(a), (c) and (g)` adds only the letters that are written.
_RANGE = re.compile(r"\s*(?:through|thru|[-–])\s*\(([a-z])\)", re.IGNORECASE)
_LIST = re.compile(r"\s*(?:,|and|or)\s*\(([a-z])\)", re.IGNORECASE)
# Final number required. `E2.A6.1.2` is the heading; `E2.A6.1.2.1.2` is not a code.
_DIRECTIVE = re.compile(r"E2\.A6\.1\.(2|3)\.(\d{1,2})(?!\d)(?!\.\d)", re.IGNORECASE)
# `FC DC 3` is Directive condition 3. `FC DC 19(c)` is the letter form, handled above.
_ORDINAL = re.compile(r"\bFC\s*(DC|MC)s?\s+(\d{1,2})(?!\d)(?!\s*\()", re.IGNORECASE)


def _ag(number: int, letter: str) -> dict | None:
    """The closed-list record for a Guideline F letter, or None."""
    row = condition(f"AG ¶ {number}({letter})")
    if row is None or row["guideline"] != "F":
        return None
    return row


def _directive(section: int, number: int) -> dict | None:
    """A real E2.A6.1.2 / E2.A6.1.3 subparagraph, or None."""
    last = _DIRECTIVE_LAST.get(section)
    if last is None or not 1 <= number <= last:
        return None
    code = f"E2.A6.1.{section}.{number}"
    hit = {
        "code": code,
        "kind": _DIRECTIVE_KIND[section],
    }
    sead4 = _DIRECTIVE_SEAD4.get((section, number))
    if sead4 is not None:
        # Refuse the equivalent unless the closed list still contains it.
        official = condition(sead4)
        if official is not None:
            hit["sead4"] = official["code"]
    return hit


def _letter_hits(text: str) -> list[tuple[int, int, list[dict]]]:
    """One occupied span per citation, covering a list or a range that follows it."""
    occupied = []
    for match in _CITE.finditer(text):
        number = int(match.group("num"))
        first = match.group("letter").lower()
        row = _ag(number, first)
        if row is None:
            continue
        items = [{
            "code": row["code"],
            "kind": row["kind"],
            "start": match.start(),
            "end": match.end(),
        }]
        cursor = match.end()
        range_match = _RANGE.match(text, cursor)
        if range_match:
            last = range_match.group(1).lower()
            # Letters that are only implied by "through" or a hyphen have no
            # token of their own, so the position is the whole range.
            for codepoint in range(ord(first) + 1, ord(last)):
                filled = _ag(number, chr(codepoint))
                if filled is None:
                    continue
                items.append({
                    "code": filled["code"],
                    "kind": filled["kind"],
                    "start": match.start(),
                    "end": range_match.end(),
                })
            if last != first:
                endpoint = _ag(number, last)
                if endpoint is not None:
                    items.append({
                        "code": endpoint["code"],
                        "kind": endpoint["kind"],
                        "start": range_match.start(),
                        "end": range_match.end(),
                    })
            cursor = range_match.end()
        while True:
            listed = _LIST.match(text, cursor)
            if not listed:
                break
            extra = _ag(number, listed.group(1))
            if extra is None:
                break
            items.append({
                "code": extra["code"],
                "kind": extra["kind"],
                "start": listed.start(),
                "end": listed.end(),
            })
            cursor = listed.end()
        occupied.append((match.start(), cursor, items))
    return occupied


def _directive_hits(text: str) -> list[tuple[int, int, list[dict]]]:
    found = []
    for match in _DIRECTIVE.finditer(text):
        hit = _directive(int(match.group(1)), int(match.group(2)))
        if hit is None:
            continue
        found.append((match.start(), match.end(), [{
            **hit,
            "start": match.start(),
            "end": match.end(),
        }]))
    for match in _ORDINAL.finditer(text):
        section = 2 if match.group(1).upper().startswith("DC") else 3
        hit = _directive(section, int(match.group(2)))
        if hit is None:
            continue
        found.append((match.start(), match.end(), [{
            **hit,
            "start": match.start(),
            "end": match.end(),
        }]))
    return found


def _scan(text: str) -> list[dict]:
    """Citations in `text`, earliest first. A span already claimed is not read again."""
    candidates = _directive_hits(text) + _letter_hits(text)
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: list[tuple[int, int]] = []
    hits: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for start, end, items in candidates:
        if any(start < taken_end and end > taken_start for taken_start, taken_end in accepted):
            continue
        accepted.append((start, end))
        for item in items:
            key = (item["code"], item["start"], item["end"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(item)
    hits.sort(key=lambda item: (item["start"], item["end"], item["code"]))
    return hits


def _regions(text: str, spans: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    # None searches the whole decision. An empty list searches nothing: a case
    # with no analysis or conclusions has no judge's citation to read.
    if spans is None:
        return [(0, len(text))]
    regions = []
    for start, end in spans:
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if end <= start:
            continue
        regions.append((max(0, start), min(len(text), end)))
    return regions


def extract(
    text: str,
    policies: tuple[int, int] | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Guideline F conditions cited in `text`, with offsets into `text`.

    `spans` limits the search to those slices (the analysis and conclusions).
    Offsets are still indexes of `text`, which is `full_text` when the caller
    passes the whole decision. `policies` is the POLICIES boilerplate span;
    a citation inside it is the guideline being recited, not one the judge
    applied, so it is dropped.

    Each hit is `code`, `kind` (`disqualifier` or `mitigator`), `start`, and
    `end`. Directive hits whose sentence is one SEAD-4 code also carry `sead4`.
    """
    if not text:
        return []
    found = []
    seen: set[tuple[str, int, int]] = set()
    for start, end in _regions(text, spans):
        for hit in _scan(text[start:end]):
            hit_start = start + hit["start"]
            hit_end = start + hit["end"]
            if policies is not None and policies[0] <= hit_start < policies[1]:
                continue
            key = (hit["code"], hit_start, hit_end)
            if key in seen:
                continue
            seen.add(key)
            recorded = {
                "code": hit["code"],
                "kind": hit["kind"],
                "start": hit_start,
                "end": hit_end,
            }
            if "sead4" in hit:
                recorded["sead4"] = hit["sead4"]
            found.append(recorded)
    found.sort(key=lambda item: (item["start"], item["end"], item["code"]))
    return found
