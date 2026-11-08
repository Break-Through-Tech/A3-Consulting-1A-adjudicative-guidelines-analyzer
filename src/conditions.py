"""Guideline F conditions the judge actually cited, with character offsets.

The subparagraph is the condition. `AG ¶ 19` names the whole disqualifying
paragraph and is not a citation; `AG ¶ 19(a)` is. Letters are accepted only
when they are on the closed list in src.sead4_reference, so a pattern cannot
invent `19(z)`.

Three forms show up in the decisions:

* SEAD-4: `AG ¶ 19(a)`, `¶ 20(b)`, `AG & 19(a)` (the pilcrow often survives
  HTML as an ampersand), and lists or ranges such as `20(a), (b), and (c)`
  or `20(a) through 20(d)`.
* Directive 5220.6: `E2.A6.1.2.n` (disqualifying) and `E2.A6.1.3.n`
  (mitigating), including a space or line break after a dot. The heading
  paragraphs `E2.A6.1.2` and `E2.A6.1.3` are not conditions.
* Short forms. `MC 20(a)` still names a paragraph. `DC 1` and
  `Mitigating Condition 3` name a position in the list, and which list that
  is depends on the case's regime and date.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from src import config
from src.sead4_reference import condition

# Directive 5220.6 Guideline F stops at these subparagraphs. A higher number
# is not a real condition. SEAD-4's list is checked through the closed list,
# which stops at 19(i) and 20(g).
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
    r"\s*(?<!\d)(?P<num>19|20)\s*\.?\s*\(\s*(?P<letter>[a-z])\s*\)",
    re.IGNORECASE,
)
# `20(a) through 20(d)`, `20(a)—(c)`, `20(a) to (c)`, `20(a) - AG ¶ 20(c)`.
# A far end that names the other paragraph is not this range.
_RANGE = re.compile(
    r"\s*(?:through|thru|to|[-–—])\s*"
    r"(?:AG\s*(?:¶{1,2}|&)\s*)?"
    r"(?:(?P<num>19|20)\s*\.?\s*)?"
    r"\((?P<letter>[a-z])\)",
    re.IGNORECASE,
)
# `20(a), (b), and (c)` keeps the item after the comma-and, not only `(b)`.
_LIST = re.compile(
    r"\s*(?:,\s*(?:and|or|&)|,|and|or|&)\s*\(([a-z])\)",
    re.IGNORECASE,
)
# Whitespace, including a line break, may follow any dot. `E2.A6.1.3.3.3`
# is still not a code: the final number cannot introduce another dotted number.
_DIRECTIVE = re.compile(
    r"E2\.A6\.\s*1\.\s*(2|3)\.\s*(\d{1,2})(?!\d)(?!\.\s*\d)",
    re.IGNORECASE,
)
# `DC 1`, `MCs 3`, `FC MC 1`, `Disqualifying Condition (DC) 1`.
# `(?!\d)` keeps `DC 20301` from being `DC 2`. A bracketed letter, as in
# `MC 20(a)`, still belongs to the paragraph pattern. A footnote such as
# `DC 1 (10)` does not.
_ORDINAL = re.compile(
    r"(?P<fc>\bFC\s*)?"
    r"(?:"
    r"(?P<short>DCs?|MCs?)"
    r"|"
    r"(?P<long>Disqualifying|Mitigating)\s+Conditions?"
    r"(?:\s*\(\s*(?:FC\s+)?(?:DC|MC)\s*\))?"
    r")"
    r"\s+(?P<num>\d{1,2})(?!\d)(?!\s*\([a-z]\))",
    re.IGNORECASE,
)
# `AG ¶ 19` with no letter, then `(a) inability` on its own list line.
# `AG & 19` and a bare `¶ 19` are the same header. A following `19(a)` is not,
# and neither is a longer number such as `AG ¶ 190` or `¶ 2001`.
_BARE_HEADER = re.compile(
    r"(?:AG\s*(?:¶{1,2}|&)|¶{1,2})\s*(?<!\d)(19|20)(?!\d)(?!\s*\.?\s*\(\s*[a-z]\s*\))",
    re.IGNORECASE,
)
_NEXT_PARAGRAPH = re.compile(
    r"(?:AG\s*(?:¶{1,2}|&)|¶{1,2})\s*\d+",
    re.IGNORECASE,
)
# A list entry starts after a colon, a semicolon, "and", "or", or a line break,
# and the bracketed letter is followed by a word.
_BARE_ITEM = re.compile(
    r"(?:(?<=[:;])|(?<=\n)|(?<=\band)|(?<=\bor))\s*(\([a-z]\))\s+[A-Za-z]",
    re.IGNORECASE,
)

_GUIDELINE_PATTERNS: list[tuple[str, re.Pattern]] = []
for _code, _name in sorted(config.GUIDELINE_NAMES.items(), key=lambda item: -len(item[1])):
    _GUIDELINE_PATTERNS.append((_code, re.compile(rf"\b{re.escape(_name)}\b", re.IGNORECASE)))
for _code in config.GUIDELINE_CODES:
    _GUIDELINE_PATTERNS.append((_code, re.compile(rf"\bGuideline\s+{_code}\b", re.IGNORECASE)))

_CASE_COLUMNS = [
    "case_number",
    "source_url",
    "full_text",
    "guidelines",
    "date",
    "guideline_regime",
    "sec_analysis_start",
    "sec_analysis_end",
    "sec_conclusions_start",
    "sec_conclusions_end",
    "sec_policies_start",
    "sec_policies_end",
]


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


def _between(number: int, start: str, end: str) -> list[str]:
    """Letters from `start` to `end`, excluding `start`. Empty when they match."""
    if start == end:
        return []
    step = 1 if ord(end) > ord(start) else -1
    letters = []
    codepoint = ord(start) + step
    while codepoint != ord(end) + step:
        letters.append(chr(codepoint))
        codepoint += step
    return letters


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
        last = first
        # Ranges and list items can alternate: `20(a)-(c), and (e)`.
        # A range always starts from the letter just read, not the first one.
        while True:
            range_match = _RANGE.match(text, cursor)
            if range_match is not None:
                far = range_match.group("num")
                if far is not None and int(far) != number:
                    range_match = None
            if range_match is not None:
                end = range_match.group("letter").lower()
                for letter in _between(number, last, end):
                    filled = _ag(number, letter)
                    if filled is None:
                        continue
                    items.append({
                        "code": filled["code"],
                        "kind": filled["kind"],
                        "start": range_match.start(),
                        "end": range_match.end(),
                    })
                last = end
                cursor = range_match.end()
                continue
            listed = _LIST.match(text, cursor)
            if listed is None:
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
            last = listed.group(1).lower()
            cursor = listed.end()
        occupied.append((match.start(), cursor, items))
    return occupied


def _bare_list_hits(text: str) -> list[tuple[int, int, list[dict]]]:
    """Letters listed under a bare `AG ¶ 19` or `AG ¶ 20`.

    The judge names the paragraph, then writes `(a) inability` and so on.
    The window stops at the next paragraph citation, or at 4,000 characters.
    The first letter must sit within 400 characters of the header, and each
    later letter within 1,200 of the one before, so a page number in the
    middle of the list does not break it. Letters have to ascend, and each
    one has to be on the closed list for that paragraph.
    """
    found = []
    for header in _BARE_HEADER.finditer(text):
        number = int(header.group(1))
        window_end = min(len(text), header.end() + 4000)
        later = _NEXT_PARAGRAPH.search(text, header.end(), window_end)
        if later is not None:
            window_end = later.start()
        items = []
        last_letter = ""
        last_at = header.end()
        first = True
        for item in _BARE_ITEM.finditer(text, header.end(), window_end):
            if first:
                if item.start() - header.end() > 400:
                    break
            elif item.start() - last_at > 1200:
                break
            letter = item.group(1)[1].lower()
            row = _ag(number, letter)
            if row is None or (last_letter and letter <= last_letter):
                continue
            bracket = item.group(1)
            # Offsets cover the bracketed letter itself, `(b)`, not the words after it.
            start = item.start() + text[item.start():item.end()].lower().find(bracket.lower())
            items.append({
                "code": row["code"],
                "kind": row["kind"],
                "start": start,
                "end": start + len(bracket),
            })
            last_letter = letter
            last_at = item.start()
            first = False
        if items:
            found.append((header.start(), items[-1]["end"], items))
    return found


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
    return found


def _scan(text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Citations in `text`, earliest first. A span already claimed is not read again."""
    candidates = _directive_hits(text) + _letter_hits(text) + _bare_list_hits(text)
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
    return hits, accepted


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


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < taken_end and end > taken_start for taken_start, taken_end in taken)


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict)):
        return False
    try:
        return bool(value != value)
    except Exception:
        return False


def _year(decided) -> int | None:
    if decided is None or _missing(decided):
        return None
    if isinstance(decided, datetime):
        return decided.year
    if isinstance(decided, date):
        return decided.year
    year = getattr(decided, "year", None)
    if year is None or _missing(year):
        return None
    return int(year)


def _rulebook(regime: str | None, decided) -> str | None:
    """Which numbering a `DC n` / `MC n` citation uses, or None to skip it.

    An explicit directive regime uses Directive numbers. A sead4 decision
    already cites the lettered AG ¶ codes, and its numbered forms are often a
    quotation of the older Directive list, so those are skipped. Unknown, and
    a missing regime, follow the date: Directive before 2006, SEAD-4 from 2008.
    2006 and 2007 use both, so those years are skipped.
    """
    name = (regime or "").strip().lower()
    if name == "directive":
        return "directive"
    if name == "sead4":
        return None
    if name not in ("", "unknown"):
        return None
    year = _year(decided)
    if year is None:
        return None
    if year < 2006:
        return "directive"
    if year >= 2008:
        return "sead4"
    return None


def _only_f(guidelines) -> bool:
    if not guidelines:
        return False
    letters = [str(item).upper() for item in guidelines if item is not None and not _missing(item)]
    return letters == ["F"]


def _guideline_marks(text: str, regions: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """(end offset, letter) for every guideline named inside the searched text."""
    marks = []
    for start, end in regions:
        chunk = text[start:end]
        for letter, pattern in _GUIDELINE_PATTERNS:
            for match in pattern.finditer(chunk):
                marks.append((start + match.end(), letter))
    marks.sort()
    return marks


def _last_guideline(marks: list[tuple[int, str]], position: int) -> str | None:
    found = None
    for end, letter in marks:
        if end <= position:
            found = letter
        else:
            break
    return found


def _ordinal_kind(match: re.Match) -> str:
    short = match.group("short")
    if short:
        return "disqualifier" if short.upper().startswith("DC") else "mitigator"
    long = match.group("long") or ""
    return "disqualifier" if long.lower().startswith("disqualifying") else "mitigator"


def _has_fc(match: re.Match) -> bool:
    if match.group("fc"):
        return True
    return re.search(r"\bFC\b", match.group(0), re.IGNORECASE) is not None


def _numbered_record(kind: str, number: int, rulebook: str) -> dict | None:
    if rulebook == "directive":
        section = 2 if kind == "disqualifier" else 3
        return _directive(section, number)
    letter = chr(ord("a") + number - 1)
    paragraph = 19 if kind == "disqualifier" else 20
    row = _ag(paragraph, letter)
    if row is None:
        return None
    return {"code": row["code"], "kind": row["kind"]}


def _numbered_hits(
    text: str,
    regions: list[tuple[int, int]],
    taken: list[tuple[int, int]],
    guidelines,
    decided,
    regime: str | None,
) -> list[dict]:
    """`DC n` / `MC n` citations that belong to Guideline F under this case's rulebook."""
    rulebook = _rulebook(regime, decided)
    if rulebook is None:
        return []
    marks = _guideline_marks(text, regions)
    found = []
    for start, end in regions:
        for match in _ORDINAL.finditer(text[start:end]):
            hit_start = start + match.start()
            hit_end = start + match.end()
            if _overlaps(hit_start, hit_end, taken):
                continue
            counts = (
                _has_fc(match)
                or _only_f(guidelines)
                or _last_guideline(marks, hit_start) == "F"
            )
            if not counts:
                continue
            record = _numbered_record(_ordinal_kind(match), int(match.group("num")), rulebook)
            if record is None:
                continue
            found.append({
                **record,
                "start": hit_start,
                "end": hit_end,
                "numbered": True,
            })
    return found


def extract(
    text: str,
    policies: tuple[int, int] | None = None,
    spans: list[tuple[int, int]] | None = None,
    guidelines: list[str] | None = None,
    decided: date | None = None,
    regime: str | None = None,
) -> list[dict]:
    """Guideline F conditions cited in `text`, with offsets into `text`.

    `spans` limits the search to those slices (the analysis and conclusions).
    Offsets are still indexes of `text`, which is `full_text` when the caller
    passes the whole decision. `policies` is the POLICIES boilerplate span;
    a citation inside it is the guideline being recited, not one the judge
    applied, so it is dropped.

    `guidelines`, `decided` and `regime` are how a numbered form (`DC 1`,
    `MC 3`) is assigned. Without a regime and a date, numbered forms are
    skipped. Paragraph citations do not need them.

    Each hit is `code`, `kind` (`disqualifier` or `mitigator`), `start`, and
    `end`. Directive hits whose sentence is one SEAD-4 code also carry `sead4`.
    Numbered hits carry `numbered` so a sample can be drawn from that form alone.
    """
    if not text:
        return []
    regions = _regions(text, spans)
    found = []
    seen: set[tuple[str, int, int]] = set()
    taken: list[tuple[int, int]] = []
    for start, end in regions:
        hits, occupied = _scan(text[start:end])
        for hit in hits:
            hit_start = start + hit["start"]
            hit_end = start + hit["end"]
            if policies is not None and not _missing(policies[0]) and policies[0] <= hit_start < policies[1]:
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
        for occ_start, occ_end in occupied:
            taken.append((start + occ_start, start + occ_end))
    for hit in _numbered_hits(text, regions, taken, guidelines, decided, regime):
        if policies is not None and not _missing(policies[0]) and policies[0] <= hit["start"] < policies[1]:
            continue
        key = (hit["code"], hit["start"], hit["end"])
        if key in seen:
            continue
        seen.add(key)
        recorded = {
            "code": hit["code"],
            "kind": hit["kind"],
            "start": hit["start"],
            "end": hit["end"],
            "numbered": True,
        }
        if "sead4" in hit:
            recorded["sead4"] = hit["sead4"]
        found.append(recorded)
    found.sort(key=lambda item: (item["start"], item["end"], item["code"]))
    return found


def _cell(row, name: str):
    if isinstance(row, dict):
        return row.get(name)
    try:
        return row[name]
    except Exception:
        return getattr(row, name, None)


def _pair(row, name: str) -> tuple[int, int] | None:
    start = _cell(row, f"sec_{name}_start")
    end = _cell(row, f"sec_{name}_end")
    if _missing(start) or _missing(end):
        return None
    start, end = int(start), int(end)
    if end <= start:
        return None
    return start, end


def _as_date(value) -> date | None:
    if _missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        converted = value.date()
        if isinstance(converted, date):
            return converted
    return None


def _as_guidelines(value) -> list[str] | None:
    if _missing(value):
        return None
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in list(value) if item is not None and not _missing(item)]
    except TypeError:
        return None


def extract_case(row) -> list[dict]:
    """Conditions cited in one `load_cases()` row.

    The search is the analysis and conclusions. POLICIES is passed so a
    recital of the guideline is not treated as a citation. The row's
    guidelines, date and regime decide what a numbered `DC n` / `MC n` means.
    """
    text = _cell(row, "full_text") or ""
    spans = []
    for name in ("analysis", "conclusions"):
        pair = _pair(row, name)
        if pair is not None:
            spans.append(pair)
    policies = _pair(row, "policies")
    decided = _as_date(_cell(row, "date"))
    regime = _cell(row, "guideline_regime")
    if _missing(regime):
        regime = None
    return extract(
        text,
        policies=policies,
        spans=spans,
        guidelines=_as_guidelines(_cell(row, "guidelines")),
        decided=decided,
        regime=None if regime is None else str(regime),
    )


def _f_cases():
    """Pool cases that put Guideline F in issue. `full_text` is loaded on purpose."""
    from src.dataset import load_cases

    df = load_cases(columns=_CASE_COLUMNS)
    has_f = df["guidelines"].map(lambda codes: codes is not None and "F" in list(codes))
    return df.loc[has_f].reset_index(drop=True)


def _summarize(df) -> list[tuple[str, int, int, int, int]]:
    rows = []
    order = ["sead4", "directive", "unknown"]
    grouped = {name: {"cases": 0, "any": 0, "mitigator": 0, "disqualifier": 0} for name in order}
    grouped["all"] = {"cases": 0, "any": 0, "mitigator": 0, "disqualifier": 0}
    for row in df.itertuples(index=False):
        regime = str(row.guideline_regime) if not _missing(row.guideline_regime) else "unknown"
        if regime not in grouped:
            grouped[regime] = {"cases": 0, "any": 0, "mitigator": 0, "disqualifier": 0}
            order.append(regime)
        hits = extract_case(row)
        kinds = {hit["kind"] for hit in hits}
        for key in (regime, "all"):
            grouped[key]["cases"] += 1
            if hits:
                grouped[key]["any"] += 1
            if "mitigator" in kinds:
                grouped[key]["mitigator"] += 1
            if "disqualifier" in kinds:
                grouped[key]["disqualifier"] += 1
    report = []
    for name in order + ["all"]:
        bucket = grouped[name]
        report.append((
            name,
            bucket["cases"],
            bucket["any"],
            bucket["mitigator"],
            bucket["disqualifier"],
        ))
    return report


def _print_report(report: list[tuple[str, int, int, int, int]]) -> None:
    print(f"{'regime':<12} {'cases':>8} {'any condition':>22} {'mitigator':>22} {'disqualifier':>22}")
    for name, cases, any_hit, mitigator, disqualifier in report:
        def share(count: int) -> str:
            pct = 100.0 * count / cases if cases else 0.0
            return f"{count:,}  {pct:5.1f}%"

        print(f"{name:<12} {cases:8,} {share(any_hit):>22} {share(mitigator):>22} {share(disqualifier):>22}")


def _hyperlink(url: str, label: str) -> str:
    safe_url = str(url).replace('"', '""')
    safe_label = str(label).replace('"', '""')
    return f'=HYPERLINK("{safe_url}","{safe_label}")'


def _write_samples(df) -> None:
    import csv

    import pandas as pd

    out_dir = config.OUT_DIR
    cases_path = out_dir / "conditions_check_cases.csv"
    numbered_path = out_dir / "conditions_check_numbered.csv"

    frames = []
    for regime, count in (("sead4", 10), ("directive", 5), ("unknown", 5)):
        part = df.loc[df["guideline_regime"] == regime]
        frames.append(part.sample(n=count, random_state=42))
    picked = pd.concat(frames, ignore_index=True)

    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_number",
            "decision",
            "regime",
            "date",
            "codes_found",
            "wrong_codes",
            "missed_codes",
            "notes",
            "source_url",
        ])
        for row in picked.itertuples(index=False):
            hits = extract_case(row)
            codes = ", ".join(sorted({hit["code"] for hit in hits}))
            decided = _as_date(row.date)
            writer.writerow([
                row.case_number,
                _hyperlink(row.source_url, row.case_number),
                row.guideline_regime,
                "" if decided is None else decided.isoformat(),
                codes,
                "",
                "",
                "",
                row.source_url,
            ])

    numbered = []
    for row in df.itertuples(index=False):
        text = row.full_text or ""
        decided = _as_date(row.date)
        for hit in extract_case(row):
            if not hit.get("numbered"):
                continue
            before = text[max(0, hit["start"] - 200):hit["start"]]
            after = text[hit["end"]:hit["end"] + 120]
            context = " ".join(f"{before} {text[hit['start']:hit['end']]} {after}".split())
            numbered.append({
                "case_number": row.case_number,
                "date": "" if decided is None else decided.isoformat(),
                "regime": row.guideline_regime,
                "code": hit["code"],
                "cited_as": " ".join(text[hit["start"]:hit["end"]].split()),
                "context": context,
            })
    chosen = pd.DataFrame(numbered).sample(n=30, random_state=42)
    with numbered_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_number",
            "date",
            "regime",
            "code",
            "cited_as",
            "context",
            "correct",
            "notes",
        ])
        for row in chosen.itertuples(index=False):
            writer.writerow([
                row.case_number,
                row.date,
                row.regime,
                row.code,
                row.cited_as,
                row.context,
                "",
                "",
            ])
    print(f"Wrote {cases_path}")
    print(f"Wrote {numbered_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Guideline F condition coverage")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="write the hand-check CSVs under Challenge_Data/",
    )
    args = parser.parse_args()
    cases = _f_cases()
    if args.sample:
        _write_samples(cases)
        return
    _print_report(_summarize(cases))


if __name__ == "__main__":
    main()
