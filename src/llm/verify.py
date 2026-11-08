"""Keep only the LLM quotes that really appear in the decision.

    from src.llm.verify import verify_guideline_quotes
    result = verify_guideline_quotes(full_text, items, "F", region=(start, end), mask_token="[CITE]")
    result.kept     # VerifiedQuote: code, start, end, text, context, n_matches
    result.counts   # Counter: "kept" plus one entry per drop reason

`verify_guideline_quotes` is the call the pipeline should use. It takes the
guideline's codes and their SEAD-4 wording from the closed list in
src/sead4_reference.py, so a code the list doesn't have, or a "quote" that is
only the condition's own wording, can't get through. `verify_quotes` underneath
takes those as arguments, for tests and for any other list.

The model is asked for `[{condition_code, exact_quote}]` and nothing else. It is
never asked for character offsets: every offset here comes from where the quote
is found. A quote that can't be found is dropped and counted, never displayed.

What counts as found
--------------------
`full_text` comes from PyMuPDF, which ends every printed line with a line break.
Measured over the analysis and conclusions of the 13,116 Guideline F pool cases,
88% of sentences contain a line break, 88% of cases use curly quotes, 42% have
runs of spaces and 26% split a word with a hyphen at a line break. A model quotes
with ordinary spacing and often straight quotes, so a plain `full_text.find()`
would reject almost every correct quote.

Quote and text are therefore compared on a matching key that ignores layout only:

- whitespace of every kind, including line breaks and non-breaking spaces;
- a hyphen or dash between two letters, so `repay-\\nment`, `repayment`,
  `self-\\nemployed` and `self-employed` all match. Between digits a hyphen
  counts: `2015-2019` does not match `20152019`. En and em dashes count as hyphens;
- curly versus straight quotes and apostrophes;
- Unicode compatibility forms, soft hyphens and zero-width characters;
- page furniture inside the passage, whether the quote keeps it or skips it: a
  page number on its own line (in 91.5% of decisions, and inside 3.95% of
  analysis sentences), a running header such as `19-01234.h1`, or the
  `file:///...` print header that browser-printed decisions repeat on every page;
- quotation marks around the whole quote, one trailing `.`, `,`, `;` or `:`, and
  the case of the first letter, since a quote lifted from mid-sentence is often
  given a capital.

Everything else must match exactly: every other letter, digit and punctuation
mark, and letter case. A changed word, a changed number or a dropped word fails.
The span returned is always the original text, line breaks included.

What it does not check
----------------------
That a quote exists is not that it supports the condition. A real passage can
also mislead out of context: "he acted responsibly" is in "he has not shown that
he acted responsibly". So every kept quote carries the whole sentence around it
(`context`) for display, and support is measured by the pilot score and the
traceability check, not here.

Drop reasons, in the order they are checked
-------------------------------------------
malformed           not a {condition_code: str, exact_quote: str} object
empty_quote         nothing left once layout is ignored
bad_code            code is blank, or not in `allowed_codes`
contains_mask       quote includes the token that replaced a citation in the input
too_short           fewer than `min_chars` matching characters; short quotes such
                    as "Applicant" occur all over a decision and prove nothing
restates_guideline  quote is the SEAD-4 condition wording itself, not case evidence
outside_region      found, but only outside the part of the decision the model saw
ellipsis            not found whole, but each piece between "..." marks is: the
                    model stitched separate passages together
case_mismatch       not found as written, but found ignoring letter case
near_miss           not found, but within a few characters of a passage: usually a
                    corrected typo, a reformatted number or one changed word. Still
                    dropped, and `span` says where, so the pilot can review them
not_found           anything else: reworded, merged or invented
duplicate           same code and same span as a quote already kept
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.sead4_reference import conditions_for

MALFORMED = "malformed"
EMPTY_QUOTE = "empty_quote"
BAD_CODE = "bad_code"
CONTAINS_MASK = "contains_mask"
TOO_SHORT = "too_short"
RESTATES_GUIDELINE = "restates_guideline"
OUTSIDE_REGION = "outside_region"
ELLIPSIS = "ellipsis"
CASE_MISMATCH = "case_mismatch"
NEAR_MISS = "near_miss"
NOT_FOUND = "not_found"
DUPLICATE = "duplicate"
KEPT = "kept"

DROP_REASONS = (
    MALFORMED, EMPTY_QUOTE, BAD_CODE, CONTAINS_MASK, TOO_SHORT, RESTATES_GUIDELINE,
    OUTSIDE_REGION, ELLIPSIS, CASE_MISMATCH, NEAR_MISS, NOT_FOUND, DUPLICATE,
)

# About four words. Measured in matching-key characters, so spaces don't count.
DEFAULT_MIN_CHARS = 20

# A quote counts as restating the guideline when this share of it is one
# unbroken run of a condition's reference wording.
RESTATE_SHARE = 0.8

# A near miss is at most this many character edits, or one per 50 characters
# for longer quotes.
NEAR_MISS_EDITS = 2

# How far either side of a quote to look for the edges of its sentence.
CONTEXT_LIMIT = 1000

# --- the matching key ----------------------------------------------------------

_QUOTE_CHARS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
}
# Hyphens and dashes, after NFKC has folded the full-width and small forms.
_DASHES = set("-‐‑‒–—―−")
# Soft hyphen, zero-width space and joiners, word joiner, byte-order mark.
_INVISIBLE = set("­​‌‍⁠﻿")

# A line holding nothing but a page number, as PyMuPDF writes page footers.
_PAGE_NUMBER_LINE = re.compile(r"\n[ \t]*\d{1,3}[ \t]*(?=\n)")
# The web-archive print header, repeated at the top of every page of decisions
# that were printed from a browser: `03-04628.h1` then `file:///...[6/24/2021 3:11:28 PM]`.
_PRINT_HEADER_LINE = re.compile(r"^\s*(?:file:///.*|\d{2}-\d{4,5}\.h\d+)\s*$")
_LINE = re.compile(r"[^\n]+")

_ELLIPSIS = re.compile(r"\.\s*\.\s*\.|…")
_TRAILING_PUNCT = ".,;:"

# Sentence edges, for context. A full stop followed by a capital starts a new
# sentence; abbreviations such as "Mr." can cut one short, which only trims context.
_SENTENCE_START = re.compile(r"[.!?][\"'’”)\]]*\s+(?=[\"'‘“(\[]?[A-Z])")
_SENTENCE_END = re.compile(r"[.!?][\"'’”)\]]*(?=\s|$)")


def _fold(ch: str) -> str:
    """One character as it appears in the matching key: '' if layout-only."""
    if ch in _QUOTE_CHARS:
        return _QUOTE_CHARS[ch]
    if ch in _DASHES:
        return "-"
    if ch.isspace() or ch in _INVISIBLE:
        return ""
    out = []
    for c in unicodedata.normalize("NFKC", ch):
        c = _QUOTE_CHARS.get(c, c)
        if c in _DASHES:
            out.append("-")
        elif not (c.isspace() or c in _INVISIBLE):
            out.append(c)
    return "".join(out)


def _could_be_header(line: str) -> bool:
    """Short, has a letter and a digit, doesn't end a sentence: `19-01234.h1`, `Page 2 of 7`.

    Needing a letter keeps amounts such as `$1,234` in debt tables from being skipped.
    """
    return (0 < len(line) <= 60 and line[-1] not in ".?!:;,"
            and any(c.isdigit() for c in line) and any(c.isalpha() for c in line))


def _furniture(text: str) -> set[int]:
    """Indices of page furniture: page numbers, print headers, running headers (3+ repeats)."""
    skipped: set[int] = set()
    for m in _PAGE_NUMBER_LINE.finditer(text):
        skipped.update(range(m.start(), m.end()))
    lines = [(m.start(), m.end(), m.group().strip()) for m in _LINE.finditer(text)]
    counts = Counter(s for _, _, s in lines if _could_be_header(s))
    for a, b, s in lines:
        if counts.get(s, 0) >= 3 or _PRINT_HEADER_LINE.match(s):
            skipped.update(range(a, b))
    return skipped


def _key(text: str, skip: set[int] | None = None) -> tuple[str, list[int]]:
    """The matching key for `text`, and for each key character its index in `text`."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if skip and i in skip:
            continue
        for c in _fold(ch):
            chars.append(c)
            index.append(i)

    # Hyphens: a run counts once, and one between two letters is layout.
    out_c: list[str] = []
    out_i: list[int] = []
    n = len(chars)
    for j, c in enumerate(chars):
        if c == "-":
            if out_c and out_c[-1] == "-":
                continue
            nxt = j + 1
            while nxt < n and chars[nxt] == "-":
                nxt += 1
            if out_c and out_c[-1].isalpha() and nxt < n and chars[nxt].isalpha():
                continue
        out_c.append(c)
        out_i.append(index[j])
    return "".join(out_c), out_i


def _unwrap(k: str) -> str:
    if len(k) > 2 and k[0] in "\"'" and k[-1] in "\"'":
        return k[1:-1]
    if k[:1] in "\"'":
        return k[1:]
    if k[-1:] in "\"'":
        return k[:-1]
    return k


def _untrail(k: str) -> str:
    return k[:-1] if k[-1:] and k[-1] in _TRAILING_PUNCT else k


def _core(qkey: str) -> str:
    """The quote's key without wrapping quotation marks or a trailing mark."""
    return _untrail(_unwrap(qkey)) or qkey


def _candidates(qkey: str) -> list[str]:
    """Forms of the quote to search for, most literal first."""
    out: list[str] = []
    for k in (qkey, _unwrap(qkey), _untrail(qkey), _untrail(_unwrap(qkey))):
        if k and k not in out:
            out.append(k)
    # A quote lifted from mid-sentence is often given a capital, or loses one.
    for k in list(out):
        if k[0].isalpha():
            swapped = k[0].swapcase() + k[1:]
            if swapped not in out:
                out.append(swapped)
    return out


# --- condition codes -------------------------------------------------------------

_SEAD4_CODE = re.compile(r"^(?:AG)?[\s¶&]*(\d{1,2})\.?\s*\(\s*([a-z])\s*\)$", re.IGNORECASE)


def normalize_code(code: str) -> str:
    """`AG ¶ 20(b)`, `20(b)`, `¶20 (B)`, `AG ¶¶ 20(b)`, `AG & 20(b)` and `20.(b)`
    all become `AG ¶ 20(b)`, the form the closed list uses.

    Directive-era codes such as `E2.A6.1.3.1` are kept, uppercased, without spaces.
    """
    text = unicodedata.normalize("NFKC", code).strip()
    m = _SEAD4_CODE.match(text)
    if m:
        return f"AG ¶ {int(m.group(1))}({m.group(2).lower()})"
    return re.sub(r"\s+", "", text).upper()


# --- results -------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifiedQuote:
    condition_code: str
    start: int            # full_text[start:end] is the quoted passage
    end: int
    text: str             # the original span, with its line breaks
    context_start: int    # full_text[context_start:context_end] is its whole sentence,
    context_end: int      # shown so a reader sees what surrounds the quote
    context: str
    n_matches: int        # times the quote occurs where it was searched; the first is kept


@dataclass(frozen=True)
class DroppedQuote:
    item: object                         # what the model returned, unchanged
    reason: str
    span: tuple[int, int] | None = None  # for outside_region and near_miss: where to look


@dataclass
class VerifyResult:
    kept: list[VerifiedQuote] = field(default_factory=list)
    dropped: list[DroppedQuote] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        counts = Counter({KEPT: len(self.kept)})
        counts.update(d.reason for d in self.dropped)
        return counts


def total_counts(results: Iterable[VerifyResult]) -> Counter:
    """Kept and dropped counts summed over many decisions, for the pilot report."""
    total: Counter = Counter()
    for result in results:
        total.update(result.counts)
    return total


# --- searching -------------------------------------------------------------------------

class _Searchable:
    """One matching key of full_text, limited to the region the model saw."""

    def __init__(self, full_text: str, region: tuple[int, int] | None, skip: set[int] | None):
        self.key, self.index = _key(full_text, skip)
        self.lo, self.hi = 0, len(self.key)
        if region is not None:
            self.lo = bisect_left(self.index, region[0])
            self.hi = bisect_left(self.index, region[1])
        self._folded: str | None = None

    def find(self, k: str) -> int:
        return self.key.find(k, self.lo, self.hi)

    def span(self, p: int, length: int) -> tuple[int, int]:
        """Original offsets of key[p:p + length]."""
        return self.index[p], self.index[p + length - 1] + 1

    def found_ignoring_case(self, k: str) -> bool:
        if self._folded is None:
            self._folded = self.key[self.lo:self.hi].casefold()
        return k.casefold() in self._folded


def _restates(core: str, reference_keys: list[str]) -> bool:
    q = core.casefold()
    for ref in reference_keys:
        match = SequenceMatcher(None, q, ref, autojunk=False).find_longest_match(0, len(q), 0, len(ref))
        if match.size >= RESTATE_SHARE * len(q):
            return True
    return False


def _edit_distance_within(q: str, w: str, budget: int) -> tuple[int, int] | None:
    """Fewest edits turning `q` into some stretch of `w`, and where that stretch ends.

    None as soon as every alignment needs more than `budget` edits.
    """
    prev = [0] * (len(w) + 1)          # the stretch may start anywhere in w
    for i, qc in enumerate(q, 1):
        cur = [i] * (len(w) + 1)
        for j in range(1, len(w) + 1):
            cur[j] = min(prev[j - 1] + (qc != w[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        if min(cur) > budget:
            return None
        prev = cur
    best = min(prev)
    return best, prev.index(best)


def _near_miss(core: str, s: _Searchable) -> tuple[int, int] | None:
    """Original offsets of a passage within a few edits of `core`, if the region has one."""
    budget = max(NEAR_MISS_EDITS, len(core) // 50)
    region = s.key[s.lo:s.hi]
    if len(core) < 12 or not region:
        return None
    # Anchor on 12-character pieces that do match exactly, then align around them.
    starts: list[int] = []
    for off in range(0, len(core) - 11, 6):
        p = region.find(core[off:off + 12])
        if p >= 0 and p - off not in starts:
            starts.append(p - off)
        if len(starts) >= 6:
            break
    slack = budget + 2
    best = None
    for st in starts:
        w_lo = max(0, st - slack)
        w_hi = min(len(region), st + len(core) + slack)
        found = _edit_distance_within(core, region[w_lo:w_hi], budget)
        if found and (best is None or found[0] < best[0]):
            best = (found[0], w_lo + found[1])
    if best is None or best[1] == 0:
        return None
    end_k = s.lo + best[1]
    start_k = max(s.lo, end_k - len(core))
    return s.span(start_k, end_k - start_k)


def _why_not_found(quote: str, core: str, candidates: list[str], keys: list[_Searchable],
                   region: tuple[int, int] | None, min_chars: int) -> tuple[str, tuple[int, int] | None]:
    """The most useful reason a quote was not found in the region, and where to look."""
    if region is not None:
        for s in keys:
            for k in candidates:
                p = s.key.find(k)
                if p >= 0:
                    return OUTSIDE_REGION, s.span(p, len(k))
    # Real passages joined with "..." -- every piece is in the text, the whole is not.
    # Judges also quote with ". . .", so a quote that merely contains one is not enough.
    pieces = [_key(part)[0] for part in _ELLIPSIS.split(quote)]
    pieces = [p for p in pieces if p]
    if len(pieces) > 1 and all(
        len(p) >= min_chars // 2 and any(s.find(c) >= 0 for s in keys for c in _candidates(p))
        for p in pieces
    ):
        return ELLIPSIS, None
    if any(s.found_ignoring_case(k) for s in keys for k in candidates):
        return CASE_MISMATCH, None
    for s in keys:
        span = _near_miss(core, s)
        if span:
            return NEAR_MISS, span
    return NOT_FOUND, None


def _sentence_around(full_text: str, start: int, end: int) -> tuple[int, int]:
    """Offsets of the sentence holding full_text[start:end], capped at CONTEXT_LIMIT each side."""
    c_start = start
    for m in _SENTENCE_START.finditer(full_text, max(0, start - CONTEXT_LIMIT), min(len(full_text), start + 3)):
        if m.end() <= start:
            c_start = m.end()
    m = _SENTENCE_END.search(full_text, max(start, end - 1), min(len(full_text), end + CONTEXT_LIMIT))
    c_end = max(end, m.end()) if m else end
    return c_start, c_end


# --- the check ----------------------------------------------------------------------

def guideline_conditions(guideline: str) -> dict[str, str]:
    """A guideline's closed-list codes with their SEAD-4 wording, e.g. 16 for F."""
    conditions = {row["code"]: row["text"] for row in conditions_for(guideline)}
    if not conditions:
        raise ValueError(f"no SEAD-4 conditions for guideline {guideline!r}")
    return conditions


def verify_guideline_quotes(
    full_text: str,
    items: Sequence[object],
    guideline: str = "F",
    *,
    region: tuple[int, int] | None = None,
    mask_token: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> VerifyResult:
    """verify_quotes with the closed list filled in for one guideline.

    Only that guideline's codes are allowed, and a quote that restates any of its
    conditions' wording is dropped. This is the call the pilot and the run use.
    """
    conditions = guideline_conditions(guideline)
    return verify_quotes(
        full_text,
        items,
        allowed_codes=conditions,
        region=region,
        mask_token=mask_token,
        reference_texts=conditions,
        min_chars=min_chars,
    )


def verify_quotes(
    full_text: str,
    items: Sequence[object],
    *,
    allowed_codes: Iterable[str] | None = None,
    region: tuple[int, int] | None = None,
    mask_token: str | None = None,
    reference_texts: Iterable[str] | Mapping[str, str] | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> VerifyResult:
    """Check each `{condition_code, exact_quote}` item against `full_text`.

    allowed_codes    the codes the model may use; compared after normalize_code.
                     None skips the check. verify_guideline_quotes fills it in.
    region           (start, end) offsets of the part of full_text the model was
                     given. A quote found only outside it is dropped.
    mask_token       the placeholder that replaced AG citations in the model's input.
    reference_texts  SEAD-4 condition wording; a quote that only restates it is
                     dropped. A mapping is read for its values.
    min_chars        shortest quote accepted, in matching-key characters.
    """
    if not isinstance(full_text, str):
        raise TypeError("full_text must be a string")
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
        raise TypeError("items must be a list of {condition_code, exact_quote} objects")

    allowed = None
    if allowed_codes is not None:
        allowed = {normalize_code(c): c for c in allowed_codes}
    if isinstance(reference_texts, Mapping):
        reference_texts = reference_texts.values()
    reference_keys = [_key(t)[0].casefold() for t in (reference_texts or ())]

    # Two keys: one keeps page furniture, so a quote that includes it still
    # matches; the other skips it, so a quote that leaves it out matches too.
    keys = [_Searchable(full_text, region, None)]
    furniture = _furniture(full_text)
    if furniture:
        keys.append(_Searchable(full_text, region, furniture))

    def search(cands: list[str]) -> tuple[_Searchable, str, int] | None:
        for s in keys:
            for k in cands:
                p = s.find(k)
                if p >= 0:
                    return s, k, p
        return None

    result = VerifyResult()
    seen: set[tuple[str, int, int]] = set()

    def drop(item: object, reason: str, span: tuple[int, int] | None = None) -> None:
        result.dropped.append(DroppedQuote(item, reason, span))

    for item in items:
        code = item.get("condition_code") if isinstance(item, Mapping) else None
        quote = item.get("exact_quote") if isinstance(item, Mapping) else None
        if not isinstance(code, str) or not isinstance(quote, str):
            drop(item, MALFORMED)
            continue

        qkey = _key(quote)[0]
        if not qkey:
            drop(item, EMPTY_QUOTE)
            continue

        norm = normalize_code(code)
        if not norm or (allowed is not None and norm not in allowed):
            drop(item, BAD_CODE)
            continue
        code_out = allowed[norm] if allowed is not None else norm

        if mask_token and mask_token in quote:
            drop(item, CONTAINS_MASK)
            continue

        core = _core(qkey)
        if len(core) < min_chars:
            drop(item, TOO_SHORT)
            continue

        if reference_keys and _restates(core, reference_keys):
            drop(item, RESTATES_GUIDELINE)
            continue

        candidates = _candidates(qkey)
        hit = search(candidates)
        if hit is None:
            drop(item, *_why_not_found(quote, core, candidates, keys, region, min_chars))
            continue

        s, k, p = hit
        start, end = s.span(p, len(k))
        if (code_out, start, end) in seen:
            drop(item, DUPLICATE)
            continue
        seen.add((code_out, start, end))
        c_start, c_end = _sentence_around(full_text, start, end)
        result.kept.append(VerifiedQuote(
            condition_code=code_out,
            start=start,
            end=end,
            text=full_text[start:end],
            context_start=c_start,
            context_end=c_end,
            context=full_text[c_start:c_end],
            n_matches=s.key.count(k, s.lo, s.hi),
        ))

    return result
