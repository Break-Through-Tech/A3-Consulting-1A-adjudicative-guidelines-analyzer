"""Text normalisation. Every function here is pure: text in, text out.

Offset contract
---------------
`full_text` is never modified in the output -- it stays byte-identical to the
source so that extracted evidence spans remain traceable. Normalisation is
applied to a working copy.

The one transform that shifts positions is the HTML header strip. It returns
the number of characters removed (`html_prefix_len`) so callers can add it back
and keep every stored offset indexing `full_text` directly. The remaining
transforms are only applied when building feature text, which carries no
offsets.

Measured artifact rates (4,000-row sample):
    bare page-number lines   73.2%
    hyphenated line breaks   35.5%
    curly quotes / runs      76.8%
    soft hyphen               1.3%
    non-breaking space        0.4%
    tab                       0.7%
No form feeds, ligatures or U+FFFD -- PyMuPDF extracted cleanly, so there is no
OCR damage to repair.
"""
from __future__ import annotations

import re

# --- 1. HTML scrape header ---------------------------------------------------
# 7,607 rows (22.6%) were sourced from archived .html files and carry a browser
# print header before the document text, e.g.
#   02-05086.h1 file:///usr.osd.mil/...Archived%20-%20HTML/02-05086.h1.html[6/24/2021 10:49:28 AM]
# Measured across all 7,607: length varies 136-169 chars over 9 distinct values
# (159 is only the mode, at 5,035 rows). A fixed-width cut would mangle 2,572
# rows, so the prefix is located by matching its trailing timestamp.
_HTML_HEADER = re.compile(
    r"^.{0,220}?file:///.{0,220}?\[\d{1,2}/\d{1,2}/\d{4}[^\]]{0,24}\]",
    re.DOTALL,
)


def strip_html_header(text: str) -> tuple[str, int]:
    """Remove the browser print header. Returns (text, chars_removed)."""
    if not text or "file:///" not in text[:400]:
        return text, 0
    match = _HTML_HEADER.match(text)
    if not match:
        return text, 0
    return text[match.end():], match.end()


# --- 2. page numbers ---------------------------------------------------------
# PDF page breaks leave a bare number on its own line, often mid-sentence. This
# is why the source outcome patterns are littered with [\s\d]* -- they were
# working around it instead of removing it.
_PAGE_NUMBER_LINE = re.compile(r"\n[ \t]*\d{1,3}[ \t]*(?=\n)")


def strip_page_numbers(text: str) -> str:
    return _PAGE_NUMBER_LINE.sub("", text)


# --- 3. hyphenated line breaks ----------------------------------------------
# Runs after page-number removal: a page break can sit between the two halves
# of a split word, and rejoining first would leave the number embedded.
# Blank lines between the halves are allowed: removing a page number from
# "finan-\n\n7\n\ncial" leaves "finan-\n\n\ncial", which a single-newline
# pattern will not rejoin. Bounded to 3 so it cannot span paragraphs.
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t\n]{0,3}([a-z])")


def rejoin_hyphenated(text: str) -> str:
    return _HYPHEN_BREAK.sub(r"\1\2", text)


# --- 4. punctuation and whitespace ------------------------------------------
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "",  # soft hyphen
    "﻿": "",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))
_HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
_BLANK_RUN = re.compile(r"\n{3,}")


def fold_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], text)


def collapse_whitespace(text: str) -> str:
    text = _HORIZONTAL_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


# --- composed ----------------------------------------------------------------

def normalize_body(text: str) -> str:
    """Full chain minus the HTML strip, for text that is already de-prefixed.

    Order matters: page numbers before hyphen rejoining, punctuation before
    whitespace collapse.
    """
    if not text:
        return ""
    text = strip_page_numbers(text)
    text = rejoin_hyphenated(text)
    text = fold_punctuation(text)
    return collapse_whitespace(text)


def build_features(full_text: str, spans: list[tuple[int, int] | None]) -> str:
    """Concatenate the given (start, end) spans of full_text and normalise.

    Spans are the leak-free sections only -- see config.FEATURE_SECTIONS. A row
    with no usable span yields "", which validate.py reports rather than hides.
    """
    parts = [full_text[s:e] for span in spans if span for s, e in [span]]
    if not parts:
        return ""
    return normalize_body("\n\n".join(parts))
