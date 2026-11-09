"""Prompt, schema and input masks for the Guideline F quote extractor.

    from src.llm.templates import build_input, SYSTEM_PROMPT, SCHEMA
    text, region = build_input(row, "blind")

The model extracts and quotes. It never classifies, scores or rates anything.
It gets the judge's analysis and conclusions, with citations hidden, and returns
`{"items": [{"condition_code", "exact_quote"}]}`. `verify_guideline_quotes` then
checks each quote against the unmasked decision.

Two input modes:

    citation  masks the judge's citations (AG ¶ 19(a), DC 1, E2.A6.1.2.3 ...)
    blind     also masks every run of 30+ characters that recites a condition's
              SEAD-4 wording, because the judge often restates the condition
              right next to the citation

The formal findings sit between analysis and conclusions and state the outcome,
so they are never sent. Whitespace is collapsed before sending; verify.py ignores
layout, so quotes still map back to the original text.
"""
from __future__ import annotations

import re

from src.conditions import _pair, _scan, extract_case
from src.llm.verify import _key, guideline_conditions

MASK_TOKEN = "[CITE]"
MODES = ("citation", "blind")
GUIDELINE = "F"

# Shortest shared run, in matching-key characters, that counts as recited wording.
RECITAL_MIN = 30

_WS = re.compile(r"\s+")


def codes() -> list[str]:
    return list(guideline_conditions(GUIDELINE))


def _system_prompt() -> str:
    lines = [f"{code}: {text}" for code, text in guideline_conditions(GUIDELINE).items()]
    return (
        "You read the analysis and conclusions of a DOHA security clearance decision "
        "under SEAD-4 Guideline F (Financial Considerations). The judge applies "
        "disqualifying conditions (AG ¶ 19) and mitigating conditions (AG ¶ 20). "
        f"Citations and recitals of the guideline have been replaced with {MASK_TOKEN}.\n\n"
        "Here are the 16 conditions:\n"
        + "\n".join(lines)
        + "\n\n"
        "Find every condition the judge addresses, whether or not the judge found that it "
        "applies. A condition is addressed when the reasoning discusses the facts it covers. "
        "Do not decide whether a condition applies. Do not rate, score or classify anything.\n\n"
        "For each condition, return one or more quotes. A quote is evidence from the judge's "
        "reasoning about the facts of this case. Rules for every quote:\n"
        "- Copy it character for character from the text. Do not fix typos or change numbers.\n"
        "- Take it from one passage. Do not join passages and do not use ellipses.\n"
        f"- Do not include {MASK_TOKEN}.\n"
        "- Do not use text that only restates the condition. Use case facts.\n"
        "- Use at least four words.\n\n"
        "If the text addresses none of the conditions, return an empty list. Answer only "
        "with JSON in the required shape."
    )


SYSTEM_PROMPT = _system_prompt()

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition_code": {"type": "string", "enum": codes()},
                    "exact_quote": {"type": "string"},
                },
                "required": ["condition_code", "exact_quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

OUTPUT_CONFIG = {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}


# --- masks --------------------------------------------------------------------

def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def citation_spans(text: str) -> list[tuple[int, int]]:
    """Every citation in `text` as (start, end), local to `text`, merged."""
    _, accepted = _scan(text)
    return _merge(list(accepted))


def wording_spans(text: str, guideline: str = GUIDELINE) -> list[tuple[int, int]]:
    """Runs of `text` that recite a condition's SEAD-4 wording, local to `text`.

    A run is a maximal stretch of at least min(30, len(wording key)) matching-key
    characters that the text shares with one condition's wording, ignoring case.
    """
    key, index = _key(text)
    key = key.lower()
    covered = [False] * len(key)
    for wording in guideline_conditions(guideline).values():
        wkey = _key(wording)[0].lower()
        size = min(RECITAL_MIN, len(wkey))
        if size == 0:
            continue
        for j in range(len(wkey) - size + 1):
            window = wkey[j:j + size]
            p = key.find(window)
            while p >= 0:
                for q in range(p, p + size):
                    covered[q] = True
                p = key.find(window, p + 1)
    spans = []
    q = 0
    while q < len(key):
        if covered[q]:
            r = q
            while r < len(key) and covered[r]:
                r += 1
            spans.append((index[q], index[r - 1] + 1))
            q = r
        else:
            q += 1
    return spans


def apply_mask(text: str, spans: list[tuple[int, int]], token: str = MASK_TOKEN) -> str:
    out, last = [], 0
    for a, b in _merge(spans):
        out.append(text[last:a])
        out.append(token)
        last = b
    out.append(text[last:])
    return "".join(out)


def mask_section(text: str, mode: str, extra: list[tuple[int, int]] | None = None) -> str:
    """One section's text with the mask for `mode` applied. `extra` are local spans."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    spans = citation_spans(text) + list(extra or [])
    if mode == "blind":
        spans += wording_spans(text)
    return apply_mask(text, spans)


# --- model input --------------------------------------------------------------

def sections(row) -> list[tuple[int, int]]:
    """(start, end) of the analysis then the conclusions, whichever exist."""
    return [p for p in (_pair(row, "analysis"), _pair(row, "conclusions")) if p is not None]


def build_input(row, mode: str, *, collapse: bool = True) -> tuple[str, tuple[int, int]]:
    """The text to send and the region of `full_text` it came from.

    The text is the masked analysis, a blank line, then the masked conclusions.
    The region runs from the first section's start to the last section's end, so
    it spans the formal findings between them; those are never in the text.
    """
    spans = sections(row)
    if not spans:
        raise ValueError("case has neither an analysis nor a conclusions section")
    full = row["full_text"]
    # Numbered forms (DC 1, MC 3) are only found by extract_case, so mask its hits too.
    hits = extract_case(row)
    parts = []
    for start, end in spans:
        extra = [(h["start"] - start, h["end"] - start) for h in hits
                 if h["start"] >= start and h["end"] <= end]
        part = mask_section(full[start:end], mode, extra)
        parts.append(_WS.sub(" ", part).strip() if collapse else part)
    text = "\n\n".join(parts)
    region = (min(a for a, _ in spans), max(b for _, b in spans))
    return text, region


def request_params(text: str, *, model: str, max_tokens: int = 8000) -> dict:
    """Messages API parameters for one case. No temperature, prefill or budget_tokens."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "output_config": OUTPUT_CONFIG,
        "messages": [{"role": "user", "content": text}],
    }
