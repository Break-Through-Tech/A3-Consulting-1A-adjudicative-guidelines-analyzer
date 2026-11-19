"""Prompts, schemas and input masks for the three LLM tasks.

The Guideline F quote extractor comes first. The rationale and the similarity
note, which src/llm/notes.py runs, are at the end.

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

from src.conditions import _pair, _scan, analysis_span, extract_case
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

def sections(row, fallback: bool = True) -> list[tuple[int, int]]:
    """(start, end) of the analysis then the conclusions, whichever exist.

    With `fallback`, an analysis the segmenter missed is found by
    `conditions.analysis_span`. The pilot passes False to keep its Nov 9 inputs.
    """
    return [p for p in (analysis_span(row, fallback), _pair(row, "conclusions")) if p is not None]


def build_input(row, mode: str, *, collapse: bool = True, fallback: bool = True) -> tuple[str, tuple[int, int]]:
    """The text to send and the region of `full_text` it came from.

    The text is the masked analysis, a blank line, then the masked conclusions.
    The region runs from the first section's start to the last section's end, so
    it spans the formal findings between them; those are never in the text.
    """
    spans = sections(row, fallback)
    if not spans:
        raise ValueError("case has neither an analysis nor a conclusions section")
    full = row["full_text"]
    # Numbered forms (DC 1, MC 3) are only found by extract_case, so mask its hits too.
    hits = extract_case(row, fallback)
    parts = []
    for start, end in spans:
        extra = [(h["start"] - start, h["end"] - start) for h in hits
                 if h["start"] >= start and h["end"] <= end]
        part = mask_section(full[start:end], mode, extra)
        parts.append(_WS.sub(" ", part).strip() if collapse else part)
    text = "\n\n".join(parts)
    region = (min(a for a, _ in spans), max(b for _, b in spans))
    return text, region


def _params(system: str, output_config: dict, text: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "output_config": output_config,
        "messages": [{"role": "user", "content": text}],
    }


def request_params(text: str, *, model: str, max_tokens: int = 8000) -> dict:
    """Messages API parameters for one case. No temperature, prefill or budget_tokens."""
    return _params(SYSTEM_PROMPT, OUTPUT_CONFIG, text, model, max_tokens)


def prompt_hash(system: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]


# --- rationale ------------------------------------------------------------------
#
# The rationale explains one schema document in plain words. Its input is only
# what the document already holds: the predicted status of each guideline at
# issue, the composed recommendation and the verified conditions with their
# quotes. No probabilities go in, so none can come out. Every passage it quotes
# is checked against `full_text` by src/llm/notes.py before it is kept.

RATIONALE_MAX_WORDS = 120
NOTE_MAX_WORDS = 60

RATIONALE_SYSTEM_PROMPT = (
    "You explain the output of a tool that analyzes DOHA security clearance decisions "
    "under the SEAD-4 adjudicative guidelines. You get the tool's predicted status for "
    "each guideline at issue, the recommendation composed from those statuses, and the "
    "conditions the tool found in the decision, each with its id, its SEAD-4 wording "
    "and the decision's own words.\n\n"
    "Write a short explanation. First state the predicted statuses and the recommendation. "
    "Then say what the decision says about the conditions found, and whether the judge's "
    "discussion points the same way as the prediction or the other way. Rules:\n"
    f"- Use three to five sentences and at most {RATIONALE_MAX_WORDS} words.\n"
    "- Use only facts that appear in the evidence given. Add no outside facts.\n"
    "- Present the statuses as the tool's prediction, not as the judge's ruling.\n"
    "- The conditions' sentences are the judge's words. Credit them to the decision or "
    "the judge, never to the tool.\n"
    "- The tool predicts from the case's facts. It does not read these conditions, so "
    "never say the prediction uses, weighs, reflects or rests on them.\n"
    "- Give no probabilities, percentages, scores or confidence, and do not say how "
    "sure the prediction is.\n"
    "- When you copy words from the decision, put them in double quotes and copy them "
    "character for character from one Evidence or Sentence line, at least three words at "
    "a time. List every such quote in `quotes` with the id of the condition it came from.\n"
    "- Do not write condition ids or citation codes in the text.\n"
    "- Put in `condition_ids` every condition the text relies on, and in "
    "`guideline_codes` every guideline it discusses.\n\n"
    "Answer only with JSON in the required shape."
)


def rationale_schema(condition_ids: list[str], guideline_codes: list[str]) -> dict:
    """The answer shape for one document. Ids and codes are limited to that document's."""
    ids = {"type": "string", "enum": list(condition_ids)}
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "condition_ids": {"type": "array", "items": ids},
            "guideline_codes": {"type": "array", "items": {"type": "string", "enum": list(guideline_codes)}},
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"condition_id": ids, "exact_quote": {"type": "string"}},
                    "required": ["condition_id", "exact_quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["text", "condition_ids", "guideline_codes", "quotes"],
        "additionalProperties": False,
    }


def at_issue(doc: dict) -> list[dict]:
    """The document's guidelines predicted FOR or AGAINST, in A to M order."""
    return [g for g in doc["guidelines"] if g["status"] != "NOT_AT_ISSUE"]


def _one_line(text: str) -> str:
    return _WS.sub(" ", text).strip()


def rationale_input(doc: dict) -> str:
    """The user message for one document: statuses, recommendation, conditions. No numbers from the heads."""
    rec = doc["recommendation"]
    lines = ["Predicted status of each guideline at issue:"]
    lines += [f"- Guideline {g['code']}, {g['name']}: {g['status']}" for g in at_issue(doc)]
    driven = ", ".join(rec["driven_by"]) or "none"
    lines += ["", f"Composed recommendation: {rec['composed']}. Guidelines predicted AGAINST: {driven}.", "",
              "Conditions found in the decision:"]
    for c in doc["conditions"]:
        lines.append(f"[{c['id']}] {c['code']}, a Guideline {c['guideline']} {c['kind']}. "
                     f"SEAD-4 wording: {_one_line(c['text'])}")
        if c["source"] == "llm":
            lines.append(f"Evidence: {_one_line(c['quote'])}")
        if c["lead_in"] is not None:
            lines.append(f"List lead-in: {_one_line(c['lead_in']['text'])}")
        lines.append(f"Sentence: {_one_line(c['context']['text'])}")
    return "\n".join(lines)


def rationale_params(doc: dict, *, model: str, max_tokens: int = 4000) -> dict:
    schema = rationale_schema([c["id"] for c in doc["conditions"]], [g["code"] for g in at_issue(doc)])
    config = {"effort": "medium", "format": {"type": "json_schema", "schema": schema}}
    return _params(RATIONALE_SYSTEM_PROMPT, config, rationale_input(doc), model, max_tokens)


# --- similarity note ------------------------------------------------------------
#
# One note per precedent: what the facts of the case under review share with the
# precedent's best paragraph from PrecedentRetriever.search. Both texts come from
# findings of fact, so neither states an outcome. The note gives no score.

NOTE_SYSTEM_PROMPT = (
    "You compare the facts of two DOHA security clearance decisions. You get the "
    "findings of fact of a case under review and one paragraph from the findings of "
    "fact of an earlier decision, a precedent.\n\n"
    "Write what the facts in the precedent paragraph share with the case under review. "
    "Rules:\n"
    f"- Use one or two sentences and at most {NOTE_MAX_WORDS} words.\n"
    "- Use only facts that appear in the two texts. Add no outside facts.\n"
    "- If they share little, say what little they share in one sentence.\n"
    "- Do not rate how similar or relevant the precedent is, and give no score, "
    "percentage or confidence.\n"
    "- Do not mention outcomes, rulings, clearances or which side should win.\n\n"
    "Answer only with JSON in the required shape."
)

NOTE_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}},
    "required": ["note"],
    "additionalProperties": False,
}

NOTE_OUTPUT_CONFIG = {"effort": "medium", "format": {"type": "json_schema", "schema": NOTE_SCHEMA}}


def note_input(query_facts: str, precedent_case: str, paragraph: str) -> str:
    return (f"Case under review, findings of fact:\n{_one_line(query_facts)}\n\n"
            f"Precedent {precedent_case}, paragraph from its findings of fact:\n{_one_line(paragraph)}")


def note_params(query_facts: str, precedent_case: str, paragraph: str, *, model: str,
                max_tokens: int = 4000) -> dict:
    return _params(NOTE_SYSTEM_PROMPT, NOTE_OUTPUT_CONFIG, note_input(query_facts, precedent_case, paragraph),
                   model, max_tokens)
