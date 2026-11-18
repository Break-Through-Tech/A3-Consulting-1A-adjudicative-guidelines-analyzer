"""The structured SEAD-4 document the analyzer returns for one decision.

    python -m src.schema --export results/schema/document.schema.json
    python -m src.schema --example

One document holds everything the app shows for a case:

    case            source_url, case_number, date, regime, split, text length
    guidelines      13 entries, A to M: calibrated probabilities from the heads,
                    the at-issue threshold and the predicted status
    recommendation  the composed outcome, DENIED when any head says AGAINST, and
                    the guidelines that drove it; `direct` stays empty until the
                    direct outcome head is settled
    conditions      cited conditions with an exact quote and its offsets into
                    `full_text`, from the rule extractor or from the LLM
    precedents      similar decisions from retrieval, with the best paragraph
    rationale       a short text and the ids of what it rests on, or null
    provenance      model names, calibration file, prompt hashes, run dates and
                    the drop counts from `verify`

The TypedDicts below are the single definition. `json_schema()` is generated
from them, and `validate()` checks a document against them and then checks the
facts a type cannot: every quote is `full_text[start:end]`, every code is on the
closed list, every head's probabilities sum to 1, all 13 guidelines are present
and the recommendation follows from the statuses. That is PROJECT_PLAN.md
section 11 item 9 as code.

`build_document()` fills what exists and leaves the rest empty, so a document
can be built before the LLM run, the notes and the rationale are done.

The LLM never supplies a probability or a confidence. Every number in
`guidelines` comes from the heads.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import types
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict, Union, get_args, get_origin, get_type_hints, is_typeddict

from src import config
from src import sead4_reference as ref

SCHEMA_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADS_RUN = "modernbert-4096"
HEADS_DIR = PROJECT_ROOT / "results" / "heads" / HEADS_RUN
SCHEMA_DIR = PROJECT_ROOT / "results" / "schema"

GUIDELINE_CODES = tuple(config.GUIDELINE_CODES)
DEFENSIBLE = tuple("BCDEFGHJ")
TIERS = {code: "defensible" if code in DEFENSIBLE else "caveat" for code in GUIDELINE_CODES}
COMPOSE_RULE = "DENIED when any guideline is predicted AGAINST, otherwise GRANTED"
SECTIONS = (
    "synopsis",
    "statement_of_the_case",
    "findings_of_fact",
    "policies",
    "analysis",
    "conclusions",
    "formal_findings",
    "decision",
    "discussion",
    "order",
)
PROBABILITY_TOLERANCE = 1e-6

Status = Literal["NOT_AT_ISSUE", "FOR", "AGAINST"]
Outcome = Literal["DENIED", "GRANTED"]
Tier = Literal["defensible", "caveat"]
Source = Literal["rule", "llm"]
Kind = Literal["disqualifier", "mitigator"]
Regime = Literal["sead4", "directive", "unknown"]


class CaseInfo(TypedDict):
    source_url: str
    case_number: str
    date: str | None          # ISO date of the decision, from `date`, never `case_year`
    regime: Regime
    split: str | None         # train, val or test for a pool case, null for new text
    text_length: int          # len(full_text); every offset in the document indexes it


class Guideline(TypedDict):
    code: str                 # A to M
    name: str
    tier: Tier                # defensible B C D E F G H J, caveat A I K L M
    p_not_at_issue: float
    p_for: float
    p_against: float
    at_issue_threshold: float  # at issue when p_for + p_against reaches it
    status: Status
    model: str                # heads run, e.g. modernbert-4096
    calibration: str          # calibration file and its hash


class DirectOutcome(TypedDict):
    outcome: Outcome
    model: str


class Recommendation(TypedDict):
    composed: Outcome
    driven_by: list[str]      # guideline codes predicted AGAINST
    rule: str
    direct: DirectOutcome | None


class Condition(TypedDict):
    id: str                   # c1, c2, ...
    code: str                 # closed-list SEAD-4 code, e.g. AG ¶ 20(b)
    guideline: str
    kind: Kind
    text: str                 # the condition's wording from sead4_reference
    source: Source
    quote: str                # exactly full_text[start:end]; for a rule hit it is the
                              # citation itself, e.g. "AG ¶ 20(b)" or "(b)" in a list
    start: int
    end: int
    section: str | None       # the section the quote starts in
    cited_as: str | None      # the Directive-era code when the decision cites one


class Precedent(TypedDict):
    id: str                   # p1, p2, ...
    case_number: str
    source_url: str
    score: float              # retrieval score, not a probability
    outcome: str
    guidelines: list[str]
    label_source: str
    start: int                # best paragraph, offsets into that case's full_text
    end: int
    note: str | None          # LLM similarity note, null until written


class Rationale(TypedDict):
    text: str
    condition_ids: list[str]
    guideline_codes: list[str]


class Provenance(TypedDict):
    schema_version: str
    heads_model: str
    calibration: str
    extractor: str
    llm_model: str | None
    prompt_hashes: dict[str, str]
    run_dates: dict[str, str]
    verify_counts: dict[str, int]   # kept plus one entry per drop reason
    unmapped_hits: dict[str, int]   # hits left out because no SEAD-4 code exists, by source


class Document(TypedDict):
    schema_version: str
    case: CaseInfo
    guidelines: list[Guideline]
    recommendation: Recommendation
    conditions: list[Condition]
    precedents: list[Precedent]
    rationale: Rationale | None
    provenance: Provenance


class DocumentError(ValueError):
    """A document that breaks the schema. `problems` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        super().__init__(f"{len(problems)} problem(s):\n  {shown}{more}")


# --- JSON Schema, generated from the TypedDicts ---------------------------------------


def _is_union(hint) -> bool:
    return get_origin(hint) in (Union, types.UnionType)


def _schema_for(hint, defs: dict[str, dict]) -> dict:
    if hint is str:
        return {"type": "string"}
    if hint is bool:
        return {"type": "boolean"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is type(None):
        return {"type": "null"}
    origin = get_origin(hint)
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(hint))}
    if _is_union(hint):
        return {"anyOf": [_schema_for(arg, defs) for arg in get_args(hint)]}
    if origin is list:
        return {"type": "array", "items": _schema_for(get_args(hint)[0], defs)}
    if origin is dict:
        return {"type": "object", "additionalProperties": _schema_for(get_args(hint)[1], defs)}
    if is_typeddict(hint):
        name = hint.__name__
        if name not in defs:
            defs[name] = {}
            fields = get_type_hints(hint)
            defs[name] = {
                "type": "object",
                "properties": {key: _schema_for(value, defs) for key, value in fields.items()},
                "required": list(fields),
                "additionalProperties": False,
            }
        return {"$ref": f"#/$defs/{name}"}
    raise TypeError(f"no JSON Schema form for {hint!r}")


def json_schema() -> dict:
    """JSON Schema, draft 2020-12, for a `Document`."""
    defs: dict[str, dict] = {}
    _schema_for(Document, defs)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SEAD-4 Adjudication Analyzer document",
        "description": f"Schema version {SCHEMA_VERSION}. Generated from src/schema.py.",
        **defs.pop("Document"),
        "$defs": defs,
    }


# --- structural check, the same walk over the TypedDicts ------------------------------


def _check_type(value, hint, path: str, problems: list[str]) -> None:
    if hint is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected a string, got {type(value).__name__}")
        return
    if hint is bool:
        if not isinstance(value, bool):
            problems.append(f"{path}: expected true or false, got {type(value).__name__}")
        return
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an integer, got {type(value).__name__}")
        return
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, got {type(value).__name__}")
        elif not math.isfinite(value):
            problems.append(f"{path}: {value} is not a finite number")
        return
    if hint is type(None):
        if value is not None:
            problems.append(f"{path}: expected null")
        return
    origin = get_origin(hint)
    if origin is Literal:
        if value not in get_args(hint):
            problems.append(f"{path}: {value!r} is not one of {list(get_args(hint))}")
        return
    if _is_union(hint):
        args = get_args(hint)
        if value is None and type(None) in args:
            return
        options = [arg for arg in args if arg is not type(None)]
        if len(options) == 1:
            _check_type(value, options[0], path, problems)
            return
        for option in options:
            trial: list[str] = []
            _check_type(value, option, path, trial)
            if not trial:
                return
        problems.append(f"{path}: matches none of {options}")
        return
    if origin is list:
        if not isinstance(value, list):
            problems.append(f"{path}: expected a list, got {type(value).__name__}")
            return
        item = get_args(hint)[0]
        for i, element in enumerate(value):
            _check_type(element, item, f"{path}[{i}]", problems)
        return
    if origin is dict:
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object, got {type(value).__name__}")
            return
        item = get_args(hint)[1]
        for key, element in value.items():
            if not isinstance(key, str):
                problems.append(f"{path}: key {key!r} is not a string")
            _check_type(element, item, f"{path}.{key}", problems)
        return
    if is_typeddict(hint):
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object, got {type(value).__name__}")
            return
        fields = get_type_hints(hint)
        for key in fields:
            if key not in value:
                problems.append(f"{path}: missing {key!r}")
        for key in value:
            if key not in fields:
                problems.append(f"{path}: unexpected key {key!r}")
        for key, field_hint in fields.items():
            if key in value:
                _check_type(value[key], field_hint, f"{path}.{key}", problems)
        return
    raise TypeError(f"cannot check {hint!r}")


# --- decoding, the same rule as the 04 baseline's decode_predictions ------------------


def decode_status(p_for: float, p_against: float, threshold: float) -> str:
    """At issue when p_for + p_against reaches the threshold; then the likelier direction."""
    if p_for + p_against < threshold:
        return "NOT_AT_ISSUE"
    return "FOR" if p_for >= p_against else "AGAINST"


def compose(statuses: Mapping[str, str]) -> tuple[str, list[str]]:
    against = [code for code in GUIDELINE_CODES if statuses.get(code) == "AGAINST"]
    return ("DENIED" if against else "GRANTED"), against


# --- validate --------------------------------------------------------------------------


def problems(doc, full_text: str) -> list[str]:
    """Every way `doc` breaks the schema for this `full_text`. Empty means valid."""
    found: list[str] = []
    _check_type(doc, Document, "doc", found)
    if found:
        return found

    if doc["schema_version"] != SCHEMA_VERSION:
        found.append(f"schema_version {doc['schema_version']!r}, expected {SCHEMA_VERSION!r}")
    if doc["provenance"]["schema_version"] != doc["schema_version"]:
        found.append("provenance.schema_version differs from schema_version")
    if doc["case"]["text_length"] != len(full_text):
        found.append(f"case.text_length is {doc['case']['text_length']}, full_text has {len(full_text)}")

    codes = [entry["code"] for entry in doc["guidelines"]]
    if codes != list(GUIDELINE_CODES):
        missing = sorted(set(GUIDELINE_CODES) - set(codes))
        extra = sorted(set(codes) - set(GUIDELINE_CODES))
        repeated = sorted({code for code in codes if codes.count(code) > 1})
        found.append(
            f"guidelines must be the 13 codes A to M in order; missing {missing}, "
            f"unknown {extra}, repeated {repeated}"
        )
    statuses = {}
    for i, entry in enumerate(doc["guidelines"]):
        where = f"guidelines[{i}] {entry['code']}"
        code = entry["code"]
        if code in TIERS:
            if entry["name"] != config.GUIDELINE_NAMES[code]:
                found.append(f"{where}: name {entry['name']!r}, expected {config.GUIDELINE_NAMES[code]!r}")
            if entry["tier"] != TIERS[code]:
                found.append(f"{where}: tier {entry['tier']!r}, expected {TIERS[code]!r}")
        probabilities = (entry["p_not_at_issue"], entry["p_for"], entry["p_against"])
        if any(p < 0 or p > 1 for p in probabilities):
            found.append(f"{where}: probabilities must lie in [0, 1], got {probabilities}")
        total = sum(probabilities)
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            found.append(f"{where}: probabilities sum to {total:.8f}, not 1")
        expected = decode_status(entry["p_for"], entry["p_against"], entry["at_issue_threshold"])
        if entry["status"] != expected:
            found.append(f"{where}: status {entry['status']}, but the probabilities and threshold give {expected}")
        statuses[code] = entry["status"]

    outcome, against = compose(statuses)
    recommendation = doc["recommendation"]
    if recommendation["composed"] != outcome:
        found.append(f"recommendation.composed is {recommendation['composed']}, the statuses give {outcome}")
    if recommendation["driven_by"] != against:
        found.append(f"recommendation.driven_by is {recommendation['driven_by']}, expected {against}")

    ids: set[str] = set()
    for i, item in enumerate(doc["conditions"]):
        where = f"conditions[{i}] {item['id']}"
        if item["id"] in ids:
            found.append(f"{where}: repeated id")
        ids.add(item["id"])
        record = ref.CONDITIONS.get(item["code"])
        if record is None:
            found.append(f"{where}: {item['code']!r} is not on the closed SEAD-4 list")
        else:
            if item["guideline"] != record["guideline"]:
                found.append(f"{where}: guideline {item['guideline']}, the code belongs to {record['guideline']}")
            if item["kind"] != record["kind"]:
                found.append(f"{where}: kind {item['kind']}, the code is a {record['kind']}")
            if item["text"] != record["text"]:
                found.append(f"{where}: text is not the closed-list wording")
        start, end = item["start"], item["end"]
        if not 0 <= start < end <= len(full_text):
            found.append(f"{where}: offsets {start}:{end} fall outside full_text of length {len(full_text)}")
        elif full_text[start:end] != item["quote"]:
            found.append(f"{where}: full_text[{start}:{end}] is not the quote")
        if item["section"] is not None and item["section"] not in SECTIONS:
            found.append(f"{where}: unknown section {item['section']!r}")

    precedent_ids: set[str] = set()
    for i, item in enumerate(doc["precedents"]):
        where = f"precedents[{i}] {item['id']}"
        if item["id"] in precedent_ids:
            found.append(f"{where}: repeated id")
        precedent_ids.add(item["id"])
        if not 0 <= item["start"] <= item["end"]:
            found.append(f"{where}: offsets {item['start']}:{item['end']} are not a span")
        unknown = [code for code in item["guidelines"] if code not in TIERS]
        if unknown:
            found.append(f"{where}: unknown guidelines {unknown}")

    rationale = doc["rationale"]
    if rationale is not None:
        if not rationale["text"].strip():
            found.append("rationale.text is empty")
        missing_ids = [cid for cid in rationale["condition_ids"] if cid not in ids]
        if missing_ids:
            found.append(f"rationale cites conditions that are not in the document: {missing_ids}")
        unknown = [code for code in rationale["guideline_codes"] if code not in TIERS]
        if unknown:
            found.append(f"rationale cites unknown guidelines {unknown}")
    return found


def validate(doc, full_text: str) -> None:
    """Raise `DocumentError` listing every problem, or return None for a valid document."""
    found = problems(doc, full_text)
    if found:
        raise DocumentError(found)


# --- build -----------------------------------------------------------------------------


def _get(row, name: str, default=None):
    if isinstance(row, Mapping):
        value = row.get(name, default)
    else:
        try:
            value = row[name]
        except (KeyError, IndexError, TypeError):
            value = getattr(row, name, default)
    return value


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _iso_date(value) -> str | None:
    if _missing(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _sections(row) -> list[tuple[str, int, int]]:
    spans = []
    for name in SECTIONS:
        start, end = _get(row, f"sec_{name}_start"), _get(row, f"sec_{name}_end")
        if _missing(start) or _missing(end):
            continue
        start, end = int(start), int(end)
        if end > start:
            spans.append((name, start, end))
    return spans


def _section_of(position: int, spans: list[tuple[str, int, int]]) -> str | None:
    for name, start, end in spans:
        if start <= position < end:
            return name
    return None


def _as_mapping(hit) -> Mapping:
    if dataclasses.is_dataclass(hit) and not isinstance(hit, type):
        return dataclasses.asdict(hit)
    return hit


def _conditions(full_text: str, row, hits: Iterable, source: str) -> tuple[list[dict], int]:
    """Closed-list conditions from extractor or `verify` hits, and how many had no code."""
    spans = _sections(row)
    out, unmapped = [], 0
    for raw in hits or []:
        hit = _as_mapping(raw)
        code = hit.get("code") or hit.get("condition_code")
        cited_as = None
        if code not in ref.CONDITIONS:
            mapped = hit.get("sead4")
            if mapped not in ref.CONDITIONS:
                unmapped += 1
                continue
            cited_as, code = code, mapped
        start, end = int(hit["start"]), int(hit["end"])
        quote = full_text[start:end]
        given = hit.get("quote", hit.get("text"))
        if given is not None and given != quote:
            raise ValueError(f"{source} hit {code} at {start}:{end} does not quote full_text there")
        record = ref.CONDITIONS[code]
        out.append({
            "id": "",
            "code": code,
            "guideline": record["guideline"],
            "kind": record["kind"],
            "text": record["text"],
            "source": source,
            "quote": quote,
            "start": start,
            "end": end,
            "section": _section_of(start, spans),
            "cited_as": cited_as,
        })
    return out, unmapped


def _precedents(items: Iterable | None) -> list[dict]:
    out = []
    for i, raw in enumerate(items or [], start=1):
        item = _as_mapping(raw)
        note = item.get("note")
        out.append({
            "id": f"p{i}",
            "case_number": str(item["case_number"]),
            "source_url": str(item["source_url"]),
            "score": float(item["score"]),
            "outcome": str(item["outcome"]),
            "guidelines": [str(code) for code in list(item.get("guidelines", []))],
            "label_source": str(item.get("label_source", "")),
            "start": int(item["start"]),
            "end": int(item["end"]),
            "note": None if _missing(note) else str(note),
        })
    return out


def build_document(
    row,
    head_probs: Mapping,
    rule_hits: Iterable | None,
    llm_hits: Iterable | None,
    precedents: Iterable | None = None,
    rationale: Mapping | None = None,
    *,
    direct: Mapping | None = None,
    provenance: Mapping | None = None,
) -> dict:
    """One document for one case. Missing parts stay empty; the result passes `validate`.

    `row` is a `load_cases()` row, or any mapping with `full_text` and the case
    columns. `head_probs` is what `load_head_outputs` returns: `model`,
    `calibration`, and per guideline code `p_not_at_issue`, `p_for`, `p_against`
    and `at_issue_threshold`. `rule_hits` come from `conditions.extract_case` and
    `llm_hits` from `verify`'s kept quotes, as `VerifiedQuote` or as dicts with
    `code`, `start` and `end`. Directive-era rule hits use their `sead4` code and
    keep the original in `cited_as`; hits with no SEAD-4 code are counted in
    `provenance.unmapped_hits` and left out.
    """
    full_text = _get(row, "full_text") or ""
    regime = _get(row, "guideline_regime")
    split = _get(row, "split")
    case = {
        "source_url": str(_get(row, "source_url", "")),
        "case_number": str(_get(row, "case_number", "")),
        "date": _iso_date(_get(row, "date")),
        "regime": "unknown" if _missing(regime) else str(regime),
        "split": None if _missing(split) else str(split),
        "text_length": len(full_text),
    }

    guidelines, statuses = [], {}
    for code in GUIDELINE_CODES:
        probs = head_probs["guidelines"][code]
        p_none, p_for, p_against = (float(probs[k]) for k in ("p_not_at_issue", "p_for", "p_against"))
        threshold = float(probs["at_issue_threshold"])
        status = decode_status(p_for, p_against, threshold)
        statuses[code] = status
        guidelines.append({
            "code": code,
            "name": config.GUIDELINE_NAMES[code],
            "tier": TIERS[code],
            "p_not_at_issue": p_none,
            "p_for": p_for,
            "p_against": p_against,
            "at_issue_threshold": threshold,
            "status": status,
            "model": str(head_probs["model"]),
            "calibration": str(head_probs["calibration"]),
        })
    outcome, against = compose(statuses)

    rule, rule_unmapped = _conditions(full_text, row, rule_hits, "rule")
    llm, llm_unmapped = _conditions(full_text, row, llm_hits, "llm")
    conditions = sorted(rule + llm, key=lambda c: (c["start"], c["end"], c["source"], c["code"]))
    for i, item in enumerate(conditions, start=1):
        item["id"] = f"c{i}"

    meta = {
        "schema_version": SCHEMA_VERSION,
        "heads_model": str(head_probs["model"]),
        "calibration": str(head_probs["calibration"]),
        "extractor": "src.conditions.extract_case",
        "llm_model": None,
        "prompt_hashes": {},
        "run_dates": {},
        "verify_counts": {},
        "unmapped_hits": {"rule": rule_unmapped, "llm": llm_unmapped},
    }
    for key, value in (provenance or {}).items():
        if key not in meta:
            raise KeyError(f"unknown provenance field {key!r}")
        meta[key] = dict(value) if isinstance(value, Mapping) else value

    return {
        "schema_version": SCHEMA_VERSION,
        "case": case,
        "guidelines": guidelines,
        "recommendation": {
            "composed": outcome,
            "driven_by": against,
            "rule": COMPOSE_RULE,
            "direct": None if direct is None else {"outcome": str(direct["outcome"]), "model": str(direct["model"])},
        },
        "conditions": conditions,
        "precedents": _precedents(precedents),
        "rationale": None if rationale is None else {
            "text": str(rationale["text"]),
            "condition_ids": [str(cid) for cid in rationale.get("condition_ids", [])],
            "guideline_codes": [str(code) for code in rationale.get("guideline_codes", [])],
        },
        "provenance": meta,
    }


# --- published head outputs ------------------------------------------------------------


@lru_cache(maxsize=4)
def _published(heads_dir: str, split: str):
    import pandas as pd

    frame = pd.read_parquet(Path(heads_dir) / f"{split}_probabilities.parquet")
    frame = frame.set_index("source_url")
    if not frame.index.is_unique:
        raise ValueError(f"{split}_probabilities.parquet repeats a source_url")
    return frame


def load_head_outputs(source_url: str, split: str, heads_dir: Path = HEADS_DIR) -> dict:
    """The published head probabilities and thresholds for one val or test case."""
    heads_dir = Path(heads_dir)
    calibration_path = heads_dir / "calibration.json"
    raw = calibration_path.read_bytes()
    calibration = json.loads(raw)
    frame = _published(str(heads_dir), split)
    if source_url not in frame.index:
        raise KeyError(f"{source_url} is not in the published {split} head outputs")
    row = frame.loc[source_url]
    return {
        "model": heads_dir.name,
        "calibration": f"results/heads/{heads_dir.name}/calibration.json sha256:{hashlib.sha256(raw).hexdigest()[:12]}",
        "guidelines": {
            code: {
                "p_not_at_issue": float(row[f"p_not_at_issue_{code}"]),
                "p_for": float(row[f"p_for_{code}"]),
                "p_against": float(row[f"p_against_{code}"]),
                "at_issue_threshold": float(calibration[code]["at_issue_threshold"]),
            }
            for code in GUIDELINE_CODES
        },
    }


# --- command line ----------------------------------------------------------------------

CASE_COLUMNS = [
    "case_number",
    "source_url",
    "date",
    "guideline_regime",
    "guidelines",
    "outcome",
    "split",
    "full_text",
    *[f"sec_{name}_{edge}" for name in SECTIONS for edge in ("start", "end")],
]


def _pick_example(df) -> int:
    """The first test case, by source_url, that shows each part of the document.

    sead4 regime, Guideline F at issue with at least two guidelines in all, rule
    hits of both kinds, and at least one guideline predicted AGAINST and one FOR.
    """
    from src import conditions as extractor

    df = df.sort_values("source_url")
    for index, row in df.iterrows():
        if row["guideline_regime"] != "sead4" or "F" not in list(row["guidelines"]) or len(row["guidelines"]) < 2:
            continue
        kinds = {hit["kind"] for hit in extractor.extract_case(row) if hit["code"] in ref.CONDITIONS}
        if kinds != {"disqualifier", "mitigator"}:
            continue
        heads = load_head_outputs(row["source_url"], "test")
        statuses = {
            code: decode_status(p["p_for"], p["p_against"], p["at_issue_threshold"])
            for code, p in heads["guidelines"].items()
        }
        if {"FOR", "AGAINST"} <= set(statuses.values()):
            return index
    raise LookupError("no test case shows every part of the document")


def write_example(out_dir: Path = SCHEMA_DIR) -> Path:
    from src import conditions as extractor
    from src.dataset import load_cases

    df = load_cases(pool=True, columns=CASE_COLUMNS)
    df = df.loc[df["split"] == "test"]
    row = df.loc[_pick_example(df)]
    doc = build_document(
        row,
        load_head_outputs(row["source_url"], "test"),
        extractor.extract_case(row),
        llm_hits=[],
    )
    validate(doc, row["full_text"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"example_{row['case_number']}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, help="write the JSON Schema to this path")
    parser.add_argument("--example", action="store_true", help="write the example document for a test case")
    args = parser.parse_args(argv)
    if not args.export and not args.example:
        parser.print_help()
        return 1
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.export}")
    if args.example:
        print(f"wrote {write_example()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
