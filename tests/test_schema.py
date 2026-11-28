"""Tests for src/schema.py, the structured SEAD-4 document.

    python tests/test_schema.py

Most tests are NEGATIVE: a shifted offset, a code off the closed list, a missing
guideline, probabilities that don't sum to 1, a status the probabilities don't
give and a rationale that cites a missing condition must each fail `validate`.
The last tests rebuild the committed example from the real data and the
published head outputs.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import parametrize, run_module  # noqa: E402
from src import schema as S  # noqa: E402
from src.llm.verify import VerifiedQuote  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS = (
    "ANALYSIS \nApplicant has delinquent debts. Two are potentially applicable: "
    "AG ¶ 19(a) and AG ¶ 19(c) apply. "
    "He said the debts arose from a layoff in 2018. "
    "Under Directive ¶ E2.A6.1.3.2 the layoff counts. E2.A6.1.3.9 is not a real code. "
)
CONCLUSIONS = "CONCLUSIONS \nAG ¶ 20(b) applies in part. "
FULL_TEXT = "STATEMENT OF THE CASE \nThe DOD issued an SOR. \n" + ANALYSIS + CONCLUSIONS
A_START = FULL_TEXT.index("ANALYSIS")
C_START = FULL_TEXT.index("CONCLUSIONS")
LAYOFF = "the debts arose from a layoff in 2018"


def _at(text: str) -> tuple[int, int]:
    start = FULL_TEXT.index(text)
    return start, start + len(text)


def _row() -> dict:
    return {
        "case_number": "19-00001",
        "source_url": "https://example.test/FileId/1/",
        "date": "2020-01-15",
        "guideline_regime": "sead4",
        "split": "test",
        "full_text": FULL_TEXT,
        "sec_statement_of_the_case_start": 0,
        "sec_statement_of_the_case_end": A_START,
        "sec_analysis_start": A_START,
        "sec_analysis_end": C_START,
        "sec_conclusions_start": C_START,
        "sec_conclusions_end": len(FULL_TEXT),
        "sec_order_start": "",
        "sec_order_end": "",
    }


def _heads(f=(0.1, 0.2, 0.7), b=(0.2, 0.7, 0.1)) -> dict:
    guidelines = {code: {"p_not_at_issue": 0.98, "p_for": 0.01, "p_against": 0.01, "at_issue_threshold": 0.5}
                  for code in S.GUIDELINE_CODES}
    for code, probs in (("F", f), ("B", b)):
        guidelines[code] = dict(zip(("p_not_at_issue", "p_for", "p_against"), probs), at_issue_threshold=0.4)
    return {"model": "test-heads", "calibration": "test-calibration", "guidelines": guidelines}


def _rule_hits() -> list[dict]:
    hits = []
    for text in ("AG ¶ 19(a)", "AG ¶ 19(c)", "AG ¶ 20(b)"):
        start, end = _at(text)
        hits.append({"code": text, "kind": "x", "start": start, "end": end})
    start, end = _at("E2.A6.1.3.2")
    hits.append({"code": "E2.A6.1.3.2", "kind": "mitigator", "start": start, "end": end, "sead4": "AG ¶ 20(b)"})
    start, end = _at("E2.A6.1.3.9")
    hits.append({"code": "E2.A6.1.3.9", "kind": "mitigator", "start": start, "end": end})
    return hits


def _llm_hits() -> list:
    start, end = _at(LAYOFF)
    return [VerifiedQuote("AG ¶ 20(b)", start, end, LAYOFF, start, end, LAYOFF, 1)]


def _doc(**kwargs) -> dict:
    return S.build_document(_row(), _heads(), _rule_hits(), _llm_hits(), **kwargs)


def _fails(doc, needle: str) -> None:
    found = S.problems(doc, FULL_TEXT)
    assert any(needle in problem for problem in found), f"expected {needle!r} in {found}"
    try:
        S.validate(doc, FULL_TEXT)
    except S.DocumentError as error:
        assert error.problems == found
    else:
        raise AssertionError("validate accepted a broken document")


# --- a good document --------------------------------------------------------------------


def test_built_document_is_valid():
    doc = _doc(
        precedents=[{"case_number": "18-1", "source_url": "u", "score": 0.8, "outcome": "DENIED",
                     "guidelines": ["F"], "label_source": "formal_ruling", "start": 10, "end": 90}],
        rationale={"text": "Debts remain.", "condition_ids": ["c1"], "guideline_codes": ["F"]},
    )
    S.validate(doc, FULL_TEXT)
    json.dumps(doc)


def test_statuses_and_recommendation():
    doc = _doc()
    statuses = {g["code"]: g["status"] for g in doc["guidelines"]}
    assert statuses["F"] == "AGAINST" and statuses["B"] == "FOR"
    assert {statuses[c] for c in S.GUIDELINE_CODES if c not in "BF"} == {"NOT_AT_ISSUE"}
    assert doc["recommendation"]["composed"] == "DENIED"
    assert doc["recommendation"]["driven_by"] == ["F"]
    assert doc["recommendation"]["direct"] is None


def test_granted_when_nothing_is_against():
    doc = S.build_document(_row(), _heads(f=(0.1, 0.8, 0.1)), [], [])
    assert doc["recommendation"]["composed"] == "GRANTED"
    assert doc["recommendation"]["driven_by"] == []
    S.validate(doc, FULL_TEXT)


def test_tiers_follow_the_plan():
    doc = _doc()
    tiers = {g["code"]: g["tier"] for g in doc["guidelines"]}
    assert [c for c, t in tiers.items() if t == "defensible"] == list("BCDEFGHJ")
    assert [c for c, t in tiers.items() if t == "caveat"] == list("AIKLM")


def test_conditions_quote_offsets_sections_and_sources():
    doc = _doc()
    for item in doc["conditions"]:
        assert FULL_TEXT[item["start"]:item["end"]] == item["quote"]
    by_quote = {item["quote"]: item for item in doc["conditions"]}
    assert by_quote["AG ¶ 19(a)"]["section"] == "analysis"
    assert by_quote["AG ¶ 20(b)"]["section"] == "conclusions"
    assert by_quote[LAYOFF]["source"] == "llm"
    assert by_quote["AG ¶ 19(c)"]["source"] == "rule"
    assert by_quote["AG ¶ 19(a)"]["kind"] == "disqualifier"
    assert [item["id"] for item in doc["conditions"]] == [f"c{i}" for i in range(1, 6)]
    assert [item["start"] for item in doc["conditions"]] == sorted(item["start"] for item in doc["conditions"])


def test_context_and_lead_in():
    by_quote = {item["quote"]: item for item in _doc()["conditions"]}
    first = by_quote["AG ¶ 19(a)"]
    assert first["context"]["text"] == "AG ¶ 19(a) and AG ¶ 19(c) apply."
    assert first["lead_in"]["text"] == "Two are potentially applicable:"
    # a heading has no full stop, so it joins the sentence that follows it
    assert by_quote["AG ¶ 20(b)"]["context"]["text"] == "CONCLUSIONS \nAG ¶ 20(b) applies in part."
    assert by_quote["AG ¶ 20(b)"]["lead_in"] is None
    assert by_quote[LAYOFF]["context"]["text"] == "He said the debts arose from a layoff in 2018."
    for item in by_quote.values():
        span = item["context"]
        assert FULL_TEXT[span["start"]:span["end"]] == span["text"]
        assert span["start"] <= item["start"] and item["end"] <= span["end"]


def test_directive_hit_uses_its_sead4_code_and_unmapped_is_counted():
    doc = _doc()
    directive = [item for item in doc["conditions"] if item["cited_as"]]
    assert len(directive) == 1
    assert directive[0]["code"] == "AG ¶ 20(b)" and directive[0]["cited_as"] == "E2.A6.1.3.2"
    assert doc["provenance"]["unmapped_hits"] == {"rule": 1, "llm": 0}
    assert "E2.A6.1.3.9" not in {item["quote"] for item in doc["conditions"]}


def test_empty_parts_stay_empty():
    doc = S.build_document(_row(), _heads(), None, None)
    assert doc["conditions"] == [] and doc["precedents"] == [] and doc["rationale"] is None
    assert doc["provenance"]["llm_model"] is None
    S.validate(doc, FULL_TEXT)


def test_provenance_fields_are_set_and_unknown_ones_refused():
    doc = _doc(provenance={"llm_model": "claude-haiku-5-5", "verify_counts": {"kept": 1, "not_found": 2}})
    assert doc["provenance"]["llm_model"] == "claude-haiku-5-5"
    assert doc["provenance"]["verify_counts"] == {"kept": 1, "not_found": 2}
    try:
        _doc(provenance={"confidence": 0.9})
    except KeyError:
        return
    raise AssertionError("an unknown provenance field was accepted")


def test_llm_hit_that_does_not_quote_the_text_is_refused():
    start, end = _at(LAYOFF)
    bad = [{"code": "AG ¶ 20(b)", "start": start + 1, "end": end + 1, "quote": LAYOFF}]
    try:
        S.build_document(_row(), _heads(), [], bad)
    except ValueError:
        return
    raise AssertionError("a quote that is not full_text[start:end] was accepted")


def test_missing_date_and_regime():
    row = _row()
    row["date"], row["guideline_regime"], row["split"] = None, "", ""
    doc = S.build_document(row, _heads(), [], [])
    assert doc["case"]["date"] is None and doc["case"]["regime"] == "unknown" and doc["case"]["split"] is None
    S.validate(doc, FULL_TEXT)


# --- broken documents must fail -------------------------------------------------------


def test_shifted_offset_fails():
    doc = _doc()
    doc["conditions"][0]["start"] += 1
    doc["conditions"][0]["end"] += 1
    _fails(doc, "is not the quote")


def test_offset_past_the_text_fails():
    doc = _doc()
    doc["conditions"][-1]["end"] = len(FULL_TEXT) + 5
    _fails(doc, "fall outside full_text")


def test_context_that_misses_the_quote_or_the_text_fails():
    doc = _doc()
    doc["conditions"][0]["context"]["start"] = doc["conditions"][0]["end"]
    _fails(doc, "does not hold the quote")
    doc = _doc()
    doc["conditions"][0]["context"]["text"] += " extra"
    _fails(doc, "context text is not full_text")


def test_lead_in_after_the_context_or_with_wrong_text_fails():
    doc = _doc()
    item = next(c for c in doc["conditions"] if c["lead_in"])
    item["lead_in"]["end"] = item["context"]["start"] + 5
    _fails(doc, "must end before the context")
    doc = _doc()
    item = next(c for c in doc["conditions"] if c["lead_in"])
    item["lead_in"]["text"] = "Something else:"
    _fails(doc, "lead_in text is not full_text")


def test_off_list_code_fails():
    doc = _doc()
    doc["conditions"][0]["code"] = "AG ¶ 19(z)"
    _fails(doc, "not on the closed SEAD-4 list")


def test_wrong_condition_wording_fails():
    doc = _doc()
    doc["conditions"][0]["text"] = "inability to pay"
    _fails(doc, "closed-list wording")


def test_code_filed_under_the_wrong_guideline_fails():
    doc = _doc()
    doc["conditions"][0]["guideline"] = "E"
    _fails(doc, "the code belongs to F")


@parametrize("code", ["A", "F", "M"])
def test_missing_guideline_fails(code):
    doc = _doc()
    doc["guidelines"] = [g for g in doc["guidelines"] if g["code"] != code]
    _fails(doc, f"missing ['{code}']")


def test_repeated_guideline_fails():
    doc = _doc()
    doc["guidelines"][1] = copy.deepcopy(doc["guidelines"][0])
    _fails(doc, "repeated ['A']")


@parametrize("p", [((0.5, 0.2, 0.2),), ((0.6, 0.3, 0.3),), ((1.2, -0.1, -0.1),)])
def test_probabilities_not_summing_to_one_or_out_of_range_fail(p):
    doc = _doc()
    entry = doc["guidelines"][5]
    entry["p_not_at_issue"], entry["p_for"], entry["p_against"] = p
    found = S.problems(doc, FULL_TEXT)
    assert any("sum to" in x or "lie in [0, 1]" in x for x in found), found


def test_status_the_probabilities_do_not_give_fails():
    doc = _doc()
    doc["guidelines"][5]["status"] = "FOR"
    _fails(doc, "the probabilities and threshold give AGAINST")


def test_recommendation_that_does_not_follow_fails():
    doc = _doc()
    doc["recommendation"]["composed"] = "GRANTED"
    _fails(doc, "the statuses give DENIED")
    doc = _doc()
    doc["recommendation"]["driven_by"] = ["B", "F"]
    _fails(doc, "driven_by")


def test_rationale_citing_a_missing_condition_fails():
    doc = _doc(rationale={"text": "x", "condition_ids": ["c99"], "guideline_codes": ["F"]})
    _fails(doc, "not in the document: ['c99']")


def test_text_length_mismatch_fails():
    doc = _doc()
    doc["case"]["text_length"] -= 1
    _fails(doc, "case.text_length")


def test_wrong_types_and_keys_fail():
    doc = _doc()
    doc["conditions"][0]["start"] = str(doc["conditions"][0]["start"])
    _fails(doc, "expected an integer")
    doc = _doc()
    doc["guidelines"][0]["confidence"] = 0.9
    _fails(doc, "unexpected key 'confidence'")
    doc = _doc()
    del doc["provenance"]
    _fails(doc, "missing 'provenance'")
    doc = _doc()
    doc["guidelines"][0]["status"] = "MAYBE"
    _fails(doc, "is not one of")
    doc = _doc()
    doc["guidelines"][0]["p_for"] = True
    _fails(doc, "expected a number")


# --- JSON Schema export ---------------------------------------------------------------


def test_json_schema_matches_the_typeddicts():
    schema = S.json_schema()
    assert schema["required"] == list(S.Document.__annotations__)
    assert schema["additionalProperties"] is False
    for name in ("CaseInfo", "Guideline", "Recommendation", "Span", "Condition", "Precedent", "Rationale", "Provenance"):
        cls = getattr(S, name)
        assert schema["$defs"][name]["required"] == [k for k in cls.__annotations__ if k in cls.__required_keys__], name
    assert schema["$defs"]["Guideline"]["properties"]["status"]["enum"] == ["NOT_AT_ISSUE", "FOR", "AGAINST"]
    json.dumps(schema)


def test_committed_schema_file_is_current():
    committed = json.loads((ROOT / "results" / "schema" / "document.schema.json").read_text(encoding="utf-8"))
    assert committed == S.json_schema(), "run python -m src.schema --export results/schema/document.schema.json"


def test_jsonschema_library_agrees_when_installed():
    try:
        import jsonschema
    except ImportError:
        print("  jsonschema is not installed; skipped")
        return
    schema = S.json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(_doc(), schema)
    broken = _doc()
    del broken["guidelines"][0]["p_for"]
    try:
        jsonschema.validate(broken, schema)
    except jsonschema.ValidationError:
        return
    raise AssertionError("jsonschema accepted a guideline without p_for")


# --- real data -------------------------------------------------------------------------


def test_decode_matches_the_baseline_on_published_test_outputs():
    import numpy as np
    import pandas as pd

    from src.heads.data import baseline

    frame = pd.read_parquet(S.HEADS_DIR / "test_probabilities.parquet")
    calibration = json.loads((S.HEADS_DIR / "calibration.json").read_text(encoding="utf-8"))
    for code in S.GUIDELINE_CODES:
        probs = frame[[f"p_not_at_issue_{code}", f"p_for_{code}", f"p_against_{code}"]].to_numpy()
        threshold = calibration[code]["at_issue_threshold"]
        expected = baseline().decode_predictions(probs, threshold)
        ours = np.array([S.decode_status(p[1], p[2], threshold) for p in probs], dtype=object)
        assert (ours == expected).all(), code


def test_committed_example_rebuilds_and_validates():
    from src import conditions
    from src.dataset import load_cases

    paths = sorted((ROOT / "results" / "schema").glob("example_*.json"))
    assert len(paths) == 1, paths
    committed = json.loads(paths[0].read_text(encoding="utf-8"))
    url = committed["case"]["source_url"]
    df = load_cases(pool=True, columns=S.CASE_COLUMNS)
    row = df.loc[df["source_url"] == url].iloc[0]
    assert paths[0].name == f"example_{row['case_number']}.json"
    assert row["split"] == "test"
    S.validate(committed, row["full_text"])
    rebuilt = S.build_document(row, S.load_head_outputs(url, "test"), conditions.extract_case(row), [])
    assert rebuilt == committed
    assert committed["conditions"], "the example should show conditions"
    statuses = {g["status"] for g in committed["guidelines"]}
    assert {"FOR", "AGAINST"} <= statuses


if __name__ == "__main__":
    raise SystemExit(run_module(globals(), "schema tests"))
