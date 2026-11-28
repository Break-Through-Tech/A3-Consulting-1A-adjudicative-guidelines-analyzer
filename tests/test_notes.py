"""Tests for the rationale and similarity note templates and src/llm/notes.py. No network calls.

    python tests/test_notes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src.llm import notes as N  # noqa: E402
from src.llm import templates as T  # noqa: E402

S1 = "Applicant owes $20,000 on six credit cards and has not paid them since 2019."
S2 = "He started a payment plan in March 2023 and has paid two accounts. AG ¶ 20(d) applies."
FULL = "HEADER\n" + S1 + " " + S2 + "\n"
FACTS = "Applicant owes $20,000 on six credit cards. He is 45 and works as an engineer."
PARAGRAPH = "Applicant has $15,000 in delinquent credit card debt from a divorce."


def _span(needle: str) -> dict:
    start = FULL.index(needle)
    return {"start": start, "end": start + len(needle), "text": needle}


def _guideline(code: str, status: str) -> dict:
    p = {"NOT_AT_ISSUE": (0.9, 0.05, 0.05), "FOR": (0.1, 0.7, 0.2), "AGAINST": (0.1, 0.2, 0.7)}[status]
    return {"code": code, "name": f"Name {code}", "tier": "defensible", "p_not_at_issue": p[0], "p_for": p[1],
            "p_against": p[2], "at_issue_threshold": 0.5, "status": status, "model": "m", "calibration": "c"}


def _doc() -> dict:
    q1 = "Applicant owes $20,000 on six credit cards"
    c1 = {"id": "c1", "code": "AG ¶ 19(a)", "guideline": "F", "kind": "disqualifier",
          "text": "inability to satisfy debts;", "source": "llm", "quote": q1, "start": FULL.index(q1),
          "end": FULL.index(q1) + len(q1), "context": _span(S1), "lead_in": None, "section": "analysis",
          "cited_as": None}
    q2 = "AG ¶ 20(d)"
    c2 = {"id": "c2", "code": "AG ¶ 20(d)", "guideline": "F", "kind": "mitigator",
          "text": "the individual initiated and is adhering to a good-faith effort to repay", "source": "rule",
          "quote": q2, "start": FULL.index(q2), "end": FULL.index(q2) + len(q2), "context": _span(S2),
          "lead_in": None, "section": "analysis", "cited_as": None}
    guidelines = [_guideline(code, "AGAINST" if code == "F" else "FOR" if code == "E" else "NOT_AT_ISSUE")
                  for code in "ABCDEFGHIJKLM"]
    return {"guidelines": guidelines, "conditions": [c1, c2],
            "recommendation": {"composed": "DENIED", "driven_by": ["F"], "rule": "r", "direct": None}}


def _good() -> dict:
    return {"text": "The tool predicts Guideline F against the applicant, so it recommends denial. The decision "
                    "says \"Applicant owes $20,000 on six credit cards\". It also describes a payment plan, "
                    "which bears on mitigation.",
            "condition_ids": ["c1", "c2"], "guideline_codes": ["F"],
            "quotes": [{"condition_id": "c1", "exact_quote": "Applicant owes $20,000 on six credit cards"}]}


# --- templates -------------------------------------------------------------------

def test_rationale_input_has_statuses_and_conditions_but_no_head_numbers():
    text = T.rationale_input(_doc())
    assert "Guideline F, Name F: AGAINST" in text and "Guideline E, Name E: FOR" in text
    assert "Guideline A" not in text                      # not at issue, left out
    assert "Composed recommendation: DENIED" in text
    assert "[c1] AG ¶ 19(a)" in text and "[c2] AG ¶ 20(d)" in text
    assert "Evidence: Applicant owes" in text             # llm quote shown
    assert text.count("Evidence:") == 1                   # a rule hit's quote is only the citation
    for p in ("0.7", "0.2", "0.05", "0.5"):
        assert p not in text


def test_rationale_schema_limits_ids_and_codes_to_the_document():
    params = T.rationale_params(_doc(), model="claude-haiku-5-5")
    schema = params["output_config"]["format"]["schema"]
    assert schema["properties"]["condition_ids"]["items"]["enum"] == ["c1", "c2"]
    assert schema["properties"]["guideline_codes"]["items"]["enum"] == ["E", "F"]
    quote = schema["properties"]["quotes"]["items"]
    assert quote["properties"]["condition_id"]["enum"] == ["c1", "c2"] and quote["additionalProperties"] is False
    assert schema["required"] == ["text", "condition_ids", "guideline_codes", "quotes"]


def test_params_avoid_rejected_fields():
    for p in (T.rationale_params(_doc(), model="m"), T.note_params(FACTS, "15-00001", PARAGRAPH, model="m")):
        for bad in ("temperature", "top_p", "top_k", "thinking", "budget_tokens"):
            assert bad not in p
        assert p["max_tokens"] == 4000 and p["output_config"]["format"]["type"] == "json_schema"
        assert len(p["messages"]) == 1 and p["messages"][0]["role"] == "user"


def test_note_schema_is_one_string():
    assert T.NOTE_SCHEMA["required"] == ["note"] and set(T.NOTE_SCHEMA["properties"]) == {"note"}


def test_note_input_collapses_layout():
    text = T.note_input("A  b\nc", "15-00001", "d\n\ne")
    assert "A b c" in text and "Precedent 15-00001" in text and "d e" in text


def test_prompts_forbid_scores_and_outside_facts():
    for prompt in (T.RATIONALE_SYSTEM_PROMPT, T.NOTE_SYSTEM_PROMPT):
        assert "Add no outside facts" in prompt and "confidence" in prompt
    assert "not as the judge's ruling" in T.RATIONALE_SYSTEM_PROMPT
    assert "Do not mention outcomes" in T.NOTE_SYSTEM_PROMPT


# --- rationale checks --------------------------------------------------------------

def test_good_rationale_is_kept():
    rationale, problems = N.check_rationale(_good(), _doc(), FULL)
    assert problems == [], problems
    assert rationale["condition_ids"] == ["c1", "c2"] and rationale["guideline_codes"] == ["F"]


def test_unparsed_rationale_is_dropped():
    assert N.check_rationale(None, _doc(), FULL) == (None, ["unparsed"])


def _with(**changes) -> dict:
    answer = _good()
    answer.update(changes)
    return answer


def test_quote_outside_its_conditions_sentence_is_dropped():
    answer = _with(quotes=[{"condition_id": "c2", "exact_quote": "Applicant owes $20,000 on six credit cards"}])
    _, problems = N.check_rationale(answer, _doc(), FULL)
    assert "quote_outside_region" in problems, problems


def test_changed_quote_is_dropped():
    answer = _with(text="It says \"Applicant owes $25,000 on six credit cards\".",
                   quotes=[{"condition_id": "c1", "exact_quote": "Applicant owes $25,000 on six credit cards"}])
    rationale, problems = N.check_rationale(answer, _doc(), FULL)
    assert rationale is None and any(p.startswith("quote_") for p in problems), problems


def test_inline_quote_must_be_listed():
    answer = _with(text="The tool predicts F against. It says \"has paid two accounts\".")
    _, problems = N.check_rationale(answer, _doc(), FULL)
    assert "unlisted_quote" in problems, problems


def test_confidence_and_numbers_are_refused():
    for text, problem in (("The tool is 70% sure Guideline F is against.", "states_confidence"),
                          ("The probability for Guideline F is high.", "states_confidence"),
                          ("Guideline F has a score of 0.7.", "states_confidence"),
                          ("He owes on 9 cards, so Guideline F is predicted against.", "unsupported_number")):
        _, problems = N.check_rationale(_with(text=text, quotes=[]), _doc(), FULL)
        assert problem in problems, (text, problems)


def test_numbers_from_the_evidence_are_allowed_outside_quotes():
    answer = _with(text="The decision describes a payment plan from March 2023, so the tool still predicts "
                        "Guideline F against.", quotes=[])
    assert N.check_rationale(answer, _doc(), FULL)[1] == []


def test_ids_codes_and_guidelines_are_checked():
    _, problems = N.check_rationale(_with(condition_ids=["c9"]), _doc(), FULL)
    assert "bad_condition_ids" in problems
    _, problems = N.check_rationale(_with(guideline_codes=["A"]), _doc(), FULL)
    assert "bad_guideline_codes" in problems                 # A is not at issue
    _, problems = N.check_rationale(_with(condition_ids=["c2"]), _doc(), FULL)
    assert "quote_from_uncited_condition" in problems
    _, problems = N.check_rationale(_with(text="Condition c1 and AG ¶ 19(a) apply.", quotes=[]), _doc(), FULL)
    assert "names_ids_or_codes" in problems


def test_rationale_that_credits_the_tool_with_the_judges_reasoning_is_dropped():
    for text in ("The tool judged that the repayment efforts were not sustained.",
                 "The tool concluded the debts remain unpaid, so Guideline F is predicted against."):
        assert "credits_the_tool" in N.check_rationale(_with(text=text, quotes=[]), _doc(), FULL)[1], text
    ok = _with(text="The tool predicts Guideline F against. The judge concluded the plan was too recent.", quotes=[])
    assert N.check_rationale(ok, _doc(), FULL)[1] == []


def test_prompt_credits_the_judge_and_denies_the_tool_reads_conditions():
    assert "never to the tool" in T.RATIONALE_SYSTEM_PROMPT
    assert "It does not read these conditions" in T.RATIONALE_SYSTEM_PROMPT
    assert "points the same way as the prediction or the other way" in T.RATIONALE_SYSTEM_PROMPT


def test_rationale_that_says_the_prediction_uses_the_conditions_is_dropped():
    for text in ("The tool's prediction weighs these mitigators against the disqualifiers.",
                 "The tool's predictions reflect where these discussions appear.",
                 "The tool's Guideline F prediction rests on the debts."):
        assert "credits_the_tool" in N.check_rationale(_with(text=text, quotes=[]), _doc(), FULL)[1], text
    ok = _with(text="The tool predicts Guideline F against, and the judge's discussion points the same way.",
               quotes=[])
    assert N.check_rationale(ok, _doc(), FULL)[1] == []


def test_a_piece_of_a_listed_quote_may_appear_in_the_text():
    answer = _with(text="The tool predicts Guideline F against. The decision says he \"owes $20,000 on six credit cards\".")
    assert N.check_rationale(answer, _doc(), FULL)[1] == []


def test_rationale_for_a_guideline_without_conditions_is_dropped():
    # Guideline E is predicted FOR in the test document but has no condition.
    _, problems = N.check_rationale(_with(guideline_codes=["E", "F"]), _doc(), FULL)
    assert "guideline_without_conditions" in problems, problems
    for text in ("The tool predicts Guideline E and Guideline F. The judge discussed the payment plan.",
                 "The tool predicts Guidelines E and F. The judge discussed the payment plan.",
                 "The tool predicts Name E and Guideline F. The judge discussed the payment plan."):
        _, problems = N.check_rationale(_with(text=text, quotes=[]), _doc(), FULL)
        assert "guideline_without_conditions" in problems, (text, problems)


def test_rationale_for_guidelines_with_conditions_still_passes():
    ok = _with(text="The tool predicts Guideline F, Name F, against the applicant. The judge discussed a "
                    "payment plan.", quotes=[])
    assert N.check_rationale(ok, _doc(), FULL)[1] == []


def test_a_guideline_named_only_inside_a_quote_is_not_flagged():
    answer = _with(text="The tool predicts Guideline F against. The decision says \"Applicant owes $20,000 on "
                        "six credit cards\". Quoted aside: \"Guideline E, Name E\".")
    _, problems = N.check_rationale(answer, _doc(), FULL)
    assert "guideline_without_conditions" not in problems, problems


def test_demo_rationale_citing_only_f_conditions_for_e_and_f_is_dropped():
    import json
    demo = json.loads((Path(__file__).resolve().parents[1] / "results" / "demo" / "01_14-01234.json")
                      .read_text(encoding="utf-8"))
    row = N.load_rows({demo["case"]["source_url"]})[0]
    answer = dict(demo["rationale"], quotes=[])
    assert answer["guideline_codes"] == ["E", "F"] and {c["guideline"] for c in demo["conditions"]} == {"F"}
    rationale, problems = N.check_rationale(answer, demo, row["full_text"])
    assert rationale is None and "guideline_without_conditions" in problems, problems


def test_rationales_batch_resends_only_rationales():
    reqs = [{"custom_id": "rat-0", "batch": "test"}, {"custom_id": "note-0-1", "batch": "test"},
            {"custom_id": "rat-1", "batch": "full"}, {"custom_id": "note-1-1", "batch": "full"}]
    assert [r["custom_id"] for r in N.members("rationales", reqs)] == ["rat-0", "rat-1"]
    assert [r["custom_id"] for r in N.members("full", reqs)] == ["rat-1", "note-1-1"]


def test_certain_as_an_ordinary_word_is_allowed():
    note = {"note": "Both applicants owe credit card debt, and an ex-spouse was ordered to pay certain debts."}
    assert N.check_note(note, FACTS, PARAGRAPH)[1] == []
    assert "states_score" in N.check_note({"note": "With some certainty, both owe card debt."}, FACTS, PARAGRAPH)[1]


def test_long_rationale_is_dropped():
    _, problems = N.check_rationale(_with(text="word " * 200, quotes=[]), _doc(), FULL)
    assert "too_long" in problems


# --- note checks -------------------------------------------------------------------

def test_good_note_is_kept():
    note, problems = N.check_note({"note": "Both applicants carry delinquent credit card debt."}, FACTS, PARAGRAPH)
    assert problems == [] and note.startswith("Both")


def test_note_numbers_must_come_from_the_texts():
    ok = {"note": "Both owe credit card debt, $20,000 here and $15,000 in the precedent."}
    assert N.check_note(ok, FACTS, PARAGRAPH)[1] == []
    bad = {"note": "Both owe about $30,000 in credit card debt."}
    assert "unsupported_number" in N.check_note(bad, FACTS, PARAGRAPH)[1]


def test_note_with_a_score_or_relevance_is_dropped():
    for text in ("This precedent is highly relevant.", "Similarity is strong: both owe card debt.",
                 "The facts match at 80%."):
        assert "states_score" in N.check_note({"note": text}, FACTS, PARAGRAPH)[1], text


def test_note_too_long_or_unparsed():
    assert "too_long" in N.check_note({"note": "debt " * 100}, FACTS, PARAGRAPH)[1]
    assert N.check_note({"text": "x"}, FACTS, PARAGRAPH) == (None, ["unparsed"])


# --- demo set ----------------------------------------------------------------------

def _rows(n: int = 30) -> tuple[list[dict], dict]:
    rows, kept = [], {}
    for i in range(n):
        url = f"https://x/{i:03d}"
        rows.append({"source_url": url, "guideline_regime": "sead4" if i % 7 else "unknown",
                     "outcome": "DENIED" if i % 2 else "GRANTED",
                     "sec_findings_of_fact_start": 100, "sec_findings_of_fact_end": 100 + 3000 + i * 10})
        kept[url] = [{}] * (2 if i % 5 == 0 else 6)
    return rows, kept


def test_demo_set_is_stratified_eligible_and_stable():
    rows, kept = _rows()
    demo = N.select_demo(rows, kept)
    assert [r["outcome"] for r in demo] == ["DENIED"] * 5 + ["GRANTED"] * 5
    assert all(r["guideline_regime"] == "sead4" and len(kept[r["source_url"]]) >= N.MIN_KEPT for r in demo)
    again = N.select_demo(list(reversed(rows)), kept)
    assert [r["source_url"] for r in again] == [r["source_url"] for r in demo]


def test_demo_set_skips_short_or_missing_facts():
    rows, kept = _rows()
    for r in rows:
        r["sec_findings_of_fact_end"] = r["sec_findings_of_fact_start"] + 500
    try:
        N.select_demo(rows, kept)
    except LookupError:
        pass
    else:
        raise AssertionError("short findings of fact were accepted")
    assert N.facts_span({"sec_findings_of_fact_start": float("nan"), "sec_findings_of_fact_end": 9}) is None
    assert N.facts_span({"sec_findings_of_fact_start": "", "sec_findings_of_fact_end": 9}) is None


def test_only_the_first_case_is_in_the_test_batch():
    assert N.batch_of(0) == "test" and {N.batch_of(i) for i in range(1, 10)} == {"full"}
    assert N.rationale_id(3) == "rat-3" and N.note_id(3, 5) == "note-3-5"


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "notes tests"))
