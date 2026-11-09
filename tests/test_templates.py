"""Tests for src/llm/templates.py. No network calls.

    python tests/test_templates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src.llm import templates as T  # noqa: E402
from src.llm.verify import verify_guideline_quotes  # noqa: E402

ANALYSIS = (
    "Applicant has $40,000 in delinquent credit cards and did not pay them after he\n"
    "lost his job. AG ¶ 19(a) applies. He had an inability to satisfy debts for\n"
    "three years. AG ¶¶ 20(a), (b), and (c) are discussed next."
)
FINDINGS = "FORMAL FINDINGS\n\nFor Applicant: Subparagraph 1.a: Against Applicant"
CONCLUSIONS = "I considered the whole person. Clearance is denied."


def _row() -> dict:
    full = "HEADER\n" + ANALYSIS + "\n" + FINDINGS + "\n" + CONCLUSIONS
    a0 = full.index(ANALYSIS)
    c0 = full.index(CONCLUSIONS)
    return {"full_text": full, "guidelines": ["F"], "date": None, "guideline_regime": "sead4",
            "sec_analysis_start": a0, "sec_analysis_end": a0 + len(ANALYSIS),
            "sec_conclusions_start": c0, "sec_conclusions_end": c0 + len(CONCLUSIONS),
            "sec_policies_start": None, "sec_policies_end": None}


def test_list_citation_is_one_mask():
    out = T.mask_section("See AG ¶¶ 20(a), (b), and (c) here.", "citation")
    assert out == f"See {T.MASK_TOKEN} here.", out


def test_single_and_ampersand_forms_masked():
    out = T.mask_section("AG ¶ 19(a) and AG & 20(b) and ¶ 20(d).", "citation")
    assert "19(a)" not in out and "20(b)" not in out and "20(d)" not in out, out
    assert out.count(T.MASK_TOKEN) == 3, out


def test_bare_paragraph_is_not_a_citation():
    out = T.mask_section("Under AG ¶ 19 the judge looks at debts.", "citation")
    assert "AG ¶ 19" in out


def test_citation_mode_keeps_wording_blind_masks_it():
    text = "He had an inability to satisfy debts for years."
    assert T.mask_section(text, "citation") == text
    blind = T.mask_section(text, "blind")
    assert T.MASK_TOKEN in blind and "inability" not in blind, blind
    assert blind.startswith("He had an ") and blind.endswith(" for years."), blind


def test_short_wording_19a_needs_only_its_own_length():
    # 19(a) is 23 key characters, under the 30 cap: the cap is min(30, len)
    out = T.mask_section("Inability   to\nsatisfy debts.", "blind")
    assert out == f"{T.MASK_TOKEN}.", out


def test_wording_run_is_case_and_layout_blind_but_exact_otherwise():
    assert "satisfy" not in T.mask_section("INABILITY TO SATISFY DEBTS", "blind")
    assert "satisfy" in T.mask_section("inability to satisfy some debts", "blind")


def test_build_input_skips_the_findings_and_spans_the_gap():
    row = _row()
    for mode in T.MODES:
        text, region = T.build_input(row, mode)
        assert "FORMAL FINDINGS" not in text and "Subparagraph" not in text
        assert text.endswith("Clearance is denied.")
        assert "\n\n" in text
        assert "  " not in text and "\n" not in text.replace("\n\n", "")
        assert region == (row["sec_analysis_start"], row["sec_conclusions_end"])
        assert row["full_text"][region[0]:region[1]].count("FORMAL FINDINGS") == 1
    cit, _ = T.build_input(row, "citation")
    blind, _ = T.build_input(row, "blind")
    assert "19(a)" not in cit and "20(a)" not in cit
    assert "inability to satisfy debts" in cit and "inability to satisfy debts" not in blind


def test_collapse_false_keeps_layout():
    text, _ = T.build_input(_row(), "citation", collapse=False)
    assert "after he\nlost" in text


def test_single_section_has_no_blank_line():
    row = _row()
    row["sec_conclusions_start"] = row["sec_conclusions_end"] = None
    text, region = T.build_input(row, "citation")
    assert "\n\n" not in text and region == (row["sec_analysis_start"], row["sec_analysis_end"])


def test_no_sections_raises():
    row = _row()
    for k in ("sec_analysis_start", "sec_analysis_end", "sec_conclusions_start", "sec_conclusions_end"):
        row[k] = None
    try:
        T.build_input(row, "citation")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_quote_from_masked_input_verifies_against_the_original():
    row = _row()
    text, region = T.build_input(row, "citation")
    quote = "Applicant has $40,000 in delinquent credit cards and did not pay them after he lost his job."
    assert quote in text
    res = verify_guideline_quotes(row["full_text"], [{"condition_code": "AG ¶ 19(c)", "exact_quote": quote}],
                                  "F", region=region, mask_token=T.MASK_TOKEN)
    assert len(res.kept) == 1 and not res.dropped
    k = res.kept[0]
    assert region[0] <= k.start < k.end <= region[1] and T.MASK_TOKEN not in k.text


def test_quote_with_mask_token_is_dropped():
    row = _row()
    _, region = T.build_input(row, "citation")
    res = verify_guideline_quotes(row["full_text"], [{"condition_code": "AG ¶ 19(a)",
                                  "exact_quote": f"{T.MASK_TOKEN} applies. He had an"}],
                                  "F", region=region, mask_token=T.MASK_TOKEN)
    assert res.dropped[0].reason == "contains_mask"


def test_schema_shape():
    s = T.SCHEMA
    assert s["additionalProperties"] is False and s["required"] == ["items"]
    item = s["properties"]["items"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["condition_code", "exact_quote"]
    enum = item["properties"]["condition_code"]["enum"]
    assert len(enum) == 16 and enum[0] == "AG ¶ 19(a)" and enum[-1] == "AG ¶ 20(g)"
    assert set(item["properties"]) == {"condition_code", "exact_quote"}   # no applies, score or confidence


def test_prompt_lists_all_codes_and_forbids_classification():
    for code in T.codes():
        assert code in T.SYSTEM_PROMPT
    assert "whether or not the judge found" in T.SYSTEM_PROMPT
    assert "Do not decide whether a condition applies" in T.SYSTEM_PROMPT
    assert T.MASK_TOKEN in T.SYSTEM_PROMPT


def test_request_params_avoid_rejected_fields():
    p = T.request_params("text", model="claude-haiku-5-5")
    for bad in ("temperature", "top_p", "top_k", "thinking", "budget_tokens"):
        assert bad not in p
    assert p["max_tokens"] == 8000
    assert p["output_config"]["effort"] == "medium"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["messages"] == [{"role": "user", "content": "text"}]   # no assistant prefill


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "template tests"))
