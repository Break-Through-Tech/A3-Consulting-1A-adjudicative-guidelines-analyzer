"""Guideline F citations, including the ones that must not count.

    python tests/test_conditions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import parametrize, run_module  # noqa: E402
from src.conditions import extract  # noqa: E402


def _codes(text, **kwargs):
    return [hit["code"] for hit in extract(text, **kwargs)]


def test_a_real_subparagraph_is_a_condition_and_carries_its_offset():
    text = "The judge applied AG ¶ 19(a) to the debts."
    hits = extract(text)
    assert len(hits) == 1
    hit = hits[0]
    assert hit["code"] == "AG ¶ 19(a)"
    assert hit["kind"] == "disqualifier"
    assert text[hit["start"]:hit["end"]] == "AG ¶ 19(a)"


@parametrize("text,expected", [
    ("AG ¶ 20(b)", "AG ¶ 20(b)"),
    ("ag ¶ 20(B)", "AG ¶ 20(b)"),
    ("AG para 20(b)", "AG ¶ 20(b)"),
    ("AG ¶20(b)", "AG ¶ 20(b)"),
    ("AG ¶¶ 19(e)", "AG ¶ 19(e)"),
    ("AG & 19(a)", "AG ¶ 19(a)"),
    ("¶ 20(a)", "AG ¶ 20(a)"),
    ("paragraph 19(c)", "AG ¶ 19(c)"),
    ("MC 20(a)", "AG ¶ 20(a)"),
    ("MC 20 (b)", "AG ¶ 20(b)"),
    ("FC MC 20(d)", "AG ¶ 20(d)"),
    ("FC DC 19(c)", "AG ¶ 19(c)"),
    ("DC 19(a)", "AG ¶ 19(a)"),
    ("Mitigating Condition 20(c)", "AG ¶ 20(c)"),
    ("bare 20(a) in the analysis", "AG ¶ 20(a)"),
])
def test_citation_forms_land_on_the_closed_list(text, expected):
    hits = extract(text)
    assert [hit["code"] for hit in hits] == [expected]
    quoted = text[hits[0]["start"]:hits[0]["end"]].lower().replace(" ", "")
    assert expected[-4:].replace(" ", "") in quoted


def test_a_list_keeps_each_written_letter_and_a_range_fills_the_gap():
    assert _codes("AG ¶¶ 19(a), 19(c) and 19(g)") == [
        "AG ¶ 19(a)", "AG ¶ 19(c)", "AG ¶ 19(g)",
    ]
    assert _codes("conditions 19(a) and (c)") == ["AG ¶ 19(a)", "AG ¶ 19(c)"]
    assert _codes("MC 20(b)-(d)") == ["AG ¶ 20(b)", "AG ¶ 20(c)", "AG ¶ 20(d)"]
    assert _codes("FCMC 20(a) through (d)") == [
        "AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)", "AG ¶ 20(d)",
    ]


def test_the_same_condition_cited_twice_keeps_both_positions():
    text = "MC 20(b) applies. Later, MC 20(b) applies again."
    hits = extract(text)
    assert [hit["code"] for hit in hits] == ["AG ¶ 20(b)", "AG ¶ 20(b)"]
    assert hits[0]["start"] != hits[1]["start"]
    assert text[hits[0]["start"]:hits[0]["end"]] == "MC 20(b)"
    assert text[hits[1]["start"]:hits[1]["end"]] == "MC 20(b)"


def test_directive_codes_and_fc_ordinals_are_the_closed_directive_list():
    text = "See Directive ¶ E2.A6.1.3.3 and FC DC 3 and FC MC 1."
    hits = extract(text)
    assert [(hit["code"], hit["kind"]) for hit in hits] == [
        ("E2.A6.1.3.3", "mitigator"),
        ("E2.A6.1.2.3", "disqualifier"),
        ("E2.A6.1.3.1", "mitigator"),
    ]
    assert hits[0]["sead4"] == "AG ¶ 20(b)"
    assert "sead4" not in hits[1]
    assert hits[2]["sead4"] == "AG ¶ 20(a)"
    assert text[hits[0]["start"]:hits[0]["end"]] == "E2.A6.1.3.3"


@parametrize("text", [
    "AG ¶ 19",
    "AG ¶ 20",
    "AG ¶ 190",
    "AG ¶ 190(a)",
    "AG ¶ 19(z)",
    "AG ¶ 20(h)",
    "AG ¶ 22(a)",
    "E2.A6.1.1",
    "E2.A6.1.2",
    "E2.A6.1.3",
    "E2.A6.1.2.9",
    "E2.A6.1.21",
    "E2.A6.1.3.3.3",
    "DC 7(d)",
    "FC DC 9",
    "Personal Conduct Disqualifying Condition 4",
    "disqualifying conditions and mitigating conditions",
    "",
])
def test_bare_paragraphs_and_invented_codes_do_not_match(text):
    assert extract(text) == []


def test_a_policies_recitation_is_not_a_citation():
    text = "POLICIES AG ¶ 19(a) and AG ¶ 20(b) are the standards.\nThe judge applied AG ¶ 20(d)."
    policies = (0, text.index("The judge"))
    assert _codes(text, policies=policies) == ["AG ¶ 20(d)"]


def test_search_can_be_limited_to_the_analysis():
    text = "AG ¶ 19(a) in the policies block. ANALYSIS The judge credited AG ¶ 20(b)."
    spans = [(text.index("ANALYSIS"), len(text))]
    hits = extract(text, spans=spans)
    assert [hit["code"] for hit in hits] == ["AG ¶ 20(b)"]
    assert hits[0]["start"] == text.index("AG ¶ 20(b)")


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "conditions"))
