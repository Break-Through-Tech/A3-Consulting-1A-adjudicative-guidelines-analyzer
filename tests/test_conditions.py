"""Guideline F citations, including the ones that must not count.

    python tests/test_conditions.py
"""
from __future__ import annotations

import sys
from datetime import date
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
    # No regime and no date skips numbered forms. 2005 selects the Directive list.
    hits = extract(text, decided=date(2005, 1, 1))
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


@parametrize("text,expected", [
    ("AG ¶¶ 20(a), (b), and (c)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)"]),
    ("AG ¶¶ 19 (a), (b), and (c) apply", ["AG ¶ 19(a)", "AG ¶ 19(b)", "AG ¶ 19(c)"]),
    ("AG ¶¶ 20(a), (b), or (d)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(d)"]),
    ("AG ¶ 20(a) through 20(d)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)", "AG ¶ 20(d)"]),
    ("AG ¶¶ 20(a) - 20(c) do not apply", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)"]),
    ("AG ¶ 20(a)—(c)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)"]),
    ("AG ¶ 20(a) to (c)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)"]),
    ("AG ¶¶ 20(a)-(c), and (e)", ["AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)", "AG ¶ 20(e)"]),
    (
        "AG ¶¶ 20(a)-(b) are potentially applicable: (c) the individual has received counseling",
        ["AG ¶ 20(a)", "AG ¶ 20(b)"],
    ),
    ("mitigated under E2.A6.1.3. 6. The", ["E2.A6.1.3.6"]),
    ("FC MC E2.A6.\n1.3.1 applies", ["E2.A6.1.3.1"]),
])
def test_ranges_lists_and_broken_directive_codes(text, expected):
    assert _codes(text) == expected


@parametrize("text,decided,guidelines,expected", [
    ("Under Guideline F, DC 1 and DC 3 apply.", date(2005, 1, 1), None,
     ["E2.A6.1.2.1", "E2.A6.1.2.3"]),
    ("Under Guideline F, MC 6 applies.", date(2005, 1, 1), None, ["E2.A6.1.3.6"]),
    ("Financial Considerations Mitigating Condition 3 applies.", date(2005, 1, 1), None,
     ["E2.A6.1.3.3"]),
    ("Under Guideline F, MC 1 applies.", date(2010, 1, 1), None, ["AG ¶ 20(a)"]),
    ("Under Guideline F, DC 3 applies.", date(2010, 1, 1), None, ["AG ¶ 19(c)"]),
    ("FC MC 1 applies.", date(2010, 1, 1), None, ["AG ¶ 20(a)"]),
    ("FC DC 3 applies.", date(2005, 1, 1), None, ["E2.A6.1.2.3"]),
    ("MC 1 applies.", date(2005, 1, 1), ["F"], ["E2.A6.1.3.1"]),
])
def test_numbered_forms_follow_the_date_when_no_regime_is_given(text, decided, guidelines, expected):
    assert _codes(text, decided=decided, guidelines=guidelines) == expected


def test_regime_beats_the_date_for_numbered_forms():
    text = "Under Guideline F, MC 2 applies."
    decided = date(2007, 3, 19)
    assert _codes(text, decided=decided, regime="directive") == ["E2.A6.1.3.2"]
    # A sead4 decision already uses lettered AG ¶ codes. Numbered forms there
    # are usually a quotation of the older Directive list, so they are skipped.
    assert _codes(text, decided=decided, regime="sead4") == []
    assert _codes(text, decided=decided, regime="unknown") == []


@parametrize("text,kwargs", [
    ("Under Guideline F, MC 1 applies.", {}),
    ("Under Guideline F, MC 1 applies.", {"decided": date(2006, 11, 1)}),
    ("Under Guideline J, MC 1 applies.", {"decided": date(2005, 1, 1)}),
    (
        "Under Guideline F, debts. Under Guideline J, MC 1 applies.",
        {"decided": date(2005, 1, 1)},
    ),
    ("MC 1 applies.", {"decided": date(2005, 1, 1), "guidelines": ["F", "J"]}),
    ("Under Guideline F, MC 7 applies.", {"decided": date(2005, 1, 1)}),
    ("Under Guideline F, DC 6 applies.", {"decided": date(2005, 1, 1)}),
    ("Under Guideline F, MC 8 applies.", {"decided": date(2010, 1, 1)}),
    ("Washington, DC 20301", {"decided": date(2010, 1, 1), "guidelines": ["F"]}),
])
def test_numbered_forms_that_are_not_guideline_f_conditions(text, kwargs):
    assert extract(text, **kwargs) == []


def test_a_bare_paragraph_picks_up_the_lettered_list_under_it():
    text = (
        "AG ¶ 19. Three are potentially applicable in this case:\n"
        "(a) inability to satisfy debts;\n"
        "(b) unwillingness to satisfy debts regardless of the ability to do so; and\n"
        "(c) a history of not meeting financial obligations."
    )
    hits = extract(text)
    assert [hit["code"] for hit in hits] == ["AG ¶ 19(a)", "AG ¶ 19(b)", "AG ¶ 19(c)"]
    assert [text[hit["start"]:hit["end"]] for hit in hits] == ["(a)", "(b)", "(c)"]


def test_a_lettered_list_survives_a_page_break():
    text = (
        "Five mitigating conditions under AG ¶ 20 are potentially applicable:\n"
        "(a) the behavior happened so long ago, was so infrequent\n"
        "\n7\n\n"
        "(b) the conditions that resulted in the financial problem were largely\n"
        "(c) the person has received or is receiving counseling\n"
        "(d) the individual initiated a good-faith effort; and\n"
        "(e) the individual has a reasonable basis to dispute the debt\n"
    )
    assert _codes(text) == [
        "AG ¶ 20(a)", "AG ¶ 20(b)", "AG ¶ 20(c)", "AG ¶ 20(d)", "AG ¶ 20(e)",
    ]


def test_a_policies_list_under_a_bare_paragraph_is_not_a_citation():
    text = (
        "POLICIES\nAG ¶ 19. The conditions are:\n"
        "(a) inability to satisfy debts;\n"
        "(b) unwillingness to satisfy debts.\n"
        "ANALYSIS The judge applied AG ¶ 20(d)."
    )
    policies = (0, text.index("ANALYSIS"))
    assert _codes(text, policies=policies) == ["AG ¶ 20(d)"]


@parametrize("text", [
    "AG ¶ 19 sets out the conditions. Applicant paid (a) small debt in 2019.",
    "AG ¶ 20 applies. " + ("x" * 500) + "\n(a) the behavior happened so long ago",
    "AG ¶ 20 provides conditions: (h) not a condition letter",
    "AG ¶ 19 and AG ¶ 20 are discussed below.",
])
def test_a_bare_paragraph_does_not_invent_a_list(text):
    assert extract(text) == []


def test_a_dot_before_the_bracket_is_still_the_paragraph():
    text = "Disqualifying Conditions 19.(a)\ninability to satisfy debts; and 19.(c) a history"
    assert _codes(text) == ["AG ¶ 19(a)", "AG ¶ 19(c)"]
    assert _codes("Mitigating Conditions 20.(c) the person has received") == ["AG ¶ 20(c)"]


def test_a_footnote_after_a_numbered_condition_does_not_hide_it():
    text = "Guideline F disqualifying condition (DC) 1 (10)\nand DC 3 (11) apply here."
    assert _codes(text, decided=date(2005, 8, 15), regime="directive") == [
        "E2.A6.1.2.1", "E2.A6.1.2.3",
    ]


def test_search_can_be_limited_to_the_analysis():
    text = "AG ¶ 19(a) in the policies block. ANALYSIS The judge credited AG ¶ 20(b)."
    spans = [(text.index("ANALYSIS"), len(text))]
    hits = extract(text, spans=spans)
    assert [hit["code"] for hit in hits] == ["AG ¶ 20(b)"]
    assert hits[0]["start"] == text.index("AG ¶ 20(b)")


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "conditions"))
