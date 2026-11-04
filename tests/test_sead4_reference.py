"""The SEAD-4 condition list is closed: 142 real codes, and nothing else.

    python tests/test_sead4_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import parametrize, run_module  # noqa: E402
from src import config  # noqa: E402
from src.sead4_reference import (  # noqa: E402
    CODES,
    CONDITIONS,
    GUIDELINES,
    canonical,
    cited,
    condition,
    conditions_for,
    is_condition,
)


# Disqualifying paragraph, then the mitigating paragraph that follows it.
_PARAGRAPHS = {
    "A": (4, 5), "B": (7, 8), "C": (10, 11), "D": (13, 14), "E": (16, 17),
    "F": (19, 20), "G": (22, 23), "H": (25, 26), "I": (28, 29), "J": (31, 32),
    "K": (34, 35), "L": (37, 38), "M": (40, 41),
}


def test_the_list_is_the_142_official_codes():
    assert len(CODES) == 142
    assert len(CONDITIONS) == 142
    assert list(GUIDELINES) == config.GUIDELINE_CODES
    assert set(CODES) == {row["code"] for row in CONDITIONS.values()}
    for code, row in CONDITIONS.items():
        disq, mit = _PARAGRAPHS[row["guideline"]]
        number = int(code.split("¶")[1].split("(")[0])
        expect = disq if row["kind"] == "disqualifier" else mit
        assert number == expect, code


def test_every_stored_code_resolves_to_itself():
    """A code that is on the list must come back unchanged."""
    for code in CODES:
        assert canonical(code) == code
        row = condition(code)
        assert row["code"] == code
        assert row["guideline"] in GUIDELINES
        assert row["text"]


def test_financial_mitigator_20b_is_the_official_wording():
    row = condition("AG ¶ 20(b)")
    assert row["guideline"] == "F"
    assert row["kind"] == "mitigator"
    assert "largely beyond the person's control" in row["text"]


def test_conditions_for_a_guideline_stay_inside_that_guideline():
    rows = conditions_for("f")
    assert [row["code"] for row in rows[:2]] == ["AG ¶ 19(a)", "AG ¶ 19(b)"]
    assert len(rows) == 16
    assert {row["guideline"] for row in rows} == {"F"}
    assert conditions_for("Z") == []
    assert conditions_for("") == []


@parametrize("citation", [
    "AG ¶ 19(z)",
    "AG ¶ 99(a)",
    "AG ¶ 19",
    "AG ¶ 190(a)",
    "AG ¶ 2(c)",
    "E2.A6.1.2.1",
    "",
])
def test_invented_codes_are_rejected(citation):
    """Bare paragraphs, unknown letters, and Directive cites are not conditions."""
    assert canonical(citation) is None
    assert condition(citation) is None
    assert is_condition(citation) is False


@parametrize("citation,expected", [
    ("AG ¶ 19(a)", "AG ¶ 19(a)"),
    ("ag ¶ 19(A)", "AG ¶ 19(a)"),
    ("AG para 19(a)", "AG ¶ 19(a)"),
    ("AG paragraph 20(b)", "AG ¶ 20(b)"),
    ("paragraph 22(a)", "AG ¶ 22(a)"),
    ("see AG¶7(d) below", "AG ¶ 7(d)"),
])
def test_citation_variants_map_onto_the_closed_list(citation, expected):
    assert canonical(citation) == expected
    assert is_condition(citation) is True


def test_a_passage_keeps_real_codes_and_drops_invented_ones():
    text = "The judge applied AG ¶ 19(a) and AG ¶ 19(z), then AG para 20(b) again AG ¶ 19(a)."
    assert cited(text) == ["AG ¶ 19(a)", "AG ¶ 20(b)"]
    assert cited("") == []
    assert cited("no citation here") == []


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "sead4 reference"))
