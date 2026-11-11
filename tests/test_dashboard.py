"""Dashboard figures: the year is the decision date, and tiers stay apart.

    python tests/test_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _harness import run_module

from src.dashboard_data import (
    apply_view,
    decision_year,
    direction_rates,
    docket_year,
    guideline_comparison,
    outcome_comparison,
    prepare,
    tier_coverage,
    year_disagreement,
)


def test_decision_year_ignores_the_docket_number():
    """03-16936 was filed in 2003 and decided on 2 August 2007."""
    dates = pd.Series(["2007-08-02", "1996-03-22"])
    years = decision_year(dates)
    assert int(years.iloc[0]) == 2007
    assert int(years.iloc[1]) == 1996
    assert docket_year("03-16936") == 2003
    assert docket_year("96-00001") == 1996
    assert years.iloc[0] != docket_year("03-16936")


def test_a_missing_date_does_not_fall_back_to_the_docket():
    years = decision_year(pd.Series([None, "not a date"]))
    assert years.isna().all()
    assert docket_year("15-03025") == 2015


def test_outcome_view_stays_inside_one_tier():
    df = prepare(pd.DataFrame({
        "outcome": ["DENIED", "GRANTED"],
        "guidelines": [["F"], ["A"]],
        "guideline_findings": [
            {"F": "AGAINST"},
            {"A": "FOR"},
        ],
        "guideline_regime": ["sead4", "sead4"],
        "label_source": ["formal_ruling", "regex_fallback"],
        "date": ["2014-01-01", "2014-06-01"],
        "case_number": ["99-00001", "14-00002"],
    }))
    view, undated = apply_view(df, "formal_ruling", ["sead4"], 1996, 2026)
    assert undated == 0
    assert list(view["label_source"]) == ["formal_ruling"]
    assert int(view["decision_year"].iloc[0]) == 2014
    assert docket_year("99-00001") == 1999


def test_direction_rate_uses_only_stated_rulings():
    df = pd.DataFrame({
        "guideline_findings": [
            {"F": "AGAINST", "B": "FOR", "A": None},
            {"F": "AGAINST", "B": None, "A": None},
            {"F": None, "B": None, "A": None},
        ]
    })
    rates = direction_rates(df).set_index("guideline")
    assert rates.loc["F", "against"] == 2
    assert rates.loc["F", "for"] == 0
    assert rates.loc["B", "for"] == 1
    assert rates.loc["A", "stated"] == 0
    assert pd.isna(rates.loc["A", "against_share"])


def test_tier_coverage_keeps_every_tier():
    df = pd.DataFrame({
        "label_source": ["formal_ruling", "formal_ruling", "regex_fallback", "none"],
    })
    coverage = tier_coverage(df).set_index("tier")
    assert list(coverage.index) == [
        "formal_ruling", "inherited_from_hearing", "regex_fallback", "none",
    ]
    assert coverage.loc["formal_ruling", "cases"] == 2
    assert coverage.loc["inherited_from_hearing", "cases"] == 0
    assert coverage.loc["regex_fallback", "share"] == 0.25


def test_cleaning_comparison_uses_the_stored_columns():
    df = prepare(pd.DataFrame({
        "outcome": ["DENIED", "GRANTED"],
        "outcome_stored": ["UNKNOWN", "GRANTED"],
        "guidelines": [["B"], ["F"]],
        "guidelines_stored": [["A", "B"], ["A", "F"]],
        "date": ["2007-08-02", "2010-01-01"],
        "case_number": ["03-16936", "10-00001"],
    }))
    outcomes = outcome_comparison(df).set_index("outcome")
    assert outcomes.loc["Unresolved", "Before cleaning"] == 1
    assert outcomes.loc["DENIED", "After cleaning"] == 1
    guidelines = guideline_comparison(df).set_index("guideline")
    assert guidelines.loc["A", "Before cleaning"] == 2
    assert guidelines.loc["A", "After cleaning"] == 0
    compared = year_disagreement(df)
    assert compared["compared"] == 2
    assert compared["differ"] == 1


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "dashboard"))
