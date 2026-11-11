"""Numbers for the decisions dashboard.

Every figure comes from `load_cases`. The year is the decision date. The
docket number is a filing year and is not the year a chart uses. Outcome
rates stay inside one label tier: formal rulings, inherited appeals, and
keyword guesses are not one population.
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import GUIDELINE_CODES, GUIDELINE_NAMES

COLUMNS = [
    "outcome",
    "outcome_stored",
    "guidelines",
    "guidelines_stored",
    "guideline_findings",
    "guideline_regime",
    "label_source",
    "date",
    "date_stored",
    "case_number",
    "case_type",
    "case_type_stored",
]

OUTCOMES = ["DENIED", "GRANTED", "REMANDED", "REVOKED"]
TIERS = [
    "formal_ruling",
    "inherited_from_hearing",
    "regex_fallback",
    "none",
]
TIER_LABELS = {
    "formal_ruling": "Judge's ruling",
    "inherited_from_hearing": "Inherited from the hearing",
    "regex_fallback": "Keyword guess",
    "none": "No label",
}
REGIMES = ["sead4", "directive", "unknown"]
REGIME_LABELS = {
    "sead4": "SEAD-4",
    "directive": "Directive 5220.6",
    "unknown": "Unstated",
}

# A docket like 03-16936 was filed in 2003. Decisions in that file are often
# later. 90 and above are 1990s; the corpus does not reach 2090.
_DOCKET = re.compile(r"^(\d{2})-")


def as_codes(value) -> list[str]:
    """Guideline letters in one cell. Missing and junk become an empty list."""
    if value is None or isinstance(value, float):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    found = []
    for item in items:
        code = str(item)
        if code in GUIDELINE_CODES and code not in found:
            found.append(code)
    return found


def decision_year(dates: pd.Series) -> pd.Series:
    """Year of the decision date. This is not the docket year."""
    return pd.to_datetime(dates, errors="coerce").dt.year.astype("Int64")


def docket_year(case_number) -> int | None:
    """Filing year encoded in the case number. Not a decision year."""
    match = _DOCKET.match("" if case_number is None else str(case_number))
    if not match:
        return None
    year = int(match.group(1))
    return 1900 + year if year >= 90 else 2000 + year


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the decision year used by every year filter and chart."""
    out = df.copy()
    out["decision_year"] = decision_year(out["date"])
    return out


def apply_view(
    df: pd.DataFrame,
    tier: str,
    regimes: list[str],
    year_min: int,
    year_max: int,
) -> tuple[pd.DataFrame, int]:
    """Rows for the charts, and how many dated rows the year slider dropped.

    `tier` of "all" does not filter. Callers that mix tiers have to say so.
    Rows with no decision date are counted and left out of a year range.
    """
    view = df
    if tier != "all":
        view = view[view["label_source"] == tier]
    if regimes:
        view = view[view["guideline_regime"].isin(regimes)]
    undated = int(view["decision_year"].isna().sum())
    dated = view["decision_year"].between(year_min, year_max)
    return view.loc[dated].copy(), undated


def outcome_label(value) -> str:
    if value in OUTCOMES:
        return str(value)
    return "Unresolved"


def outcome_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per decision: outcome, guideline letters, year, regime."""
    rows = []
    for outcome, codes, year, regime in zip(
        df["outcome"], df["guidelines"], df["decision_year"], df["guideline_regime"]
    ):
        label = outcome_label(outcome)
        letters = as_codes(codes) or [None]
        for letter in letters:
            rows.append({
                "outcome": label,
                "guideline": letter,
                "year": None if pd.isna(year) else int(year),
                "regime": regime if regime in REGIME_LABELS else "unknown",
            })
    return pd.DataFrame(rows, columns=["outcome", "guideline", "year", "regime"])


def direction_rates(df: pd.DataFrame) -> pd.DataFrame:
    """FOR and AGAINST among rulings that state a direction.

    The denominator is rulings on that guideline, not every case. A guideline
    with no direction contributes a row of zeros rather than a blank.
    """
    rows = []
    findings = df["guideline_findings"] if "guideline_findings" in df else [None] * len(df)
    for code in GUIDELINE_CODES:
        toward = 0
        against = 0
        for value in findings:
            if not isinstance(value, dict):
                continue
            direction = value.get(code)
            if direction == "FOR":
                toward += 1
            elif direction == "AGAINST":
                against += 1
        stated = toward + against
        rows.append({
            "guideline": code,
            "name": GUIDELINE_NAMES[code],
            "for": toward,
            "against": against,
            "stated": stated,
            "against_share": (against / stated) if stated else None,
        })
    return pd.DataFrame(rows)


def tier_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many cases sit in each label tier. This ignores the tier filter."""
    counts = df["label_source"].value_counts(dropna=False)
    total = int(counts.sum())
    rows = []
    for tier in TIERS:
        count = int(counts.get(tier, 0))
        rows.append({
            "tier": tier,
            "label": TIER_LABELS[tier],
            "cases": count,
            "share": (count / total) if total else 0.0,
        })
    return pd.DataFrame(rows)


def outcome_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Stored outcome beside the outcome re-read from the verdict."""
    order = OUTCOMES + ["Unresolved"]
    before = df["outcome_stored"].map(lambda value: "Unresolved" if value in (None, "UNKNOWN", "Unknown") or (isinstance(value, float)) else (value if value in OUTCOMES else "Unresolved"))
    after = df["outcome"].map(outcome_label)
    rows = []
    before_counts = before.value_counts()
    after_counts = after.value_counts()
    for name in order:
        rows.append({
            "outcome": name,
            "Before cleaning": int(before_counts.get(name, 0)),
            "After cleaning": int(after_counts.get(name, 0)),
        })
    return pd.DataFrame(rows)


def guideline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Cases listing each guideline, stored keywords beside the cleaned labels."""
    before = {code: 0 for code in GUIDELINE_CODES}
    after = {code: 0 for code in GUIDELINE_CODES}
    for stored, cleaned in zip(df["guidelines_stored"], df["guidelines"]):
        for code in as_codes(stored):
            before[code] += 1
        for code in as_codes(cleaned):
            after[code] += 1
    return pd.DataFrame({
        "guideline": GUIDELINE_CODES,
        "Before cleaning": [before[code] for code in GUIDELINE_CODES],
        "After cleaning": [after[code] for code in GUIDELINE_CODES],
    })


def year_disagreement(df: pd.DataFrame) -> dict[str, int]:
    """How often the docket year is not the decision year.

    Charts do not use this. It is the reason the year slider reads `date`.
    """
    decision = df["decision_year"]
    docket = df["case_number"].map(docket_year)
    docket = pd.Series(docket, index=df.index, dtype="Int64")
    both = decision.notna() & docket.notna()
    differ = int((decision[both] != docket[both]).sum())
    return {
        "compared": int(both.sum()),
        "differ": differ,
    }
