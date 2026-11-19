"""DOHA decision dashboard.

    streamlit run app/dashboard.py

The page is one pass: what happened, then direction, then whether the labels
can be trusted, then what cleaning changed. Filters stay in the sidebar so
the choice is small and in a place people already expect.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard_data import (
    COLUMNS,
    OUTCOMES,
    REGIMES,
    REGIME_LABELS,
    TIER_LABELS,
    TIERS,
    apply_view,
    direction_rates,
    guideline_comparison,
    outcome_comparison,
    outcome_frame,
    prepare,
    tier_coverage,
    year_disagreement,
)
from src.dataset import load_cases

OUTCOME_COLORS = {
    "DENIED": "#8c3a3a",
    "GRANTED": "#2c6e8a",
    "REMANDED": "#c48a2a",
    "REVOKED": "#5c4a6a",
    "Unresolved": "#9a958c",
}
DIRECTION_COLORS = {"AGAINST": "#8c3a3a", "FOR": "#2c6e8a"}


@st.cache_data(show_spinner="Loading decisions…")
def load() -> pd.DataFrame:
    return prepare(load_cases(pool=False, columns=COLUMNS))


def _outcome_colors() -> alt.Scale:
    domain = OUTCOMES + ["Unresolved"]
    return alt.Scale(domain=domain, range=[OUTCOME_COLORS[name] for name in domain])


def outcome_by(data: pd.DataFrame, column: str, sort: list) -> alt.Chart:
    grouped = (
        data.groupby([column, "outcome"], dropna=False)
        .size()
        .reset_index(name="cases")
    )
    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:N", title=None, sort=sort),
            y=alt.Y("cases:Q", title="Decisions", stack="zero"),
            color=alt.Color("outcome:N", title="Outcome", scale=_outcome_colors()),
            tooltip=[f"{column}:N", "outcome:N", "cases:Q"],
        )
        .properties(height=320)
    )


def main() -> None:
    st.set_page_config(page_title="DOHA decisions", layout="wide")
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.4rem; max-width: 1080px;}
            h1 {font-weight: 600; letter-spacing: -0.02em;}
            h2 {font-weight: 600; padding-top: 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load()
    years = df["decision_year"].dropna()
    year_min = int(years.min()) if len(years) else 1996
    year_max = int(years.max()) if len(years) else 2026

    st.sidebar.header("Filters")
    tier_choice = st.sidebar.radio(
        "Label tier",
        options=["formal_ruling", "inherited_from_hearing", "regex_fallback", "none", "all"],
        format_func=lambda key: "All tiers" if key == "all" else TIER_LABELS[key],
        index=0,
    )
    regime_choice = st.sidebar.multiselect(
        "Citation regime",
        options=REGIMES,
        default=REGIMES,
        format_func=lambda key: REGIME_LABELS[key],
    )
    chosen_years = st.sidebar.slider(
        "Decision year",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    st.sidebar.caption(
        "The year is the date of the decision, not the year in the docket number. "
        "A file opened in 2003 can be decided in 2007."
    )

    if tier_choice == "all":
        st.sidebar.warning(
            "All tiers are on. Keyword guesses and judge-written rulings are not the same kind of label."
        )

    view, undated = apply_view(
        df, tier_choice, regime_choice, chosen_years[0], chosen_years[1]
    )

    st.title("DOHA security-clearance decisions")
    st.caption(
        "Outcomes by guideline, decision year, and citation regime. "
        "Direction rates use only rulings that say FOR or AGAINST."
    )

    if view.empty:
        st.info("Nothing is in this filter. Widen the year or turn a regime back on.")
        return

    outcomes = view["outcome"].map(lambda value: value if value in OUTCOMES else "Unresolved")
    total = len(view)
    denied = int((outcomes == "DENIED").sum())
    granted = int((outcomes == "GRANTED").sum())
    open_cases = int((outcomes == "Unresolved").sum())

    one, two, three, four = st.columns(4)
    one.metric("Decisions in view", f"{total:,}")
    two.metric("Denied", f"{denied / total:.0%}" if total else "—")
    three.metric("Granted", f"{granted / total:.0%}" if total else "—")
    four.metric("No outcome", f"{open_cases:,}")
    if undated:
        st.caption(f"{undated:,} decisions in this tier and regime have no usable date, so they are not in the year range.")

    named = outcome_frame(view).dropna(subset=["guideline"])
    cases = pd.DataFrame({
        "outcome": outcomes,
        "year": view["decision_year"].map(lambda year: None if pd.isna(year) else int(year)),
        "regime": view["guideline_regime"].map(lambda key: REGIME_LABELS.get(key, "Unstated")),
    })

    st.header("Outcomes")
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            outcome_by(named, "guideline", list("ABCDEFGHIJKLM")),
            use_container_width=True,
        )
        st.caption("By guideline. A decision that cites two guidelines is counted under both.")
    with right:
        by_year = cases.dropna(subset=["year"])
        st.altair_chart(
            outcome_by(by_year, "year", sorted(by_year["year"].unique())),
            use_container_width=True,
        )
        st.caption("By decision year. Each decision is counted once.")

    st.altair_chart(
        outcome_by(cases, "regime", [REGIME_LABELS[key] for key in REGIMES]),
        use_container_width=True,
    )
    st.caption("By citation regime. SEAD-4 and Directive 5220.6 number the rules differently.")

    st.header("FOR and AGAINST")
    st.caption(
        "Share of stated rulings that went AGAINST the applicant. "
        "The count is rulings on that guideline, not every decision in the filter."
    )
    directions = direction_rates(view)
    stated = directions[directions["stated"] > 0].copy()
    if stated.empty:
        st.info("This tier does not record a FOR or AGAINST on any guideline.")
    else:
        long = stated.melt(
            id_vars=["guideline", "stated"],
            value_vars=["for", "against"],
            var_name="direction",
            value_name="rulings",
        )
        long["direction"] = long["direction"].map({"for": "FOR", "against": "AGAINST"})
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("guideline:N", title=None, sort=list("ABCDEFGHIJKLM")),
                y=alt.Y("rulings:Q", title="Rulings", stack="normalize"),
                color=alt.Color(
                    "direction:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["AGAINST", "FOR"],
                        range=[DIRECTION_COLORS["AGAINST"], DIRECTION_COLORS["FOR"]],
                    ),
                ),
                tooltip=["guideline:N", "direction:N", "rulings:Q", "stated:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        table = stated[["guideline", "name", "against", "for", "stated"]].rename(columns={
            "guideline": "Guideline",
            "name": "Name",
            "against": "Against",
            "for": "For",
            "stated": "Stated",
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.header("Label tiers")
    st.caption(
        "This count ignores the tier filter. It still respects the year and the regime. "
        "Keyword guesses are about 60% precise, so they are not mixed into the rates above "
        "unless you chose All tiers."
    )
    tier_base, _ = apply_view(df, "all", regime_choice, chosen_years[0], chosen_years[1])
    coverage = tier_coverage(tier_base)
    coverage["Cases"] = coverage["cases"]
    coverage["Share"] = coverage["share"].map(lambda value: f"{value:.0%}")
    st.dataframe(
        coverage[["label", "Cases", "Share"]].rename(columns={"label": "Tier"}),
        hide_index=True,
        use_container_width=True,
    )
    tier_chart = (
        alt.Chart(coverage)
        .mark_bar(color="#2c6e8a")
        .encode(
            x=alt.X("label:N", title=None, sort=[TIER_LABELS[tier] for tier in TIERS]),
            y=alt.Y("cases:Q", title="Decisions"),
            tooltip=["label:N", "cases:Q"],
        )
        .properties(height=240)
    )
    st.altair_chart(tier_chart, use_container_width=True)

    st.header("Before and after cleaning")
    st.caption(
        "Stored values are what the old parser wrote. Cleaned values are what this project re-read. "
        "Both columns are on the same surviving decisions."
    )
    compared = year_disagreement(df)
    differ_share = compared["differ"] / compared["compared"] if compared["compared"] else 0
    st.metric(
        "Decision year differs from the docket year",
        f"{compared['differ']:,}",
        help="The docket year is the filing year in the case number. Charts on this page do not use it.",
    )
    st.caption(
        f"That is {differ_share:.0%} of {compared['compared']:,} decisions that have both years. "
        "Using the docket year would date those decisions too early."
    )

    outcomes_wide = outcome_comparison(df).melt(
        "outcome", var_name="version", value_name="cases"
    )
    guidelines_wide = guideline_comparison(df).melt(
        "guideline", var_name="version", value_name="cases"
    )
    before_after = alt.Scale(
        domain=["Before cleaning", "After cleaning"],
        range=["#9a958c", "#1f4e3d"],
    )
    outcome_chart = (
        alt.Chart(outcomes_wide)
        .mark_bar()
        .encode(
            x=alt.X("outcome:N", title=None, sort=OUTCOMES + ["Unresolved"]),
            y=alt.Y("cases:Q", title="Decisions"),
            color=alt.Color("version:N", title=None, scale=before_after),
            xOffset="version:N",
            tooltip=["outcome:N", "version:N", "cases:Q"],
        )
        .properties(height=280, title="Outcomes")
    )
    guideline_chart = (
        alt.Chart(guidelines_wide)
        .mark_bar()
        .encode(
            x=alt.X("guideline:N", title=None, sort=list("ABCDEFGHIJKLM")),
            y=alt.Y("cases:Q", title="Decisions listing it"),
            color=alt.Color("version:N", title=None, scale=before_after),
            xOffset="version:N",
            tooltip=["guideline:N", "version:N", "cases:Q"],
        )
        .properties(height=280, title="Guidelines")
    )
    st.altair_chart(outcome_chart, use_container_width=True)
    st.altair_chart(guideline_chart, use_container_width=True)
    st.caption(
        "Guideline A collapses after cleaning because the old pattern treated paragraph 2, "
        "which almost every decision quotes, as Allegiance to the United States."
    )


if __name__ == "__main__":
    main()
