"""Draw the 50-case label-check sample (PROJECT_PLAN.md section 7).

    python -m src.build_label_check

Produces:
    Challenge_Data/label_check.csv

The judge-ruled labels were parsed from each decision's FORMAL FINDINGS by our
own code, and their precision has never been measured. This sample is the
hand check every model metric is reported beside.

50 pool cases are drawn at random with a fixed seed, so the same 50 come back
every time. A plain random draw is what makes the result a corpus-wide accuracy
(about +/-6 points at 95% confidence if it lands near 95%).

One row per stored guideline. For each case, the reviewer opens the decision from
its link (the DOHA site sends the PDF as a download), reads FORMAL FINDINGS
near the end, and fills in:

    guideline_correct   Y if the judge ruled on this guideline, N if not
    judge_direction     what the judge wrote: FOR, AGAINST, or none
    missed_guidelines   on the case's first row only: any guideline the judge
                        ruled on that has no row here, e.g. "E AGAINST", else blank
    notes               anything odd

Read the whole FORMAL FINDINGS section in the PDF, not a summary of it, so that
mistakes in our own section boundaries and PDF conversion are caught too.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.dataset import load_cases

LABEL_CHECK_PATH = config.OUT_DIR / "label_check.csv"
N_CASES = 50
SEED = 42
FILL_IN = ["guideline_correct", "judge_direction", "missed_guidelines", "notes"]
# Consecutive blocks of an already random sample, so each share is random too.
REVIEWERS = [("Rashed", 13), ("Sam", 13), ("Aryan", 12), ("Brian", 12)]


def reviewer_for(case: int) -> str:
    """Who checks case number `case` (1-based)."""
    end = 0
    for name, count in REVIEWERS:
        end += count
        if case <= end:
            return name
    raise ValueError(f"case {case} is beyond the {end} assigned")


def build_sample(n: int = N_CASES, seed: int = SEED) -> pd.DataFrame:
    """One row per stored guideline of `n` pool cases drawn at random."""
    pool = load_cases(columns=[
        "case_number", "source_url", "date", "guideline_regime", "guidelines", "guideline_findings",
    ])
    sample = pool.sample(n=n, random_state=seed).reset_index(drop=True)

    rows = []
    for i, case in sample.iterrows():
        for code in sorted(case["guidelines"]):
            rows.append({
                "case": i + 1,
                "reviewer": reviewer_for(i + 1),
                "case_number": case["case_number"],
                # Opens the decision from Excel or Google Sheets; source_url keeps the plain link.
                "decision": f'=HYPERLINK("{case["source_url"]}","{case["case_number"]}")',
                "date": case["date"].date().isoformat() if pd.notna(case["date"]) else "",
                "regime": case["guideline_regime"],
                "guideline": f"{code} {config.GUIDELINE_NAMES[code]}",
                "stored_direction": case["guideline_findings"].get(code) or "none",
                **{col: "" for col in FILL_IN},
                "source_url": case["source_url"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_sample()
    df.to_csv(LABEL_CHECK_PATH, index=False)
    print(f"{df['case'].nunique()} cases, {len(df)} guideline rows -> {LABEL_CHECK_PATH}")
    print("stored directions:", df["stored_direction"].value_counts().to_dict())
    print("regimes (cases):", df.drop_duplicates("case")["regime"].value_counts().to_dict())
    cases = df.drop_duplicates("case")
    for name, _ in REVIEWERS:
        mine = cases[cases["reviewer"] == name]["case"]
        rows = (df["reviewer"] == name).sum()
        print(f"  {name}: cases {mine.min()}-{mine.max()}, {rows} rows")


if __name__ == "__main__":
    main()
