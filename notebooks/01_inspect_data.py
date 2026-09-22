"""Step 5: inspect the DOHA parquet shards.

Run from anywhere:  python notebooks\01_inspect_data.py
"""
import glob
from pathlib import Path

import pandas as pd

# Resolve the doha repo relative to THIS file, so cwd doesn't matter.
DATA_DIR = Path(__file__).resolve().parents[2] / "doha" / "doha_parsed_cases"


def main():
    files = sorted(glob.glob(str(DATA_DIR / "all_cases_*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {DATA_DIR}")

    print("=== Files ===")
    for f in files:
        print(f"  {Path(f).name:<24} {Path(f).stat().st_size / 1024**2:>7.1f} MB")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    print(f"\n=== Shape ===\n  {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n=== Columns ===")
    for c in df.columns:
        print(f"  {c}")

    print("\n=== Outcome ===")
    print(df["outcome"].value_counts(dropna=False).to_string())

    print("\n=== Case type ===")
    print(df["case_type"].value_counts(dropna=False).to_string())

    print("\n=== Guideline letters (list column, exploded) ===")
    print(df["guidelines"].explode().value_counts().to_string())

    print("\n=== full_text length (chars) ===")
    print(df["full_text"].str.len().describe().to_string())

    print("\n=== Missing values (top 10) ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10).to_string() or "  none")

    print("\n=== First 2 rows (metadata only) ===")
    cols = ["case_number", "date", "case_type", "outcome", "guidelines", "judge"]
    print(df[cols].head(2).to_string())


if __name__ == "__main__":
    main()
