"""Generate the frozen train / val / test split for the trainable pool.

    python -m src.build_splits

Produces:
    Challenge_Data/splits.parquet
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.dataset import load_cases

SPLITS_PATH = config.OUT_DIR / "splits.parquet"


def get_rarest_guideline(glist: list[str] | np.ndarray | None, freq: dict[str, int]) -> str:
    """Return the rarest guideline in this case, absorbing Guideline A into co-occurring guidelines.

    Per PROJECT_PLAN.md:
    Guideline A's 4 cases can't form a stratum of their own (smaller than fold count),
    but carry a second guideline (E, B, M, E), so they stratify on that instead.
    """
    if glist is None or len(glist) == 0:
        return "NONE"
    candidates = [g for g in glist if g != "A"]
    if not candidates:
        return "A"
    return min(candidates, key=lambda g: freq.get(g, 0))


def compute_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Partition the pool DataFrame into train (~71.4%), val (~14.3%), test (~14.3%).

    Uses a two-stage StratifiedGroupKFold:
    - Stage 1: n_splits=7, shuffle=True, random_state=seed (1 fold test, 6 remainder)
    - Stage 2: n_splits=6, shuffle=True, random_state=seed on remainder (1 fold val, 5 train)
    - Grouped on split_group (case_number)
    - Stratified on rarest guideline in each case
    """
    all_guidelines = [g for glist in df["guidelines"] if glist is not None for g in glist]
    freq = Counter(all_guidelines)

    y = np.array([get_rarest_guideline(g, freq) for g in df["guidelines"]])
    groups = df["split_group"].to_numpy()
    indices = np.arange(len(df))

    # Stage 1: 7 folds, 1 fold as test (approx 1/7 = 14.3%)
    sgkf1 = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf1.split(indices, y, groups))

    # Stage 2: 6 folds on remainder, 1 fold as val (approx 1/6 of 6/7 = 14.3%), 5 as train (~71.4%)
    y_rem = y[train_val_idx]
    groups_rem = groups[train_val_idx]
    indices_rem = np.arange(len(train_val_idx))

    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf2.split(indices_rem, y_rem, groups_rem))

    final_train_idx = train_val_idx[train_sub_idx]
    final_val_idx = train_val_idx[val_sub_idx]
    final_test_idx = test_idx

    split_series = pd.Series(index=df.index, dtype=str)
    split_series.iloc[final_train_idx] = "train"
    split_series.iloc[final_val_idx] = "val"
    split_series.iloc[final_test_idx] = "test"

    out = pd.DataFrame({
        "case_number": df["case_number"],
        "case_type": df["case_type"],
        "source_url": df["source_url"],
        "split_group": df["split_group"],
        "split": split_series,
        "stratum": y,
    })
    return out


def main() -> None:
    print("Loading trainable pool cases...")
    df = load_cases(pool=True)
    print(f"Loaded {len(df):,} trainable rows.")

    splits_df = compute_splits(df, seed=42)

    counts = splits_df["split"].value_counts()
    print("\nSplit breakdown:")
    for split_name, count in counts.items():
        print(f"  {split_name}: {count:,} ({count / len(splits_df):.1%})")

    # Verify no group leakage
    train_groups = set(splits_df.loc[splits_df["split"] == "train", "split_group"])
    val_groups = set(splits_df.loc[splits_df["split"] == "val", "split_group"])
    test_groups = set(splits_df.loc[splits_df["split"] == "test", "split_group"])

    assert len(train_groups & val_groups) == 0, "Leakage: Train and Val share groups!"
    assert len(train_groups & test_groups) == 0, "Leakage: Train and Test share groups!"
    assert len(val_groups & test_groups) == 0, "Leakage: Val and Test share groups!"
    print("\n[PASS] Group leakage check: 0 group overlap across all splits.")

    splits_df.to_parquet(SPLITS_PATH, index=False)
    print(f"\nSaved frozen splits to {SPLITS_PATH}")


if __name__ == "__main__":
    main()

