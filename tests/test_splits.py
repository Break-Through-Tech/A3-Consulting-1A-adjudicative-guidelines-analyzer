"""Unit tests for the frozen train/val/test splits.

    python tests/test_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _harness import run_module
from src import config
from src.build_splits import get_rarest_guideline


def test_rarest_guideline_absorbs_guideline_a():
    freq = {"A": 4, "B": 100, "E": 200, "L": 40}
    # When A co-occurs with B, A should be ignored and B returned (or L if L is rarer)
    assert get_rarest_guideline(["A", "B"], freq) == "B"
    assert get_rarest_guideline(["A", "B", "L"], freq) == "L"
    # Bare A returns A
    assert get_rarest_guideline(["A"], freq) == "A"
    assert get_rarest_guideline([], freq) == "NONE"
    assert get_rarest_guideline(None, freq) == "NONE"


def test_splits_file_integrity():
    splits_path = config.OUT_DIR / "splits.parquet"
    assert splits_path.exists(), f"Missing {splits_path}"

    df = pd.read_parquet(splits_path)
    assert len(df) == 21959, f"Expected 21,959 rows, got {len(df)}"

    req_cols = {"case_number", "case_type", "source_url", "split_group", "split"}
    assert req_cols.issubset(set(df.columns)), f"Missing required columns: {req_cols - set(df.columns)}"

    valid_splits = set(df["split"].unique())
    assert valid_splits == {"train", "val", "test"}, f"Invalid split values: {valid_splits}"

    train_groups = set(df.loc[df["split"] == "train", "split_group"])
    val_groups = set(df.loc[df["split"] == "val", "split_group"])
    test_groups = set(df.loc[df["split"] == "test", "split_group"])

    assert len(train_groups & val_groups) == 0, "Train and Val share groups!"
    assert len(train_groups & test_groups) == 0, "Train and Test share groups!"
    assert len(val_groups & test_groups) == 0, "Val and Test share groups!"

    train_pct = (df["split"] == "train").mean()
    val_pct = (df["split"] == "val").mean()
    test_pct = (df["split"] == "test").mean()

    assert 0.68 <= train_pct <= 0.74, f"Train proportion unexpected: {train_pct:.1%}"
    assert 0.12 <= val_pct <= 0.17, f"Val proportion unexpected: {val_pct:.1%}"
    assert 0.12 <= test_pct <= 0.17, f"Test proportion unexpected: {test_pct:.1%}"


def test_load_cases_and_load_splits_integration():
    from src.dataset import load_cases, load_splits
    splits = load_splits()
    assert len(splits) == 21959
    assert set(splits["split"].unique()) == {"train", "val", "test"}

    # Default pool=True attaches split
    pool = load_cases(pool=True)
    assert "split" in pool.columns
    assert len(pool) == 21959
    assert (pool["split"] == splits["split"]).all()

    # Explicit column subset with 'split'
    sub = load_cases(pool=True, columns=["text_features", "outcome", "split"])
    assert list(sub.columns) == ["text_features", "outcome", "split"]


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "split tests"))

