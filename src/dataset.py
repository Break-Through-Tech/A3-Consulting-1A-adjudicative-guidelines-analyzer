"""Load the cleaned Challenge_Data for analysis and modeling."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src import config


def load_cases(pool: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """Load cleaned cases from Challenge_Data parquet shards.
    
    Args:
        pool: If True, filter to training pool (use_for_training=True, 
              formal_ruling, non-empty text_features). If False, load all cases.
        columns: List of columns to load. If None, loads all columns.
    
    Returns:
        DataFrame with requested cases and columns.
    """
    data_dir = Path(__file__).resolve().parents[1] / "Challenge_Data"
    files = sorted(glob.glob(str(data_dir / "cases_clean_*.parquet")))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    
    if pool:
        #Filter to training pool
        df = df[
            (df['use_for_training'] == True) &  # noqa: E712
            (df['label_source'] == 'formal_ruling') &
            (df['text_features'].str.len() > 0)
        ].copy()
    
    return df
