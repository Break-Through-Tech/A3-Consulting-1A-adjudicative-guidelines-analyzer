"""Load the cleaned dataset. Every analysis should come through here.

    from src.dataset import load_cases
    pool = load_cases()                                      # the trainable pool
    corpus = load_cases(pool=False, columns=["outcome"])     # every row, for EDA

The committed shards have four quirks, each of which has already caused a bug
in a script that handled them by hand. They are fixed here once:

1. Schema drift. pyarrow inferred the `guideline_findings` struct per output
   shard, so cases_clean_1-3 carry keys B-M and cases_clean_4 carries A-M.
   Every row is normalised to the full A-M set -- the same repair
   load._reconcile_formal_findings makes for the raw source.
2. Direction casing. `guideline_findings` says FOR/AGAINST while
   `guideline_subparagraphs[i]["finding"]` says For/Against. Both are uppercased.
3. Missing text is '' rather than null, so `.notna()` keeps everything. The
   pool filter tests string length instead.
4. Size. `full_text` is most of the bytes, so it is only read when asked for.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config, labels

SHARD_GLOB = "cases_clean_*.parquet"

# Read only on request: full_text alone is most of the dataset's size.
HEAVY_COLUMNS = {"full_text"}

# The pool filter needs these, so they are read even if the caller didn't ask.
_POOL_COLUMNS = ["use_for_training", "label_source", "text_features"]


def shard_files() -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(config.OUT_DIR / SHARD_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"No cleaned shards at {config.OUT_DIR}. "
            "Run `python -m src.build_dataset --full` to rebuild them."
        )
    return paths


def _normalise_findings(value) -> dict:
    """One guideline_findings cell as a full A-M dict of FOR / AGAINST / None."""
    out = {code: None for code in config.GUIDELINE_CODES}
    if not isinstance(value, dict):
        return out
    for code, finding in value.items():
        if code in out and isinstance(finding, str):
            out[code] = finding.upper()
    return out


def _normalise_subparagraphs(value) -> list[dict]:
    """Subparagraph findings with For/Against uppercased.

    Values arrive as numpy arrays of dicts, so emptiness is checked by length --
    `if value:` raises on an ndarray.
    """
    if value is None:
        return []
    items = []
    for item in value:
        item = dict(item)
        if isinstance(item.get("finding"), str):
            item["finding"] = item["finding"].upper()
        items.append(item)
    return items


SPLITS_PATH = config.OUT_DIR / "splits.parquet"


def load_splits() -> pd.DataFrame:
    """Return the frozen splits DataFrame (train / val / test)."""
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Missing splits at {SPLITS_PATH}. Run: python -m src.build_splits")
    return pd.read_parquet(SPLITS_PATH)


def load_cases(pool: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned dataset.

    pool=True   the trainable pool: use_for_training, a formal_ruling label and
                non-empty text_features (21,959 rows). Attaches 'split' column
                if splits.parquet exists.
    pool=False  every row (30,517), normalised but unfiltered -- for EDA, which
                needs to see the empty texts and weaker label tiers.
    columns     the columns to return. Default: everything except full_text.
                Can include 'split' to explicitly request split assignment.
    """
    paths = shard_files()
    attach_split = False
    if columns is None:
        columns = [c for c in pq.read_schema(paths[0]).names if c not in HEAVY_COLUMNS]
        attach_split = pool and SPLITS_PATH.exists()
    else:
        columns = list(columns)
        if "split" in columns:
            attach_split = True
            columns.remove("split")

    helpers = [c for c in _POOL_COLUMNS if pool and c not in columns]

    df = pd.concat(
        [pd.read_parquet(p, columns=columns + helpers, engine=config.PARQUET_ENGINE) for p in paths],
        ignore_index=True,
    )
    if "guideline_findings" in df:
        df["guideline_findings"] = df["guideline_findings"].map(_normalise_findings)
    if "guideline_subparagraphs" in df:
        df["guideline_subparagraphs"] = df["guideline_subparagraphs"].map(_normalise_subparagraphs)

    if pool:
        keep = (
            df["use_for_training"]
            & (df["label_source"] == labels.SOURCE_FORMAL)
            & (df["text_features"].str.len() > 0)
        )
        df = df.loc[keep, columns].reset_index(drop=True)

    if attach_split and SPLITS_PATH.exists():
        splits_df = pd.read_parquet(SPLITS_PATH)
        if pool and len(df) == len(splits_df):
            df["split"] = splits_df["split"].to_numpy()
        elif "source_url" in df.columns:
            df = df.merge(splits_df[["source_url", "split"]], on="source_url", how="left")

    return df
