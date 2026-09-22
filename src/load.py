"""Read the doha parquet shards. Read-only: nothing here writes to doha/.

Two source-side problems are repaired at load time:

1. Schema drift. pyarrow inferred the `formal_findings` struct separately per
   shard, so shard 1 has keys B-M while shard 2 has A-M. Concatenating them
   yields rows with inconsistent key sets. We normalise both to the full A-M set.

2. Stringified nesting. `formal_findings[X]['subparagraphs']` round-tripped
   through parquet as a Python repr string rather than a list of dicts, e.g.
   "[{'finding': 'Against', 'para': '1.a-1.b'}]". We parse it back.

Neither repair is load-bearing for labels -- labels.py re-derives those from
full_text -- but carrying a coherent column costs little and the damage should
be visible in the quality report rather than silently passed on.
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path

import pandas as pd

from src import config


def _parse_subparagraphs(value):
    """Recover a stringified list-of-dicts. Returns [] on anything unparseable."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _reconcile_formal_findings(value) -> dict:
    """Normalise one formal_findings cell to a full A-M dict."""
    out = {code: None for code in config.GUIDELINE_CODES}
    if not isinstance(value, dict):
        return out
    for code, finding in value.items():
        if code not in out or finding is None:
            continue
        if isinstance(finding, dict):
            repaired = dict(finding)
            repaired["subparagraphs"] = _parse_subparagraphs(
                repaired.get("subparagraphs")
            )
            out[code] = repaired
        else:
            out[code] = finding
    return out


def source_files() -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(config.SOURCE_DIR / config.SOURCE_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"No parquet shards at {config.SOURCE_DIR}. "
            "Expected the doha repo as a sibling of this one."
        )
    return paths


def load_raw() -> tuple[pd.DataFrame, dict]:
    """Load and concatenate every shard.

    Returns the dataframe plus a stats dict for the quality report.
    """
    paths = source_files()
    frames, stats = [], {"shards": [], "subparagraph_parse_failures": 0}

    for path in paths:
        frame = pd.read_parquet(path)
        frame["shard"] = path.name
        frames.append(frame)
        stats["shards"].append({"file": path.name, "rows": len(frame),
                                "size_mb": round(path.stat().st_size / 1024**2, 1)})

    df = pd.concat(frames, ignore_index=True)

    if "formal_findings" in df.columns:
        before = _count_subparagraphs(df["formal_findings"])
        df["formal_findings"] = df["formal_findings"].map(_reconcile_formal_findings)
        after = _count_subparagraphs(df["formal_findings"])
        stats["subparagraph_parse_failures"] = max(before - after, 0)

    stats["rows_loaded"] = len(df)
    stats["columns_loaded"] = len(df.columns)
    return df, stats


def _count_subparagraphs(series: pd.Series) -> int:
    """How many cells carry a non-empty subparagraphs value, string or list.

    Values arrive as numpy arrays as often as lists, so emptiness is checked by
    length -- `if value:` raises on an ndarray.
    """
    total = 0
    for cell in series:
        if not isinstance(cell, dict):
            continue
        for finding in cell.values():
            if not isinstance(finding, dict):
                continue
            value = finding.get("subparagraphs")
            if value is None:
                continue
            try:
                if len(value) > 0:
                    total += 1
            except TypeError:
                continue
    return total
