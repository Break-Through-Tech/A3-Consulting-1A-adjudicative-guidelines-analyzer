"""Publish one run's calibrated head probabilities as small files that git can hold.

    python -m src.heads.publish --run runs/modernbert-4096

Reads the gitignored run folder and writes, under ``results/heads/<run>/``:
    val_probabilities.parquet   one row per val case, 39 probability columns
    test_probabilities.parquet  one row per test case, the same columns
    calibration.json            per-head temperature and at-issue threshold

The temperatures and thresholds are the ones ``evaluate.py`` stored in the run's
``calibration.json``. This script recomputes both from the val logits with
``evaluate.py``'s own functions and stops if they differ. It never calls
``evaluate_run``, so the run folder and the other ``results/heads`` tables stay
untouched.

Then it scores the published test file with ``evaluate.py``'s ``baseline_tables``
and compares the result with the committed ``results/heads/<run>*.csv`` rows.
``--check-only`` skips the build and repeats that check from the published
files alone, which needs no run folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_cases
from src.heads import data as heads_data
from src.heads.data import CLASSES, GUIDELINES, IGNORE_INDEX, baseline, head_labels
from src.heads.evaluate import (
    COMPARISON_METRICS,
    RESULTS_DIR,
    _comparison_block,
    _load_rows,
    baseline_tables,
    fit_temperature,
    softmax,
)

PROB_COLUMNS = [f"p_{label.lower()}" for label in CLASSES]
ID_COLUMNS = ["case_number", "source_url", "split"]
TOLERANCE = 1e-9


def wide_columns() -> list[str]:
    return [f"{column}_{code}" for code in GUIDELINES for column in PROB_COLUMNS]


def _pool_with_urls() -> pd.DataFrame:
    """The pool the evaluate code uses, plus the source URL of each row."""
    df = heads_data.load_pool()
    meta = load_cases(pool=True, columns=["case_number", "source_url", "split"])
    if not np.array_equal(meta["case_number"].astype(str).to_numpy(), df["case_number"].astype(str).to_numpy()):
        raise ValueError("The pool and the source_url frame are in different row orders")
    df = df.copy()
    df["source_url"] = meta["source_url"].to_numpy()
    return df


def _wide_frame(df: pd.DataFrame, rows: np.ndarray, split: str, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "case_number": df["case_number"].iloc[rows].astype(str).to_numpy(),
            "source_url": df["source_url"].iloc[rows].astype(str).to_numpy(),
            "split": split,
        }
    )
    for head, code in enumerate(GUIDELINES):
        for index, column in enumerate(PROB_COLUMNS):
            frame[f"{column}_{code}"] = probabilities[:, head, index]
    return frame


def _sanity(frame: pd.DataFrame, split: str) -> list[str]:
    """Problems with one published frame, empty when it is sound."""
    problems = []
    if not frame["split"].eq(split).all():
        problems.append(f"{split}: split column is not all {split}")
    if frame["source_url"].str.len().eq(0).any() or frame["source_url"].isna().any():
        problems.append(f"{split}: empty source_url")
    if frame["source_url"].duplicated().any():
        problems.append(f"{split}: duplicate source_url")
    values = frame[wide_columns()].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        problems.append(f"{split}: NaN probability")
    if ((values < 0) | (values > 1)).any():
        problems.append(f"{split}: probability outside [0, 1]")
    sums = values.reshape(len(frame), len(GUIDELINES), len(CLASSES)).sum(axis=-1)
    worst = float(np.abs(sums - 1.0).max())
    if worst > 1e-6:
        problems.append(f"{split}: a head sums to 1 only within {worst:.2e}")
    return problems


def _thresholds_agree(stored: dict, recomputed: dict) -> list[str]:
    mismatches = []
    for code in GUIDELINES:
        for key in ("temperature", "at_issue_threshold"):
            a, b = float(stored[code][key]), float(recomputed[code][key])
            if abs(a - b) > 1e-6:
                mismatches.append(f"{code} {key}: stored {a!r}, recomputed {b!r}")
    return mismatches


def build(run_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    df = _pool_with_urls()
    targets = heads_data.target_matrix(df)
    stored = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    missing = [code for code in GUIDELINES if "at_issue_threshold" not in stored.get(code, {})]
    print(f"Stored thresholds present for {len(GUIDELINES) - len(missing)} of {len(GUIDELINES)} heads")
    if missing:
        raise ValueError(f"calibration.json has no at_issue_threshold for {missing}")

    rows = {split: _load_rows(run_dir, split, df) for split in ("val", "test")}
    logits = {split: np.load(run_dir / f"logits_{split}.npy").astype(np.float64) for split in rows}

    recomputed: dict[str, dict] = {}
    for head, code in enumerate(GUIDELINES):
        val_y = targets[rows["val"], head]
        known = val_y != IGNORE_INDEX
        temperature = fit_temperature(logits["val"][known, head], val_y[known])
        calibrated = softmax(logits["val"][known, head] / temperature)
        threshold = baseline().choose_at_issue_threshold(pd.Series(head_labels(val_y[known])), calibrated)
        recomputed[code] = {"temperature": temperature, "at_issue_threshold": threshold}
    mismatches = _thresholds_agree(stored, recomputed)
    worst_t = max(abs(stored[c]["temperature"] - recomputed[c]["temperature"]) for c in GUIDELINES)
    worst_th = max(abs(stored[c]["at_issue_threshold"] - recomputed[c]["at_issue_threshold"]) for c in GUIDELINES)
    print(f"Recomputed from val logits: max temperature diff {worst_t:.3e}, max threshold diff {worst_th:.3e}")
    if mismatches:
        raise ValueError("Stored calibration does not match the val logits:\n" + "\n".join(mismatches))

    # The published probabilities use the stored temperatures, as evaluate.py did.
    frames = {}
    for split in ("val", "test"):
        probabilities = np.stack(
            [softmax(logits[split][:, head] / stored[code]["temperature"]) for head, code in enumerate(GUIDELINES)],
            axis=1,
        )
        frames[split] = _wide_frame(df, rows[split], split, probabilities)
        problems = _sanity(frames[split], split)
        if problems:
            raise ValueError("\n".join(problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_parquet(out_dir / f"{split}_probabilities.parquet", index=False)
    shutil.copyfile(run_dir / "calibration.json", out_dir / "calibration.json")

    # The run folder already holds the long test file. The published one must agree.
    long_path = run_dir / "test_probabilities.parquet"
    if long_path.exists():
        long = pd.read_parquet(long_path)
        diffs = []
        for head, code in enumerate(GUIDELINES):
            part = long.loc[long["guideline"].eq(code)].reset_index(drop=True)
            if not np.array_equal(part["case_number"].astype(str).to_numpy(), frames["test"]["case_number"].to_numpy()):
                diffs.append(f"{code}: row order differs")
                continue
            for column in PROB_COLUMNS:
                diff = float(np.abs(part[column].to_numpy() - frames["test"][f"{column}_{code}"].to_numpy()).max())
                if diff > 1e-9:
                    diffs.append(f"{code} {column}: max diff {diff:.2e}")
        print("Published test file against the run's long test file:", "identical" if not diffs else diffs)
        if diffs:
            raise ValueError("Published test probabilities differ from the run's test_probabilities.parquet")
    return frames


# --- the check ---------------------------------------------------------------


def _score(out_dir: Path, df: pd.DataFrame, targets: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the published test file with evaluate.py's table builder."""
    info = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    test = pd.read_parquet(out_dir / "test_probabilities.parquet")
    by_url = pd.Series(np.arange(len(df)), index=df["source_url"].to_numpy())
    if by_url.index.duplicated().any():
        raise ValueError("source_url is not unique in the pool")
    rows = by_url.loc[test["source_url"].to_numpy()].to_numpy()
    if not np.array_equal(df["case_number"].astype(str).iloc[rows].to_numpy(), test["case_number"].to_numpy()):
        raise ValueError("The published test file does not match the pool by case_number")
    test_df = df.iloc[rows]
    val_rows = by_url.loc[pd.read_parquet(out_dir / "val_probabilities.parquet")["source_url"].to_numpy()].to_numpy()

    per_guideline: dict[str, dict] = {}
    for head, code in enumerate(GUIDELINES):
        test_y = targets[rows, head]
        val_y = targets[val_rows, head]
        train_y = targets[(df["split"] == "train").to_numpy(), head]
        known = test_y != IGNORE_INDEX
        probabilities = test[[f"{column}_{code}" for column in PROB_COLUMNS]].to_numpy(dtype=np.float64)[known]
        threshold = float(info[code]["at_issue_threshold"])
        counts = pd.Series(np.bincount(train_y[train_y != IGNORE_INDEX], minlength=len(CLASSES)), index=list(CLASSES))
        per_guideline[code] = {
            "targets": pd.Series(head_labels(test_y[known]), index=test_df.index[known]),
            "predictions": baseline().decode_predictions(probabilities, threshold),
            "probabilities": probabilities,
            "threshold": threshold,
            "val_targets": pd.Series(head_labels(val_y[val_y != IGNORE_INDEX])),
            "train_class_counts": counts,
            "trained_classes": ",".join(sorted(label for label, count in counts.items() if count > 0)),
        }
    return baseline_tables(test_df, per_guideline)


def _compare(new: pd.DataFrame, old: pd.DataFrame, keys: list[str], label: str) -> tuple[bool, float]:
    """Whether two tables match, and the largest numeric difference."""
    new = new.sort_values(keys).reset_index(drop=True)
    old = old.sort_values(keys).reset_index(drop=True)
    shared = [c for c in old.columns if c in new.columns]
    if len(new) != len(old):
        print(f"  {label}: {len(new)} rows against {len(old)} committed")
        return False, float("nan")
    worst, ok = 0.0, True
    for column in shared:
        a, b = new[column], old[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            both_nan = a.isna() & b.isna()
            diff = (a.astype(float) - b.astype(float)).abs()[~both_nan]
            if diff.isna().any():
                ok = False
                print(f"  {label}: {column} has NaN on one side only")
            elif len(diff):
                worst = max(worst, float(diff.max()))
        elif not a.astype(str).reset_index(drop=True).eq(b.astype(str).reset_index(drop=True)).all():
            ok = False
            print(f"  {label}: column {column} differs")
    return ok and worst <= TOLERANCE, worst


def check(out_dir: Path, results_dir: Path = RESULTS_DIR) -> bool:
    """Score the published files and compare with the committed tables."""
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    name = out_dir.name
    df = _pool_with_urls()
    targets = heads_data.target_matrix(df)
    ok = True

    for split in ("val", "test"):
        problems = _sanity(pd.read_parquet(out_dir / f"{split}_probabilities.parquet"), split)
        print(f"{split} file: {'sound, every head sums to 1 within 1e-6, no NaN' if not problems else problems}")
        ok &= not problems
    info = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    has_thresholds = all("at_issue_threshold" in info.get(code, {}) for code in GUIDELINES)
    print(f"calibration.json has temperature and at_issue_threshold for all {len(GUIDELINES)} heads: {has_thresholds}")
    ok &= has_thresholds

    results, summary = _score(out_dir, df, targets)
    committed_results = pd.read_csv(results_dir / f"{name}.csv", keep_default_na=False, na_values=[""])
    committed_summary = pd.read_csv(results_dir / f"{name}_summary.csv", keep_default_na=False, na_values=[""])
    r_ok, r_worst = _compare(results, committed_results, ["regime", "guideline"], f"{name}.csv")
    s_ok, s_worst = _compare(summary, committed_summary, ["regime"], f"{name}_summary.csv")

    comparison = pd.read_csv(results_dir / "comparison.csv", keep_default_na=False, na_values=[""])
    committed_block = comparison.loc[comparison["run"].eq(name)].drop(
        columns=["run", "model", "max_length", "delta_defensible_macro_f1_vs_tfidf"]
    )
    block = _comparison_block(name, "", "", summary, results).drop(columns=["run", "model", "max_length"])
    c_ok, c_worst = _compare(block, committed_block, ["regime"], f"comparison.csv rows for {name}")
    print(f"per-guideline table against {name}.csv: {'match' if r_ok else 'MISMATCH'}, max diff {r_worst:.2e}")
    print(f"summary table against {name}_summary.csv: {'match' if s_ok else 'MISMATCH'}, max diff {s_worst:.2e}")
    print(f"comparison.csv rows for {name}: {'match' if c_ok else 'MISMATCH'}, max diff {c_worst:.2e}")
    ok &= r_ok and s_ok and c_ok

    print("\nRescored from the published test file, then the committed value")
    shown = committed_summary.set_index("regime")
    for _, row in summary.iterrows():
        regime = row["regime"]
        for metric in ("defensible_macro_f1", "macro_f1", "routing_f1", "for_f1", "against_f1"):
            print(f"  {regime:10s} {metric:20s} {row[metric]:.7f}  {shown.loc[regime, metric]:.7f}")
    print("\nPublished files check:", "PASS" if ok else "FAIL")
    return bool(ok)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="the run folder, such as runs/modernbert-4096")
    parser.add_argument("--out-dir", help="where the files go, default results/heads/<run name>")
    parser.add_argument("--check-only", action="store_true", help="skip the build and check the published files")
    parser.add_argument("--no-check", action="store_true", help="build without the check")
    args = parser.parse_args(argv)

    if args.check_only:
        if not args.out_dir and not args.run:
            parser.error("--check-only needs --run or --out-dir")
        out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / Path(args.run).name
    else:
        if not args.run:
            parser.error("--run is required")
        run_dir = Path(args.run)
        out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / run_dir.name
        build(run_dir, out_dir)
        for path in sorted(out_dir.iterdir()):
            print(f"wrote {path}  {path.stat().st_size:,} bytes")
    if not args.no_check or args.check_only:
        if not check(out_dir):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
