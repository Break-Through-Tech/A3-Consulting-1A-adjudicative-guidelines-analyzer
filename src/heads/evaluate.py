"""Calibrate, threshold and score one run in the same tables as the 04 baseline.

    python -m src.heads.evaluate --run runs/legalbert-512

Steps, per guideline head:
1. Fit a temperature on val by LBFGS on the masked negative log-likelihood.
2. Pick the at-issue threshold on the calibrated val probabilities with the
   baseline's ``choose_at_issue_threshold``.
3. Decode test with the baseline's ``decode_predictions`` and score it with the
   baseline's ``classification_metrics``.

Writes, under ``results/heads/``:
    <run>.csv              per-guideline metrics by regime, the 04 columns
    <run>_summary.csv      by-regime summary, the 04 columns
    <run>_calibration.csv  reliability bins on test, before and after scaling
    comparison.csv         every evaluated run next to the TF-IDF baseline
and, under the run folder, ``test_probabilities.parquet`` and
``calibration.json`` for integration and the app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.heads import data as heads_data
from src.heads.data import CLASSES, DEFENSIBLE, GUIDELINES, IGNORE_INDEX, baseline, head_labels

PROJECT_ROOT = heads_data.PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results" / "heads"
BASELINE_RESULTS = PROJECT_ROOT / "guideline_baseline_frozen_split.csv"
BASELINE_SUMMARY = PROJECT_ROOT / "guideline_baseline_frozen_split_summary.csv"
BASELINE_NAME = "tfidf-logreg"
N_BINS = 15

COMPARISON_METRICS = [
    "defensible_macro_f1",
    "macro_f1",
    "accuracy",
    "routing_f1",
    "routing_average_precision",
    "for_f1",
    "against_f1",
    "macro_average_precision",
]


# --- shared helpers, also used by train.py ----------------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def head_metrics(logits: np.ndarray, targets: np.ndarray) -> tuple[dict[str, dict], float]:
    """Per-head metrics from argmax predictions, and the defensible-8 macro-F1 mean.

    Used to pick the best checkpoint on val. The final test numbers use the
    calibrated threshold decode instead.
    """
    probabilities = softmax(logits)
    per_head: dict[str, dict] = {}
    for head, code in enumerate(GUIDELINES):
        known = targets[:, head] != IGNORE_INDEX
        if not known.any():
            continue
        y_true = pd.Series(head_labels(targets[known, head]))
        y_pred = head_labels(probabilities[known, head].argmax(axis=-1))
        per_head[code] = baseline().classification_metrics(y_true, y_pred, probabilities[known, head])
    scores = [per_head[code]["macro_f1"] for code in DEFENSIBLE if code in per_head]
    return per_head, float(np.mean(scores)) if scores else float("nan")


# --- calibration -------------------------------------------------------------


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """The temperature that minimises NLL of softmax(logits / T) on known targets."""
    if len(targets) == 0:
        return 1.0
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.detach().exp())
    if not np.isfinite(temperature):
        return 1.0
    return float(np.clip(temperature, 0.05, 20.0))


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    if len(targets) == 0:
        return float("nan")
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64) / temperature
    return float(F.cross_entropy(x, torch.as_tensor(np.asarray(targets), dtype=torch.long)))


def reliability(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = N_BINS) -> tuple[float, pd.DataFrame]:
    """Top-label expected calibration error and its equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(targets) == 0:
        return float("nan"), pd.DataFrame()
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        count = int(in_bin.sum())
        mean_conf = float(confidence[in_bin].mean()) if count else float("nan")
        accuracy = float(correct[in_bin].mean()) if count else float("nan")
        if count:
            ece += count / len(targets) * abs(accuracy - mean_conf)
        rows.append(
            {
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
            }
        )
    return float(ece), pd.DataFrame(rows)


# --- the 04 tables -----------------------------------------------------------


def baseline_tables(test_df: pd.DataFrame, per_guideline: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-guideline and summary tables, built exactly as the 04 script builds them."""
    tiers = baseline().GUIDELINE_TIERS
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for regime, regime_df in test_df.groupby("guideline_regime", dropna=False):
        regime_name = "missing" if pd.isna(regime) else str(regime)
        regime_positions = regime_df.index.to_numpy()
        all_true: list[str] = []
        all_predicted: list[str] = []
        all_probabilities: list[np.ndarray] = []

        for code in GUIDELINES:
            result = per_guideline[code]
            targets = result["targets"]
            known_regime = targets.index.isin(regime_positions)
            y_true = targets.loc[known_regime]
            known_positions = np.flatnonzero(known_regime)
            y_pred = result["predictions"][known_positions]
            probabilities = result["probabilities"][known_positions]
            n_cases = int(regime_df.loc[y_true.index, "case_number"].nunique())
            rows.append(
                {
                    "regime": regime_name,
                    "guideline": code,
                    "tier": tiers[code],
                    "regime_test_cases": int(regime_df["case_number"].nunique()),
                    "known_test_cases": n_cases,
                    "at_issue_threshold": result["threshold"],
                    **baseline().classification_metrics(y_true, y_pred, probabilities),
                    "train_class_counts": ";".join(
                        f"{label}={int(count)}" for label, count in result["train_class_counts"].items()
                    ),
                    "val_known_pairs": int(len(result["val_targets"])),
                    "trained_classes": result["trained_classes"],
                }
            )
            all_true.extend(y_true.tolist())
            all_predicted.extend(y_pred.tolist())
            all_probabilities.extend(probabilities)

        if all_true:
            aggregate = baseline().classification_metrics(
                pd.Series(all_true), np.asarray(all_predicted), np.asarray(all_probabilities)
            )
            regime_defensible_macro_f1 = np.mean(
                [row["macro_f1"] for row in rows if row["regime"] == regime_name and row["tier"] == "defensible"]
            )
            summary_rows.append(
                {
                    "regime": regime_name,
                    "test_cases": int(regime_df["case_number"].nunique()),
                    "defensible_macro_f1": float(regime_defensible_macro_f1),
                    **aggregate,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


# --- comparison --------------------------------------------------------------


def _comparison_block(
    run: str, model: str, max_length: object, summary: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    block = summary[["regime", "test_cases", *COMPARISON_METRICS]].copy()
    f_rows = results.loc[results["guideline"].eq("F")].set_index("regime")
    block["f_for_f1"] = block["regime"].map(f_rows["for_f1"])
    block["f_against_f1"] = block["regime"].map(f_rows["against_f1"])
    block.insert(0, "max_length", max_length)
    block.insert(0, "model", model)
    block.insert(0, "run", run)
    return block


def update_comparison(
    results_dir: Path,
    run: str,
    model: str,
    max_length: int,
    summary: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Upsert this run's rows into comparison.csv, with TF-IDF refreshed first."""
    path = results_dir / "comparison.csv"
    blocks = []
    if BASELINE_SUMMARY.exists() and BASELINE_RESULTS.exists():
        blocks.append(
            _comparison_block(
                BASELINE_NAME,
                "TF-IDF + logistic regression",
                "",
                pd.read_csv(BASELINE_SUMMARY),
                pd.read_csv(BASELINE_RESULTS),
            )
        )
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False, na_values=[""])
        previous = previous.loc[~previous["run"].isin([BASELINE_NAME, run])]
        if len(previous):
            blocks.append(previous)
    blocks.append(_comparison_block(run, model, max_length, summary, results))
    table = pd.concat(blocks, ignore_index=True)

    base = table.loc[table["run"].eq(BASELINE_NAME)].set_index("regime")["defensible_macro_f1"]
    table["delta_defensible_macro_f1_vs_tfidf"] = table["defensible_macro_f1"] - table["regime"].map(base)
    run_order = {name: i for i, name in enumerate(dict.fromkeys(table["run"]))}
    table["_order"] = table["run"].map(run_order)
    table = table.sort_values(["regime", "_order"], kind="stable").drop(columns="_order")
    table.to_csv(path, index=False)
    return table


# --- the run -----------------------------------------------------------------


def _load_rows(run_dir: Path, split: str, df: pd.DataFrame) -> np.ndarray:
    cases = pd.read_csv(run_dir / f"cases_{split}.csv")
    rows = cases["row"].to_numpy()
    if rows.max(initial=-1) >= len(df):
        raise ValueError(f"cases_{split}.csv points past the pool; was the data rebuilt?")
    stored = cases["case_number"].astype(str).to_numpy()
    current = df["case_number"].iloc[rows].astype(str).to_numpy()
    if not np.array_equal(stored, current):
        raise ValueError(f"cases_{split}.csv does not match the pool order; was the data rebuilt?")
    return rows


def evaluate_run(run_dir: Path, results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    name = run_config["args"]["run_name"]
    model_name = run_config["args"]["model"]
    max_length = int(run_config["args"]["max_length"])

    df = heads_data.load_pool()
    targets = heads_data.target_matrix(df)
    rows = {split: _load_rows(run_dir, split, df) for split in heads_data.SPLITS}
    logits = {split: np.load(run_dir / f"logits_{split}.npy") for split in ("val", "test")}
    for split in ("val", "test"):
        if logits[split].shape != (len(rows[split]), len(GUIDELINES), len(CLASSES)):
            raise ValueError(f"logits_{split}.npy has shape {logits[split].shape}")

    split_targets = {split: targets[rows[split]] for split in heads_data.SPLITS}
    test_df = df.iloc[rows["test"]]
    per_guideline: dict[str, dict] = {}
    calibration_frames: list[pd.DataFrame] = []
    calibration_info: dict[str, dict] = {}
    test_calibrated = np.zeros_like(logits["test"], dtype=np.float64)
    thresholds: dict[str, float] = {}

    for head, code in enumerate(GUIDELINES):
        val_y = split_targets["val"][:, head]
        test_y = split_targets["test"][:, head]
        train_y = split_targets["train"][:, head]
        val_known = val_y != IGNORE_INDEX
        test_known = test_y != IGNORE_INDEX

        val_logits = logits["val"][:, head].astype(np.float64)
        test_logits = logits["test"][:, head].astype(np.float64)
        temperature = fit_temperature(val_logits[val_known], val_y[val_known])
        val_calibrated = softmax(val_logits / temperature)
        test_raw = softmax(test_logits)
        test_calibrated[:, head] = softmax(test_logits / temperature)

        val_labels = pd.Series(head_labels(val_y[val_known]))
        threshold = baseline().choose_at_issue_threshold(val_labels, val_calibrated[val_known])
        thresholds[code] = threshold
        predictions = baseline().decode_predictions(test_calibrated[test_known, head], threshold)
        train_counts = pd.Series(
            np.bincount(train_y[train_y != IGNORE_INDEX], minlength=len(CLASSES)), index=list(CLASSES)
        )
        per_guideline[code] = {
            "targets": pd.Series(head_labels(test_y[test_known]), index=test_df.index[test_known]),
            "predictions": predictions,
            "probabilities": test_calibrated[test_known, head],
            "threshold": threshold,
            "val_targets": val_labels,
            "train_class_counts": train_counts,
            "trained_classes": ",".join(sorted(label for label, count in train_counts.items() if count > 0)),
        }

        ece_raw, bins_raw = reliability(test_raw[test_known], test_y[test_known])
        ece_cal, bins_cal = reliability(test_calibrated[test_known, head], test_y[test_known])
        calibration_info[code] = {
            "temperature": temperature,
            "at_issue_threshold": threshold,
            "val_nll_before": nll(val_logits[val_known], val_y[val_known]),
            "val_nll_after": nll(val_logits[val_known], val_y[val_known], temperature),
            "test_ece_before": ece_raw,
            "test_ece_after": ece_cal,
            "test_known_pairs": int(test_known.sum()),
        }
        if code in DEFENSIBLE:
            for stage, bins, ece in (("before", bins_raw, ece_raw), ("after", bins_cal, ece_cal)):
                frame = bins.copy()
                frame.insert(0, "ece", ece)
                frame.insert(0, "stage", stage)
                frame.insert(0, "temperature", temperature)
                frame.insert(0, "guideline", code)
                calibration_frames.append(frame)

    results, summary = baseline_tables(test_df, per_guideline)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    results.to_csv(results_dir / f"{name}.csv", index=False)
    summary.to_csv(results_dir / f"{name}_summary.csv", index=False)
    calibration.to_csv(results_dir / f"{name}_calibration.csv", index=False)
    comparison = update_comparison(results_dir, name, model_name, max_length, summary, results)

    (run_dir / "calibration.json").write_text(json.dumps(calibration_info, indent=2), encoding="utf-8")
    long_rows = []
    test_cases = df["case_number"].iloc[rows["test"]].to_numpy()
    for head, code in enumerate(GUIDELINES):
        probabilities = test_calibrated[:, head]
        y = split_targets["test"][:, head]
        long_rows.append(
            pd.DataFrame(
                {
                    "row": rows["test"],
                    "case_number": test_cases,
                    "guideline": code,
                    "target": np.where(y == IGNORE_INDEX, None, head_labels(np.clip(y, 0, None))),
                    "p_not_at_issue": probabilities[:, 0],
                    "p_for": probabilities[:, 1],
                    "p_against": probabilities[:, 2],
                    "temperature": calibration_info[code]["temperature"],
                    "at_issue_threshold": thresholds[code],
                    "prediction": baseline().decode_predictions(probabilities, thresholds[code]),
                }
            )
        )
    pd.concat(long_rows, ignore_index=True).to_parquet(run_dir / "test_probabilities.parquet", index=False)

    return {"results": results, "summary": summary, "calibration": calibration, "comparison": comparison, "info": calibration_info}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="the run folder, such as runs/legalbert-512")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="where the result tables go")
    args = parser.parse_args(argv)

    out = evaluate_run(Path(args.run), Path(args.results_dir))
    fmt = lambda value: f"{value:.3f}"  # noqa: E731
    print("=== Test summary by regime ===")
    print(out["summary"][["regime", "test_cases", *COMPARISON_METRICS]].to_string(index=False, float_format=fmt))
    print("\n=== Calibration per head: temperature, test ECE before and after ===")
    info = pd.DataFrame(out["info"]).T[["temperature", "test_ece_before", "test_ece_after", "at_issue_threshold"]]
    print(info.astype(float).to_string(float_format=fmt))
    print("\n=== Comparison ===")
    columns = ["run", "regime", "defensible_macro_f1", "delta_defensible_macro_f1_vs_tfidf", "routing_f1", "for_f1", "against_f1"]
    print(out["comparison"][columns].to_string(index=False, float_format=fmt))
    print(f"\nTables written to {Path(args.results_dir)}")


if __name__ == "__main__":
    main()
