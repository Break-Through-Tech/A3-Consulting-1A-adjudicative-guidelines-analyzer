"""Frozen-split TF-IDF baseline for 13 three-way guideline classifiers.

Each guideline predicts NOT_AT_ISSUE, FOR, or AGAINST from leak-free
``text_features``. Cases with a listed guideline but no formal direction are
masked for that guideline, not treated as NOT_AT_ISSUE.

The model is trained on train, its at-issue threshold is selected on val, and
metrics are reported on test, overall and by citation regime.

Run from the repository root or any working directory:
    python notebooks\\04_guideline_classifier_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_cases

TEXT_COL = "text_features"
SPLITS = ("train", "val", "test")
CLASSES = ("NOT_AT_ISSUE", "FOR", "AGAINST")
DIRECTIONS = ("FOR", "AGAINST")
MAX_FEATURES = 20_000
DEFENSIBLE_GUIDELINES = {"B", "C", "D", "E", "F", "G", "H", "J"}
GUIDELINE_TIERS = {
    code: "defensible" if code in DEFENSIBLE_GUIDELINES else "caveat"
    for code in config.GUIDELINE_CODES
}


def targets_for_guideline(df: pd.DataFrame, code: str) -> pd.Series:
    """Build targets, masking listed guidelines with no observed direction."""
    listed = df["guidelines"].map(lambda codes: code in codes)
    findings = df["guideline_findings"].map(lambda values: values.get(code))

    targets = pd.Series(pd.NA, index=df.index, dtype="object")
    targets.loc[~listed] = "NOT_AT_ISSUE"
    directional = listed & findings.isin(DIRECTIONS)
    targets.loc[directional] = findings.loc[directional]
    return targets


def aligned_probabilities(
    model: LogisticRegression, features, classes: tuple[str, ...] = CLASSES
) -> np.ndarray:
    """Return probabilities in the fixed class order, including absent classes."""
    probabilities = np.zeros((features.shape[0], len(classes)), dtype=float)
    model_probabilities = model.predict_proba(features)
    for column, label in enumerate(model.classes_):
        probabilities[:, classes.index(label)] = model_probabilities[:, column]
    return probabilities


def choose_at_issue_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Maximize at-issue F1 on validation, preferring higher thresholds on ties."""
    issue_scores = probabilities[:, 1] + probabilities[:, 2]
    is_at_issue = y_true.to_numpy() != "NOT_AT_ISSUE"
    if not is_at_issue.any():
        return 1.01

    candidates = np.unique(np.append(issue_scores, 1.01))
    scores = [
        f1_score(is_at_issue, issue_scores >= threshold, zero_division=0)
        for threshold in candidates
    ]
    best = max(scores)
    return float(max(threshold for threshold, score in zip(candidates, scores) if score == best))


def decode_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the at-issue threshold, then select the more likely direction."""
    issue_scores = probabilities[:, 1] + probabilities[:, 2]
    predictions = np.full(len(issue_scores), "NOT_AT_ISSUE", dtype=object)
    at_issue = issue_scores >= threshold
    predictions[at_issue] = np.where(
        probabilities[at_issue, 1] >= probabilities[at_issue, 2],
        "FOR",
        "AGAINST",
    )
    return predictions


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
    is_at_issue = y_true.to_numpy() != "NOT_AT_ISSUE"
    predicted_at_issue = y_pred != "NOT_AT_ISSUE"
    ap = (
        float(average_precision_score(is_at_issue, probabilities[:, 1] + probabilities[:, 2]))
        if is_at_issue.any()
        else None
    )
    metrics: dict[str, float | int | None] = {
        "n_pairs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="micro", zero_division=0)),
        "routing_precision": float(
            precision_recall_fscore_support(
                is_at_issue, predicted_at_issue, average="binary", zero_division=0
            )[0]
        ),
        "routing_recall": float(
            precision_recall_fscore_support(
                is_at_issue, predicted_at_issue, average="binary", zero_division=0
            )[1]
        ),
        "routing_f1": float(
            f1_score(is_at_issue, predicted_at_issue, zero_division=0)
        ),
        "routing_average_precision": ap,
    }
    class_average_precisions: list[float] = []
    for index, (label, p, r, score, count) in enumerate(
        zip(CLASSES, precision, recall, f1, support)
    ):
        metrics[f"{label.lower()}_precision"] = float(p)
        metrics[f"{label.lower()}_recall"] = float(r)
        metrics[f"{label.lower()}_f1"] = float(score)
        metrics[f"{label.lower()}_support"] = int(count)
        class_ap = (
            float(
                average_precision_score(
                    y_true.to_numpy() == label,
                    probabilities[:, index],
                )
            )
            if count
            else None
        )
        metrics[f"{label.lower()}_average_precision"] = class_ap
        if class_ap is not None:
            class_average_precisions.append(class_ap)
    metrics["macro_average_precision"] = (
        float(np.mean(class_average_precisions)) if class_average_precisions else None
    )
    return metrics


def main() -> None:
    df = load_cases(
        pool=True,
        columns=[
            "case_number",
            "guidelines",
            "guideline_findings",
            "guideline_regime",
            TEXT_COL,
            "split",
        ],
    )
    if df["split"].isna().any():
        raise ValueError("Some trainable cases are missing frozen split assignments")
    if not set(df["split"].unique()).issubset(SPLITS):
        raise ValueError(f"Unexpected split values: {sorted(set(df['split'].unique()))}")

    train_mask = df["split"].eq("train")
    val_mask = df["split"].eq("val")
    test_mask = df["split"].eq("test")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
    )
    x_train = vectorizer.fit_transform(df.loc[train_mask, TEXT_COL])
    x_val = vectorizer.transform(df.loc[val_mask, TEXT_COL])
    x_test = vectorizer.transform(df.loc[test_mask, TEXT_COL])
    train_positions = np.flatnonzero(train_mask.to_numpy())
    val_positions = np.flatnonzero(val_mask.to_numpy())
    test_positions = np.flatnonzero(test_mask.to_numpy())

    print(
        f"Loaded {len(df):,} trainable cases; "
        f"train={int(train_mask.sum()):,}, val={int(val_mask.sum()):,}, "
        f"test={int(test_mask.sum()):,}; features={x_train.shape[1]:,}"
    )

    per_guideline: dict[str, dict[str, object]] = {}
    for code in config.GUIDELINE_CODES:
        targets = targets_for_guideline(df, code)
        train_targets = targets.iloc[train_positions]
        train_known = train_targets.notna().to_numpy()
        y_train = train_targets.loc[train_known].to_numpy()

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_train[train_known], y_train)

        val_targets = targets.iloc[val_positions]
        val_known = val_targets.notna().to_numpy()
        val_probabilities = aligned_probabilities(model, x_val[val_known])
        threshold = choose_at_issue_threshold(
            val_targets.loc[val_known],
            val_probabilities,
        )

        test_targets = targets.iloc[test_positions]
        test_known = test_targets.notna().to_numpy()
        test_probabilities = aligned_probabilities(model, x_test[test_known])
        test_predictions = decode_predictions(test_probabilities, threshold)
        per_guideline[code] = {
            "targets": test_targets.loc[test_known],
            "predictions": test_predictions,
            "probabilities": test_probabilities,
            "threshold": threshold,
            "model": model,
            "val_targets": val_targets.loc[val_known],
            "train_class_counts": train_targets.value_counts(dropna=True).reindex(
                CLASSES, fill_value=0
            ),
        }

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    test_df = df.loc[test_mask]
    for regime, regime_df in test_df.groupby("guideline_regime", dropna=False):
        regime_name = "missing" if pd.isna(regime) else str(regime)
        regime_positions = regime_df.index.to_numpy()
        all_true: list[str] = []
        all_predicted: list[str] = []
        all_probabilities: list[np.ndarray] = []

        for code in config.GUIDELINE_CODES:
            result = per_guideline[code]
            targets = result["targets"]
            known_regime = targets.index.isin(regime_positions)
            y_true = targets.loc[known_regime]
            # The stored prediction arrays correspond only to known target rows.
            known_positions = np.flatnonzero(
                targets.index.isin(regime_positions)
            )
            y_pred = result["predictions"][known_positions]
            probabilities = result["probabilities"][known_positions]
            n_cases = int(regime_df.loc[y_true.index, "case_number"].nunique())

            val_targets = result["val_targets"]
            rows.append(
                {
                    "regime": regime_name,
                    "guideline": code,
                    "tier": GUIDELINE_TIERS[code],
                    "regime_test_cases": int(regime_df["case_number"].nunique()),
                    "known_test_cases": n_cases,
                    "at_issue_threshold": result["threshold"],
                    **classification_metrics(y_true, y_pred, probabilities),
                    "train_class_counts": ";".join(
                        f"{label}={int(count)}"
                        for label, count in result["train_class_counts"].items()
                    ),
                    "val_known_pairs": int(len(val_targets)),
                    "trained_classes": ",".join(result["model"].classes_),
                }
            )
            all_true.extend(y_true.tolist())
            all_predicted.extend(y_pred.tolist())
            all_probabilities.extend(probabilities)

        if all_true:
            aggregate = classification_metrics(
                pd.Series(all_true),
                np.asarray(all_predicted),
                np.asarray(all_probabilities),
            )
            regime_defensible_macro_f1 = np.mean(
                [
                    row["macro_f1"]
                    for row in rows
                    if row["regime"] == regime_name and row["tier"] == "defensible"
                ]
            )
            summary_rows.append(
                {
                    "regime": regime_name,
                    "test_cases": int(regime_df["case_number"].nunique()),
                    "defensible_macro_f1": float(regime_defensible_macro_f1),
                    **aggregate,
                }
            )

    results = pd.DataFrame(rows)
    results_path = PROJECT_ROOT / "guideline_baseline_frozen_split.csv"
    results.to_csv(results_path, index=False)

    print("\n=== Frozen test metrics by citation regime (13 guideline heads) ===")
    summary = pd.DataFrame(summary_rows)
    summary_path = PROJECT_ROOT / "guideline_baseline_frozen_split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n=== Per-guideline metrics by regime ===")
    display_columns = [
        "regime",
        "guideline",
        "regime_test_cases",
        "known_test_cases",
        "n_pairs",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "routing_f1",
        "routing_average_precision",
        "for_average_precision",
        "against_average_precision",
        "for_support",
        "against_support",
    ]
    print(results[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nPer-guideline metrics saved to {results_path}")
    print(f"Regime summary saved to {summary_path}")

    a_rows = results.loc[results["guideline"].eq("A")]
    if not (a_rows[["for_support", "against_support"]].to_numpy().sum()):
        print(
            "\nCaveat: the frozen test split contains no FOR/AGAINST Guideline A "
            "examples; its test metrics only reflect NOT_AT_ISSUE."
        )


if __name__ == "__main__":
    main()
