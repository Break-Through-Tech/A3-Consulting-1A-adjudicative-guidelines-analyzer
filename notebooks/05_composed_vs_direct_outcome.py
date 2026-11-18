"""Compare composed and directly-trained GRANTED/DENIED predictions.

The composed recommendation uses the 13 frozen-split guideline heads:
DENIED when at least one head predicts AGAINST, GRANTED otherwise.

The direct recommendation trains a separate TF-IDF + logistic outcome head.
Both are evaluated once on test and broken out by citation regime.
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
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_cases
from src.outcome_comparison import (
    OUTCOME_CLASSES,
    compose_outcomes,
    outcome_metrics,
)

TEXT_COL = "text_features"
SPLITS = ("train", "val", "test")
DIRECTION_CLASSES = ("NOT_AT_ISSUE", "FOR", "AGAINST")
DIRECTIONS = ("FOR", "AGAINST")
MAX_FEATURES = 20_000


def targets_for_guideline(df: pd.DataFrame, code: str) -> pd.Series:
    """Build targets while masking unobserved directions on listed guidelines."""
    listed = df["guidelines"].map(lambda codes: code in codes)
    findings = df["guideline_findings"].map(lambda values: values.get(code))

    targets = pd.Series(pd.NA, index=df.index, dtype="object")
    targets.loc[~listed] = "NOT_AT_ISSUE"
    directional = listed & findings.isin(DIRECTIONS)
    targets.loc[directional] = findings.loc[directional]
    return targets


def aligned_probabilities(
    model: LogisticRegression,
    features,
    classes: tuple[str, ...] = DIRECTION_CLASSES,
) -> np.ndarray:
    probabilities = np.zeros((features.shape[0], len(classes)), dtype=float)
    model_probabilities = model.predict_proba(features)
    for column, label in enumerate(model.classes_):
        probabilities[:, classes.index(label)] = model_probabilities[:, column]
    return probabilities


def choose_at_issue_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    issue_scores = probabilities[:, 1] + probabilities[:, 2]
    is_at_issue = y_true.to_numpy() != "NOT_AT_ISSUE"
    if not is_at_issue.any():
        return 1.01

    candidates = np.unique(np.append(issue_scores, 1.01))
    scores = [
        float(
            f1_score(
                is_at_issue,
                issue_scores >= threshold,
                zero_division=0,
            )
        )
        for threshold in candidates
    ]
    best = max(scores)
    return float(max(threshold for threshold, score in zip(candidates, scores) if score == best))


def decode_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    issue_scores = probabilities[:, 1] + probabilities[:, 2]
    predictions = np.full(len(issue_scores), "NOT_AT_ISSUE", dtype=object)
    at_issue = issue_scores >= threshold
    predictions[at_issue] = np.where(
        probabilities[at_issue, 1] >= probabilities[at_issue, 2],
        "FOR",
        "AGAINST",
    )
    return predictions


def main() -> None:
    df = load_cases(
        pool=True,
        columns=[
            "case_number",
            "guidelines",
            "guideline_findings",
            "guideline_regime",
            "outcome",
            TEXT_COL,
            "split",
        ],
    )
    if df["split"].isna().any():
        raise ValueError("Some trainable cases are missing frozen split assignments")
    unexpected_splits = set(df["split"].unique()) - set(SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected frozen split values: {sorted(unexpected_splits)}")

    df = df.loc[
        df["outcome"].isin(OUTCOME_CLASSES)
        & df[TEXT_COL].fillna("").str.strip().ne("")
    ].copy()
    train_mask = df["split"].eq("train")
    val_mask = df["split"].eq("val")
    test_mask = df["split"].eq("test")

    print(
        f"Loaded {len(df):,} outcome-labelled cases; "
        f"train={int(train_mask.sum()):,}, val={int(val_mask.sum()):,}, "
        f"test={int(test_mask.sum()):,}"
    )

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

    composed_predictions: dict[str, np.ndarray] = {}
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

        full_test_predictions = np.full(int(test_mask.sum()), "NOT_AT_ISSUE", dtype=object)
        full_test_predictions[np.flatnonzero(test_known)] = test_predictions
        composed_predictions[code] = full_test_predictions

    test_df = df.loc[test_mask].reset_index(drop=True)
    composed = compose_outcomes(composed_predictions)
    test_df["composed_prediction"] = composed

    direct_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    y_direct_train = df.loc[train_mask, "outcome"]
    direct_model.fit(x_train, y_direct_train)
    direct = direct_model.predict(x_test)
    test_df["direct_prediction"] = direct
    test_df["true_outcome"] = test_df["outcome"]

    comparison_rows: list[dict[str, object]] = []
    for method in ("composed", "direct"):
        for regime, regime_df in test_df.groupby("guideline_regime", dropna=False):
            regime_name = "missing" if pd.isna(regime) else str(regime)
            y_pred = regime_df[f"{method}_prediction"].to_numpy()
            comparison_rows.append(
                {
                    "method": method,
                    "regime": regime_name,
                    **outcome_metrics(regime_df["outcome"], y_pred),
                }
            )

        overall = outcome_metrics(
            test_df["outcome"],
            test_df[f"{method}_prediction"].to_numpy(),
        )
        comparison_rows.append(
            {
                "method": method,
                "regime": "overall",
                **overall,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = PROJECT_ROOT / "composed_vs_direct_outcome_by_regime.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n=== Composed vs direct frozen-test outcome metrics ===")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMetrics saved to {comparison_path}")

    for method in ("composed", "direct"):
        y_pred = test_df[f"{method}_prediction"].to_numpy()
        print(f"\n=== {method.title()} confusion matrix (rows=true, cols=predicted) ===")
        print(
            pd.DataFrame(
                confusion_matrix(test_df["outcome"], y_pred, labels=list(OUTCOME_CLASSES)),
                index=OUTCOME_CLASSES,
                columns=OUTCOME_CLASSES,
            ).to_string()
        )
        print(classification_report(
            test_df["outcome"],
            y_pred,
            labels=list(OUTCOME_CLASSES),
            digits=3,
            zero_division=0,
        ))


if __name__ == "__main__":
    main()
