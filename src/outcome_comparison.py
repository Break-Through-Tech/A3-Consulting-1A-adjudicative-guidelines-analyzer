"""Helpers for composed and directly-trained outcome prediction."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


OUTCOME_CLASSES = ("DENIED", "GRANTED")
DIRECTION_CLASSES = ("NOT_AT_ISSUE", "FOR", "AGAINST")
DIRECTIONS = ("FOR", "AGAINST")


def compose_outcome(predictions: Mapping[str, str]) -> str:
    """Return DENIED if any guideline head predicts AGAINST, otherwise GRANTED."""
    return "DENIED" if "AGAINST" in predictions.values() else "GRANTED"


def compose_outcomes(predictions_by_guideline: Mapping[str, Iterable[str]]) -> np.ndarray:
    """Compose one outcome array from a mapping of guideline-code to predictions."""
    arrays = {code: np.asarray(values, dtype=object) for code, values in predictions_by_guideline.items()}
    if not arrays:
        raise ValueError("At least one guideline prediction is required")
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("All guideline prediction arrays must have the same length")
    length = lengths.pop()
    result = np.full(length, "GRANTED", dtype=object)
    for values in arrays.values():
        result[values == "AGAINST"] = "DENIED"
    return result


def outcome_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(OUTCOME_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_cases": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "denied_precision": float(report["DENIED"]["precision"]),
        "denied_recall": float(report["DENIED"]["recall"]),
        "denied_f1": float(report["DENIED"]["f1-score"]),
        "denied_support": int(report["DENIED"]["support"]),
        "granted_precision": float(report["GRANTED"]["precision"]),
        "granted_recall": float(report["GRANTED"]["recall"]),
        "granted_f1": float(report["GRANTED"]["f1-score"]),
        "granted_support": int(report["GRANTED"]["support"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
