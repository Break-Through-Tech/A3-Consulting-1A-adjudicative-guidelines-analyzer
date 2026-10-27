"""Step 6: TF-IDF + Logistic Regression baseline for outcome prediction.

Leak-free version. Two things differ from the first baseline:

1. Features come from `text_features`, the cleaned text built from the
   leak-free sections only (verdict region excluded), not from `full_text`.
2. The committed frozen split is used, so results are comparable with later
   models. It groups cases by case_number and provides train, val, and test.

The validation split is held out. The model trains on train and is evaluated
once on test. The earlier random grouped split scored about 0.827; the
full_text baseline scored about 0.954, mostly from reading the ruling.

Reads Challenge_Data/ through `src.dataset.load_cases()`.

Run from anywhere:
    python notebooks\02_baseline.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_cases

# The corpus has 5 outcome values, but REMANDED (110) and REVOKED (7) are too
# rare to learn and UNKNOWN means "the parser could not tell". Keep the two
# real classes.
KEEP = ["DENIED", "GRANTED"]

TEXT_COL = "text_features"
SPLITS = ("train", "val", "test")


def main():
    df = load_cases(
        pool=True,
        columns=["case_number", "guideline_regime", "outcome", TEXT_COL, "split"],
    )
    print(f"Loaded {len(df):,} trainable rows with frozen split assignments")

    if df["split"].isna().any():
        raise ValueError("Some trainable rows are missing a frozen split assignment")
    unexpected_splits = set(df["split"].unique()) - set(SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected frozen split values: {sorted(unexpected_splits)}")

    before = len(df)
    df = df[df["outcome"].isin(KEEP)]
    print(f"Dropped {before - len(df):,} rows outside {KEEP} -> {len(df):,} rows")

    empty = df[TEXT_COL].str.strip() == ""
    if empty.any():
        print(f"Dropped {int(empty.sum()):,} rows with empty {TEXT_COL}")
        df = df[~empty]

    parts = {name: df.loc[df["split"] == name].copy() for name in SPLITS}
    for name, part in parts.items():
        print(
            f"{name.capitalize():5} {len(part):,} rows "
            f"({part['case_number'].nunique():,} case numbers)"
        )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(parts[left]["case_number"]) & set(parts[right]["case_number"])
        if overlap:
            raise ValueError(
                f"Frozen split leakage: {len(overlap)} case numbers in {left} and {right}"
            )

    train, test = parts["train"], parts["test"]
    X_train, y_train = train[TEXT_COL], train["outcome"]
    X_test, y_test = test[TEXT_COL], test["outcome"]

    print("\nVectorizing...")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"  {Xtr.shape[1]:,} features")

    print("Fitting...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)

    # A model that always guesses the majority class is the bar to beat.
    baseline = y_test.value_counts(normalize=True).max()
    accuracy = accuracy_score(y_test, preds)
    print(f"\nMajority-class accuracy to beat: {baseline:.3f}")
    print(f"Frozen test accuracy: {accuracy:.3f}")

    print("\n=== Classification report ===")
    print(classification_report(y_test, preds, labels=KEEP, digits=3, zero_division=0))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=KEEP),
        index=KEEP, columns=KEEP,
    ).to_string())

    print("\n=== Frozen test metrics by citation regime ===")
    for regime, regime_test in test.groupby("guideline_regime", dropna=False):
        regime_preds = clf.predict(vec.transform(regime_test[TEXT_COL]))
        regime_accuracy = accuracy_score(regime_test["outcome"], regime_preds)
        regime_name = "missing" if pd.isna(regime) else regime
        print(f"\n{regime_name} (n={len(regime_test):,}, accuracy={regime_accuracy:.3f})")
        print(classification_report(
            regime_test["outcome"],
            regime_preds,
            labels=KEEP,
            digits=3,
            zero_division=0,
        ))

    # Which words drive the prediction. With the verdict excluded, words like
    # "denied" or "granted" should no longer dominate these lists. If they do,
    # something is still leaking.
    names = vec.get_feature_names_out()
    coefs = pd.Series(clf.coef_[0], index=names).sort_values()
    print(f"\n=== Top 15 features -> {clf.classes_[0]} ===")
    print(coefs.head(15).to_string())
    print(f"\n=== Top 15 features -> {clf.classes_[1]} ===")
    print(coefs.tail(15)[::-1].to_string())


if __name__ == "__main__":
    main()