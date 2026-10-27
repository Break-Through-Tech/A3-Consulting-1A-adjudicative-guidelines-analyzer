"""Step 6: TF-IDF + Logistic Regression baseline for outcome prediction.

Leak-free version. Two things differ from the first baseline:

1. Features come from `text_features`, the cleaned text built from the
   leak-free sections only (verdict region excluded), not from `full_text`.
2. The train/test split is grouped by case_number, so a hearing and its appeal
   (which share a case number) can never land on opposite sides of the split.

Expected accuracy is about 0.827. The earlier full_text baseline scored about
0.954, which was mostly the model reading the ruling.

Reads Challenge_Data/ (built by `python -m src.build_dataset --full`).

Run from anywhere:
    python notebooks\02_baseline.py                  # full corpus
    python notebooks\02_baseline.py --sample 2000    # quick smoke test
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# parents[0] is notebooks/, parents[1] is the project root.
DATA_DIR = Path(__file__).resolve().parents[1] / "Challenge_Data"

# The corpus has 5 outcome values, but REMANDED (110) and REVOKED (7) are too
# rare to learn and UNKNOWN means "the parser could not tell". Keep the two
# real classes.
KEEP = ["DENIED", "GRANTED"]

TEXT_COL = "text_features"
GROUP_COL = "case_number"
EXPECTED_ACCURACY = 0.827


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset (all shards) from Challenge_Data/.

    src.load only has load_raw(), which reads doha's untouched shards. This
    reads the repaired output instead. Only the columns the baseline needs
    are read.
    """
    files = sorted(glob.glob(str(DATA_DIR / "cases_clean_*.parquet")))
    if not files:
        raise SystemExit(
            f"No parquet files found in {DATA_DIR}\n"
            "Build the dataset first from the project root:\n"
            "    python -m src.build_dataset --full"
        )
    cols = [TEXT_COL, GROUP_COL, "outcome"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} file(s)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="use only N rows (faster; omit for full corpus)")
    args = ap.parse_args()

    df = load_clean()

    df = df.dropna(subset=[TEXT_COL, GROUP_COL, "outcome"])
    before = len(df)
    df = df[df["outcome"].isin(KEEP)]
    print(f"Dropped {before - len(df):,} rows outside {KEEP} -> {len(df):,} rows")

    # A row whose leak-free sections could not be found has empty features.
    empty = df[TEXT_COL].str.strip() == ""
    if empty.any():
        print(f"Dropped {int(empty.sum()):,} rows with empty {TEXT_COL}")
        df = df[~empty]

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampled down to {len(df):,} rows")

    df = df.reset_index(drop=True)

    # Grouped split: every document of a given case_number goes to the same side.
    # (GroupShuffleSplit cannot stratify, so class balance is checked below.)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["outcome"], groups=df[GROUP_COL]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train[GROUP_COL]) & set(test[GROUP_COL])
    if overlap:
        raise RuntimeError(f"{len(overlap)} case numbers appear in both train and test")
    print(f"Train {len(train):,} / test {len(test):,} "
          f"({train[GROUP_COL].nunique():,} / {test[GROUP_COL].nunique():,} case numbers, "
          f"0 shared)")
    print("Class share  train: "
          + ", ".join(f"{k} {v:.3f}" for k, v in train['outcome'].value_counts(normalize=True).items())
          + "  |  test: "
          + ", ".join(f"{k} {v:.3f}" for k, v in test['outcome'].value_counts(normalize=True).items()))

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
    print(f"Accuracy: {accuracy:.3f}  (expected about {EXPECTED_ACCURACY:.3f}; "
          f"leaky full_text baseline was about 0.954)")
    if args.sample:
        print("  (sampled run, so expect more noise than the full corpus)")

    print("\n=== Classification report ===")
    print(classification_report(y_test, preds, digits=3))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=clf.classes_),
        index=clf.classes_, columns=clf.classes_,
    ).to_string())

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