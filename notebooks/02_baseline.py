"""Step 6: TF-IDF + Logistic Regression baseline for outcome prediction.

Run from anywhere:
    python notebooks\02_baseline.py              # full corpus
    python notebooks\02_baseline.py --sample 2000  # quick smoke test
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "doha" / "doha_parsed_cases"

# The corpus has 5 outcome values, but REMANDED (110) and REVOKED (7) are too
# rare to learn and UNKNOWN means "the parser could not tell". Keep the two
# real classes.
KEEP = ["DENIED", "GRANTED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="use only N rows (faster; omit for full corpus)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(DATA_DIR / "all_cases_*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {DATA_DIR}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=["full_text", "outcome"])
    before = len(df)
    df = df[df["outcome"].isin(KEEP)]
    print(f"Dropped {before - len(df):,} rows outside {KEEP} -> {len(df):,} rows")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampled down to {len(df):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        df["full_text"], df["outcome"],
        test_size=0.2, random_state=42, stratify=df["outcome"],
    )
    print(f"Train {len(X_train):,} / test {len(X_test):,}")

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
    print(f"\nMajority-class accuracy to beat: {baseline:.3f}")

    print("\n=== Classification report ===")
    print(classification_report(y_test, preds, digits=3))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=clf.classes_),
        index=clf.classes_, columns=clf.classes_,
    ).to_string())

    # Which words drive the prediction. Worth reading: full_text contains the
    # judge's conclusion, so verdict language can leak in and inflate scores.
    names = vec.get_feature_names_out()
    coefs = pd.Series(clf.coef_[0], index=names).sort_values()
    print(f"\n=== Top 15 features -> {clf.classes_[0]} ===")
    print(coefs.head(15).to_string())
    print(f"\n=== Top 15 features -> {clf.classes_[1]} ===")
    print(coefs.tail(15)[::-1].to_string())


if __name__ == "__main__":
    main()
