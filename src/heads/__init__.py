"""Transformer guideline heads: Legal-BERT and ModernBERT.

One encoder reads a case's leak-free ``text_features`` and 13 heads, one per
SEAD-4 guideline A-M, each predict NOT_AT_ISSUE, FOR or AGAINST the applicant.
The labels, the at-issue threshold and every metric come from the TF-IDF
baseline in ``notebooks/04_guideline_classifier_baseline.py``, so the two are
compared like for like.

Only ``text_features`` is ever used as input. ``full_text`` holds the judge's
formal findings and decision, so it would leak the answer.

    python -m src.heads.train --model nlpaueb/legal-bert-base-uncased --max-length 512 --run-name legalbert-512
    python -m src.heads.evaluate --run runs/legalbert-512
"""
