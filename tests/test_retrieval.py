"""Unit tests for the precedent search index and retrieval module (Task 13).

    python tests/test_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
import pandas as pd
from _harness import run_module
from src import config
from src.retrieval import (
    INDEX_PATH,
    METADATA_PATH,
    VECTOR_DIM,
    PrecedentRetriever,
)


def test_precedent_artifacts_exist():
    assert INDEX_PATH.exists(), f"Missing FAISS index at {INDEX_PATH}"
    assert METADATA_PATH.exists(), f"Missing metadata at {METADATA_PATH}"


def test_faiss_index_properties():
    index = faiss.read_index(str(INDEX_PATH))
    assert index.ntotal == 24238, f"Expected 24,238 vectors, got {index.ntotal}"
    assert index.d == VECTOR_DIM, f"Expected dimension {VECTOR_DIM}, got {index.d}"


def test_precedent_metadata_schema_and_alignment():
    meta = pd.read_parquet(METADATA_PATH)
    assert len(meta) == 24238, f"Expected 24,238 rows in metadata, got {len(meta)}"

    expected_cols = {
        "case_number",
        "source_url",
        "outcome",
        "guidelines",
        "label_source",
        "guideline_regime",
        "case_type",
        "date",
        "case_year",
        "split",
        "fof_text",
    }
    assert expected_cols.issubset(set(meta.columns)), f"Missing columns: {expected_cols - set(meta.columns)}"

    # All fof_text must be non-empty
    assert (meta["fof_text"].str.len() > 0).all(), "Found empty Findings of Fact texts in metadata"

    # All outcomes must be resolved (DENIED, GRANTED, REVOKED)
    assert set(meta["outcome"].unique()).issubset({"DENIED", "GRANTED", "REVOKED"})


def test_retriever_search_and_highlighting():
    retriever = PrecedentRetriever()
    query = "Applicant accumulated significant delinquent credit card debt following medical expenses."

    hits = retriever.search(query, top_k=3, highlight_paragraph=True)
    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"

    top = hits[0]
    expected_keys = {
        "rank",
        "case_number",
        "score",
        "outcome",
        "guidelines",
        "label_source",
        "guideline_regime",
        "case_type",
        "case_year",
        "date",
        "split",
        "source_url",
        "best_paragraph",
        "best_paragraph_score",
    }
    assert expected_keys.issubset(set(top.keys())), f"Missing keys in hit: {expected_keys - set(top.keys())}"

    assert 0.0 <= top["score"] <= 1.0, f"Score out of range: {top['score']}"
    assert len(top["best_paragraph"]) > 20, "Expected non-empty highlighted paragraph"
    # Guideline F should be in top hit for financial debt query
    assert "F" in top["guidelines"], f"Expected Guideline F in top hit, got {top['guidelines']}"


def test_retriever_filters():
    retriever = PrecedentRetriever()
    query = "Foreign contacts and family members residing abroad."

    # Test guideline filter
    hits_b = retriever.search(query, top_k=3, guideline="B", highlight_paragraph=False)
    for h in hits_b:
        assert "B" in h["guidelines"], f"Guideline filter violated: {h['guidelines']}"

    # Test outcome filter
    hits_granted = retriever.search(query, top_k=3, outcome="GRANTED", highlight_paragraph=False)
    for h in hits_granted:
        assert h["outcome"] == "GRANTED", f"Outcome filter violated: {h['outcome']}"


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "retrieval tests"))
