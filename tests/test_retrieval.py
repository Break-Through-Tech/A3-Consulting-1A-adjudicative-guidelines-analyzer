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
        "para_offsets_fof",
    }
    assert expected_cols.issubset(set(meta.columns)), f"Missing columns: {expected_cols - set(meta.columns)}"

    # fof_text should be excluded to keep metadata under 2 MB
    assert "fof_text" not in meta.columns, "fof_text should be omitted from metadata to save space"

    # All para_offsets_fof must have at least one paragraph offset
    has_paras = meta["para_offsets_fof"].apply(lambda p: len(p) > 0 if hasattr(p, "__len__") else False)
    assert has_paras.all(), "Found cases without paragraph offsets in metadata"

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
        "best_paragraph_start",
        "best_paragraph_end",
        "best_paragraph_score",
    }
    assert expected_keys.issubset(set(top.keys())), f"Missing keys in hit: {expected_keys - set(top.keys())}"

    assert 0.0 <= top["score"] <= 1.0, f"Score out of range: {top['score']}"
    assert len(top["best_paragraph"]) > 20, "Expected non-empty highlighted paragraph"
    assert top["best_paragraph_end"] > top["best_paragraph_start"], "Expected valid paragraph char offsets"
    # Guideline F should be in top hit for financial debt query
    assert "F" in top["guidelines"], f"Expected Guideline F in top hit, got {top['guidelines']}"


def test_retriever_filters():
    retriever = PrecedentRetriever()
    query = "Foreign contacts and family members residing abroad, with delinquent debts."

    # Test guideline filter
    hits_b = retriever.search(query, top_k=3, guideline="B", highlight_paragraph=False)
    assert len(hits_b) == 3, f"Expected 3 hits for Guideline B filter, got {len(hits_b)}"
    for h in hits_b:
        assert "B" in h["guidelines"], f"Guideline filter violated: {h['guidelines']}"

    # Test outcome filter
    hits_granted = retriever.search(query, top_k=3, outcome="GRANTED", highlight_paragraph=False)
    assert len(hits_granted) == 3, f"Expected 3 hits for outcome GRANTED filter, got {len(hits_granted)}"
    for h in hits_granted:
        assert h["outcome"] == "GRANTED", f"Outcome filter violated: {h['outcome']}"

    # Test guideline H filter across whole index on debts and foreign family query
    # (H is not in top unfiltered semantic hits, so this asserts whole-index scan)
    hits_h = retriever.search(query, top_k=3, guideline="H", highlight_paragraph=False)
    assert len(hits_h) == 3, f"Expected 3 hits for Guideline H across whole index, got {len(hits_h)}"
    for h in hits_h:
        assert "H" in h["guidelines"], f"Guideline filter violated: {h['guidelines']}"


def test_bm25_index_and_docket_search():
    from src.bm25 import BM25_INDEX_PATH, get_bm25_index

    assert BM25_INDEX_PATH.exists(), f"Missing BM25 index at {BM25_INDEX_PATH}"
    bm25 = get_bm25_index()
    assert bm25.n_docs == 24238, f"Expected 24,238 documents in BM25, got {bm25.n_docs}"

    # Exact docket search should return the exact case at rank 1 with high score
    hits = bm25.search("ISCR Case No. 03-11614", top_k=5)
    assert len(hits) > 0, "Expected BM25 hits for docket query"
    top_doc_id, top_score = hits[0]
    assert bm25.case_numbers[top_doc_id] == "03-11614", f"Expected case 03-11614 at rank 1, got {bm25.case_numbers[top_doc_id]}"
    assert top_score > 30.0, f"Expected high BM25 score for exact docket, got {top_score}"


def test_hybrid_reciprocal_rank_fusion():
    retriever = PrecedentRetriever()

    # 1. Exact docket search under hybrid mode must return the target case at rank 1
    docket_hits = retriever.search("ISCR Case No. 03-11614", top_k=3, mode="hybrid", highlight_paragraph=False)
    assert len(docket_hits) == 3, f"Expected 3 hits, got {len(docket_hits)}"
    assert docket_hits[0]["case_number"] == "03-11614", f"Expected case 03-11614 at rank 1, got {docket_hits[0]['case_number']}"
    assert docket_hits[0]["rrf_score"] is not None, "Expected rrf_score in hybrid results"
    assert docket_hits[0]["bm25_score"] > 30.0, f"Expected strong BM25 component score, got {docket_hits[0]['bm25_score']}"

    # 2. Semantic query under hybrid mode combines dense and BM25 signals
    sem_hits = retriever.search("Applicant accumulated delinquent credit card debt and medical bills.", top_k=3, mode="hybrid", highlight_paragraph=True)
    assert len(sem_hits) == 3, f"Expected 3 hits, got {len(sem_hits)}"
    assert "F" in sem_hits[0]["guidelines"], f"Expected Guideline F in top hit, got {sem_hits[0]['guidelines']}"
    assert sem_hits[0]["dense_score"] > 0.7, f"Expected high dense similarity, got {sem_hits[0]['dense_score']}"


def test_50_query_test_set_structure():
    import json
    from src.query_set import QUERY_SET_PATH, STRATIFICATION_TARGETS

    assert QUERY_SET_PATH.exists(), f"Missing 50-query test set at {QUERY_SET_PATH}"
    with open(QUERY_SET_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert len(queries) == 50, f"Expected exactly 50 queries, got {len(queries)}"

    expected_keys = {
        "query_id",
        "case_number",
        "source_url",
        "split",
        "primary_guideline",
        "guidelines",
        "direction",
        "outcome",
        "query_text",
        "exact_docket_citation",
        "relevance_definition",
    }
    for q in queries:
        assert expected_keys.issubset(set(q.keys())), f"Missing keys in query {q.get('query_id')}"
        assert q["split"] == "test", f"All queries must be from held-out test split, got {q['split']}"
        assert len(q["query_text"]) >= 50, f"Query text too short: {q['query_text']}"

    # Check stratification counts
    g_counts = {}
    for q in queries:
        g = q["primary_guideline"]
        g_counts[g] = g_counts.get(g, 0) + 1

    for g, (n_ag, n_for) in STRATIFICATION_TARGETS.items():
        assert g_counts.get(g, 0) == (n_ag + n_for), f"Guideline {g} count mismatch: {g_counts.get(g, 0)} vs expected {n_ag + n_for}"


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "retrieval tests"))
