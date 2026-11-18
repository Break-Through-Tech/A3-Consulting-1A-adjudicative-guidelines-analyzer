"""Okapi BM25 keyword search index for DOHA precedent retrieval.

Supports exact citation matching (e.g. 'AG ¶ 19(a)'), statutory authorities
(e.g. '10 U.S.C. § 986', 'Directive 5220.6'), and docket number lookups
(e.g. 'ISCR Case No. 03-11614') across the 24,238 qualified precedent cases.
"""
from __future__ import annotations

import gzip
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

BM25_INDEX_PATH = config.OUT_DIR / "precedents_bm25.pkl.gz"


def tokenize(text: str) -> list[str]:
    """Tokenize legal text preserving docket numbers, statutory codes, and AG paragraph citations.

    Examples:
        'ISCR Case No. 03-11614' -> ['iscr', 'case', 'no', '03-11614', '03', '11614']
        'AG ¶ 19(a)' -> ['ag', '19(a)', 'ag19(a)', '19']
        '10 U.S.C. § 986' -> ['10', 'u', 's', 'c', '986']
    """
    if not text:
        return []

    raw = re.findall(r"(?:ag\s*¶?\s*\d+\([a-z0-9]+\)|\d{2}-\d{4,6}|[a-zA-Z0-9]+)", text.lower())
    tokens: list[str] = []

    for t in raw:
        tokens.append(t)

        # AG paragraph citations: emit normalized and sub-code variants
        if "ag" in t and "(" in t:
            clean = re.sub(r"[\s¶]+", "", t)
            tokens.append(clean)  # e.g. 'ag19(a)'
            sub = re.search(r"\d+\([a-z0-9]+\)", clean)
            if sub:
                tokens.append(sub.group(0))  # e.g. '19(a)'
                num_only = re.match(r"\d+", sub.group(0))
                if num_only:
                    tokens.append(num_only.group(0))  # e.g. '19'

        # Docket numbers: e.g. '03-11614' -> emit components '03' and '11614'
        elif "-" in t and re.match(r"^\d{2}-\d{4,6}$", t):
            parts = t.split("-")
            tokens.extend(parts)

    return tokens


class BM25Index:
    """Okapi BM25 keyword search index over DOHA precedent decisions."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs: int = 0
        self.avgdl: float = 0.0
        self.doc_lens: np.ndarray = np.array([], dtype=np.int32)
        self.case_numbers: list[str] = []
        self.source_urls: list[str] = []
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def build(self, precedents_df: pd.DataFrame) -> BM25Index:
        """Build BM25 index from qualified precedent DataFrame.

        The DataFrame must contain 'case_number', 'source_url', and 'text_features'.
        """
        t0 = time.time()
        self.n_docs = len(precedents_df)
        self.case_numbers = list(precedents_df["case_number"])
        self.source_urls = list(precedents_df["source_url"])

        # Tokenize each document: combines case number, guidelines, and text features
        corpus_tokens: list[list[str]] = []
        for _, row in precedents_df.iterrows():
            g_str = " ".join(row["guidelines"]) if hasattr(row.get("guidelines"), "__iter__") else ""
            doc_str = f"case {row['case_number']} iscr case no. {row['case_number']} {g_str} {row['text_features']}"
            corpus_tokens.append(tokenize(doc_str))

        self.doc_lens = np.array([len(t) for t in corpus_tokens], dtype=np.int32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.n_docs > 0 else 0.0

        # Compute document frequencies and IDF
        df_counts: Counter[str] = Counter()
        for doc in corpus_tokens:
            df_counts.update(set(doc))

        self.idf = {
            term: float(np.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in df_counts.items()
        }

        # Build inverted index mapping term -> (doc_indices, term_frequencies)
        temp_inv: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, doc in enumerate(corpus_tokens):
            tf = Counter(doc)
            for term, count in tf.items():
                if term not in temp_inv:
                    temp_inv[term] = ([], [])
                temp_inv[term][0].append(doc_id)
                temp_inv[term][1].append(count)

        self.inverted_index = {
            term: (np.array(docs, dtype=np.int32), np.array(counts, dtype=np.float32))
            for term, (docs, counts) in temp_inv.items()
        }

        elapsed = time.time() - t0
        print(
            f"Built BM25 index for {self.n_docs:,} documents ({len(self.inverted_index):,} terms) in {elapsed:.2f}s"
        )
        return self

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Search the BM25 index for a query string.

        Returns:
            List of (doc_index, bm25_score) tuples sorted by score descending.
        """
        if not query or self.n_docs == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl

        for term in q_tokens:
            if term in self.inverted_index:
                doc_ids, tfs = self.inverted_index[term]
                # Vectorized Okapi BM25 term score
                lens = self.doc_lens[doc_ids]
                tf_norm = (tfs * (k1 + 1.0)) / (tfs + k1 * (1.0 - b + b * (lens / avgdl)))
                scores[doc_ids] += self.idf[term] * tf_norm

        non_zero_indices = np.where(scores > 0)[0]
        if len(non_zero_indices) == 0:
            return []

        # Sort top-k matching candidates
        k_eval = min(top_k, len(non_zero_indices))
        top_sub_idx = np.argpartition(-scores[non_zero_indices], k_eval - 1)[:k_eval]
        top_doc_ids = non_zero_indices[top_sub_idx]
        top_doc_ids = top_doc_ids[np.argsort(-scores[top_doc_ids])]

        return [(int(doc_id), float(scores[doc_id])) for doc_id in top_doc_ids]

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        """Save the BM25 index compressed with gzip."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_lens": self.doc_lens,
            "case_numbers": self.case_numbers,
            "source_urls": self.source_urls,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved BM25 index to {path} ({mb:.2f} MB)")

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> BM25Index:
        """Load a saved BM25 index."""
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}. Build it first.")
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.n_docs = data["n_docs"]
        idx.avgdl = data["avgdl"]
        idx.doc_lens = data["doc_lens"]
        idx.case_numbers = data["case_numbers"]
        idx.source_urls = data["source_urls"]
        idx.idf = data["idf"]
        idx.inverted_index = data["inverted_index"]
        return idx


_CACHED_BM25: BM25Index | None = None


def get_bm25_index(index_path: Path = BM25_INDEX_PATH) -> BM25Index:
    """Return singleton BM25 index, building or loading from disk as needed."""
    global _CACHED_BM25
    if _CACHED_BM25 is not None:
        return _CACHED_BM25

    if index_path.exists():
        _CACHED_BM25 = BM25Index.load(index_path)
        return _CACHED_BM25

    print(f"BM25 index not found at {index_path}. Building from precedent corpus...")
    from src.dataset import load_cases

    cols = ["case_number", "source_url", "outcome", "guidelines", "text_features", "para_offsets_fof"]
    df = load_cases(pool=False, columns=cols)
    cond = (
        (df["text_features"].str.len() > 0)
        & (df["outcome"].isin(["DENIED", "GRANTED", "REVOKED"]))
        & (df["para_offsets_fof"].apply(lambda x: len(x) > 0 if hasattr(x, "__len__") else False))
    )
    precedents = df[cond].reset_index(drop=True)

    bm25 = BM25Index()
    bm25.build(precedents)
    bm25.save(index_path)
    _CACHED_BM25 = bm25
    return _CACHED_BM25
