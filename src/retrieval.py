"""Dense semantic precedent retrieval index using sentence-transformers and FAISS.

Task 13 (October milestone): Precedent search index v1.
    python -m src.retrieval --build        # Build and save FAISS index & metadata
    python -m src.retrieval --smoke-test   # Run validation queries
    python -m src.retrieval --query "..."  # Search precedent cases
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from sentence_transformers import SentenceTransformer

from src import config
from src.dataset import load_cases, load_splits, shard_files
from src.normalize import normalize_body

# Precedent corpus artifact paths
INDEX_PATH = config.OUT_DIR / "precedents.faiss"
METADATA_PATH = config.OUT_DIR / "precedents_metadata.parquet"
CHECKPOINT_PATH = config.OUT_DIR / ".precedents_embeddings_checkpoint.npy"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


def get_default_device() -> str:
    """Detect fastest available acceleration hardware (CUDA, MPS, or CPU fallback)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_precedent_corpus() -> tuple[pd.DataFrame, list[str], list[tuple[int, int]]]:
    """Load qualified precedent cases and extract all individual Findings of Fact paragraphs.

    Per PROJECT_PLAN.md §4:
    Usable corpus requires:
    - non-empty text_features
    - resolved outcome (DENIED, GRANTED, REVOKED)
    - non-empty para_offsets_fof (findings-of-fact paragraphs)
    Exactly 24,238 cases qualify.

    Returns:
        corpus: Metadata DataFrame (without heavy full_text/fof_text to keep file size ~1 MB)
        all_paras: Flattened list of all normalized paragraph strings across the corpus
        case_para_ranges: List of (start_idx, end_idx) mapping each case to its paragraphs in all_paras
    """
    cols = [
        "case_number",
        "source_url",
        "outcome",
        "guidelines",
        "label_source",
        "guideline_regime",
        "case_type",
        "date",
        "case_year",
        "sec_findings_of_fact_start",
        "sec_findings_of_fact_end",
        "para_offsets_fof",
        "text_features",
        "full_text",
    ]
    df = load_cases(pool=False, columns=cols)
    cond = (
        (df["text_features"].str.len() > 0)
        & (df["outcome"].isin(["DENIED", "GRANTED", "REVOKED"]))
        & (df["para_offsets_fof"].apply(lambda x: len(x) > 0 if hasattr(x, "__len__") else False))
    )
    corpus = df[cond].copy().reset_index(drop=True)

    # Attach split assignment if splits.parquet exists
    try:
        splits = load_splits()
        corpus = corpus.merge(splits[["source_url", "split"]], on="source_url", how="left")
        corpus["split"] = corpus["split"].fillna("unassigned")
    except Exception:
        corpus["split"] = "unassigned"

    all_paras: list[str] = []
    case_para_ranges: list[tuple[int, int]] = []

    for _, row in corpus.iterrows():
        ft = row["full_text"] or ""
        offsets = row["para_offsets_fof"]
        c_start = len(all_paras)

        for item in offsets:
            ps, pe = int(item[0]), int(item[1])
            if pe > ps:
                pt = normalize_body(ft[ps:pe])
                if len(pt) >= 20:
                    all_paras.append(pt)

        # Fallback to whole Findings of Fact if all paragraph slices were < 20 chars
        if len(all_paras) == c_start:
            fs, fe = int(row["sec_findings_of_fact_start"]), int(row["sec_findings_of_fact_end"])
            all_paras.append(normalize_body(ft[fs:fe]))

        case_para_ranges.append((c_start, len(all_paras)))

    meta_cols = [
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
    ]
    return corpus[meta_cols], all_paras, case_para_ranges


def build_precedent_index(
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    case_chunk_size: int = 1000,
    device: str | None = None,
    output_index_path: Path = INDEX_PATH,
    output_metadata_path: Path = METADATA_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> tuple[faiss.Index, pd.DataFrame]:
    """Encode all paragraphs in memory-safe chunks, average per case, and save FAISS index & metadata.

    Each case's embedding is the L2-normalized mean of its L2-normalized paragraph
    embeddings. Processing in chunks of cases prevents GPU/MPS memory accumulation
    and supports resume on interruption.

    Produces:
        Challenge_Data/precedents.faiss (IndexFlatIP, 24,238 vectors, 384-dim)
        Challenge_Data/precedents_metadata.parquet (~1 MB, offsets only)
    """
    if device is None:
        device = get_default_device()

    print(f"Loading precedent corpus and extracting paragraphs (target: 24,238 cases)...")
    t0 = time.time()
    corpus, all_paras, case_para_ranges = get_precedent_corpus()
    n_cases = len(corpus)
    print(
        f"Loaded {n_cases:,} cases with {len(all_paras):,} paragraphs in {time.time() - t0:.2f}s"
    )

    print(f"Loading embedding model: {model_name} on device '{device}'...")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256

    # Check for existing checkpoint to resume
    case_embeddings = np.zeros((n_cases, VECTOR_DIM), dtype=np.float32)
    start_chunk = 0
    if checkpoint_path.exists():
        try:
            saved = np.load(checkpoint_path)
            if saved.shape == (n_cases, VECTOR_DIM):
                non_zero_cases = np.where(np.linalg.norm(saved, axis=1) > 0)[0]
                if len(non_zero_cases) > 0:
                    start_case = int(non_zero_cases[-1]) + 1
                    start_chunk = start_case // case_chunk_size
                    case_embeddings = saved
                    print(
                        f"Found checkpoint at {checkpoint_path}: resuming from chunk {start_chunk} (case {start_chunk * case_chunk_size:,})"
                    )
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")

    n_chunks = (n_cases + case_chunk_size - 1) // case_chunk_size
    t_start_encoding = time.time()

    for chunk_idx in range(start_chunk, n_chunks):
        c_start_case = chunk_idx * case_chunk_size
        c_end_case = min((chunk_idx + 1) * case_chunk_size, n_cases)

        # Slice paragraphs for this chunk of cases
        chunk_case_ranges = case_para_ranges[c_start_case:c_end_case]
        chunk_start_para = chunk_case_ranges[0][0]
        chunk_end_para = chunk_case_ranges[-1][1]
        chunk_paras = all_paras[chunk_start_para:chunk_end_para]

        t_c = time.time()
        with torch.no_grad():
            chunk_para_embs = model.encode(
                chunk_paras,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        # Compute averaged and normalized vector for each case in this chunk
        for i, (p_start, p_end) in enumerate(chunk_case_ranges):
            case_idx = c_start_case + i
            cs = p_start - chunk_start_para
            ce = p_end - chunk_start_para
            mean_vec = np.mean(chunk_para_embs[cs:ce], axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            case_embeddings[case_idx] = mean_vec

        # Free GPU / MPS memory buffers
        del chunk_para_embs
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_c
        total_elapsed = time.time() - t_start_encoding
        done_cases = c_end_case
        est_total = (total_elapsed / max(1, chunk_idx - start_chunk + 1)) * (n_chunks - start_chunk)
        eta = max(0, est_total - total_elapsed)

        print(
            f"Chunk {chunk_idx + 1}/{n_chunks} ({done_cases:,}/{n_cases:,} cases, "
            f"{len(chunk_paras):,} paras) encoded in {elapsed:.1f}s | ETA: {eta/60:.1f} min"
        )

        # Save checkpoint after each chunk
        np.save(checkpoint_path, case_embeddings)

    total_encode_time = time.time() - t_start_encoding
    print(
        f"All case embeddings complete in {total_encode_time:.2f}s "
        f"({n_cases / total_encode_time:.1f} cases/sec)"
    )

    print("Constructing FAISS IndexFlatIP (cosine similarity on normalized vectors)...")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(np.ascontiguousarray(case_embeddings, dtype=np.float32))
    print(f"Index built: {index.ntotal:,} vectors of dimension {index.d}")

    # Save FAISS index
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_path))
    index_mb = output_index_path.stat().st_size / (1024 * 1024)
    print(f"Saved FAISS index to {output_index_path} ({index_mb:.2f} MB)")

    # Save metadata parquet table (offsets only, no full text to save space)
    corpus.to_parquet(output_metadata_path, engine=config.PARQUET_ENGINE, compression="gzip")
    meta_mb = output_metadata_path.stat().st_size / (1024 * 1024)
    print(f"Saved metadata to {output_metadata_path} ({meta_mb:.2f} MB)")

    # Remove temporary checkpoint file upon successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return index, corpus


class PrecedentRetriever:
    """Precedent case retriever supporting dense semantic search and paragraph highlighting."""

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.device = device if device is not None else get_default_device()

        self._index: faiss.Index | None = None
        self._metadata: pd.DataFrame | None = None
        self._model: SentenceTransformer | None = None
        self._dataset: ds.Dataset | None = None

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"Precedent index not found at {self.index_path}. "
                    "Run `python -m src.retrieval --build` first."
                )
            self._index = faiss.read_index(str(self.index_path))
        return self._index

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Precedent metadata not found at {self.metadata_path}. "
                    "Run `python -m src.retrieval --build` first."
                )
            self._metadata = pd.read_parquet(self.metadata_path)
        return self._metadata

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _fetch_full_texts(self, source_urls: list[str]) -> dict[str, str]:
        """Fetch verbatim full_text on demand for a list of source_urls from parquet shards."""
        if not source_urls:
            return {}
        if self._dataset is None:
            self._dataset = ds.dataset(shard_files(), format="parquet")
        table = self._dataset.to_table(
            filter=ds.field("source_url").isin(source_urls),
            columns=["source_url", "full_text"],
        )
        df_lookup = table.to_pandas()
        return dict(zip(df_lookup["source_url"], df_lookup["full_text"]))

    def search(
        self,
        query: str,
        top_k: int = 5,
        guideline: str | None = None,
        outcome: str | None = None,
        split: str | None = None,
        highlight_paragraph: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve most similar precedent cases for a fact pattern or query string.

        Args:
            query: Free-text fact pattern or case description.
            top_k: Number of precedent cases to return.
            guideline: Optional filter for specific guideline letter (e.g. 'F', 'B').
            outcome: Optional filter for outcome ('DENIED', 'GRANTED', 'REVOKED').
            split: Optional filter for split ('train', 'val', 'test').
            highlight_paragraph: Whether to find and score the best-matching paragraph within each case.

        Returns:
            List of result dictionaries sorted by similarity score descending.
        """
        if not query or not query.strip():
            return []

        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        q_vec = np.ascontiguousarray([q_emb], dtype=np.float32)

        # Retrieve a broader candidate pool if filters are applied
        has_filter = bool(guideline or outcome or split)
        candidate_k = min(top_k * 15 if has_filter else top_k, self.index.ntotal)

        scores, indices = self.index.search(q_vec, candidate_k)
        scores = scores[0]
        indices = indices[0]

        candidates = []
        for sim, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = self.metadata.iloc[idx]

            # Apply optional filters
            if guideline:
                glist = row["guidelines"]
                if glist is None or guideline.upper() not in glist:
                    continue

            if outcome:
                if str(row["outcome"]).upper() != outcome.upper():
                    continue

            if split:
                if str(row["split"]).lower() != split.lower():
                    continue

            item = {
                "rank": len(candidates) + 1,
                "case_number": row["case_number"],
                "score": round(float(sim), 4),
                "outcome": row["outcome"],
                "guidelines": list(row["guidelines"]) if row["guidelines"] is not None else [],
                "label_source": row["label_source"],
                "guideline_regime": row["guideline_regime"],
                "case_type": row["case_type"],
                "case_year": row["case_year"],
                "date": str(row["date"]),
                "split": row["split"],
                "source_url": row["source_url"],
                "para_offsets": row["para_offsets_fof"],
                "best_paragraph": "",
                "best_paragraph_start": 0,
                "best_paragraph_end": 0,
                "best_paragraph_score": round(float(sim), 4),
            }
            candidates.append(item)
            if len(candidates) >= top_k:
                break

        # Highlight best matching paragraph within each retrieved case using para_offsets_fof
        if highlight_paragraph and candidates:
            urls = [c["source_url"] for c in candidates]
            full_texts = self._fetch_full_texts(urls)

            for c in candidates:
                ft = full_texts.get(c["source_url"], "")
                offsets = c.pop("para_offsets", [])
                paras: list[str] = []
                spans: list[tuple[int, int]] = []

                if offsets is not None:
                    for item in offsets:
                        ps, pe = int(item[0]), int(item[1])
                        if pe > ps:
                            pt = normalize_body(ft[ps:pe])
                            if len(pt) >= 20:
                                paras.append(pt)
                                spans.append((ps, pe))

                if paras:
                    p_embs = self.model.encode(
                        paras, normalize_embeddings=True, show_progress_bar=False
                    )
                    p_sims = np.dot(p_embs, q_emb)
                    best_i = int(np.argmax(p_sims))
                    c["best_paragraph"] = paras[best_i]
                    c["best_paragraph_start"] = spans[best_i][0]
                    c["best_paragraph_end"] = spans[best_i][1]
                    c["best_paragraph_score"] = round(float(p_sims[best_i]), 4)
                else:
                    c["best_paragraph"] = normalize_body(ft[:400])
                    c["best_paragraph_start"] = 0
                    c["best_paragraph_end"] = min(400, len(ft))
        else:
            for c in candidates:
                c.pop("para_offsets", None)

        return candidates

    def format_results(self, results: list[dict[str, Any]], query: str | None = None) -> str:
        """Format retrieval results as a readable markdown summary."""
        lines = []
        if query:
            lines.append(f"### Precedent Search Results for Fact Pattern\n> *\"{query}\"*\n")
        lines.append(f"Found **{len(results)}** precedent cases:\n")

        for r in results:
            g_str = ", ".join(r["guidelines"]) if r["guidelines"] else "None"
            outcome_badge = f"**{r['outcome']}**"
            trust_warning = (
                " ⚠️ *(candidate-only labels)*" if r["label_source"] == "regex_fallback" else ""
            )

            lines.append(f"#### #{r['rank']} Case {r['case_number']} — Similarity: `{r['score']:.4f}`")
            lines.append(
                f"- **Outcome:** {outcome_badge} | **Guidelines:** [{g_str}] | **Regime:** `{r['guideline_regime']}`"
            )
            lines.append(
                f"- **Proceeding:** {r['case_type'].title()} ({r['case_year']}) | **Date:** {r['date'][:10]} | **Split:** `{r['split']}`"
            )
            lines.append(f"- **Label Trust Tier:** `{r['label_source']}`{trust_warning}")
            lines.append(f"- **Source Document:** [{r['case_number']}.pdf]({r['source_url']})")

            if r.get("best_paragraph"):
                snippet = r["best_paragraph"].replace("\n", " ")
                span_info = f"offsets `[{r['best_paragraph_start']}:{r['best_paragraph_end']}]`"
                lines.append(
                    f"- **Most Relevant Findings Excerpt** (sim `{r['best_paragraph_score']:.4f}`, {span_info}):"
                )
                lines.append(f"  > \"{snippet}\"\n")
            else:
                lines.append("")

        return "\n".join(lines)


def run_smoke_test(retriever: PrecedentRetriever) -> bool:
    """Run smoke test queries across diverse guidelines and assert relevancy."""
    print("=" * 70)
    print("Running Precedent Index Smoke Tests (Paragraph-Averaged Dense Vectors)")
    print("=" * 70)

    test_queries = [
        {
            "topic": "Guideline F (Financial Considerations)",
            "query": "Applicant accumulated delinquent credit card debt, hospital medical bills, and mortgage foreclosure following divorce.",
            "expected_guideline": "F",
        },
        {
            "topic": "Guideline B (Foreign Influence)",
            "query": "Applicant has immediate family members who are citizens and residents of China, and owns real property abroad.",
            "expected_guideline": "B",
        },
        {
            "topic": "Guideline H (Drug Involvement)",
            "query": "Applicant admitted to recreational marijuana and cannabis use during college and after submitting security clearance questionnaire.",
            "expected_guideline": "H",
        },
        {
            "topic": "Guideline E (Personal Conduct)",
            "query": "Applicant deliberately failed to disclose multiple past misdemeanor arrests on Standard Form 86.",
            "expected_guideline": "E",
        },
    ]

    all_passed = True
    for item in test_queries:
        print(f"\n[Test] Query Topic: {item['topic']}")
        print(f"Query: \"{item['query']}\"")
        hits = retriever.search(item["query"], top_k=3)

        if not hits:
            print("  [FAIL] No results returned!")
            all_passed = False
            continue

        top_hit = hits[0]
        has_expected_guideline = item["expected_guideline"] in top_hit["guidelines"]
        score_in_range = 0.0 <= top_hit["score"] <= 1.0
        has_paragraph = len(top_hit["best_paragraph"]) > 20
        has_offsets = top_hit["best_paragraph_end"] > top_hit["best_paragraph_start"]

        print(
            f"  Top Match: Case {top_hit['case_number']} "
            f"(Score: {top_hit['score']:.4f}, Outcome: {top_hit['outcome']})"
        )
        print(f"  Guidelines in Top Match: {top_hit['guidelines']}")
        print(
            f"  Span in full_text: [{top_hit['best_paragraph_start']}:{top_hit['best_paragraph_end']}]"
        )
        print(f"  Excerpt: \"{top_hit['best_paragraph'][:140]}...\"")

        if score_in_range and has_paragraph and has_expected_guideline and has_offsets:
            print(
                f"  [PASS] Expected guideline '{item['expected_guideline']}' matched with sensible similarity score and full_text span."
            )
        else:
            print(
                f"  [FAIL] Score={top_hit['score']}, GuidelineMatch={has_expected_guideline}, "
                f"Paragraph={has_paragraph}, Offsets={has_offsets}"
            )
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL SMOKE TESTS PASSED!")
    else:
        print("SOME SMOKE TESTS FAILED!")
    print("=" * 70)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="DOHA Precedent Search Index (Task 13)")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index and metadata")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run validation smoke tests on index"
    )
    parser.add_argument("--query", type=str, help="Search for precedent cases matching a query")
    parser.add_argument("--guideline", type=str, help="Optional guideline filter (e.g. F, B)")
    parser.add_argument("--outcome", type=str, help="Optional outcome filter (DENIED, GRANTED)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of precedent cases to return")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation (default: 128)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of cases per chunk to prevent memory accumulation (default: 1000)",
    )
    args = parser.parse_args()

    if args.build:
        build_precedent_index(batch_size=args.batch_size, case_chunk_size=args.chunk_size)
        return

    retriever = PrecedentRetriever()

    if args.smoke_test:
        success = run_smoke_test(retriever)
        sys.exit(0 if success else 1)

    if args.query:
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            guideline=args.guideline,
            outcome=args.outcome,
        )
        print(retriever.format_results(results, query=args.query))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
