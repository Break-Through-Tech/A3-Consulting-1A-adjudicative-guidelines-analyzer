"""50-Query Stratified Test Set and Precedent Retrieval Benchmark.

Builds and evaluates the 50-query test set for precedent retrieval (Task 13 milestone).
Queries are drawn exclusively from the held-out test split (Challenge_Data/splits.parquet),
stratified across all 12 test guidelines (B through M), both directions (FOR and AGAINST),
and decision regimes (SEAD-4 and Directive 5220.6).

Relevance Definition (established strictly beforehand):
    A candidate precedent counts as relevant to a query iff:
    1. It is not the query case itself (source_url / case_number differs).
    2. It shares at least one guideline with the query (set intersection non-empty).
    3. The direction on the shared guideline(s) matches:
       - If per-guideline formal rulings are present, the candidate agrees with the query's
         direction (FOR vs. AGAINST) on at least one shared guideline.
       - Otherwise, the candidate's overall finding direction matches the query's finding direction
         (DENIED/REVOKED maps to AGAINST, GRANTED maps to FOR).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.dataset import load_cases
from src.normalize import normalize_body

QUERY_SET_PATH = config.OUT_DIR / "retrieval_queries_50.json"

# Stratification targets: (n_against, n_for) per primary guideline
STRATIFICATION_TARGETS: dict[str, tuple[int, int]] = {
    "F": (5, 5),  # Financial Considerations: 10 queries
    "E": (4, 4),  # Personal Conduct: 8 queries
    "B": (3, 3),  # Foreign Influence: 6 queries
    "H": (3, 3),  # Drug Involvement: 6 queries
    "G": (2, 2),  # Alcohol Consumption: 4 queries
    "J": (2, 2),  # Criminal Conduct: 4 queries
    "C": (2, 1),  # Foreign Preference: 3 queries
    "D": (1, 1),  # Sexual Behavior: 2 queries
    "I": (1, 1),  # Psychological Conditions: 2 queries
    "K": (1, 1),  # Handling Protected Information: 2 queries
    "M": (1, 1),  # Use of Information Technology: 2 queries
    "L": (1, 0),  # Outside Activities: 1 query (total: 50)
}

RELEVANCE_DEFINITION = (
    "A precedent counts as relevant when it shares the guidelines and direction of the query. "
    "Specifically: (1) candidate != query, (2) len(set(candidate.guidelines) & set(query.guidelines)) > 0, "
    "and (3) the formal finding direction (FOR/AGAINST) on the shared guideline(s) matches the query direction."
)


def is_precedent_relevant(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Evaluate whether a candidate precedent is relevant to a query under the predefined criteria."""
    # Rule 1: A case cannot cite itself as a past precedent
    if candidate.get("source_url") == query.get("source_url") or candidate.get("case_number") == query.get("case_number"):
        return False

    q_guidelines = set(query.get("guidelines") or [])
    c_guidelines = set(candidate.get("guidelines") or [])

    # Rule 2: Must share at least one guideline
    shared = q_guidelines & c_guidelines
    if not shared:
        return False

    # Rule 3: Direction match
    # Check per-guideline formal findings if available
    q_findings = query.get("guideline_findings") or {}
    c_findings = candidate.get("guideline_findings") or {}

    matching_findings = [
        g for g in shared
        if q_findings.get(g) and c_findings.get(g) and q_findings.get(g) == c_findings.get(g)
    ]
    if matching_findings:
        return True

    # Fallback to overall direction alignment
    q_dir = query.get("direction")
    if not q_dir:
        q_dir = "AGAINST" if query.get("outcome") in ["DENIED", "REVOKED"] else "FOR"

    c_dir = candidate.get("direction")
    if not c_dir:
        c_dir = "AGAINST" if candidate.get("outcome") in ["DENIED", "REVOKED"] else "FOR"

    return q_dir == c_dir


def build_50_query_test_set(output_path: Path = QUERY_SET_PATH, seed: int = 42) -> list[dict[str, Any]]:
    """Build reproducible 50-query stratified test set from the held-out test split."""
    cols = [
        "case_number",
        "source_url",
        "split",
        "guidelines",
        "outcome",
        "guideline_findings",
        "guideline_regime",
        "text_features",
        "full_text",
        "para_offsets_fof",
    ]
    df = load_cases(pool=True, columns=cols)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    rng = np.random.RandomState(seed)
    selected_rows: list[dict[str, Any]] = []
    used_case_numbers: set[str] = set()

    query_counter = 1

    for guideline, (target_against, target_for) in STRATIFICATION_TARGETS.items():
        # Match cases containing this guideline
        g_mask = test_df["guidelines"].apply(lambda gl: guideline in gl if hasattr(gl, "__iter__") else False)
        g_candidates = test_df[g_mask]

        # Select AGAINST cases
        against_pool = g_candidates[
            (g_candidates["outcome"] == "DENIED") & (~g_candidates["case_number"].isin(used_case_numbers))
        ]
        if len(against_pool) < target_against:
            raise ValueError(f"Insufficient AGAINST cases for Guideline {guideline}")
        shuffled_against = against_pool.sample(n=target_against, random_state=seed)

        for _, row in shuffled_against.iterrows():
            used_case_numbers.add(row["case_number"])
            q_text = _extract_query_text(row)
            findings_clean = {k: v for k, v in row["guideline_findings"].items() if v is not None} if isinstance(row.get("guideline_findings"), dict) else {}

            item = {
                "query_id": f"Q{query_counter:02d}",
                "case_number": row["case_number"],
                "source_url": row["source_url"],
                "split": "test",
                "primary_guideline": guideline,
                "guidelines": list(row["guidelines"]),
                "direction": "AGAINST",
                "guideline_findings": findings_clean,
                "outcome": row["outcome"],
                "guideline_regime": row["guideline_regime"],
                "query_text": q_text,
                "exact_docket_citation": f"ISCR Case No. {row['case_number']}",
                "relevance_definition": RELEVANCE_DEFINITION,
            }
            selected_rows.append(item)
            query_counter += 1

        # Select FOR cases
        if target_for > 0:
            for_pool = g_candidates[
                (g_candidates["outcome"] == "GRANTED") & (~g_candidates["case_number"].isin(used_case_numbers))
            ]
            if len(for_pool) < target_for:
                raise ValueError(f"Insufficient FOR cases for Guideline {guideline}")
            shuffled_for = for_pool.sample(n=target_for, random_state=seed)

            for _, row in shuffled_for.iterrows():
                used_case_numbers.add(row["case_number"])
                q_text = _extract_query_text(row)
                findings_clean = {k: v for k, v in row["guideline_findings"].items() if v is not None} if isinstance(row.get("guideline_findings"), dict) else {}

                item = {
                    "query_id": f"Q{query_counter:02d}",
                    "case_number": row["case_number"],
                    "source_url": row["source_url"],
                    "split": "test",
                    "primary_guideline": guideline,
                    "guidelines": list(row["guidelines"]),
                    "direction": "FOR",
                    "guideline_findings": findings_clean,
                    "outcome": row["outcome"],
                    "guideline_regime": row["guideline_regime"],
                    "query_text": q_text,
                    "exact_docket_citation": f"ISCR Case No. {row['case_number']}",
                    "relevance_definition": RELEVANCE_DEFINITION,
                }
                selected_rows.append(item)
                query_counter += 1

    assert len(selected_rows) == 50, f"Expected 50 queries, got {len(selected_rows)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_rows, f, indent=2)

    print(f"Generated {len(selected_rows)} stratified retrieval queries -> {output_path}")
    return selected_rows


def _extract_query_text(row: pd.Series) -> str:
    """Extract clean factual narrative representing the query fact pattern without verdict text."""
    ft = row.get("full_text") or ""
    offsets = row.get("para_offsets_fof")

    # If paragraph offsets exist, combine the first 1-2 factual paragraphs
    if offsets is not None and len(offsets) > 0:
        paras = []
        for item in offsets[:3]:
            ps, pe = int(item[0]), int(item[1])
            if pe > ps:
                clean_p = normalize_body(ft[ps:pe])
                if len(clean_p) >= 30:
                    paras.append(clean_p)
                if len(" ".join(paras)) >= 300:
                    break
        if paras:
            return " ".join(paras)

    # Fallback to initial section of text_features
    tf = row.get("text_features") or ""
    return normalize_body(tf[:600])


def load_query_set(path: Path = QUERY_SET_PATH) -> list[dict[str, Any]]:
    """Load the 50-query test set from disk."""
    if not path.exists():
        return build_50_query_test_set(output_path=path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval_benchmark(
    retriever: Any,
    queries: list[dict[str, Any]] | None = None,
    modes: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    """Evaluate retrieval benchmark across modes on the 50-query test set.

    Computes:
        - Precision@1, Precision@3, Precision@5
        - Mean Reciprocal Rank (MRR)
        - Exact Docket / Citation Match Rate@1 and Match Rate@5
    """
    if queries is None:
        queries = load_query_set()
    if modes is None:
        modes = ["hybrid", "dense", "bm25"]

    results: dict[str, dict[str, float]] = {}

    for mode in modes:
        p1_list: list[float] = []
        p3_list: list[float] = []
        p5_list: list[float] = []
        rr_list: list[float] = []
        exact_docket_hits_1: int = 0
        exact_docket_hits_5: int = 0

        for q in queries:
            # 1. Factual pattern search (fetch top_k + 2 to account for potential self-match)
            hits = retriever.search(q["query_text"], top_k=top_k + 2, mode=mode, highlight_paragraph=False)

            # Filter out self-match if present
            clean_hits = [h for h in hits if h["source_url"] != q["source_url"] and h["case_number"] != q["case_number"]][:top_k]

            relevant_flags = [is_precedent_relevant(q, h) for h in clean_hits]

            # Precision@K
            p1 = 1.0 if (len(relevant_flags) > 0 and relevant_flags[0]) else 0.0
            p3 = sum(relevant_flags[:3]) / min(3, max(1, len(clean_hits)))
            p5 = sum(relevant_flags[:5]) / min(5, max(1, len(clean_hits)))

            p1_list.append(p1)
            p3_list.append(p3)
            p5_list.append(p5)

            # Reciprocal Rank of first relevant precedent
            rr = 0.0
            for rank, is_rel in enumerate(relevant_flags, 1):
                if is_rel:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)

            # 2. Exact docket query search (tests exact citation retrieval)
            docket_query = q["exact_docket_citation"]
            d_hits = retriever.search(docket_query, top_k=5, mode=mode, highlight_paragraph=False)
            d_case_numbers = [h["case_number"] for h in d_hits]

            if d_case_numbers and d_case_numbers[0] == q["case_number"]:
                exact_docket_hits_1 += 1
            if q["case_number"] in d_case_numbers:
                exact_docket_hits_5 += 1

        n_q = max(1, len(queries))
        results[mode] = {
            "queries": float(n_q),
            "precision@1": round(float(np.mean(p1_list)), 4),
            "precision@3": round(float(np.mean(p3_list)), 4),
            "precision@5": round(float(np.mean(p5_list)), 4),
            "mrr": round(float(np.mean(rr_list)), 4),
            "exact_docket_hit@1": round(exact_docket_hits_1 / n_q, 4),
            "exact_docket_hit@5": round(exact_docket_hits_5 / n_q, 4),
        }

    return results


def export_team_annotation_tasks(
    retriever: Any,
    queries: list[dict[str, Any]] | None = None,
    output_json: Path = config.OUT_DIR / "retrieval_annotation_tasks.json",
    output_md: Path = config.OUT_DIR / "retrieval_annotation_tasks.md",
) -> dict[str, Any]:
    """Export top-5 hybrid search hits for all 50 test queries partitioned across 4 teammates.

    Assignments (Task 6 in Milestone #3):
        - Aryan: Q01 - Q13 (13 queries)
        - Brian: Q14 - Q25 (12 queries)
        - Rashed: Q26 - Q37 (12 queries)
        - Sam: Q38 - Q50 (13 queries)
    """
    if queries is None:
        queries = load_query_set()

    assignments = {
        "Aryan": queries[0:13],
        "Brian": queries[13:25],
        "Rashed": queries[25:37],
        "Sam": queries[37:50],
    }

    team_export: dict[str, list[dict[str, Any]]] = {}
    md_lines: list[str] = [
        "# Milestone #3 Task 6: Precedent Retrieval Relevance Labeling\n",
        "**Relevance Standard:** A precedent is relevant (`1`) if it is not the query case itself, ",
        "shares at least one guideline with the query, and agrees with the query's formal direction (FOR / AGAINST).\n",
        "Otherwise mark `0`.\n",
    ]

    for member, member_queries in assignments.items():
        team_export[member] = []
        md_lines.append(f"\n## Assignee: {member} ({len(member_queries)} queries)\n")

        for q in member_queries:
            hits = retriever.search(q["query_text"], top_k=7, mode="hybrid", highlight_paragraph=True)
            clean_hits = [h for h in hits if h["source_url"] != q["source_url"] and h["case_number"] != q["case_number"]][:5]

            candidates_data = []
            md_lines.append(f"### Query `{q['query_id']}` (Case {q['case_number']})")
            md_lines.append(f"- **Primary Guideline:** `{q['primary_guideline']}` | **Direction:** `{q['direction']}` | **Regime:** `{q['guideline_regime']}`")
            md_lines.append(f"- **Fact Pattern:** \"{q['query_text'][:250]}...\"\n")
            md_lines.append("| Rank | Case Number | Guidelines | Outcome | Score (RRF) | Rule Match | Human Label [0/1] | Notes |")
            md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

            for rank, c in enumerate(clean_hits, 1):
                rule_match = 1 if is_precedent_relevant(q, c) else 0
                g_str = ",".join(c.get("guidelines") or [])
                md_lines.append(
                    f"| {rank} | [{c['case_number']}]({c['source_url']}) | {g_str} | {c['outcome']} | {c['score']:.4f} | {rule_match} | [ ] | |"
                )
                candidates_data.append({
                    "rank": rank,
                    "case_number": c["case_number"],
                    "source_url": c["source_url"],
                    "guidelines": c.get("guidelines") or [],
                    "outcome": c["outcome"],
                    "score": c["score"],
                    "dense_score": c.get("dense_score"),
                    "bm25_score": c.get("bm25_score"),
                    "rule_relevant": rule_match,
                    "human_label": None,
                })

            md_lines.append("")
            team_export[member].append({
                "query_id": q["query_id"],
                "case_number": q["case_number"],
                "source_url": q["source_url"],
                "primary_guideline": q["primary_guideline"],
                "guidelines": q.get("guidelines") or [],
                "direction": q["direction"],
                "query_text": q["query_text"],
                "candidates": candidates_data,
            })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(team_export, f, indent=2)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Exported annotation tasks to {output_json} and {output_md}")
    return team_export


def main():
    import argparse
    from src.retrieval import PrecedentRetriever

    parser = argparse.ArgumentParser(description="50-Query Precedent Retrieval Benchmark (Task 3 & 10)")
    parser.add_argument("--build", action="store_true", help="Build or rebuild the 50-query test set")
    parser.add_argument("--eval", action="store_true", help="Run retrieval evaluation across dense, bm25, and hybrid")
    parser.add_argument("--export-annotations", action="store_true", help="Export labeling batches for the 4 team members")
    args = parser.parse_args()

    if args.build:
        build_50_query_test_set()
        return

    retriever = PrecedentRetriever()

    if args.export_annotations:
        export_team_annotation_tasks(retriever)
        return

    if args.eval:
        print("Running Precedent Retrieval Benchmark across 50 Stratified Test Queries...")
        queries = load_query_set()
        results = evaluate_retrieval_benchmark(retriever, queries=queries, modes=["dense", "bm25", "hybrid"])

        print("\n" + "=" * 80)
        print("PRE-DEFINED RELEVANCE RETRIEVAL BENCHMARK RESULTS (50 QUERIES)")
        print("=" * 80)
        print(f"{'Metric':<24} | {'Dense (FAISS)':<16} | {'BM25':<16} | {'Hybrid (RRF)':<16}")
        print("-" * 80)
        for metric in ["precision@1", "precision@3", "precision@5", "mrr", "exact_docket_hit@1", "exact_docket_hit@5"]:
            d_val = results["dense"].get(metric, 0.0)
            b_val = results["bm25"].get(metric, 0.0)
            h_val = results["hybrid"].get(metric, 0.0)
            print(f"{metric:<24} | {d_val:<16.4f} | {b_val:<16.4f} | {h_val:<16.4f}")
        print("=" * 80)

        out_res = config.OUT_DIR / "retrieval_benchmark_results.json"
        with open(out_res, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved benchmark results to {out_res}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
