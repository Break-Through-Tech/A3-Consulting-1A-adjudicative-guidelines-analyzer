"""Build the cleaned dataset.

    python -m src.build_dataset --full
    python -m src.build_dataset --sample 1000

Reads doha's parquet shards (read-only), repairs what is broken, writes
Challenge_Data/. The source is never modified.

Pipeline order matters in one place: the HTML scrape header is stripped BEFORE
segmentation, because it occupies the first ~150 characters -- exactly the
region every header-based extractor reads.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config, dedupe, fields, labels, load, normalize, segment


def _span_columns(spans: dict, offset: int) -> dict:
    """Section spans as flat start/end columns, re-based onto original full_text."""
    shifted = segment.shift(spans, offset)
    out = {}
    for name in config.EXPORTED_SECTIONS:
        column = segment.column_name(name)
        span = shifted.get(name)
        out[f"{column}_start"] = span[0] if span else pd.NA
        out[f"{column}_end"] = span[1] if span else pd.NA
    return out


def process_row(row) -> dict:
    """All per-document work. Offsets returned index the ORIGINAL full_text."""
    raw = row.full_text or ""
    body, prefix_len = normalize.strip_html_header(raw)

    spans, used_fuzzy, partial = segment.segment(body)
    verdict = segment.verdict_span(spans)
    feature_spans = segment.feature_spans(spans)

    outcome, outcome_source = fields.extract_outcome(body, verdict)
    date, date_source = fields.extract_date(body, row.case_number)
    judge, judge_status = fields.extract_judge(body)
    case_type, case_type_corrected = fields.extract_case_type(body, row.case_type)
    regime = fields.extract_regime(body)

    codes, findings, subparagraphs, label_source = labels.label_row(
        body, spans.get("POLICIES")
    )
    for item in subparagraphs:
        item["offset"] += prefix_len

    record = {
        "outcome": outcome,
        "outcome_source": outcome_source,
        "outcome_changed": bool(
            isinstance(outcome, str) and outcome != row.outcome
        ),
        "date": date,
        "date_source": date_source,
        "judge": judge,
        "judge_status": judge_status,
        "case_type": case_type,
        "case_type_corrected": case_type_corrected,
        "guideline_regime": regime,
        "guidelines": codes,
        "guideline_findings": findings,
        "guideline_subparagraphs": subparagraphs,
        "label_source": label_source,
        "html_prefix_len": prefix_len,
        # Where the outcome was actually read from. This is NOT sec_decision:
        # "Decision" often appears as a caption label at ~3% through the
        # document while the ruling sits under "Conclusion" at ~98%. The
        # verdict region is chosen by position, so it is recorded separately.
        "verdict_span_start": (verdict[0] + prefix_len) if verdict else pd.NA,
        "verdict_span_end": (verdict[1] + prefix_len) if verdict else pd.NA,
        "segmentation_fuzzy": used_fuzzy,
        "segmentation_partial": partial,
        "case_year": fields.docket_year(row.case_number),
        "text_features": normalize.build_features(body, feature_spans),
        "para_offsets_fof": [
            (s + prefix_len, e + prefix_len)
            for s, e in segment.paragraph_offsets(body, spans.get("FINDINGS OF FACT"))
        ],
    }
    record.update(_span_columns(spans, prefix_len))
    return record


def inherit_appeal_labels(df: pd.DataFrame, texts: list[str]) -> int:
    """Tier 1b: an appeal inherits the guidelines of the hearing it reviews.

    Sound because the Appeal Board reviews the same alleged guidelines, and
    95.8% of appeals cite their own case number. Still inference rather than
    observation, so it gets its own label_source.
    """
    hearings = {
        row.case_number: row.guidelines
        for row in df.itertuples(index=False)
        if row.label_source == labels.SOURCE_FORMAL and len(row.guidelines) > 0
    }
    if not hearings:
        return 0

    inherited = 0
    for index, (row, text) in enumerate(zip(df.itertuples(index=False), texts)):
        if row.label_source == labels.SOURCE_FORMAL:
            continue
        candidates = [row.case_number] + labels.cited_case_numbers(text)
        for candidate in candidates:
            source = hearings.get(candidate)
            if source:
                df.at[df.index[index], "guidelines"] = list(source)
                df.at[df.index[index], "label_source"] = labels.SOURCE_INHERITED
                inherited += 1
                break
    return inherited


def build(limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    raw, stats = load.load_raw()
    if limit:
        raw = raw.sample(n=min(limit, len(raw)), random_state=42).reset_index(drop=True)

    records = [process_row(row) for row in raw.itertuples(index=False)]
    derived = pd.DataFrame.from_records(records, index=raw.index)

    # Preserve every original value that we produce a cleaned version of.
    out = raw.rename(columns={
        "outcome": "outcome_stored",
        "guidelines": "guidelines_stored",
        "judge": "judge_raw",
        "date": "date_stored",
        "case_type": "case_type_stored",
    })
    out = pd.concat([out, derived], axis=1)

    inherited = inherit_appeal_labels(out, list(raw["full_text"]))
    stats["labels_inherited"] = inherited

    out = dedupe.classify(out)
    excluded = dedupe.excluded_report(out)
    stats["rows_before_dedupe"] = len(out)
    stats["true_duplicates_deleted"] = int(out["is_duplicate"].sum())
    stats["case_number_collisions"] = int(out["case_number_collision"].sum())
    stats["degenerate_flagged"] = int(out["is_degenerate"].sum())
    out = dedupe.drop_duplicates(out)
    stats["rows_after_dedupe"] = len(out)
    stats["elapsed_seconds"] = round(time.time() - started, 1)
    return out, stats, excluded


def write(df: pd.DataFrame, stats: dict, excluded: pd.DataFrame) -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split by row position rather than np.array_split, which returns ndarrays
    # on current numpy and loses the DataFrame.
    total = len(df)
    size = -(-total // config.N_SHARDS)  # ceiling division
    shards = [df.iloc[start:start + size] for start in range(0, total, size)]
    written = []
    for index, shard in enumerate(shards, start=1):
        path = config.OUT_DIR / f"cases_clean_{index}.parquet"
        shard.to_parquet(
            path, index=False,
            engine=config.PARQUET_ENGINE, compression=config.PARQUET_COMPRESSION,
        )
        with open(path, "rb") as handle:          # parquet footer magic
            handle.seek(-4, 2)
            if handle.read(4) != b"PAR1":
                raise ValueError(f"{path.name} is not a valid parquet file")
        written.append({"file": path.name, "rows": len(shard),
                        "size_mb": round(path.stat().st_size / 1024**2, 1)})
    stats["shards_written"] = written

    excluded.to_csv(config.OUT_DIR / "excluded_rows.csv", index=False)

    # Judge spelling variants that share a (first, last) key. Emitted for human
    # confirmation rather than auto-merged: Jean E. Smallin and Jean S. Smallin
    # may be two different people.
    groups: dict[tuple, set] = {}
    for name in df["judge"].dropna().unique():
        key = fields.name_key(name)
        if key != ("", ""):
            groups.setdefault(key, set()).add(name)
    variants = [
        {"first": k[0], "last": k[1], "variants": " | ".join(sorted(v))}
        for k, v in groups.items() if len(v) > 1
    ]
    pd.DataFrame(variants).to_csv(config.OUT_DIR / "judge_review.csv", index=False)
    stats["judge_variant_groups"] = len(variants)

    (config.OUT_DIR / "build_stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the cleaned DOHA dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="process every row")
    group.add_argument("--sample", type=int, metavar="N", help="process N random rows")
    args = parser.parse_args()

    df, stats, excluded = build(limit=None if args.full else args.sample)
    write(df, stats, excluded)

    print(f"rows   {stats['rows_before_dedupe']:,} -> {stats['rows_after_dedupe']:,} "
          f"({stats['true_duplicates_deleted']:,} true duplicates deleted)")
    print(f"labels {dict(df['label_source'].value_counts())}")
    print(f"time   {stats['elapsed_seconds']}s")
    print(f"wrote  {config.OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
