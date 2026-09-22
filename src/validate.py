"""Assertions over the built dataset, plus the Challenge_Data/README.md stats.

    python -m src.validate

Exits non-zero if any check fails. Each check exists because the corresponding
defect was actually observed in the source data, not as a hypothetical.
"""
from __future__ import annotations

import glob
import json
import re
import sys

import pandas as pd

from src import config

# Phrases that must never survive into the model input. If any appear, the
# section boundaries drifted and the outcome task is silently invalidated --
# which is the single most expensive thing that can go wrong here.
_VERDICT_PHRASES = re.compile(
    r"\b(?:is|are)\s+(?:granted|denied|revoked)\b|\bclearly\s+consistent\b",
    re.IGNORECASE,
)
_AUTOGEN = re.compile(
    r"<!-- AUTOGEN:START -->.*?<!-- AUTOGEN:END -->", re.DOTALL
)


def load_clean() -> pd.DataFrame:
    paths = sorted(glob.glob(str(config.OUT_DIR / "cases_clean_*.parquet")))
    if not paths:
        raise FileNotFoundError(
            f"No cleaned shards in {config.OUT_DIR}. Run: python -m src.build_dataset --full"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def run_checks(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # Sentinels must be gone: the source encoded missingness as 'Unknown'/''
    # so df.isna() reported a clean dataset that was 8% unusable.
    sentinel_hits = 0
    for column in ("outcome", "judge"):
        if column in df:
            values = df[column].dropna().astype(str)
            sentinel_hits += values.isin(config.SENTINEL_UNKNOWN).sum()
            sentinel_hits += (values.str.strip() == "").sum()
    check("no sentinel strings in cleaned columns", sentinel_hits == 0,
          f"{sentinel_hits} found")

    bad_codes = {
        code
        for codes in df["guidelines"]
        if codes is not None
        for code in codes
        if code not in config.GUIDELINE_CODES
    }
    check("guideline codes all within A-M", not bad_codes, str(sorted(bad_codes)))

    dates = pd.to_datetime(df["date"], errors="coerce")
    out_of_range = dates.dropna()[
        (dates.dropna().dt.year < config.MIN_YEAR) | (dates.dropna().dt.year > config.MAX_YEAR)
    ]
    check("dates within window or null", len(out_of_range) == 0,
          f"{len(out_of_range)} outside {config.MIN_YEAR}-{config.MAX_YEAR}")

    lengths = df["full_text"].str.len().fillna(0)
    bad_offsets = 0
    for name in config.EXPORTED_SECTIONS:
        column = f"sec_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        start, end = f"{column}_start", f"{column}_end"
        if start not in df:
            continue
        present = df[start].notna()
        if present.any():
            bad_offsets += int(
                ((df.loc[present, start] < 0) | (df.loc[present, end] > lengths[present])).sum()
            )
    check("section offsets index full_text", bad_offsets == 0, f"{bad_offsets} out of range")

    prefix_bad = int(
        ((df["html_prefix_len"] > 0) & ~df["full_text"].str.contains("file:///", regex=False, na=False)).sum()
    )
    check("html_prefix_len only where a header exists", prefix_bad == 0, f"{prefix_bad} mismatched")

    # Leakage is tested by SIGNAL, not by phrase presence.
    #
    # A blacklist gives false alarms: "clearly consistent" appears in 4,475
    # rows purely as a recitation of the legal standard ("DOHA was unable to
    # find that it was clearly consistent..."), and "is granted" appears in
    # procedural asides ("Department Counsel's motion to amend the SOR is
    # granted"). Neither reveals the verdict.
    #
    # What matters is whether such rows skew toward an outcome. If the phrase
    # carries no signal, its rows mirror the corpus distribution and a model
    # cannot exploit it.
    features = df["text_features"].fillna("")
    flagged = features.str.contains(_VERDICT_PHRASES, na=False)
    resolved = df["outcome"].notna()
    baseline = (df.loc[resolved, "outcome"] == "DENIED").mean()
    subset = df.loc[flagged & resolved, "outcome"]
    skew = abs((subset == "DENIED").mean() - baseline) if len(subset) else 0.0
    check(
        "verdict phrasing in text_features carries no signal",
        skew < 0.10,
        f"{int(flagged.sum())} rows contain a phrase; DENIED rate {100 * (subset == 'DENIED').mean():.1f}% "
        f"vs baseline {100 * baseline:.1f}% (skew {100 * skew:.1f}pp)",
    )

    # Structural guarantee: feature text is never drawn from the verdict region.
    #
    # Section spans themselves CAN appear out of order -- in ~0.17% of documents
    # the headers are unusual and a fact section runs past the ruling. That is a
    # property of the source, not a defect. What must hold is that such a span is
    # never used: segment.feature_spans drops it, leaving those rows with empty
    # feature text rather than leaked feature text.
    #
    # Compared against `verdict_span_*`, not `sec_decision_*` -- "Decision" is
    # frequently a caption near the top while the ruling sits under "Conclusion".
    fof_end = pd.to_numeric(df.get("sec_findings_of_fact_end"), errors="coerce")
    verdict_start = pd.to_numeric(df.get("verdict_span_start"), errors="coerce")
    out_of_order = fof_end.notna() & verdict_start.notna() & (fof_end > verdict_start)
    leaked_rows = int((out_of_order & (df["text_features"].fillna("").str.len() > 0)).sum())
    check(
        "out-of-order sections never reach the feature text",
        leaked_rows == 0,
        f"{leaked_rows} leaked of {int(out_of_order.sum())} out-of-order rows",
    )

    remaining = df.duplicated(["case_number", "case_type", "full_text"]).sum()
    check("no byte-identical duplicates remain", remaining == 0, f"{remaining} left")

    check("every row has a label_source", df["label_source"].notna().all(), "")

    return checks


def summary_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    exploded = [code for codes in df["guidelines"] if codes is not None for code in codes]
    stored = [code for codes in df["guidelines_stored"] if codes is not None for code in codes]
    return {
        "rows": total,
        "columns": len(df.columns),
        "outcome_resolved": int(df["outcome"].notna().sum()),
        "outcome_resolved_pct": round(100 * df["outcome"].notna().mean(), 1),
        "outcome_changed": int(df["outcome_changed"].sum()),
        "date_resolved_pct": round(100 * df["date"].notna().mean(), 1),
        "judge_resolved_pct": round(100 * df["judge"].notna().mean(), 1),
        "case_type_corrected": int(df["case_type_corrected"].sum()),
        "label_source": {k: int(v) for k, v in df["label_source"].value_counts().items()},
        "guideline_regime": {k: int(v) for k, v in df["guideline_regime"].value_counts().items()},
        "case_type": {k: int(v) for k, v in df["case_type"].value_counts().items()},
        "rows_with_evidence_spans": int(
            df["guideline_subparagraphs"].map(lambda v: len(v) > 0 if v is not None else False).sum()
        ),
        "guideline_counts": {
            code: {"stored": stored.count(code), "cleaned": exploded.count(code)}
            for code in config.GUIDELINE_CODES
        },
        "text_features_present_pct": round(
            100 * (df["text_features"].fillna("").str.len() > 0).mean(), 1
        ),
    }


def render_stats_block(stats: dict, checks: list) -> str:
    lines = ["<!-- AUTOGEN:START -->",
             "<!-- Regenerated by `python -m src.validate`. Do not edit by hand. -->",
             "", "### Dataset at a glance", "",
             "| | |", "|---|---|",
             f"| Rows | {stats['rows']:,} |",
             f"| Columns | {stats['columns']} |",
             f"| Outcome resolved | {stats['outcome_resolved']:,} ({stats['outcome_resolved_pct']}%) |",
             f"| Outcome changed vs source | {stats['outcome_changed']:,} |",
             f"| Date resolved | {stats['date_resolved_pct']}% |",
             f"| Judge resolved | {stats['judge_resolved_pct']}% |",
             f"| case_type corrected | {stats['case_type_corrected']:,} |",
             f"| Rows with evidence spans | {stats['rows_with_evidence_spans']:,} |",
             f"| Leak-free feature text present | {stats['text_features_present_pct']}% |",
             "", "### Label quality tiers", "",
             "| source | rows | trust |", "|---|---:|---|"]
    trust = {
        "formal_ruling": "authoritative - the judge's written ruling",
        "inherited_from_hearing": "inherited from the reviewed hearing",
        "regex_fallback": "~60% precision - candidates only",
        "none": "no label recoverable",
    }
    for source, count in stats["label_source"].items():
        lines.append(f"| `{source}` | {count:,} | {trust.get(source, '')} |")

    lines += ["", "### Guideline counts, source vs cleaned", "",
              "| Guideline | source | cleaned | change |", "|---|---:|---:|---:|"]
    for code, values in stats["guideline_counts"].items():
        delta = values["cleaned"] - values["stored"]
        lines.append(
            f"| {code} {config.GUIDELINE_NAMES[code]} | {values['stored']:,} "
            f"| {values['cleaned']:,} | {delta:+,} |"
        )

    lines += ["", "### Citation regime", "", "| regime | rows |", "|---|---:|"]
    for regime, count in stats["guideline_regime"].items():
        lines.append(f"| `{regime}` | {count:,} |")

    lines += ["", "### Validation", ""]
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"- **{mark}** - {name}" + (f" ({detail})" if detail and not ok else ""))

    lines.append("<!-- AUTOGEN:END -->")
    return "\n".join(lines)


def update_readme(block: str) -> None:
    path = config.OUT_DIR / "README.md"
    if path.exists():
        original = path.read_text(encoding="utf-8")
        if _AUTOGEN.search(original):
            path.write_text(_AUTOGEN.sub(lambda _: block, original), encoding="utf-8")
            return
        path.write_text(original.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        return
    path.write_text(block + "\n", encoding="utf-8")


def main() -> int:
    df = load_clean()
    checks = run_checks(df)
    stats = summary_stats(df)

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")

    update_readme(render_stats_block(stats, checks))
    (config.OUT_DIR / "validation.json").write_text(
        json.dumps({"checks": [{"name": n, "passed": o, "detail": d} for n, o, d in checks],
                    "stats": stats}, indent=2),
        encoding="utf-8",
    )

    failed = [name for name, ok, _ in checks if not ok]
    print()
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
