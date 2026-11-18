"""Draw the 100-finding traceability sample (task 31) and score it once it is filled in.

    python -m src.build_traceability                       # write the sheet and the key
    python -m src.build_traceability --score filled.csv    # pass rates from a filled sheet

Produces:
    Challenge_Data/traceability.csv       the reviewer sheet, with no hint of the source
    Challenge_Data/traceability_key.csv   finding id, source (rule or llm) and extractor detail

A finding is one condition code attached to one span of a decision's text. The
question for each finding is whether that span supports the condition. A kept
quote can be a real passage and still not support its condition, and an
extractor hit can be a real citation that the sentence does not apply. Neither
the offset check in src/llm/verify.py nor the pilot score tells us which, so a
person reads each one.

The sample is 100 findings. Seed 42 throughout.

Rule side, 50 findings
    1. Pool cases that put Guideline F in issue are sorted by source_url, and 400
       are drawn with `DataFrame.sample(random_state=42)`.
    2. `conditions.extract_case` runs over those cases. Each hit's code is its
       `sead4` field when it has one, else its `code`, as src/llm/pilot.py does.
       A hit is dropped when that code is not on the closed list in
       src.sead4_reference, or when the text at its offsets is blank. Directive
       5220.6 hits with no single SEAD-4 code fall out here.
    3. Quotas by regime follow each regime's share of the hit pool, with a floor
       of 5 findings for every regime that has hits. Without a floor the two
       small regimes would get two findings each, too few to read anything from.
       The largest regime gives up the difference.
    4. Inside a regime, hits are grouped by code and each group is shuffled. The
       draw then goes round-robin across the codes, one hit per code per pass,
       until the quota is met. No single code can dominate.

LLM side, 50 findings
    The kept quotes of the pilot in results/llm/pilot/kept.csv, joined to
    results/llm/pilot/cases.csv for the case. The pilot cases are all SEAD-4
    cases, so the quota is 50 in that regime. The citation and blind modes often
    keep the same span, so rows are merged on case, code and offsets, and the
    key lists both modes. The same round-robin runs over the codes. When the
    task 4 run on the test split finishes, point KEPT_PATH and CASES_PATH at it.

Both sides
    Every candidate is re-checked against the case's full_text: the quote must
    equal full_text[start:end]. A failure is dropped, counted and replaced by the
    next candidate. A case is used at most twice across the 100 findings. The
    rule side is drawn first, then the LLM side. The 100 rows are shuffled so the
    sources interleave, numbered 1 to 100, and handed out in blocks of 25.

Context
    A rule hit's span is usually a bare citation such as `(a)`, which a reviewer
    can't judge alone. Every row therefore also shows `context`, the sentence or
    list item around the span from `conditions.hit_context`, and `lead_in`, the
    sentence that opens the list from `conditions.list_lead_in`, blank outside a
    list. The lead-in shows whether the judge applied the listed conditions or
    only recited them. Both sources get the same two columns.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src import config
from src.conditions import _CASE_COLUMNS, extract_case, hit_context, list_lead_in
from src.dataset import load_cases
from src.sead4_reference import CODES, condition

SHEET_PATH = config.OUT_DIR / "traceability.csv"
KEY_PATH = config.OUT_DIR / "traceability_key.csv"
KEPT_PATH = config.REPO_ROOT / "results" / "llm" / "pilot" / "kept.csv"
CASES_PATH = config.REPO_ROOT / "results" / "llm" / "pilot" / "cases.csv"

SEED = 42
N_RULE = 50
N_LLM = 50
RULE_CASES = 400
REGIME_FLOOR = 5
MAX_PER_CASE = 2
REVIEWERS = [("Rashed", 25), ("Sam", 25), ("Aryan", 25), ("Brian", 25)]
SHEET_COLUMNS = ["finding", "reviewer", "case_number", "decision", "date", "regime", "code",
                 "condition_text", "quote", "context", "lead_in", "start", "end", "supports", "notes",
                 "source_url"]
ANSWERS = ("yes", "no", "unsure")


# --- candidates ----------------------------------------------------------------------

def load_pool() -> pd.DataFrame:
    """Pool cases that put Guideline F in issue, with full_text, sorted by source_url."""
    df = load_cases(columns=_CASE_COLUMNS + ["split"])
    has_f = df["guidelines"].map(lambda g: g is not None and "F" in list(g))
    return df[has_f].sort_values("source_url").reset_index(drop=True)


def _usable(full_text: str, code: str | None, start: int, end: int) -> bool:
    return code in CODES and 0 <= start < end <= len(full_text) and bool(full_text[start:end].strip())


def rule_candidates(pool: pd.DataFrame, n_cases: int = RULE_CASES, seed: int = SEED) -> tuple[list[dict], dict]:
    """Every usable extractor hit in a fixed-seed sample of pool cases."""
    sample = pool.sample(n=n_cases, random_state=seed).sort_values("source_url")
    out, dropped = [], Counter()
    began = time.perf_counter()
    for row in sample.to_dict("records"):
        for hit in extract_case(row):
            code = hit.get("sead4") or hit["code"]
            if not _usable(row["full_text"], code, hit["start"], hit["end"]):
                dropped["off_list_or_blank"] += 1
                continue
            out.append({
                "source": "rule", "source_url": row["source_url"], "code": code,
                "start": hit["start"], "end": hit["end"],
                "detail": {"extractor_code": hit["code"], "kind": hit["kind"],
                           "numbered": bool(hit.get("numbered", False))},
            })
    stats = {"cases": len(sample), "seconds": time.perf_counter() - began,
             "hits": len(out), "dropped": dict(dropped)}
    return out, stats


def llm_candidates() -> list[dict]:
    """Kept pilot quotes, one per case, code and span, with the modes that kept it."""
    kept = pd.read_csv(KEPT_PATH, keep_default_na=False)
    cases = pd.read_csv(CASES_PATH, keep_default_na=False)
    # The committed file has LF line breaks inside quotes. A Windows checkout with
    # autocrlf turns them into CRLF, which would fail the offset check for no real reason.
    kept["quote"] = kept["quote"].str.replace("\r\n", "\n", regex=False)
    kept = kept.merge(cases[["custom_id", "source_url"]], on="custom_id", how="left", validate="m:1")
    if kept["source_url"].eq("").any() or kept["source_url"].isna().any():
        raise ValueError("a kept quote has no case in cases.csv")
    merged: dict[tuple, dict] = {}
    for r in kept.sort_values(["source_url", "start", "end", "code", "custom_id"]).to_dict("records"):
        key = (r["source_url"], r["code"], int(r["start"]), int(r["end"]))
        if key not in merged:
            merged[key] = {
                "source": "llm", "source_url": r["source_url"], "code": r["code"],
                "start": int(r["start"]), "end": int(r["end"]), "quote": r["quote"],
                "detail": {"custom_ids": [], "modes": []},
            }
        merged[key]["detail"]["custom_ids"].append(r["custom_id"])
        merged[key]["detail"]["modes"].append(r["mode"])
    return list(merged.values())


# --- the draw --------------------------------------------------------------------------

def quotas(counts: dict[str, int], n: int, floor: int = REGIME_FLOOR) -> dict[str, int]:
    """Findings per regime: proportional to `counts`, at least `floor`, summing to `n`."""
    live = {k: v for k, v in counts.items() if v > 0}
    total = sum(live.values())
    out = {k: max(min(floor, v), round(n * v / total)) for k, v in live.items()}
    while sum(out.values()) > n:
        biggest = max(out, key=lambda k: (out[k] - min(floor, live[k]), k))
        out[biggest] -= 1
    while sum(out.values()) < n:
        room = {k: live[k] - out[k] for k in out if live[k] > out[k]}
        out[max(room, key=lambda k: (room[k], k))] += 1
    return out


def draw(candidates: list[dict], n: int, rng: np.random.Generator, regimes: dict[str, str],
         full_text: dict[str, str], case_use: Counter, floor: int = REGIME_FLOOR) -> tuple[list[dict], int]:
    """`n` findings spread over regimes, then codes. Returns them and the offset failures."""
    by_regime: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(candidates, key=lambda c: (c["source_url"], c["start"], c["end"], c["code"])):
        by_regime[regimes[c["source_url"]]][c["code"]].append(c)
    plan = quotas({r: sum(len(v) for v in codes.values()) for r, codes in by_regime.items()}, n, floor)

    chosen, failures = [], 0
    for regime in sorted(plan):
        groups = by_regime[regime]
        codes = sorted(groups)
        for code in codes:
            order = rng.permutation(len(groups[code]))
            groups[code] = [groups[code][i] for i in order]
        codes = [codes[i] for i in rng.permutation(len(codes))]
        taken = 0
        while taken < plan[regime] and any(groups[c] for c in codes):
            for code in codes:
                if taken >= plan[regime]:
                    break
                while groups[code]:
                    cand = groups[code].pop(0)
                    if case_use[cand["source_url"]] >= MAX_PER_CASE:
                        continue
                    text = full_text[cand["source_url"]]
                    quote = text[cand["start"]:cand["end"]]
                    if "quote" in cand and cand["quote"] != quote:
                        failures += 1
                        continue
                    cand["quote"] = quote
                    case_use[cand["source_url"]] += 1
                    chosen.append(cand)
                    taken += 1
                    break
        if taken < plan[regime]:
            raise RuntimeError(f"only {taken} of {plan[regime]} findings available for {regime}")
    return chosen, failures


def reviewer_for(finding: int) -> str:
    """Who reads finding number `finding` (1-based)."""
    end = 0
    for name, count in REVIEWERS:
        end += count
        if finding <= end:
            return name
    raise ValueError(f"finding {finding} is beyond the {end} assigned")


def build_sample(seed: int = SEED, pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(sheet, key, stats). The same seed gives the same rows."""
    pool = load_pool() if pool is None else pool
    regimes = dict(zip(pool["source_url"], pool["guideline_regime"]))
    full_text = dict(zip(pool["source_url"], pool["full_text"]))
    rng = np.random.default_rng(seed)
    case_use: Counter = Counter()

    rule_pool, rule_stats = rule_candidates(pool, RULE_CASES, seed)
    llm_pool = [c for c in llm_candidates() if c["source_url"] in regimes]
    rule, _ = draw(rule_pool, N_RULE, rng, regimes, full_text, case_use)
    llm, failures = draw(llm_pool, N_LLM, rng, regimes, full_text, case_use)
    stats = {"rule": rule_stats, "llm_pool": len(llm_pool), "llm_offset_failures": failures}

    findings = rule + llm
    findings = [findings[i] for i in rng.permutation(len(findings))]
    info = pool.set_index("source_url")[["case_number", "date", "guideline_regime", "split"]]
    sheet_rows, key_rows = [], []
    for number, f in enumerate(findings, start=1):
        case = info.loc[f["source_url"]]
        d = f["detail"]
        text = full_text[f["source_url"]]
        c_start, c_end = hit_context(text, f["start"], f["end"])
        lead = list_lead_in(text, c_start)
        sheet_rows.append({
            "finding": number,
            "reviewer": reviewer_for(number),
            "case_number": case["case_number"],
            "decision": f'=HYPERLINK("{f["source_url"]}","{case["case_number"]}")',
            "date": case["date"].date().isoformat() if pd.notna(case["date"]) else "",
            "regime": case["guideline_regime"],
            "code": f["code"],
            "condition_text": condition(f["code"])["text"],
            "quote": f["quote"],
            "context": text[c_start:c_end],
            "lead_in": "" if lead is None else text[lead[0]:lead[1]],
            "start": f["start"],
            "end": f["end"],
            "supports": "",
            "notes": "",
            "source_url": f["source_url"],
        })
        key_rows.append({
            "finding": number,
            "source": f["source"],
            "source_url": f["source_url"],
            "case_number": case["case_number"],
            "split": case["split"],
            "regime": case["guideline_regime"],
            "code": f["code"],
            "start": f["start"],
            "end": f["end"],
            "custom_id": "|".join(d.get("custom_ids", [])),
            "mode": "|".join(d.get("modes", [])),
            "extractor_code": d.get("extractor_code", ""),
            "kind": d.get("kind", ""),
            "numbered": d.get("numbered", ""),
        })
    return pd.DataFrame(sheet_rows, columns=SHEET_COLUMNS), pd.DataFrame(key_rows), stats


# --- scoring -----------------------------------------------------------------------------

def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for `passes` out of `n`."""
    if n == 0:
        return (math.nan, math.nan)
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def score_sheet(filled: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One row per source, then per source and regime: yes, no, unsure, answered, rate, interval."""
    sheet = filled[["finding", "supports"]].copy()
    sheet["supports"] = sheet["supports"].fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(sheet["supports"]) - set(ANSWERS) - {""})
    if bad:
        raise ValueError(f"supports must be yes, no or unsure; found {bad}")
    sheet["finding"] = sheet["finding"].astype(int)
    df = sheet.merge(key[["finding", "source", "regime"]], on="finding", how="left", validate="1:1")
    if df["source"].isna().any():
        raise ValueError("a finding in the sheet is not in the key")
    rows = []
    groups = [(s, "all", df[df["source"] == s]) for s in ("rule", "llm")]
    for s in ("rule", "llm"):
        for regime in sorted(df.loc[df["source"] == s, "regime"].unique()):
            groups.append((s, regime, df[(df["source"] == s) & (df["regime"] == regime)]))
    for source, regime, part in groups:
        yes, no = int((part["supports"] == "yes").sum()), int((part["supports"] == "no").sum())
        low, high = wilson(yes, yes + no)
        rows.append({"source": source, "regime": regime, "findings": len(part), "yes": yes, "no": no,
                     "unsure": int((part["supports"] == "unsure").sum()),
                     "blank": int((part["supports"] == "").sum()),
                     "answered": yes + no, "pass_rate": yes / (yes + no) if yes + no else math.nan,
                     "low": low, "high": high})
    return pd.DataFrame(rows)


def print_scores(table: pd.DataFrame) -> None:
    print("pass rate = yes / (yes + no); unsure and blank rows are not counted as passes or fails")
    for r in table.to_dict("records"):
        rate = "n/a" if r["answered"] == 0 else (
            f"{r['pass_rate']:.1%} (95% interval {r['low']:.1%} to {r['high']:.1%})")
        print(f"{r['source']:>4} {r['regime']:<9} {r['yes']} yes, {r['no']} no of {r['answered']} answered: {rate}; "
              f"{r['unsure']} unsure, {r['blank']} blank")


def run_score(path: str) -> None:
    filled = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(KEY_PATH, keep_default_na=False)
    print_scores(score_sheet(filled, key))


# --- command line --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", metavar="FILLED_CSV", help="print pass rates from a filled sheet")
    args = parser.parse_args(argv)
    if args.score:
        run_score(args.score)
        return

    sheet, key, stats = build_sample()
    sheet.to_csv(SHEET_PATH, index=False)
    key.to_csv(KEY_PATH, index=False)
    print(f"{len(sheet)} findings -> {SHEET_PATH}\nkey -> {KEY_PATH}")
    r = stats["rule"]
    print(f"extract_case over {r['cases']} cases: {r['seconds']:.1f} s, {r['hits']} usable hits, dropped {r['dropped']}")
    print(f"LLM candidates after merging modes: {stats['llm_pool']}; offset re-check failures: {stats['llm_offset_failures']}")
    print("source x regime:\n", pd.crosstab(key["source"], key["regime"]))
    print("source x code:\n", pd.crosstab(key["code"], key["source"]))
    print("cases used:", key["source_url"].nunique(), "; most uses of one case:", key["source_url"].value_counts().max())
    for name, _ in REVIEWERS:
        print(f"  {name}: {(sheet['reviewer'] == name).sum()} rows")


if __name__ == "__main__":
    main(sys.argv[1:])
