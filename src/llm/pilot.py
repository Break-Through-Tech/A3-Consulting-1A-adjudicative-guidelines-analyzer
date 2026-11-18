"""The Guideline F LLM pilot: cohort, cost prediction, batch runs, scoring, report.

    python -m src.llm.pilot predict             # free: count tokens for all 110 requests
    python -m src.llm.pilot test                # paid: 5 cases, both modes
    python -m src.llm.pilot pilot               # paid: 50 cases, both modes
    python -m src.llm.pilot score test|pilot    # free: verify quotes, score against src/conditions.py
    python -m src.llm.pilot report              # free: results/llm/REPORT.md

    python -m src.llm.pilot run-predict         # free: count tokens for the 174 test-split requests
    python -m src.llm.pilot run-test            # paid: 10-case batch, 5 uncited and 5 cited
    python -m src.llm.pilot run-full            # paid: the other 164 cases
    python -m src.llm.pilot run-score           # free: results/llm/run/*.csv from both batches
    python -m src.llm.pilot run-report          # free: "Test split run" section of REPORT.md

Spend is capped at $1.00 in results/llm/spend.jsonl. A stage with a saved batch id
is resumed, never resubmitted, unless --resubmit is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.llm import client as C
from src.llm import templates as T
from src.llm.verify import total_counts, verify_guideline_quotes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "llm"
STAGES = {"test": (50, 55), "pilot": (0, 50)}   # slice of the shuffled eligible list
SEED = 42
BOOTSTRAPS = 2000
CHARS_PER_TOKEN = 3.08
SYSTEM_TOKENS = 1800
OUTPUT_PER_REQUEST = 2300          # estimate until the test run replaces it
CENTRAL_LIMIT = 0.50               # predicted central cost must not pass this

CHECK_VALUES = {                   # measured Nov 9, before whitespace is collapsed
    "eligible": 7624,
    "pilot_chars": 366_867,
    "pilot_codes": 319,
    "test_chars": [6269, 6106, 5881, 6717, 7433],
    "test_codes": 39,
}


# --- cohort ---------------------------------------------------------------------

def truth_codes(row, fallback: bool = True) -> set[str] | None:
    """Closed-list codes the judge cited, or None if any hit is off the list."""
    from src.conditions import extract_case

    closed = set(T.codes())
    codes = set()
    for hit in extract_case(row, fallback):
        code = hit.get("sead4") or hit["code"]
        if code not in closed:
            return None
        codes.add(code)
    return codes


def cohort() -> dict:
    """The eligible count, and the pilot and test-run cases in shuffled order."""
    from src.conditions import _CASE_COLUMNS
    from src.dataset import load_cases

    df = load_cases(columns=_CASE_COLUMNS + ["split"])
    has_f = df["guidelines"].map(lambda g: g is not None and "F" in list(g))
    df = df[has_f & (df["split"] == "train") & (df["guideline_regime"] == "sead4")]
    keep, truths = [], []
    for i, row in enumerate(df.to_dict("records")):
        if not T.sections(row, fallback=False):
            continue
        codes = truth_codes(row, fallback=False)
        if not codes:
            continue
        keep.append(i)
        truths.append(sorted(codes))
    eligible = df.iloc[keep].copy()
    eligible["truth"] = truths
    eligible = eligible.sort_values("source_url").reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(eligible))
    out = {"eligible": len(eligible)}
    for stage, (a, b) in STAGES.items():
        out[stage] = eligible.iloc[order[a:b]].reset_index(drop=True)
    return out


# --- requests --------------------------------------------------------------------

def custom_id(stage: str, position: int, mode: str) -> str:
    return f"{stage}-{position}-{mode}"


def build_requests(stage: str, cases) -> tuple[list[dict], list[dict]]:
    """(requests, case table rows). One request per case and mode."""
    requests, table = [], []
    for pos, row in enumerate(cases.to_dict("records")):
        for mode in T.MODES:
            text, region = T.build_input(row, mode, fallback=False)
            cid = custom_id(stage, pos, mode)
            requests.append({"custom_id": cid, "params": T.request_params(text, model=C.MODEL)})
            table.append({"custom_id": cid, "stage": stage, "position": pos, "mode": mode,
                          "source_url": row["source_url"], "case_number": row["case_number"],
                          "region_start": region[0], "region_end": region[1],
                          "input_chars": len(text), "truth": "|".join(row["truth"])})
    return requests, table


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- cost ledger -------------------------------------------------------------------

LEDGER = OUT / "spend.jsonl"


def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line]


def ledger_append(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def spent() -> float:
    """Actual cost where known, otherwise the ceiling, per batch."""
    per_batch: dict[str, float] = {}
    for r in ledger_rows():
        if r["event"] == "submit":
            per_batch.setdefault(r["batch_id"], r["ceiling"])
        elif r["event"] == "actual":
            per_batch[r["batch_id"]] = r["cost"]
    return sum(per_batch.values())


def guard(ceiling: float, budget: float = C.BUDGET) -> None:
    if spent() + ceiling > budget + 1e-12:
        raise SystemExit(f"refused: spent ${spent():.4f} plus ceiling ${ceiling:.4f} passes ${budget:.2f}")


def ceiling_for(input_tokens: int, n_requests: int, tier: str = "batch",
                max_tokens: int = 8000) -> float:
    return C.cost(input_tokens, n_requests * max_tokens, tier)


# --- predict -----------------------------------------------------------------------

def predict() -> dict:
    """Free. Count tokens for every real request and write cost_prediction.json."""
    co = cohort()
    cl = C.make_client()
    stages = {}
    for stage in STAGES:
        requests, table = build_requests(stage, co[stage])
        counts = {r["custom_id"]: C.count_tokens(cl, r["params"]) for r in requests}
        chars = [t["input_chars"] for t in table]
        n = len(requests)
        est_in = sum(CHARS_PER_TOKEN ** -1 * c + SYSTEM_TOKENS for c in chars)
        tin = sum(counts.values())
        stages[stage] = {
            "requests": n,
            "counted_input_tokens": tin,
            "estimated_input_tokens": round(est_in),
            "output_tokens_per_request": OUTPUT_PER_REQUEST,
            "output_tokens": n * OUTPUT_PER_REQUEST,
            "central_cost": C.cost(tin, n * OUTPUT_PER_REQUEST),
            "ceiling": ceiling_for(tin, n),
            "counts": counts,
        }
    pilot_cases, test_cases = co["pilot"], co["test"]
    cit_pilot = [T.build_input(r, "citation", collapse=False, fallback=False)[0] for r in pilot_cases.to_dict("records")]
    cit_test = [T.build_input(r, "citation", collapse=False, fallback=False)[0] for r in test_cases.to_dict("records")]
    checks = {
        "eligible": co["eligible"],
        "pilot_chars": sum(map(len, cit_pilot)),
        "pilot_codes": sum(len(t) for t in pilot_cases["truth"]),
        "test_chars": [len(t) for t in cit_test],
        "test_codes": sum(len(t) for t in test_cases["truth"]),
    }
    total = {
        "requests": sum(s["requests"] for s in stages.values()),
        "counted_input_tokens": sum(s["counted_input_tokens"] for s in stages.values()),
        "central_cost": sum(s["central_cost"] for s in stages.values()),
        "ceiling": sum(s["ceiling"] for s in stages.values()),
    }
    result = {"model": C.MODEL, "rates": C.RATES["batch"], "stages": stages, "total": total,
              "cohort_checks": checks, "cohort_expected": CHECK_VALUES,
              "cohort_checks_match": checks == CHECK_VALUES,
              "proceed": total["central_cost"] <= CENTRAL_LIMIT}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_prediction() -> dict:
    path = OUT / "cost_prediction.json"
    if not path.exists():
        raise SystemExit("run `predict` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --- running a stage -----------------------------------------------------------------

def parse_items(text: str | None) -> list | None:
    """The `items` list from a response, or None if the text is not the expected JSON."""
    try:
        data = json.loads(text or "")
        items = data["items"]
    except (ValueError, KeyError, TypeError):
        return None
    return items if isinstance(items, list) else None


def _state_path(stage: str) -> Path:
    return OUT / ("test_run" if stage == "test" else "pilot") / "batch.json"


def _stage_dir(stage: str) -> Path:
    return _state_path(stage).parent


def run_stage(stage: str, resubmit: bool = False, log=print) -> None:
    pred = load_prediction()
    if not pred["proceed"]:
        raise SystemExit("prediction is over the central limit; not running")
    d = _stage_dir(stage)
    state_path = _state_path(stage)
    co = cohort()
    requests, table = build_requests(stage, co[stage])
    write_csv(d / "cases.csv", table)
    cl = C.make_client()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    if state.get("batch_id") and not resubmit:
        log(f"resuming saved batch {state['batch_id']}")
    else:
        sp = pred["stages"][stage]
        ceiling = ceiling_for(sp["counted_input_tokens"], len(requests))
        guard(ceiling)
        batch_id = C.submit_batch(cl, requests)
        state = {"batch_id": batch_id, "tier": "batch"}
        state_path.write_text(json.dumps(state), encoding="utf-8")       # saved before polling
        ledger_append({"event": "submit", "stage": stage, "batch_id": batch_id, "tier": "batch",
                       "ceiling": ceiling})
        log(f"submitted batch {batch_id}, ceiling ${ceiling:.4f}")

    batch_id, tier = state["batch_id"], state.get("tier", "batch")
    results = state.get("results")
    if results is None:
        if tier == "batch" and C.wait_for_batch(cl, batch_id, log=log):
            results = C.collect_batch(cl, batch_id)
        else:
            if tier == "batch":                      # cancelled for running too long: bill what finished
                _record_actual(stage, batch_id, C.collect_batch(cl, batch_id), "batch")
            results = _fallback_standard(cl, stage, requests, pred, log)
            tier = "standard"
            state["tier"] = tier
        state["results"] = results
        state_path.write_text(json.dumps(state), encoding="utf-8")

    primary_key = batch_id if tier == "batch" else f"{stage}-standard"
    _record_actual(stage, primary_key, results, tier)       # before any retry is merged in

    # Retry errored requests once.
    bad = [r for r in requests if results.get(r["custom_id"], {}).get("status") != "succeeded"
           and not state.get("retried")]
    if bad:
        ceiling = ceiling_for(sum(pred["stages"][stage]["counts"][r["custom_id"]] for r in bad), len(bad), tier)
        guard(ceiling)
        log(f"retrying {len(bad)} requests once")
        if tier == "batch":
            rid = C.submit_batch(cl, bad)
            ledger_append({"event": "submit", "stage": stage, "batch_id": rid, "tier": "batch",
                           "ceiling": ceiling, "retry": True})
            retried = C.collect_batch(cl, rid) if C.wait_for_batch(cl, rid, log=log) else {}
            _record_actual(stage, rid, retried, "batch")
        else:
            retried = C.run_standard(cl, bad)
            _record_actual(stage, f"{stage}-standard-retry", retried, "standard")
        results.update({k: v for k, v in retried.items() if v["status"] == "succeeded"})
        state["retried"] = True
    state["results"] = results
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with (d / "raw.jsonl").open("w", encoding="utf-8") as f:
        for r in requests:
            cid = r["custom_id"]
            row = results.get(cid, {"status": "missing", "usage": None, "stop_reason": None,
                                    "text": None, "error": None})
            f.write(json.dumps({"custom_id": cid, **row}) + "\n")
    log(f"{stage}: wrote {d / 'raw.jsonl'}")


def _fallback_standard(cl, stage, requests, pred, log) -> dict:
    tin = pred["stages"][stage]["counted_input_tokens"]
    ceiling = ceiling_for(tin, len(requests), "standard")
    guard(ceiling)
    ledger_append({"event": "submit", "stage": stage, "batch_id": f"{stage}-standard",
                   "tier": "standard", "ceiling": ceiling})
    log(f"switching {stage} to the standard API at double rates, ceiling ${ceiling:.4f}")
    return C.run_standard(cl, requests)


def _record_actual(stage: str, key: str, results: dict, tier: str) -> None:
    """Write the actual cost of one batch (or standard run) to the ledger, once per key."""
    if any(r["event"] == "actual" and r["batch_id"] == key for r in ledger_rows()):
        return
    used = [v["usage"] for v in results.values() if v.get("usage")]
    ledger_append({"event": "actual", "stage": stage, "batch_id": key, "tier": tier,
                   "cost": sum(C.usage_cost(u, tier) for u in used),
                   "input_tokens": sum(u["input_tokens"] for u in used),
                   "output_tokens": sum(u["output_tokens"] for u in used)})


# --- scoring ---------------------------------------------------------------------------

def kind_of(code: str) -> str:
    return "disqualifier" if "19(" in code else "mitigator"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def micro(pairs: list[tuple[set, set]], kind: str | None = None) -> dict:
    """pairs: (truth set, predicted set) per case."""
    tp = fp = fn = 0
    for truth, pred in pairs:
        if kind:
            truth = {c for c in truth if kind_of(c) == kind}
            pred = {c for c in pred if kind_of(c) == kind}
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def bootstrap(pairs: list[tuple[set, set]], n: int = BOOTSTRAPS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(n, len(pairs)))
    vals = {"precision": [], "recall": [], "f1": []}
    for row in idx:
        m = micro([pairs[i] for i in row])
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def per_code(pairs: list[tuple[set, set]]) -> list[dict]:
    rows = []
    for code in T.codes():
        tp = sum(code in t and code in p for t, p in pairs)
        fp = sum(code not in t and code in p for t, p in pairs)
        fn = sum(code in t and code not in p for t, p in pairs)
        pr, rc, f1 = prf(tp, fp, fn)
        rows.append({"code": code, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)})
    return rows


def score(stage: str) -> dict:
    """Verify every saved response and score it. Needs only raw.jsonl and the dataset."""
    d = _stage_dir(stage)
    raw = {}
    for line in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        raw[r["custom_id"]] = r
    table = read_csv(d / "cases.csv")
    co = cohort()
    cases = co[stage].to_dict("records")

    kept_rows, dropped_rows, disagreements, results_by_mode = [], [], [], defaultdict(list)
    pairs = {m: [] for m in T.MODES}
    pairs_before = {m: [] for m in T.MODES}
    parse_fail = Counter()
    stops = Counter()
    for t in table:
        cid, mode = t["custom_id"], t["mode"]
        case = cases[int(t["position"])]
        assert case["source_url"] == t["source_url"]
        r = raw[cid]
        stops[(mode, r.get("stop_reason") or r["status"])] += 1
        items = parse_items(r.get("text"))
        if items is None:
            parse_fail[mode] += 1
            items = []
        region = (int(t["region_start"]), int(t["region_end"]))
        res = verify_guideline_quotes(case["full_text"], items, "F", region=region,
                                      mask_token=T.MASK_TOKEN)
        results_by_mode[mode].append(res)
        truth = set(case["truth"])
        pred = {k.condition_code for k in res.kept}
        before = {i["condition_code"] for i in items
                  if isinstance(i, dict) and i.get("condition_code") in set(T.codes())}
        pairs[mode].append((truth, pred))
        pairs_before[mode].append((truth, before))
        for k in res.kept:
            kept_rows.append({"custom_id": cid, "mode": mode, "code": k.condition_code,
                              "start": k.start, "end": k.end, "quote": k.text,
                              "context": k.context})
        for q in res.dropped:
            item = q.item if isinstance(q.item, dict) else {}
            dropped_rows.append({"custom_id": cid, "mode": mode, "reason": q.reason,
                                 "code": item.get("condition_code"), "quote": item.get("exact_quote"),
                                 "span": q.span})
        quotes_by_code = defaultdict(list)
        for k in res.kept:
            quotes_by_code[k.condition_code].append(k.text)
        for code in sorted(pred - truth):
            disagreements.append({"custom_id": cid, "mode": mode, "type": "false_positive",
                                  "code": code, "text": " || ".join(quotes_by_code[code])})
        for code in sorted(truth - pred):
            cited = [case["full_text"][h["start"]:h["end"]] for h in _hits(case, fallback=False)
                     if (h.get("sead4") or h["code"]) == code]
            disagreements.append({"custom_id": cid, "mode": mode, "type": "false_negative",
                                  "code": code, "text": " || ".join(cited[:3])})

    scores, per_code_rows = [], []
    for mode in T.MODES:
        counts = total_counts(results_by_mode[mode])
        for label, pp in (("after_verify", pairs[mode]), ("before_verify", pairs_before[mode])):
            m = micro(pp)
            ci = bootstrap(pp)
            scores.append({"mode": mode, "stage": label, "cases": len(pp), **m,
                           "precision_ci": ci["precision"], "recall_ci": ci["recall"], "f1_ci": ci["f1"],
                           "disq_f1": micro(pp, "disqualifier")["f1"],
                           "disq_p": micro(pp, "disqualifier")["precision"],
                           "disq_r": micro(pp, "disqualifier")["recall"],
                           "mit_f1": micro(pp, "mitigator")["f1"],
                           "mit_p": micro(pp, "mitigator")["precision"],
                           "mit_r": micro(pp, "mitigator")["recall"],
                           "kept": counts.get("kept", 0),
                           "dropped": sum(v for k, v in counts.items() if k != "kept"),
                           "parse_failures": parse_fail[mode]})
        for row in per_code(pairs[mode]):
            per_code_rows.append({"mode": mode, **row})

    drop_counts = Counter()
    for q in dropped_rows:
        drop_counts[(q["mode"], q["reason"])] += 1
    write_csv(d / "kept.csv", kept_rows, ["custom_id", "mode", "code", "start", "end", "quote", "context"])
    write_csv(d / "dropped.csv", dropped_rows, ["custom_id", "mode", "reason", "code", "quote", "span"])
    write_csv(d / "scores.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in s.items()}
                                 for s in scores])
    write_csv(d / "per_code.csv", per_code_rows)
    write_csv(d / "disagreements.csv", disagreements, ["custom_id", "mode", "type", "code", "text"])
    usage = [r["usage"] for r in raw.values() if r.get("usage")]
    tier = json.loads(_state_path(stage).read_text(encoding="utf-8")).get("tier", "batch")
    cost = {"tier": tier, "requests": len(raw),
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
            "cost": sum(C.usage_cost(u, tier) for u in usage),
            "stop_reasons": {f"{m}:{s}": n for (m, s), n in stops.items()},
            "drops": {f"{m}:{s}": n for (m, s), n in drop_counts.items()}}
    (d / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    return {"scores": scores, "cost": cost, "pairs": pairs, "pairs_before": pairs_before}


def _hits(case, fallback: bool = True) -> list[dict]:
    from src.conditions import extract_case
    return extract_case(case, fallback)


# --- test-run checks ----------------------------------------------------------------------

def check_test(log=print) -> dict:
    out = score("test")
    pred = load_prediction()
    cost = out["cost"]
    sp = pred["stages"]["test"]
    in_ratio = cost["input_tokens"] / sp["counted_input_tokens"]
    per_request = cost["cost"] / cost["requests"]
    cit = next(s for s in out["scores"] if s["mode"] == "citation" and s["stage"] == "after_verify")
    kept = sum(s["kept"] for s in out["scores"] if s["stage"] == "after_verify")
    dropped = sum(s["dropped"] for s in out["scores"] if s["stage"] == "after_verify")
    ok_stops = all(k.split(":")[1] in ("end_turn",) for k in cost["stop_reasons"])
    checks = {
        "input_within_15pct": abs(in_ratio - 1) <= 0.15,
        "input_ratio": in_ratio,
        "cost_100_requests_under_0.50": per_request * 100 <= 0.50,
        "mean_cost_per_request": per_request,
        "all_requests_ok_no_refusal_or_max_tokens": ok_stops and cost["requests"] == 10
        and sum(s["parse_failures"] for s in out["scores"]) == 0,
        "quote_survival": kept / (kept + dropped) if kept + dropped else 0.0,
        "quote_survival_ok": (kept / (kept + dropped) if kept + dropped else 0.0) >= 0.70,
        "citation_precision": cit["precision"],
        "citation_recall": cit["recall"],
        "citation_smoke_ok": cit["precision"] >= 0.5 and cit["recall"] >= 0.5,
        "manual_read_needed": "read every dropped quote and 10 kept quotes",
        "measured_output_per_request": cost["output_tokens"] / cost["requests"],
    }
    (_stage_dir("test") / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def update_prediction_with_measured(measured_output_per_request: float) -> None:
    """Replace the output estimate with the test run's measured mean."""
    pred = load_prediction()
    total_cost = 0.0
    for name, s in pred["stages"].items():
        if name == "test":
            continue
        s["output_tokens_per_request"] = round(measured_output_per_request)
        s["output_tokens"] = s["requests"] * round(measured_output_per_request)
        s["central_cost"] = C.cost(s["counted_input_tokens"], s["output_tokens"])
    pred["output_estimate_replaced_with_measured"] = round(measured_output_per_request)
    pred["total"]["central_cost_after_test"] = sum(s["central_cost"] for s in pred["stages"].values())
    (OUT / "cost_prediction.json").write_text(json.dumps(pred, indent=2), encoding="utf-8")


# --- report ------------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report() -> str:
    pred = load_prediction()
    lines = ["# Guideline F LLM pilot", ""]
    lines += ["Model: Claude Haiku 5.5 on the Batch API. Truth: the judge's citations read by `src/conditions.py`.", ""]
    lines += ["## Cost, predicted against actual", "",
              "| Stage | Requests | Predicted input | Actual input | Predicted output | Actual output | Predicted cost | Actual cost |",
              "|---|---|---|---|---|---|---|---|"]
    for stage, name in (("test", "test_run"), ("pilot", "pilot")):
        p = pred["stages"][stage]
        cpath = OUT / name / "cost.json"
        if not cpath.exists():
            continue
        c = json.loads(cpath.read_text(encoding="utf-8"))
        lines.append(f"| {stage} | {c['requests']} | {p['counted_input_tokens']:,} | {c['input_tokens']:,} "
                     f"| {p['requests'] * OUTPUT_PER_REQUEST:,} | {c['output_tokens']:,} "
                     f"| ${p['requests'] * 0 + C.cost(p['counted_input_tokens'], p['requests'] * OUTPUT_PER_REQUEST):.4f} "
                     f"| ${c['cost']:.4f} |")
    lines += ["", f"Ledger total: ${spent():.4f} of ${C.BUDGET:.2f}.", ""]
    for stage, name in (("pilot", "pilot"),):
        spath = OUT / name / "scores.csv"
        if not spath.exists():
            continue
        lines += ["## Scores", "", "| Mode | Stage | P | R | F1 | F1 interval | Disq F1 | Mit F1 | Kept | Dropped |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for s in read_csv(spath):
            lo, hi = json.loads(s["f1_ci"])
            lines.append(f"| {s['mode']} | {s['stage']} | {_pct(float(s['precision']))} | {_pct(float(s['recall']))} "
                         f"| {_pct(float(s['f1']))} | {_pct(lo)} to {_pct(hi)} | {_pct(float(s['disq_f1']))} "
                         f"| {_pct(float(s['mit_f1']))} | {s['kept']} | {s['dropped']} |")
        cost = json.loads((OUT / name / "cost.json").read_text(encoding="utf-8"))
        lines += ["", "## Drops by reason", "", "| Mode:reason | Count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(cost["drops"].items())]
        lines += ["", "## Per code", "", "| Mode | Code | Support | TP | FP | FN | P | R | F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in read_csv(OUT / name / "per_code.csv"):
            lines.append(f"| {r['mode']} | {r['code']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} "
                         f"| {r['precision']} | {r['recall']} | {r['f1']} |")
        lines.append("")
    manual = OUT / "notes_manual.md"
    if manual.exists():
        lines += [manual.read_text(encoding="utf-8")]
    text = "\n".join(lines) + "\n"
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    return text


# --- test split run --------------------------------------------------------------------------
#
# 74 uncited test F cases (citation mode) and a 100-case validation sample of the 1,668
# cited test F cases whose hits all map to the closed list (blind mode). One request per
# case. A 10-case test batch (5 + 5 from the front of each shuffled list) goes first, then a
# full batch of the other 164, so no request is paid for twice.

RUN = OUT / "run"
RUN_SAMPLE = 100
RUN_TEST_PER_GROUP = 5
RUN_MODE = {"uncited": "citation", "cited": "blind"}
RUN_BATCHES = ("test", "full")
RUN_FIELDS = ["custom_id", "stage", "position", "mode", "source_url", "case_number", "region_start",
              "region_end", "input_chars", "truth", "regime", "group", "batch"]


def run_cohort(df=None) -> dict:
    """The test-split cases for the run. `df` is the test-split Guideline F frame (tests pass a fake).

    uncited      extract_case finds no condition at all, any regime
    cited_pool   at least one hit, and every hit maps to the closed list
    cited        the first RUN_SAMPLE of the shuffled pool
    Both lists are sorted by source_url, then permuted by default_rng(SEED), as cohort() does.
    """
    if df is None:
        from src.conditions import _CASE_COLUMNS
        from src.dataset import load_cases

        df = load_cases(columns=_CASE_COLUMNS + ["split"])
        has_f = df["guidelines"].map(lambda g: g is not None and "F" in list(g))
        df = df[has_f & (df["split"] == "test")]
    unc_idx, cit_idx, truths, no_sections = [], [], {}, Counter()
    for i, row in enumerate(df.to_dict("records")):
        hits = _hits(row)
        group = None
        if not hits:
            group = "uncited"
        else:
            codes = truth_codes(row)
            if codes:
                group, truths[i] = "cited", sorted(codes)
        if group is None:
            continue
        if not T.sections(row):
            no_sections[group] += 1
            continue
        (unc_idx if group == "uncited" else cit_idx).append(i)

    def shuffled(idx, group):
        part = df.iloc[idx].copy()
        part["truth"] = [truths.get(i, []) for i in idx]
        part["group"] = group
        part = part.sort_values("source_url").reset_index(drop=True)
        order = np.random.default_rng(SEED).permutation(len(part))
        return part.iloc[order].reset_index(drop=True)

    uncited, pool = shuffled(unc_idx, "uncited"), shuffled(cit_idx, "cited")
    return {"uncited": uncited, "cited_pool": pool, "cited": pool.iloc[:RUN_SAMPLE].reset_index(drop=True),
            "no_sections": dict(no_sections)}


def run_case_list(co: dict) -> list[dict]:
    """Uncited cases then the cited sample, as records. Position is the index in this list."""
    return co["uncited"].to_dict("records") + co["cited"].to_dict("records")


def run_batch_of(position: int, n_uncited: int) -> str:
    """'test' for the first RUN_TEST_PER_GROUP cases of each group, else 'full'."""
    first = position if position < n_uncited else position - n_uncited
    return "test" if first < RUN_TEST_PER_GROUP else "full"


def run_custom_id(position: int, mode: str) -> str:
    return f"run-{position}-{mode}"


def build_run_requests(co: dict) -> tuple[list[dict], list[dict]]:
    """(requests, table rows) for all cases. One request each, in the group's mode."""
    cases = run_case_list(co)
    n_unc = len(co["uncited"])
    requests, table = [], []
    for pos, row in enumerate(cases):
        mode = RUN_MODE[row["group"]]
        text, region = T.build_input(row, mode)
        cid = run_custom_id(pos, mode)
        requests.append({"custom_id": cid, "params": T.request_params(text, model=C.MODEL)})
        table.append({"custom_id": cid, "stage": "run", "position": pos, "mode": mode,
                      "source_url": row["source_url"], "case_number": row["case_number"],
                      "region_start": region[0], "region_end": region[1], "input_chars": len(text),
                      "truth": "|".join(row["truth"]), "regime": row["guideline_regime"],
                      "group": row["group"], "batch": run_batch_of(pos, n_unc)})
    return requests, table


def _run_state(name: str) -> Path:
    return RUN / f"{name}_batch.json"


def _run_ledger_stage(name: str) -> str:
    return f"run-{name}"


def measured_output_per_request() -> tuple[float, str]:
    """Output tokens per request from the pilot's actual cost file."""
    c = json.loads((OUT / "pilot" / "cost.json").read_text(encoding="utf-8"))
    return c["output_tokens"] / c["requests"], "results/llm/pilot/cost.json, 100 requests"


def run_predict(log=print) -> dict:
    """Free. Count tokens for the run's requests and write results/llm/run/cost_prediction.json."""
    co = run_cohort()
    requests, table = build_run_requests(co)
    batch_of = {t["custom_id"]: t["batch"] for t in table}
    cl = C.make_client()
    counts = {r["custom_id"]: C.count_tokens(cl, r["params"]) for r in requests}
    out_per, source = measured_output_per_request()
    stages = {}
    running = spent()
    for name in RUN_BATCHES:
        ids = [c for c in counts if batch_of[c] == name]
        n, tin = len(ids), sum(counts[c] for c in ids)
        out_tokens = round(n * out_per)
        ceiling = ceiling_for(tin, n)
        stages[_run_ledger_stage(name)] = {
            "requests": n, "counted_input_tokens": tin,
            "output_tokens_per_request": out_per, "output_tokens": out_tokens,
            "central_cost": C.cost(tin, out_tokens), "ceiling": ceiling,
            "guard_allows": running + ceiling <= C.BUDGET + 1e-12,
            "counts": {c: counts[c] for c in ids}}
        running += ceiling
    total = {"requests": sum(s["requests"] for s in stages.values()),
             "counted_input_tokens": sum(s["counted_input_tokens"] for s in stages.values()),
             "central_cost": sum(s["central_cost"] for s in stages.values()),
             "ceiling": sum(s["ceiling"] for s in stages.values())}
    result = {"model": C.MODEL, "rates": C.RATES["batch"], "output_tokens_per_request_source": source,
              "cohort": {"uncited": len(co["uncited"]), "cited_pool": len(co["cited_pool"]),
                         "cited_sample": len(co["cited"]), "no_sections": co["no_sections"]},
              "stages": stages, "total": total, "spent_so_far": spent(), "cap": C.BUDGET,
              "ceiling_fits_under_cap": spent() + total["ceiling"] <= C.BUDGET + 1e-12,
              "proceed": total["central_cost"] <= CENTRAL_LIMIT
              and spent() + total["ceiling"] <= C.BUDGET + 1e-12}
    RUN.mkdir(parents=True, exist_ok=True)
    write_csv(RUN / "cases.csv", table, RUN_FIELDS)
    (RUN / "cost_prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_run_prediction() -> dict:
    path = RUN / "cost_prediction.json"
    if not path.exists():
        raise SystemExit("run `run-predict` first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_batch(name: str, resubmit: bool = False, log=print) -> None:
    """Paid. Submit one of the run's two batches, or resume its saved batch id."""
    if name not in RUN_BATCHES:
        raise SystemExit(f"unknown run batch {name!r}")
    if name == "full" and not _run_state("test").exists():
        raise SystemExit("run the 10-case test batch first")
    pred = load_run_prediction()
    if not pred["proceed"]:
        raise SystemExit("prediction is over its limits; not running")
    stage = _run_ledger_stage(name)
    state_path = _run_state(name)
    co = run_cohort()
    all_requests, table = build_run_requests(co)
    write_csv(RUN / "cases.csv", table, RUN_FIELDS)
    wanted = {t["custom_id"] for t in table if t["batch"] == name}
    requests = [r for r in all_requests if r["custom_id"] in wanted]
    cl = C.make_client()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    if state.get("batch_id") and not resubmit:
        log(f"resuming saved batch {state['batch_id']}")
    else:
        sp = pred["stages"][stage]
        ceiling = ceiling_for(sp["counted_input_tokens"], len(requests))
        guard(ceiling)
        batch_id = C.submit_batch(cl, requests)
        state = {"batch_id": batch_id, "tier": "batch"}
        state_path.write_text(json.dumps(state), encoding="utf-8")       # saved before polling
        ledger_append({"event": "submit", "stage": stage, "batch_id": batch_id, "tier": "batch",
                       "ceiling": ceiling})
        log(f"submitted batch {batch_id}, ceiling ${ceiling:.4f}")

    batch_id, tier = state["batch_id"], state.get("tier", "batch")
    results = state.get("results")
    if results is None:
        if tier == "batch" and C.wait_for_batch(cl, batch_id, log=log):
            results = C.collect_batch(cl, batch_id)
        else:
            if tier == "batch":
                _record_actual(stage, batch_id, C.collect_batch(cl, batch_id), "batch")
            results = _fallback_standard(cl, stage, requests, pred, log)
            tier = "standard"
            state["tier"] = tier
        state["results"] = results
        state_path.write_text(json.dumps(state), encoding="utf-8")

    primary_key = batch_id if tier == "batch" else f"{stage}-standard"
    _record_actual(stage, primary_key, results, tier)

    bad = [r for r in requests if results.get(r["custom_id"], {}).get("status") != "succeeded"
           and not state.get("retried")]
    if bad:
        ceiling = ceiling_for(sum(pred["stages"][stage]["counts"][r["custom_id"]] for r in bad), len(bad), tier)
        guard(ceiling)
        log(f"retrying {len(bad)} requests once")
        if tier == "batch":
            rid = C.submit_batch(cl, bad)
            ledger_append({"event": "submit", "stage": stage, "batch_id": rid, "tier": "batch",
                           "ceiling": ceiling, "retry": True})
            retried = C.collect_batch(cl, rid) if C.wait_for_batch(cl, rid, log=log) else {}
            _record_actual(stage, rid, retried, "batch")
        else:
            retried = C.run_standard(cl, bad)
            _record_actual(stage, f"{stage}-standard-retry", retried, "standard")
        results.update({k: v for k, v in retried.items() if v["status"] == "succeeded"})
        state["retried"] = True
    state["results"] = results
    state_path.write_text(json.dumps(state), encoding="utf-8")
    log(f"{stage}: saved {len(results)} results in {state_path}")


def _bootstrap_or_none(pairs):
    return bootstrap(pairs) if len(pairs) >= 2 else {k: [None, None] for k in ("precision", "recall", "f1")}


def score_run(d: Path | None = None, cases: list[dict] | None = None) -> dict:
    """Free. Verify the saved responses of both batches and score the cited sample.

    `d` is the run folder (default results/llm/run); `cases` are the case records in
    position order (default from run_cohort(), which loads the dataset).
    """
    d = d or RUN
    table = read_csv(d / "cases.csv")
    cases = cases if cases is not None else run_case_list(run_cohort())
    results = {}
    for name in RUN_BATCHES:
        p = d / f"{name}_batch.json"
        if p.exists():
            results.update(json.loads(p.read_text(encoding="utf-8")).get("results") or {})
    closed = set(T.codes())

    raw_rows, kept_rows, dropped_rows, found_rows = [], [], [], []
    results_by_group = {"uncited": [], "cited": []}
    parse_fail, stops = Counter(), Counter()
    cited = []          # (regime, truth, pred, before)
    code_cases = defaultdict(set)
    code_quotes = Counter()
    scored_ids = []
    for t in table:
        cid, group = t["custom_id"], t["group"]
        if cid not in results:
            continue                                   # its batch has not been run yet
        scored_ids.append(cid)
        case = cases[int(t["position"])]
        assert case["source_url"] == t["source_url"]
        r = results[cid]
        raw_rows.append({"custom_id": cid, **r})
        stops[(group, r.get("stop_reason") or r["status"])] += 1
        items = parse_items(r.get("text"))
        if items is None:
            parse_fail[group] += 1
            items = []
        region = (int(t["region_start"]), int(t["region_end"]))
        res = verify_guideline_quotes(case["full_text"], items, "F", region=region, mask_token=T.MASK_TOKEN)
        results_by_group[group].append(res)
        pred = {k.condition_code for k in res.kept}
        for k in res.kept:
            kept_rows.append({"custom_id": cid, "group": group, "regime": t["regime"], "mode": t["mode"],
                              "code": k.condition_code, "start": k.start, "end": k.end,
                              "quote": k.text, "context": k.context})
            if group == "uncited":
                code_quotes[k.condition_code] += 1
        for code in pred:
            if group == "uncited":
                code_cases[code].add(cid)
        for q in res.dropped:
            item = q.item if isinstance(q.item, dict) else {}
            dropped_rows.append({"custom_id": cid, "group": group, "regime": t["regime"],
                                 "mode": t["mode"], "reason": q.reason, "code": item.get("condition_code"),
                                 "quote": item.get("exact_quote"), "span": q.span})
        if group == "uncited":
            found_rows.append({"custom_id": cid, "source_url": t["source_url"], "regime": t["regime"],
                               "kept_quotes": len(res.kept), "codes": "|".join(sorted(pred)),
                               "dropped_quotes": len(res.dropped)})
        else:
            before = {i["condition_code"] for i in items
                      if isinstance(i, dict) and i.get("condition_code") in closed}
            cited.append((t["regime"], set(t["truth"].split("|")) if t["truth"] else set(), pred, before))

    scores = []
    for scope in ("overall", "sead4", "directive", "unknown"):
        sel = [c for c in cited if scope == "overall" or c[0] == scope]
        for label, idx in (("after_verify", 2), ("before_verify", 3)):
            pp = [(c[1], c[idx]) for c in sel]
            m = micro(pp)
            ci = _bootstrap_or_none(pp)
            scores.append({"scope": scope, "stage": label, "cases": len(pp), **m,
                           "precision_ci": ci["precision"], "recall_ci": ci["recall"], "f1_ci": ci["f1"],
                           "disq_f1": micro(pp, "disqualifier")["f1"], "mit_f1": micro(pp, "mitigator")["f1"]})
    cited_pairs = [(c[1], c[2]) for c in cited]
    per_code_rows = [{"group": "cited", "kept_quotes": "", "cases_found": "", **r} for r in per_code(cited_pairs)]
    for code in T.codes():
        per_code_rows.append({"group": "uncited", "code": code, "support": "", "tp": "", "fp": "", "fn": "",
                              "precision": "", "recall": "", "f1": "", "kept_quotes": code_quotes[code],
                              "cases_found": len(code_cases[code])})

    drops = Counter((q["group"], q["reason"]) for q in dropped_rows)
    kept_by_group = {g: total_counts(results_by_group[g]).get("kept", 0) for g in results_by_group}
    d.mkdir(parents=True, exist_ok=True)
    with (d / "raw.jsonl").open("w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r) + "\n")
    write_csv(d / "kept.csv", kept_rows, ["custom_id", "group", "regime", "mode", "code", "start", "end",
                                          "quote", "context"])
    write_csv(d / "dropped.csv", dropped_rows, ["custom_id", "group", "regime", "mode", "reason", "code",
                                                "quote", "span"])
    write_csv(d / "scores.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in s.items()}
                                 for s in scores],
              ["scope", "stage", "cases", "tp", "fp", "fn", "precision", "recall", "f1", "precision_ci",
               "recall_ci", "f1_ci", "disq_f1", "mit_f1"])
    write_csv(d / "per_code.csv", per_code_rows,
              ["group", "code", "support", "tp", "fp", "fn", "precision", "recall", "f1", "kept_quotes",
               "cases_found"])
    write_csv(d / "found_uncited.csv", found_rows,
              ["custom_id", "source_url", "regime", "kept_quotes", "codes", "dropped_quotes"])
    usage = [r["usage"] for r in raw_rows if r.get("usage")]
    tier = "batch"
    for name in RUN_BATCHES:
        p = d / f"{name}_batch.json"
        if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("tier") == "standard":
            tier = "standard"
    unc_rows = [r for r in found_rows]
    summary = {
        "requests": len(raw_rows), "tier": tier,
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "cost": sum(C.usage_cost(u, tier) for u in usage),
        "stop_reasons": {f"{g}:{s}": n for (g, s), n in stops.items()},
        "parse_failures": dict(parse_fail),
        "kept": kept_by_group,
        "drops": {f"{g}:{s}": n for (g, s), n in drops.items()},
        "uncited": {"cases": len(unc_rows),
                    "cases_with_kept_quote": sum(r["kept_quotes"] > 0 for r in unc_rows),
                    "kept_quotes": kept_by_group["uncited"],
                    "by_regime": {rg: {"cases": sum(r["regime"] == rg for r in unc_rows),
                                       "with_kept_quote": sum(r["regime"] == rg and r["kept_quotes"] > 0
                                                              for r in unc_rows)}
                                  for rg in ("sead4", "directive", "unknown")}},
        "scored_requests": len(scored_ids),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"scores": scores, "summary": summary}


RUN_HEADING = "## Test split run"


def run_report(d: Path | None = None, report_path: Path | None = None) -> str:
    """Free. Put a 'Test split run' section in REPORT.md, replacing an older one."""
    d = d or RUN
    report_path = report_path or OUT / "REPORT.md"
    pred = json.loads((d / "cost_prediction.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    coh = pred["cohort"]
    lines = [RUN_HEADING, "",
             f"The run covers the test split Guideline F cases. {coh['uncited']} cases have no citation "
             f"that the extractor can read. They go to the model in citation mode, and the report counts "
             f"what it finds. {coh['cited_sample']} cases come from the {coh['cited_pool']:,} test cases "
             "whose citations all map to the closed list. A fixed seed draws them from a list sorted by "
             "source URL. They go in blind mode, so the model cannot copy the answer, and the extractor's "
             "citations are the truth for scoring.", "",
             "### Limitations", "",
             "- The closed list holds SEAD-4 `AG ¶` codes. Directive cases cite Directive codes. For them, "
             "the truth is the `sead4` mapping that `src/conditions.py` returns, so a directive case is "
             "scored on a translated truth.",
             "- The truth is what the judge cited, not what the judge discussed. A correct quote under a "
             "code the judge did not cite counts as a false positive.",
             "- The uncited cases have no truth. The numbers below say what the model found, not how "
             "often it is right.", "",
             "### Cost", "",
             "| Item | Requests | Input tokens | Output tokens | Cost |", "|---|---|---|---|---|"]
    for name in ("run-test", "run-full"):
        s = pred["stages"][name]
        lines.append(f"| {name} predicted | {s['requests']} | {s['counted_input_tokens']:,} "
                     f"| {s['output_tokens']:,} | ${s['central_cost']:.4f} |")
    lines.append(f"| actual | {summ['requests']} | {summ['input_tokens']:,} | {summ['output_tokens']:,} "
                 f"| ${summ['cost']:.4f} |")
    lines += ["", f"Ledger total: ${spent():.4f} of ${C.BUDGET:.2f}.", "",
              "### Scores on the cited sample", "",
              "| Scope | Stage | Cases | P | R | F1 | F1 interval | Disq F1 | Mit F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in read_csv(d / "scores.csv"):
        lo, hi = json.loads(s["f1_ci"])
        interval = "too few cases" if lo is None else f"{_pct(lo)} to {_pct(hi)}"
        lines.append(f"| {s['scope']} | {s['stage']} | {s['cases']} | {_pct(float(s['precision']))} "
                     f"| {_pct(float(s['recall']))} | {_pct(float(s['f1']))} | {interval} "
                     f"| {_pct(float(s['disq_f1']))} | {_pct(float(s['mit_f1']))} |")
    u = summ["uncited"]
    lines += ["", "### Found on the uncited cases", "",
              f"The model returned at least one kept quote for {u['cases_with_kept_quote']} of "
              f"{u['cases']} cases, {u['kept_quotes']} quotes in all.", "",
              "| Regime | Cases | With a kept quote |", "|---|---|---|"]
    for rg, v in u["by_regime"].items():
        lines.append(f"| {rg} | {v['cases']} | {v['with_kept_quote']} |")
    lines += ["", "| Code | Cases | Kept quotes |", "|---|---|---|"]
    for r in read_csv(d / "per_code.csv"):
        if r["group"] == "uncited" and int(r["kept_quotes"] or 0):
            lines.append(f"| {r['code']} | {r['cases_found']} | {r['kept_quotes']} |")
    lines += ["", "### Drops by reason", "", "| Group and reason | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(summ["drops"].items())]
    lines += ["", f"Kept quotes: {summ['kept'].get('cited', 0)} on the cited sample and "
              f"{summ['kept'].get('uncited', 0)} on the uncited cases. Every kept quote passed "
              "`src/llm/verify.py`.", ""]
    section = "\n".join(lines)
    old = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    import re
    pattern = re.compile(r"\n?" + re.escape(RUN_HEADING) + r"\n.*?(?=\n## |\Z)", re.S)
    if pattern.search(old):
        text = pattern.sub(lambda _: "\n" + section, old, count=1)
    else:
        text = old.rstrip("\n") + "\n\n" + section
    report_path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    return section


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "test", "pilot", "score", "report",
                                      "run-predict", "run-test", "run-full", "run-score", "run-report"])
    ap.add_argument("which", nargs="?", choices=["test", "pilot"])
    ap.add_argument("--resubmit", action="store_true", help="submit again even if a batch id is saved")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.stage == "run-predict":
        r = run_predict()
        print(json.dumps({k: v for k, v in r.items() if k != "stages"}, indent=2))
        for n, s in r["stages"].items():
            print(n, {k: v for k, v in s.items() if k != "counts"})
    elif args.stage in ("run-test", "run-full"):
        run_batch(args.stage.split("-")[1], resubmit=args.resubmit)
    elif args.stage == "run-score":
        score_run()
    elif args.stage == "run-report":
        run_report()
    elif args.stage == "predict":
        r = predict()
        print(json.dumps({k: v for k, v in r.items() if k != "stages"}, indent=2))
        for n, s in r["stages"].items():
            print(n, {k: v for k, v in s.items() if k != "counts"})
    elif args.stage in ("test", "pilot"):
        run_stage(args.stage, resubmit=args.resubmit)
    elif args.stage == "score":
        score(args.which or "pilot")
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
