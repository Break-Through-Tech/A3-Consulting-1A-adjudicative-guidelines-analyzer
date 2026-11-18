"""LLM part two: a rationale and similarity notes for the demo cases (task 34).

    python -m src.llm.notes predict     # free: demo set, retrieval, documents, token counts
    python -m src.llm.notes test        # paid: the first demo case, 1 rationale and 5 notes
    python -m src.llm.notes full        # paid: the other 9 demo cases
    python -m src.llm.notes score       # free: check every answer, write the documents

The demo set is 10 test cases from the Guideline F test-split run, 5 DENIED and
5 GRANTED, so each has verified LLM quotes as well as rule hits. For each case:

    rationale   one request. Its input is the schema document's statuses,
                recommendation and conditions, and nothing else. The answer is
                kept only if every passage it quotes is found in `full_text`
                inside the sentence of the condition it names, every number in
                it appears in the input, and it states no probability or score.
    notes       five requests, one per precedent from PrecedentRetriever.search
                over train cases, with the case's findings of fact as the query.
                The note says what the facts share and gives no score.

`predict` saves the retrieval results and the exact requests, so the paid stages
send what was counted. Spend goes in results/llm/spend.jsonl under the same
$1.00 cap as the pilot and the run. Precedents come from index v1; re-run all
four stages once hybrid search lands.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from src.llm import client as C
from src.llm import templates as T
from src.llm.pilot import OUT, _record_actual, ceiling_for, guard, ledger_append, measured_output_per_request, \
    parse_items, read_csv, spent, write_csv
from src.llm.verify import _key, verify_quotes

NOTES = OUT / "notes"
RUN = OUT / "run"
SEED = 42
PER_OUTCOME = 5                    # demo cases per actual outcome, DENIED and GRANTED
TOP_K = 5                          # precedents, so notes, per case
FACTS_CHARS = (2_000, 9_000)       # findings of fact length, so inputs stay pilot-sized
MIN_KEPT = 4                       # verified LLM quotes the case needs
MAX_TOKENS = 4000
BATCHES = ("test", "full")
RUN_DATE = "2026-11-17"            # the test-split run's full batch, results/llm/run
CENTRAL_LIMIT = 0.10
QUOTE_MIN_CHARS = 10               # a rationale may quote shorter passages than an extraction

DEMO_FIELDS = ["position", "source_url", "case_number", "outcome", "guidelines", "regime", "facts_start",
               "facts_end", "n_rule", "n_llm", "composed", "batch"]

_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")
_NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")
_CONFIDENCE = re.compile(r"%|percent|probab|confiden|\bscor(?:e|ed|es|ing)\b|\bcertain", re.I)
_CODES_IN_TEXT = re.compile(r"\bc\d+\b|\[c\d+\]|AG\s*¶|\b(?:19|20)\([a-g]\)", re.I)


# --- demo set ------------------------------------------------------------------------

def run_tables() -> tuple[list[dict], dict[str, list[dict]], dict[str, Counter]]:
    """The run's cases, its kept quotes by source_url, and its verify counts by source_url."""
    cases = read_csv(RUN / "cases.csv")
    url_of = {c["custom_id"]: c["source_url"] for c in cases}
    kept: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    for k in read_csv(RUN / "kept.csv"):
        url = url_of[k["custom_id"]]
        kept[url].append({"code": k["code"], "start": int(k["start"]), "end": int(k["end"]), "quote": k["quote"]})
        counts[url]["kept"] += 1
    for d in read_csv(RUN / "dropped.csv"):
        counts[url_of[d["custom_id"]]][d["reason"]] += 1
    return cases, dict(kept), dict(counts)


def facts_span(row) -> tuple[int, int] | None:
    try:
        start, end = int(row.get("sec_findings_of_fact_start")), int(row.get("sec_findings_of_fact_end"))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def select_demo(rows: list[dict], kept: dict[str, list[dict]]) -> list[dict]:
    """PER_OUTCOME DENIED then PER_OUTCOME GRANTED cases, in seeded order.

    Eligible: sead4, findings of fact between FACTS_CHARS long, at least MIN_KEPT
    verified quotes. Sorted by source_url, then permuted by default_rng(SEED).
    """
    ok = []
    for row in rows:
        span = facts_span(row)
        if row["guideline_regime"] != "sead4" or span is None:
            continue
        if not FACTS_CHARS[0] <= span[1] - span[0] <= FACTS_CHARS[1]:
            continue
        if len(kept.get(row["source_url"], [])) < MIN_KEPT:
            continue
        ok.append(row)
    ok.sort(key=lambda r: r["source_url"])
    order = np.random.default_rng(SEED).permutation(len(ok))
    picked = {"DENIED": [], "GRANTED": []}
    for i in order:
        row = ok[i]
        bucket = picked.get(row["outcome"])
        if bucket is not None and len(bucket) < PER_OUTCOME:
            bucket.append(row)
    if any(len(b) < PER_OUTCOME for b in picked.values()):
        raise LookupError(f"too few eligible demo cases: { {k: len(v) for k, v in picked.items()} }")
    return picked["DENIED"] + picked["GRANTED"]


def batch_of(position: int) -> str:
    return "test" if position == 0 else "full"


def load_rows(urls: set[str]) -> list[dict]:
    from src import schema
    from src.conditions import _CASE_COLUMNS
    from src.dataset import load_cases

    columns = list(dict.fromkeys(schema.CASE_COLUMNS + _CASE_COLUMNS))
    df = load_cases(columns=columns)
    return df[df["source_url"].isin(urls)].to_dict("records")


def provenance(counts: Counter, notes_date: str | None = None) -> dict:
    dates = {"extract": RUN_DATE}
    if notes_date:
        dates["notes"] = notes_date
    return {
        "llm_model": C.MODEL,
        "prompt_hashes": {"extract": T.prompt_hash(T.SYSTEM_PROMPT),
                          "rationale": T.prompt_hash(T.RATIONALE_SYSTEM_PROMPT),
                          "note": T.prompt_hash(T.NOTE_SYSTEM_PROMPT)},
        "run_dates": dates,
        "verify_counts": dict(counts),
    }


def base_document(row, kept: list[dict], counts: Counter, precedents=None, rationale=None,
                  notes_date: str | None = None) -> dict:
    from src import schema
    from src.conditions import extract_case

    doc = schema.build_document(row, schema.load_head_outputs(row["source_url"], "test"), extract_case(row),
                                kept, precedents, rationale, provenance=provenance(counts, notes_date))
    schema.validate(doc, row["full_text"])
    return doc


# --- retrieval -----------------------------------------------------------------------

def retrieve(rows: list[dict]) -> dict[str, list[dict]]:
    """source_url -> TOP_K train precedents for that case's findings of fact."""
    from src.normalize import normalize_body
    from src.retrieval import PrecedentRetriever

    retriever = PrecedentRetriever()
    out = {}
    for row in rows:
        a, b = facts_span(row)
        found = retriever.search(normalize_body(row["full_text"][a:b]), top_k=TOP_K, split="train")
        out[row["source_url"]] = [{
            "case_number": str(p["case_number"]), "source_url": p["source_url"], "score": p["score"],
            "outcome": str(p["outcome"]), "guidelines": list(p["guidelines"]),
            "label_source": str(p["label_source"]), "start": int(p["best_paragraph_start"]),
            "end": int(p["best_paragraph_end"]), "paragraph": p["best_paragraph"],
            "paragraph_score": p["best_paragraph_score"],
        } for p in found]
    return out


# --- requests ------------------------------------------------------------------------

def rationale_id(position: int) -> str:
    return f"rat-{position}"


def note_id(position: int, rank: int) -> str:
    return f"note-{position}-{rank}"


def build_requests(demo: list[dict], docs: list[dict], precedents: dict[str, list[dict]]) -> list[dict]:
    requests = []
    for pos, (row, doc) in enumerate(zip(demo, docs)):
        requests.append({"custom_id": rationale_id(pos), "batch": batch_of(pos),
                         "params": T.rationale_params(doc, model=C.MODEL, max_tokens=MAX_TOKENS)})
        a, b = facts_span(row)
        for rank, p in enumerate(precedents[row["source_url"]], start=1):
            requests.append({"custom_id": note_id(pos, rank), "batch": batch_of(pos),
                             "params": T.note_params(row["full_text"][a:b], p["case_number"], p["paragraph"],
                                                     model=C.MODEL, max_tokens=MAX_TOKENS)})
    return requests


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def predict(log=print) -> dict:
    """Free. Pick the demo set, retrieve, build the documents and requests, count tokens."""
    cases, kept, counts = run_tables()
    rows = {r["source_url"]: r for r in load_rows({c["source_url"] for c in cases})}
    run_rows = [dict(rows[c["source_url"]]) for c in cases if c["source_url"] in rows]
    demo = select_demo(run_rows, kept)
    log(f"demo set: {[r['case_number'] for r in demo]}")
    precedents = retrieve(demo)
    docs = [base_document(r, kept[r["source_url"]], counts[r["source_url"]]) for r in demo]
    requests = build_requests(demo, docs, precedents)

    cl = C.make_client()
    counted = {r["custom_id"]: C.count_tokens(cl, r["params"]) for r in requests}
    out_per, source = measured_output_per_request()
    stages, running = {}, spent()
    for name in BATCHES:
        ids = [r["custom_id"] for r in requests if r["batch"] == name]
        n, tin = len(ids), sum(counted[i] for i in ids)
        ceiling = ceiling_for(tin, n, max_tokens=MAX_TOKENS)
        stages[name] = {"requests": n, "counted_input_tokens": tin, "output_tokens_per_request": out_per,
                        "central_cost": C.cost(tin, round(n * out_per)), "ceiling": ceiling,
                        "guard_allows": running + ceiling <= C.BUDGET + 1e-12}
        running += ceiling
    total = {k: sum(s[k] for s in stages.values()) for k in ("requests", "counted_input_tokens", "central_cost",
                                                              "ceiling")}
    result = {"model": C.MODEL, "rates": C.RATES["batch"], "max_tokens": MAX_TOKENS,
              "output_tokens_per_request_source": source, "stages": stages, "total": total,
              "spent_so_far": spent(), "cap": C.BUDGET,
              "proceed": total["central_cost"] <= CENTRAL_LIMIT and spent() + total["ceiling"] <= C.BUDGET + 1e-12,
              "counts": counted}

    table = []
    for pos, (row, doc) in enumerate(zip(demo, docs)):
        a, b = facts_span(row)
        table.append({"position": pos, "source_url": row["source_url"], "case_number": row["case_number"],
                      "outcome": row["outcome"], "guidelines": "|".join(row["guidelines"]),
                      "regime": row["guideline_regime"], "facts_start": a, "facts_end": b,
                      "n_rule": sum(c["source"] == "rule" for c in doc["conditions"]),
                      "n_llm": sum(c["source"] == "llm" for c in doc["conditions"]),
                      "composed": doc["recommendation"]["composed"], "batch": batch_of(pos)})
    write_csv(NOTES / "demo.csv", table, DEMO_FIELDS)
    (NOTES / "precedents.json").write_text(json.dumps(precedents, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(NOTES / "requests.jsonl", requests)
    (NOTES / "cost_prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --- batches -------------------------------------------------------------------------

def _state(name: str) -> Path:
    return NOTES / f"{name}_batch.json"


def run_batch(name: str, resubmit: bool = False, log=print) -> None:
    """Paid. Submit one batch of the saved requests, or resume its saved batch id."""
    if name not in BATCHES:
        raise SystemExit(f"unknown batch {name!r}")
    if name == "full" and not _state("test").exists():
        raise SystemExit("run the test batch first")
    pred_path = NOTES / "cost_prediction.json"
    if not pred_path.exists():
        raise SystemExit("run `predict` first")
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    if not pred["proceed"]:
        raise SystemExit("prediction is over its limits; not running")
    requests = [{"custom_id": r["custom_id"], "params": r["params"]}
                for r in _read_jsonl(NOTES / "requests.jsonl") if r["batch"] == name]
    stage = f"notes-{name}"
    state_path = _state(name)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    cl = C.make_client()

    if state.get("batch_id") and not resubmit:
        log(f"resuming saved batch {state['batch_id']}")
    else:
        ceiling = ceiling_for(pred["stages"][name]["counted_input_tokens"], len(requests), max_tokens=MAX_TOKENS)
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
            ceiling = ceiling_for(pred["stages"][name]["counted_input_tokens"], len(requests), "standard",
                                  max_tokens=MAX_TOKENS)
            guard(ceiling)
            ledger_append({"event": "submit", "stage": stage, "batch_id": f"{stage}-standard",
                           "tier": "standard", "ceiling": ceiling})
            log(f"switching to the standard API at double rates, ceiling ${ceiling:.4f}")
            results, tier = C.run_standard(cl, requests), "standard"
            state["tier"] = tier
        state["results"] = results
        state["collected"] = date.today().isoformat()
        state_path.write_text(json.dumps(state), encoding="utf-8")

    _record_actual(stage, batch_id if tier == "batch" else f"{stage}-standard", results, tier)

    bad = [r for r in requests if results.get(r["custom_id"], {}).get("status") != "succeeded"]
    if bad and not state.get("retried"):
        counted = pred["counts"]
        ceiling = ceiling_for(sum(counted[r["custom_id"]] for r in bad), len(bad), tier, max_tokens=MAX_TOKENS)
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


# --- checks --------------------------------------------------------------------------

def parse_answer(text: str | None) -> dict | None:
    try:
        data = json.loads(text or "")
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _outside_quotes(text: str) -> str:
    return _QUOTED.sub(" ", text)


def _unsupported_numbers(text: str, source: str) -> list[str]:
    """Numbers outside quotes in `text` that are not whole numbers in `source`."""
    known = set(_NUMBER.findall(source))
    return [n for n in _NUMBER.findall(_outside_quotes(text)) if n not in known]


def _same(a: str, b: str) -> bool:
    return _key(a)[0].strip(" .,;:").casefold() == _key(b)[0].strip(" .,;:").casefold()


def check_rationale(answer: dict | None, doc: dict, full_text: str) -> tuple[dict | None, list[str]]:
    """(rationale for the document, []) when every check passes, else (None, problems)."""
    if answer is None or not isinstance(answer.get("text"), str):
        return None, ["unparsed"]
    problems: list[str] = []
    text = answer["text"].strip()
    by_id = {c["id"]: c for c in doc["conditions"]}
    issue = {g["code"] for g in T.at_issue(doc)}
    ids = [str(i) for i in answer.get("condition_ids", [])]
    codes = [str(g) for g in answer.get("guideline_codes", [])]
    quotes = answer.get("quotes", [])
    if not text:
        problems.append("empty")
    if len(text.split()) > round(T.RATIONALE_MAX_WORDS * 1.25):
        problems.append("too_long")
    if not ids or any(i not in by_id for i in ids):
        problems.append("bad_condition_ids")
    if not codes or any(g not in issue for g in codes):
        problems.append("bad_guideline_codes")
    if _CONFIDENCE.search(_outside_quotes(text)):
        problems.append("states_confidence")
    if _CODES_IN_TEXT.search(_outside_quotes(text)):
        problems.append("names_ids_or_codes")
    if _unsupported_numbers(text, T.rationale_input(doc)):
        problems.append("unsupported_number")

    listed = []
    for q in quotes:
        cid, quote = q.get("condition_id"), q.get("exact_quote")
        if cid not in by_id or not isinstance(quote, str):
            problems.append("bad_quote_entry")
            continue
        if cid not in ids:
            problems.append("quote_from_uncited_condition")
        ctx = by_id[cid]["context"]
        res = verify_quotes(full_text, [{"condition_code": cid, "exact_quote": quote}],
                            region=(ctx["start"], ctx["end"]), min_chars=QUOTE_MIN_CHARS)
        if not res.kept:
            problems.append(f"quote_{res.dropped[0].reason}")
        listed.append(quote)
    for inline in _QUOTED.findall(text):
        if not any(_same(inline, q) for q in listed):
            problems.append("unlisted_quote")
            break
    if problems:
        return None, sorted(set(problems))
    return {"text": text, "condition_ids": ids, "guideline_codes": [g["code"] for g in doc["guidelines"] if g["code"] in codes]}, []


def check_note(answer: dict | None, query_facts: str, paragraph: str) -> tuple[str | None, list[str]]:
    if answer is None or not isinstance(answer.get("note"), str):
        return None, ["unparsed"]
    note = answer["note"].strip()
    problems = []
    if not note:
        problems.append("empty")
    if len(note.split()) > round(T.NOTE_MAX_WORDS * 1.25):
        problems.append("too_long")
    if _CONFIDENCE.search(note) or re.search(r"\b(?:similarity|relevan\w*)\b", note, re.I):
        problems.append("states_score")
    if _unsupported_numbers(note, T.note_input(query_facts, "", paragraph)):
        problems.append("unsupported_number")
    return (None, problems) if problems else (note, [])


# --- score ---------------------------------------------------------------------------

def score(log=print) -> dict:
    """Free. Check the saved answers and write the demo documents, tables and summary."""
    results, collected = {}, None
    for name in BATCHES:
        if _state(name).exists():
            state = json.loads(_state(name).read_text(encoding="utf-8"))
            results.update(state.get("results") or {})
            collected = state.get("collected", collected)
    demo_table = read_csv(NOTES / "demo.csv")
    precedents = json.loads((NOTES / "precedents.json").read_text(encoding="utf-8"))
    _, kept, counts = run_tables()
    rows = {r["source_url"]: r for r in load_rows({d["source_url"] for d in demo_table})}

    rat_rows, note_rows, raw = [], [], []
    tally = Counter()
    docs_dir = NOTES / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for d in demo_table:
        pos, url = int(d["position"]), d["source_url"]
        if not any(k.startswith(f"rat-{pos}") or k.startswith(f"note-{pos}-") for k in results):
            continue
        row = rows[url]
        a, b = facts_span(row)
        facts = row["full_text"][a:b]
        base = base_document(row, kept[url], counts[url])

        res = results.get(rationale_id(pos), {})
        raw.append({"custom_id": rationale_id(pos), **res})
        answer = parse_answer(res.get("text")) if res.get("status") == "succeeded" else None
        if answer is not None and res.get("stop_reason") != "end_turn":
            answer = None
        rationale, problems = check_rationale(answer, base, row["full_text"])
        tally["rationale_kept" if rationale else "rationale_dropped"] += 1
        rat_rows.append({"position": pos, "case_number": d["case_number"], "composed": d["composed"],
                         "outcome": d["outcome"], "kept": bool(rationale), "problems": "|".join(problems),
                         "words": len((answer or {}).get("text", "").split()),
                         "condition_ids": "|".join((answer or {}).get("condition_ids", [])),
                         "guideline_codes": "|".join((answer or {}).get("guideline_codes", [])),
                         "n_quotes": len((answer or {}).get("quotes", [])),
                         "text": (answer or {}).get("text", "")})

        prec_out = []
        for rank, p in enumerate(precedents[url], start=1):
            res = results.get(note_id(pos, rank), {})
            raw.append({"custom_id": note_id(pos, rank), **res})
            answer = parse_answer(res.get("text")) if res.get("status") == "succeeded" else None
            if answer is not None and res.get("stop_reason") != "end_turn":
                answer = None
            note, problems = check_note(answer, facts, p["paragraph"])
            tally["note_kept" if note else "note_dropped"] += 1
            note_rows.append({"position": pos, "case_number": d["case_number"], "rank": rank,
                              "precedent": p["case_number"], "precedent_url": p["source_url"],
                              "retrieval_score": p["score"], "kept": bool(note), "problems": "|".join(problems),
                              "words": len((answer or {}).get("note", "").split()),
                              "note": (answer or {}).get("note", ""), "paragraph": p["paragraph"]})
            prec_out.append({**{k: p[k] for k in ("case_number", "source_url", "score", "outcome", "guidelines",
                                                    "label_source", "start", "end")}, "note": note})

        doc = base_document(row, kept[url], counts[url], prec_out, rationale, notes_date=collected)
        path = docs_dir / f"{pos:02d}_{d['case_number']}.json"
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_jsonl(NOTES / "raw.jsonl", raw)
    write_csv(NOTES / "rationales.csv", rat_rows)
    write_csv(NOTES / "notes.csv", note_rows)
    ok = [r for r in raw if r.get("status") == "succeeded"]
    usage = [r["usage"] for r in ok if r.get("usage")]
    summary = {
        "requests": len(raw), "succeeded": len(ok),
        "end_turn": sum(r.get("stop_reason") == "end_turn" for r in ok),
        **dict(tally),
        "rationale_problems": dict(Counter(p for r in rat_rows for p in r["problems"].split("|") if p)),
        "note_problems": dict(Counter(p for r in note_rows for p in r["problems"].split("|") if p)),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "cost": sum(C.usage_cost(u) for u in usage),
        "spent_total": spent(),
    }
    (NOTES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "test", "full", "score"])
    ap.add_argument("--resubmit", action="store_true", help="submit again even if a batch id is saved")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.stage == "predict":
        r = predict()
        print(json.dumps({k: v for k, v in r.items() if k != "counts"}, indent=2))
    elif args.stage in BATCHES:
        run_batch(args.stage, resubmit=args.resubmit)
    else:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
