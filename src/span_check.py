"""Check that every span the app displays resolves exactly in full_text.

    python -m src.span_check

PROJECT_PLAN.md section 11, item 9. A span resolves when `full_text[start:end]`
equals the text stored with it. Two sources are checked:

    demo      the 10 documents in results/demo/. The kinds are condition
              quote, context and lead_in, each compared with full_text
              directly. `schema.problems()` also runs, and any other problem
              it reports fails the run. A precedent paragraph stores offsets
              only, so it is checked for being a non-empty range inside the
              precedent case's own full_text.
    llm       the quotes in results/llm/run/kept.csv against each case's
              full_text, joined on source_url through cases.csv.

Writes results/span_check.json and results/span_check.md. Exit code 1 when any
span fails or a demo document has any other schema problem.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config, dataset, schema

DEMO_DIR = config.REPO_ROOT / "results" / "demo"
KEPT_PATH = config.REPO_ROOT / "results" / "llm" / "run" / "kept.csv"
CASES_PATH = config.REPO_ROOT / "results" / "llm" / "run" / "cases.csv"
OUT_JSON = config.REPO_ROOT / "results" / "span_check.json"
OUT_MD = config.REPO_ROOT / "results" / "span_check.md"

DEMO_KINDS = ("quote", "context", "lead_in", "precedent")
_PROBLEM = re.compile(r"^conditions\[(\d+)\] (\S+): (.*)$")


def load_texts(urls) -> dict[str, str]:
    """full_text for just these source_urls; no other text is read into memory."""
    wanted = sorted({u for u in urls if u})
    texts: dict[str, str] = {}
    if not wanted:
        return texts
    for path in dataset.shard_files():
        table = pq.read_table(path, columns=["source_url", "full_text"],
                              filters=[("source_url", "in", wanted)])
        texts.update(zip(table.column("source_url").to_pylist(), table.column("full_text").to_pylist()))
    return texts


def _kind_of_problem(message: str) -> str | None:
    """The span kind a `schema.problems()` line is about, or None for any other problem."""
    if "fall outside full_text" in message or "is not the quote" in message:
        return "quote"
    if message.startswith("context text is not full_text"):
        return "context"
    if message.startswith("lead_in") and ("text is not full_text" in message or "must end before" in message):
        return "lead_in"
    return None


def check_document(doc: dict, full_text: str, precedent_texts: dict[str, str] | None = None) -> dict:
    """Count and test the spans of one document.

    Returns {"total": {kind: n}, "failed": [{kind, id, start, end, detail}], "other": [str]}.
    `other` holds problems() lines that are not about a span resolving.
    """
    precedent_texts = precedent_texts or {}
    total = {kind: 0 for kind in DEMO_KINDS}
    failed: list[dict] = []
    conditions = doc["conditions"]
    for item in conditions:
        spans = [("quote", item["start"], item["end"], item["quote"]),
                 ("context", item["context"]["start"], item["context"]["end"], item["context"]["text"])]
        if item["lead_in"] is not None:
            spans.append(("lead_in", item["lead_in"]["start"], item["lead_in"]["end"], item["lead_in"]["text"]))
        for kind, start, end, text in spans:
            total[kind] += 1
            if not 0 <= start < end <= len(full_text):
                detail = f"offsets fall outside full_text of length {len(full_text)}"
            elif full_text[start:end] != text:
                detail = "full_text[start:end] is not the stored text"
            else:
                continue
            failed.append({"kind": kind, "id": item["id"], "start": start, "end": end, "detail": detail})

    # The direct check above decides every span. schema.problems() runs too, and any line it gives
    # that is not about one of those spans is kept as `other`, which also fails the run.
    other: list[str] = []
    for message in schema.problems(doc, full_text):
        match = _PROBLEM.match(message)
        if match is None or _kind_of_problem(match.group(3)) is None:
            other.append(message)

    for item in doc["precedents"]:
        total["precedent"] += 1
        text = precedent_texts.get(item["source_url"])
        if text is None:
            detail = "precedent case text not found"
        elif not 0 <= item["start"] < item["end"] <= len(text):
            detail = f"range is not inside the precedent text of length {len(text)}"
        else:
            continue
        failed.append({"kind": "precedent", "id": item["id"], "start": item["start"], "end": item["end"],
                       "detail": detail})
    return {"total": total, "failed": failed, "other": other}


def _tally(kinds, total: dict, failed: list[dict]) -> dict:
    out = {}
    for kind in kinds:
        n = total.get(kind, 0)
        bad = sum(1 for f in failed if f["kind"] == kind)
        out[kind] = {"spans": n, "resolved": n - bad, "failed": bad,
                     "pass_rate": (n - bad) / n if n else None}
    n = sum(total.get(k, 0) for k in kinds)
    bad = len(failed)
    out["all"] = {"spans": n, "resolved": n - bad, "failed": bad, "pass_rate": (n - bad) / n if n else None}
    return out


def check_demo(demo_dir: Path = DEMO_DIR) -> dict:
    docs = [(p.name, json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(demo_dir.glob("[0-9][0-9]_*.json"))]
    urls = {d["case"]["source_url"] for _, d in docs} | {p["source_url"] for _, d in docs for p in d["precedents"]}
    texts = load_texts(urls)
    total = {kind: 0 for kind in DEMO_KINDS}
    failed: list[dict] = []
    other: list[dict] = []
    for name, doc in docs:
        url = doc["case"]["source_url"]
        if url not in texts:
            failed.append({"kind": "quote", "case": doc["case"]["case_number"], "file": name, "id": "",
                           "start": -1, "end": -1, "detail": "case text not found"})
            continue
        result = check_document(doc, texts[url], texts)
        for kind, n in result["total"].items():
            total[kind] += n
        for f in result["failed"]:
            failed.append({"case": doc["case"]["case_number"], "file": name, **f})
        other += [{"file": name, "problem": m} for m in result["other"]]
    return {"documents": len(docs), "kinds": _tally(DEMO_KINDS, total, failed), "failed": failed,
            "other_problems": other}


def check_kept(kept_path: Path = KEPT_PATH, cases_path: Path = CASES_PATH) -> dict:
    kept = pd.read_csv(kept_path, keep_default_na=False)
    cases = pd.read_csv(cases_path, keep_default_na=False, usecols=["custom_id", "source_url", "case_number"])
    kept["quote"] = kept["quote"].str.replace("\r\n", "\n", regex=False)
    kept = kept.merge(cases, on="custom_id", how="left", validate="m:1")
    texts = load_texts(kept["source_url"].dropna())
    failed: list[dict] = []
    for r in kept.to_dict("records"):
        text = texts.get(r["source_url"])
        start, end = int(r["start"]), int(r["end"])
        if text is None:
            detail = "case text not found"
        elif not 0 <= start < end <= len(text):
            detail = f"offsets fall outside full_text of length {len(text)}"
        elif text[start:end] != r["quote"]:
            detail = "full_text[start:end] is not the quote"
        else:
            continue
        failed.append({"kind": "quote", "case": r["case_number"], "id": r["custom_id"], "code": r["code"],
                       "start": start, "end": end, "detail": detail})
    return {"cases": int(kept["source_url"].nunique()), "kinds": _tally(("quote",), {"quote": len(kept)}, failed),
            "failed": failed}


def _pct(rate) -> str:
    return "n/a" if rate is None else f"{100 * rate:.2f}%"


def report(result: dict) -> str:
    lines = ["# Span check", "",
             "PROJECT_PLAN.md section 11, item 9: every span the app displays resolves exactly in `full_text`.",
             "A span resolves when `full_text[start:end]` equals its stored text. A precedent paragraph stores offsets",
             "only, so it is checked as a non-empty range inside the precedent's own text.", ""]
    for name, label in (("demo", "Demo documents (results/demo)"),
                        ("llm", "LLM kept quotes (results/llm/run/kept.csv)")):
        part = result[name]
        extent = f"{part['documents']} documents" if name == "demo" else f"{part['cases']} cases"
        lines += [f"## {label}", "", f"{extent}.", "",
                  "| kind | spans | resolved | failed | pass rate |", "|---|---|---|---|---|"]
        for kind, c in part["kinds"].items():
            lines.append(f"| {kind} | {c['spans']} | {c['resolved']} | {c['failed']} | {_pct(c['pass_rate'])} |")
        lines.append("")
        if part["failed"]:
            lines += ["Failing spans:", ""]
            lines += [f"- case {f.get('case', '')} {f['kind']} {f.get('id', '')} offsets {f['start']}:{f['end']}: "
                      f"{f['detail']}" for f in part["failed"]]
            lines.append("")
        if part.get("other_problems"):
            lines += ["Other schema problems:", ""] + [f"- {o['file']}: {o['problem']}"
                                                       for o in part["other_problems"]] + [""]
    total = result["total"]
    lines += ["## Total", "",
              f"{total['resolved']} of {total['spans']} spans resolve ({_pct(total['pass_rate'])}).", ""]
    return "\n".join(lines)


def run() -> dict:
    demo, llm = check_demo(), check_kept()
    spans = demo["kinds"]["all"]["spans"] + llm["kinds"]["all"]["spans"]
    bad = demo["kinds"]["all"]["failed"] + llm["kinds"]["all"]["failed"]
    return {"demo": demo, "llm": llm,
            "total": {"spans": spans, "resolved": spans - bad, "failed": bad,
                      "pass_rate": (spans - bad) / spans if spans else None}}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    result = run()
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(report(result), encoding="utf-8", newline="\n")
    print(report(result))
    return 1 if result["total"]["failed"] or result["demo"]["other_problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
