"""Tests for src/span_check.py, PROJECT_PLAN.md section 11 item 9.

    python tests/test_span_check.py

The demo documents must all pass. The negative tests break one span and expect
exactly that kind to fail: a quote shifted by one, a context shifted by one, a
lead-in shifted by one, a precedent range past the end of its text, and a kept
quote that is not in full_text. Two more check that a span fails even when
`schema.problems()` reports nothing, and that a problem in wording the check
does not know still fails the run.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src import span_check as S  # noqa: E402

_CACHE: dict = {}


def _demo():
    """The first demo document that has a lead-in, with the texts it needs."""
    if not _CACHE:
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(S.DEMO_DIR.glob("[0-9][0-9]_*.json"))]
        doc = next(d for d in docs if any(c["lead_in"] for c in d["conditions"]))
        urls = {doc["case"]["source_url"]} | {p["source_url"] for p in doc["precedents"]}
        _CACHE.update(doc=doc, texts=S.load_texts(urls))
    doc = copy.deepcopy(_CACHE["doc"])
    return doc, _CACHE["texts"][doc["case"]["source_url"]], _CACHE["texts"]


def _kinds(result):
    return sorted(f["kind"] for f in result["failed"])


def test_all_demo_documents_pass():
    result = S.check_demo()
    assert result["documents"] == 10
    assert result["failed"] == [] and result["other_problems"] == []
    kinds = result["kinds"]
    assert kinds["all"]["pass_rate"] == 1.0
    assert kinds["quote"]["spans"] == kinds["context"]["spans"] > 0
    assert kinds["lead_in"]["spans"] > 0 and kinds["precedent"]["spans"] > 0


def test_a_clean_document_has_no_failures():
    doc, text, texts = _demo()
    result = S.check_document(doc, text, texts)
    assert result["failed"] == [] and result["other"] == []
    assert result["total"]["quote"] == len(doc["conditions"])
    assert result["total"]["precedent"] == len(doc["precedents"])


def test_shifted_condition_start_fails_as_a_quote():
    doc, text, texts = _demo()
    doc["conditions"][0]["start"] += 1
    result = S.check_document(doc, text, texts)
    assert _kinds(result) == ["quote"]
    assert result["failed"][0]["id"] == doc["conditions"][0]["id"]


def test_shifted_context_fails_as_a_context():
    doc, text, texts = _demo()
    doc["conditions"][0]["context"]["start"] -= 1
    assert _kinds(S.check_document(doc, text, texts)) == ["context"]


def test_shifted_lead_in_fails_as_a_lead_in():
    doc, text, texts = _demo()
    item = next(c for c in doc["conditions"] if c["lead_in"])
    item["lead_in"]["start"] += 1
    result = S.check_document(doc, text, texts)
    assert _kinds(result) == ["lead_in"]
    assert result["failed"][0]["id"] == item["id"]


def test_precedent_range_past_its_text_fails():
    doc, text, texts = _demo()
    doc["precedents"][0]["end"] = len(texts[doc["precedents"][0]["source_url"]]) + 1
    assert _kinds(S.check_document(doc, text, texts)) == ["precedent"]


def test_kept_quote_missing_from_full_text_fails():
    doc, text, texts = _demo()
    url = doc["case"]["source_url"]
    good = text[100:140].replace('"', '""')
    header = "custom_id,group,regime,mode,code,start,end,quote,context\n"
    rows = [
        f'a,uncited,unknown,citation,AG ¶ 19(a),100,140,"{good}","{good}"\n',
        'b,uncited,unknown,citation,AG ¶ 19(a),100,140,"this sentence is not in the decision","x"\n',
    ]
    with tempfile.TemporaryDirectory() as tmp:
        kept, cases = Path(tmp) / "kept.csv", Path(tmp) / "cases.csv"
        kept.write_text(header + "".join(rows), encoding="utf-8", newline="\n")
        case_row = f"{doc['case']['case_number']},{url}"
        cases.write_text("custom_id,case_number,source_url\n" + "\n".join(f"{c},{case_row}" for c in "ab") + "\n",
                         encoding="utf-8")
        result = S.check_kept(kept, cases)
    assert [f["id"] for f in result["failed"]] == ["b"]
    assert result["kinds"]["quote"]["spans"] == 2 and result["kinds"]["quote"]["resolved"] == 1


def test_kept_quote_with_crlf_still_resolves():
    doc, text, texts = _demo()
    url = doc["case"]["source_url"]
    start = text.index("\n")
    end = start + 40
    quote = text[start:end].replace("\n", "\r\n")
    header = "custom_id,group,regime,mode,code,start,end,quote,context\n"
    row = 'a,uncited,unknown,citation,AG ¶ 19(a),%d,%d,"%s","x"\n' % (start, end, quote.replace('"', '""'))
    with tempfile.TemporaryDirectory() as tmp:
        kept, cases = Path(tmp) / "kept.csv", Path(tmp) / "cases.csv"
        kept.write_bytes((header + row).encode("utf-8"))
        cases.write_text(f"custom_id,case_number,source_url\na,{doc['case']['case_number']},{url}\n", encoding="utf-8")
        assert S.check_kept(kept, cases)["failed"] == []


def test_committed_kept_quotes_all_resolve():
    result = S.check_kept()
    assert result["failed"] == []
    assert result["kinds"]["quote"]["spans"] > 0


def test_direct_check_fails_a_shifted_quote_when_problems_is_silent():
    doc, text, texts = _demo()
    doc["conditions"][0]["end"] += 1
    real = S.schema.problems
    S.schema.problems = lambda *_: []
    try:
        result = S.check_document(doc, text, texts)
    finally:
        S.schema.problems = real
    assert _kinds(result) == ["quote"]


def test_unrecognised_schema_problem_is_kept_and_fails_the_run():
    doc, text, texts = _demo()
    real = S.schema.problems
    S.schema.problems = lambda *_: ["conditions[0] x: some new wording", "guidelines: something else"]
    try:
        result = S.check_document(doc, text, texts)
    finally:
        S.schema.problems = real
    assert result["failed"] == []
    assert result["other"] == ["conditions[0] x: some new wording", "guidelines: something else"]

    clean = {"spans": 1, "resolved": 1, "failed": 0, "pass_rate": 1.0}
    fake = {"demo": {"documents": 1, "kinds": {"all": clean}, "failed": [],
                     "other_problems": [{"file": "01_x.json", "problem": "guidelines: something else"}]},
            "llm": {"cases": 1, "kinds": {"all": clean}, "failed": []},
            "total": clean}
    saved = S.run, S.OUT_JSON, S.OUT_MD
    with tempfile.TemporaryDirectory() as tmp:
        S.run, S.OUT_JSON, S.OUT_MD = (lambda: fake), Path(tmp) / "s.json", Path(tmp) / "s.md"
        try:
            assert S.main([]) == 1
            fake["demo"]["other_problems"] = []
            assert S.main([]) == 0
        finally:
            S.run, S.OUT_JSON, S.OUT_MD = saved


if __name__ == "__main__":
    raise SystemExit(run_module(globals(), "span check tests"))
