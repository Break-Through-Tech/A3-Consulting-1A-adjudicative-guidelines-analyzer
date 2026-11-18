"""Tests for the test split run in src/llm/pilot.py. No network calls.

    python tests/test_run.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src.llm import pilot as P  # noqa: E402
from src.llm import templates as T  # noqa: E402

_CO = {}


def co() -> dict:
    if not _CO:
        _CO.update(P.run_cohort())
    return _CO


def test_uncited_cohort_counts():
    unc = co()["uncited"]
    assert len(unc) == 74, len(unc)
    assert dict(Counter(unc["guideline_regime"])) == {"sead4": 32, "unknown": 34, "directive": 8}
    assert co()["no_sections"] == {}


def test_cited_pool_count_and_sample_size():
    assert len(co()["cited_pool"]) == 1668
    assert len(co()["cited"]) == 100
    assert all(len(t) > 0 for t in co()["cited"]["truth"])


def test_sample_is_fixed_by_the_seed_and_the_sort():
    import numpy as np

    pool = co()["cited_pool"].sort_values("source_url")["source_url"].tolist()
    order = np.random.default_rng(P.SEED).permutation(len(pool))
    expect = [pool[i] for i in order[:100]]
    assert co()["cited"]["source_url"].tolist() == expect
    assert P.run_cohort()["cited"]["source_url"].tolist() == expect


def test_groups_do_not_overlap_and_are_sorted_free_of_duplicates():
    unc, cit = set(co()["uncited"]["source_url"]), set(co()["cited_pool"]["source_url"])
    assert not unc & cit
    assert len(unc) == 74 and len(cit) == 1668


def test_modes_by_group():
    assert P.RUN_MODE == {"uncited": "citation", "cited": "blind"}


def _table() -> list[dict]:
    return P.read_csv(P.RUN / "cases.csv")


def test_174_unique_ids_that_do_not_overlap_the_pilot():
    table = _table()
    ids = [t["custom_id"] for t in table]
    assert len(ids) == 174 and len(set(ids)) == 174
    old = set()
    for p in (P.OUT / "pilot" / "cases.csv", P.OUT / "test_run" / "cases.csv"):
        old |= {r["custom_id"] for r in P.read_csv(p)}
    assert not set(ids) & old
    assert all(i.startswith("run-") for i in ids)
    for t in table:
        assert t["mode"] == P.RUN_MODE[t["group"]]
        assert t["custom_id"] == P.run_custom_id(int(t["position"]), t["mode"])
    assert Counter(t["group"] for t in table) == {"uncited": 74, "cited": 100}


def test_ten_case_batch_and_the_rest_do_not_overlap():
    table = _table()
    test = [t for t in table if t["batch"] == "test"]
    full = [t for t in table if t["batch"] == "full"]
    assert (len(test), len(full)) == (10, 164)
    assert not {t["custom_id"] for t in test} & {t["custom_id"] for t in full}
    assert Counter(t["group"] for t in test) == {"uncited": 5, "cited": 5}
    assert [int(t["position"]) for t in test] == [0, 1, 2, 3, 4, 74, 75, 76, 77, 78]
    assert P.run_batch_of(4, 74) == "test" and P.run_batch_of(5, 74) == "full"
    assert P.run_batch_of(78, 74) == "test" and P.run_batch_of(79, 74) == "full"


def test_case_table_matches_the_cohort():
    c = co()
    cases = P.run_case_list(c)
    for t in _table():
        row = cases[int(t["position"])]
        assert row["source_url"] == t["source_url"]
        assert row["group"] == t["group"] and row["guideline_regime"] == t["regime"]
        assert t["truth"] == "|".join(row["truth"])
        if t["group"] == "uncited":
            assert t["truth"] == ""


def test_regime_split_of_the_sample_and_test_batch_is_recorded():
    table = _table()
    sample = Counter(t["regime"] for t in table if t["group"] == "cited")
    assert sum(sample.values()) == 100
    test = Counter(t["regime"] for t in table if t["group"] == "cited" and t["batch"] == "test")
    assert sum(test.values()) == 5


def test_pilot_check_values_still_match():
    pc = P.cohort()
    pilot, test = pc["pilot"], pc["test"]
    cp = [T.build_input(r, "citation", collapse=False)[0] for r in pilot.to_dict("records")]
    ct = [T.build_input(r, "citation", collapse=False)[0] for r in test.to_dict("records")]
    got = {"eligible": pc["eligible"], "pilot_chars": sum(map(len, cp)),
           "pilot_codes": sum(len(t) for t in pilot["truth"]), "test_chars": [len(t) for t in ct],
           "test_codes": sum(len(t) for t in test["truth"])}
    assert got == P.CHECK_VALUES, got
    assert P.STAGES == {"test": (50, 55), "pilot": (0, 50)}


def _with_ledger(fn):
    old = P.LEDGER
    with tempfile.TemporaryDirectory() as d:
        P.LEDGER = Path(d) / "spend.jsonl"
        try:
            fn()
        finally:
            P.LEDGER = old


def test_guard_refuses_a_run_ceiling_that_passes_the_cap():
    pred = json.loads((P.RUN / "cost_prediction.json").read_text(encoding="utf-8"))
    full = pred["stages"]["run-full"]["ceiling"]

    def run():
        P.ledger_append({"event": "submit", "batch_id": "b1", "ceiling": 1.0 - full + 0.01})
        try:
            P.guard(full)
        except SystemExit:
            return
        raise AssertionError("expected a refusal")

    _with_ledger(run)
    _with_ledger(lambda: P.guard(full))          # an empty ledger lets it through


def test_prediction_file_is_consistent():
    pred = json.loads((P.RUN / "cost_prediction.json").read_text(encoding="utf-8"))
    assert pred["stages"]["run-test"]["requests"] == 10 and pred["stages"]["run-full"]["requests"] == 164
    assert len(pred["stages"]["run-test"]["counts"]) + len(pred["stages"]["run-full"]["counts"]) == 174
    assert pred["total"]["ceiling"] == sum(s["ceiling"] for s in pred["stages"].values())


# --- run-score on synthetic results -----------------------------------------------------------

TEXT = ("Applicant owes the lender $12,345 on a car loan that went unpaid for two years. "
        "He has not made a payment since 2021 and gave no reason for the delay.")
REAL = "He has not made a payment since 2021"
REWORDED = "He never paid anything on that loan after 2021"
KEPT_CODE, DROP_CODE = "AG ¶ 19(a)", "AG ¶ 19(c)"


def _synthetic(d: Path) -> list[dict]:
    cases = [
        {"source_url": "u0", "full_text": TEXT, "group": "uncited", "guideline_regime": "sead4", "truth": []},
        {"source_url": "u1", "full_text": TEXT, "group": "uncited", "guideline_regime": "directive", "truth": []},
        {"source_url": "c0", "full_text": TEXT, "group": "cited", "guideline_regime": "sead4",
         "truth": [KEPT_CODE]},
        {"source_url": "c1", "full_text": TEXT, "group": "cited", "guideline_regime": "directive",
         "truth": [KEPT_CODE]},
    ]
    n_unc = 2
    table = []
    for pos, c in enumerate(cases):
        mode = P.RUN_MODE[c["group"]]
        table.append({"custom_id": P.run_custom_id(pos, mode), "stage": "run", "position": pos, "mode": mode,
                      "source_url": c["source_url"], "case_number": f"n{pos}", "region_start": 0,
                      "region_end": len(TEXT), "input_chars": len(TEXT), "truth": "|".join(c["truth"]),
                      "regime": c["guideline_regime"], "group": c["group"],
                      "batch": "test" if pos in (0, 2) else "full"})
    P.write_csv(d / "cases.csv", table, P.RUN_FIELDS)

    def ok(items, tin=100, tout=50):
        return {"status": "succeeded", "usage": {"input_tokens": tin, "output_tokens": tout},
                "stop_reason": "end_turn", "text": json.dumps({"items": items}), "error": None}
    item = {"condition_code": KEPT_CODE, "exact_quote": REAL}
    bad = {"condition_code": DROP_CODE, "exact_quote": REWORDED}
    by_id = {t["custom_id"]: t for t in table}
    first = {table[0]["custom_id"]: ok([item, bad]), table[2]["custom_id"]: ok([item])}
    second = {table[1]["custom_id"]: ok([]), table[3]["custom_id"]: ok([item, bad])}
    assert set(first) | set(second) == set(by_id)
    (d / "test_batch.json").write_text(json.dumps({"batch_id": "b1", "tier": "batch", "results": first}))
    (d / "full_batch.json").write_text(json.dumps({"batch_id": "b2", "tier": "batch", "results": second}))
    return cases


def test_run_score_keeps_real_quotes_drops_reworded_ones_and_writes_every_file():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = _synthetic(d)
        out = P.score_run(d, cases)
        for name in ("raw.jsonl", "kept.csv", "dropped.csv", "per_code.csv", "scores.csv",
                     "found_uncited.csv", "summary.json"):
            assert (d / name).exists(), name
        assert len((d / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 4
        kept = P.read_csv(d / "kept.csv")
        assert len(kept) == 3 and {k["code"] for k in kept} == {KEPT_CODE}
        for k in kept:
            assert k["quote"].replace("\r\n", "\n") == REAL or REAL in k["quote"]
            assert TEXT[int(k["start"]):int(k["end"])] == k["quote"]
        dropped = P.read_csv(d / "dropped.csv")
        assert len(dropped) == 2 and {r["reason"] for r in dropped} == {"not_found"}
        assert {r["group"] for r in dropped} == {"uncited", "cited"}
        found = {r["source_url"]: r for r in P.read_csv(d / "found_uncited.csv")}
        assert found["u0"]["kept_quotes"] == "1" and found["u1"]["kept_quotes"] == "0"
        s = out["summary"]
        assert s["uncited"]["cases"] == 2 and s["uncited"]["cases_with_kept_quote"] == 1
        assert s["drops"] == {"uncited:not_found": 1, "cited:not_found": 1}
        assert s["kept"] == {"uncited": 1, "cited": 2}
        scores = {(r["scope"], r["stage"]): r for r in P.read_csv(d / "scores.csv")}
        overall = scores[("overall", "after_verify")]
        assert overall["cases"] == "2" and float(overall["f1"]) == 1.0
        assert scores[("directive", "after_verify")]["cases"] == "1"
        assert json.loads(scores[("directive", "after_verify")]["f1_ci"]) == [None, None]
        assert scores[("unknown", "after_verify")]["cases"] == "0"
        code = {r["code"]: r for r in P.read_csv(d / "per_code.csv") if r["group"] == "uncited"}
        assert code[KEPT_CODE]["kept_quotes"] == "1" and code[KEPT_CODE]["cases_found"] == "1"


def test_run_score_with_only_the_test_batch_scores_what_is_there():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = _synthetic(d)
        (d / "full_batch.json").unlink()
        out = P.score_run(d, cases)
        assert out["summary"]["requests"] == 2


def test_run_report_adds_one_section_and_replaces_it_on_rerun():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = _synthetic(d)
        pred = json.loads((P.RUN / "cost_prediction.json").read_text(encoding="utf-8"))
        (d / "cost_prediction.json").write_text(json.dumps(pred))
        P.score_run(d, cases)
        rp = d / "REPORT.md"
        rp.write_text("# Guideline F LLM pilot\n\n## Scores\n\nx\n\n## Notes\n\nlast\n", encoding="utf-8")
        P.run_report(d, rp)
        P.run_report(d, rp)
        text = rp.read_text(encoding="utf-8")
        assert text.count("## Test split run") == 1
        assert "## Scores" in text and "## Notes" in text and "### Limitations" in text
        assert "—" not in text.split("## Test split run")[1]


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "run tests"))
