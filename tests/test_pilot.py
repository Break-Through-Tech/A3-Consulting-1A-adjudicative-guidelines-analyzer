"""Tests for src/llm/client.py and src/llm/pilot.py. No network calls.

    python tests/test_pilot.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src.llm import client as C  # noqa: E402
from src.llm import pilot as P  # noqa: E402

A, B, C20 = "AG ¶ 19(a)", "AG ¶ 20(b)", "AG ¶ 20(d)"


def approx(x, y, tol=1e-9):
    assert abs(x - y) <= tol, (x, y)


def test_batch_cost_arithmetic():
    approx(C.cost(1_000_000, 0), 0.05)
    approx(C.cost(0, 1_000_000), 0.25)
    approx(C.cost(448_000, 253_000), 0.02240 + 0.06325)
    approx(C.cost(1_000_000, 1_000_000, "standard"), 0.60)
    approx(C.usage_cost({"input_tokens": 2_000_000, "output_tokens": 400_000}), 0.10 + 0.10)


def test_plan_prediction_matches_the_plan_table():
    approx(C.cost(410_000, 100 * 2300), 0.0205 + 0.0575)         # pilot, about $0.08
    approx(P.ceiling_for(410_000, 100), 0.0205 + 100 * 8000 * 0.25 / 1e6)   # about $0.22


def test_micro_prf_hand_built():
    pairs = [({A, B}, {A}), ({C20}, {C20, B}), ({A}, set())]
    m = P.micro(pairs)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    approx(m["precision"], 2 / 3)
    approx(m["recall"], 0.5)
    approx(m["f1"], 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_micro_split_by_kind():
    pairs = [({A, B}, {A, C20})]
    d = P.micro(pairs, "disqualifier")
    m = P.micro(pairs, "mitigator")
    assert (d["tp"], d["fp"], d["fn"]) == (1, 0, 0)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_empty_cases_do_not_divide_by_zero():
    m = P.micro([(set(), set())])
    assert m["precision"] == m["recall"] == m["f1"] == 0.0


def test_per_code_table():
    rows = {r["code"]: r for r in P.per_code([({A}, {A}), ({A, B}, {B}), (set(), {B})])}
    assert (rows[A]["tp"], rows[A]["fn"], rows[A]["fp"]) == (1, 1, 0)
    assert (rows[B]["tp"], rows[B]["fp"], rows[B]["fn"]) == (1, 1, 0)
    assert len(rows) == 16


def test_bootstrap_is_fixed_and_brackets_the_estimate():
    pairs = [({A}, {A})] * 6 + [({A}, set())] * 2 + [(set(), {B})] * 2
    one, two = P.bootstrap(pairs, n=300), P.bootstrap(pairs, n=300)
    assert one == two
    lo, hi = one["f1"]
    assert 0 <= lo <= P.micro(pairs)["f1"] <= hi <= 1


def test_kind_of():
    assert P.kind_of(A) == "disqualifier" and P.kind_of(B) == "mitigator"


def test_parse_items():
    assert P.parse_items('{"items": []}') == []
    assert P.parse_items('{"items": [{"condition_code": "x", "exact_quote": "y"}]}')[0]["exact_quote"] == "y"
    for bad in ("", None, "not json", '{"other": 1}', '{"items": "x"}', "[]"):
        assert P.parse_items(bad) is None, bad


def _with_ledger(fn):
    old = P.LEDGER
    with tempfile.TemporaryDirectory() as d:
        P.LEDGER = Path(d) / "spend.jsonl"
        try:
            fn()
        finally:
            P.LEDGER = old


def test_ledger_refuses_a_submit_that_would_pass_the_cap():
    def run():
        P.ledger_append({"event": "submit", "batch_id": "b1", "ceiling": 0.60})
        approx(P.spent(), 0.60)
        P.guard(0.39)
        try:
            P.guard(0.41)
        except SystemExit:
            return
        raise AssertionError("expected a refusal")
    _with_ledger(run)


def test_ledger_actual_replaces_ceiling_and_retry_adds():
    def run():
        P.ledger_append({"event": "submit", "batch_id": "b1", "ceiling": 0.60})
        P.ledger_append({"event": "actual", "batch_id": "b1", "cost": 0.01})
        P.ledger_append({"event": "submit", "batch_id": "b2", "ceiling": 0.30})
        approx(P.spent(), 0.31)
        P.guard(0.69)
    _with_ledger(run)


def test_record_actual_writes_once():
    def run():
        res = {"a": {"usage": {"input_tokens": 1_000_000, "output_tokens": 0}},
               "b": {"usage": None}}
        P._record_actual("test", "b1", res, "batch")
        P._record_actual("test", "b1", res, "batch")
        rows = [r for r in P.ledger_rows() if r["event"] == "actual"]
        assert len(rows) == 1 and abs(rows[0]["cost"] - 0.05) < 1e-12
    _with_ledger(run)


def test_custom_ids_follow_stage_position_mode():
    assert P.custom_id("pilot", 7, "blind") == "pilot-7-blind"
    assert P.custom_id("test", 0, "citation") == "test-0-citation"


def test_stage_slices_do_not_overlap():
    (a0, a1), (b0, b1) = P.STAGES["pilot"], P.STAGES["test"]
    assert (a0, a1) == (0, 50) and (b0, b1) == (50, 55)


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "pilot tests"))
