"""Tests for src/build_traceability.py: the sample, the sheet and the scoring mode.

    python tests/test_traceability.py

The draw runs once per test run on the real pool, and the second run for the
same-seed check reuses the loaded pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import build_traceability as B
from src.sead4_reference import CODES
from tests._harness import run_module

_cache: dict = {}


def pool():
    if "pool" not in _cache:
        _cache["pool"] = B.load_pool()
    return _cache["pool"]


def sample():
    if "sample" not in _cache:
        _cache["sample"] = B.build_sample(pool=pool())
    return _cache["sample"]


def test_quote_equals_full_text_slice():
    sheet, key, _ = sample()
    text = dict(zip(pool()["source_url"], pool()["full_text"]))
    assert len(sheet) == 100
    for r in sheet.to_dict("records"):
        assert text[r["source_url"]][r["start"]:r["end"]] == r["quote"], r["finding"]
        assert r["quote"].strip()


def test_context_holds_the_quote_and_lead_in_comes_before_it():
    sheet, key, _ = sample()
    text = dict(zip(pool()["source_url"], pool()["full_text"]))
    for r in sheet.to_dict("records"):
        t = text[r["source_url"]]
        c_start, c_end = B.hit_context(t, r["start"], r["end"])
        assert t[c_start:c_end] == r["context"], r["finding"]
        assert c_start <= r["start"] and r["end"] <= c_end
        if r["lead_in"]:
            assert r["lead_in"].endswith(":")
            assert t.rfind(r["lead_in"], 0, c_start) >= 0, r["finding"]
    rule = set(key.loc[key["source"] == "rule", "finding"])
    rule_rows = sheet[sheet["finding"].isin(rule)]
    assert rule_rows["context"].str.len().median() > 3 * rule_rows["quote"].str.len().median()


def test_codes_are_on_the_closed_list():
    sheet, key, _ = sample()
    assert set(sheet["code"]) <= CODES
    assert set(key["code"]) <= CODES
    for r in sheet.to_dict("records"):
        assert r["condition_text"]


def test_fifty_rule_and_fifty_llm():
    _, key, _ = sample()
    assert key["source"].value_counts().to_dict() == {"rule": 50, "llm": 50}
    assert sorted(key["finding"]) == list(range(1, 101))


def test_no_case_used_more_than_twice():
    _, key, _ = sample()
    assert key["source_url"].value_counts().max() <= B.MAX_PER_CASE


def test_same_seed_gives_the_same_draw():
    sheet, key, _ = sample()
    sheet2, key2, _ = B.build_sample(pool=pool())
    pd.testing.assert_frame_equal(sheet, sheet2)
    pd.testing.assert_frame_equal(key, key2)


def test_reviewers_get_twenty_five_rows_each():
    sheet, _, _ = sample()
    counts = sheet["reviewer"].value_counts().to_dict()
    assert counts == {"Rashed": 25, "Sam": 25, "Aryan": 25, "Brian": 25}
    assert list(sheet["reviewer"][:25].unique()) == ["Rashed"]
    assert list(sheet["reviewer"][75:].unique()) == ["Brian"]


def test_sheet_columns_and_blind_review():
    sheet, _, _ = sample()
    assert list(sheet.columns) == B.SHEET_COLUMNS
    assert (sheet["supports"] == "").all() and (sheet["notes"] == "").all()
    for col in sheet.columns:
        assert "source" not in col or col == "source_url"
    for r in sheet.to_dict("records"):
        assert r["decision"] == f'=HYPERLINK("{r["source_url"]}","{r["case_number"]}")'


def test_quotas_sum_and_floor():
    q = B.quotas({"sead4": 2235, "directive": 97, "unknown": 96}, 50)
    assert sum(q.values()) == 50
    assert q["directive"] >= 5 and q["unknown"] >= 5
    assert B.quotas({"sead4": 900}, 50) == {"sead4": 50}


def test_committed_files_match_the_draw():
    sheet, key, _ = sample()
    disk = pd.read_csv(B.KEY_PATH, keep_default_na=False)
    for col in ("finding", "source", "code", "start", "end", "source_url"):
        assert list(disk[col]) == list(key[col]), col
    disk_sheet = pd.read_csv(B.SHEET_PATH, keep_default_na=False)
    assert list(disk_sheet["finding"]) == list(sheet["finding"])
    assert list(disk_sheet["code"]) == list(sheet["code"])
    assert [q.replace("\r\n", "\n") for q in disk_sheet["quote"]] == list(sheet["quote"])


def test_wilson_interval():
    low, high = B.wilson(45, 50)
    assert abs(low - 0.7864) < 0.001 and abs(high - 0.9565) < 0.001
    assert B.wilson(0, 0)[0] != B.wilson(0, 0)[0]   # nan


def _synthetic():
    key = pd.DataFrame({
        "finding": [1, 2, 3, 4, 5, 6, 7, 8],
        "source": ["rule"] * 4 + ["llm"] * 4,
        "regime": ["sead4", "sead4", "directive", "sead4", "sead4", "sead4", "sead4", "sead4"],
    })
    filled = pd.DataFrame({
        "finding": [1, 2, 3, 4, 5, 6, 7, 8],
        "supports": ["yes", "no", "yes", "", "yes", "Yes ", "unsure", ""],
    })
    return filled, key


def test_scoring_counts_unsure_apart_and_skips_blanks():
    filled, key = _synthetic()
    t = B.score_sheet(filled, key).set_index(["source", "regime"])
    rule = t.loc[("rule", "all")]
    assert (rule["yes"], rule["no"], rule["unsure"], rule["blank"], rule["answered"]) == (2, 1, 0, 1, 3)
    assert abs(rule["pass_rate"] - 2 / 3) < 1e-9
    llm = t.loc[("llm", "all")]
    assert (llm["yes"], llm["no"], llm["unsure"], llm["blank"], llm["answered"]) == (2, 0, 1, 1, 2)
    assert llm["pass_rate"] == 1.0
    assert t.loc[("rule", "directive")]["yes"] == 1
    assert t.loc[("rule", "sead4")]["answered"] == 2


def test_scoring_rejects_other_answers():
    filled, key = _synthetic()
    filled.loc[0, "supports"] = "maybe"
    try:
        B.score_sheet(filled, key)
    except ValueError:
        return
    raise AssertionError("an answer other than yes, no or unsure was accepted")


if __name__ == "__main__":
    sys.exit(run_module(globals(), "traceability tests"))
