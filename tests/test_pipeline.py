"""End-to-end tests for src/pipeline.py, PROJECT_PLAN.md section 11 item 10. No network calls.

    python tests/test_pipeline.py

One held-out test case goes in as text and a JSON document comes out. Every
offset in it must quote the text it claims. The case is the first test-split
case of the LLM run, by source_url, that is not a demo case, so the run uses the
published ModernBERT heads, the cached LLM quotes and live retrieval. The TF-IDF
test needs runs/pipeline/tfidf_heads.joblib from `python -m src.pipeline --fit-fallback`.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the embedding model is read from the local cache
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src import config, pipeline, schema  # noqa: E402
from src.heads import infer  # noqa: E402
from src.llm import notes  # noqa: E402

MISSING_CHECKPOINT = str(Path(__file__).resolve().parent / "no_such_checkpoint.pt")


@contextlib.contextmanager
def _checkpoint(path: str):
    """Point MODERNBERT_CHECKPOINT at `path` for the block."""
    old = os.environ.get(infer.CHECKPOINT_ENV)
    os.environ[infer.CHECKPOINT_ENV] = path
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(infer.CHECKPOINT_ENV, None)
        else:
            os.environ[infer.CHECKPOINT_ENV] = old


@lru_cache(maxsize=1)
def _case() -> dict:
    """The held-out test case: its dataset row, as load_cases returns it."""
    demo = {d["source_url"] for d in notes.read_csv(notes.NOTES / "demo.csv")}
    urls = sorted({c["source_url"] for c in notes.read_csv(notes.RUN / "cases.csv")} - demo)
    from src.dataset import load_cases

    columns = list(dict.fromkeys(["case_number", "source_url", "split", "full_text", "text_features", "guidelines",
                                  "guideline_regime", "date"]
                                 + [f"{segment_col}_{edge}" for segment_col in
                                    (pipeline.segment.column_name(n) for n in config.EXPORTED_SECTIONS)
                                    for edge in ("start", "end")]))
    df = load_cases(columns=columns)
    df = df[df["source_url"].isin(urls[:20])].sort_values("source_url")
    for row in df.to_dict("records"):
        if row["split"] == "test" and notes.facts_span(row):
            return row
    raise LookupError("no held-out test case with findings of fact")


@lru_cache(maxsize=1)
def _analyzed() -> tuple[dict, str]:
    row = _case()
    return pipeline.analyze_with_text(row["full_text"], source_url=row["source_url"], use_cache=False)


def _same(a, b) -> bool:
    missing = lambda v: v is None or v != v or v == ""  # noqa: E731
    if missing(a) and missing(b):
        return True
    return not missing(a) and not missing(b) and int(a) == int(b)


def test_prepare_rebuilds_the_dataset_row():
    row = _case()
    built = pipeline.prepare(row["full_text"], case_number=row["case_number"])
    for name in config.EXPORTED_SECTIONS:
        col = pipeline.segment.column_name(name)
        for edge in ("start", "end"):
            key = f"{col}_{edge}"
            assert _same(built[key], row[key]), (key, built[key], row[key])
    assert built["text_features"] == row["text_features"]
    assert list(built["guidelines"]) == list(row["guidelines"])
    assert built["guideline_regime"] == row["guideline_regime"]
    assert schema._iso_date(built["date"]) == schema._iso_date(row["date"])


def test_case_number_is_read_from_the_header():
    row = _case()
    assert pipeline.case_number_of(pipeline.normalize.strip_html_header(row["full_text"])[0]) == row["case_number"]


def test_end_to_end_document_validates_and_every_offset_quotes_its_text():
    doc, text = _analyzed()
    assert text == _case()["full_text"]
    schema.validate(doc, text)
    assert doc["case"]["text_length"] == len(text) and doc["case"]["split"] == "test"
    assert doc["conditions"], "the held-out case should have conditions"
    for c in doc["conditions"]:
        assert text[c["start"]:c["end"]] == c["quote"], c["id"]
        assert text[c["context"]["start"]:c["context"]["end"]] == c["context"]["text"], c["id"]
        if c["lead_in"] is not None:
            assert text[c["lead_in"]["start"]:c["lead_in"]["end"]] == c["lead_in"]["text"], c["id"]
    assert json.loads(json.dumps(doc, ensure_ascii=False)) == doc


def test_end_to_end_uses_published_heads_cached_llm_quotes_and_train_precedents():
    doc, _ = _analyzed()
    assert {g["model"] for g in doc["guidelines"]} == {schema.HEADS_RUN}
    assert any(c["source"] == "llm" for c in doc["conditions"]), "the case is in the LLM run"
    assert doc["provenance"]["llm_model"] == "claude-haiku-5-5"
    assert len(doc["precedents"]) == pipeline.TOP_K
    assert all(p["note"] is None for p in doc["precedents"]) and doc["rationale"] is None
    assert _case()["source_url"] not in {p["source_url"] for p in doc["precedents"]}


def test_precedent_offsets_quote_each_precedents_own_text():
    doc, _ = _analyzed()
    from src.dataset import load_cases

    urls = [p["source_url"] for p in doc["precedents"]]
    df = load_cases(columns=["source_url", "full_text", "split"])
    texts = dict(zip(df["source_url"], df["full_text"]))
    splits = dict(zip(df["source_url"], df["split"]))
    for p in doc["precedents"]:
        assert splits[p["source_url"]] == "train", p["case_number"]
        body = texts[p["source_url"]]
        assert 0 <= p["start"] < p["end"] <= len(body) and body[p["start"]:p["end"]].strip(), p["case_number"]
    assert urls


def test_new_text_falls_back_to_the_tfidf_heads():
    if not pipeline.FALLBACK_PATH.exists():
        raise AssertionError("run `python -m src.pipeline --fit-fallback` first")
    row = _case()
    with _checkpoint(MISSING_CHECKPOINT):
        doc, text = pipeline.analyze_with_text(row["full_text"], retrieve=False, use_cache=False)
    schema.validate(doc, text)
    assert {g["model"] for g in doc["guidelines"]} == {pipeline.FALLBACK_MODEL}
    assert doc["provenance"]["heads_model"] == pipeline.FALLBACK_MODEL and "warnings" not in doc["provenance"]
    assert doc["case"]["source_url"] == "" and doc["case"]["split"] is None
    assert not any(c["source"] == "llm" for c in doc["conditions"]) and doc["precedents"] == []
    assert doc["case"]["case_number"] == row["case_number"]


def test_demo_cache_serves_a_demo_case_by_its_text_alone():
    index = json.loads((pipeline.DEMO_DIR / "index.json").read_text(encoding="utf-8"))
    assert len(index["by_url"]) == 10 and len(index["by_hash"]) == 10
    url, name = sorted(index["by_url"].items())[0]
    row = notes.load_rows({url})[0]
    doc = pipeline.analyze(row["full_text"])            # no source_url: found by the text's hash
    stored = json.loads((pipeline.DEMO_DIR / name).read_text(encoding="utf-8"))
    assert doc == stored and doc["case"]["source_url"] == url
    assert all(p["note"] is not None for p in doc["precedents"][:1]) or doc["precedents"] == []


def test_demo_documents_carry_the_task_6_notes_and_rationales():
    index = json.loads((pipeline.DEMO_DIR / "index.json").read_text(encoding="utf-8"))
    rationales = notes_kept = 0
    for name in index["by_url"].values():
        doc = json.loads((pipeline.DEMO_DIR / name).read_text(encoding="utf-8"))
        rationales += doc["rationale"] is not None
        notes_kept += sum(p["note"] is not None for p in doc["precedents"])
    assert rationales == 9 and notes_kept == 49, (rationales, notes_kept)


def test_reading_text_returns_it_and_bad_input_is_refused():
    assert pipeline.read("plain decision text") == "plain decision text"
    try:
        pipeline.read(42)
    except TypeError:
        return
    raise AssertionError("an int was accepted")


def test_headings_without_facts_are_refused_and_the_message_names_the_section():
    text = "DISCUSSION\nThe applicant mitigated the concern.\n\nDECISION\nClearance is granted.\n"
    try:
        pipeline.analyze(text, use_cache=False)
    except pipeline.NoFeaturesError as exc:
        assert "FINDINGS OF FACT" in str(exc) and "DISCUSSION" in str(exc), str(exc)
        return
    raise AssertionError("empty model input produced a recommendation")


def test_empty_text_is_refused():
    for text in ("", "   \n\n"):
        try:
            pipeline.analyze(text, use_cache=False)
        except pipeline.NoFeaturesError:
            continue
        raise AssertionError("empty text produced a recommendation")


def test_a_paste_with_no_headings_is_read_as_the_facts_with_a_warning():
    text = ("Applicant is 41 years old. He owes about $23,000 in delinquent credit card debt and has not made "
            "payments since 2021. He disclosed the debts on his security clearance application.")
    with _checkpoint(MISSING_CHECKPOINT):
        doc, full_text = pipeline.analyze_with_text(text, retrieve=False, use_cache=False)
    schema.validate(doc, full_text)
    assert doc["provenance"]["warnings"] == [pipeline.NO_HEADINGS_WARNING]
    assert "no decision headings" in doc["provenance"]["warnings"][0].lower()
    row = pipeline.prepare(text)
    assert row["text_features"] == pipeline.normalize.build_features(text, [(0, len(text))])
    assert "delinquent credit card debt" in row["text_features"]


def test_a_full_decision_carries_no_warning():
    doc, _ = _analyzed()
    assert "warnings" not in doc["provenance"]
    assert pipeline.prepare(_case()["full_text"])["warnings"] == []


def test_the_schema_accepts_warnings_and_does_not_require_them():
    provenance = schema.json_schema()["$defs"]["Provenance"]
    assert provenance["properties"]["warnings"]["items"] == {"type": "string"}
    assert "warnings" not in provenance["required"]
    doc, text = _analyzed()
    schema.validate(doc, text)
    with_warning = json.loads(json.dumps(doc))
    with_warning["provenance"]["warnings"] = ["a caution"]
    schema.validate(with_warning, text)
    with_warning["provenance"]["warnings"] = "a caution"
    assert any("warnings" in p for p in schema.problems(with_warning, text))


def test_pdf_pages_are_read_and_joined_as_doha_did():
    import fitz

    doc = fitz.open()
    for words in ("FINDINGS OF FACT first page", "second page text"):
        doc.new_page().insert_text((72, 72), words)
    data = doc.tobytes()
    doc.close()
    expected_doc = fitz.open(stream=data, filetype="pdf")
    expected = "\n".join(page.get_text() for page in expected_doc)
    expected_doc.close()
    text = pipeline.read(data)
    assert text == expected and "first page" in text and "second page" in text
    assert text.index("first page") < text.index("second page")


def test_live_modernbert_with_a_missing_checkpoint_raises():
    with _checkpoint(MISSING_CHECKPOINT):
        try:
            pipeline.heads({"source_url": "", "text_features": "x"}, backend="modernbert")
        except FileNotFoundError as exc:
            assert "checkpoint" in str(exc)
            return
    raise AssertionError("live ModernBERT ran without a checkpoint")


def test_backend_order_is_published_then_live_then_tfidf():
    row = _case()
    assert pipeline.heads(row)["model"] == schema.HEADS_RUN            # a published test case
    new = {"source_url": "", "text_features": row["text_features"]}
    with _checkpoint(MISSING_CHECKPOINT):
        assert not infer.available()
        assert pipeline.heads(new)["model"] == pipeline.FALLBACK_MODEL
        try:
            pipeline.heads(new, backend="published")
        except KeyError:
            pass
        else:
            raise AssertionError("published backend answered for an unpublished case")
    if not infer.available():
        print("  (live half skipped: no ModernBERT checkpoint)")
        return
    assert pipeline.heads(row)["model"] == schema.HEADS_RUN            # published still wins with the checkpoint
    live = pipeline.heads(new)
    assert live["model"] == infer.LIVE_MODEL


def test_live_probabilities_match_the_published_val_outputs():
    if not infer.available():
        print(f"  (skipped: no ModernBERT checkpoint at {infer.checkpoint_path()})")
        return
    import numpy as np

    from src.dataset import load_cases

    published = pd_read_val()
    df = load_cases(columns=["source_url", "split", "text_features"])
    df = df[df["split"].eq("val") & df["source_url"].isin(published.index) & df["text_features"].ne("")]
    df = df.assign(n=df["text_features"].str.len()).sort_values("n").iloc[[0, len(df) // 2, len(df) - 1]]
    worst = 0.0
    for row in df.to_dict("records"):
        live = pipeline.heads({"source_url": "", "text_features": row["text_features"]}, backend="modernbert")
        assert live["model"] == infer.LIVE_MODEL
        want = schema.load_head_outputs(row["source_url"], "val")["guidelines"]
        for code in config.GUIDELINE_CODES:
            got = live["guidelines"][code]
            assert abs(got["at_issue_threshold"] - want[code]["at_issue_threshold"]) < 1e-12
            assert abs(got["p_not_at_issue"] + got["p_for"] + got["p_against"] - 1) < 1e-9
            for key in ("p_not_at_issue", "p_for", "p_against"):
                worst = max(worst, abs(got[key] - want[code][key]))
    print(f"  (live versus published, max probability difference {worst:.4f} over {len(df)} val cases)")
    assert worst < 0.02, worst


def pd_read_val():
    import pandas as pd

    frame = pd.read_parquet(schema.HEADS_DIR / "val_probabilities.parquet")
    return frame.set_index("source_url")


def test_a_live_document_names_the_live_model():
    if not infer.available():
        print("  (skipped: no ModernBERT checkpoint)")
        return
    text = _case()["full_text"]
    doc, full_text = pipeline.analyze_with_text(text, retrieve=False, use_cache=False)   # no source_url: not published
    schema.validate(doc, full_text)
    assert {g["model"] for g in doc["guidelines"]} == {infer.LIVE_MODEL}
    assert doc["provenance"]["heads_model"] == infer.LIVE_MODEL


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "pipeline tests"))
