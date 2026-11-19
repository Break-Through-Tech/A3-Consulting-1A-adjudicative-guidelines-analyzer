"""One decision in, one SEAD-4 document out (task 38).

    from src.pipeline import analyze
    doc = analyze("decision.pdf")                       # or the decision's text
    doc = analyze(text, source_url=url)                 # a pool case: cached heads and LLM quotes

    python -m src.pipeline decision.pdf --out doc.json
    python -m src.pipeline --fit-fallback               # train the TF-IDF heads once, into runs/
    python -m src.pipeline --build-demo                 # write results/demo/ for the 10 demo cases

The steps, each a function below:

    read        PyMuPDF, the call doha's download_pdfs.py used to build full_text:
                every page's get_text(), joined with "\\n"
    prepare     the build_dataset steps for one row: strip the HTML header,
                segment, labels, regime, date, the section columns and the
                leak-free text_features
    heads       the published ModernBERT-4096 outputs when the case is a val or
                test pool case, otherwise the TF-IDF baseline heads. Live
                ModernBERT needs the 600 MB checkpoint in runs/ and a GPU, so it is
                a stub until Rashed decides how to ship it
    conditions  src.conditions.extract_case
    llm         verified quotes from the cached test-split run; never a live call
    precedents  PrecedentRetriever.search over train cases with the findings of
                fact, index v1. Hybrid search plugs in here when it lands
    notes       rationale and similarity notes from the task 6 cache, demo cases only
    validate    schema.validate against full_text

The demo cache in results/demo/ holds the finished documents for the 10 task 6
demo cases, keyed by source_url and by the sha256 of full_text, so the demo runs
with no GPU and no API key. `analyze` checks it first.

Every offset in the result indexes the full_text that `read` returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import conditions, config, fields, labels, normalize, schema, segment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "results" / "demo"
NOTES_DOCS = PROJECT_ROOT / "results" / "llm" / "notes" / "documents"
FALLBACK_PATH = PROJECT_ROOT / "runs" / "pipeline" / "tfidf_heads.joblib"
FALLBACK_MODEL = "tfidf-baseline"
TOP_K = 5

_CASE_NUMBER = re.compile(r"\b(?:ISCR|ADP)?\s*Case\s+No\.?\s*:?\s*(\d{2}-\d{4,5})", re.I)
_DOCKET = re.compile(r"\b(\d{2}-\d{5})\b")


# --- read -------------------------------------------------------------------------------

def read(source) -> str:
    """full_text for a PDF path or PDF bytes, or the text itself."""
    if isinstance(source, (bytes, bytearray)):
        return _pdf_text(bytes(source))
    if isinstance(source, Path) or (isinstance(source, str) and len(source) < 1024 and source.lower().endswith(".pdf")):
        return _pdf_text(Path(source).read_bytes())
    if isinstance(source, str):
        return source
    raise TypeError("analyze takes decision text, a .pdf path or PDF bytes")


def _pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF, the library doha used
    except ImportError as exc:  # not in requirements.txt yet
        raise ImportError("Reading a PDF needs PyMuPDF: pip install PyMuPDF~=1.26.0, the version doha used") from exc
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def text_hash(full_text: str) -> str:
    return "sha256:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest()


# --- prepare ----------------------------------------------------------------------------

def case_number_of(body: str) -> str:
    head = body[:3000]
    m = _CASE_NUMBER.search(head) or _DOCKET.search(head)
    return m.group(1) if m else ""


def prepare(full_text: str, *, case_number: str | None = None, source_url: str = "",
            split: str | None = None) -> dict:
    """One row shaped like load_cases() output, built the way build_dataset.process_row builds it."""
    body, prefix = normalize.strip_html_header(full_text)
    spans, _, _ = segment.segment(body)
    case_number = case_number or case_number_of(body)
    codes, findings, _, label_source = labels.label_row(body, spans.get("POLICIES"))
    date, _ = fields.extract_date(body, case_number)
    row = {
        "full_text": full_text,
        "case_number": case_number,
        "source_url": source_url,
        "split": split,
        "date": None if pd.isna(date) else date,
        "guideline_regime": fields.extract_regime(body),
        "guidelines": codes,
        "guideline_findings": findings,
        "label_source": label_source,
        "text_features": normalize.build_features(body, segment.feature_spans(spans)),
    }
    shifted = segment.shift(spans, prefix)
    for name in config.EXPORTED_SECTIONS:
        column = segment.column_name(name)
        start, end = shifted.get(name, (None, None))
        row[f"{column}_start"], row[f"{column}_end"] = start, end
    for name in schema.SECTIONS:                          # every column extract_case and schema read
        row.setdefault(f"sec_{name}_start", None)
        row.setdefault(f"sec_{name}_end", None)
    return row


# --- heads ------------------------------------------------------------------------------

def published_split(source_url: str) -> str | None:
    """'val' or 'test' when the published ModernBERT outputs hold this case."""
    for split in ("val", "test"):
        try:
            schema.load_head_outputs(source_url, split)
        except KeyError:
            continue
        return split
    return None


def heads(row: dict, backend: str = "auto") -> dict:
    """Head outputs in `schema.load_head_outputs` form.

    auto       published ModernBERT for a val or test pool case, else TF-IDF
    published  published ModernBERT only
    tfidf      the TF-IDF baseline heads
    modernbert live inference, not built yet
    """
    if backend == "modernbert":
        raise NotImplementedError("live ModernBERT needs the runs/modernbert-4096 checkpoint; not shipped yet")
    if backend in ("auto", "published") and row.get("source_url"):
        split = published_split(row["source_url"])
        if split:
            return schema.load_head_outputs(row["source_url"], split)
    if backend == "published":
        raise KeyError(f"no published head outputs for {row.get('source_url')!r}")
    return tfidf_heads(row["text_features"])


def fit_fallback(path: Path = FALLBACK_PATH, log=print) -> Path:
    """Train the TF-IDF baseline heads on train, thresholds on val, as notebook 04 does. Saved to runs/."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from src.dataset import load_cases
    from src.heads.data import baseline

    b = baseline()
    df = load_cases(pool=True, columns=["case_number", "guidelines", "guideline_findings", "guideline_regime",
                                        b.TEXT_COL, "split"])
    train, val = df["split"].eq("train").to_numpy(), df["split"].eq("val").to_numpy()
    vectorizer = TfidfVectorizer(max_features=b.MAX_FEATURES, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(df.loc[train, b.TEXT_COL])
    x_val = vectorizer.transform(df.loc[val, b.TEXT_COL])
    models, thresholds = {}, {}
    for code in config.GUIDELINE_CODES:
        targets = b.targets_for_guideline(df, code)
        y_train, y_val = targets[train], targets[val]
        known = y_train.notna().to_numpy()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_train[known], y_train[known].to_numpy())
        vknown = y_val.notna().to_numpy()
        thresholds[code] = b.choose_at_issue_threshold(y_val[vknown], b.aligned_probabilities(model, x_val[vknown]))
        models[code] = model
        log(f"fitted {code}, threshold {thresholds[code]:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "models": models, "thresholds": thresholds,
                 "classes": b.CLASSES}, path)
    _fallback.cache_clear()
    return path


@lru_cache(maxsize=1)
def _fallback(path: str) -> tuple[dict, str]:
    import joblib

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No TF-IDF heads at {p}. Run `python -m src.pipeline --fit-fallback` once.")
    raw = p.read_bytes()
    return joblib.load(p), hashlib.sha256(raw).hexdigest()[:12]


def tfidf_heads(text_features: str, path: Path = FALLBACK_PATH) -> dict:
    from src.heads.data import baseline

    bundle, digest = _fallback(str(path))
    x = bundle["vectorizer"].transform([text_features])
    out = {}
    for code in config.GUIDELINE_CODES:
        p = baseline().aligned_probabilities(bundle["models"][code], x, tuple(bundle["classes"]))[0]
        p = p / p.sum()
        out[code] = {"p_not_at_issue": float(p[0]), "p_for": float(p[1]), "p_against": float(p[2]),
                     "at_issue_threshold": float(bundle["thresholds"][code])}
    return {"model": FALLBACK_MODEL,
            "calibration": f"runs/pipeline/{path.name} sha256:{digest}, val at-issue thresholds, no temperature",
            "guidelines": out}


# --- LLM quotes, precedents, notes --------------------------------------------------------

@lru_cache(maxsize=1)
def _run_cache():
    from src.llm import notes

    _, kept, counts = notes.run_tables()
    return kept, counts


def llm_hits(source_url: str) -> tuple[list[dict], dict]:
    """Cached verified quotes and verify counts for a test-split run case, else nothing."""
    if not source_url:
        return [], {}
    kept, counts = _run_cache()
    return kept.get(source_url, []), dict(counts.get(source_url, {}))


@lru_cache(maxsize=1)
def _retriever():
    from src.retrieval import PrecedentRetriever

    return PrecedentRetriever()


def precedents(row: dict, top_k: int = TOP_K) -> list[dict]:
    """Train precedents for the case's findings of fact, from index v1. Hybrid search plugs in here."""
    start, end = row.get("sec_findings_of_fact_start"), row.get("sec_findings_of_fact_end")
    if start is None or end is None:
        return []
    query = normalize.normalize_body(row["full_text"][int(start):int(end)])
    found = _retriever().search(query, top_k=top_k + 1, split="train")
    found = [p for p in found if p["source_url"] != row.get("source_url")][:top_k]
    return [{"case_number": str(p["case_number"]), "source_url": p["source_url"], "score": p["score"],
             "outcome": str(p["outcome"]), "guidelines": list(p["guidelines"]),
             "label_source": str(p["label_source"]), "start": int(p["best_paragraph_start"]),
             "end": int(p["best_paragraph_end"]), "note": None} for p in found]


@lru_cache(maxsize=1)
def _notes_documents() -> dict[str, dict]:
    out = {}
    for path in sorted(NOTES_DOCS.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[doc["case"]["source_url"]] = doc
    return out


def cached_notes(source_url: str) -> dict | None:
    """The task 6 document for a demo case: its precedents with notes, its rationale, its provenance."""
    return _notes_documents().get(source_url) if source_url else None


# --- the demo cache -----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _demo_index() -> dict:
    path = DEMO_DIR / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"by_url": {}, "by_hash": {}}


def cached_document(full_text: str, source_url: str = "") -> dict | None:
    index = _demo_index()
    name = index["by_url"].get(source_url) if source_url else None
    name = name or index["by_hash"].get(text_hash(full_text))
    if not name:
        return None
    doc = json.loads((DEMO_DIR / name).read_text(encoding="utf-8"))
    schema.validate(doc, full_text)          # a cached document must still quote this exact text
    return doc


# --- analyze ------------------------------------------------------------------------------

class NoFeaturesError(ValueError):
    """The decision has no leak-free section for the heads to read.

    About 9% of DOHA hearings, and most decisions from other agencies, whose
    headings the segmenter doesn't know. Empty input would decode to every
    guideline NOT_AT_ISSUE and a GRANTED recommendation, which says nothing.
    """


def analyze_with_text(source, *, source_url: str = "", case_number: str | None = None, split: str | None = None,
                      heads_backend: str = "auto", retrieve: bool = True, use_cache: bool = True) -> tuple[dict, str]:
    """(document, full_text). The document passes schema.validate against full_text."""
    full_text = read(source)
    if use_cache:
        doc = cached_document(full_text, source_url)
        if doc is not None:
            return doc, full_text

    row = prepare(full_text, case_number=case_number, source_url=source_url, split=split)
    if not row["text_features"]:
        raise NoFeaturesError(
            "No findings of fact or other pre-verdict section was found, so the heads would read nothing and "
            "report every guideline as not at issue. Refusing to return a recommendation from empty input.")
    head_probs = heads(row, heads_backend)
    if row["split"] is None and head_probs["model"] != FALLBACK_MODEL and source_url:
        row["split"] = published_split(source_url)
    rule = conditions.extract_case(row)
    llm, verify_counts = llm_hits(source_url)

    cached = cached_notes(source_url)
    if cached is not None:
        found, rationale = cached["precedents"], cached["rationale"]
        meta = {k: cached["provenance"][k] for k in ("llm_model", "prompt_hashes", "run_dates", "verify_counts")}
    else:
        found, rationale = (precedents(row) if retrieve else []), None
        meta = {"verify_counts": verify_counts}
        if llm:
            from src.llm import client, notes as llm_notes, templates

            meta.update({"llm_model": client.MODEL,
                         "prompt_hashes": {"extract": templates.prompt_hash(templates.SYSTEM_PROMPT)},
                         "run_dates": {"extract": llm_notes.RUN_DATE}})
    doc = schema.build_document(row, head_probs, rule, llm, found, rationale, provenance=meta)
    schema.validate(doc, full_text)
    return doc, full_text


def analyze(source, **kwargs) -> dict:
    """The SEAD-4 document for one decision: a .pdf path, PDF bytes or the decision's text."""
    return analyze_with_text(source, **kwargs)[0]


def build_demo(log=print) -> Path:
    """Write results/demo/: the finished document for each task 6 demo case, and the index."""
    from src.llm import notes

    demo = notes.read_csv(notes.NOTES / "demo.csv")
    rows = {r["source_url"]: r for r in notes.load_rows({d["source_url"] for d in demo})}
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    index = {"by_url": {}, "by_hash": {}}
    for d in demo:
        row = rows[d["source_url"]]
        doc, full_text = analyze_with_text(row["full_text"], source_url=d["source_url"],
                                           case_number=d["case_number"], use_cache=False)
        name = f"{int(d['position']):02d}_{d['case_number']}.json"
        (DEMO_DIR / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        index["by_url"][d["source_url"]] = name
        index["by_hash"][text_hash(full_text)] = name
        log(f"wrote {name}: {doc['recommendation']['composed']}, {len(doc['conditions'])} conditions")
    (DEMO_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    _demo_index.cache_clear()
    return DEMO_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="a decision .pdf or .txt file")
    ap.add_argument("--source-url", default="", help="the case's source_url, for a pool case")
    ap.add_argument("--out", type=Path, help="write the document here instead of printing it")
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--fit-fallback", action="store_true", help="train the TF-IDF heads into runs/pipeline/")
    ap.add_argument("--build-demo", action="store_true", help="write results/demo/ for the task 6 demo cases")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.fit_fallback:
        print(f"wrote {fit_fallback()}")
    if args.build_demo:
        print(f"wrote {build_demo()}")
    if args.source:
        path = Path(args.source)
        source = path if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        doc = analyze(source, source_url=args.source_url, retrieve=not args.no_retrieval)
        text = json.dumps(doc, indent=2, ensure_ascii=False)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    if not (args.source or args.fit_fallback or args.build_demo):
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
