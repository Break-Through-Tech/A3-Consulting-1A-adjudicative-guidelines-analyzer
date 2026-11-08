"""verify.py against real decisions, not just a hand-made fixture.

    python tests/test_verify_realdata.py

Takes 150 Guideline F pool decisions (fixed seed) and quotes sentences from their
analysis and conclusions the way a model writes them: ordinary spacing, straight
quotes, a word split at a line end joined back up, page numbers left out, and
sometimes a capital added or dropped on the first letter. Every one of those must
be kept, with offsets that point at the original text.

Then it changes one word, or one digit, in each quote. None of those may be kept.

Loads `full_text` for the pool, so it takes about a minute.
"""
from __future__ import annotations

import random
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_module  # noqa: E402
from src.dataset import load_cases  # noqa: E402
from src.llm import verify as V  # noqa: E402

N_DECISIONS = 150
SENTENCES_PER_DECISION = 4
MIN_KEEP_RATE = 0.995

_PAGE_NUMBER = re.compile(r"\n[ \t]*\d{1,3}[ \t]*(?=\n)")
_SPLIT_WORD = re.compile(r"([a-z])-[ \t]*\n\s*([a-z])")
_STRAIGHT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


@lru_cache(maxsize=1)
def _quotes() -> list[tuple[str, tuple[int, int], str]]:
    """(full_text, region, model-style quote) for real analysis sentences."""
    cols = ["guidelines", "full_text", "sec_analysis_start", "sec_analysis_end",
            "sec_conclusions_start", "sec_conclusions_end"]
    df = load_cases(columns=cols)
    df = df[df["guidelines"].map(lambda g: "F" in list(g))]
    df = df.sample(N_DECISIONS, random_state=0)
    rng = random.Random(0)

    out = []
    for row in df.itertuples():
        starts = [v for v in (row.sec_analysis_start, row.sec_conclusions_start) if v == v and v >= 0]
        ends = [v for v in (row.sec_analysis_end, row.sec_conclusions_end) if v == v and v >= 0]
        if not starts or not ends:
            continue
        region = (int(min(starts)), int(max(ends)))
        segment = row.full_text[region[0]:region[1]]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", segment)]
        sentences = [s for s in sentences if 60 <= len(s) <= 400]
        for sentence in rng.sample(sentences, min(SENTENCES_PER_DECISION, len(sentences))):
            quote = _PAGE_NUMBER.sub("", sentence + "\n")
            quote = _SPLIT_WORD.sub(lambda m: m.group(1) + rng.choice(["", "-"]) + m.group(2), quote)
            quote = re.sub(r"\s+", " ", quote).strip()
            if rng.random() < 0.7:
                quote = quote.translate(_STRAIGHT)
            if rng.random() < 0.3 and quote[:1].isalpha():
                quote = quote[0].swapcase() + quote[1:]
            out.append((row.full_text, region, quote))
    return out


def _kept(full_text: str, region: tuple[int, int], quote: str) -> list[V.VerifiedQuote]:
    item = {"condition_code": "AG ¶ 20(b)", "exact_quote": quote}
    return V.verify_guideline_quotes(full_text, [item], "F", region=region).kept


def test_correct_quotes_are_kept_and_point_at_the_original_text():
    quotes = _quotes()
    assert len(quotes) >= 500, f"only {len(quotes)} sentences sampled"
    kept = 0
    for full_text, region, quote in quotes:
        hits = _kept(full_text, region, quote)
        if hits:
            kept += 1
            span = hits[0]
            assert full_text[span.start:span.end] == span.text
            assert region[0] <= span.start < span.end <= region[1]
            assert span.text in span.context
    assert kept / len(quotes) >= MIN_KEEP_RATE, f"kept {kept}/{len(quotes)}"


def test_one_changed_word_is_never_kept():
    rng = random.Random(1)
    tried = 0
    for full_text, region, quote in _quotes():
        words = [m for m in re.finditer(r"\b[a-z]{5,}\b", quote) if m.group(0) != m.group(0)[::-1]]
        if not words:
            continue
        m = rng.choice(words)
        changed = quote[:m.start()] + m.group(0)[::-1] + quote[m.end():]
        assert not _kept(full_text, region, changed), changed
        tried += 1
    assert tried >= 400


def test_one_changed_digit_is_never_kept():
    tried = 0
    for full_text, region, quote in _quotes():
        if not re.search(r"\d", quote):
            continue
        changed = re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), quote, count=1)
        assert not _kept(full_text, region, changed), changed
        tried += 1
    assert tried >= 100


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "verify on real decisions"))
