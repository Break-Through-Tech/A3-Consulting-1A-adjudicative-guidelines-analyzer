"""Unit tests for the cleaning pipeline.

    python -m pytest tests/ -q

Many of these are NEGATIVE tests -- they assert a pattern does *not* match.
That is deliberate. The source repo's CI has 146 regex tests and every single
one asserts `should_match=True`, which is precisely why two label bugs survived
it and corrupted ~21,000 rows. A pattern that matches everything passes every
positive test ever written.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import parametrize, run_module  # noqa: E402
from src import fields, labels, normalize, segment  # noqa: E402


# --- the two bugs that shipped ----------------------------------------------

@parametrize("text", ["AG \u00b6 20(b)", "AG \u00b6 22(a)", "AG \u00b6 24", "AG \u00b6 27", "AG \u00b6 28"])
def test_guideline_a_ignores_higher_paragraph_numbers(text):
    """`AG para 2` must not match inside 20/22/24/27/28.

    Those paragraphs belong to Guidelines F, G, H and I. Without the (?!\\d)
    boundary this flagged Guideline A on 20,662 cases; the true count is 3.
    """
    assert "A" not in labels.fallback_labels(text, None)


@parametrize("text", ["AG \u00b6 2(c)", "AG \u00b6 2(b)", "Guideline A", "Allegiance to the United States"])
def test_guideline_a_still_matches_genuine_references(text):
    assert "A" in labels.fallback_labels(text, None)


def test_guideline_c_does_not_match_the_word_can():
    """`Guideline C` without \\b matched inside 'Guideline can'."""
    assert "C" not in labels.fallback_labels("Adjudication under this Guideline can apply", None)


def test_guideline_c_matches_a_real_citation():
    assert "C" in labels.fallback_labels("Guideline C (Foreign Preference)", None)


def test_policies_span_is_excluded_from_fallback():
    """POLICIES recites every guideline; leaving it in poisons the labels."""
    text = "Guideline F applies.\nPOLICIES BLOCK Guideline H Guideline J\nend."
    policies = (text.index("POLICIES BLOCK"), text.index("\nend."))
    assert "H" not in labels.fallback_labels(text, policies)
    assert "F" in labels.fallback_labels(text, policies)


# --- outcome: negation is the whole ballgame --------------------------------

@parametrize("text,expected", [
    ("Eligibility for access to classified information is denied.", "DENIED"),
    ("Applicant's eligibility for a security clearance is granted.", "GRANTED"),
    ("It is clearly consistent with the national interest to grant.", "GRANTED"),
    ("It is not clearly consistent with the national interest to grant.", "DENIED"),
    ("It is clearly not consistent with the national interest.", "DENIED"),
    ("Applicant's security clearance is revoked.", "REVOKED"),
    ("The case is remanded to the Administrative Judge.", "REMANDED"),
])
def test_outcome_direct_phrasing(text, expected):
    assert fields.extract_outcome(text, (0, len(text)))[0] == expected


def test_negation_beats_the_phrase_it_contains():
    """'not clearly consistent' contains 'clearly consistent'.

    Both match and both END at the same character. Choosing by start position
    hands the win to the shorter inner match, which silently flipped 2,170 rows
    from DENIED to GRANTED before it was caught.
    """
    text = "it is not clearly consistent with the national interest"
    assert fields.extract_outcome(text, (0, len(text)))[0] == "DENIED"


@parametrize("text,expected", [
    ("The Judge's decision denying Applicant a clearance is AFFIRMED.", "DENIED"),
    ("The decision of the Administrative Judge granting Applicant a clearance is AFFIRMED.", "GRANTED"),
    ("The Judge's adverse security clearance decision is REVERSED.", "GRANTED"),
    ("The Judge's favorable decision is REVERSED.", "DENIED"),
])
def test_appeal_outcome_is_a_composition(text, expected):
    """An appeal outcome is underlying-decision x board-action.

    The span contains neither 'is denied' nor 'is granted', so flat phrase
    matching cannot resolve it.
    """
    assert fields.extract_outcome(text, (0, len(text)))[0] == expected


def test_unresolvable_outcome_is_null_not_a_guess():
    import pandas as pd
    text = "The parties submitted written argument on the matter."
    assert pd.isna(fields.extract_outcome(text, (0, len(text)))[0])


# --- HTML scrape header ------------------------------------------------------

def test_html_header_length_is_variable_not_fixed():
    """Measured across all 7,607 affected rows: 136-169 chars, 9 distinct
    lengths. 159 is only the mode. A fixed-width cut damages 2,572 rows."""
    short = "02-0.h1 file:///a/b.html[6/1/2021 1:02:03 AM] BODY"
    long = "02-05086.h1 file:///usr.osd.mil/" + "x" * 60 + "/02-05086.h1.html[12/24/2021 10:49:28 AM] BODY"
    for text in (short, long):
        cleaned, removed = normalize.strip_html_header(text)
        assert cleaned.strip() == "BODY"
        assert removed == text.index("]") + 1   # ends at the timestamp, not the body


def test_documents_without_the_header_are_untouched():
    text = "DECISION OF ADMINISTRATIVE JUDGE\n\nFINDINGS OF FACT\n"
    assert normalize.strip_html_header(text) == (text, 0)


# --- normalisation order -----------------------------------------------------

def test_page_numbers_are_removed_before_hyphens_are_rejoined():
    """A page break can land inside a split word. Rejoining first leaves the
    page number embedded in the middle of it."""
    assert "financial" in normalize.normalize_body("finan-\n\n7\n\ncial obligations")


def test_full_text_is_never_mutated_by_normalisation():
    original = "Some \u201ctext\u201d with  runs\n\n5\n\nand a finan-\ncial word."
    normalize.normalize_body(original)
    assert original == "Some \u201ctext\u201d with  runs\n\n5\n\nand a finan-\ncial word."


# --- segmentation ------------------------------------------------------------

DOC = """DECISION OF ADMINISTRATIVE JUDGE

SYNOPSIS

Applicant did not mitigate the concerns.

STATEMENT OF THE CASE

The SOR was issued on 1 June 2019.

FINDINGS OF FACT

Applicant owes $12,000 on three accounts.

POLICIES

Guideline F, Guideline H and Guideline J are recited here generically.

FORMAL FINDINGS

Paragraph 1, Guideline F:

AGAINST APPLICANT

Conclusion

Eligibility for access to classified information is denied.
"""


def test_sections_are_located():
    spans, _, partial = segment.segment(DOC)
    assert not partial
    for name in ("SYNOPSIS", "STATEMENT OF THE CASE", "FINDINGS OF FACT", "POLICIES", "FORMAL FINDINGS"):
        assert name in spans


def test_feature_spans_exclude_synopsis_and_verdict():
    """The synopsis states the outcome in 61% of real documents."""
    spans, _, _ = segment.segment(DOC)
    feature_text = "".join(DOC[s:e] for s, e in segment.feature_spans(spans))
    assert "owes $12,000" in feature_text
    assert "did not mitigate" not in feature_text
    assert "is denied" not in feature_text


def test_verdict_span_is_chosen_by_position_not_name():
    """'Decision' is a caption at ~3% here; the ruling is under 'Conclusion'."""
    spans, _, _ = segment.segment(DOC)
    span = segment.verdict_span(spans)
    assert "is denied" in DOC[span[0]:span[1]]


def test_segment_never_raises_on_junk():
    for text in ("", "   ", "no headers here at all", "\n\n\n"):
        spans, _, partial = segment.segment(text)
        assert isinstance(spans, dict)
        assert partial


# --- formal rulings ----------------------------------------------------------

def test_formal_ruling_reads_the_verdict_from_the_next_line():
    """The FOR/AGAINST routinely sits on its own line below the heading."""
    codes, findings, _ = labels.extract_formal_rulings(DOC)
    assert codes == ["F"]
    assert findings == {"F": "AGAINST"}


def test_subparagraphs_do_not_capture_the_guideline_heading():
    """Before the digit-dot-letter shape was required, this pattern recorded
    'Paragraph 1, Guideline B' as if it were a subparagraph reference."""
    text = "Paragraph 1, Guideline B:\n\nAGAINST APPLICANT\n\nSubparagraphs 1.a-1.c:\n\nAgainst Applicant\n"
    _, _, subparagraphs = labels.extract_formal_rulings(text)
    assert [s["para"] for s in subparagraphs] == ["1.a-1.c"]


def test_formal_ruling_beats_fallback():
    codes, _, _, source = labels.label_row(DOC, None)
    assert source == labels.SOURCE_FORMAL
    assert codes == ["F"]


# --- judge -------------------------------------------------------------------

@parametrize("junk", [
    "Board reverses the", "evidence before the", "The Appeal Board",
    "Administrative Judge", "Security Clearance Decision",
])
def test_sentence_fragments_are_rejected_as_judge_names(junk):
    """The source extractor stored all of these as judge names."""
    assert not fields._looks_like_name(junk)


@parametrize("name", ["James F. Duffy", "Darlene Lokey Anderson", "Michael Y. Ra'anan"])
def test_real_names_are_accepted(name):
    assert fields._looks_like_name(name)


def test_apostrophe_variants_share_an_identity_key():
    assert fields.name_key("Michael Y. Ra\u2019anan") == fields.name_key("Michael Y. Ra'anan")


# --- dates -------------------------------------------------------------------

def test_implausible_years_are_nulled_not_kept():
    """Source dates parsed to years 214 and 9201."""
    import pandas as pd
    value, source = fields.extract_date("Date: March 3, 0214\n", "03-12345")
    assert pd.isna(value)
    assert source == "unresolved"


def test_date_must_be_consistent_with_the_docket_year():
    """A decision cannot predate the docket. 19-02453 cannot be decided in 1999."""
    import pandas as pd
    value, _ = fields.extract_date("Date: June 1, 1999\n", "19-02453")
    assert pd.isna(value)


# --- citation regime ---------------------------------------------------------

def test_regime_detection():
    assert fields.extract_regime("cited under AG \u00b6 19(a)") == "sead4"
    assert fields.extract_regime("under E2.A2.1.2.1 of the Directive") == "directive"
    assert fields.extract_regime("no citation scheme present") == "unknown"


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "unit tests"))
