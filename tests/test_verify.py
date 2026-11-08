"""Tests for src/llm/verify.py, the LLM quote checker.

    python tests/test_verify.py

Most of these are NEGATIVE tests: a reworded, invented, stitched or boilerplate
quote must be dropped, and dropped for the right reason. The positive tests use
the layout PyMuPDF actually produces -- line breaks mid-sentence, curly quotes,
words split at line ends, page numbers and running headers inside sentences --
because that is what a model's tidily spaced quote has to be matched against.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import parametrize, run_module  # noqa: E402
from src.llm import verify as V  # noqa: E402

RSQ = "’"            # curly apostrophe
LDQ, RDQ = "“", "”"
EM = "—"
HEADER = "19-01234.h1"    # running header, printed on every page
PRINT_HEADER = (          # browser print header, as it sits between pages
    "03-04628.h1\nfile:///usr.osd.mil/...Computer/Desktop/DOHA%20transfer/DOHA-Kane/dodogc/doha/"
    "industrial/Archived%20-%20HTML/03-04628.h1.html[6/24/2021 3:11:28 PM]"
)

REF_20B = (
    "the conditions that resulted in the financial problem were largely beyond the "
    "person's control (e.g., loss of employment, a business downturn, unexpected medical "
    "emergency, a death, divorce or separation, clear victimization by predatory lending "
    "practices, or identity theft), and the individual acted responsibly under the circumstances"
)

# A small decision laid out the way PyMuPDF writes one.
STATEMENT = f"STATEMENT OF THE CASE \nOn May 2, 2019, the DOD issued an SOR. \n{HEADER}\n"
POLICIES = (
    "POLICIES \n"
    f"Each guideline lists conditions. \n{HEADER}\n"
    "An \nindividual who is financially overextended is at greater risk of having to \n"
    "engage in illegal or otherwise questionable acts to generate funds. \n"
)
ANALYSIS = (
    "Analysis \n"
    f"Applicant{RSQ}s debts arose after he was laid off in March 2015 and his wife{RSQ}s \n"
    "medical bills reached $12,000. He contacted every creditor in 2020, and set up a plan for repay-\n"
    "ment of the delinquent accounts. She was self-\nemployed until 2019 and the debts were \n"
    " \n4 \n"
    "incurred after the layoff. AG ¶ 20(b) provides: \n"
    f"{REF_20B.replace(' financial ', ' financial \n')}. \n"
    f"The record shows he {LDQ}did what he could{RDQ} to resolve the debts. \n"
    "He paid the first account. He paid the first account. \n"
    "Applicant testified that he . . . paid each creditor in full by 2021. \n"
    "He recieved financial counseling in 2021 and has followed its budget. \n"
    f"From 2015-2019 he was unemployed, and his debts{EM}including two car loans{EM}went unpaid. \n"
    f"Applicant{RSQ}s monthly budget shows a \n{HEADER}\nsmall surplus each month. \n"
    "He has not shown that he acted responsibly toward his creditors. \n"
    f"The SOR alleged seven delinquent \n{PRINT_HEADER}\ndebts totaling $9,800. \n"
)
FULL_TEXT = STATEMENT + POLICIES + ANALYSIS
REGION = (FULL_TEXT.index("Analysis"), len(FULL_TEXT))


def item(quote, code="AG ¶ 20(b)"):
    return {"condition_code": code, "exact_quote": quote}


def check(quote, code="AG ¶ 20(b)", **kw):
    kw.setdefault("region", REGION)
    return V.verify_quotes(FULL_TEXT, [item(quote, code)], **kw)


def dropped(quote, **kw):
    res = check(quote, **kw)
    assert not res.kept, f"expected a drop, kept {res.kept}"
    return res.dropped[0]


def reason(quote, **kw):
    return dropped(quote, **kw).reason


def kept(quote, **kw):
    res = check(quote, **kw)
    assert res.kept, f"expected a keep, dropped as {res.dropped[0].reason if res.dropped else '?'}"
    return res.kept[0]


# --- what a correct quote looks like when a model writes it ---------------------

def test_exact_quote_is_kept_and_offsets_point_at_it():
    q = "He contacted every creditor in 2020"
    v = kept(q)
    assert FULL_TEXT[v.start:v.end] == q == v.text
    assert v.condition_code == "AG ¶ 20(b)"


def test_quote_across_line_breaks_is_kept_with_original_layout():
    v = kept("his wife's medical bills reached $12,000. He contacted every creditor in 2020")
    assert "\n" in v.text
    assert v.text.startswith("his wife") and v.text.endswith("2020")


def test_straight_apostrophe_matches_curly():
    v = kept("Applicant's debts arose after he was laid off in March 2015")
    assert RSQ in v.text


def test_straight_double_quotes_match_curly():
    kept('The record shows he "did what he could" to resolve the debts')


@parametrize("quote", [
    "set up a plan for repayment of the delinquent accounts",
    "set up a plan for repay-ment of the delinquent accounts",
])
def test_word_split_at_a_line_end_matches_either_way(quote):
    kept(quote)


def test_hyphenated_word_split_at_a_line_end_keeps_its_hyphen():
    kept("She was self-employed until 2019")


@parametrize("quote", [
    "From 2015-2019 he was unemployed",
    "From 2015 - 2019 he was unemployed",
    "From 2015–2019 he was unemployed",
])
def test_dash_variants_between_years_match(quote):
    kept(quote)


@parametrize("quote", [
    f"his debts{EM}including two car loans{EM}went unpaid",
    "his debts - including two car loans - went unpaid",
    "his debts--including two car loans--went unpaid",
])
def test_dash_variants_between_words_match(quote):
    kept(quote)


@parametrize("quote", [
    "the debts were incurred after the layoff",
    "the debts were 4 incurred after the layoff",
])
def test_page_number_line_inside_the_passage_is_ignored(quote):
    v = kept(quote)
    assert "\n4 \n" in v.text


@parametrize("quote", [
    "Applicant's monthly budget shows a small surplus each month",
    f"Applicant's monthly budget shows a {HEADER} small surplus each month",
])
def test_running_header_inside_the_passage_is_ignored(quote):
    v = kept(quote)
    assert HEADER in v.text


def test_browser_print_header_between_pages_is_ignored():
    v = kept("The SOR alleged seven delinquent debts totaling $9,800")
    assert "file:///" in v.text


@parametrize("quote", [
    "he contacted every creditor in 2020",          # capital lost
    "Medical bills reached $12,000",                # capital added mid-sentence
])
def test_case_of_the_first_letter_is_forgiven(quote):
    kept(quote)


def test_wrapping_quotation_marks_and_a_trailing_period_are_forgiven():
    kept('"He contacted every creditor in 2020."')


def test_non_breaking_space_and_soft_hyphen_are_layout():
    kept("He contacted every cred­i­tor in 2020")


def test_judges_own_ellipsis_is_kept_when_quoted_verbatim():
    kept("Applicant testified that he . . . paid each creditor in full by 2021")


@parametrize("code", ["AG ¶ 20(b)", "20(b)", "¶20 (B)", "AG ¶¶ 20(b)", " ag ¶ 20(b) "])
def test_code_forms_resolve_to_the_code_list_entry(code):
    v = kept("He contacted every creditor in 2020", code=code, allowed_codes=["AG ¶ 20(b)", "AG ¶ 20(d)"])
    assert v.condition_code == "AG ¶ 20(b)"


def test_directive_codes_are_normalised_not_mangled():
    assert V.normalize_code(" e2.a6.1.3.1 ") == "E2.A6.1.3.1"


def test_repeated_passage_keeps_the_first_and_counts_both():
    v = kept("He paid the first account.")
    assert v.n_matches == 2
    assert v.start == FULL_TEXT.index("He paid the first account.")


def test_one_passage_can_support_two_conditions():
    q = "He contacted every creditor in 2020"
    res = V.verify_quotes(FULL_TEXT, [item(q, "AG ¶ 20(b)"), item(q, "AG ¶ 20(d)")], region=REGION)
    assert len(res.kept) == 2 and not res.dropped


def test_spans_start_and_end_on_text_not_layout():
    v = kept("  He contacted every creditor in 2020  ")
    assert not v.text[0].isspace() and not v.text[-1].isspace()


# --- context: the whole sentence around a kept quote ---------------------------------

def test_context_is_the_whole_sentence():
    v = kept("He contacted every creditor in 2020")
    assert v.context == ("He contacted every creditor in 2020, and set up a plan for repay-\n"
                         "ment of the delinquent accounts.")
    assert FULL_TEXT[v.context_start:v.context_end] == v.context


def test_context_shows_a_negation_the_quote_leaves_out():
    v = kept("he acted responsibly toward his creditors")
    assert v.context.startswith("He has not shown that he acted responsibly")


def test_context_of_a_quote_that_ends_its_sentence_stops_there():
    v = kept("He paid the first account.")
    assert v.context == "He paid the first account."


# --- what must be dropped, and why ----------------------------------------------------

@parametrize("quote", [
    "Applicant's debts arose after he lost his job in March 2015",       # reworded
    "He contacted every creditor, and set up a plan",                   # words dropped
    "Applicant has paid all of his federal student loans in full",      # from another case
])
def test_quotes_not_in_the_decision_are_dropped(quote):
    assert reason(quote) == V.NOT_FOUND


@parametrize("quote,near", [
    ("his wife's medical bills reached $12000", "medical bills"),                    # number reformatted
    ("his wife's medical bills reached $13,000", "medical bills"),                   # number changed
    ("He received financial counseling in 2021", "counseling"),                      # judge's typo fixed
    ("From 20152019 he was unemployed", "unemployed"),                               # hyphen between digits
])
def test_near_misses_are_dropped_and_located(quote, near):
    d = dropped(quote)
    assert d.reason == V.NEAR_MISS
    assert near in FULL_TEXT[d.span[0]:d.span[1]]


@parametrize("bad", [
    None,
    "AG 20(b): he contacted every creditor",
    ["AG ¶ 20(b)", "He contacted every creditor in 2020"],
    {"condition_code": 20, "exact_quote": "He contacted every creditor in 2020"},
    {"condition_code": "AG ¶ 20(b)"},
    {"exact_quote": "He contacted every creditor in 2020"},
    {"condition_code": "AG ¶ 20(b)", "exact_quote": None},
])
def test_malformed_items_are_dropped(bad):
    res = V.verify_quotes(FULL_TEXT, [bad], region=REGION)
    assert not res.kept and res.dropped[0].reason == V.MALFORMED


@parametrize("quote", ["", "   ", " \n\t ", "­​"])
def test_blank_quotes_are_dropped(quote):
    assert reason(quote) == V.EMPTY_QUOTE


@parametrize("code", ["AG ¶ 20(z)", "AG ¶ 19(b)", "", "   ", "mitigating condition b"])
def test_codes_outside_the_list_are_dropped(code):
    assert reason("He contacted every creditor in 2020", code=code,
                  allowed_codes=["AG ¶ 20(b)", "AG ¶ 20(d)"]) == V.BAD_CODE


def test_blank_code_is_dropped_even_without_a_code_list():
    assert reason("He contacted every creditor in 2020", code="  ") == V.BAD_CODE


def test_quote_with_the_citation_mask_is_dropped():
    assert reason("[CITATION] provides that the conditions", mask_token="[CITATION]") == V.CONTAINS_MASK


@parametrize("quote", ["Applicant", "the debts", "He paid."])
def test_short_quotes_are_dropped_even_though_they_occur(quote):
    assert reason(quote) == V.TOO_SHORT


@parametrize("quote", [
    "the conditions that resulted in the financial problem were largely beyond the person's control",
    "AG ¶ 20(b) provides: the conditions that resulted in the financial problem were largely "
    "beyond the person's control",
])
def test_reciting_the_condition_wording_is_not_evidence(quote):
    assert reason(quote, reference_texts={"AG ¶ 20(b)": REF_20B}) == V.RESTATES_GUIDELINE


def test_case_facts_are_not_mistaken_for_condition_wording():
    kept("He contacted every creditor in 2020", reference_texts={"AG ¶ 20(b)": REF_20B})


def test_quote_from_outside_what_the_model_saw_is_dropped():
    q = "individual who is financially overextended is at greater risk"
    d = dropped(q)
    assert d.reason == V.OUTSIDE_REGION
    assert FULL_TEXT[d.span[0]:d.span[1]].startswith("individual")
    assert V.verify_quotes(FULL_TEXT, [item(q)]).kept  # fine when no region is given


def test_quote_running_past_the_region_end_is_dropped():
    region = (REGION[0], FULL_TEXT.index("in 2020"))
    assert reason("He contacted every creditor in 2020", region=region) == V.OUTSIDE_REGION


def test_stitched_passages_are_dropped_as_ellipsis():
    assert reason("He contacted every creditor in 2020 ... She was self-employed until 2019") == V.ELLIPSIS


def test_ellipsis_over_invented_text_is_plain_not_found():
    assert reason("He contacted every creditor in 2020 ... and then he won the lottery last year") == V.NOT_FOUND


def test_changed_letter_case_after_the_first_letter_is_dropped_and_named():
    assert reason("He Contacted Every Creditor in 2020") == V.CASE_MISMATCH


def test_the_same_code_and_passage_twice_is_a_duplicate():
    q = "He contacted every creditor in 2020"
    res = V.verify_quotes(FULL_TEXT, [item(q), item(q.upper().lower().capitalize())], region=REGION)
    assert len(res.kept) == 1
    assert [d.reason for d in res.dropped] == [V.DUPLICATE]


def test_counts_cover_every_item_and_sum_across_decisions():
    items = [item("He contacted every creditor in 2020"), item("Applicant"), None, item("")]
    res = V.verify_quotes(FULL_TEXT, items, region=REGION)
    assert res.counts == {V.KEPT: 1, V.TOO_SHORT: 1, V.MALFORMED: 1, V.EMPTY_QUOTE: 1}
    assert sum(res.counts.values()) == len(items)
    assert V.total_counts([res, res])[V.KEPT] == 2


def test_amounts_repeated_in_a_table_are_not_treated_as_headers():
    text = "Account 1\n$1,250\nAccount 2\n$1,250\nAccount 3\n$1,250\nHe owes $1,250 on each of three accounts.\n"
    assert not V._furniture(text)
    # Skipping the amounts would let a quote that leaves them out pass.
    res = V.verify_quotes(text, [item("Account 1 Account 2 Account 3")])
    assert not res.kept


@parametrize("bad_items", ["He contacted every creditor", {"condition_code": "x", "exact_quote": "y"}, None])
def test_items_must_be_a_list(bad_items):
    try:
        V.verify_quotes(FULL_TEXT, bad_items)
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def test_full_text_must_be_a_string():
    try:
        V.verify_quotes(None, [])
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def test_empty_decision_finds_nothing():
    res = V.verify_quotes("", [item("He contacted every creditor in 2020")])
    assert res.dropped[0].reason == V.NOT_FOUND


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals()), "verify tests"))
