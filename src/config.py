"""Shared constants for the DOHA cleaning pipeline.

Paths resolve relative to this file, so the working directory doesn't matter —
same approach as notebooks/01_inspect_data.py. The `doha/` repo is read-only:
nothing in this package writes to it.
"""
from pathlib import Path

# --- paths -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DOHA_ROOT = REPO_ROOT.parent / "doha"
SOURCE_DIR = DOHA_ROOT / "doha_parsed_cases"
SOURCE_GLOB = "all_cases_*.parquet"

OUT_DIR = REPO_ROOT / "Challenge_Data"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# Four shards keeps each file under GitHub's 100MB limit. Source is 171MB and we
# add text_features (~25% of corpus text), so budget ~205MB total.
N_SHARDS = 4
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "gzip"  # matches the source files

# --- the 13 SEAD-4 guidelines ------------------------------------------------

GUIDELINE_NAMES = {
    "A": "Allegiance to the United States",
    "B": "Foreign Influence",
    "C": "Foreign Preference",
    "D": "Sexual Behavior",
    "E": "Personal Conduct",
    "F": "Financial Considerations",
    "G": "Alcohol Consumption",
    "H": "Drug Involvement",
    "I": "Psychological Conditions",
    "J": "Criminal Conduct",
    "K": "Handling Protected Information",
    "L": "Outside Activities",
    "M": "Use of Information Technology",
}
GUIDELINE_CODES = list(GUIDELINE_NAMES)

OUTCOMES = ["GRANTED", "DENIED", "REVOKED", "REMANDED"]

# --- sentinels ---------------------------------------------------------------
# The source has zero true nulls. Missingness is encoded as one of these, and
# the two mean different things:
#   ''        -> field not applicable to this case type
#   'Unknown' -> extractor ran and failed
# Both become pd.NA; the distinction is preserved in the *_status columns.
SENTINEL_EMPTY = ""
SENTINEL_UNKNOWN = {"Unknown", "UNKNOWN", "unknown"}

# --- document section vocabulary ---------------------------------------------
# Casing varies (`Formal Findings` 66.5% / `FORMAL FINDINGS` 24.8%), so every
# match is case-insensitive. Order here is document order, which segment.py
# relies on to resolve section boundaries.

HEARING_SECTIONS = [
    "SYNOPSIS",
    "STATEMENT OF THE CASE",
    "FINDINGS OF FACT",
    "POLICIES",
    "ANALYSIS",
    "CONCLUSIONS",
    "FORMAL FINDINGS",
    "DECISION",
]

APPEAL_SECTIONS = [
    "SYNOPSIS",
    "STATEMENT OF THE CASE",
    "THE JUDGE'S FINDINGS OF FACT",
    "DISCUSSION",
    "ORDER",
]

# Regex alternates for headers whose wording varies. Keys are the canonical
# name used in column names (sec_<lowercased, underscored>).
SECTION_ALIASES = {
    "CONCLUSIONS": r"CONCLUSIONS?",
    "FORMAL FINDINGS": r"FORMAL FINDINGS?",
    "THE JUDGE'S FINDINGS OF FACT": r"(?:THE\s+)?JUDGE['’]?S\s+FINDINGS\s+OF\s+FACT",
}

# Sections written to the output as (start, end) offset columns.
EXPORTED_SECTIONS = [
    "SYNOPSIS",
    "STATEMENT OF THE CASE",
    "FINDINGS OF FACT",
    "POLICIES",
    "ANALYSIS",
    "CONCLUSIONS",
    "FORMAL FINDINGS",
    "DECISION",
    "DISCUSSION",
    "ORDER",
]

# Sections that make up the leak-free model input. Deliberately excludes
# SYNOPSIS (states the outcome in 61% of documents) and everything from
# ANALYSIS onward (the judge's reasoning and verdict).
#
# Measured coverage: hearings 91.0%, appeals 0.4% -- an appeal is a review OF a
# decision, so most contain no pre-verdict factual section at all. Including the
# appeal-side equivalent rescues 1,542 of them; the rest genuinely cannot supply
# leak-free input and are reported as such rather than patched over.
FEATURE_SECTIONS = [
    "STATEMENT OF THE CASE",
    "FINDINGS OF FACT",
    "THE JUDGE'S FINDINGS OF FACT",
]

# --- guideline citation regimes ----------------------------------------------
# Pre-2006 cases were adjudicated under Directive 5220.6, which numbers its
# paragraphs E2.An.n.n. SEAD-4 uses AG para n. The patterns are not
# interchangeable, so every row records which regime it belongs to.
REGIME_SEAD4 = "sead4"
REGIME_DIRECTIVE = "directive"
REGIME_UNKNOWN = "unknown"

# --- date validation ---------------------------------------------------------
# Observed parsed years span 214 to 9201. Anything outside this window is
# nulled and flagged rather than guessed at.
MIN_YEAR = 1990
MAX_YEAR = 2027

# --- judge extraction --------------------------------------------------------
# Tokens that prove a captured "name" is really a sentence fragment. The source
# extractor produced values like 'Board reverses the' and 'evidence before the'.
JUDGE_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "evidence", "board",
    "decision", "applicant", "government", "appeal", "case", "judge",
    "administrative", "security", "clearance", "guideline", "record",
}
JUDGE_MAX_LEN = 40
