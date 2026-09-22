"""Duplicate resolution.

A repeated `(case_number, case_type)` is two different things, and conflating
them is a mistake that deletes real cases:

  A. True duplicate (857 groups). Byte-identical `full_text`; 842 are identical
     across all 18 source columns and 15 differ only in `source_url`. DOHA lists
     some cases on more than one archive page, so the scraper collected them
     twice. One copy is DELETED.

  B. Case-number collision (221 groups). Different `FileId`, different length,
     different labels -- two genuinely different documents that happen to share
     a case number. BOTH ARE KEPT and flagged.

Why the obvious rule is wrong: "keep the longest full_text" picks the 63,735-char
row for 03-11420, which has outcome UNKNOWN, and discards the 9,312-char row
that actually records DENIED. Length is not completeness.
"""
from __future__ import annotations

import pandas as pd

MIN_USABLE_CHARS = 200


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_duplicate / case_number_collision / is_degenerate / use_for_training.

    `is_duplicate` marks rows to be deleted; nothing else is ever deleted.
    """
    out = df.copy()
    out["is_duplicate"] = False
    out["case_number_collision"] = False
    out["is_degenerate"] = out["full_text"].str.len().fillna(0) < MIN_USABLE_CHARS

    # Pass 1 -- exact duplicates, keyed on the text itself. Doing this before
    # the collision check matters: a group of three rows holding two identical
    # documents plus one different one is BOTH a duplicate and a collision, and
    # checking nunique() first would classify it as a collision and retain the
    # redundant copy.
    ordered = out.sort_values("source_url", kind="stable")
    exact = ordered.duplicated(["case_number", "case_type", "full_text"], keep="first")
    out.loc[exact[exact].index, "is_duplicate"] = True

    # Pass 2 -- among the survivors, a repeated key with differing text means
    # two genuinely different documents share a case number. Keep both.
    survivors = out[~out["is_duplicate"]]
    repeated = survivors.duplicated(["case_number", "case_type"], keep=False)
    collision_index = survivors[repeated].index
    out.loc[collision_index, "case_number_collision"] = True

    out["use_for_training"] = ~(out["is_duplicate"] | out["is_degenerate"])
    # Group key for splitting, so a hearing and its own appeal -- or the two
    # halves of a collision -- can never straddle a train/test boundary.
    out["split_group"] = out["case_number"]
    return out


def excluded_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per excluded record, with the reason and what was discarded."""
    frames = []

    dupes = df[df["is_duplicate"]]
    if len(dupes):
        frames.append(pd.DataFrame({
            "case_number": dupes["case_number"],
            "case_type": dupes["case_type"],
            "reason": "true_duplicate_deleted",
            "detail": "byte-identical full_text; discarded source_url: "
                      + dupes["source_url"].astype(str),
        }))

    degen = df[df["is_degenerate"]]
    if len(degen):
        frames.append(pd.DataFrame({
            "case_number": degen["case_number"],
            "case_type": degen["case_type"],
            "reason": "degenerate_text_flagged",
            "detail": "full_text under "
                      + str(MIN_USABLE_CHARS) + " chars: "
                      + degen["full_text"].str.len().astype(str),
        }))

    collide = df[df["case_number_collision"]]
    if len(collide):
        frames.append(pd.DataFrame({
            "case_number": collide["case_number"],
            "case_type": collide["case_type"],
            "reason": "case_number_collision_kept",
            "detail": "different document sharing this case number; both retained",
        }))

    if not frames:
        return pd.DataFrame(columns=["case_number", "case_type", "reason", "detail"])
    return pd.concat(frames, ignore_index=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only the rows flagged as true duplicates."""
    return df[~df["is_duplicate"]].reset_index(drop=True)
