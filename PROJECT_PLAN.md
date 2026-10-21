# Project Plan — SEAD-4 Adjudication Analyzer

**Break Through Tech AI Studio, Fall 2026 · A3 Consulting challenge project**

This is our team's plan for the Challenge Project *Explainable AI for Security Clearance Decisions*.
It records the target we chose, the measurements that justify it, and what we are building in October
and November. Every number below was measured directly against `Challenge_Data/` (the corpus-size chain
in §8 also uses the source `doha` files); the commands to reproduce them are in §10.

---

## 1. What we are building

> **Given the facts of an undecided case, how will each SEAD-4 guideline be resolved, why, and what
> did judges do in the most similar past cases?**

Four outputs from one pipeline:

| # | Output | How |
|---|---|---|
| 1 | **Routing** — which of the 13 guidelines are at issue | Multi-label classifier over A–M |
| 2 | **Resolution** — for each, the likely finding FOR or AGAINST the applicant | Same model, calibrated |
| 3 | **Reasoning** — which specific conditions the judge will credit, with the text they rest on | Rule-based + LLM extraction of `AG ¶` citations, with character offsets |
| 4 | **Precedent** — the most similar decided cases and how they came out | Hybrid BM25 + dense retrieval over findings-of-fact |

Delivered as a Streamlit app that accepts a decision as PDF or text, plus a structured SEAD-4 JSON
document carrying provenance on every claim.

---

## 2. The target, and why

Our prediction target is **the judge's per-guideline formal finding** — FOR or AGAINST the applicant on
each guideline at issue. The cleaned dataset carries roughly **29,600 of these (guideline, direction)
pairs**, taken from the judges' own written rulings.

Three reasons this is the right target:

**It is the unit an adjudicator reasons in.** A single case can be resolved FOR the applicant on
Guideline E and AGAINST on Guideline F. One GRANTED/DENIED bit cannot express that; a per-guideline
finding can.

**It subsumes overall outcome prediction.** We tested the rule *"any guideline AGAINST ⇒ case DENIED"*
across the 20,236 DENIED/GRANTED cases that carry at least one directional finding: it holds **99.28%** of the
time (145 exceptions). So the case recommendation composes from the per-guideline heads rather than
needing a separate model. We will report both the composed recommendation and a directly-trained
outcome head, and treat the comparison as a result. The 145 exceptions are interesting in themselves —
79 are DENIED with all-FOR findings, which points at a non-guideline statutory bar such as the Bond
Amendment, or a finding the parser missed.

**It subsumes the multi-label A–M task.** Collapsing {FOR, AGAINST} into "at issue" gives exactly the
multi-label guideline classification the brief specifies, at no additional cost. One model, three of
the brief's four outputs.

A fourth reason, less about the model and more about having something to say: **direction rates differ
sharply by guideline, and nobody has reported it.** Guideline B (Foreign Influence) is resolved against
the applicant 37.3% of the time; Guideline D (Sexual Behavior) 71.5%; Guideline F 64.3%. Guidelines are
mitigated at very different rates.

For contrast, standalone case-outcome prediction is close to exhausted — 82.5% accuracy against a 64.3%
majority baseline (the DENIED share of that experiment's test split) using plain TF-IDF on leak-free text,
measured during data cleaning and since reproduced at 82.7%. That becomes our baseline chapter, not our
headline.

---

## 3. Scope: all 13 guidelines, depth on Guideline F

We classify all 13 guidelines, but we **pre-register two reporting tiers**, because the labels do not
honestly support uniform claims. Counts are directional findings in the trainable pool:

| Tier | Guidelines | Reporting |
|---|---|---|
| **Defensible** | F 12,353 · E 5,906 · B 2,667 · H 2,667 · J 2,415 · G 1,467 · C 924 · D 386 | Headline per-label precision/recall/F1; macro-F1 computed over these 8 |
| **Reported with caveat** | I 186 · K 207 · M 202 · L 29 · A 3 | Counts printed beside every metric; never quoted without them |

Guideline A is the clearest case for why this matters. It appears on 960 rows of the cleaned dataset,
but **99.6% of those labels come from keyword matching rather than the judge's ruling**. Restricted to
judge-confirmed labels, Guideline A has **4 cases**. A per-label F1 for Guideline A would be noise
presented as a measurement.

All depth work — condition extraction, provenance, precedent retrieval, the demo's worked example —
targets **Guideline F, Financial Considerations**: 13,119 cases with F at issue (12,353 of them with a
FOR/AGAINST finding), the largest population and the richest condition structure.

**Out of scope, deliberately:** appeals as a modeling target. Appeal Board decisions contain no formal
findings section, so only 0.15% of them (7 of 4,763) carry the labels this approach depends on, and the trainable pool
is 21,952 hearings to 7 appeals. Appeals remain in the dataset and in the precedent index; they are not
a prediction target.

---

## 4. What the data supports

Measured against `Challenge_Data/` (30,517 rows × 66 columns).

### The trainable pool

Filtering to `use_for_training & (label_source == "formal_ruling") & (len(text_features) > 0)` gives
**21,959 rows** — 21,952 hearings and 7 appeals, across 21,928 distinct case numbers.

| Split | Count |
|---|---:|
| `guideline_regime == "sead4"` | 17,428 |
| `guideline_regime == "directive"` | 2,629 |
| `guideline_regime == "unknown"` | 1,902 |
| Outcome DENIED / GRANTED / REVOKED / unresolved | 13,882 / 8,039 / 14 / 24 |

The 24 unresolved rows keep their guideline labels but are left out of outcome metrics.

Metrics must be reported **split by regime**. `sead4` means the `AG ¶` numbering shared by the 2006
Adjudicative Guidelines and SEAD-4 — 65% of these cases predate SEAD-4's June 2017 effective date — and
`directive` means the pre-2006 Directive 5220.6 numbering (`E2.A…`). They are different legal standards
with incompatible citation schemes; averaging them averages incomparable things.

### Document lengths — this decides the model

| percentile | characters | ≈ tokens (characters ÷ 4) |
|---|---:|---:|
| 50th | 6,998 | 1,750 |
| 75th | 9,900 | 2,475 |
| 95th | 17,362 | 4,340 |
| max | 78,398 | 19,600 |

**98.0% of documents exceed 512 tokens. Only 0.3% exceed 8,192.** A 512-token encoder such as
Legal-BERT would truncate essentially every document to its first ~2,000 characters. A long-context
encoder such as ModernBERT-base (8,192 tokens) fits the corpus nearly untruncated; on our hardware (free
Colab, an 8 GB laptop GPU) we cap it at 4,096 tokens, which still keeps ~94% of documents whole (§6 October). We will run both, because
domain pre-training versus seeing the whole document is a real trade-off and which one wins here is worth
reporting.

### Guideline F conditions — the extraction target

Nothing in the pipeline currently captures `AG ¶` paragraph citations. `src/labels.py` matches `AG ¶ 19`
only to recover the guideline letter "F" and discards the subparagraph — but the subparagraph is the
part that names the specific condition. Measured over the 13,119 Guideline F cases in the pool:

| Citation | Cases | Share |
|---|---:|---:|
| Any condition cited | 10,212 | 77.8% |
| A **mitigating** condition cited (`AG ¶ 20(x)`) | 8,587 | 65.5% |
| A **disqualifying** condition cited (`AG ¶ 19(x)`) | 6,173 | 47.1% |
| Both | 5,283 | 40.3% |
| Nothing cited | 2,907 | 22.2% |

Mitigator frequencies are workable for a five-class model: `20(a)` 6,855 · `20(b)` 6,663 ·
`20(c)` 5,075 · `20(d)` 4,882 · `20(e)` 2,842. (`20(g)` 767 and `20(f)` 275 are reported only.)

Coverage depends entirely on the citation regime, and **both schemes must be implemented**:

| regime | F cases | extractable | scheme |
|---|---:|---:|---|
| `sead4` | 11,251 | **9,415 (83.7%)** | `AG ¶ 19(x)` / `AG ¶ 20(x)` |
| `directive` | 937 | **735 (78.4%)** | `E2.A6.1.2.x` / `E2.A6.1.3.x` |
| `unknown` | 931 | **62 (6.7%)** with these patterns | short forms — see below |

The two regimes are cleanly disjoint: zero `sead4` F cases cite `E2.A6`, and zero `directive` F cases
cite `AG ¶`. Implementing only the SEAD-4 patterns would silently drop 937 cases. `E2.A6` is verified as
the Directive-era Guideline F citation — it appears in only 0.30% of a 3,000-case sample of non-F
decisions.

The `unknown` cases are not citation-free. In their analysis and conclusions, 593 of the 931 (64%) cite a
condition in a short form the patterns above miss — bare `20(a)` / `19(a)`, `¶ 20(a)` without `AG`,
`MC 20(a)`, `DC 1` / `MC 3`, `FC MC 1`, or `Disqualifying Condition 1` — so `src/conditions.py` covers
them too. `DC n` / `MC n` don't name their guideline, so they are tied to the F discussion by position and
hand-checked before any coverage is claimed. The remaining 338 (36%) are a stated limitation.

### Evidence and retrieval assets

`para_offsets_fof` holds pre-computed, verdict-free paragraph boundaries inside FINDINGS OF FACT:
present on **95.2% of the trainable pool and 94.7% of Guideline F cases**, mean 8.3 paragraphs per case.
These are ready-made units for both evidence provenance and retrieval chunking.

For the precedent index, the usable corpus is **wider than the trainable pool** — retrieval needs
non-empty `text_features`, a resolved outcome (DENIED, GRANTED or REVOKED) and findings-of-fact
paragraphs to index, not clean guideline labels. **24,238 cases** qualify.

| granularity | vectors | size @ 384-dim float32 |
|---|---:|---:|
| **case-level (chosen)** | **24,238** | **37 MB** |
| paragraph-level | 187,691 | 288 MB |

Case-level keeps this inside free-tier compute. We highlight the best-matching paragraph *within* a
retrieved case rather than indexing every paragraph. Note 3,261 of those cases carry `regex_fallback`
guideline labels — fine for retrieval, which matches on text and shows the independently-resolved
outcome, but precedent cards must display `label_source` so a reader knows when a hit's guideline tags
are the weaker kind.

---

## 5. Data issues to fix first

Four in the dataset, four in the EDA script. All measured; each has a short fix.

### In `Challenge_Data/`

**`guideline_findings` struct keys differ across shards.** pyarrow inferred the struct schema per output
shard, so `cases_clean_1–3` carry 12 keys (B–M) and `cases_clean_4` carries 13 (A–M). After
`pd.concat`, 22,890 rows have no `A` key at all and 7,627 do — so `d.get("A")` means "key absent" or
"value null" depending only on which file a row came from. Most likely no findings were lost, since
Guideline A has ~3 corpus-wide and they land in shard 4, but the schema is not self-consistent. Fix in
the writer in `src/build_dataset.py` with an explicit A–M schema, then rebuild (~2 min); a loader-side
patch would leave the files wrong. Until that rebuild, `load_cases()` normalizes every row to A–M on load.

**Missing values are empty strings, not nulls.** `text_features` has **0 nulls and 4,199 empty
strings**. Any filter written as `.notna()` therefore excludes nothing. Use `.str.len() > 0`.

**Direction casing is inconsistent.** `guideline_findings` uses `FOR`/`AGAINST`;
`guideline_subparagraphs[i]["finding"]` uses `For`/`Against`. Normalize on load.

**`case_year` is not the decision year.** It is the docket-number prefix and agrees with the year of
`date` only 29.5% of the time (76% within one year). Use `date` for anything temporal.

### In `notebooks/03_exploratory_data_analysis.py`

**1. The text-length statistics need the empty-string filter.** `analyzeTextLengths` filters
`text_features` with `.notna()`, which excludes nothing:

| statistic | as filtered | non-empty only |
|---|---:|---:|
| n | 30,517 | **26,318** |
| min | 0 | **348** |
| mean | 6,651 | **7,712** |
| median | 5,974 | **6,652** |

**2. The guideline-frequency chart mixes label tiers.** `dataFrame["guidelines"].explode()` runs across
all 30,517 rows, so `formal_ruling` labels are pooled with the inherited tier and the ~60%-precision
`regex_fallback` tier that `Challenge_Data/README.md` flags as not comparable. The effect is large and
uneven — Guideline A is 99.6% keyword-tier; J is 40% and L is 44% not judge-ruled (keyword plus
inherited). Either plot per tier or filter and label the chart accordingly.

**3. Dependencies were not pinned.** The script needs `matplotlib` and `seaborn`, and `requirements.txt`
was empty. It now pins pandas, pyarrow, numpy, scikit-learn, matplotlib and seaborn.

**4. The same sentinel problem elsewhere, and the charts inherit it.** `analyzeDataQuality` counts
missing `text_features` and `guidelines` with `.isna()`, so it reports 0 for both — really 4,199 empty
texts and 115 empty guideline lists — and its training-ready count omits the empty-text filter (22,105,
not 21,959). `loadData` reads all 66 columns. Of the committed charts, `guideline_analysis.png` mixes
label tiers, `text_length_analysis.png` counts empty texts as length 0, and `outcome_distribution.png` has
the right counts but its pie silently drops the 888 unresolved outcomes and its small-slice labels
overlap. All three need regenerating.

**The shared fix for all of these is one loader.** `src/dataset.py` exposes
`load_cases(pool=True, columns=None)`: it normalizes the dataset issues on load and applies the pool filter,
and `pool=False` returns every row for EDA. The EDA script will be refactored onto it. Its default
`columns` excludes `full_text` — loading all 66 columns of all four shards pulls multiple GB into memory,
which is why every measurement in this document was taken with a column subset.

---

## 6. The work: October and November

We have two months, October and November, and four people: Rashed, Aryan, Brian and Sam. Every task
below has an owner and a due date, and nothing is due before the work it depends on.

Check-ins fall in the 2nd and 4th week of each month (Oct 8–14, Oct 22–28, Nov 8–14, Nov 22–28), so each
check-in's showcase items are due just before it; if a check-in falls early in its week, move that week's
dates up. Thanksgiving is Thursday Nov 26, so the final deliverables are due Tuesday Nov 24, with Nov 25–30
as buffer. All LLM calls are Rashed's, so prompts, credentials and spending stay in one place; the app shows
those outputs from his cached runs.

### Schedule

| Due | Owner | Deliverable | Needs first |
|---|---|---|---|
| Oct 11 (done) | Rashed | Loader `src/dataset.py`, pinned `requirements.txt`, this plan | — |
| Tue Oct 13 | Sam | Leak-free baseline committed: the existing TF-IDF model re-run on `text_features` with a grouped split | Loader |
| Wed Oct 14 (done) | Aryan | `Challenge_Data/splits.parquet`, plus split checks in `src/validate.py` | Loader |
| Fri Oct 16 | Brian | EDA fixes (§5) and `notebooks/EDA_REPORT.md` | Loader |
| Fri Oct 16 | Sam | Baseline re-run on the frozen split | Split |
| Fri Oct 16 | Rashed | Laptop GPU ready: CUDA 12.8+ PyTorch, local LLM installed | — |
| Tue Oct 20 | Aryan | `Challenge_Data/DATA_CARD.md` | EDA report |
| Tue Oct 20 | Brian | `src/sead4_reference.py`, the closed list of condition codes | — |
| Fri Oct 23 | Everyone | Label check: 50 cases, ~13 each (§7) | — |
| Fri Oct 23 | Sam | 13-head TF-IDF baseline (per-guideline FOR/AGAINST) | Split, baseline |
| Fri Oct 23 | Rashed | LLM prompt templates and `src/llm/verify.py` | Code list |
| Tue Oct 27 | Brian | Rule-based extractor `src/conditions.py`, with negative tests | Code list |
| Tue Oct 27 | Aryan | Precedent index v1: embeddings + FAISS, smoke-tested | Loader |
| Fri Oct 30 | Rashed | LLM pilot: ~50 cited cases, scored against the rule-based extractor | Templates, extractor |
| Fri Oct 30 | Sam | First ModernBERT run (2,048-token cap) | 13-head baseline |
| Fri Nov 6 | Rashed | LLM run on the test split's uncited F cases; `src/schema.py` | Pilot, split, extractor |
| Fri Nov 6 | Sam | ModernBERT at 4,096 tokens and the Legal-BERT comparison; per-head calibration; composed-vs-direct outcome | First run |
| Fri Nov 6 | Aryan | Hybrid retrieval (BM25 + dense, RRF); the 50-query set with relevance defined | Index v1 |
| Fri Nov 6 | Brian | Dashboard | Loader |
| Fri Nov 6 | Everyone | Traceability check: ~25 findings each (§7) | Extractor, pilot |
| Wed Nov 11 | Everyone | Retrieval relevance labels: ~13 queries each (§7) | Query set |
| Fri Nov 13 | Rashed | LLM layer, part two: similarity notes and narrative rationale | Retrieval, classifier |
| Fri Nov 13 | Sam | Mitigator head; metrics against the judge-ruled labels by regime, with the label check's accuracy | Extractor, classifier, label check |
| Fri Nov 13 | Aryan | precision@5 and MRR | Relevance labels |
| Fri Nov 13 | Brian | Analyzer app shell running on cached outputs | `schema.py` |
| Wed Nov 18 | Rashed | Pipeline integration, plus cached outputs for the demo | Classifier, extraction, retrieval, LLM layer |
| Fri Nov 20 | Brian | Analyzer wired to the pipeline, end to end from a raw PDF | Integration |
| Fri Nov 20 | Sam | Evaluation write-up, model card, README draft | All metrics |
| Fri Nov 20 | Aryan | CI: `pytest`, pre-commit and GitHub Actions running the tests and `src.validate` | — |
| Tue Nov 24 | Everyone | Final README, demo rehearsal, final presentation | Everything above |

**Two chains decide the final date.** Extraction: code list (Oct 20) → extractor (Oct 27) → LLM pilot
(Oct 30) → LLM run (Nov 6). Retrieval: index (Oct 27) → hybrid retrieval (Nov 6) → similarity notes
(Nov 13). Both feed integration (Nov 18) → the wired app (Nov 20) → the final week (Nov 24). A slip on
either chain moves everything after it, so tasks on them get help first.

**If something has to give,** cut in this order: the Legal-BERT comparison, the local-vs-Claude comparison,
then the stretch goals. The heaviest week is Sam's, Oct 30 to Nov 6.

### Foundations — due by October 20

**Goal: a dataset everyone loads the same way, a frozen split, a committed leak-free baseline and an
honest EDA report.** These are the brief's September items; they come first because everything else
builds on them.

| Deliverable | Owner | Due | Notes |
|---|---|---|---|
| `src/dataset.py` | Rashed | done | The single loader described above |
| `PROJECT_PLAN.md` | Rashed | done | This document |
| Baseline | Sam | Oct 13; on the split Oct 16 | The TF-IDF + logistic-regression model is already trained: `notebooks/02_baseline.py`. As committed it trains on `full_text` with a random split — 95.4%, leaky. The fix is a re-run of the same model with three changes — `load_cases()`, `text_features`, a grouped split — which reproduces at 82.7% vs 64.3% |
| `Challenge_Data/splits.parquet` | Aryan | done | Frozen train/val/test, committed |
| `notebooks/EDA_REPORT.md` | Brian | Oct 16 | Written EDA report with corrected figures |
| `Challenge_Data/DATA_CARD.md` | Aryan | Oct 20 | Data dictionary, per-tier counts, known gaps, ethics note |

**On the split:** use `StratifiedGroupKFold`, not `GroupShuffleSplit` — the latter takes no `stratify`
argument, and we need grouping and stratification together. Group on `split_group` (= `case_number`) so
a hearing and its own appeal cannot land on opposite sides of a boundary. Stratify on **the rarest
guideline present in each case**: tested, that gives 13 strata with a minimum of 4. Guideline A's 4 cases
can't form a stratum of their own, since it would be smaller than the fold count, but all four carry a
second guideline (E, B, M, E), so they are stratified on that instead; the smallest stratum is then L,
with 40. Do *not* stratify on `(guideline, regime)` — tested, 37 strata of which 4 are smaller than the
fold count — and do not stratify on the exact guideline-presence pattern, which produces strata of size 1.
Check regime balance afterwards instead; at 17,428 / 2,629 / 1,902 it balances on its own. We considered
iterative stratification for this multi-label split and kept the rarest-guideline method. Iterative
stratification can't keep a hearing and its appeal together, and our split already holds every guideline
within about one point of its pool rate in all three parts.

`StratifiedGroupKFold` makes k equal folds, so 70/15/15 takes two stages: `n_splits=7` (shuffled, seed 42)
with one fold as test, then `n_splits=6` on the remainder with one fold as validation — about 71/14/14.

**Done when (Oct 20):** `python -m src.validate` passes with the new split assertions, `splits.parquet` is
committed, the leak-free baseline is committed, and the EDA report carries the corrected figures.

---

### October — Guideline Classification & Extraction (through Oct 30)

**Goal: a validated multi-label A–M classifier, disqualifier/mitigator extraction built two ways, and a
precedent index under construction.**

**1. The classifier — 13 three-way heads.** Per guideline, one 3-class head: *not at issue / FOR /
AGAINST*. Collapsing {FOR, AGAINST} gives the multi-label A–M classifier, which we report as such.

- Baseline: TF-IDF + one-vs-rest logistic regression, 13 × 3-class, extending `02_baseline.py`. This has
  to be beaten before a transformer is justified.
- Main: fine-tuned ModernBERT-base on `text_features`, with Legal-BERT at 512 tokens as the comparison.
- Compute: fine-tuning runs on free Colab (T4, 16 GB). Rashed's RTX 5060 laptop GPU (8 GB) runs the local
  LLM — template testing, the pilot and the run — from Oct 19 to Nov 6, and is free for training outside
  that window. Iterate at a
  2,048-token cap and run the final model at 4,096, which keeps ~94% of documents whole (estimated at
  characters ÷ 4 — confirm with the real tokenizer); bf16, gradient checkpointing, small batches with
  accumulation.
- Missing directions: 1,823 pool cases (8.3%) list guidelines with no FOR/AGAINST at all, and 223 more
  lack it on some guidelines — 3,234 (case, guideline) pairs in all. Those pairs are masked out of the
  direction loss, not treated as *not at issue*.
- Imbalance: per-head class weights.
- Calibration: temperature scaling **per head**, not one global temperature — a head trained on 12,353
  Guideline F findings is not miscalibrated the same way as one trained on 386 Guideline D findings.
  Reliability curve per defensible guideline plus the pooled curve.
- Decision thresholds: tune each head's cutoff on the validation set instead of taking the most likely
  class, and report precision-recall curves. Taking the most likely class under-predicts the rare
  direction on small heads such as D, I, K and M.
- Case recommendation: compose via "any AGAINST ⇒ DENIED" *and* train a direct head; report both.
- Metrics: Accuracy, Macro-F1, Micro-F1, per-label precision/recall/F1 and average precision for all 13
  with counts shown, AUC where applicable, **split by regime**. Within each guideline, report precision
  and recall for FOR and for AGAINST separately.
- Routing baseline: the statement of the case usually names the alleged guidelines, so a bare keyword
  match on `text_features` already gets 0.90–0.99 precision and ~0.8 recall (B/E/F/G/H/J). Report it
  beside the model — direction, not routing, is the real test.

**2. Extraction, channel one — rule-based.** `src/conditions.py`, a deterministic extractor over
`sec_analysis_*` ∪ `sec_conclusions_*` covering `AG ¶ 19(a)–(i)`, `AG ¶ 20(a)–(g)` and Directive-era
`E2.A6.1.2.x` / `E2.A6.1.3.x`, each with a character offset into `full_text`. Feeds a mitigator
prediction head (5 classes) that reuses the classifier above, and `src/schema.py`, the structured
SEAD-4 JSON with offsets on every finding.

> The extractor reads post-verdict sections, so it produces **labels, never features**. This distinction
> has to be explicit in the write-up — it is the line between an honest result and a leaked one.

**3. Extraction, channel two — LLM prompt templates.** The rule-based channel only fires where the judge
cited a paragraph number, which leaves 2,907 F cases citing nothing and 4,532 citing no `AG ¶ 20`
mitigator (that count includes Directive-era cases that cite `E2.A6` mitigators instead). That gap is the
LLM's job.

- **Model and runs (Rashed):** the default is an open-source model run locally — a 7–8B instruct model,
  4-bit, e.g. via Ollama, on the 8 GB laptop GPU — so the run needs no key and no spending. Prompts are
  drafted and hand-checked in Claude on a Pro subscription first; it has no API key, so it can't run a
  script. The pilot (Oct 30) runs ~50 cited cases and scores them against the rule-based channel; the run
  (Nov 6) covers the test split's uncited F cases (~650) plus a validation sample of cited ones. If the
  advisor covers a small API credit (§8, question 4), the same pilot also runs on Claude Haiku 5.5 through
  the Batch API (under $1 at this scope), and whichever model scores better does the run.
- `src/sead4_reference.py` — the closed list of 142 `AG ¶` condition codes with their paragraph text, so
  the model selects from a fixed set and cannot invent a code. Vendored, with attribution, from
  `sead4_llm/config/guidelines.py` in the `doha` repo (§8, question 1, answered).
- `src/llm/templates.py` — templates returning strict JSON `{condition_code, exact_quote}`.
- `src/llm/verify.py` — runs `full_text.find(quote)` on every returned quote and **drops any that does
  not resolve exactly**. We never ask the model for character offsets; we recover them ourselves from
  the verified quote.
- **Validation without hand-labeling:** on the cases where the judge *did* cite, the rule-based channel
  is ground truth for the LLM channel. Measure precision/recall there, publish it, then apply the LLM to
  the uncited cases with that error rate attached.
- **Hard boundary:** the LLM extracts and quotes. It never emits a classification, a relevance score or
  a confidence value — those come from calibrated model outputs.

Where a regex beats a language model at legal citation extraction, and where it loses, is a result worth
presenting.

**4. Start the precedent index.** Embeddings plus vector store, built and smoke-tested this month;
evaluation follows in November. `sentence-transformers` for embeddings, FAISS for the store, case-level
over FINDINGS OF FACT paragraphs. Never index `full_text` — retrieving a verdict-bearing passage would
leak the answer into the demo.

**5. Label check — due Oct 23.** 50 pool cases, about 13 each. See §7.

**Done when (Oct 30):** the 13-head TF-IDF baseline and a first ModernBERT run, each in a 13-label table
with tier caveats; rule-based extractor validated on a hand-checked sample; LLM pilot scored against it;
index returns sensible neighbours; 50-case label check done. Calibration curves, the Legal-BERT comparison
and the composed-vs-direct outcome comparison follow by Nov 6.

---

### November — End-to-End Analyzer & Demo (through Nov 24)

**Goal: everything wired into one app that takes a PDF and returns a traceable assessment, with every
number tied to a hand-checked sample.**

**1. Finish retrieval.** Hybrid: dense (`bge-small-en-v1.5` or `all-mpnet-base-v2`) plus **BM25**, fused
with Reciprocal Rank Fusion. Dense-only search over general-purpose embeddings misses the exact `AG ¶`
and docket-number matches that legal search depends on. Evaluate precision@5 and MRR on 50 stratified
queries, with relevance defined *before* looking at results as sharing the query's guideline set and
direction.

**2. LLM layer, part two (Rashed).** Two more templates, extending October's `src/llm/`: short **similarity
notes** per retrieved precedent (the brief asks for these by name) and a **narrative rationale** built
only from findings the classifier and extractor already established. Same guardrails, unchanged.

**3. Streamlit app and dashboard.**

- *Analyzer:* upload a PDF or paste text → routed guidelines, per-guideline FOR/AGAINST with
  calibrated confidence, composed case recommendation, conditions with the cited span highlighted in the
  source, similar precedents with scores and notes, and the structured JSON downloadable. PDFs are read
  with the same PyMuPDF call that built `full_text` (`"\n".join(page.get_text() for page in doc)`); on two
  real decisions this reproduces the stored text, sections, labels and `text_features` exactly.
- *Dashboard:* outcome × guideline × year (from `date`) × regime; direction rate per guideline;
  label-tier coverage; and the cleaning before-and-after. Built on `src/dataset.py` so it cannot repeat
  the tier-mixing issue.
- Ships with cached model outputs — no API key and no GPU required to run the demo. For a newly uploaded
  PDF, the classifier and the rule-based channel run offline; the LLM channel runs if the local model is
  installed.

**4. Evaluation and portfolio.** Report every metric against the judge-ruled labels, with the label
check's measured accuracy beside it. Traceability pass rate and retrieval relevance (§7). Rewrite
`README.md` — it is still the unedited Break Through Tech template and it is the graded portfolio
artifact. Keep `requirements.txt` pinned as dependencies are added. Add `pytest`, pre-commit hooks and
GitHub Actions CI running `tests/test_units.py` and `python -m src.validate`. Model card. Final
presentation.

New tests must include **negative cases**. `tests/test_units.py` exists precisely because the upstream
repo's 147 regex tests all assert `should_match=True`, which is how two label bugs shipped undetected.

**At each check-in:** Oct 8–14 the leak-free baseline and the split; Oct 22–28 the 13-head baseline, the
extractor and index v1; Nov 8–14 retrieval, the calibrated classifier and the LLM layer; Nov 22–28 the full
app and the presentation. Each item's date is in the schedule above.

**Done when (Nov 24):** the app runs end to end from a raw PDF, offline; every displayed span resolves in
`full_text`; metrics carry the label check's measured accuracy; README is no longer a template; CI is
green.

---

### Stretch goals, if the above is finished

1. **Temporal consistency audit** — the brief asks whether similar fact patterns across years produce
   consistent outputs. Cluster by fact-pattern similarity using the retrieval index, then check whether
   outcome and direction rates within a cluster drift by decision year. Split by `guideline_regime`: the
   2006 move from Directive 5220.6 to the Adjudicative Guidelines (and to SEAD-4 in 2017) is a real change
   in the legal standard and would otherwise read as model inconsistency. Do this one first.
2. **Counterfactual judge-name swap** — if a prediction flips when only the judge's name changes, the
   model learned a spurious correlation.
3. **FCRA / policy-authority citation layer** — rule-based: a credit-report-based `AG ¶ 19(c)` finding
   surfaces the governing FCRA sections.
4. **`cleanlab` confident learning** to recover usable labels from the 6,142 `regex_fallback` rows.
5. **Fine-tuning a compact open LLM** is the brief's fourth stretch goal. We are consciously not
   pursuing it, so that the write-up records a decision rather than an omission.

---

## 7. Evaluation

The dataset's own README states that `formal_ruling` label precision **is not established**, and that
validating it against the source's `formal_findings` column is circular because both come from nearly
the same regex. Against an independent signal it achieves 98.0% recall, but precision awaits a
hand-verified sample.

The brief requires manual checking in two places — retrieval ("precision@k, MRR on a manually labeled
sample") and explainability ("manual traceability pass rate on a random sample"). We add one small check
on the labels themselves and stop there: three focused checks, about six hours across the team in total,
rather than a full gold set.

| Check | What a person does | When |
|---|---|---|
| **Label check** — 50 pool cases, ~13 each | Read the decision's FORMAL FINDINGS and confirm each stored guideline and FOR/AGAINST, noting any guideline the labels missed | Due Oct 23 |
| **Traceability** — ~100 extracted findings | Confirm each cited span supports the `AG ¶` condition it is attached to | Due Nov 6, after the extractor (Oct 27) and the LLM pilot (Oct 30) |
| **Retrieval relevance** — 50 queries × top 5 | Mark each precedent relevant or not, against a definition fixed before looking at results | Due Nov 11, after the query set (Nov 6) |

**What 50 cases buys, stated plainly:** the judge-ruled labels' accuracy to roughly ±6 points (95% CI, if
it is near 95%), corpus-wide. Not per guideline, and not the Guideline F conditions — the LLM channel is
scored against the rule-based channel instead (§6 October, item 3). Every model metric is reported against
the judge-ruled labels with this measured accuracy beside it.

The **traceability pass rate** is what fraction of cited spans resolve to the correct text and the correct
AG paragraph. A number, not a paragraph of reassurance.

**Start the label check now.** It needs nothing else to exist, and it is the task that silently does not
get done if left until November.

---

## 8. Open questions for our Challenge Advisor

1. **May we vendor `sead4_llm/config/guidelines.py` into this repo, with attribution?** It holds all 13
   guidelines with 142 `AG ¶` codes and full paragraph text — exactly the closed list our prompt
   templates and UI need, and the brief points at `sead4_llm/` for it. There is no LICENSE file in the
   `doha` repo, so we would rather ask than assume. If preferred, we can regenerate it from the official
   SEAD-4 PDF instead. **Answered (Oct 20):** yes. Both repos are ours to use, so we vendor it with
   attribution instead of regenerating it from the PDF.
2. **Is restricting the modeling target to hearing decisions acceptable?** Appeal Board decisions carry
   no formal findings section, so only 0.15% (7 of 4,763) have judge-written per-guideline labels. Appeals stay in the
   dataset and the precedent index, but not as a prediction target.
3. **Does the two-tier reporting approach work for you?** We would rather publish 8 guidelines with
   defensible per-label metrics and 5 clearly marked as under-supported than present 13 numbers of
   uneven quality.
4. **Can the program cover a small Claude API credit?** Optional: our default is an open-source model run
   locally. A credit would let us compare it against Claude Haiku 5.5 on the same pilot, which costs under
   $1 at our scope (§6 October, item 3). A Claude subscription doesn't include API access.

### One clarification on corpus size

The brief describes ~36,700 documents and we will be reporting 30,517, so the chain is worth recording.
The **36,703** collected links (that is what the ~36,700 figure counts) point to **33,477** distinct
documents — 3,226 links repeat one already listed — and **709** of those were never collected. The two
source shards hold **33,610** rows covering 32,768 documents (842 rows repeat a URL). Removing **3,093**
byte-identical rows — those 842 repeats, plus 2,251 rows whose URL differs but whose text is another
case's, which is what appeals overwriting one another's PDFs (they share filenames) looks like — leaves
**30,517**.

Deleting identical copies discarded nothing, but about **2,960** real decisions are missing from the
corpus: 709 never collected and 2,251 overwritten. (`failed_cases.json` lists 4,286 failures, but it is a
stale log — every one of those URLs is present in the source shards.)

Relatedly, the brief describes the shards as containing sectioned text. In the raw parquet those columns
are largely empty (`sor_allegations` 3.6%, `judges_analysis` 6.9% of the 33,610 rows), which is why our
cleaning pipeline replaced them with character offsets computed from the document structure.

---

## 9. How this maps to the Challenge Project Overview

| Month | Key activity from the brief | Where |
|---|---|---|
| Sep | Load committed Parquet shards | Done — `src/load.py` |
| Sep | EDA: outcome distribution | `analyzeOutcomes` |
| Sep | EDA: guideline frequency | `analyzeGuidelines` (see §5) |
| Sep | EDA: text lengths | `analyzeTextLengths` (see §5) |
| Sep | TF-IDF + LogReg baseline for outcome prediction | `notebooks/02_baseline.py` — leak-free version due Oct 13 (§6 Foundations) |
| Sep | Short EDA report | `notebooks/EDA_REPORT.md` |
| Sep | Reproducible train/val/test split | `Challenge_Data/splits.parquet` |
| Oct | Multi-label classifier for guidelines A–M | §6 October, item 1 |
| Oct | LLM prompt templates for extracting disqualifiers/mitigators | §6 October, item 3 |
| Oct | Rule-based heuristics for the same | §6 October, item 2 |
| Oct | Start a RAG index (embeddings + vector store) | §6 October, item 4 |
| Nov | Integrate classifier + extraction + RAG into one pipeline | §6 November, item 3 |
| Nov | Structured SEAD-4 JSON with provenance | `src/schema.py` |
| Nov | Streamlit demo UI | §6 November, item 3 |
| Nov | Quantitative evaluation | §7 |
| Nov | Manual qualitative evaluation | §7 — label check, traceability pass rate, retrieval relevance |
| Nov | Finalize README and presentation | §6 November, item 4 |

The brief's September items are our Foundations block, due by Oct 20 (§6).

| Success criterion | Where |
|---|---|
| Accuracy, Macro/Micro F1, per-class P/R, AUC | §6 October, item 1 |
| Per-label P/R/F1 and average precision for A–M | §3 two-tier reporting |
| precision@k, MRR on a manually labeled sample | §6 November, item 1 |
| Extracted conditions traceable to text spans or SEAD-4 paragraphs | §6 October, items 2–3 |
| Provenance for LLM outputs, no obvious hallucination | `src/llm/verify.py` — unverified quotes dropped, never displayed |
| Manual traceability pass rate | §7 |
| Demo accepts PDF/text, outputs recommendation + confidence, per-guideline assessments, cited evidence, precedents | §6 November, item 3 |
| Calibration analysis, including per-label | §6 October, item 1 — treated as core, not stretch |
| Automated explanation-traceability evaluation | §7 — treated as core, not stretch |

Recommended stack, all used: `pandas`, `pyarrow`, `PyMuPDF`, `scikit-learn`, `transformers`, `PyTorch`,
`sentence-transformers`, `faiss`, an open-source LLM run locally (Claude through the API as an optional
comparison), `streamlit`, `pytest`, pre-commit, GitHub Actions. The brief also suggests small open models
for reproducibility — running the extraction channel on one by default means the write-up reports a fully
reproducible, zero-key version.

---

## 10. Reproducing the numbers in this document

Every figure above came from the committed parquet. To check any of them:

```python
import glob, pandas as pd

files = sorted(glob.glob("Challenge_Data/cases_clean_*.parquet"))

# Always pass columns= . Loading all 66 columns of all four shards is multiple GB.
cols = ["case_number", "case_type", "label_source", "guidelines",
        "guideline_findings", "text_features", "outcome",
        "guideline_regime", "use_for_training", "split_group"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)

# guideline_findings struct keys differ per shard (see section 5) - normalize first.
def directions(d):
    if d is None:
        return {}
    return {k: (v.upper() if isinstance(v, str) else None) for k, v in dict(d).items()}

df["gf"] = df["guideline_findings"].map(directions)

# The trainable pool: 21,959 rows.
pool = df[df.use_for_training
          & (df.label_source == "formal_ruling")
          & (df.text_features.str.len() > 0)]
```

Or use the loader, which does the normalization and the pool filter: `from src.dataset import load_cases;
pool = load_cases()` (21,959 rows).

The frozen split is `Challenge_Data/splits.parquet` (train 15,685 / val 3,137 / test 3,137, from
`python -m src.build_splits`). `load_cases()` attaches it as a `split` column, so filter on that rather
than re-splitting, and results stay comparable across the team.

---

## 11. Verification checklist

Anything that would invalidate a result, checked as a test rather than by eye:

1. `python -m src.validate` — existing checks plus split and condition assertions.
2. `python tests/test_units.py` — including negative cases for the new `AG ¶ 19(x)` / `20(x)` and
   Directive-era patterns: they must not match bare `AG ¶ 19`, `AG ¶ 190`, or a POLICIES recitation.
3. **Leakage** — retrain the outcome baseline on `text_features` and confirm it lands near 82.5%, not
   95.7%. Movement toward 95% means verdict text has re-entered the features.
4. **Split integrity** — zero `split_group` overlap across train/val/test, and every guideline and
   regime present in all three folds at close to the pool rate. The tail guidelines are what will fail
   this; catch it here, not in the results.
5. **Sentinel check** — no column filtered with `.isna()` where `''` is the real sentinel. Make it a
   test, because this pattern has already caused two separate issues.
6. Model metrics against the judge-ruled labels, never the keyword tier, with the label check's measured
   accuracy beside them, broken out by regime.
7. LLM extraction scored against the rule-based channel on the cases where the judge cited, before being
   applied to those where they did not. Dropped quotes counted and reported, not silently discarded.
8. Retrieval precision@5 and MRR on the 50-query set, relevance criterion fixed beforehand.
9. Every span the app displays resolves exactly in `full_text`; report the pass rate.
10. End to end: feed a held-out case never seen in training, download the JSON, and confirm each offset
    quotes the text it claims to.

---

*Maintenance note: when a figure here stops being true — labels regenerated, split frozen, index built —
update this file. It is only useful while it is trustworthy.*
