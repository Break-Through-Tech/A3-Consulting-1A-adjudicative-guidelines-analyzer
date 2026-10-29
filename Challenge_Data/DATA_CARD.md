# Data Card: DOHA Security Clearance Adjudications Dataset

**Dataset Name:** Defense Office of Hearings and Appeals (DOHA) Security Clearance Decisions (Cleaned & Segmented Corpus)  
**Host Organization:** A3 Consulting LLC  
**Program:** Break Through Tech AI Studio — Fall 2026  
**Curators & Authors:** Aryan Sigdel (Data Infrastructure Lead), Rashed Alsaidi, Brian Martinez Flores, Samriddhi Srivastava  
**Version:** 1.0 (Frozen Splits & Segmented Features)  
**Date:** October 2026  

---

## 1. Dataset Overview & Provenance

### Purpose
The DOHA Security Clearance Decisions dataset is an explainable natural language processing (NLP) corpus curated to analyze industrial security clearance appeals and hearings. The corpus enables machine learning models to:
1. Predict final adjudication outcomes (`DENIED` vs `GRANTED`),
2. Classify relevant SEAD-4 Adjudicative Guidelines (A through M),
3. Extract specific mitigating conditions (`AG ¶ 20(x)`) and disqualifying conditions (`AG ¶ 19(x)`) with exact character-level text spans, and
4. Retrieve relevant historical precedent cases for explainable decision support.

### Source & Acquisition
- **Primary Origin:** Public decisions published on the Defense Office of Hearings and Appeals (DOHA) web archive (`https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/`).
- **Raw Scrape & Parser:** Initial scraping and PDF text extraction were executed by Teneika Askew via [TeneikaAskew/doha](https://github.com/TeneikaAskew/doha).
- **Cleaning & Segmentation Pipeline:** Processed through `src/build_dataset.py`, which fixed scraping contamination, structural section boundaries, corrupt metadata, and severe label fabrication in the original raw ingest.

---

## 2. Dataset Scope & Cohorts

| Cohort | Case Count | Description | Primary Use |
| :--- | :---: | :--- | :--- |
| **Total Cleaned Decisions** | **30,517** | All deduplicated, normalized decisions spanning 1996–2026 across hearing and appeal types. | Exploratory Data Analysis (EDA) and corpus-level search. |
| **Trainable Pool** | **21,959** | Subset where `use_for_training=True`, `label_source='formal_ruling'`, and `len(text_features) > 0`. | Multi-label guideline classification and extraction models. |
| **Outcome Benchmark Set** | **21,921** | Trainable pool decisions with resolved binary outcomes (`DENIED` or `GRANTED`). Excludes 24 unresolved cases and 14 REVOKED cases. | Outcome prediction baseline and transformer evaluation. |
| **Train Split** | **15,685** | Frozen 71.4% training partition (15,656 resolved binary outcomes). | Model training and feature selection. |
| **Validation Split** | **3,137** | Frozen 14.3% validation partition (3,132 resolved binary outcomes). | Hyperparameter tuning and model checkpointing. |
| **Held-Out Test Split** | **3,137** | Frozen 14.3% test partition (3,133 resolved binary outcomes). | Immutable benchmarking across all teams and models. |

---

## 3. Data Dictionary & Column Schema

The cleaned dataset contains **66 columns** in `Challenge_Data/cases_clean_*.parquet`. Every raw field was preserved alongside the cleaned counterpart to ensure complete provenance.

> [!NOTE]
> **Partitioning Columns (`split`, `stratum`)**:  
> The dataset partition columns `split` (`train`, `val`, `test`) and `stratum` are stored in `Challenge_Data/splits.parquet` (and attached automatically by `src.dataset.load_cases(pool=True)`), not in the parquet shards directly.

### 3.1 Case Identification & Source Metadata
| Column | Type | Empty % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `case_number` | String | 0.0% | Official DOHA docket number (e.g. `03-11614`). Not globally unique across hearings and appeals. | Metadata only |
| `case_type` | String | 0.0% | Corrected proceeding type: `hearing` or `appeal`. | Safe |
| `case_type_stored` | String | 0.0% | Original proceeding type from web scraper. | Metadata only |
| `case_type_corrected`| Boolean | 0.0% | Flag indicating whether the cleaning pipeline corrected the proceeding type. | Metadata only |
| `case_year` | Integer | 0.04% | Docket-number year prefix, not the decision year. Matches the year of `date` in only 29.5% of rows. Use `date` for anything time-based. | Metadata only |
| `date` | Timestamp | 0.08% | Exact date of the written decision (validated against plausible window). | Safe |
| `date_stored` | String | 0.0% | Raw unparsed date string from source documents. | Metadata only |
| `date_source` | String | 0.0% | Resolution source for the date (`signature_block`, `title_header`, `docket_fallback`). | Metadata only |
| `source_url` | String | 0.0% | Canonical URL to the original DOHA PDF decision document. 100% unique primary key. | Identifier |
| `shard` | String | 0.0% | Shard identifier corresponding to source parquet partitions. | Metadata only |
| `summary` | String | 0.02% | Scraped procedural summary or case syllabus. | ⚠️ Leaky (contains verdict) |
| `full_text` | String | 0.0% | Verbatim decision text extracted from source PDF. | ⚠️ Leaky (contains verdict) |

### 3.2 Adjudication Targets & Cleaned Metadata
| Column | Type | Empty % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `outcome` | String | 2.91% | Final adjudicated ruling: `DENIED`, `GRANTED`, `REMANDED`, or `REVOKED`. | **Target variable** |
| `outcome_stored` | String | 0.0% | Original scraped outcome string. | Target audit only |
| `outcome_source` | String | 0.0% | How outcome was derived (`verdict_section`, `appeal_composite`, `synopsis`). | Metadata only |
| `outcome_changed` | Boolean | 0.0% | True if cleaning pipeline corrected the source outcome. | Metadata only |
| `judge` | String | 12.19% | Standardized administrative judge name extracted from signature block. | Safe |
| `judge_raw` | String | 0.0% | Raw uncleaned judge text from source documents. | Metadata only |
| `judge_status` | String | 0.0% | Parsing status of judge entity (`clean`, `inferred`, `unresolved`). | Metadata only |
| `appeal_board_members`| List[Str] | 85.0% | Names of Appeal Board judges participating in panel decisions (empty for hearings). | Safe |
| `who_appealed` | String | 84.2% | Identity of appealing party (`Applicant`, `Department Counsel`). | Safe |

### 3.3 Guideline Classifications & Provenance
| Column | Type | Empty % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `guideline_regime` | String | 0.0% | Citation scheme governing the case: `sead4`, `directive` (Directive 5220.6), or `unknown`. | Safe |
| `guidelines` | List[Str] | 0.38% | Cleaned list of guidelines at issue (letters A through M). | **Target variable** |
| `guidelines_stored` | List[Str] | 0.19% | Source scraped guideline list (severely fabricated; matched non-adjudicated policies). | Unusable |
| `guideline_findings`| Struct | 33.6% | Per-guideline formal judge ruling: dictionary mapping letters to `FOR`, `AGAINST`, or `None` (33.6% with no finding; every value None). | **Target variable** |
| `guideline_subparagraphs`| List[Struct]| 34.5% | Subparagraph-level rulings with paragraph codes, findings, and character text offsets. | **Evidence target** |
| `formal_findings` | Struct | 31.9% | Raw parsed formal findings table from source parser. | Audit only |
| `label_source` | String | 0.0% | Trust tier of guideline labels: `formal_ruling` (72.4%), `inherited_from_hearing` (7.1%, not in the training pool), `regex_fallback` (20.1%), `none` (0.4%). | **Filtering mask** |

### 3.4 Document Segmentation & Leak-Free Features
| Column | Type | Empty % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `text_features` | String | 13.76% | **The canonical model input.** Built from STATEMENT OF THE CASE, FINDINGS OF FACT and, for appeals, THE JUDGE'S FINDINGS OF FACT (`config.FEATURE_SECTIONS`). The synopsis and everything from ANALYSIS onward are excluded. | **Primary input** |
| `html_prefix_len` | Integer | 0.0% | Character length of browser scraping header (`file:///...`) stripped from document start. | Metadata only |
| `verdict_span_start` | Integer | 1.23% | Character index where the judge's final verdict begins. | Boundary marker |
| `verdict_span_end` | Integer | 1.23% | Character index where the judge's final verdict ends. | Boundary marker |
| `segmentation_fuzzy` | Boolean | 0.0% | Flag indicating whether section headers required approximate regex matching. | Quality marker |
| `segmentation_partial`| Boolean | 0.0% | Flag indicating whether some structural sections were missing in source text. | Quality marker |
| `para_offsets_fof` | List[List[Int]]| 20.3% | Start and end character offsets for paragraphs within Findings of Fact. | RAG retrieval index |
| `sec_*_start / end` | Integer | Varies | Start and end character offsets for individual sections: `synopsis`, `statement_of_the_case`, `findings_of_fact`, `policies`, `analysis`, `conclusions`, `formal_findings`, `decision`, `discussion`, `order`. | Section slicing |

### 3.5 Quality & Grouping Flags
| Column | Type | Empty % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `use_for_training` | Boolean | 0.0% | Master inclusion flag. False for corrupt documents, missing text, or severe parsing flaws. | Filter flag |
| `split_group` | String | 0.0% | Docket grouping key (`case_number`). Prevents hearing/appeal multi-document leakage. | Grouping key |
| `is_duplicate` | Boolean | 0.0% | True if row is an exact byte-identical duplicate of an earlier document. | Filter flag |
| `case_number_collision`| Boolean | 0.0% | True if two different cases share the same docket identifier. | Metadata only |
| `is_degenerate` | Boolean | 0.0% | True if document text contains insufficient substantive content. | Filter flag |

### 3.6 Raw source sections (do not use)
These uncleaned sections from the source parser are largely superseded by document segmentation:

| Column | Empty % | Description |
| :--- | :---: | :--- |
| `sor_allegations` | 96.1% | Allegation text from Statement of Reasons. Rarely enumerated verbatim in decisions. |
| `mitigating_factors` | 17.8% | Unsegmented mitigating factors text from raw source scraper. |
| `judges_findings_of_fact` | 89.7% | Raw text from source parser (present mainly in appeals). |
| `judges_analysis` | 95.4% | Raw judge analysis text from source parser; superseded by `sec_analysis_*` offsets. |
| `discussion` | 93.4% | Raw discussion text from source parser; superseded by `sec_discussion_*` offsets. |
| `order` | 88.4% | Raw order text from source parser; superseded by `sec_order_*` offsets. |

---

## 4. Label Quality Tiers

Because the DOHA corpus has undergone multiple parsing iterations, labels must **never be pooled across quality tiers**:

```
┌─────────────────────────┬────────────┬───────────┬────────────────────────────────────────────────────────┐
│ Label Tier              │ Case Count │ Share     │ Trust Level & Appropriate Usage                        │
├─────────────────────────┼────────────┼───────────┼────────────────────────────────────────────────────────┤
│ formal_ruling           │ 22,105     │ 72.44%    │ Authoritative. Directly extracted from formal ruling   │
│                         │            │           │ sections. Mandatory for model training and evaluation. │
│ inherited_from_hearing  │ 2,155      │ 7.06%     │ Sound but inferred. Guideline labels inherited by an   │
│                         │            │           │ appeal from its underlying hearing. Not in the         │
│                         │            │           │ training pool.                                         │
│ regex_fallback          │ 6,142      │ 20.13%    │ Candidate-only (~60% precision). Keyword regex hits    │
│                         │            │           │ where formal findings were missing. Do not train on.   │
│ none                    │ 115        │ 0.38%     │ Unrecoverable. No valid guideline mentions found.      │
└─────────────────────────┴────────────┴───────────┴────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Canonical Training Filter**:  
> Always load training data via `src.dataset.load_cases(pool=True)` or apply:
> ```python
> pool = df[df["use_for_training"] & (df["label_source"] == "formal_ruling") & (df["text_features"].str.len() > 0)]
> ```
> This isolates the exact **21,959 training-ready cases**.

---

## 5. Dataset Distributions & Imbalances

### 5.1 Outcome Distribution
- **Whole-Corpus Outcomes**: Across all 30,517 decisions in the corpus, **62.8% are DENIED** (19,163), **32.8% are GRANTED** (10,013), 1.4% are REMANDED (437), 0.05% are REVOKED (16), and 2.9% have no resolved outcome (888).
- **Training Pool (21,959 cases)**: 13,882 DENIED, 8,039 GRANTED, 14 REVOKED, and 24 unresolved.
- **Binary Benchmark Set (21,921 cases)**: Consists of pool decisions with resolved binary outcomes (`DENIED` or `GRANTED`): **63.2% DENIED** (13,882) and **36.6% GRANTED** (8,039).
- **Binary Outcome Counts by Split**:
  - `train`: 15,656
  - `val`: 3,132
  - `test`: 3,133
- **Appeals vs Hearings**: Among decisions with a resolved outcome, appeals are denied more often (77.0%) than hearings (62.8%), and 16.4% of appeals have no resolved outcome.

### 5.2 Guideline Frequencies (A through M in Formal Rulings)
Adjudications exhibit extreme long-tail distribution:
* **Dominant Guidelines**:
  * **Guideline F (Financial Considerations)**: 59.7% of formal rulings (13,204 instances).
  * **Guideline E (Personal Conduct)**: 30.5% of formal rulings (6,746 instances).
  * Together, **76.9% of formal rulings** cite at least one of these two guidelines.
* **Moderate Frequency**:
  * Guideline B (Foreign Influence): 14.2% (3,143)
  * Guideline H (Drug Involvement): 13.0% (2,882)
  * Guideline J (Criminal Conduct): 13.0% (2,870)
  * Guideline G (Alcohol Consumption): 7.6% (1,671)
  * Guideline C (Foreign Preference): 5.4% (1,199)
* **Rare Guidelines**:
  * Guideline D (Sexual Behavior): 1.99% (439)
  * Guideline K (Handling Protected Information): 1.03% (227)
  * Guideline M (Use of Information Technology): 1.02% (226)
  * Guideline I (Psychological Conditions): 0.87% (193)
  * Guideline L (Outside Activities): 0.18% (40)
  * **Guideline A (Allegiance to the United States)**: **0.02% (4 cases total)**. All 4 co-occur with E, B, or M.

### 5.3 Citation Regimes
- **`sead4` (79.4% of pool)**: Evaluated under the 2017 SEAD-4 (and 2006 Adjudicative Guidelines) numbering scheme (`AG ¶ 19(x)` / `AG ¶ 20(x)`).
- **`directive` (12.0% of pool)**: Evaluated under DoD Directive 5220.6 (`E2.A6.1.2.x`).
- **`unknown` (8.7% of pool)**: Informal, narrative, or short-form citations.

---

## 6. Known Gaps, Limitations & Leakage Prevention

### 1. Missing Decisions (About 2,960 Decisions)
- About 2,960 decisions are missing from the collected corpus, of which 709 were never collected and 2,251 were overwritten (PROJECT_PLAN.md §8). The 36,703 collected links pointed to 33,477 distinct documents; 2,251 rows carried another case's text (appeals sharing filenames whose PDFs overwrote one another) and were removed as byte-identical duplicates.

### 2. Verdict Leakage in Raw Text
- **Vulnerability**: In raw `full_text`, the outcome appears before the halfway mark in 61% of cases. Models trained on `full_text` scored an artificial 95.7% accuracy by memorizing phrases like *"is denied"*, *"against applicant"*, or *"is granted"*.
- **Mitigation**: All models must train strictly on `text_features`. The leak-free baseline scores an honest **82.5%** accuracy (**81.6%** macro F1) against a 64.3% majority baseline (reproduced at 82.7% on the frozen split).

### 3. Empty Text Features (4,199 Cases, 13.76% of Corpus)
- 3,816 of the 4,199 are appeals, which usually have no pre-verdict factual section (src/config.py). The shared data loader (`src.dataset.load_cases(pool=True)`) automatically filters these out from the trainable pool.

### 4. Missing Outcomes (888 Cases, 2.91% of Corpus)
- 888 cases lack an extractable verdict phrase in the decision section. These 888 cases keep their guideline labels but are excluded from outcome metrics and the precedent index.

### 5. Circular Ground Truth Caveat
- The source column `formal_findings` was produced by an earlier automated regex. Comparing our parsed `formal_ruling` labels against that raw column is partially circular. External recall against guideline subheadings in the Analysis section is **98.0%**, with formal manual accuracy measured via the 50-case label check (PROJECT_PLAN.md §7).

---

## 7. Ethics, Bias & Responsible AI Considerations

### Privacy and Personally Identifiable Information (PII)
* **Publication**: DOHA publishes these decisions publicly, and the full text is committed to this public repo.
* **Sensitive Content**: Applicants are not named. Decisions say "Applicant", and captions show [Redacted], X's or dashes. However, the decisions describe sensitive personal details, including consumer debts, criminal history, substance use, and health records.
* **Usage Restrictions**: This dataset is intended exclusively for educational, statistical, and assistive adjudicative research. **Users must never attempt to build automated screening tools that target individuals, perform background checks, or profile private citizens.**

### Financial Considerations & Mitigation (Guideline F)
* Guideline F is at issue in 59.7% of formal rulings, and SEAD-4 treats hardship largely beyond the person's control as a mitigating condition (AG ¶ 20(b)).
* Machine learning models trained on this corpus risk learning spurious correlations that equate financial misfortune with a lack of trustworthiness or loyalty. Models must distinguish between intentional financial misconduct and conditions where mitigating factors apply.

### Adjudicative Standard
* Sourced from the data: **65.5% of decisions state the standard that granting a clearance must be "clearly consistent with the national interest", and 50.2% cite *Department of the Navy v. Egan* (484 U.S. 518)**.

### Human-in-the-Loop Requirement
* Automated classifiers and retrieval indices are decision-support aids designed to surface relevant precedent and flag specific SEAD-4 conditions.
* Under no circumstances should an automated pipeline make autonomous, binding, or unreviewed eligibility determinations. All outputs must maintain traceable provenance linking directly to verbatim text spans in the decision record.
