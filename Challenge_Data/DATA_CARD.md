# Data Card: DOHA Security Clearance Adjudications Dataset

**Dataset Name:** Defense Office of Hearings and Appeals (DOHA) Security Clearance Decisions (Cleaned & Segmented Corpus)  
**Host Organization:** A3 Consulting LLC  
**Program:** Break Through Tech AI Studio — Fall 2026  
**Curators & Authors:** Aryan Sigdel (Data Infrastructure Lead), Rashed Alsa, Brian Flores, Samriddhi Srivastava  
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
| **Outcome Benchmark Set** | **21,935** | Trainable pool decisions with resolved binary outcomes (`DENIED` or `GRANTED`). Excludes 24 rare/ambiguous cases. | Outcome prediction baseline and transformer evaluation. |
| **Train Split** | **15,685** | Frozen 71.4% training partition (~15,656 resolved outcomes). | Model training and feature selection. |
| **Validation Split** | **3,137** | Frozen 14.3% validation partition (~3,132 resolved outcomes). | Hyperparameter tuning and model checkpointing. |
| **Held-Out Test Split** | **3,137** | Frozen 14.3% test partition (~3,133 resolved outcomes). | Immutable benchmarking across all teams and models. |

---

## 3. Data Dictionary & Column Schema

The cleaned dataset contains **66 columns** in `Challenge_Data/cases_clean_*.parquet`. Every raw field was preserved alongside the cleaned counterpart to ensure complete provenance.

### 3.1 Case Identification & Source Metadata
| Column | Type | Null % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `case_number` | String | 0.0% | Official DOHA docket number (e.g. `03-11614`, `18-02453`). Not globally unique across hearings and appeals. | Metadata only |
| `case_type` | String | 0.0% | Corrected proceeding type: `hearing` (first-instance) or `appeal` (three-judge panel). | Safe |
| `case_type_stored` | String | 0.0% | Original proceeding type from the web scraper (often mislabelled). | Metadata only |
| `case_type_corrected`| Boolean | 0.0% | Flag indicating whether the cleaning pipeline corrected the proceeding type. | Metadata only |
| `case_year` | Integer | 0.04% | Year the decision was rendered (1996–2026). Cross-validated against docket year. | Safe |
| `date` | Timestamp | 0.08% | Exact date of the written decision (validated against plausible window). | Safe |
| `date_stored` | String | 0.0% | Raw unparsed date string from source documents. | Metadata only |
| `date_source` | String | 0.0% | Resolution source for the date (`signature_block`, `title_header`, `docket_fallback`). | Metadata only |
| `source_url` | String | 0.0% | Canonical URL to the original DOHA PDF decision document. 100% unique primary key. | Identifier |
| `shard` | String | 0.0% | Shard identifier corresponding to source parquet partitions. | Metadata only |
| `summary` | String | 0.0% | Scraped procedural summary or case syllabus. | ⚠️ Leaky (contains verdict) |
| `full_text` | String | 0.0% | Verbatim decision text extracted from source PDF. | ⚠️ Leaky (contains verdict) |

### 3.2 Adjudication Targets & Cleaned Metadata
| Column | Type | Null % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `outcome` | String | 2.91% | Final adjudicated ruling: `DENIED`, `GRANTED`, `REMANDED`, or `REVOKED`. | **Target variable** |
| `outcome_stored` | String | 0.0% | Original scraped outcome string (frequently inaccurate on appeals). | Target audit only |
| `outcome_source` | String | 0.0% | How outcome was derived (`verdict_section`, `appeal_composite`, `synopsis`). | Metadata only |
| `outcome_changed` | Boolean | 0.0% | True if our cleaning pipeline corrected the source outcome. | Metadata only |
| `judge` | String | 12.19%| Standardized administrative judge name extracted from signature block. | Safe |
| `judge_raw` | String | 0.0% | Raw uncleaned judge text from source documents. | Metadata only |
| `judge_status` | String | 0.0% | Parsing status of judge entity (`clean`, `inferred`, `unresolved`). | Metadata only |
| `appeal_board_members`| List[Str] | 0.0% | Names of Appeal Board judges participating in panel decisions. | Safe |
| `who_appealed` | String | 84.2% empty | Identity of appealing party (`Applicant`, `Department Counsel`). | Safe |

### 3.3 Guideline Classifications & Provenance
| Column | Type | Null % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `guideline_regime` | String | 0.0% | Citation scheme governing the case: `sead4`, `directive` (Directive 5220.6), or `unknown`. | Safe |
| `guidelines` | List[Str] | 0.38% | Cleaned list of guidelines at issue (letters A through M). | **Target variable** |
| `guidelines_stored` | List[Str] | 0.0% | Source scraped guideline list (severely fabricated; matched non-adjudicated policies). | Unusable |
| `guideline_findings`| Struct | 0.0% | Per-guideline formal judge ruling: dictionary mapping letters to `FOR`, `AGAINST`, or `None`. | **Target variable** |
| `guideline_subparagraphs`| List[Struct]| 0.0% | Subparagraph-level rulings with paragraph codes, findings, and character text offsets. | **Evidence target** |
| `label_source` | String | 0.0% | Trust tier of guideline labels: `formal_ruling`, `inherited_from_hearing`, `regex_fallback`, `none`. | **Filtering mask** |
| `formal_findings` | Struct | 0.0% | Raw parsed formal findings table from source parser. | Audit only |

### 3.4 Document Segmentation & Leak-Free Features
| Column | Type | Null % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `text_features` | String | 13.76% empty | **The canonical model input.** Composed strictly of pre-verdict sections (Findings of Fact + Discussion). | **Primary input** |
| `html_prefix_len` | Integer | 0.0% | Character length of browser scraping header (`file:///...`) stripped from document start. | Metadata only |
| `verdict_span_start` | Integer | 1.23% | Character index where the judge's final verdict begins. | Boundary marker |
| `verdict_span_end` | Integer | 1.23% | Character index where the judge's final verdict ends. | Boundary marker |
| `segmentation_fuzzy` | Boolean | 0.0% | Flag indicating whether section headers required approximate regex matching. | Quality marker |
| `segmentation_partial`| Boolean | 0.0% | Flag indicating whether some structural sections were missing in source text. | Quality marker |
| `para_offsets_fof` | List[List[Int]]| 0.0% | Start and end character offsets for paragraphs within Findings of Fact. | RAG retrieval index |
| `sec_*_start / end` | Integer | Varies | Start and end character offsets for individual sections: `synopsis`, `statement_of_the_case`, `findings_of_fact`, `policies`, `analysis`, `conclusions`, `formal_findings`, `decision`, `discussion`, `order`. | Section slicing |

### 3.5 Quality, Partitioning & Grouping Flags
| Column | Type | Null % | Description | Modeling Safety |
| :--- | :--- | :---: | :--- | :--- |
| `use_for_training` | Boolean | 0.0% | Master inclusion flag. False for corrupt documents, missing text, or severe parsing flaws. | Filter flag |
| `split_group` | String | 0.0% | Docket grouping key (`case_number`). Prevents hearing/appeal multi-document leakage. | Grouping key |
| `split` | String | 0.0% | Frozen dataset partition: `train`, `val`, or `test` (available in `splits.parquet` and loader). | Partition key |
| `stratum` | String | 0.0% | Rarest guideline stratum used by `StratifiedGroupKFold`. | Partition audit |
| `is_duplicate` | Boolean | 0.0% | True if row is an exact byte-identical duplicate of an earlier document. | Filter flag |
| `case_number_collision`| Boolean | 0.0% | True if two different cases share the same docket identifier. | Metadata only |
| `is_degenerate` | Boolean | 0.0% | True if document text contains insufficient substantive content. | Filter flag |

---

## 4. Label Quality Tiers

Because the DOHA corpus has undergone multiple parsing iterations, labels must **never be pooled across quality tiers**:

```
┌─────────────────────────┬────────────┬───────────┬────────────────────────────────────────────────────────┐
│ Label Tier              │ Case Count │ Share     │ Trust Level & Appropriate Usage                        │
├─────────────────────────┼────────────┼───────────┼────────────────────────────────────────────────────────┤
│ formal_ruling           │ 22,105     │ 72.44%    │ Authoritative. Directly extracted from formal ruling   │
│                         │            │           │ sections. Mandatory for model training and evaluation. │
│ inherited_from_hearing  │ 2,155      │ 7.06%     │ Sound but Inferred. Guideline labels inherited by an   │
│                         │            │           │ appeal from its underlying hearing. Evaluation only.   │
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

### 5.1 Outcome Distribution (Trainable Pool)
- **Class Imbalance**: 62.8% `DENIED` vs 32.8% `GRANTED`.
- **Procedural Disparity**: Appeals are denied at a higher rate (77.0%) than first-instance Hearings (62.8%), because the Appeal Board operates under an "arbitrary and capricious" deferential review standard.
- **Scarcity Classes**: `REMANDED` (437 cases, 1.4%) and `REVOKED` (16 cases, 0.05%) represent interlocutory or administrative dispositions and are excluded from binary outcome benchmarking.

### 5.2 Guideline Frequencies (A through M in Formal Rulings)
Adjudications exhibit extreme long-tail distribution:
* **Dominant Guidelines**:
  * **Guideline F (Financial Considerations)**: 59.7% of cases (13,204 instances).
  * **Guideline E (Personal Conduct)**: 30.5% of cases (6,746 instances).
  * Together, 76.9% of all decisions cite at least one of these two guidelines.
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

### 1. Verdict Leakage in Raw Text
- **Vulnerability**: In raw `full_text`, the outcome was stated in the synopsis or first third of the document in 61% of cases. Models trained on `full_text` scored an artificial 95.4% accuracy by memorizing phrases like *"clearance is denied"*.
- **Mitigation**: All models must train strictly on `text_features`. The leak-free baseline scores an honest **82.4%** on the frozen test split.

### 2. Empty Text Features (13.76% of Corpus)
- 4,199 cases in the full corpus have empty `text_features` because structural headers were missing or scans were unsegmented. The shared data loader automatically filters these out when `pool=True`.

### 3. Missing Outcomes (2.91% of Corpus)
- 888 cases lack an extractable verdict phrase in the decision section. They retain guideline and metadata tags for unsupervised retrieval but are excluded from outcome benchmarks.

### 4. Circular Ground Truth Caveat
- The source column `formal_findings` was produced by an earlier automated regex. Comparing our parsed `formal_ruling` labels against that raw column is partially circular. External recall against guideline subheadings in the Analysis section is **98.0%**, with formal manual accuracy measured via Task 9.

---

## 7. Ethics, Bias & Responsible AI Considerations

### Privacy and Personally Identifiable Information (PII)
* **Legal Status**: DOHA decisions are public records released pursuant to the Freedom of Information Act (FOIA).
* **Sensitive Content**: These documents contain unredacted names of applicants, witnesses, spouses, and employers, alongside deeply sensitive disclosures: detailed consumer debt balances, criminal arrest histories, substance dependency treatments, and mental health diagnoses.
* **Usage Restrictions**: This dataset is intended exclusively for educational, statistical, and assistive adjudicative research. **Users must never attempt to build automated screening tools that target individuals, perform background checks, or profile private citizens.**

### Socioeconomic and Systemic Disparities
* **Financial Bias (Guideline F)**: 60% of all DOHA cases revolve around delinquent debts (credit cards, medical bills, foreclosures, past-due student loans). Economic hardships often stem from involuntary life events—such as catastrophic illness, job layoffs, or divorce.
* **Model Fairness Concern**: Machine learning models trained on this corpus risk learning spurious correlations that equate financial misfortune with a lack of trustworthiness or loyalty. Models must distinguish between intentional financial misconduct (tax evasion, embezzlement) and resolved past debt with good-faith repayment plans.

### The Asymmetric Legal Standard
* **"Clearly Consistent with National Interest"**: Under Department of the Navy v. Egan (484 U.S. 518), clearance eligibility requires an affirmative finding that granting access is clearly consistent with the national security of the United States. Any reasonable doubt must be resolved in favor of national security.
* **Baseline Skew**: This evidentiary standard explains why nearly **63% of adjudicated decisions result in clearance denial**. This outcome distribution reflects national security statutory mandates, not judicial prejudice.

### Human-in-the-Loop Requirement
* Automated classifiers and retrieval indices are decision-support aids designed to surface relevant precedent and flag specific SEAD-4 conditions.
* Under no circumstances should an automated pipeline make autonomous, binding, or unreviewed eligibility determinations. All outputs must maintain traceable provenance linking directly to verbatim text spans in the decision record.
