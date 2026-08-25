# Explainable AI for Security Clearance Decision

**Company / Org:** A3 Consulting LLC  
**Challenge Advisor:** Teneika Askew, teneika.askew@a3consultingllc.com  
**AI Studio Coach:** Shaun Figueiro, shaun.figueiro@breakthroughtech.org                 
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About A3 Consulting LLC

A3 Consulting works at the intersection of data science and national security policy, building tools that help reviewers and adjudicators reason over public precedent. This project adapts their SEAD‑4 expertise into a reproducible educational pipeline that demonstrates explainable automated analysis of security‑clearance decisions.

---

## 🎯 The Challenge

### Project Summary
Build an explainable NLP + LLM pipeline that ingests ~36,700 public DOHA security‑clearance decision documents to:

- Predict the likely adjudication outcome for a case,
- Identify which SEAD‑4 guideline letters (A–M) are relevant (multi‑label classification),
- Extract specific disqualifiers and mitigating factors with supporting evidence (text spans and AG paragraph references),
- Retrieve and surface relevant precedent cases (RAG) with relevance scores and short similarity notes, so a human reviewer can quickly verify the model's reasoning and recommendations.

### Success Criteria
- Quantitative:
  - Strong baseline-to‑improved outcome prediction on held‑out cases (report Accuracy, Macro/Micro F1; per‑label metrics for A–M).
  - Reliable multi‑label guideline classification (per‑label precision/recall/F1).

- Qualitative:
  - Extracted disqualifiers/mitigators are traceable to source text spans or explicit SEAD‑4 paragraphs; show provenance for LLM outputs.
  - Precedent retrieval returns semantically relevant cases with interpretable similarity indicators and no obvious hallucination.

- Deliverable:
  - A working end‑to‑end demo (Streamlit or similar) that accepts a case (PDF/text) and outputs: recommendation + confidence, per‑guideline assessments, cited evidence (text spans + AG references), and similar precedents.

### Stretch Goals
- Calibration analysis of confidence scores and per‑label calibration.
- Fairness / consistency audits (do similar fact patterns across years produce consistent outputs?).
- Fine‑tune a compact open LLM on the parsed DOHA corpus for improved extraction quality.
- Automated evaluation of explanation traceability (matching extracted spans against ground truth).
  
### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month     | Milestone             | Key Activities                                                      |
|-----------|-----------------------|---------------------------------------------------------------------|
| **September** | [Title] | Onboard team onto the provided DOHA dataset (parsed Parquet, no scraping needed). Exploratory data analysis on case outcomes, guideline frequency, and text characteristics. Establish a baseline classification model (TF-IDF plus logistic regression / lightweight transformer) predicting granted vs. denied, and a labeled subset for evaluation.          |
| **October** | [Title] | Build the guideline-classification component (multi-label A through M) and the LLM-based disqualifier/mitigator extraction using the free-tier Gemini API. Stand up the RAG precedent-retrieval pipeline against the provided vector index. Compare LLM-based vs. traditional ML approaches on the same eval set.         |
| **November** | [Title] | Integrate components into an end-to-end analyzer that outputs a structured assessment (recommendation, confidence, guideline breakdown, cited precedents). Build a Streamlit demo UI. Run final evaluation, error analysis, and document limitations. Prepare presentation and GitHub deliverable.            |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** DOHA Security Clearance Decision Documents  
**Format:** CSV, JSON, Parquet, PDF  
**Size:** 5gb to 10gb  
**Location:** https://github.com/TeneikaAskew/doha

### Key Details
- ~36,700 publicly available DOHA (Defense Office of Hearings and Appeals) documents containing numerical, categorical, and text data, stored in CSV, JSON, Parquet, and PDF formats.
- Some documents may require preprocessing to handle inconsistencies in formatting or missing information.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- Accuracy, Precision/Recall, F1 score, AUC

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Link to an article or blog post about the problem domain]
- [Link to an industry report or case study]

**Technical Tutorials:**
- [Link to a free tutorial on the ML technique(s) involved]
- [Link to documentation for a key library or tool]

**Code Examples:**
- [Link to a relevant GitHub repo]
- [Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session C).
