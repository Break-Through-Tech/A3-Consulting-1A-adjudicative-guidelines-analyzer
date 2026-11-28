# Span check

PROJECT_PLAN.md section 11, item 9: every span the app displays resolves exactly in `full_text`.
A span resolves when `full_text[start:end]` equals its stored text. A precedent paragraph stores offsets
only, so it is checked as a non-empty range inside the precedent's own text.

## Demo documents (results/demo)

10 documents.

| kind | spans | resolved | failed | pass rate |
|---|---|---|---|---|
| quote | 125 | 125 | 0 | 100.00% |
| context | 125 | 125 | 0 | 100.00% |
| lead_in | 38 | 38 | 0 | 100.00% |
| precedent | 50 | 50 | 0 | 100.00% |
| all | 338 | 338 | 0 | 100.00% |

## LLM kept quotes (results/llm/run/kept.csv)

156 cases.

| kind | spans | resolved | failed | pass rate |
|---|---|---|---|---|
| quote | 1001 | 1001 | 0 | 100.00% |
| all | 1001 | 1001 | 0 | 100.00% |

## Total

1339 of 1339 spans resolve (100.00%).
