# Guideline F LLM pilot

Model: Claude Haiku 5.5 on the Batch API. Truth: the judge's citations read by `src/conditions.py`.

## Cost, predicted against actual

| Stage | Requests | Predicted input | Actual input | Predicted output | Actual output | Predicted cost | Actual cost |
|---|---|---|---|---|---|---|---|
| test | 10 | 36,758 | 36,758 | 23,000 | 17,953 | $0.0076 | $0.0063 |
| pilot | 100 | 403,553 | 403,553 | 230,000 | 169,050 | $0.0777 | $0.0624 |

Ledger total: $0.0688 of $1.00.

## Scores

| Mode | Stage | P | R | F1 | F1 interval | Disq F1 | Mit F1 | Kept | Dropped |
|---|---|---|---|---|---|---|---|---|---|
| citation | after_verify | 83.8% | 82.8% | 83.3% | 78.8% to 87.6% | 77.9% | 86.3% | 325 | 1 |
| citation | before_verify | 83.9% | 83.1% | 83.5% | 79.3% to 87.6% | 78.4% | 86.3% | 325 | 1 |
| blind | after_verify | 82.4% | 77.7% | 80.0% | 75.7% to 84.0% | 73.8% | 83.5% | 311 | 0 |
| blind | before_verify | 82.4% | 77.7% | 80.0% | 75.7% to 84.0% | 73.8% | 83.5% | 311 | 0 |

## Drops by reason

| Mode:reason | Count |
|---|---|
| citation:not_found | 1 |

## Per code

| Mode | Code | Support | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| citation | AG ¶ 19(a) | 44 | 36 | 2 | 8 | 0.947 | 0.818 | 0.878 |
| citation | AG ¶ 19(b) | 3 | 0 | 6 | 3 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 19(c) | 44 | 43 | 2 | 1 | 0.956 | 0.977 | 0.966 |
| citation | AG ¶ 19(d) | 2 | 0 | 0 | 2 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 19(e) | 1 | 0 | 6 | 1 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 19(f) | 9 | 9 | 10 | 0 | 0.474 | 1.0 | 0.643 |
| citation | AG ¶ 19(g) | 6 | 0 | 1 | 6 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 19(h) | 1 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 19(i) | 1 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| citation | AG ¶ 20(a) | 44 | 36 | 0 | 8 | 1.0 | 0.818 | 0.9 |
| citation | AG ¶ 20(b) | 42 | 40 | 3 | 2 | 0.93 | 0.952 | 0.941 |
| citation | AG ¶ 20(c) | 43 | 32 | 1 | 11 | 0.97 | 0.744 | 0.842 |
| citation | AG ¶ 20(d) | 42 | 39 | 7 | 3 | 0.848 | 0.929 | 0.886 |
| citation | AG ¶ 20(e) | 24 | 20 | 3 | 4 | 0.87 | 0.833 | 0.851 |
| citation | AG ¶ 20(f) | 5 | 1 | 1 | 4 | 0.5 | 0.2 | 0.286 |
| citation | AG ¶ 20(g) | 8 | 8 | 9 | 0 | 0.471 | 1.0 | 0.64 |
| blind | AG ¶ 19(a) | 44 | 33 | 2 | 11 | 0.943 | 0.75 | 0.835 |
| blind | AG ¶ 19(b) | 3 | 1 | 7 | 2 | 0.125 | 0.333 | 0.182 |
| blind | AG ¶ 19(c) | 44 | 40 | 3 | 4 | 0.93 | 0.909 | 0.92 |
| blind | AG ¶ 19(d) | 2 | 0 | 0 | 2 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 19(e) | 1 | 0 | 9 | 1 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 19(f) | 9 | 9 | 9 | 0 | 0.5 | 1.0 | 0.667 |
| blind | AG ¶ 19(g) | 6 | 0 | 1 | 6 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 19(h) | 1 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 19(i) | 1 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 20(a) | 44 | 36 | 1 | 8 | 0.973 | 0.818 | 0.889 |
| blind | AG ¶ 20(b) | 42 | 40 | 2 | 2 | 0.952 | 0.952 | 0.952 |
| blind | AG ¶ 20(c) | 43 | 29 | 0 | 14 | 1.0 | 0.674 | 0.806 |
| blind | AG ¶ 20(d) | 42 | 39 | 7 | 3 | 0.848 | 0.929 | 0.886 |
| blind | AG ¶ 20(e) | 24 | 17 | 2 | 7 | 0.895 | 0.708 | 0.791 |
| blind | AG ¶ 20(f) | 5 | 0 | 0 | 5 | 0.0 | 0.0 | 0.0 |
| blind | AG ¶ 20(g) | 8 | 4 | 10 | 4 | 0.286 | 0.5 | 0.364 |

## Reading the disagreements

I read 10 false positives and 10 false negatives in blind mode, picked at random with a fixed seed. Files: `pilot/disagreements.csv`.

False positives. All 10 quotes show the judge discussing the facts of the condition without citing it. 8 are clear. Two are arguable: a small-debts remark filed under 20(a) and a "did not take steps until he needed a clearance" remark filed under 19(b). These are places where the regex loses, because the judge reasons about the condition in plain words. Examples: repayment agreements on tax debts without a 20(g) cite, and spending on four properties without a 19(e) cite. So the false positive count understates the model, and precision against citations is a floor.

False negatives. All 10 cited conditions sit in a list of conditions that the judge recites with the guideline's wording, such as "Four are potentially applicable in this case: (a) inability to satisfy debts; (c) a history of not meeting financial obligations". Blind mode masks that wording, so the model has nothing to quote, and the passage holds no case facts. Here the regex wins. It reads a recital of an applicable condition as a citation, and the model correctly finds no case evidence. Three of the ten come from one case, where the judge listed 19(i), 19(e) and 20(c) from an older version of the guideline and applied none of them.

## Where the regex wins and loses

- The regex wins on recitals. The judge names the conditions that apply in one list sentence, and the model, which asks for case facts, cannot see them in blind mode. This drives most false negatives, especially 20(c), 20(e) and 20(g).
- The regex loses where the judge applies a condition in words only. 19(e), 19(b) and 20(g) show the most false positives, and the sample suggests most are real discussion.
- Both modes miss the rare codes: 19(d), 19(g), 19(h), 19(i) have 1 to 6 cases each, so per-code scores on them are noise.

## Weak spots in the test run

The blind mask removes only runs of 30 or more shared characters. Paraphrases of a condition, such as "conditions largely beyond his control", stay in the text. Blind mode is therefore a stricter test than citation mode, not a clean one.

## Cost per case

The pilot cost $0.0624 for 50 cases in two modes, so $0.00125 per case, or $0.00062 per case and mode. The test run cost $0.0063. The ledger total is $0.0688 against a $1.00 cap. Billed input matched the counted input exactly. Output came in at 1,795 tokens per request in the test run and 1,690 in the pilot, against the 2,300 guess, so the thinking guess was high. Predicted cost in the table above uses the original 2,300 guess. After the test run the output estimate was replaced with the measured 1,795, which put the pilot prediction at $0.0651 against $0.0624 actual.

For the November run on the 1,848 test cases in one mode, the same rates give about $1.15. That is over this plan's $1.00 cap, so it needs its own budget. This extrapolation has not been run.

## Limits

- Truth comes from the judge's citations. A cited condition may be only recited, and an applied condition may go uncited. Scores against citations are a floor for precision and a ceiling for what the regex proves.
- Citation mode flatters the model. Measured on Nov 9, the wording of 88% of cited conditions stays in the text when only the citation is masked. Blind mode removes runs of 30 or more shared characters but not paraphrases.
- 50 cases. The intervals above are wide, and codes with fewer than 10 cases carry no usable per-code score.
- SEAD-4 train cases only. Directive-era decisions and the validation and test splits were not run.
- One run, one prompt, effort medium. Nothing was tuned after the pilot.
- The quote check shows a quote exists and sits in the judge's reasoning. It does not show that the quote supports the code. The reading above is 10 and 10 cases, not a measurement.

## Edits other files need

- `PROJECT_PLAN.md`: replace the local model with Claude Haiku 5.5 on the Batch API; mark the prompt templates and the pilot done; replace 2,907 uncited cases and about 650 in test with 895 and 118 under the merged extractor.
- `TIMELINE.md` and `notes/handoff.md`: replace the Gemini direction with Haiku 5.5; the $10 Gemini credit note is wrong.
- `requirements.txt` pins `anthropic==2.7.0` and will conflict with PRs 24 and 26 at merge.

