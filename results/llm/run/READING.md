# Test split run reading

Read on Nov 17 after the full batch. The numbers are in the "Test split run" section of `results/llm/REPORT.md`.

## What changed before the run

The first 10-case test batch showed that 3 of 5 uncited cases sent the model only 181 to 285 characters, the closing lines of the decision. Their analysis has a "Discussion" heading, which the segmenter does not know, so the row had no analysis section. 91 of the 118 uncited test cases had this problem.

`conditions.analysis_span` now falls back to the text from the end of POLICIES, or of the findings of fact, to the start of FORMAL FINDINGS. The pilot keeps the old behaviour. With the fallback, the uncited test cases fell from 118 to 74 and the uncited pool cases from 895 to 603. That first test batch cost $0.0047 and is kept in `first_test_batch/`. Its results are not part of the scores.

## Checks

- 174 of 174 requests ended with `end_turn` and parsed.
- Every one of the 1,001 kept quotes equals `full_text` at its offsets. Re-running `verify` on the raw answers gives the same counts: 1,001 kept, 3 `restates_guideline`, 3 `contains_mask`, 2 `not_found` and 1 `near_miss`.
- Billed input matched the counted tokens exactly. The run cost $0.0996 against a central estimate of $0.1086 and ceilings of $0.3831.

## Dropped quotes

The 3 `contains_mask` drops are real sentences where the model copied the mask into its quote, such as "Applicant did not dispute any debts, so [CITE] does not apply." The drop is correct, since the text has no "[CITE]". A prompt line that says to leave the mask out of quotes would likely keep them. The pilot had none of these.

## Ten uncited quotes, picked at random

8 support the condition they are filed under. Two do not.

- AG ¶ 19(c): "She has further demonstrated that since October 1995, she has steadfastly paid in a timely manner all of her legitimate bills". It describes paying on time, which is mitigation, not a history of missed obligations.
- AG ¶ 19(f): "Applicant failed to document that he". It is a fragment cut mid-sentence.

## Uncited cases with nothing kept

18 of 74 uncited cases have no kept quote. 6 still send under 2,500 characters. The other 12 are sead4 cases with 7,500 to 13,000 characters, and in 11 of them the input analyzes another guideline, mostly B, with almost no mention of finances. The empty answer looks right for those. These cases carry a Guideline F finding in our labels, so they are worth a look in the label check.

## Directive and unknown

The cited sample has 3 directive and 3 unknown cases. Directive F1 is 50.0% on 3 cases, so it says little. Directive truth uses the `sead4` mapping.
