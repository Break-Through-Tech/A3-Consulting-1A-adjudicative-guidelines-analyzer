# Test run reading

Both checks passed on the first run, so no prompt fix was made.

Cost check: billed input 36,758 tokens against 36,758 counted. Mean cost per request $0.00063, so 100 requests cost about $0.063.

Results check: 10 of 10 requests ended with `end_turn` and parsed. 62 of 63 quotes survived `verify`. Citation mode scored precision 93.3% and recall 71.8% over the 5 cases.

Dropped quote, read in full:

- `test-1-blind`, AG ¶ 20(b): "Arguably, these conditions were beyond his control." Reason `not_found`. It is a reworded sentence, not a copy.

Ten kept quotes, picked at random, read against their condition:

- 9 support the condition they are filed under.
- 1 is weak: `test-0-citation`, AG ¶ 20(a), "He demonstrated poor judgment by using his company credit card in an manner not authorized." It bears on the 20(a) judgment clause but is not clearly about it.
- In blind mode the model still quotes sentences that paraphrase a condition, for example "conditions largely beyond his control". The wording mask only removes runs of 30 or more shared characters, so paraphrases stay. This is a limit of blind mode, not a defect in the quotes.
