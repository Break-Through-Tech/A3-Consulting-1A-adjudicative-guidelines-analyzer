# Traceability check instructions

Our tools pick out the passages in a decision that point to a financial condition, such as "AG ¶ 20(b)", the mitigator for debts that arose from causes beyond the person's control. Nobody has checked by hand whether those passages really support the condition they are tied to. This check measures that, so please read carefully.

Each of us checks 25 findings. It takes about 3 minutes per finding, so around 75 minutes in total.

Due date: Rashed will set it.

## Your findings

| Person | Findings | Rows |
|---|---|---|
| Rashed | 1 to 25 | 25 |
| Sam | 26 to 50 | 25 |
| Aryan | 51 to 75 | 25 |
| Brian | 76 to 100 | 25 |

The `reviewer` column in the sheet shows the same split. Filter on your name.

## The sheet

Rashed will share the sheet in the group chat. It is made from `Challenge_Data/traceability.csv`.

Only fill in the two empty columns, `supports` and `notes`. Don't change anything else.

## What a finding is

A finding is one row. It ties a span of a decision's text to one condition. The row gives you:

- `code` and `condition_text`, the condition and what it says.
- `quote`, the span of text we tied to it.
- `start` and `end`, where the span sits in the decision's text.

The question is whether the span supports the condition. You are checking the span, not the case. The applicant can lose or win. It makes no difference here.

## For each finding

1. Click the case number in the `decision` column. The DOHA site downloads the decision as a PDF with a name like `24-01749.h1`. Open it.
2. Find the span. Search the PDF for the first few words of `quote`. Line breaks and hyphens at the end of a line can differ from the sheet, so if the full quote finds nothing, search for a shorter piece.
3. For a short span such as `(a)` or `AG ¶ 19(c)`, search for the whole thing and read the sentence around it. There can be several matches. The right one sits in the judge's analysis or conclusions, near the end of the decision, not in the POLICIES section that lists every condition.
4. The `start` and `end` numbers count characters in the text our tools read, which is not the PDF. They are not page numbers. Use them only to tell two matches apart. The first match in the PDF is usually the lower number.
5. Read the span with the sentence around it. Compare it with `condition_text`. Fill in `supports`.

## What to write

| Column | What to write |
|---|---|
| `supports` | `yes`, `no` or `unsure`. Lower case, nothing else. |
| `notes` | Anything odd. Always write a note for `no` and `unsure`. |

## When to answer

- `yes` when the span says or cites what the condition says. A long passage that describes the facts the condition covers is a yes. A short citation such as `AG ¶ 20(b)` is a yes when the judge uses it for this condition in that sentence.
- `no` when the span is about something else, when it is a real passage that does not bear on the condition, or when it points at a different condition from the one in the row. A citation that the judge only lists in a recital, with no application to the case, is also a `no`.
- `unsure` when you can't find the span, when the PDF won't open, or when you can't decide after reading the sentence around it. Leave a note that says which.

Leave `supports` blank only for rows you haven't reached.

## Example

Say the row is condition `AG ¶ 20(b)`, "the conditions that resulted in the financial problem were largely beyond the person's control". The quote reads "his hours were cut during the pandemic, and he was out of work for four months".

The passage describes a cause outside the applicant's control, so you write `yes`.

Now say the quote reads "He did not provide documents showing a payment plan". That is evidence about something else, so you write `no` and a note such as "about a payment plan, not about cause".

## Rules

- One finding at a time. Judge each row on its own, even when the same decision shows up in more than one row.
- Read the PDF yourself. Don't use our dataset's text, and don't ask an AI tool to read it for you. The check measures our tools against a person reading the decision.
- Don't try to guess where a row came from. Judge only the span and the condition.
- If something is unclear, ask in the group chat instead of guessing.

## When you're done

Post in the group chat that your rows are filled in. Rashed scores the sheet with `python -m src.build_traceability --score <filled csv>`.
