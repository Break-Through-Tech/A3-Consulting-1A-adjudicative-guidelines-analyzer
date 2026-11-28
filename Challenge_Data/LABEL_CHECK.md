# Label check instructions

Our guideline labels come from code that reads each decision's FORMAL FINDINGS. Nobody has checked them by hand yet. Every model score we report will sit next to the accuracy this check measures, so please read carefully.

Each of us checks about 13 decisions. It takes about 2 minutes per decision, so around 30 minutes in total.

Due Saturday, November 28, at 8 pm.

## Your cases

| Person | Cases | Rows |
|---|---|---|
| Rashed | 1 to 13 | 25 |
| Sam | 14 to 26 | 18 |
| Aryan | 27 to 38 | 21 |
| Brian | 39 to 50 | 15 |

The sheet has no reviewer column. Find your rows by the `case` column.

## The sheet

Rashed will share the sheet in the group chat. It is made from `Challenge_Data/label_check.csv`.

Each row is one guideline we stored for a case. A case with three guidelines has three rows.

Only fill in the four empty columns. Don't change anything else.

## For each case

1. Click the case number in the `decision` column. The DOHA site downloads the decision as a PDF with a name like `24-01749.h1`. Open it.
2. Find FORMAL FINDINGS near the end of the decision. Searching the PDF for "Formal Findings" gets you there.
3. Read the lines that start with "Paragraph". Each one names a guideline and gives the judge's ruling.
4. Fill in every row for that case.

## What to write

| Column | What to write |
|---|---|
| `guideline_correct` | Y if FORMAL FINDINGS has a Paragraph line for this row's guideline. N if it doesn't. |
| `judge_direction` | The ruling on that Paragraph line, FOR or AGAINST. Write none if the line gives no ruling. Leave it blank if you wrote N. |
| `missed_guidelines` | First row of the case only. Any guideline that has a Paragraph line but no row in the sheet, with its ruling, such as `E FOR`. Leave it blank if nothing is missing. |
| `notes` | Anything odd. For example, the PDF won't open, you can't find FORMAL FINDINGS, or the decision date in the PDF differs from the `date` column. |

## Example

Say the sheet has one row for a case, guideline F with `stored_direction` AGAINST. The PDF's FORMAL FINDINGS says this:

```
Paragraph 1, Guideline F:      AGAINST APPLICANT
  Subparagraphs 1.a - 1.c:     Against Applicant
  Subparagraph 1.d:            For Applicant
Paragraph 2, Guideline E:      FOR APPLICANT
  Subparagraph 2.a:            For Applicant
```

You would fill in the F row like this:

| guideline_correct | judge_direction | missed_guidelines | notes |
|---|---|---|---|
| Y | AGAINST | E FOR | |

Guideline F has a Paragraph line, so it gets Y. The line says AGAINST. Guideline E has a Paragraph line but no row in the sheet, so it goes in `missed_guidelines`.

## Rules

- Only the Paragraph lines count. Skip the subparagraphs. A guideline can be AGAINST overall even when some of its subparagraphs are FOR, so always take the ruling from the Paragraph line.
- Only FORMAL FINDINGS counts. Searching for "Guideline F" also finds it in POLICIES, which lists every guideline, and in the Statement of the Case. Ignore those.
- Write what the judge wrote, even when it differs from `stored_direction`. Don't fix the stored columns. Rashed compares the two afterwards.
- Read the PDF yourself. Don't use our dataset's text, and don't ask an AI tool to read it for you. The check measures our labels against a person reading the ruling.
- Two rows have `stored_direction` none, cases 26 and 45. If the Paragraph line there has no overall ruling and only the subparagraphs do, write none and explain in `notes`.
- If something is unclear, ask in the group chat instead of guessing.

## When you're done

Post in the group chat that your rows are filled in.
