# Week 3: monitoring, live grading, and a privacy filter for uploads

Three pieces for the public page, 3 to 6 November 2026. Every figure below comes from a file in this repo, named next to it.

## 1. A private usage dashboard (monitoring and cost)

```bash
uv run streamlit run scripts/usage_dashboard.py --server.port 8503
```

Questions, visits, cost, time to answer (median and 95th percentile), thumbs up and down, and answers that weren't ok, all per day. It also has tables of errors and thumbs-down answers. It's built on `scripts/usage_report.py` and reads the same log.

- **Read-only, enforced by the database.** `UsageLog(read_only=True)` runs every Postgres query inside `BEGIN READ ONLY`, opens SQLite with `mode=ro`, and skips the `CREATE TABLE`. A write attempted through it against the hosted database came back `ReadOnlySqlTransaction`. The text report uses the same mode.
- **Private.** It's served to this computer only (`.streamlit/config.toml`). It's a separate script, not part of `hosted_app.py`, so the public page doesn't include it.

## 2. Monthly grading of live answers

```bash
uv run python scripts/usage_report.py --month 2026-11 --sample 20     # pick 20, seed 0
uv run python scripts/usage_report.py --grades out/grading/sample_2026-11_seed0.csv
```

The sample comes with a Markdown copy to read: each question, the answer, and the code the agent ran with its output, plus a three-line marking guide (`y`, `n`, `skip`). Every question counts, not just finished answers, so a crash is an `n`. Grading saves the rate and its 95% Wilson interval with ids and verdicts only, so the result can be committed without any visitor's words.

**Status: no rate yet.** The hosted log holds 1 question, and it's Ak's own launch test (`evals/live_grading/log_snapshot_2026-11-03.json`). On 3 November the page was asleep for lack of traffic. The kit was tested end to end on a made-up log (`tests/test_usage_report.py`).

## 3. A privacy filter for uploads

`privacy.py` checks each column of an uploaded CSV for personal data: names, emails, phone numbers, street addresses, postcodes, dates of birth, ID and card numbers, IP addresses, home locations, usernames, and free text containing contact details. It uses rules only: value patterns (Luhn-checked card numbers, IBAN, SSN, National Insurance, Aadhaar and PAN shapes, UK postcodes, phone formats) plus column-name hints that the values must agree with. It's instant and free, and it runs on the page's server, so the check sends the file nowhere.

When it flags a file, the page:

- **Warns the visitor:** it lists the columns and what they look like.
- **Keeps the words out of the log:** for that file it records `[withheld: ...]` instead of the question, and no answer or code. Cost, time, status and the thumbs rating are still kept, so the dashboard and budget stay right.

This was checked end to end on the page with a fake agent: a file of names, emails and phones got the warning and a withheld log row, and a file without personal data was logged as before.

### How it was measured

| Step | Commit |
|---|---|
| Labelling rules (`evals/privacy/LABELS.md`) and 225 labelled columns, before any filter code | `26d1a00` |
| Filter built on the 90 dev columns; its verdicts on all 225 committed before the test split was scored or any paid call | `6e609fe` |
| Test split scored; then Claude Opus 5.5 judged the same 135 test columns, given the same rules | this branch |

The 225 columns: 101 personal and 124 not. Their names come from real exports (Shopify, Mailchimp, Stripe, Google Forms, HR and CRM files) with made-up values, plus all 27 columns of the three sample datasets with their real values. Look-alikes are planted on purpose, such as a product "Name" column, store postcodes, email opt-in flags and 13-digit barcodes. Built by `evals/privacy/make_columns.py`, seeded.

### Results (`evals/privacy/results.json`), on the 135 held-out test columns

| | Precision | Recall | Cost |
|---|---|---|---|
| Rules (`privacy.py`) | 100% (60/60, 95% CI 94% to 100%) | 98% (60/61, 91% to 100%) | free, instant |
| Claude Opus 5.5, low effort | 98% (61/62, 91% to 100%) | 100% (61/61, 94% to 100%) | $0.80 for 135 columns, median 1.4 s each |

The rules missed one column: a column of names with no telling column name, where too few of the names were on its short list of common first names. Claude flagged one column too many: a store's phone number, which the rules exclude as a business number. The two are within each other's intervals. The page uses the rules: they're free, instant, and don't send the file anywhere for the check.

### Limits

- **The numbers are optimistic.** The same author wrote the labels, the value generator and the rules, and the values are made up. Real uploads will be messier, so expect lower recall on real files. Ak's spot-check of 25 labels (`evals/privacy/spotcheck.md`) is the check on the labels themselves.
- **Names without a column hint** are only caught when at least half of them are common first names (a short built-in list).
- **What isn't covered:**
  - Only the first 500 rows are checked.
  - Personal data typed into the question itself isn't caught.
  - Pseudonymous IDs (customer numbers, hashed emails) aren't flagged, by design (see `LABELS.md`).
- **A flagged file still goes to Anthropic's sandbox** for the analysis, as every upload did before. What changes is the warning and what the page's own log keeps.

## Spend

$0.80 of the $2 budget, all of it the Claude comparison (`results.json`, `cost_usd`). Everything else ran locally, or with a fake agent.
