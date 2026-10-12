# Putting the agent online

`hosted_app.py` is a public version of the web page. Anyone with the link can ask questions about the sample datasets or their own CSV. It runs on your API key, inside a monthly budget, and records every question so you can see how people use it.

It differs from `app.py`, which only runs on your own computer, in four ways:

- **The agent's code runs in Anthropic's sandbox, never on the server.** `cloud_agent.py` uses Claude's code execution tool: an isolated container per conversation, with no internet access and none of the server's secrets. A visitor can't get the agent to read the API key or the database password, because that code never runs where they are.
- **One setup:** Sonnet 5.5 at high effort, the setup both evals measured on the sandbox (see the README).
- **Limits:** the page stops answering once the month's spend reaches `MONTHLY_BUDGET_USD`, and each visit gets `QUESTIONS_PER_VISIT` questions.
- **A usage log** (`usage_log.py`): each question, its answer, the code the agent ran, the cost, the time taken, and the visitor's thumbs up or down. For uploaded files, what the code printed isn't kept, because that output is the visitor's data. The uploads themselves are deleted from Anthropic's Files API when the visitor switches data or starts over, or within a day.

Setup takes about 20 minutes, in three accounts.

## 1. A separate API key with a spending limit

In the [Claude Console](https://console.anthropic.com):

1. Create a workspace for the page, for example `data-analyst-agent-public`.
2. Set that workspace's monthly spend limit to $10. The page stops itself at $10 too, but the Console limit is the one that can't be bypassed.
3. Create an API key in that workspace.

Use a workspace of its own for two reasons. The page can't spend your other projects' budget. And once an hour it deletes files older than a day from its workspace's Files API storage, which would also delete files your other projects keep there.

## 2. A database for the usage log

Any Postgres works. [Neon](https://neon.tech)'s free tier is enough:

1. Create a project, for example `data-analyst-agent`.
2. Copy its connection string. It looks like `postgresql://user:password@host/db?sslmode=require`.

The page creates its one table on first start.

## 3. Deploy on Streamlit Community Cloud

1. Make sure your latest commits are pushed to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository `adityakannan-2003/data-analyst-agent`, branch `main`, main file path `hosted_app.py`. Pick a URL, such as `data-analyst-agent.streamlit.app`.
4. Under **Advanced settings**, choose Python 3.12 and paste these secrets, with your own values:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   DATABASE_URL = "postgresql://..."
   MONTHLY_BUDGET_USD = "10"
   QUESTIONS_PER_VISIT = "10"
   ```

5. Deploy. Streamlit installs the dependencies from `uv.lock`.

The first visitor to pick the UK retail data after a restart waits a minute or two while the page downloads it from UCI. Community Cloud puts an app to sleep after a few days without visitors, and the next visitor wakes it up.

## 4. Check that it works

1. Open the page, pick the coffee data, and click an example question.
2. Open **How it worked** to see the code it ran, then give the answer a thumbs up.
3. On your computer, add the same `DATABASE_URL` to `.env` and run:

   ```bash
   uv run python scripts/usage_report.py
   ```

   The question should be there, with its cost and your rating.

## Reading the log

```bash
uv run python scripts/usage_report.py                  # this month: questions, cost, time, ratings, visits
uv run python scripts/usage_report.py --since 2026-10-01
```

Live questions have no answer key, so to measure accuracy on real traffic, grade a random sample by hand:

```bash
uv run python scripts/usage_report.py --sample 20
```

That writes `out/grading/sample_<date>.csv` with each question, the answer and the code. Put `y` or `n` in the `correct` column, then:

```bash
uv run python scripts/usage_report.py --grades out/grading/sample_<date>.csv
```

## What it costs

- **Claude:** about 3 to 5 cents per answer on Sonnet 5.5: the evals measured $0.025 on the retail data and $0.039 on the coffee data. A $10 budget covers roughly 250 answers.
- **Code execution:** billed by container time, but each organization gets 1,550 free hours a month, far more than this page uses.
- **Streamlit Community Cloud and Neon's free tier:** nothing.

## If something goes wrong

- **"This page has no API key set":** the secrets are missing or misspelled. Fix them in the app's settings on share.streamlit.io, then reboot the app.
- **The deploy log says the health check failed:** the likely cause is `address = "localhost"` in `.streamlit/config.toml`, which keeps the local page private to your computer. Remove that line, and run the local page with `uv run streamlit run app.py --server.address localhost` from then on.
- **Answers stop with "Something went wrong on our side":** the real error is in the `error` column of the log. A spend limit reached in the Console shows up there.
