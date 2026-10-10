# LinkedIn post: the data analyst agent

Attach a short screen recording of a live run on the real data, like last time (30-60 seconds).
A post can carry one video, or images, not both. LinkedIn may not animate an uploaded GIF, so a
video is safer. See "Before you post" for what to record.

---

## Post

I built an AI data analyst and tested it on 541,909 real transactions from a UK online shop. It passed all 15 of my questions on every run. Along the way, 3 of the 4 failures I chased turned out to be bugs in my own answer key.

Why now: a cheaper model that's "almost as good" comes out every few weeks. A leaderboard can't tell you if it's good enough for your data. Your own questions can.

The agent answers questions about any CSV in plain English. It writes pandas code, runs it, fixes its own errors and draws charts. The loop is about 70 lines of Python on the Claude API, with no agent framework.

The real data is the UCI Online Retail dataset, and it's messy in all the ways real exports are:

🔁 3,836 cancelled orders, one of them for 80,995 paper crafts, cancelled 12 minutes later
❓ a quarter of rows with no customer ID
📦 postage and Amazon fees mixed in with the products
📅 a last month that stops on 9 December

That's where the traps come from. The "best-selling product" is the cancelled order. The "weakest month" is the one with 9 days of data. "Saturday sales" don't exist. The agent avoided every one.

I also built a made-up dataset with planted patterns, where I know every answer. On its 22 questions, Claude Sonnet 5.5 matched Opus 5.5 at about half the cost after one prompt line, so Sonnet is the default.

My favourite moment: the agent "failed" the best-seller question. It had grouped products by stock code, and said so. One code is sold under two names, so it wins that way. My key had only counted by name. The bug was mine.

What I took away:

1. Real data needs definitions, not just answers. Before or after cancellations? Name or stock code? A fair check accepts each reasonable choice, as long as the answer says which it used.
2. Read failures before trusting the score. 3 of the 4 I chased were bugs in my answer key.
3. Test the grader too. Mine once passed a wrong number because the caveat around it was right.
4. Hold questions back. A fix written after seeing a failure only proves it fixes that failure.
5. Measure on your own data before picking a model. Here, Sonnet matched Opus at half the cost.

Code, all 37 questions with every answer and grade, and a demo are in the first comment.

Measure first. Then pick the model.

How do you test AI on data where the right answer depends on a definition?

#AIAgents #GenAI #LLM #Python

---

## First comment (post right after publishing)

Code, evals and results: https://github.com/adityakannan-2003/data-analyst-agent

All 37 questions (15 on the real data, 22 on a made-up dataset with planted traps), every answer and its grade are in the repo, along with the script that tests the grader. The tests run on GitHub on every push and need no API key.

The real data is the UCI Online Retail dataset (Chen, 2015, CC BY 4.0). A script in the repo downloads it. Running the agent needs an Anthropic API key, and a question costs about 2 cents on Sonnet 5.5.

---

## Other opening lines (swap in for the first line)

- "My AI agent's best-selling product was a single order for 80,995 paper crafts. It was cancelled 12 minutes later. The agent noticed."
- "My AI agent failed my test. It was right. My answer key was wrong. That happened 3 times out of 4."
- "I tested an AI data analyst on 541,909 messy, real transactions. The hardest part wasn't the AI. It was deciding what the right answer was."

---

## Before you post

1. **Record the video.** Download the data with `uv run python scripts/fetch_online_retail.py`, then run
   `uv run python agent.py data/online_retail.csv`. Ask "Which product sold the most units?" and record
   it in QuickTime. It shows the agent spotting the cancelled 80,995-unit order. Trim the dead time with
   Edit → Trim. If you'd rather post an image, use `docs/price-increase-chart.png`.
2. **Link placement.** The link is in the first comment, because LinkedIn tends to show posts with
   outside links in the body to fewer people. If you'd rather have it in the post, replace the
   "Code, all 37 questions..." line with the link.
3. **Timing.** Post on a weekday morning in your audience's time zone, and reply to early comments in
   the first hour.

Every number in the post comes from the README and the saved eval results in the repo.

---

## Project description (detailed)

For a portfolio page, a CV, or the description of a GitHub or LinkedIn project.

**Data Analyst Agent: ask questions about any CSV in plain English**

An AI agent that answers questions about spreadsheet data. You ask something like "Which product sold the most units?", and the agent writes pandas code, runs it, reads the output and fixes its own errors until it can answer. It draws charts when they help and looks at each chart it draws to check it's readable. Every number in an answer comes from code the agent actually ran.

*How it's built*

- **A hand-written agent loop.** About 70 lines of Python on the Claude API, with no agent framework. The loop sends the question to the model, runs every tool call in the response, returns all the results in one message, and stops when the model answers. It handles refusals, tool input cut off at the output limit, malformed tool input, a cap on steps, and rolling back the conversation after an error or Ctrl+C.
- **Two tools.** `run_python` runs code against the data, and `save_chart` draws a chart and sends the image back to the model.
- **A sandbox with persistent state.** Code runs in a separate process that keeps its variables between calls, like notebook cells. If code hangs, the process is restarted after 60 seconds. Errors are trimmed to the agent's own lines so it can see what to fix.
- **Context up front.** The system prompt includes a profile of the data: column types, missing values, distinct counts and examples. That saves a tool call on every question.
- **Default model chosen by evidence.** Claude Sonnet 5.5, picked after the eval below showed it matched Claude Opus 5.5 at about half the cost.

*How it's tested*

- **Two evals, 37 questions.** One uses a made-up coffee-chain dataset (33,237 rows) with planted patterns and planted data problems, so every answer is known: 22 questions, including 5 held back until after the last prompt change. The other uses the real UCI Online Retail dataset: 541,909 transactions from a UK online shop, with 15 questions.
- **Answer keys computed from the data, never typed in.** Where a fair cleaning choice changes the answer, such as before or after cancellations, or product name vs. stock code, the check accepts each fair reading as long as the answer says which it used.
- **A model as grader, and a test of the grader.** Claude Sonnet 5.5 grades each answer against concrete checks, using structured output. A separate script confirms that correct answers pass and that wrong ones fail: empty answers, "I don't know", answers to a different question, and the real mistakes agents made.
- **A careful runner.** Repeated runs, resume after a crash, a time limit per question, failed runs kept out of the score, a check that the expected model answered, and re-grading saved answers after a check changes.
- **Code tests.** 16 unit tests, including a fake model client that drives the agent loop with no API key. They run on GitHub Actions on every push.

*Results*

- **Real data:** 15 of 15 questions passed on every run, at $0.017 and about 9 seconds per answer. The agent avoided every trap: a cancelled 80,995-unit order posing as the best seller, postage posing as the top product, a month with 9 days of data posing as the weakest, and Saturday sales that don't exist.
- **Made-up data:** 22 of 22 passed on every run.
- **Model comparison:** Sonnet 5.5 needed one prompt line to match Opus 5.5, about averaging products that weren't on sale for the whole period. After that it matched Opus at about half the cost and 40% faster. Re-grading Sonnet's answers with Opus as the grader agreed on 63 of 64 checks.
- **What the evals taught:** 3 of the 4 failures I investigated were bugs in my answer key, not the agent. The grader was also too lenient twice, and both cases are now regression tests.

*Tools:* Python, Claude API (Anthropic SDK), pandas, matplotlib, Pillow, pytest, uv, GitHub Actions.

*Code:* https://github.com/adityakannan-2003/data-analyst-agent

---

## Plain-English summary

I built a program you can ask questions about a spreadsheet in everyday language, like "Which product sold the most?". It answers by working the numbers out itself, using an AI model (Claude) that writes and runs small pieces of code, rather than guessing.

To check it really works, I gave it two exams. One used made-up sales data where I knew every answer. The other used more than half a million real sales records from a UK online shop, full of the mess real data has: cancelled orders, missing customer details and shipping charges mixed in with products. It passed every question on both, and it spotted the traps, like a huge order that looked like the best seller but was cancelled minutes later.

The exams also showed that a cheaper AI model did as well as the expensive one, so the program now uses the cheaper one. And when the AI seemed to get something wrong, it was usually my answer sheet that was wrong, not the AI.
