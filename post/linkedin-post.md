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
