"""A private dashboard for the hosted page's usage log: questions per day, cost, time, ratings, errors.

    uv run streamlit run scripts/usage_dashboard.py

It opens the log read-only (the database refuses writes), and .streamlit/config.toml serves it to this
computer only. It isn't part of the public page, which runs hosted_app.py. Reads DATABASE_URL from .env,
like scripts/usage_report.py; without it, the local SQLite log.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.usage_report import percentile  # noqa: E402
from usage_log import UsageLog  # noqa: E402

st.set_page_config(page_title="Usage dashboard", page_icon="📈", layout="wide")
load_dotenv(ROOT / ".env")
st.title("Hosted page usage")
since = st.sidebar.date_input("Since", date.today() - timedelta(days=30))
rows = UsageLog(read_only=True).rows(f"{since}T00:00:00+00:00")
if not rows:
    st.info(f"No questions since {since}.")
    st.stop()

df = pd.DataFrame(rows)
df["day"] = pd.to_datetime(df["asked_at"].str[:10])
df["error"] = df["error"].fillna("")
latencies = df["latency_s"].dropna().tolist()
rated = df["feedback"].notna().sum()
up, down = (df["feedback"] == 1).sum(), (df["feedback"] == -1).sum()

metrics = [
    ("Questions", len(df)), ("Visits", df["session_id"].nunique()), ("Cost", f"${df['cost_usd'].fillna(0).sum():.2f}"),
    ("Median time", f"{percentile(latencies, 0.5):.0f}s"), ("Thumbs up", up), ("Thumbs down", down),
    ("Not ok", (df["status"] != "ok").sum()),
]
for start in range(0, len(metrics), 4):  # rows of four, so the numbers fit a narrow window
    for col, (label, value) in zip(st.columns(4), metrics[start:start + 4]):
        col.metric(label, value)
st.caption(f"${df['cost_usd'].fillna(0).mean():.3f} per question · {percentile(latencies, 0.95):.0f}s at the 95th "
           f"percentile · {rated} of {len(df)} answers rated · {(df['status'] == 'error').sum()} errors")

days = pd.date_range(since, max(date.today(), df["day"].max().date()), name="day")  # days with no questions show as 0
daily = df.groupby("day").agg(
    questions=("id", "size"),
    cost_usd=("cost_usd", "sum"),
    median_s=("latency_s", "median"),
    p95_s=("latency_s", lambda s: s.quantile(0.95)),
    thumbs_up=("feedback", lambda f: (f == 1).sum()),
    thumbs_down=("feedback", lambda f: (f == -1).sum()),
    not_ok=("status", lambda s: (s != "ok").sum()),
).reindex(days).fillna({"questions": 0, "cost_usd": 0, "thumbs_up": 0, "thumbs_down": 0, "not_ok": 0})

left, right = st.columns(2)
left.subheader("Questions per day")
left.bar_chart(daily[["questions"]])
right.subheader("Cost per day (USD)")
right.bar_chart(daily[["cost_usd"]])
left.subheader("Time to answer (seconds)")
left.line_chart(daily[["median_s", "p95_s"]])
right.subheader("Ratings and answers that weren't ok, per day")
right.bar_chart(daily[["thumbs_up", "thumbs_down", "not_ok"]])

st.subheader("By dataset and outcome")
st.dataframe(pd.crosstab(df["dataset"], df["status"], margins=True, margins_name="total"))

st.subheader("Answers that weren't ok")
st.dataframe(df[df["status"] != "ok"][["asked_at", "dataset", "status", "error", "question"]].iloc[::-1],
             hide_index=True, width="stretch")

st.subheader("Thumbs down")
st.dataframe(df[df["feedback"] == -1][["asked_at", "dataset", "question", "answer"]].iloc[::-1],
             hide_index=True, width="stretch")

with st.expander("Daily table"):
    st.dataframe(daily, width="stretch")
