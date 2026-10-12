"""The public web page: anyone with the link can ask questions, on your API key, within a budget.

    uv run streamlit run hosted_app.py

How it differs from app.py, which runs on your own computer:
- The agent's code runs in Anthropic's sandbox (cloud_agent.py), never on this server.
- One setup, the one the evals measured: Sonnet 5.5 at high effort.
- A monthly budget, and a limit on questions per visit.
- Every question is recorded (usage_log.py), and visitors can rate each answer.
- Uploaded files are deleted from Anthropic's Files API when the visitor moves on, or within a day.

Settings, from Streamlit secrets or environment variables:
    ANTHROPIC_API_KEY      a key for a workspace used only by this page (see DEPLOY.md)
    DATABASE_URL           Postgres for the usage log; without it, a local SQLite file
    MONTHLY_BUDGET_USD     stop answering once this month's spend reaches it (default 10)
    QUESTIONS_PER_VISIT    default 10
    MAX_UPLOAD_MB          default 20
"""

import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import MODEL
from app import DATASETS, EXAMPLES, UPLOAD, latest_charts, md
from cloud_agent import CloudAgent, safe_filename
from pricing import PRICES, cost, usage_dict
from sandbox import comment_lines
from usage_log import UsageLog, month_start

ROOT = Path(__file__).resolve().parent
EFFORT = "high"
HELD_BACK_USD = 0.15  # kept in reserve per question, so a run that starts near the budget can't go far past it
REPO_URL = "https://github.com/adityakannan-2003/data-analyst-agent"
STEP_LABELS = {"bash": "Ran code", "write_file": "Wrote a script", "edit_file": "Edited a script", "view_file": "Checked a chart"}
SOURCES = {
    "UK online retailer (real)": "UCI Online Retail dataset (Chen, 2015, CC BY 4.0): 541,909 transactions from a UK gift shop.",
    "Coffee chain (made up)": "A year of made-up sales at three coffee shops, with problems planted on purpose.",
    "Stock prices (real)": "Monthly prices for eight companies and two indexes, 1990 to 2022, from Yahoo Finance.",
}


def setting(name: str, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file, as when running locally
        value = None
    return value if value not in (None, "") else os.environ.get(name, default)


@st.cache_resource
def api() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=setting("ANTHROPIC_API_KEY"), max_retries=4)


@st.cache_resource
def usage_log() -> UsageLog:
    return UsageLog(setting("DATABASE_URL"))


@st.cache_resource(show_spinner="Downloading the UK retail data. This happens once after the page restarts and takes a minute or two...")
def ensure_dataset(path: str) -> bool:
    if not Path(path).exists() and Path(path).name == "online_retail.csv":
        from scripts.fetch_online_retail import main as fetch

        try:
            fetch()
        except SystemExit:  # the download didn't match its checksum
            return False
    return Path(path).exists()


@st.cache_resource
def shared_uploads() -> set[str]:
    """The file ids of the sample datasets uploaded since the server started."""
    return set()


@st.cache_resource(show_spinner="Getting the data ready...")
def shared_upload(path: str) -> str:
    """A sample dataset goes up to the Files API the first time a visitor picks it after the server
    starts, and every visitor shares that upload."""
    file_id = api().files.upload(file=Path(path)).id
    shared_uploads().add(file_id)
    return file_id


@st.cache_data(ttl=3600, show_spinner=False)
def sweep_old_files(keep: tuple[str, ...]) -> int:
    """Delete files older than a day from the Files API, apart from the shared sample datasets.
    A visitor's upload is normally deleted when they switch data or start over; this catches the
    rest. It assumes the API key's workspace is used only by this page (see DEPLOY.md)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    deleted = 0
    for file in api().files.list():
        if file.id not in keep and file.created_at < cutoff:
            api().files.delete(file.id)
            deleted += 1
    return deleted


def show_steps(steps: list[dict]) -> None:
    for step in steps:
        if step["type"] == "tool_call":
            st.markdown(f"**{STEP_LABELS.get(step['name'], 'Used a tool')}**")
            st.code(step["code"], language="bash" if step["name"] == "bash" else "python")
        elif step["type"] == "tool_result":
            output = step["output"].strip() or "(no output)"
            st.caption(("✓ " if step["ok"] else "✗ ") + ("output" if step["ok"] else "error"))
            st.code(output[:2000] + ("\n..." if len(output) > 2000 else ""), language=None)
        elif step["type"] == "note":
            st.markdown(f"_{md(step['text'])}_")


def save_feedback(question_id: str, key: str) -> None:
    value = st.session_state.get(key)  # st.feedback: 1 thumbs up, 0 thumbs down, None cleared
    usage_log().set_feedback(question_id, {1: 1, 0: -1}.get(value))


def show_answer(message: dict) -> None:
    calls = sum(step["type"] == "tool_call" for step in message["steps"])
    if calls:
        with st.expander(f"How it worked: {calls} step{'s' if calls != 1 else ''}"):
            show_steps(message["steps"])
    if message["status"] == "refused":
        st.warning("The model declined this request. Try asking it another way.")
    elif message["status"] == "error":
        st.error("Something went wrong on our side, so there's no answer. Try again in a minute.")
    elif message["status"] == "max_steps":
        st.warning("The agent stopped before it found an answer. Try a narrower question.")
    if message["content"]:
        st.markdown(md(message["content"]))
    if message["status"] == "truncated":
        st.warning("The answer was cut off at the length limit.")
    for chart in message["charts"]:
        st.image(chart)
    if message.get("id"):
        key = f"feedback_{message['id']}"
        st.feedback("thumbs", key=key, on_change=save_feedback, args=(message["id"], key))


def get_agent(csv_path: Path, file_id: str | None) -> CloudAgent:
    """One agent per visitor, rebuilt when the data changes or they start over."""
    key = (str(csv_path), csv_path.stat().st_mtime, file_id)  # a new upload under the same name starts over
    if st.session_state.get("agent_key") != key:
        if "agent" in st.session_state:
            st.session_state.agent.close()  # deletes the previous upload, if the visitor made one
        out_dir = Path(tempfile.gettempdir()) / "analyst-charts" / uuid.uuid4().hex
        out_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.agent = CloudAgent(csv_path, out_dir, MODEL, EFFORT, quiet=True, file_id=file_id, client=api())
        st.session_state.agent_key = key
        st.session_state.chat = []
    return st.session_state.agent


def ask(agent: CloudAgent, question: str, dataset: str) -> dict:
    """Run one question, showing the steps live; record it; return the message to keep in the chat."""
    status = st.status("Working on it. Questions usually take 20 to 60 seconds...", expanded=False)
    answer_box = st.empty()
    steps, charts, streamed, note = [], [], [], []

    def on_event(event: dict) -> None:
        if event["type"] == "text":
            streamed.append(event["text"])
            answer_box.markdown(md("".join(streamed)) + " ▌")
        elif event["type"] == "note":
            note.append(event["text"])
        elif event["type"] == "tool_call":
            if note:  # a progress note introduces the tool call that follows it
                steps.append({"type": "note", "text": "".join(note)})
                status.markdown(f"_{md(''.join(note))}_")
                note.clear()
            streamed.clear()  # text before a tool call was narration, not the answer
            answer_box.empty()
            steps.append(event)
            calls = sum(step["type"] == "tool_call" for step in steps)
            status.update(label=f"Step {calls}: {STEP_LABELS.get(event['name'], 'working').lower()}...")
            with status:
                show_steps([event])
        elif event["type"] == "tool_result":
            steps.append(event)
            with status:
                show_steps([event])
        elif event["type"] == "chart":
            charts.append(event["path"])

    agent.on_event = on_event
    start, finished, t0 = len(agent.messages), False, time.monotonic()
    try:
        answer = agent.ask(question)
        finished = True
    finally:
        if not finished:  # the visitor left or reran the page midway; drop the half-finished turn
            del agent.messages[start:]
    latency = time.monotonic() - t0
    calls = sum(step["type"] == "tool_call" for step in steps)
    status.update(label=f"Done in {calls} step{'s' if calls != 1 else ''}", state="complete" if answer.status == "ok" else "error")
    answer_box.empty()

    usage = usage_dict(answer.usage)
    served = answer.models[-1] if answer.models else MODEL
    # A fallback model without a price here is costed at the dearest price we know, so the budget never undercounts.
    spent = cost(served, usage)
    if spent is None:
        spent = max(cost(model, usage) for model in PRICES)
    # Keep the code for every question, but what it printed only for the sample datasets:
    # for a visitor's own file, that output is their data.
    logged_steps = [
        {"type": s["type"], "name": s.get("name"), "code": s.get("code"), "ok": s.get("ok"),
         "output": (s.get("output") or "")[:1000] if dataset != UPLOAD else None, "text": s.get("text")}
        for s in steps
    ]
    try:
        question_id = usage_log().record(
            session_id=st.session_state.session_id, dataset="upload" if dataset == UPLOAD else dataset,
            data_rows=agent.shape[0], data_cols=agent.shape[1], question=question, answer=answer.text,
            status=answer.status, error=answer.error, steps=logged_steps, model=served,
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
            cache_read_tokens=usage["cache_read_input_tokens"], cache_write_tokens=usage["cache_creation_input_tokens"],
            tool_calls=answer.usage.tool_calls, cost_usd=spent, latency_s=round(latency, 1),
        )
    except Exception as exc:  # the visitor still gets their answer; the server log says what went wrong
        print(f"Couldn't record the question: {exc!r}", file=sys.stderr)
        question_id = None
    message = {"role": "assistant", "id": question_id, "content": answer.text, "status": answer.status,
               "steps": steps, "charts": latest_charts(charts)}
    show_answer({**message, "steps": []})  # the steps are already in the status box above
    return message


def sidebar() -> tuple[str, Path | None]:
    with st.sidebar:
        st.header("Data")
        # Start on the coffee data: it's small, so a first visit after the page wakes up doesn't wait on the
        # 45 MB retail download.
        choice = st.radio("Dataset", [*DATASETS, UPLOAD], index=list(DATASETS).index("Coffee chain (made up)"),
                          label_visibility="collapsed")
        csv_path = None
        if choice == UPLOAD:
            limit = float(setting("MAX_UPLOAD_MB", 20))
            uploaded = st.file_uploader(f"CSV file, up to {limit:g} MB", type="csv")
            if uploaded is not None:
                if uploaded.size > limit * 1_000_000:
                    st.error(f"That file is {uploaded.size / 1e6:.0f} MB. This demo takes files up to {limit:g} MB.")
                else:
                    folder = Path(tempfile.gettempdir()) / "analyst-uploads" / st.session_state.session_id
                    folder.mkdir(parents=True, exist_ok=True)
                    csv_path = folder / safe_filename(uploaded.name)
                    if not csv_path.exists() or csv_path.stat().st_size != uploaded.size:
                        csv_path.write_bytes(uploaded.getvalue())
        else:
            csv_path = DATASETS[choice]
            st.caption(SOURCES[choice])
        if st.button("New conversation"):
            st.session_state.pop("agent_key", None)

        st.divider()
        st.markdown(
            "**What this page keeps.** Your questions, the answers, the code the agent ran, what each answer cost "
            "and how long it took, and your ratings. I use them to see how people use the agent and where it goes "
            "wrong. For files you upload, what the code printed isn't kept, and the file is deleted from Anthropic's "
            "servers when you switch data or start over, or within a day. Please don't upload personal or "
            "confidential data."
        )
        st.markdown(f"The agent's code runs in Anthropic's sandbox, never on this server. [Code and evals on GitHub]({REPO_URL})")
    return choice, csv_path


def main() -> None:
    st.set_page_config(page_title="Data Analyst Agent", page_icon="📊")
    load_dotenv(ROOT / ".env")  # when trying the page locally; the hosted page uses its secrets
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("asked", 0)
    choice, csv_path = sidebar()

    st.title("Data Analyst Agent")
    if not setting("ANTHROPIC_API_KEY"):
        st.error("This page has no API key set. Add ANTHROPIC_API_KEY to its secrets.")
        st.stop()
    if csv_path is None:
        st.info("Choose a CSV file in the sidebar to start.")
        st.stop()
    if not ensure_dataset(str(csv_path)):
        st.error("This dataset couldn't be downloaded. Pick another one.")
        st.stop()

    file_id = None if choice == UPLOAD else shared_upload(str(csv_path))
    sweep_old_files(tuple(sorted(shared_uploads())))
    try:
        agent = get_agent(csv_path, file_id)
    except Exception as exc:  # a file pandas can't read
        st.error(f"That file couldn't be read as a CSV: {exc}")
        st.stop()

    rows, cols = agent.shape
    st.caption(f"Ask about **{csv_path.name}** ({rows:,} rows × {cols} columns) in plain English. The agent "
               "writes and runs Python to work out every number, and shows its code. Follow-up questions "
               "remember the conversation.")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(md(message["content"]))
            else:
                show_answer(message)

    budget = float(setting("MONTHLY_BUDGET_USD", 10))
    per_visit = int(setting("QUESTIONS_PER_VISIT", 10))
    blocked = None
    try:
        spent = usage_log().spent_since(month_start())
    except Exception as exc:  # without the log, the page can't keep to its budget, so it pauses
        print(f"Couldn't read the usage log: {exc!r}", file=sys.stderr)
        spent = None
    if spent is None:
        blocked = "This demo can't check its budget right now, so it's paused. Please try again later."
    elif spent + HELD_BACK_USD > budget:
        blocked = (f"This demo has used its budget for the month, so it's resting until the 1st. "
                   f"You can run it yourself with your own API key: [code on GitHub]({REPO_URL}).")
    elif st.session_state.asked >= per_visit:
        blocked = (f"That's {per_visit} questions, the limit for one visit to this demo. "
                   f"To keep going, run it yourself with your own API key: [code on GitHub]({REPO_URL}).")
    if blocked:
        st.info(blocked)

    clicked = None
    if not st.session_state.chat:
        with st.expander("Preview the data"):
            st.dataframe(pd.read_csv(csv_path, skiprows=len(comment_lines(csv_path)), nrows=20))
        st.write("Try one of these, or type your own question below:")
        clicked = [q for q in EXAMPLES[choice] if st.button(q, key=q, disabled=bool(blocked))]
    question = st.chat_input("Ask a question about the data", disabled=bool(blocked)) or (clicked[0] if clicked else None)
    if question and not blocked:
        st.session_state.asked += 1
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            st.session_state.chat.append(ask(agent, question, choice))
        st.rerun()  # redraw with the feedback buttons live and the example buttons gone


if __name__ == "__main__":
    main()
