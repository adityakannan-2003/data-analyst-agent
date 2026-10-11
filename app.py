"""A web page for the data analyst agent.

    uv run streamlit run app.py

Pick a dataset or upload a CSV in the sidebar, then ask questions in the chat box. Each
answer shows the steps the agent took (the code it ran and what came back), and its
charts appear under the answer. It runs on your computer and uses your API key.
"""

import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import MODEL, Agent, has_credentials
from sandbox import comment_lines

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "UK online retailer (real)": ROOT / "data" / "online_retail.csv",
    "Coffee chain (made up)": ROOT / "data" / "coffee_sales.csv",
    "Stock prices (real)": ROOT / "data" / "stocks.csv",
}
UPLOAD = "Upload your own CSV"
MODELS = {"Sonnet 5.5": MODEL, "Opus 5.5 (more careful, about 2x the cost)": "claude-opus-5-5"}
EXAMPLES = {
    "UK online retailer (real)": [
        "Which product sold the most units?",
        "Show me monthly sales as a chart.",
        "Which countries buy the most, apart from the UK?",
    ],
    "Coffee chain (made up)": [
        "Did the April price increase hurt latte sales?",
        "Which store makes the most revenue?",
        "When do iced drinks sell best? Show me a chart.",
    ],
    "Stock prices (real)": [
        "Which stock gave investors the best return?",
        "How did these stocks do in 2008?",
        "Chart Apple against the S&P 500.",
    ],
    UPLOAD: ["What's in this data?", "Are there any data quality problems?", "What stands out?"],
}


def md(text: str) -> str:
    """Show the agent's plain text as Markdown: keep its line breaks, and stop "$5 to $6" turning into maths."""
    return text.replace("$", "\\$").replace("\n", "  \n")


def latest_charts(paths: list[str]) -> list[str]:
    """The agent sometimes redraws a chart to fix it. Keep the last version of each chart name."""
    latest = {}
    for path in paths:
        latest[re.sub(r"^chart_\d+_", "", Path(path).name)] = path
    return list(latest.values())


def show_steps(steps: list[dict]) -> None:
    for step in steps:
        if step["type"] == "tool_call":
            st.markdown(f"**{'Drew a chart' if step['name'] == 'save_chart' else 'Ran code'}**")
            st.code(step["code"], language="python")
        elif step["type"] == "tool_result":
            output = step["output"].strip() or "(no output)"
            st.caption(("✓ " if step["ok"] else "✗ ") + ("output" if step["ok"] else "error"))
            st.code(output[:2000] + ("\n..." if len(output) > 2000 else ""), language=None)
        elif step["type"] == "note":
            st.markdown(f"_{md(step['text'])}_")


def show_answer(message: dict) -> None:
    calls = sum(step["type"] == "tool_call" for step in message["steps"])
    if calls:
        with st.expander(f"How it worked: {calls} step{'s' if calls != 1 else ''}"):
            show_steps(message["steps"])
    if message["status"] == "refused":
        st.warning("The model declined this request.")
    elif message["status"] == "error":
        st.error(f"Something went wrong: {message['error']}")
    elif message["status"] == "max_steps":
        st.warning("The agent stopped after 25 steps without a final answer.")
    if message["content"]:
        st.markdown(md(message["content"]))
    if message["status"] == "truncated":
        st.warning("The answer was cut off at the output limit.")
    for chart in message["charts"]:
        st.image(chart)
    st.caption(message["usage"])


def get_agent(csv_path: Path, model: str, effort: str) -> Agent:
    """One agent per browser session, rebuilt when the data or model changes."""
    key = (str(csv_path), csv_path.stat().st_mtime, model, effort)
    if st.session_state.get("agent_key") != key:
        if "agent" in st.session_state:
            st.session_state.agent.sandbox.close()
        with st.spinner("Loading the data..."):
            out_dir = ROOT / "out" / "web" / uuid.uuid4().hex[:8]  # per session, so charts never collide
            out_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.agent = Agent(csv_path, out_dir, model=model, effort=effort, quiet=True)
        st.session_state.agent_key = key
        st.session_state.chat = []
    return st.session_state.agent


def ask(agent: Agent, question: str) -> dict:
    """Run one question, showing the steps live, and return the message to keep in the chat."""
    status = st.status("Working on it...", expanded=False)
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
            status.update(label=f"Step {calls}: {'drawing a chart' if event['name'] == 'save_chart' else 'running code'}...")
            with status:
                show_steps([event])
        elif event["type"] == "tool_result":
            steps.append(event)
            with status:
                show_steps([event])
        elif event["type"] == "chart":
            charts.append(event["path"])

    agent.on_event = on_event
    start, finished = len(agent.messages), False
    try:
        answer = agent.ask(question)
        finished = True
    finally:
        if not finished:  # the page stopped this run midway; drop the half-finished turn
            del agent.messages[start:]
    calls = sum(step["type"] == "tool_call" for step in steps)
    status.update(label=f"Done in {calls} step{'s' if calls != 1 else ''}", state="complete" if answer.status == "ok" else "error")
    answer_box.empty()
    message = {"role": "assistant", "content": answer.text, "status": answer.status, "error": answer.error,
               "steps": steps, "charts": latest_charts(charts), "usage": str(answer.usage)}
    show_answer({**message, "steps": []})  # the steps are already in the status box above
    return message


def main() -> None:
    st.set_page_config(page_title="Data Analyst Agent", page_icon="📊")
    load_dotenv(ROOT / ".env")

    with st.sidebar:
        st.header("Data")
        choice = st.radio("Dataset", [*DATASETS, UPLOAD], label_visibility="collapsed")
        if choice == UPLOAD:
            uploaded = st.file_uploader("CSV file", type="csv")
            if uploaded is None:
                st.info("Choose a CSV file to start.")
                st.stop()
            csv_path = ROOT / "out" / "uploads" / Path(uploaded.name).name
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not csv_path.exists() or csv_path.stat().st_size != uploaded.size:
                csv_path.write_bytes(uploaded.getvalue())
        else:
            csv_path = DATASETS[choice]
        st.header("Model")
        model = MODELS[st.radio("Model", list(MODELS), label_visibility="collapsed")]
        effort = st.select_slider("Effort", ["low", "medium", "high", "xhigh", "max"], value="high",
                                  help="How much the model thinks before acting. Higher is more careful and costs more.")
        if st.button("New conversation"):
            st.session_state.pop("agent_key", None)
        st.caption("Each question costs about 2 cents with Sonnet, on your API key. The agent runs the code it "
                   "writes on this computer, so only use files you trust.")

    if not has_credentials():
        st.error("No Anthropic API key found. Copy .env.example to .env and paste your key there, then reload.")
        st.stop()
    if not csv_path.exists():
        st.error("This dataset isn't downloaded yet. In a terminal, run: uv run python scripts/fetch_online_retail.py")
        st.stop()

    agent = get_agent(csv_path, model, effort)
    rows, cols = agent.shape
    st.title("Data Analyst Agent")
    st.caption(f"Ask about **{csv_path.name}** ({rows:,} rows × {cols} columns) in plain English. "
               "Every number comes from code the agent runs. Follow-up questions remember the conversation.")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(md(message["content"]))
            else:
                show_answer(message)

    clicked = None
    if not st.session_state.chat:
        with st.expander("Preview the data"):
            st.dataframe(pd.read_csv(csv_path, skiprows=len(comment_lines(csv_path)), nrows=20))
        st.write("Try one of these, or type your own question below:")
        clicked = [q for q in EXAMPLES[choice] if st.button(q, key=q)]  # a list, so every button is drawn
    question = st.chat_input("Ask a question about the data") or (clicked[0] if clicked else None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            st.session_state.chat.append(ask(agent, question))
        if clicked:
            st.rerun()  # redraw without the example buttons


# The guard matters: the sandbox starts a new Python process, which re-imports this file
# under another name. Without it, that copy would try to start the whole app again.
if __name__ == "__main__":
    main()
