"""Tests for the eval runner's --loop option, with a fake model and no judge (no API calls)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_graph_agent import FakeModel, call, turn

from agent import MODEL
from graph_agent import GraphAgent

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))  # run_eval imports the answer keys from there
import run_eval  # noqa: E402


def test_a_langgraph_run_saves_its_loop_turns_and_trace(csv_path, tmp_path, monkeypatch):
    def scripted_agent(*args, **kwargs):
        made = GraphAgent(*args, **kwargs)
        made.llm = FakeModel([turn(call("t1", "run_python", code="df.sales.sum()"), text="Let me add them up."),
                              turn(text="Total sales are 22.")])
        return made

    monkeypatch.setattr(run_eval, "GraphAgent", scripted_agent)
    monkeypatch.setattr(run_eval, "CSV", csv_path)
    monkeypatch.setattr(run_eval, "FLOW", tmp_path)
    monkeypatch.setattr(run_eval, "grade", lambda row, *rest: row)  # no judge
    args = SimpleNamespace(executor="local", loop="langgraph", model=MODEL, effort="high", judge_model=None)
    case = {"id": "total", "question": "What are total sales?", "tags": []}

    row, trace = run_eval.run_case(case, 0, args, tmp_path / "langgraph-v1", None, {})

    assert (row["meta"]["loop"], row["model_turns"], row["tool_calls"]) == ("langgraph", 2, 1)
    assert row["meta"]["answer"] == "Total sales are 22."
    # The turn's text comes before its tool call, as the model sent them.
    assert [t["role"] for t in trace] == ["system", "user", "assistant", "tool_call", "tool_result", "assistant"]
    assert trace[4]["content"] == "22\n"


@pytest.mark.parametrize("flags", [
    ["--loop", "langgraph", "--variant", "v5"],             # LangGraph results in a hand-written folder
    ["--variant", "langgraph-v1"],                           # hand-written results in a LangGraph folder
    ["--loop", "langgraph", "--variant", "langgraph-v1", "--executor", "cloud"],
])
def test_the_two_loops_never_share_a_results_folder(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *flags])
    with pytest.raises(SystemExit) as stopped:
        run_eval.main()
    assert "--" in str(stopped.value)  # a message that says which option to change, not a bare exit code
