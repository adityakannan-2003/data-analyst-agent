"""Tests for the agent loop, with a fake model client in place of the API."""

from types import SimpleNamespace as NS

import pytest

from agent import MODEL, Agent


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([])

    def get_final_message(self):
        return self.message


class FakeClient:
    """Returns scripted model turns in order and records each request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return FakeStream(self.turns.pop(0))


def turn(*content, stop_reason="end_turn"):
    usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=list(content), stop_reason=stop_reason, usage=usage, model=MODEL)


def tool_use(block_id, tool, **input):
    return NS(type="tool_use", id=block_id, name=tool, input=input)


def text(value):
    return NS(type="text", text=value)


@pytest.fixture
def agent(csv_path):
    a = Agent(csv_path, csv_path.parent)
    yield a
    a.sandbox.close()


def test_runs_tools_until_the_model_answers(agent):
    agent.client = FakeClient([
        turn(tool_use("t1", "run_python", code="df.sales.sum()"), stop_reason="tool_use"),
        turn(text("Total sales are 22.")),
    ])
    answer = agent.ask("What are total sales?")

    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert (answer.text, answer.status, answer.usage.tool_calls) == ("Total sales are 22.", "ok", 1)
    assert answer.models == [MODEL, MODEL]
    result = agent.messages[2]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": "22\n", "is_error": False}
    assert len(agent.client.requests) == 2


def test_parallel_tool_calls_return_in_one_message(agent):
    agent.client = FakeClient([
        turn(
            tool_use("a", "run_python", code="1 + 1"),
            tool_use("b", "run_python", code="undefined_name"),
            stop_reason="tool_use",
        ),
        turn(text("Done.")),
    ])
    agent.ask("Two things")

    results = agent.messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert [r["is_error"] for r in results] == [False, True]


def test_invalid_tool_input_is_reported_not_run(agent):
    agent.client = FakeClient([
        turn(tool_use("t1", "run_python", script="oops"), stop_reason="tool_use"),
        turn(text("Sorry.")),
    ])
    agent.ask("Question")

    result = agent.messages[2]["content"][0]
    assert result["is_error"] and "Invalid input" in result["content"]


def test_truncated_tool_input_is_not_run(agent):
    agent.client = FakeClient([
        turn(tool_use("t1", "run_python", code="open('should_not_exist', 'w')"), stop_reason="max_tokens"),
        turn(text("Retrying later.")),
    ])
    agent.ask("Question")

    assert agent.messages[2]["content"][0]["is_error"]
    assert not (agent.out_dir / "should_not_exist").exists()


def test_chart_comes_back_as_an_image(agent):
    agent.client = FakeClient([
        turn(tool_use("c1", "save_chart", code="df.sales.plot()", name="Sales over rows"), stop_reason="tool_use"),
        turn(text("Here is the chart.")),
    ])
    agent.ask("Chart it")

    result = agent.messages[2]["content"][0]
    assert [part["type"] for part in result["content"]] == ["text", "image"]
    assert (agent.out_dir / "chart_1_sales-over-rows.png").exists()


def test_refusal_rolls_back_the_question(agent):
    agent.client = FakeClient([
        turn(text("Earlier answer.")),
        turn(stop_reason="refusal"),
    ])
    agent.ask("First question")
    answer = agent.ask("Declined question")

    assert answer.status == "refused"
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_quiet_mode_prints_nothing(csv_path, capsys):
    quiet = Agent(csv_path, csv_path.parent, quiet=True)
    try:
        quiet.client = FakeClient([
            turn(tool_use("t1", "run_python", code="print('hi')"), stop_reason="tool_use"),
            turn(text("Done.")),
        ])
        assert quiet.ask("Question").text == "Done."
    finally:
        quiet.sandbox.close()
    assert capsys.readouterr().out == ""
