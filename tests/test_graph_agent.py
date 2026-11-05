"""Tests for the LangGraph loop, with a fake model in place of ChatAnthropic."""

import json
import signal
from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

pytest.importorskip("langgraph", reason="the LangGraph loop's packages aren't installed: uv sync --all-groups")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402
from test_agent import FakeClient  # noqa: E402
from test_agent import text as hand_text  # noqa: E402
from test_agent import turn as hand_turn  # noqa: E402

from agent import MAX_STEPS, MODEL, Agent, StreamPrinter  # noqa: E402
from graph_agent import GraphAgent, TurnPrinter, api_usage  # noqa: E402


class FakeModel:
    """Returns scripted model turns in order (or raises the ones that are exceptions) and records
    the messages each turn was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(list(messages))
        result = self.turns.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def turn(*tool_calls, text="", stop_reason=None):
    """A model turn shaped like the ones ChatAnthropic streams: each tool call's input arrives as raw JSON."""
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": {}, "partial_json": c["raw"]}
                for c in tool_calls]
    return AIMessage(
        content=content,
        response_metadata={"stop_reason": stop_reason or ("tool_use" if tool_calls else "end_turn"), "model_name": MODEL},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        "input_token_details": {"cache_read": 0, "cache_creation": 0}},
    )


def call(call_id, tool, raw=None, **args):
    """A tool call in a turn: its input as the model streamed it (raw JSON), by default the JSON of args."""
    return {"id": call_id, "name": tool, "raw": json.dumps(args) if raw is None else raw}


# Tool input JSON the SDK's parser rejects (a trailing comma).
UNPARSEABLE = '{"code": "1",}'


@pytest.fixture
def agent(csv_path):
    a = GraphAgent(csv_path, csv_path.parent, quiet=True)
    yield a
    a.close()


def test_runs_tools_until_the_model_answers(agent):
    agent.llm = FakeModel([
        turn(call("t1", "run_python", code="df.sales.sum()")),
        turn(text="Total sales are 22."),
    ])
    answer = agent.ask("What are total sales?")

    assert (answer.text, answer.status, answer.usage.tool_calls) == ("Total sales are 22.", "ok", 1)
    assert type(answer.text) is str  # not LangChain's str subclass, so callers get what Agent gives them
    assert answer.models == [MODEL, MODEL]
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    result = agent.messages[2]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": "22\n", "is_error": False}
    first, second = agent.llm.requests
    assert first[0].content == agent.system  # the system prompt goes first on every turn
    assert [m.type for m in second] == ["system", "human", "ai", "tool"]


def test_the_conversation_is_saved_once_per_question(agent):
    # By default LangGraph saves a full copy of the conversation after every step, so memory grows fast.
    agent.llm = FakeModel([turn(call("t1", "run_python", code="1")), turn(text="One."), turn(text="Two.")])
    agent.ask("First question")
    agent.ask("Second question")

    assert len(list(agent.graph.checkpointer.list(agent.config))) == 2


def test_follow_up_questions_see_the_conversation(agent):
    agent.llm = FakeModel([turn(text="22."), turn(text="Paris.")])
    agent.ask("What are total sales?")
    agent.ask("Which city sells most?")

    sent = agent.llm.requests[1][1:]  # the second question's request, after the system prompt
    assert [m.text for m in sent] == ["What are total sales?", "22.", "Which city sells most?"]


def test_cache_writes_are_counted_when_split_by_lifetime():
    message = AIMessage("", usage_metadata={
        "input_tokens": 1100, "output_tokens": 50, "total_tokens": 1150,
        "input_token_details": {"cache_read": 600, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 0},
    })
    usage = api_usage(message)
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (100, 600, 400)
    assert usage.output_tokens == 50


def test_request_settings_match_the_hand_written_loop(csv_path):
    """ChatAnthropic must send what Agent._call_model sends, so the loop is the only difference."""
    hand = Agent(csv_path, csv_path.parent, quiet=True)
    hand.client = FakeClient([hand_turn(hand_text("Hi."))])
    hand.ask("Hi")
    hand.close()
    expected = {k: v for k, v in hand.client.requests[0].items() if k != "messages"}

    graph = GraphAgent(csv_path, csv_path.parent, quiet=True)
    # A private method, so this test fails loudly if a LangChain update changes what gets sent.
    payload = graph.llm.bound._get_request_payload([SystemMessage(graph.system), HumanMessage("Hi")], **graph.llm.kwargs)
    graph.close()
    assert {k: payload.get(k) for k in expected} == expected
    assert graph.llm.bound.default_request_timeout == 600  # the SDK's read timeout; LangChain's default is none


def test_parallel_tool_calls_run_in_order_and_return_in_one_message(agent):
    real_model = agent.llm
    agent.llm = FakeModel([
        turn(call("a", "run_python", code="x = 20"), call("b", "run_python", code="x + 2"),
             call("c", "run_python", code="undefined_name")),
        turn(text="Done."),
    ])
    agent.ask("Three things")

    results = [block for message in agent.messages[2:5] for block in message["content"]]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert [r["content"] for r in results[:2]] == ["(no output)", "22\n"]  # b saw the x that a made
    assert [r["is_error"] for r in results] == [False, False, True]
    # LangChain keeps one message per result, and merges them into one user message when it sends them.
    history = agent.graph.get_state(agent.config).values["messages"]
    sent = real_model.bound._get_request_payload(history, **real_model.kwargs)["messages"]
    assert [b["tool_use_id"] for b in sent[2]["content"]] == ["a", "b", "c"]


def test_invalid_tool_input_is_reported_not_run(agent):
    agent.llm = FakeModel([turn(call("t1", "run_python", script="oops")), turn(text="Sorry.")])
    agent.ask("Question")

    result = agent.messages[2]["content"][0]
    assert result["is_error"] and "Invalid input" in result["content"]


def test_unparseable_tool_input_runs_the_turn_again(agent):
    agent.llm = FakeModel([turn(call("t1", "run_python", raw=UNPARSEABLE)), turn(text="Done.")])
    answer = agent.ask("Question")

    assert (answer.text, answer.models) == ("Done.", [MODEL])  # the broken turn is dropped, like the SDK error
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_unparseable_tool_input_three_times_is_an_error(agent):
    agent.llm = FakeModel([turn(call("t1", "run_python", raw=UNPARSEABLE)) for _ in range(3)])
    answer = agent.ask("Question")

    assert answer.status == "error" and "couldn't be parsed" in answer.error
    assert agent.messages == []


def test_truncated_tool_input_is_not_run(agent):
    agent.llm = FakeModel([
        turn(call("t1", "run_python", code=f"open({str(agent.out_dir / 'should_not_exist')!r}, 'w')"),
             stop_reason="max_tokens"),
        turn(text="Retrying later."),
    ])
    agent.ask("Question")

    assert agent.messages[2]["content"][0]["is_error"]
    assert not (agent.out_dir / "should_not_exist").exists()


def test_chart_comes_back_as_an_image(agent):
    agent.llm = FakeModel([
        turn(call("c1", "save_chart", code="df.sales.plot()", name="Sales over rows")),
        turn(text="Here is the chart."),
    ])
    agent.ask("Chart it")

    result = agent.messages[2]["content"][0]
    assert [part["type"] for part in result["content"]] == ["text", "image"]
    assert (agent.out_dir / "chart_1_sales-over-rows.png").exists()


def test_refusal_rolls_back_the_question(agent):
    agent.llm = FakeModel([turn(text="Earlier answer."), turn(stop_reason="refusal")])
    agent.ask("First question")
    answer = agent.ask("Declined question")

    assert answer.status == "refused"
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_refusal_of_the_first_question_rolls_back(agent):
    agent.llm = FakeModel([turn(stop_reason="refusal"), turn(text="Fine.")])
    assert agent.ask("Declined question").status == "refused"
    assert agent.messages == []

    assert agent.ask("Next question").text == "Fine."
    assert [m.text for m in agent.llm.requests[1][1:]] == ["Next question"]


@pytest.mark.parametrize("failure, status", [
    (KeyboardInterrupt(), "cancelled"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), "error"),
])
def test_cancel_or_api_error_rolls_back_the_question(agent, failure, status):
    agent.llm = FakeModel([turn(text="Earlier answer."), turn(call("t1", "run_python", code="1")), failure,
                           turn(text="Next answer.")])
    agent.ask("First question")
    answer = agent.ask("Interrupted question")

    assert answer.status == status
    assert answer.usage.tool_calls == 1  # work done before the failure still counts, as in Agent.ask
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
    agent.ask("Next question")
    assert [m.text for m in agent.llm.requests[-1][1:]] == ["First question", "Earlier answer.", "Next question"]


def test_stops_after_max_steps_model_turns(agent):
    agent.llm = FakeModel([turn(call(f"t{i}", "run_python", code="1")) for i in range(MAX_STEPS)]
                          + [turn(text="Late answer.")])
    answer = agent.ask("Loop forever")

    assert answer.status == "max_steps"
    assert (len(answer.models), answer.usage.tool_calls) == (MAX_STEPS, MAX_STEPS)
    assert len(agent.llm.turns) == 1  # the next turn was never requested
    # Like Agent.ask, the conversation is kept, and the next question starts a fresh run.
    assert agent.ask("Answer now").text == "Late answer."


def test_quiet_mode_prints_nothing(agent, capsys):
    agent.llm = FakeModel([turn(call("t1", "run_python", code="print('hi')")), turn(text="Done.")])
    assert agent.ask("Question").text == "Done."
    assert capsys.readouterr().out == ""


def test_streamed_turns_print_like_the_hand_written_loop(capsys):
    # One model turn as the SDK streams it to Agent._call_model, printed by StreamPrinter...
    expected = []
    hand = StreamPrinter(out=lambda *parts, end="\n", flush=False: expected.append("".join(parts) + end))
    for event in [
        NS(type="content_block_start", content_block=NS(type="thinking")), NS(type="thinking", thinking="Summing."),
        NS(type="content_block_stop"),
        NS(type="content_block_start", content_block=NS(type="text")), NS(type="text", text="Total sales "),
        NS(type="text", text="are 22."), NS(type="content_block_stop"),
    ]:
        hand.handle(event)
    hand.finish()

    # ...and the same turn as ChatAnthropic hands it to the LangGraph loop's callback, chunk by chunk.
    events = []
    printer = TurnPrinter(print, events.append)
    printer.on_chat_model_start({}, [])
    for content in [
        [],  # the message start
        [{"type": "thinking", "thinking": "Summing.", "index": 0}],
        [{"type": "thinking", "thinking": "", "signature": "s", "index": 0}],
        [{"type": "text", "text": "Total sales ", "index": 1}],
        [{"type": "text", "text": "are 22.", "index": 1}],
    ]:
        printer.on_llm_new_token("", chunk=NS(message=AIMessageChunk(content=content)))
    printer.on_llm_end(None)

    assert capsys.readouterr().out == "".join(expected)
    assert events == [{"type": "note", "text": "Summing."}, {"type": "text", "text": "Total sales "},
                      {"type": "text", "text": "are 22."}]


def test_a_refused_turn_runs_none_of_its_tool_calls(agent):
    # The API can stop a turn as refused after a tool call has streamed. Agent.ask checks for the refusal first.
    agent.llm = FakeModel([turn(text="Earlier answer."),
                           turn(call("t1", "run_python", code=f"open({str(agent.out_dir / 'ran')!r}, 'w')"),
                                stop_reason="refusal")])
    agent.ask("First question")
    answer = agent.ask("Declined question")

    assert answer.status == "refused"
    assert not (agent.out_dir / "ran").exists()
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


@pytest.mark.parametrize("raw, result", [
    ('{"code": "print(1)" "x"}', "1\n"),                          # the SDK keeps the complete part and runs it
    ('{"code": "print(1)', "Invalid input for run_python: {}"),  # and drops a string cut off mid-stream
])
def test_tool_input_is_read_as_the_sdk_reads_it(agent, raw, result):
    agent.llm = FakeModel([turn(call("t1", "run_python", raw=raw)), turn(text="Done.")])
    agent.ask("Question")

    assert agent.messages[2]["content"][0]["content"] == result


def test_ctrl_c_stops_the_model_call_at_once(agent):
    class SlowModel:
        finished = False

        def invoke(self, messages):
            signal.raise_signal(signal.SIGINT)  # what pressing Ctrl+C does, while the model is still streaming
            SlowModel.finished = True
            return turn(text="Too late.")

    agent.llm = FakeModel([turn(text="Earlier answer.")])
    agent.ask("First question")
    agent.llm = SlowModel()
    answer = agent.ask("Interrupted question")

    assert answer.status == "cancelled"
    assert not SlowModel.finished  # the call was stopped where it was, not left to finish in another thread
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
