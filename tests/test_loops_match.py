"""Both loops on the same streamed API responses, offline: they must send the same requests, give the same
answers and print the same output, so the loop is the only difference between them in the evals.

The responses are SSE streams shaped like a recorded real one. Unlike tests/test_graph_agent.py, which fakes
the model object, these go through LangChain's real handling of the stream (ChatAnthropic).
"""

import copy
import dataclasses
import json
import signal

import anthropic
import httpx2
import pytest

pytest.importorskip("langgraph", reason="the LangGraph loop's packages aren't installed: uv sync --all-groups")

from agent import MODEL, Agent  # noqa: E402
from graph_agent import GraphAgent  # noqa: E402

USAGE = {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1500,
         "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0}, "output_tokens": 1}


def stream(*blocks, stop="end_turn"):
    """One streamed model turn. message_delta repeats the full token counts, as the API does."""
    events = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
                                                    "content": [], "stop_reason": None, "stop_sequence": None,
                                                    "usage": USAGE}}]
    for i, (start, deltas) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": i, "content_block": start})
        events += [{"type": "content_block_delta", "index": i, "delta": delta} for delta in deltas]
        events.append({"type": "content_block_stop", "index": i})
    usage = {**{k: v for k, v in USAGE.items() if k != "cache_creation"}, "output_tokens": 50}
    events += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": usage},
               {"type": "message_stop"}]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def thinking(note):
    return ({"type": "thinking", "thinking": "", "signature": ""},
            [{"type": "thinking_delta", "thinking": note}, {"type": "signature_delta", "signature": "c2ln"}])


def text(words):
    return {"type": "text", "text": ""}, [{"type": "text_delta", "text": words}]


def tool(call_id, name, raw):
    """A tool call whose input JSON (raw, valid or not) streams in two pieces."""
    half = len(raw) // 2
    return ({"type": "tool_use", "id": call_id, "name": name, "input": {}, "caller": {"type": "direct"}},
            [{"type": "input_json_delta", "partial_json": raw[:half]}, {"type": "input_json_delta", "partial_json": raw[half:]}])


def broken(*blocks):
    """A stream with a malformed event after its first block: both loops get a ValueError and run the turn again."""
    good = stream(*blocks).decode()
    cut = good.index("event: content_block_start", good.index("content_block_stop"))
    return (good[:cut] + "event: content_block_delta\ndata: {not json\n\n").encode()


OVERLOADED = (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


def run(loop, csv_path, monkeypatch, responses, questions, ctrl_c_at=None):
    """Ask the questions with one loop, answering its requests in order from `responses`."""
    sent, printed, events = [], [], []

    def answer(request):
        i = len(sent)
        sent.append(json.loads(request.read()))
        if i == ctrl_c_at:
            signal.raise_signal(signal.SIGINT)  # Ctrl+C while this request is in flight
        status, body = responses[i] if isinstance(responses[i], tuple) else (200, responses[i])
        if status != 200:
            return httpx2.Response(status, json=body)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    transport = httpx2.MockTransport(answer)
    out_dir = csv_path.parent / "out"  # the same folder for both loops: tool results name the chart's path
    out_dir.mkdir(exist_ok=True)
    if loop == "hand-written":
        agent = Agent(csv_path, out_dir, on_event=events.append)
        agent.client = anthropic.Anthropic(api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport))
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        agent = GraphAgent(csv_path, out_dir, on_event=events.append, max_retries=0)
        monkeypatch.setattr(agent.llm.bound._client._client, "_transport", transport)  # LangChain's own client
    agent.out = lambda *parts, end="\n", flush=False: printed.append(" ".join(str(p) for p in parts) + end)
    try:
        answers = [dataclasses.asdict(agent.ask(q)) for q in questions]
    finally:
        agent.close()
    return {"requests": without_default_is_error(sent), "answers": answers, "printed": "".join(printed), "events": events}


def without_default_is_error(requests):
    """LangChain sends "is_error": false on a successful tool result; the hand-written loop leaves the key out of
    a chart's result. False is the API's default, so the two requests mean the same. Drop it before comparing."""
    requests = copy.deepcopy(requests)
    for request in requests:
        for message in request["messages"]:
            for block in message["content"] if isinstance(message["content"], list) else []:
                if block.get("type") == "tool_result" and block.get("is_error") is False:
                    del block["is_error"]
    return requests


def both(csv_path, monkeypatch, responses, questions, ctrl_c_at=None):
    hand = run("hand-written", csv_path, monkeypatch, responses, questions, ctrl_c_at)
    graph = run("langgraph", csv_path, monkeypatch, responses, questions, ctrl_c_at)
    for key in ("requests", "answers", "printed", "events"):
        assert graph[key] == hand[key], f"the loops differ in {key}"
    return hand


def tool_results(request):
    return [block for block in request["messages"][-1]["content"] if block.get("type") == "tool_result"]


def test_tool_input_is_handled_the_same(csv_path, monkeypatch):
    responses = [
        # 1. Input the SDK's parser reads only part of: it runs print(1).
        stream(thinking("Checking."), text("Let me check."), tool("a1", "run_python", '{"code": "print(1)" "x"}')),
        stream(text("It printed 1.")),
        # 2. A string cut off mid-stream (no max_tokens): the SDK drops it, so nothing runs.
        stream(tool("b1", "run_python", '{"code": "print(2)'), stop="tool_use"),
        stream(text("No input.")),
        # 3. Input the SDK rejects (a trailing comma): the turn runs again.
        stream(tool("c1", "run_python", '{"code": "3",}')),
        stream(tool("c2", "run_python", '{"code": "print(3)"}')),
        stream(text("Three.")),
        # 4. Input cut off at the output limit: not run.
        stream(tool("d1", "run_python", '{"code": "print(4)\\nprint(5)"}'[:20]), stop="max_tokens"),
        stream(text("Shorter next time.")),
        # 5. Two tool calls in one turn, one of them a chart that goes back to the model as an image.
        stream(thinking("Plotting."), tool("e1", "run_python", '{"code": "x = df.sales.sum()"}'),
               tool("e2", "save_chart", '{"code": "df.sales.plot()", "name": "Sales"}')),
        stream(text("Here is the chart.")),
        # 6. A stream that breaks partway: the turn runs again, after the cut-off note is finished.
        broken(thinking("Checking."), text("Lost.")),
        stream(text("Six.")),
    ]
    hand = both(csv_path, monkeypatch, responses, ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])

    requests = hand["requests"]
    assert [a["status"] for a in hand["answers"]] == ["ok"] * 6
    assert tool_results(requests[1])[0]["content"] == "1\n"
    assert tool_results(requests[3])[0]["content"] == "Invalid input for run_python: {}"
    assert requests[5] == requests[4]  # the rejected turn was asked for again, unchanged
    assert tool_results(requests[6])[0]["content"] == "3\n"
    assert "cut off at the output limit" in tool_results(requests[8])[0]["content"]
    chart = tool_results(requests[10])[1]
    assert [part["type"] for part in chart["content"]] == ["text", "image"]
    assert requests[12] == requests[11]  # the broken turn was asked for again, unchanged
    assert hand["answers"][5]["text"] == "Six."


def test_refusals_errors_and_ctrl_c_roll_back_the_same(csv_path, monkeypatch):
    responses = [
        # 1. Refused after a tool call streamed: nothing runs, the question is rolled back.
        stream(tool("r1", "run_python", json.dumps({"code": f"open({str(csv_path.parent / 'ran')!r}, 'w')"})),
               stop="refusal"),
        # 2. The API fails partway through the question.
        stream(tool("s1", "run_python", '{"code": "print(1)"}')),
        OVERLOADED,
        # 3. Ctrl+C partway through the question (the second response is never sent).
        stream(tool("u1", "run_python", '{"code": "print(2)"}')),
        stream(text("Too late.")),
        # 4. A normal answer, with none of the rolled-back questions in its history.
        stream(text("Fine.")),
    ]
    hand = both(csv_path, monkeypatch, responses, ["Q1", "Q2", "Q3", "Q4"], ctrl_c_at=4)

    assert [a["status"] for a in hand["answers"]] == ["refused", "error", "cancelled", "ok"]
    assert hand["answers"][1]["error"].startswith("OverloadedError: ")
    assert not (csv_path.parent / "ran").exists()
    assert hand["requests"][-1]["messages"] == [{"role": "user", "content": "Q4"}]
