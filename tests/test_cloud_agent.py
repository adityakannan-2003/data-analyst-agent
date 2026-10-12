"""Tests for the hosted agent's loop, with a fake API client: no key, no network, no cost."""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from agent import MODEL
from cloud_agent import CloudAgent, final_text, safe_filename


class FakeStream:
    """Yields a content_block_stop event for each block, as the SDK's stream does, then the message."""

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([NS(type="content_block_stop", content_block=block) for block in self.message.content])

    def get_final_message(self):
        return self.message


class FakeFiles:
    def __init__(self, names=None):
        self.names = names or {}  # file_id -> filename, for files the agent made
        self.uploads, self.deleted = [], []

    def upload(self, file):
        self.uploads.append(file)
        return NS(id=f"upload-{len(self.uploads)}")

    def retrieve_metadata(self, file_id):
        return NS(filename=self.names[file_id])

    def download(self, file_id):
        return NS(write_to_file=lambda path: path.write_bytes(b"\x89PNG fake"))

    def delete(self, file_id):
        self.deleted.append(file_id)


class FakeClient:
    """Returns scripted responses in order, or raises them, and records each request."""

    def __init__(self, turns, names=None):
        self.turns = list(turns)
        self.requests = []
        self.files = FakeFiles(names)
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return FakeStream(turn)


def response(*content, stop_reason="end_turn", container="container-1"):
    usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=list(content), stop_reason=stop_reason, usage=usage, model=MODEL, container=NS(id=container))


def run(command):
    return NS(type="server_tool_use", name="bash_code_execution", input={"command": command})


def result(stdout="", return_code=0, files=()):
    content = NS(type="bash_code_execution_result", stdout=stdout, stderr="", return_code=return_code,
                 content=[NS(file_id=f) for f in files])
    return NS(type="bash_code_execution_tool_result", content=content)


def text(value):
    return NS(type="text", text=value)


@pytest.fixture
def make_agent(csv_path):
    def make(turns, names=None, **kwargs):
        return CloudAgent(csv_path, csv_path.parent, client=FakeClient(turns, names), **kwargs)
    return make


def test_first_question_mounts_the_data_and_follow_ups_reuse_the_container(make_agent):
    agent = make_agent([
        response(run("python total.py"), result("22\n"), text("Total sales are 22.")),
        response(text("Paris sold 17.")),
    ])
    first = agent.ask("What are total sales?")
    second = agent.ask("And Paris?")

    assert (first.text, first.status, first.usage.tool_calls) == ("Total sales are 22.", "ok", 1)
    assert second.text == "Paris sold 17."
    first_request, second_request = agent.client.requests
    assert first_request["messages"][0]["content"] == [
        {"type": "container_upload", "file_id": "upload-1"},
        {"type": "text", "text": "What are total sales?"},
    ]
    assert "container" not in first_request
    assert second_request["container"] == "container-1"
    assert second_request["messages"][-1] == {"role": "user", "content": "And Paris?"}
    assert len(agent.client.files.uploads) == 1


def test_a_paused_turn_is_sent_back_unchanged_to_resume(make_agent):
    agent = make_agent([
        response(run("python step1.py"), result("ok"), stop_reason="pause_turn"),
        response(run("python step2.py"), result("22"), text("Total sales are 22.")),
    ])
    answer = agent.ask("What are total sales?")

    assert (answer.text, answer.usage.tool_calls) == ("Total sales are 22.", 2)
    resumed = agent.client.requests[1]
    assert [m["role"] for m in resumed["messages"]] == ["user", "assistant"]  # no extra "continue" message
    assert resumed["container"] == "container-1"


def test_the_answer_is_the_text_after_the_last_tool_result():
    content = [text("Let me load the data."), run("python a.py"), result("22"), text("Total sales "), text("are 22.")]
    assert final_text(content) == "Total sales are 22."
    assert final_text([text("No tools needed.")]) == "No tools needed."


def test_charts_are_downloaded_shown_and_deleted_from_the_files_api(make_agent):
    events = []
    agent = make_agent(
        [response(run("python chart.py"), result("saved", files=["chart-file", "table-file"]), text("Here it is."))],
        names={"chart-file": "../../Revenue by month.png", "table-file": "summary.csv"},
        on_event=events.append,
    )
    agent.ask("Chart revenue by month.")

    charts = [e["path"] for e in events if e["type"] == "chart"]
    assert len(charts) == 1 and charts[0].endswith("chart_1_revenue-by-month.png")
    assert (agent.out_dir / "chart_1_revenue-by-month.png").read_bytes().startswith(b"\x89PNG")
    assert agent.saved == {"chart-file": charts[0]}
    assert set(agent.client.files.deleted) == {"chart-file", "table-file"}  # only charts are kept, here


def test_steps_are_reported_for_the_web_page(make_agent):
    events = []
    agent = make_agent([response(run("python -c 'print(22)'"), result("22\n"), text("22."))], on_event=events.append)
    agent.ask("What are total sales?")

    steps = [e for e in events if e["type"] in ("tool_call", "tool_result")]
    assert steps == [
        {"type": "tool_call", "name": "bash", "code": "python -c 'print(22)'"},
        {"type": "tool_result", "ok": True, "output": "22\n"},
    ]


def test_a_refusal_drops_the_question_so_the_next_one_starts_cleanly(make_agent):
    agent = make_agent([response(stop_reason="refusal"), response(text("22."))])
    assert agent.ask("Something declined").status == "refused"
    assert agent.messages == []

    agent.ask("What are total sales?")
    assert agent.client.requests[1]["messages"][0]["content"][0]["type"] == "container_upload"


def test_an_api_error_rolls_back_the_question(make_agent):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    agent = make_agent([error])
    answer = agent.ask("What are total sales?")

    assert answer.status == "error" and "APIConnectionError" in answer.error
    assert agent.messages == []


def test_close_deletes_only_an_upload_the_agent_made(make_agent):
    own = make_agent([response(text("22."))])
    own.ask("What are total sales?")
    own.close()
    assert own.client.files.deleted == ["upload-1"]

    shared = make_agent([response(text("22."))], file_id="shared-sample")
    shared.ask("What are total sales?")
    shared.close()
    assert shared.client.files.deleted == [] and shared.client.files.uploads == []


def test_file_names_are_made_safe_for_the_load_line(tmp_path):
    assert safe_filename("Q3 sales [final].csv") == "Q3_sales_final.csv"
    assert safe_filename("../../etc/passwd") == "passwd.csv"
    unsafe = tmp_path / "my data.csv"
    unsafe.write_text("a\n1\n")
    with pytest.raises(ValueError):
        CloudAgent(unsafe, tmp_path, client=FakeClient([]))


def test_the_load_line_skips_comment_lines(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("# Source: somewhere\n# Monthly\ndate,price\n2020-01,1\n")
    agent = CloudAgent(path, tmp_path, client=FakeClient([]))
    assert 'glob.glob("/files/input/*/prices.csv")[0], skiprows=2)' in agent.system
