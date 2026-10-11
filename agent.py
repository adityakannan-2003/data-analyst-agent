"""Data analyst agent: ask questions about a CSV file in plain English.

Claude answers by writing pandas code, running it, reading the output, and
trying again when something fails. The agent loop is written by hand, with no
agent framework, so every step is visible:

    question -> Claude -> tool calls -> run them -> results -> Claude -> ... -> answer

Usage:
    uv run python agent.py data/coffee_sales.csv
    uv run python agent.py data/coffee_sales.csv -q "Which store makes the most money?"
"""

import argparse
import base64
import contextlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from sandbox import Sandbox, comment_lines, read_data

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (arrow keys and history in input())

MODEL = "claude-sonnet-5-5"  # passes every eval question (see evals/) at under half the cost of claude-opus-5-5
MAX_STEPS = 25  # model turns per question, so a confused run can't loop forever
BETAS = [
    "server-side-fallback-2026-07-01",  # fallbacks="default": retry a declined request on another model
    "thinking-display-updates-2026-08-18",  # display="updates": short progress notes between tool calls
]

DIM, RED, RESET = "\033[2m", "\033[31m", "\033[0m"

SYSTEM_PROMPT = """\
You are a data analyst. The user has loaded a CSV file into a pandas DataFrame called `df`, and you answer their questions by running Python against it.

How to work:
- Check the columns you rely on before trusting them. Real exports have numbers stored as text, inconsistent spellings, missing values, duplicate rows, and refunds or corrections. Clean what matters for the question, and say what you changed.
- Every number in your answer must come from code you ran in this session. If the data can't answer the question, say so and say what data would.
- When you average or compare over time, check that everything was around for the whole period, such as a product launched partway through. Average it over the time it existed, or say plainly that the full-period figure understates it.
- Make a chart when it shows the answer better than numbers do (trends over time, comparisons across many groups). Look at the image you get back, and redraw it if it's hard to read.
- Variables persist between run_python calls, so build on earlier work instead of recomputing it.

Your answer is shown in a terminal: lead with the direct answer, then the supporting numbers, then caveats. Keep it short. Plain text only: no Markdown headings, bold, or tables; use simple "-" lists.

Dataset profile:
{profile}
"""

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Run Python in a persistent session. `df` is the dataset (a pandas DataFrame) and `pd`, `np`, and `plt` "
            "are imported. Variables persist between calls. Returns what the code prints plus the value of the last "
            "expression, like a notebook cell, or the error if it fails. Output over about 6,000 characters is "
            "truncated, so print summaries rather than whole tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"],
        },
        "eager_input_streaming": True,
    },
    {
        "name": "save_chart",
        "description": (
            "Draw a chart with matplotlib and save it as a PNG for the user. The code runs in the same session as "
            "run_python and should draw one figure (for example with df.plot or plt.bar) with a title and axis "
            "labels. Don't call plt.savefig or plt.show. Returns the file path and the rendered image so you can "
            "check it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Matplotlib code that draws the chart."},
                "name": {"type": "string", "description": "A short name for the file, like 'revenue by month'."},
            },
            "required": ["code", "name"],
        },
        "eager_input_streaming": True,
    },
]


def profile(df: pd.DataFrame, name: str, comments: list[str] = (), max_columns: int = 60) -> str:
    """A compact description of the data. It goes in the system prompt so the
    model doesn't spend a tool call on df.info() for every question."""
    lines = [f"File: {name}"]
    if comments:
        lines += ["Comment lines at the top of the file (skipped when loading):", *comments]
    lines += [f"{len(df):,} rows x {df.shape[1]} columns", ""]
    for col in df.columns[:max_columns]:
        s = df[col]
        examples = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in s.dropna().unique()[:4])
        lines.append(
            f"- {col} ({s.dtype}): {s.isna().sum():,} missing, {s.nunique():,} distinct, e.g. {examples[:120]}"
        )
    if df.shape[1] > max_columns:
        lines.append(f"- ... and {df.shape[1] - max_columns} more columns (see df.columns)")
    return "\n".join(lines)


@dataclass
class Usage:
    tool_calls: int = 0
    input_tokens: int = 0  # all input, including tokens read from or written to the cache
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage) -> None:
        cache_read, cache_write = usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        self.input_tokens += usage.input_tokens + cache_write + cache_read
        self.output_tokens += usage.output_tokens

    def __str__(self) -> str:
        return (
            f"{self.tool_calls} tool calls · {self.input_tokens / 1000:.1f}k input tokens "
            f"({self.cache_read_tokens / 1000:.1f}k from cache) · {self.output_tokens / 1000:.1f}k output tokens"
        )


@dataclass
class Answer:
    """What ask() returns: the final answer and what it took to get there."""

    text: str = ""
    status: str = "ok"  # ok, truncated, refused, max_steps, cancelled, or error
    error: str = ""
    usage: Usage = field(default_factory=Usage)
    models: list[str] = field(default_factory=list)  # the model that served each turn


def silent(*args, **kwargs) -> None:
    pass


class StreamPrinter:
    """Prints one streamed model turn: progress notes dimmed, answer text as it arrives.
    Also passes notes and text to on_event, for front ends other than the terminal."""

    def __init__(self, out=print, on_event=None):
        self.out = out
        self.on_event = on_event or silent
        self.in_note = False
        self.at_line_start = True

    def handle(self, event) -> None:
        if event.type == "thinking" and event.thinking:
            self.on_event({"type": "note", "text": event.thinking})
            # With display="updates", non-empty thinking text is a progress note; reasoning stays hidden.
            if not self.in_note:
                self.newline()
                self.write(f"{DIM}· ")
                self.in_note = True
            self.write(event.thinking)
        elif event.type == "content_block_start" and event.content_block.type == "text":
            self.newline()
        elif event.type == "text":
            self.on_event({"type": "text", "text": event.text})
            self.write(event.text)
        elif event.type == "content_block_stop":
            if self.in_note:
                self.write(RESET)
                self.in_note = False
            self.newline()

    def write(self, text: str) -> None:
        self.out(text, end="", flush=True)
        if text and text != RESET:
            self.at_line_start = text.endswith("\n")

    def newline(self) -> None:
        if not self.at_line_start:
            self.out()
            self.at_line_start = True

    def finish(self) -> None:
        if self.in_note:
            self.write(RESET)
        self.newline()


def show_tool_call(name: str, code: str, out=print, max_lines: int = 8) -> None:
    lines = code.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    out(f"{DIM}  ▸ {name}")
    for line in lines:
        out(f"    │ {line}")
    out(RESET, end="")


def show_result(ok: bool, output: str, out=print) -> None:
    lines = output.strip().splitlines()
    if ok:
        first = lines[0][:90] if lines else "(no output)"
        more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        out(f"{DIM}    ✓ {first}{more}{RESET}")
    else:
        out(f"{RED}    ✗ {lines[-1][:120] if lines else 'error'}{RESET}")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "chart"


class Agent:
    def __init__(
        self, csv_path: Path, out_dir: Path, model: str = MODEL, effort: str = "high", quiet: bool = False, on_event=None
    ):
        df = read_data(csv_path)
        self.shape = df.shape
        self.system = SYSTEM_PROMPT.format(profile=profile(df, csv_path.name, comment_lines(csv_path)))
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.out_dir = out_dir
        self.out = silent if quiet else print  # quiet mode lets several agents run side by side (see evals/)
        # on_event receives each step as a dict: note, text, tool_call, tool_result or chart (see app.py)
        self.on_event = on_event or silent
        self.sandbox = Sandbox(csv_path, out_dir)
        self.messages: list = []  # the whole conversation, so follow-up questions have context
        self.charts = 0

    def ask(self, question: str) -> Answer:
        """Answer one question: call the model, run the tools it asks for, repeat until it answers."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        answer = Answer()
        try:
            for _ in range(MAX_STEPS):
                response = self._call_model()
                answer.usage.add(response.usage)
                answer.models.append(response.model)

                if response.stop_reason == "refusal":
                    del self.messages[start:]  # drop the declined question so later questions still work
                    self.out(f"\n{RED}The model declined this request.{RESET}")
                    answer.status = "refused"
                    return answer

                # Append the turn unchanged (thinking blocks included); the API expects history as it was sent.
                self.messages.append({"role": "assistant", "content": response.content})
                tool_calls = [block for block in response.content if block.type == "tool_use"]

                if not tool_calls:
                    answer.text = "".join(block.text for block in response.content if block.type == "text")
                    if response.stop_reason == "max_tokens":
                        answer.status = "truncated"
                        self.out(f"{RED}[The answer was cut off at the output limit.]{RESET}")
                    self.out(f"\n{DIM}({answer.usage}){RESET}")
                    return answer

                if response.stop_reason == "max_tokens":
                    # A tool input cut off mid-stream still parses as a valid-looking object. Don't run it.
                    results = [
                        self._error(block, "Your tool input was cut off at the output limit. Send a shorter one.")
                        for block in tool_calls
                    ]
                else:
                    results = [self._run_tool(block) for block in tool_calls]
                answer.usage.tool_calls += len(tool_calls)
                # All results go back in one user message, which keeps parallel tool calls working.
                self.messages.append({"role": "user", "content": results})

            self.out(f"\n{RED}Stopped after {MAX_STEPS} steps without a final answer.{RESET}")
            answer.status = "max_steps"
        except KeyboardInterrupt:
            del self.messages[start:]
            self.out(f"\n{DIM}[cancelled]{RESET}")
            answer.status = "cancelled"
        except (anthropic.APIError, RuntimeError) as exc:
            del self.messages[start:]
            self.out(f"\n{RED}Error: {exc}{RESET}")
            answer.status, answer.error = "error", f"{type(exc).__name__}: {exc}"
        return answer

    def _call_model(self):
        """One model turn, streamed so progress notes and the answer print as they arrive."""
        for _ in range(3):
            printer = StreamPrinter(self.out, self.on_event)
            try:
                with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=64000,
                    betas=BETAS,
                    fallbacks="default",
                    thinking={"type": "adaptive", "display": "updates"},
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},  # cache the growing conversation between turns
                    system=self.system,
                    tools=TOOLS,
                    messages=self.messages,
                ) as stream:
                    for event in stream:
                        printer.handle(event)
                    printer.finish()
                    return stream.get_final_message()
            except ValueError:
                # Tool input JSON the SDK couldn't parse at all (possible with eager input streaming).
                # No tool_use block completed, so there is nothing to answer: run the turn again.
                printer.finish()
        raise RuntimeError("The model's tool input couldn't be parsed after 3 tries.")

    def _run_tool(self, block) -> dict:
        args = block.input
        # With eager input streaming the API doesn't validate tool input, so check it before running anything.
        valid = isinstance(args, dict) and isinstance(args.get("code"), str)
        if block.name == "save_chart":
            valid = valid and isinstance(args.get("name"), str)
        if not valid:
            return self._error(block, f"Invalid input for {block.name}: {json.dumps(args)[:500]}")

        show_tool_call(block.name, args["code"], self.out)
        self.on_event({"type": "tool_call", "name": block.name, "code": args["code"]})
        if block.name == "run_python":
            ok, output = self.sandbox.run(args["code"])
            show_result(ok, output, self.out)
            self.on_event({"type": "tool_result", "ok": ok, "output": output})
            return {"type": "tool_result", "tool_use_id": block.id, "content": output or "(no output)", "is_error": not ok}

        if block.name == "save_chart":
            self.charts += 1
            filename = f"chart_{self.charts}_{slug(args['name'])}.png"
            ok, output = self.sandbox.chart(args["code"], filename)
            if not ok:
                show_result(ok, output, self.out)
                self.on_event({"type": "tool_result", "ok": False, "output": output})
                return {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": True}
            path = self.out_dir / filename
            self.out(f"{DIM}    ✓ saved {path}{RESET}")
            self.on_event({"type": "chart", "path": str(path)})
            note = f"Saved to {path}." + (f"\nOutput:\n{output}" if output.strip() else "")
            image = base64.standard_b64encode(path.read_bytes()).decode()
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": [
                    {"type": "text", "text": note},
                    # Send the chart back so the model can see it and fix unreadable labels, wrong scales, etc.
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
                ],
            }

        return self._error(block, f"Unknown tool: {block.name}")

    def _error(self, block, message: str) -> dict:
        self.out(f"{RED}    ✗ {message[:120]}{RESET}")
        self.on_event({"type": "tool_result", "ok": False, "output": message})
        return {"type": "tool_result", "tool_use_id": block.id, "content": message, "is_error": True}


def open_file(path: Path) -> None:
    """Open a file in the computer's default viewer (Preview on a Mac)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass  # no viewer available; the path is printed anyway


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()  # an `ant auth login` profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a CSV file in plain English.")
    parser.add_argument("csv", type=Path, help="The CSV file to analyze.")
    parser.add_argument("-q", "--question", help="Ask one question and exit, instead of starting a chat.")
    parser.add_argument("--model", default=MODEL, help=f"Claude model (default: {MODEL}).")
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="How much the model thinks before acting (default: high).",
    )
    parser.add_argument("--out", type=Path, default=Path("out"), help="Folder for charts (default: out/).")
    parser.add_argument("--no-open", action="store_true", help="Don't open charts automatically in chat mode.")
    args = parser.parse_args()

    load_dotenv()
    if not has_credentials():
        sys.exit("No Anthropic API key found. Set ANTHROPIC_API_KEY, or copy .env.example to .env and add it there.")
    if not args.csv.is_file():
        sys.exit(f"File not found: {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)

    charts: list[str] = []
    agent = Agent(args.csv, args.out, args.model, args.effort,
                  on_event=lambda event: charts.append(event["path"]) if event["type"] == "chart" else None)
    try:
        if args.question:
            agent.ask(args.question)
            return
        rows, cols = agent.shape
        print(f"Loaded {args.csv.name}: {rows:,} rows x {cols} columns. Model: {args.model}, effort: {args.effort}.")
        print("Ask a question about the data. Follow-ups remember the conversation. Type 'exit' to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                agent.ask(question)
                if charts and not args.no_open:
                    open_file(Path(charts[-1]))  # the last one: the agent sometimes redraws a chart to fix it
                charts.clear()
    finally:
        agent.sandbox.close()


if __name__ == "__main__":
    main()
