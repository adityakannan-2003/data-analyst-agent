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
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from sandbox import Sandbox

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (arrow keys and history in input())

MODEL = "claude-opus-5-5"
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


def profile(df: pd.DataFrame, name: str, max_columns: int = 60) -> str:
    """A compact description of the data. It goes in the system prompt so the
    model doesn't spend a tool call on df.info() for every question."""
    lines = [f"File: {name}", f"{len(df):,} rows x {df.shape[1]} columns", ""]
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
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage) -> None:
        cache_read = usage.cache_read_input_tokens or 0
        self.cached_tokens += cache_read
        self.input_tokens += usage.input_tokens + (usage.cache_creation_input_tokens or 0) + cache_read
        self.output_tokens += usage.output_tokens

    def __str__(self) -> str:
        return (
            f"{self.tool_calls} tool calls · {self.input_tokens / 1000:.1f}k input tokens "
            f"({self.cached_tokens / 1000:.1f}k from cache) · {self.output_tokens / 1000:.1f}k output tokens"
        )


class StreamPrinter:
    """Prints one streamed model turn: progress notes dimmed, answer text as it arrives."""

    def __init__(self):
        self.in_note = False
        self.at_line_start = True

    def handle(self, event) -> None:
        if event.type == "thinking" and event.thinking:
            # With display="updates", non-empty thinking text is a progress note; reasoning stays hidden.
            if not self.in_note:
                self.newline()
                self.write(f"{DIM}· ")
                self.in_note = True
            self.write(event.thinking)
        elif event.type == "content_block_start" and event.content_block.type == "text":
            self.newline()
        elif event.type == "text":
            self.write(event.text)
        elif event.type == "content_block_stop":
            if self.in_note:
                self.write(RESET)
                self.in_note = False
            self.newline()

    def write(self, text: str) -> None:
        print(text, end="", flush=True)
        if text and text != RESET:
            self.at_line_start = text.endswith("\n")

    def newline(self) -> None:
        if not self.at_line_start:
            print()
            self.at_line_start = True

    def finish(self) -> None:
        if self.in_note:
            self.write(RESET)
        self.newline()


def show_tool_call(name: str, code: str, max_lines: int = 8) -> None:
    lines = code.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    print(f"{DIM}  ▸ {name}")
    for line in lines:
        print(f"    │ {line}")
    print(RESET, end="")


def show_result(ok: bool, output: str) -> None:
    lines = output.strip().splitlines()
    if ok:
        first = lines[0][:90] if lines else "(no output)"
        more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        print(f"{DIM}    ✓ {first}{more}{RESET}")
    else:
        print(f"{RED}    ✗ {lines[-1][:120] if lines else 'error'}{RESET}")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "chart"


class Agent:
    def __init__(self, csv_path: Path, out_dir: Path, model: str = MODEL, effort: str = "high"):
        df = pd.read_csv(csv_path)
        self.shape = df.shape
        self.system = SYSTEM_PROMPT.format(profile=profile(df, csv_path.name))
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.out_dir = out_dir
        self.sandbox = Sandbox(csv_path, out_dir)
        self.messages: list = []  # the whole conversation, so follow-up questions have context
        self.charts = 0

    def ask(self, question: str) -> None:
        """Answer one question: call the model, run the tools it asks for, repeat until it answers."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        usage = Usage()
        try:
            for _ in range(MAX_STEPS):
                response = self._call_model()
                usage.add(response.usage)

                if response.stop_reason == "refusal":
                    del self.messages[start:]  # drop the declined question so later questions still work
                    print(f"\n{RED}The model declined this request.{RESET}")
                    return

                # Append the turn unchanged (thinking blocks included); the API expects history as it was sent.
                self.messages.append({"role": "assistant", "content": response.content})
                tool_calls = [block for block in response.content if block.type == "tool_use"]

                if not tool_calls:
                    if response.stop_reason == "max_tokens":
                        print(f"{RED}[The answer was cut off at the output limit.]{RESET}")
                    print(f"\n{DIM}({usage}){RESET}")
                    return

                if response.stop_reason == "max_tokens":
                    # A tool input cut off mid-stream still parses as a valid-looking object. Don't run it.
                    results = [
                        self._error(block, "Your tool input was cut off at the output limit. Send a shorter one.")
                        for block in tool_calls
                    ]
                else:
                    results = [self._run_tool(block) for block in tool_calls]
                usage.tool_calls += len(tool_calls)
                # All results go back in one user message, which keeps parallel tool calls working.
                self.messages.append({"role": "user", "content": results})

            print(f"\n{RED}Stopped after {MAX_STEPS} steps without a final answer.{RESET}")
        except KeyboardInterrupt:
            del self.messages[start:]
            print(f"\n{DIM}[cancelled]{RESET}")
        except (anthropic.APIError, RuntimeError) as exc:
            del self.messages[start:]
            print(f"\n{RED}Error: {exc}{RESET}")

    def _call_model(self):
        """One model turn, streamed so progress notes and the answer print as they arrive."""
        for _ in range(3):
            printer = StreamPrinter()
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

        show_tool_call(block.name, args["code"])
        if block.name == "run_python":
            ok, output = self.sandbox.run(args["code"])
            show_result(ok, output)
            return {"type": "tool_result", "tool_use_id": block.id, "content": output or "(no output)", "is_error": not ok}

        if block.name == "save_chart":
            self.charts += 1
            filename = f"chart_{self.charts}_{slug(args['name'])}.png"
            ok, output = self.sandbox.chart(args["code"], filename)
            if not ok:
                show_result(ok, output)
                return {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": True}
            path = self.out_dir / filename
            print(f"{DIM}    ✓ saved {path}{RESET}")
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

    @staticmethod
    def _error(block, message: str) -> dict:
        print(f"{RED}    ✗ {message[:120]}{RESET}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": message, "is_error": True}


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
    args = parser.parse_args()

    load_dotenv()
    if not has_credentials():
        sys.exit("No Anthropic API key found. Set ANTHROPIC_API_KEY, or copy .env.example to .env and add it there.")
    if not args.csv.is_file():
        sys.exit(f"File not found: {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)

    agent = Agent(args.csv, args.out, args.model, args.effort)
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
    finally:
        agent.sandbox.close()


if __name__ == "__main__":
    main()
