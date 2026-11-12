"""Compare the hand-written loop with the LangGraph one on the same evals: the tables in docs/LANGGRAPH.md.

    uv run python evals/compare_loops.py

Reads the runs made on the same day: the LangGraph loop in langgraph-v1, the hand-written loop in v5 (coffee)
and v2 (retail). No API calls. Prints markdown.
"""

import ast
import io
import json
import random
import statistics as st
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pricing import cost  # noqa: E402

FLOW = ROOT / "evals" / "runs"
# suite: (results folder, LangGraph variant, hand-written variant)
SUITES = {"coffee": ("analyst-answers", "langgraph-v1", "v5"), "retail": ("retail-answers", "langgraph-v1", "v2")}
LOOPS = ("LangGraph", "hand-written")
# Each suite's first batch started with a cold cache (see docs/LANGGRAPH.md): that run is left out of "warm" costs.
COLD = {("coffee", "LangGraph"): 0, ("retail", "hand-written"): 0}
# Code that only one loop needs. Shared code (prompt, tools, tool code, sandbox, StreamPrinter) isn't counted.
CODE = {
    "hand-written": (ROOT / "agent.py", ["Agent.ask", "Agent._call_model"]),
    "LangGraph": (ROOT / "graph_agent.py", ["GraphAgent", "TurnPrinter", "route", "build_graph", "read_tool_input",
                                            "api_usage", "api_message"]),
}


def load(suite: str, loop: str) -> list[dict]:
    folder, graph, hand = SUITES[suite]
    path = FLOW / folder / (graph if loop == "LangGraph" else hand) / "results.jsonl"
    rows = [json.loads(line) for line in path.open()]
    for row in rows:
        row["cost"] = cost(row["model"], row["usage"])
    return rows


def passed_every_run(rows: list[dict]) -> int:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["prompt_id"]].append(row["grade"].get("pass") == 1)
    return sum(all(runs) for runs in by_question.values())


def warm(rows: list[dict], suite: str, loop: str) -> list[dict]:
    return [r for r in rows if r["rep"] != COLD.get((suite, loop))]


def paired_gap(graph: list[dict], hand: list[dict], key: str) -> tuple[float, float, float]:
    """LangGraph minus hand-written, averaged question by question, with a 95% bootstrap interval over questions.
    Comparing each question with itself cancels out how hard the questions are."""
    def means(rows):
        by_question = defaultdict(list)
        for row in rows:
            by_question[row["prompt_id"]].append(row[key])
        return {q: st.mean(v) for q, v in by_question.items()}
    g, h = means(graph), means(hand)
    gaps = [g[q] - h[q] for q in g if q in h]
    rng = random.Random(0)  # fixed, so the table is the same every time
    resampled = sorted(st.mean(rng.choices(gaps, k=len(gaps))) for _ in range(10_000))
    return st.mean(gaps), resampled[249], resampled[9_749]


def code_lines(path: Path, names: list[str]) -> dict[str, int]:
    """Lines of code in the named classes and functions: no blank lines, comments or docstrings."""
    source = path.read_text()
    tree = ast.parse(source)
    spans, docstrings = {}, set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.update(range(body[0].lineno, body[0].end_lineno + 1))

    def find(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                name = prefix + child.name
                if name in names:
                    spans[name] = (child.lineno, child.end_lineno)
                find(child, name + ".")
    find(tree)
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    code = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in skip:
            code.update(range(token.start[0], token.end[0] + 1))
    missing = set(names) - set(spans)
    assert not missing, f"not found in {path.name}: {missing}"
    return {name: sum(1 for line in range(a, b + 1) if line in code and line not in docstrings)
            for name, (a, b) in spans.items()}


def main() -> None:
    rows = {(suite, loop): load(suite, loop) for suite in SUITES for loop in LOOPS}
    columns = [(suite, loop) for suite in SUITES for loop in LOOPS]

    def row(label, value):
        return f"| {label} | " + " | ".join(value(rows[c], *c) for c in columns) + " |"

    print("| Per answer, unless stated | " + " | ".join(f"{s.capitalize()}: {loop}" for s, loop in columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    print(row("Runs (questions x runs)", lambda r, s, loop: f"{len(r)}"))
    print(row("Questions passed on every run",
              lambda r, s, loop: f"{passed_every_run(r)} of {len({x['prompt_id'] for x in r})}"))
    print(row("Model turns", lambda r, s, loop: f"{st.mean(x['model_turns'] for x in r):.2f}"))
    print(row("Tool calls", lambda r, s, loop: f"{st.mean(x['tool_calls'] for x in r):.2f}"))
    print(row("Agent cost, all runs", lambda r, s, loop: f"${st.mean(x['cost'] for x in r):.4f}"))
    print(row("Agent cost, warm runs only", lambda r, s, loop: f"${st.mean(x['cost'] for x in warm(r, s, loop)):.4f}"))
    print(row("Time", lambda r, s, loop: f"{st.mean(x['latency_s'] for x in r):.1f} s"))

    print("\nBoth suites together (74 runs per loop):\n")
    print("| Per answer | " + " | ".join(LOOPS) + " |\n|---|---|---|")
    both = {loop: rows[("coffee", loop)] + rows[("retail", loop)] for loop in LOOPS}
    for label, key, fmt in [("Model turns", "model_turns", "{:.2f}"), ("Tool calls", "tool_calls", "{:.2f}"),
                            ("Agent cost", "cost", "${:.4f}"), ("Time", "latency_s", "{:.1f} s")]:
        print(f"| {label} | " + " | ".join(fmt.format(st.mean(x[key] for x in both[loop])) for loop in LOOPS) + " |")
    print(f"| Questions passed on every run | "
          + " | ".join(f"{sum(passed_every_run(rows[(s, loop)]) for s in SUITES)} of 37" for loop in LOOPS) + " |")

    print("\nLangGraph minus hand-written, question by question (95% interval):\n")
    print("| Per answer | " + " | ".join(s.capitalize() for s in SUITES) + " |\n|---|---|---|")
    for label, key, fmt, only_warm in [("Model turns", "model_turns", "{:+.2f}", False),
                                       ("Tool calls", "tool_calls", "{:+.2f}", False),
                                       ("Agent cost, all runs", "cost", "{:+.4f}", False),
                                       ("Agent cost, warm runs only", "cost", "{:+.4f}", True),
                                       ("Time (s)", "latency_s", "{:+.1f}", False)]:
        cells = []
        for suite in SUITES:
            graph, hand = rows[(suite, "LangGraph")], rows[(suite, "hand-written")]
            if only_warm:
                graph, hand = warm(graph, suite, "LangGraph"), warm(hand, suite, "hand-written")
            mean, low, high = paired_gap(graph, hand, key)
            cells.append(f"{fmt.format(mean)} ({fmt.format(low)} to {fmt.format(high)})")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\nLines of code only one loop needs (no blank lines, comments or docstrings):\n")
    print("| Loop | Lines | Where |\n|---|---|---|")
    for loop, (path, names) in CODE.items():
        counts = code_lines(path, names)
        where = ", ".join(f"{name} {n}" for name, n in counts.items())
        print(f"| {loop} | {sum(counts.values())} | {path.name}: {where} |")


if __name__ == "__main__":
    main()
