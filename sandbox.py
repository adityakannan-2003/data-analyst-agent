"""Runs model-written Python in a child process with a persistent namespace.

The child holds the DataFrame and every variable the agent creates, so later
tool calls build on earlier ones, like cells in a notebook. Because it is a
separate process, a crash or an infinite loop can't take the agent down: the
parent kills the child after a timeout and starts a fresh one.

This protects the agent from crashes and hangs. It is NOT a security boundary:
the code runs with your user's permissions. See "Safety" in the README.
"""

import ast
import contextlib
import io
import multiprocessing as mp
import signal
import traceback
from pathlib import Path

import numpy as np

MAX_OUTPUT_CHARS = 6000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    cut = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n... [{cut:,} characters truncated] ...\n{text[-half:]}"


def _run_cell(code: str, namespace: dict) -> None:
    """Exec `code`, then print the value of a trailing expression, like a notebook."""
    tree = ast.parse(code, "<cell>")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<cell>", "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
        if isinstance(value, np.generic):
            value = value.item()  # show 22, not np.int64(22)
        if value is not None:
            print(repr(value))


def _format_error(exc: BaseException, code: str) -> str:
    """A traceback trimmed to the lines of the cell, without pandas internals."""
    lines = code.splitlines()
    out = []
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<cell>" and frame.lineno and frame.lineno <= len(lines):
            out.append(f"  line {frame.lineno}: {lines[frame.lineno - 1].strip()}")
    out += traceback.format_exception_only(exc)
    return "\n".join(line.rstrip() for line in out)


def _worker(conn, csv_path: str, out_dir: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the parent

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    try:
        namespace = {"pd": pd, "np": np, "plt": plt, "df": pd.read_csv(csv_path)}
    except Exception as exc:
        conn.send((False, f"Could not load {csv_path}: {exc}"))
        return
    conn.send((True, ""))

    while True:
        request = conn.recv()
        if request is None:
            return
        kind, code, chart_path = request
        output = io.StringIO()
        ok = True
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                _run_cell(code, namespace)
                if kind == "chart":
                    if not plt.get_fignums():
                        raise ValueError("The code didn't draw anything, so there is no chart to save.")
                    fig = plt.gcf()
                    fig.tight_layout()
                    fig.savefig(Path(out_dir) / chart_path, dpi=100)
        except Exception as exc:
            ok = False
            output.write(_format_error(exc, code))
        finally:
            if kind == "chart":
                plt.close("all")
        conn.send((ok, _truncate(output.getvalue())))


class Sandbox:
    def __init__(self, csv_path: Path, out_dir: Path, timeout: float = 60):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.timeout = timeout
        self._ctx = mp.get_context("spawn")
        self._start()

    def run(self, code: str) -> tuple[bool, str]:
        """Run code in the session. Returns (ok, output or error)."""
        return self._call("run", code, None)

    def chart(self, code: str, filename: str) -> tuple[bool, str]:
        """Run drawing code and save the current figure as out_dir/filename."""
        return self._call("chart", code, filename)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.send(None)
        self._proc.join(timeout=2)
        if self._proc.is_alive():
            self._proc.kill()

    def _start(self) -> None:
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker,
            args=(child_conn, str(self.csv_path), str(self.out_dir)),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()  # only the child holds its end now, so recv() fails fast if the child dies
        if not self._conn.poll(self.timeout):
            self._proc.kill()
            raise RuntimeError(f"The Python process didn't start within {self.timeout:.0f}s.")
        try:
            ok, error = self._conn.recv()
        except EOFError:
            raise RuntimeError("The Python process exited while loading the data.") from None
        if not ok:
            raise RuntimeError(error)

    def _restart(self) -> None:
        self._proc.kill()
        self._proc.join()
        self._start()

    def _call(self, kind: str, code: str, chart_path: str | None) -> tuple[bool, str]:
        if not self._proc.is_alive():
            self._restart()
        self._conn.send((kind, code, chart_path))
        try:
            if self._conn.poll(self.timeout):
                return self._conn.recv()
        except EOFError:
            self._restart()
            return False, "The Python process crashed. It was restarted: `df` was reloaded and all other variables were lost."
        except KeyboardInterrupt:
            self._restart()  # the child may still be running; don't let its late reply desync the pipe
            raise
        self._restart()
        return False, (
            f"Timed out after {self.timeout:.0f}s. The Python process was restarted: "
            "`df` was reloaded and all other variables were lost."
        )
