"""Record the agent answering a question and turn the session into an animated GIF.

    uv run python scripts/make_demo_gif.py                          # record and render (one agent run)
    uv run python scripts/make_demo_gif.py "Which store earns most?"  # a different question
    uv run python scripts/make_demo_gif.py --replay out/demo_recording.json  # re-render, no API calls

Everything the agent prints is replayed as recorded. Two things are edited so the GIF is
watchable: waits longer than 1.5 seconds (mostly the model thinking) are shortened to 1.5
seconds, and the typing of the command and question is animated, because the question is
piped in during recording. The window is sized so the final answer fits on screen, and the
last frame is held for 12 seconds so it can be read.
"""

import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "out" / "demo_recording.json"
DEFAULT_QUESTION = "Did the April price increase hurt latte sales?"

COLS, FONT_SIZE, LINE_H, PAD, TITLE_BAR = 100, 13, 17, 16, 30
MAX_WAIT = 1.5  # seconds; longer pauses in the recording are shortened to this
FRAME = 0.06  # output arriving within 60ms shares a frame (browsers slow down GIF frames under 20ms)
TYPE_COMMAND, TYPE_QUESTION, HOLD_END = 0.035, 0.045, 12.0

BG, BAR_BG, BAR_FG, CURSOR = (24, 27, 33), (40, 44, 52), (150, 156, 168), (200, 205, 212)
WINDOW_DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
STYLES = {  # style: (color, weight)
    "normal": ((214, 218, 225), "regular"),
    "dim": ((125, 133, 146), "regular"),
    "red": ((244, 112, 103), "regular"),
    "prompt": ((126, 231, 135), "bold"),
    "input": ((255, 255, 255), "bold"),
}
ANSI = {"\x1b[2m": "dim", "\x1b[31m": "red", "\x1b[0m": "normal"}  # the codes agent.py prints


def record(question: str, csv: str) -> list:
    """Run the agent's chat, send it the question, and capture its output with timestamps."""
    start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "agent.py", csv],
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    proc.stdin.write(f"{question}\nexit\n".encode())
    proc.stdin.close()
    decoder = codecs.getincrementaldecoder("utf-8")()  # a symbol like ✓ can be split across reads
    events = []
    while chunk := proc.stdout.read1(4096):
        events.append([round(time.monotonic() - start, 3), decoder.decode(chunk)])
    if proc.wait() != 0:
        sys.exit("The agent exited with an error:\n" + "".join(text for _, text in events))
    return events


def type_text(actions: list, t: float, text: str, delay: float) -> float:
    for ch in text:
        actions.append((t, ch, "input"))
        t += delay
    actions.append((t, "\n", "normal"))
    return t


def timeline(events: list, command: str, question: str) -> list:
    """Turn the recording into (time, text, style) actions, adding the typed command and question.
    A style of None means the text carries its own ANSI color codes."""
    actions = [(0.0, "$ ", "prompt")]
    t = type_text(actions, 0.6, command, TYPE_COMMAND) + 0.4
    prompts, last = 0, events[0][0]
    for real, chunk in events:
        t += min(real - last, MAX_WAIT)
        last = real
        while "\n> " in chunk:  # the chat's input prompt
            before, chunk = chunk.split("\n> ", 1)
            actions += [(t, before + "\n", None), (t, "> ", "prompt")]
            prompts += 1
            if prompts == 2:  # the recording typed "exit" here; end on the empty prompt
                return actions
            t = type_text(actions, t + 0.8, question, TYPE_QUESTION) + 0.3
        actions.append((t, chunk, None))
    return actions


class Terminal:
    """Just enough of a terminal for agent.py's output: text, newlines, and three colors."""

    def __init__(self):
        self.lines = [[]]  # each line is a list of (char, style)
        self.style = "normal"

    def feed(self, text: str, style: str | None = None) -> None:
        for part in re.split(r"(\x1b\[[0-9;]*m)", text):
            if part.startswith("\x1b["):
                self.style = ANSI.get(part, self.style)
                continue
            for ch in part:
                if ch == "\n":
                    self.lines.append([])
                else:
                    self.lines[-1].append((ch, style or self.style))

    def rows(self) -> list:
        """Lay the lines out at COLS wide: cut code previews, wrap prose at word boundaries."""
        rows = []
        for line in self.lines:
            text = "".join(ch for ch, _ in line)
            if len(line) > COLS and text.startswith("    │ "):
                rows.append(line[: COLS - 1] + [("…", line[COLS - 1][1])])
                continue
            if len(line) > COLS and text.startswith("    ✓ ") and "  (+" in text:  # keep the "(+N lines)" count
                suffix = line[text.rindex("  (+") :]
                rows.append(line[: COLS - len(suffix) - 1] + [("…", suffix[0][1])] + suffix)
                continue
            indent = [(" ", "normal")] * (2 if text.lstrip().startswith("- ") else 0)
            while len(line) > COLS:
                cut = max((i for i in range(1, COLS) if line[i][0] == " "), default=COLS)
                rows.append(line[:cut])
                line = indent + line[cut + 1 if line[cut][0] == " " else cut :]
            rows.append(line)
        return rows


def load_fonts() -> tuple:
    """SF Mono on macOS, then Menlo, then DejaVu Sans Mono on Linux. Returns (regular, bold, symbols)."""
    sf_mono, menlo = Path("/System/Library/Fonts/SFNSMono.ttf"), Path("/System/Library/Fonts/Menlo.ttc")
    dejavu = Path("/usr/share/fonts/truetype/dejavu")
    if sf_mono.exists():
        regular, bold = ImageFont.truetype(sf_mono, FONT_SIZE), ImageFont.truetype(sf_mono, FONT_SIZE)
        regular.set_variation_by_name("Regular")
        bold.set_variation_by_name("Bold")
    elif menlo.exists():
        regular, bold = ImageFont.truetype(menlo, FONT_SIZE, index=0), ImageFont.truetype(menlo, FONT_SIZE, index=1)
    elif dejavu.exists():
        regular = ImageFont.truetype(dejavu / "DejaVuSansMono.ttf", FONT_SIZE)
        bold = ImageFont.truetype(dejavu / "DejaVuSansMono-Bold.ttf", FONT_SIZE)
    else:
        sys.exit("No monospace font found (looked for SF Mono, Menlo, and DejaVu Sans Mono).")
    symbols_path = Path("/System/Library/Fonts/Apple Symbols.ttf")
    symbols = ImageFont.truetype(symbols_path, FONT_SIZE) if symbols_path.exists() else regular
    return regular, bold, symbols


def render(recording: dict, out_path: Path) -> str:
    regular, bold, symbols = load_fonts()
    fonts = {"regular": regular, "bold": bold}
    no_glyph = bytes(regular.getmask(""))
    glyph_font = {}  # char -> font that has it

    def font_for(ch: str, weight: str):
        if ch not in glyph_font:
            glyph_font[ch] = None if bytes(regular.getmask(ch)) != no_glyph else symbols
        return glyph_font[ch] or fonts[weight]

    command = f"uv run python agent.py {recording['csv']}"
    actions = timeline(recording["events"], command, recording["question"])

    # Size the window so the answer (everything after the last tool call) fits.
    term = Terminal()
    for _, text, style in actions:
        term.feed(text, style)
    full = [("".join(ch for ch, _ in row)) for row in term.rows()]
    tool_rows = [i for i, row in enumerate(full) if row.startswith(("  ▸", "    │", "    ✓", "    ✗"))]
    answer_rows = len(full) - (tool_rows[-1] + 1 if tool_rows else 0)
    rows_on_screen = min(max(answer_rows + 1, 24), 60)

    char_w = regular.getlength("M")
    width = int(PAD * 2 + COLS * char_w)
    height = TITLE_BAR + PAD * 2 + rows_on_screen * LINE_H

    def draw(term: Terminal, cursor: bool) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, TITLE_BAR], fill=BAR_BG)
        for i, color in enumerate(WINDOW_DOTS):
            d.ellipse([14 + i * 20, TITLE_BAR // 2 - 6, 26 + i * 20, TITLE_BAR // 2 + 6], fill=color)
        d.text((width / 2, TITLE_BAR / 2), "data-analyst-agent", font=regular, fill=BAR_FG, anchor="mm")
        rows = term.rows()[-rows_on_screen:]
        for y, row in enumerate(rows):
            top = TITLE_BAR + PAD + y * LINE_H
            for x, (ch, style) in enumerate(row):
                color, weight = STYLES[style]
                d.text((PAD + x * char_w, top + LINE_H / 2), ch, font=font_for(ch, weight), fill=color, anchor="lm")
        if cursor:
            x, top = PAD + len(rows[-1]) * char_w, TITLE_BAR + PAD + (len(rows) - 1) * LINE_H
            d.rectangle([x, top + 2, x + char_w - 1, top + LINE_H - 2], fill=CURSOR)
        return img

    term, frames, durations, i = Terminal(), [], [], 0
    while i < len(actions):
        frame_time = actions[i][0]
        while i < len(actions) and actions[i][0] < frame_time + FRAME:
            term.feed(actions[i][1], actions[i][2])
            i += 1
        next_time = actions[i][0] if i < len(actions) else frame_time + HOLD_END
        at_prompt = "".join(ch for ch, _ in term.lines[-1]).endswith(("$ ", "> "))
        typing = i < len(actions) and actions[i][2] == "input"
        frames.append(draw(term, cursor=at_prompt or typing))
        durations.append(max(int((next_time - frame_time) * 1000), 40))

    # A fixed palette: ramps from the background to each text color cover anti-aliased edges.
    colors = [BG, BAR_BG, BAR_FG, CURSOR, *WINDOW_DOTS]
    for color, _ in STYLES.values():
        colors += [tuple(int(BG[k] + (color[k] - BG[k]) * s / 15) for k in range(3)) for s in range(1, 16)]
    palette = Image.new("P", (1, 1))
    flat = [v for c in colors[:256] for v in c]
    palette.putpalette(flat + [0] * (768 - len(flat)))
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # optimize=False keeps one shared palette; per-frame palettes make the file about a third bigger.
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    size_mb = out_path.stat().st_size / 1e6
    return f"{len(frames)} frames, {sum(durations) / 1000:.1f}s, {width}x{height}px, {size_mb:.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the agent and render the session as a GIF.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="The question to ask the agent.")
    parser.add_argument("--csv", default="data/coffee_sales.csv", help="The CSV file, relative to the project root.")
    parser.add_argument("--replay", type=Path, help="Re-render a saved recording instead of running the agent.")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif", help="Where to write the GIF.")
    args = parser.parse_args()

    if args.replay:
        recording = json.loads(args.replay.read_text())
    else:
        print(f"Running the agent on: {args.question}")
        recording = {"question": args.question, "csv": args.csv, "events": record(args.question, args.csv)}
        RECORDING.parent.mkdir(exist_ok=True)
        RECORDING.write_text(json.dumps(recording))
        print(f"Saved the recording to {RECORDING.relative_to(ROOT)}. Re-render it for free with --replay.")

    print(f"Wrote {args.out}: {render(recording, args.out)}")


if __name__ == "__main__":
    main()
