"""How the hosted page is being used, from its usage log.

    uv run python scripts/usage_report.py                        # this month so far
    uv run python scripts/usage_report.py --since 2026-10-01
    uv run python scripts/usage_report.py --month 2026-10 --sample 20   # pick 20 of October's questions to grade
    uv run python scripts/usage_report.py --grades out/grading/sample_2026-10_seed0.csv

Live questions have no answer key, so the way to measure live accuracy is to grade a random sample
yourself, once a month: --sample writes a Markdown file to read (each question, the answer, and the code
the agent ran with its output) and a CSV with a blank "correct" column. Fill it in with y, n or skip, then
--grades reports the share correct with a 95% interval and saves it, ids and verdicts only, to
evals/live_grading/.

Reads DATABASE_URL from .env, the same Postgres as the hosted page; without it, the local SQLite log.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from usage_log import WITHHELD, UsageLog, month_start  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0.0


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    rate = passed / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def report(rows: list[dict], since: str) -> None:
    print(f"Since {since[:10]}: {len(rows)} questions from {len({r['session_id'] for r in rows})} visits")
    if not rows:
        return

    print("\nPer day:")
    for day, n in sorted(Counter(r["asked_at"][:10] for r in rows).items())[-14:]:
        print(f"  {day}  {n:>4}  {'█' * min(n, 60)}")

    print("\nData:     " + ", ".join(f"{name} {n}" for name, n in Counter(r["dataset"] for r in rows).most_common()))
    print("Outcome:  " + ", ".join(f"{status} {n}" for status, n in Counter(r["status"] for r in rows).most_common()))

    costs = [r["cost_usd"] or 0 for r in rows]
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    print(f"Cost:     ${sum(costs):.2f} total, ${sum(costs) / len(costs):.3f} per question")
    print(f"Time:     {percentile(latencies, 0.5):.0f}s median, {percentile(latencies, 0.95):.0f}s at the 95th percentile")
    print(f"Steps:    {sum(r['tool_calls'] or 0 for r in rows) / len(rows):.1f} per question")

    rated = [r for r in rows if r["feedback"] is not None]
    up = sum(r["feedback"] == 1 for r in rated)
    print(f"Ratings:  {up} up, {len(rated) - up} down ({len(rated)} of {len(rows)} answers rated)")
    for r in [r for r in rated if r["feedback"] == -1][-5:]:
        print(f"  down  {r['id'][:8]}  {r['dataset']}: {r['question'][:90]}")

    per_visit = Counter(r["session_id"] for r in rows)
    print(f"Visits:   {sum(n == 1 for n in per_visit.values())} asked one question, "
          f"{sum(n > 1 for n in per_visit.values())} asked follow-ups, most in one visit: {max(per_visit.values())}")


GUIDE = """How to grade (about 1 to 2 minutes each). Put one of these in the `correct` column of the CSV:

- y: it answers the question asked, and its numbers and claims follow from the code it ran (and the
  output, where kept), with no mistake a careful analyst would catch. A fair "the data can't tell us
  that" counts as y.
- n: a wrong number or claim, a misread question, a problem in the data it missed that changes the
  answer, or no usable answer (an error, stopped early, or refused a fair question).
- skip: your own test questions, or one you can't judge. Skips are left out of the rate and counted.
"""


def export_sample(rows: list[dict], n: int, seed: int, name: str) -> Path:
    """Write n random questions to grade: a CSV for the verdicts and a Markdown copy to read them in.
    Every question counts, not just finished answers, so the rate is the share of questions answered right."""
    # cancelled: the visitor left; withheld: the file looked personal, so nothing was kept to grade
    gradable = [r for r in rows if r["status"] != "cancelled" and r["question"] != WITHHELD]
    picked = random.Random(seed).sample(gradable, min(n, len(gradable)))
    out = ROOT / "out" / "grading" / f"sample_{name}_seed{seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    reading = [f"# Live answers to grade: {name} (seed {seed}, {len(picked)} of {len(gradable)} questions)\n", GUIDE]
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "asked_at", "dataset", "status", "question", "answer", "code", "correct", "notes"])
        for i, r in enumerate(picked, 1):
            steps = json.loads(r["steps"] or "[]")
            code = "\n\n".join(s["code"] for s in steps if s.get("type") == "tool_call" and s.get("code"))
            writer.writerow([r["id"], r["asked_at"], r["dataset"], r["status"], r["question"], r["answer"], code, "", ""])
            reading += [f"\n---\n\n## {i}. `{r['id'][:8]}` · {r['dataset']} · {r['status']}\n",
                        f"**Question:** {r['question']}\n", f"**Answer:**\n\n{r['answer'] or '(none)'}\n"]
            if r["error"]:
                reading.append(f"**Error:** {r['error']}\n")
            for s in steps:
                if s.get("type") == "note" and s.get("text"):
                    reading.append(f"_{s['text']}_\n")
                elif s.get("type") == "tool_call" and s.get("code"):
                    reading.append(f"Ran:\n\n```\n{s['code']}\n```\n")
                elif s.get("type") == "tool_result":
                    output = s.get("output")
                    reading.append("Output: not kept (the visitor's own file)\n" if output is None else
                                   f"{'Output' if s.get('ok') else 'Error'}:\n\n```\n{output}\n```\n")
    out.with_suffix(".md").write_text("\n".join(reading))
    print(f"\nWrote {len(picked)} questions to grade (seed {seed}): read {out.with_suffix('.md').relative_to(ROOT)}, "
          f"grade in {out.relative_to(ROOT)}")
    return out


def summarize_grades(path: Path) -> dict:
    """Print the share graded correct with a 95% interval, and save it with only ids and verdicts (no
    visitor text, so it can be committed) to evals/live_grading/."""
    rows = list(csv.DictReader(path.open()))
    verdicts = [r["correct"].strip().lower() for r in rows]
    graded = [v for v in verdicts if v in ("y", "n")]
    passed = graded.count("y")
    low, high = wilson(passed, len(graded))
    print(f"{passed} of {len(graded)} graded answers correct ({passed / max(len(graded), 1):.0%}, 95% CI {low:.0%} to {high:.0%})")
    skipped, blank = verdicts.count("skip"), sum(v not in ("y", "n", "skip") for v in verdicts)
    if skipped or blank:
        print(f"{skipped} skipped, {blank} not graded yet (correct must be y, n or skip)")
    result = {
        "sample": path.name, "questions": len(rows), "graded": len(graded), "correct": passed, "skipped": skipped,
        "not_graded": blank, "rate": round(passed / len(graded), 4) if graded else None,
        "ci95_wilson": [round(low, 4), round(high, 4)],
        "rows": [{"id": r["id"], "dataset": r["dataset"],
                  "status": r["status"], "correct": v or None} for r, v in zip(rows, verdicts)],
    }
    out = ROOT / "evals" / "live_grading" / f"{path.stem.removeprefix('sample_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Saved to {out.relative_to(ROOT)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on the hosted page's usage log.")
    parser.add_argument("--since", help="Start date, YYYY-MM-DD (default: the start of this month).")
    parser.add_argument("--month", help="One calendar month, YYYY-MM, instead of --since.")
    parser.add_argument("--sample", type=int, help="Also write this many random questions to grade by hand.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --sample, so the pick can be repeated.")
    parser.add_argument("--grades", type=Path, help="Summarize a sample CSV you've graded, and stop.")
    args = parser.parse_args()

    if args.grades:
        summarize_grades(args.grades)
        return
    load_dotenv(ROOT / ".env")
    if args.month:
        start = datetime.strptime(args.month, "%Y-%m").replace(tzinfo=timezone.utc)
        since, until = month_start(start), month_start(start + timedelta(days=32))
    else:
        since, until = f"{args.since}T00:00:00+00:00" if args.since else month_start(), "9999"
    rows = [r for r in UsageLog(read_only=True).rows(since) if r["asked_at"] < until]
    report(rows, since)
    if args.sample:
        export_sample(rows, args.sample, args.seed, args.month or f"{datetime.now(timezone.utc):%Y-%m-%d}")


if __name__ == "__main__":
    main()
