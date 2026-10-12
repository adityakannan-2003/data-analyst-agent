"""How the hosted page is being used, from its usage log.

    uv run python scripts/usage_report.py                        # this month so far
    uv run python scripts/usage_report.py --since 2026-10-01
    uv run python scripts/usage_report.py --sample 20            # also pick 20 answers to grade by hand
    uv run python scripts/usage_report.py --grades out/grading/sample_2026-11-01.csv

Live questions have no answer key, so the way to measure live accuracy is to grade a random sample
yourself: --sample writes a CSV with each question, the answer and the code the agent ran, with a
blank "correct" column. Fill it in with y or n, then --grades reports the share correct with a 95%
interval.

Reads DATABASE_URL from .env, the same Postgres as the hosted page; without it, the local SQLite log.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from usage_log import UsageLog, month_start  # noqa: E402


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


def export_sample(rows: list[dict], n: int, seed: int) -> Path:
    answered = [r for r in rows if r["status"] == "ok"]
    picked = random.Random(seed).sample(answered, min(n, len(answered)))
    out = ROOT / "out" / "grading" / f"sample_{datetime.now(timezone.utc):%Y-%m-%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "asked_at", "dataset", "question", "answer", "code", "correct", "notes"])
        for r in picked:
            steps = json.loads(r["steps"] or "[]")
            code = "\n\n".join(s["code"] for s in steps if s.get("type") == "tool_call" and s.get("code"))
            writer.writerow([r["id"], r["asked_at"], r["dataset"], r["question"], r["answer"], code, "", ""])
    print(f"\nWrote {len(picked)} answers to grade to {out.relative_to(ROOT)} (seed {seed}).")
    return out


def summarize_grades(path: Path) -> None:
    rows = list(csv.DictReader(path.open()))
    graded = [r for r in rows if r["correct"].strip().lower() in ("y", "n")]
    passed = sum(r["correct"].strip().lower() == "y" for r in graded)
    low, high = wilson(passed, len(graded))
    print(f"{passed} of {len(graded)} graded answers correct ({passed / max(len(graded), 1):.0%}, 95% CI {low:.0%} to {high:.0%})")
    if len(graded) < len(rows):
        print(f"{len(rows) - len(graded)} rows not graded yet (correct must be y or n)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on the hosted page's usage log.")
    parser.add_argument("--since", help="Start date, YYYY-MM-DD (default: the start of this month).")
    parser.add_argument("--sample", type=int, help="Also write this many random answers to a CSV to grade by hand.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --sample, so the pick can be repeated.")
    parser.add_argument("--grades", type=Path, help="Summarize a sample CSV you've graded, and stop.")
    args = parser.parse_args()

    if args.grades:
        summarize_grades(args.grades)
        return
    load_dotenv(ROOT / ".env")
    since = f"{args.since}T00:00:00+00:00" if args.since else month_start()
    rows = UsageLog().rows(since)
    report(rows, since)
    if args.sample:
        export_sample(rows, args.sample, args.seed)


if __name__ == "__main__":
    main()
