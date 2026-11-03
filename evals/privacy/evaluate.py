"""Measure the privacy filter (privacy.py) on the labelled columns in columns.jsonl.

    uv run python evals/privacy/evaluate.py rules               # privacy.py's verdicts -> predictions_rules.jsonl
    uv run python evals/privacy/evaluate.py score --split dev   # precision and recall on dev, while building it
    uv run python evals/privacy/evaluate.py claude              # Claude's verdicts on the test split (paid)
    uv run python evals/privacy/evaluate.py score               # every split and predictor -> results.json

`claude` refuses to run until predictions_rules.jsonl is committed and unchanged, so the filter's
numbers are fixed before any paid call. It stops if the spend passes $5.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import privacy  # noqa: E402
from pricing import cost, usage_dict  # noqa: E402
from scripts.usage_report import wilson  # noqa: E402

MODEL = "claude-opus-5-5"
CAP_USD = 5.0


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_rules(columns: list[dict]) -> None:
    rows = []
    for c in columns:
        found = privacy.personal_columns(pd.DataFrame({c["column"]: c["values"]}))
        kind = found.get(c["column"])
        rows.append({"id": c["id"], "personal": kind is not None, "kind": kind})
    (HERE / "predictions_rules.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote predictions for {len(rows)} columns to predictions_rules.jsonl")


def committed_unchanged(path: Path) -> bool:
    relative = str(path.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode == 0
    return tracked and clean


def run_claude(columns: list[dict]) -> None:
    import anthropic

    if not committed_unchanged(HERE / "predictions_rules.jsonl"):
        sys.exit("Commit predictions_rules.jsonl first: the filter's verdicts are fixed before any paid call.")
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    rules = (HERE / "LABELS.md").read_text().split("## Spot-check")[0]
    system = ("You check columns of a CSV file that a visitor uploaded, to decide whether each column holds personal "
              "data. Follow these rules exactly.\n\n" + rules)
    schema = {
        "type": "object",
        "properties": {"personal": {"type": "boolean"}, "kind": {"type": "string", "enum": [*privacy.KINDS, "none"]}},
        "required": ["personal", "kind"],
        "additionalProperties": False,
    }
    out_path = HERE / f"predictions_{MODEL}.jsonl"
    done = {r["id"]: r for r in load(out_path)} if out_path.exists() else {}  # rerunning resumes
    spent = sum(r["cost_usd"] for r in done.values())
    test = [c for c in columns if c["split"] == "test"]
    for i, c in enumerate(test, 1):
        if c["id"] in done:
            continue
        if spent > CAP_USD:
            sys.exit(f"Stopped at the ${CAP_USD:.0f} cap after {len(done)} columns.")
        values = "\n".join(c["values"])
        t0 = time.monotonic()
        response = client.beta.messages.create(
            model=MODEL, max_tokens=4000, output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": f"Column name: {c['column']}\nIts first {len(c['values'])} values, one per line:\n{values}"}],
        )
        usage = usage_dict(response.usage)
        spent_now = cost(response.model, usage) or cost(MODEL, usage)
        text = next((b.text for b in response.content if b.type == "text"), "")
        verdict = json.loads(text) if response.stop_reason != "refusal" and text else {"personal": None, "kind": None}
        row = {"id": c["id"], "personal": verdict["personal"], "kind": None if verdict["kind"] == "none" else verdict["kind"],
               "model": response.model, "stop_reason": response.stop_reason, "input_tokens": usage["input_tokens"],
               "output_tokens": usage["output_tokens"], "cost_usd": round(spent_now, 6), "latency_s": round(time.monotonic() - t0, 2)}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done[c["id"]] = row
        spent += spent_now
        if i % 25 == 0 or i == len(test):
            print(f"  {i}/{len(test)} columns, ${spent:.3f} so far")
    print(f"Claude ({MODEL}) judged {len(done)} test columns for ${spent:.3f}")


def metrics(pairs: list[tuple[dict, dict]]) -> dict:
    tp = sum(c["personal"] and p["personal"] is True for c, p in pairs)
    fp = sum(not c["personal"] and p["personal"] is True for c, p in pairs)
    fn = sum(c["personal"] and p["personal"] is not True for c, p in pairs)
    tn = len(pairs) - tp - fp - fn
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "columns": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "precision_ci95": [round(x, 4) for x in wilson(tp, tp + fp)],
        "recall": round(recall, 4), "recall_ci95": [round(x, 4) for x in wilson(tp, tp + fn)],
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "missed": [f"{c['id']} {c['column']} ({c['kind']})" for c, p in pairs if c["personal"] and p["personal"] is not True],
        "false_alarms": [f"{c['id']} {c['column']} (said {p['kind']})" for c, p in pairs if not c["personal"] and p["personal"] is True],
    }


def score(columns: list[dict], splits: list[str]) -> dict:
    results = {}
    for path in sorted(HERE.glob("predictions_*.jsonl")):
        predictor = path.stem.removeprefix("predictions_")
        predicted = {r["id"]: r for r in load(path)}
        results[predictor] = {}
        for split in splits:
            pairs = [(c, predicted[c["id"]]) for c in columns if c["id"] in predicted and split in (c["split"], "all")]
            if pairs:
                results[predictor][split] = metrics(pairs)
        test = [(c, predicted[c["id"]]) for c in columns if c["id"] in predicted and c["split"] == "test" and c["personal"]]
        caught = Counter(c["kind"] for c, p in test if p["personal"] is True)
        results[predictor]["test_recall_by_kind"] = {k: f"{caught[k]}/{n}" for k, n in Counter(c["kind"] for c, _ in test).most_common()}
        rows = list(predicted.values())
        if rows and "cost_usd" in rows[0]:
            results[predictor]["cost_usd"] = round(sum(r["cost_usd"] for r in rows), 4)
            results[predictor]["median_latency_s"] = sorted(r["latency_s"] for r in rows)[len(rows) // 2]
    for predictor, by_split in results.items():
        for split in splits:
            if split in by_split:
                m = by_split[split]
                print(f"{predictor:>18} {split:>4}: precision {m['precision']:.0%} ({m['tp']}/{m['tp'] + m['fp']}, "
                      f"95% CI {m['precision_ci95'][0]:.0%}-{m['precision_ci95'][1]:.0%}), recall {m['recall']:.0%} "
                      f"({m['tp']}/{m['tp'] + m['fn']}, {m['recall_ci95'][0]:.0%}-{m['recall_ci95'][1]:.0%})")
                for line in m["missed"]:
                    print(f"{'':>24}missed       {line}")
                for line in m["false_alarms"]:
                    print(f"{'':>24}false alarm  {line}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["rules", "claude", "score"])
    parser.add_argument("--split", choices=["dev", "test", "all"], help="Score one split only (default: all three, saved).")
    args = parser.parse_args()
    columns = load(HERE / "columns.jsonl")
    if args.command == "rules":
        run_rules(columns)
    elif args.command == "claude":
        run_claude(columns)
    elif args.split:
        score(columns, [args.split])
    else:
        results = score(columns, ["dev", "test", "all"])
        (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
        print("Saved results.json")


if __name__ == "__main__":
    main()
