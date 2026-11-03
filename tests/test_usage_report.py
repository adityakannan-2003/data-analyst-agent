"""Tests for the monthly grading kit in scripts/usage_report.py."""

import csv
import json

from scripts import usage_report
from usage_log import UsageLog


def test_wilson_interval_matches_a_known_value():
    low, high = usage_report.wilson(17, 20)
    assert (round(low, 3), round(high, 3)) == (0.640, 0.948)


def test_sample_is_repeatable_and_graded_results_keep_no_visitor_text(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_report, "ROOT", tmp_path)
    log = UsageLog(database_url="", sqlite_path=tmp_path / "usage.db")
    for i in range(30):
        log.record(session_id=f"s{i}", dataset="upload" if i % 3 else "Coffee chain (made up)", question=f"secret question {i}",
                   answer=f"secret answer {i}", status="cancelled" if i == 0 else "ok",
                   steps=[{"type": "tool_call", "name": "bash", "code": f"print({i})"},
                          {"type": "tool_result", "ok": True, "output": None if i % 3 else f"{i}"}])
    rows = log.rows()

    path = usage_report.export_sample(rows, 20, seed=0, name="2026-10")
    again = usage_report.export_sample(rows, 20, seed=0, name="2026-10")
    picked = list(csv.DictReader(path.open()))
    assert path == again and len(picked) == 20
    assert "secret question 0" not in path.with_suffix(".md").read_text()  # a cancelled question isn't graded
    assert "Output: not kept" in path.with_suffix(".md").read_text()

    for row, verdict in zip(picked, ["y"] * 15 + ["n"] * 3 + ["skip", ""]):
        row["correct"] = verdict
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=picked[0].keys())
        writer.writeheader()
        writer.writerows(picked)

    result = usage_report.summarize_grades(path)
    assert (result["graded"], result["correct"], result["skipped"], result["not_graded"]) == (18, 15, 1, 1)
    saved = (tmp_path / "evals" / "live_grading" / "2026-10_seed0.json").read_text()
    assert json.loads(saved)["rate"] == round(15 / 18, 4)
    assert "secret" not in saved
