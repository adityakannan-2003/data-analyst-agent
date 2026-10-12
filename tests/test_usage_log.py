"""Tests for the hosted page's usage log, on SQLite."""

import json
from datetime import datetime, timezone

from usage_log import UsageLog, month_start


def make_log(tmp_path):
    return UsageLog(database_url="", sqlite_path=tmp_path / "usage.db")


def test_records_questions_and_totals_this_months_spend(tmp_path):
    log = make_log(tmp_path)
    first = log.record(session_id="s1", dataset="Coffee chain (made up)", question="Top store?", answer="Downtown.",
                       status="ok", steps=[{"type": "tool_call", "name": "bash", "code": "python a.py"}], cost_usd=0.06)
    log.record(session_id="s1", dataset="upload", question="What's in it?", status="error", cost_usd=0.02)

    assert round(log.spent_since(month_start()), 2) == 0.08
    rows = log.rows()
    assert [r["question"] for r in rows] == ["Top store?", "What's in it?"]
    assert rows[0]["id"] == first
    assert json.loads(rows[0]["steps"])[0]["code"] == "python a.py"


def test_spend_from_earlier_months_does_not_count(tmp_path):
    log = make_log(tmp_path)
    log.record(asked_at="2026-09-30T23:59:59+00:00", session_id="s", dataset="d", question="q", status="ok", cost_usd=5.0)
    log.record(asked_at="2026-10-01T00:00:00+00:00", session_id="s", dataset="d", question="q", status="ok", cost_usd=1.0)

    october = month_start(datetime(2026, 10, 12, 15, 30, tzinfo=timezone.utc))
    assert october == "2026-10-01T00:00:00+00:00"
    assert log.spent_since(october) == 1.0


def test_feedback_can_be_given_changed_and_cleared(tmp_path):
    log = make_log(tmp_path)
    question_id = log.record(session_id="s", dataset="d", question="q", status="ok", cost_usd=0.05)

    log.set_feedback(question_id, 1)
    assert log.rows()[0]["feedback"] == 1
    log.set_feedback(question_id, -1)
    assert log.rows()[0]["feedback"] == -1
    log.set_feedback(question_id, None)
    assert log.rows()[0]["feedback"] is None
