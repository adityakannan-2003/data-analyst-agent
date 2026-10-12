"""A record of every question the hosted page answers: what was asked, what the agent did, what it
cost, how long it took, and whether the visitor gave it a thumbs up or down.

It also answers "how much has the page spent this month?", which is how the monthly budget is
enforced. Postgres when DATABASE_URL is set (the hosted page); otherwise a SQLite file, so the
page and the tests run locally without a database server.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    asked_at TEXT NOT NULL,          -- UTC, ISO 8601
    session_id TEXT NOT NULL,        -- random per browser session; says nothing about who it is
    dataset TEXT NOT NULL,           -- a sample dataset's name, or 'upload'
    data_rows INTEGER,
    data_cols INTEGER,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,            -- ok, truncated, refused, max_steps, error, or cancelled
    error TEXT,
    steps TEXT,                      -- JSON: the code the agent ran, and for sample datasets what came back
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    tool_calls INTEGER,
    cost_usd REAL,
    latency_s REAL,
    feedback INTEGER,                -- 1 thumbs up, -1 thumbs down, NULL none
    feedback_at TEXT
)
"""
COLUMNS = [
    "id", "asked_at", "session_id", "dataset", "data_rows", "data_cols", "question", "answer", "status", "error",
    "steps", "model", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "tool_calls",
    "cost_usd", "latency_s",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def month_start(at: datetime | None = None) -> str:
    at = at or datetime.now(timezone.utc)
    return at.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


class UsageLog:
    def __init__(self, database_url: str | None = None, sqlite_path: Path = Path("out/usage.db")):
        self.database_url = os.environ.get("DATABASE_URL") if database_url is None else database_url
        if self.database_url:
            import psycopg  # only needed on the hosted page

            self.connect = lambda: psycopg.connect(self.database_url, autocommit=True)
            self.mark = "%s"
        else:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.connect = lambda: sqlite3.connect(sqlite_path)
            self.mark = "?"
        self._run(SCHEMA)

    def record(self, **row) -> str:
        """Save one answered question. Returns its id, which feedback refers to."""
        row = {"id": uuid.uuid4().hex, "asked_at": now(), **row}
        if isinstance(row.get("steps"), list):
            row["steps"] = json.dumps(row["steps"])
        values = [row.get(column) for column in COLUMNS]
        self._run(f"INSERT INTO questions ({', '.join(COLUMNS)}) VALUES ({', '.join([self.mark] * len(COLUMNS))})", values)
        return row["id"]

    def set_feedback(self, question_id: str, value: int | None) -> None:
        self._run(f"UPDATE questions SET feedback = {self.mark}, feedback_at = {self.mark} WHERE id = {self.mark}",
                  [value, now() if value is not None else None, question_id])

    def spent_since(self, since: str) -> float:
        rows = self._run(f"SELECT COALESCE(SUM(cost_usd), 0) FROM questions WHERE asked_at >= {self.mark}", [since], fetch=True)
        return float(rows[0][0])

    def rows(self, since: str = "") -> list[dict]:
        rows = self._run(f"SELECT {', '.join(COLUMNS)}, feedback FROM questions WHERE asked_at >= {self.mark} ORDER BY asked_at",
                         [since], fetch=True)
        return [dict(zip([*COLUMNS, "feedback"], row)) for row in rows]

    def _run(self, sql: str, params: list | None = None, fetch: bool = False):
        connection = self.connect()
        try:
            cursor = connection.cursor()
            cursor.execute(sql, params or [])
            result = cursor.fetchall() if fetch else None
            connection.commit()
            return result
        finally:
            connection.close()
