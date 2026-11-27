"""Validate a sampled two-turn coding task with a schema tool and SQLite fixture."""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from openai_probe import chat, message_text, usage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Return the schema and allowed state values for the catalog.",
            "parameters": {
                "type": "object",
                "properties": {"catalog": {"type": "string"}},
                "required": ["catalog"],
            },
        },
    }
]
SCHEMA = {
    "ddl": (
        "CREATE TABLE projects (id INTEGER PRIMARY KEY);"
        "CREATE TABLE attempts (id INTEGER PRIMARY KEY, project_id INTEGER, "
        "task_id INTEGER, attempted_at INTEGER, status TEXT);"
    ),
    "statuses": ["passed", "failed"],
}
PROMPT = (
    "Write one SQLite SELECT query for the job_tracking catalog. First use "
    "inspect_schema to look up the schema. Count how many tasks latest passed "
    "in each project, including projects with no attempts. Latest means highest "
    "attempted_at and then highest attempt id for the same project and task. "
    "Return columns project_id and passed_tasks, ordered by project_id. "
    "After the tool result, reply with only the SQL query."
)


def run_query(query: str) -> tuple[list[str], list[tuple]]:
    """Run one read-only statement against a small, in-memory fixture."""
    with sqlite3.connect(":memory:") as database:
        database.executescript(SCHEMA["ddl"])
        database.executemany("INSERT INTO projects VALUES (?)", [(1,), (2,), (3,), (4,)])
        database.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?)",
            [
                (1, 1, 10, 100, "passed"),
                (2, 1, 10, 200, "failed"),
                (3, 1, 11, 201, "passed"),
                (4, 1, 11, 201, "failed"),
                (5, 1, 12, 300, "passed"),
                (6, 2, 10, 100, "passed"),
                (7, 2, 13, 300, "failed"),
                (8, 4, 10, 300, "failed"),
            ],
        )
        database.commit()
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        functions = {
            "coalesce", "count", "dense_rank", "ifnull", "max", "rank", "row_number", "sum"
        }
        database.set_authorizer(
            lambda action, arg1, name, *context: sqlite3.SQLITE_OK
            if action in allowed or (action == sqlite3.SQLITE_FUNCTION and name in functions)
            else sqlite3.SQLITE_DENY
        )
        database.set_progress_handler(lambda: 1, 100_000)
        cursor = database.execute(query)
        if cursor.description is None:
            raise ValueError("Expected a query that returns rows.")
        return [column[0] for column in cursor.description], cursor.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--model", default="dsv41-angelspec")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    sampling = dict(
        temperature=0.6,
        max_tokens=2048,
        reasoning_effort="high",
        extra={"seed": 1729},
        tools=TOOLS,
        timeout=50,
    )
    messages = [{"role": "user", "content": PROMPT}]
    report = {"ok": False, "turns": []}
    started = time.monotonic()
    try:
        first, elapsed = chat(args.base_url, args.model, messages, **sampling)
        content, _, calls = message_text(first)
        report["turns"].append({"elapsed_s": elapsed, "usage": usage(first)})
        if len(calls) != 1 or calls[0]["function"]["name"] != "inspect_schema":
            raise ValueError(f"Wrong tool call: {calls}")
        call = calls[0]
        if json.loads(call["function"]["arguments"]) != {"catalog": "job_tracking"}:
            raise ValueError(f"Wrong tool arguments: {calls}")
        messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(SCHEMA)})

        second, elapsed = chat(args.base_url, args.model, messages, **sampling)
        content, _, calls = message_text(second)
        report["turns"].append({"elapsed_s": elapsed, "usage": usage(second)})
        if calls:
            raise ValueError(f"Unexpected second tool call: {calls}")
        query = content.strip()
        fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*(.*?)```", query, re.S | re.I)
        if fenced:
            query = fenced[1].strip()
        report["query"] = query
        columns, rows = run_query(query)
        report["result"] = {"columns": columns, "rows": rows}
        report["ok"] = columns == ["project_id", "passed_tasks"] and rows == [
            (1, 1), (2, 1), (3, 0), (4, 0),
        ]
    except (ValueError, OSError, sqlite3.Error) as exc:
        report["error"] = str(exc)
    report["task_wall_s"] = round(time.monotonic() - started, 3)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ok": report["ok"], "task_wall_s": report["task_wall_s"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
