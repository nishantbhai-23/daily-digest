"""
Tasks Parser
=============
Loads tasks.json into a list of dicts.

Unlike email/calendar/notes, tasks are already structured data — priority,
status, due_date, and assignee are typed fields set when the task was
created, not something to extract from prose. This is a thin loader, not
an extraction layer; see tasks_signals.py for the deterministic analysis
built on top of it.

Usage:
    from digest.parsers.tasks_parser import load_tasks

    tasks = load_tasks("data/tasks/tasks.json")
"""

import json
import os


def load_tasks(path: str) -> list[dict]:
    """Load tasks from a JSON file.

    Args:
        path: Path to the tasks JSON file.

    Returns:
        List of task dicts, or an empty list if the file is missing/invalid.
    """
    if not os.path.exists(path):
        print(f"❌ Error: Tasks file '{path}' not found.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse '{path}': {e}")
        return []
