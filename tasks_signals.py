"""
Tasks Signals
==============
Deterministic, LLM-free analysis of tasks.json — overdue-ness, blocked-task
staleness, stalled progress, priority-weighted ranking.

Deliberately not a MAP-REDUCE agent: email/calendar/notes all start as raw,
unstructured signal that needs extraction (an email doesn't come with a
priority field). Tasks are already the *output* of that kind of process —
priority, status, due_date, and subtask completion are typed fields, not
prose to interpret. There's nothing here for an LLM to extract; what's
missing is purely time-based arithmetic (is this overdue, how long has it
been blocked), the same category of "hard fact" the other agents already
compute in code rather than asking a model to infer (deep_work_conflicts,
sender staleness, checklist staleness).

This is meant primarily as input for a future orchestrator — tasks.json is
more useful as the ground truth other sources get cross-checked against
(does a task exist for this email promise? is the note's stalled decision
also a stalled task?) than as a fourth standalone digest.

Usage:
    from tasks_parser import load_tasks
    from tasks_signals import compute_task_signals, format_task_signals

    tasks = load_tasks("data/tasks/tasks.json")
    signals = compute_task_signals(tasks)
    print(format_task_signals(signals))
"""

from datetime import date, datetime

# A task open this many days with less than this fraction of subtasks done
# gets flagged as stalled. Tunable heuristic, not a hard rule.
STALLED_MIN_DAYS_OPEN = 7
STALLED_MAX_PROGRESS_RATIO = 0.34

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def _parse_date(value: str) -> date:
    """Parse a bare date ('2026-07-18') or an ISO datetime with offset
    ('2026-06-26T09:00:00-07:00') into a date.
    """
    if "T" in value:
        return datetime.fromisoformat(value).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def compute_task_signals(tasks: list[dict], reference_date: date = None) -> dict:
    """Compute deterministic time-sensitive signals across all tasks.

    Args:
        tasks: Parsed task dicts (from tasks_parser.load_tasks).
        reference_date: "Today" for overdue/staleness math; defaults to the
            real current date.

    Returns:
        {
          "overdue": [...],    # due_date in the past, not done
          "due_soon": [...],   # due within 7 days, not done
          "blocked": [...],    # status == "blocked", with days_blocked
          "stalled": [...],    # open a while with little/no subtask progress
        }
        Each list is sorted priority-first, then most-urgent-first.
    """
    reference_date = reference_date or datetime.now().date()

    overdue, due_soon, blocked, stalled = [], [], [], []

    for task in tasks:
        status = task.get("status", "todo")
        is_done = status == "done"
        due = _parse_date(task["due_date"]) if task.get("due_date") else None
        created = _parse_date(task["created_at"]) if task.get("created_at") else None

        subtasks = task.get("subtasks", [])
        done_count = sum(1 for s in subtasks if s.get("done"))
        total_count = len(subtasks)

        entry = {
            "id": task["id"],
            "title": task["title"],
            "priority": task.get("priority", "?"),
            "status": status,
            "due_date": task.get("due_date"),
            "subtasks_done": done_count,
            "subtasks_total": total_count,
        }

        if due and not is_done:
            days_until_due = (due - reference_date).days
            if days_until_due < 0:
                overdue.append({**entry, "days_overdue": -days_until_due})
            elif days_until_due <= 7:
                due_soon.append({**entry, "days_until_due": days_until_due})

        if status == "blocked":
            days_blocked = (reference_date - created).days if created else None
            blocked.append({
                **entry,
                "blocked_by": task.get("blocked_by", "unspecified"),
                "days_blocked": days_blocked,
            })

        if not is_done and created:
            days_open = (reference_date - created).days
            progress_ratio = (done_count / total_count) if total_count else 0.0
            if days_open >= STALLED_MIN_DAYS_OPEN and progress_ratio < STALLED_MAX_PROGRESS_RATIO:
                stalled.append({
                    **entry,
                    "days_open": days_open,
                    "progress_ratio": round(progress_ratio, 2),
                })

    overdue.sort(key=lambda t: (_PRIORITY_ORDER.get(t["priority"], 9), -t["days_overdue"]))
    due_soon.sort(key=lambda t: (_PRIORITY_ORDER.get(t["priority"], 9), t["days_until_due"]))
    blocked.sort(key=lambda t: (_PRIORITY_ORDER.get(t["priority"], 9), -(t["days_blocked"] or 0)))
    stalled.sort(key=lambda t: (_PRIORITY_ORDER.get(t["priority"], 9), -t["days_open"]))

    return {"overdue": overdue, "due_soon": due_soon, "blocked": blocked, "stalled": stalled}


def format_task_signals(signals: dict) -> str:
    """Render computed signals as readable text (for a CLI check, or as
    ground-truth context handed to an orchestrator/LLM later).
    """
    lines = []

    if signals["overdue"]:
        lines.append("OVERDUE:")
        for t in signals["overdue"]:
            lines.append(
                f"  - [{t['priority']}] {t['id']} {t['title']} — "
                f"{t['days_overdue']}d overdue (due {t['due_date']}), "
                f"{t['subtasks_done']}/{t['subtasks_total']} subtasks done"
            )

    if signals["due_soon"]:
        lines.append("DUE SOON (within 7 days):")
        for t in signals["due_soon"]:
            lines.append(
                f"  - [{t['priority']}] {t['id']} {t['title']} — "
                f"due in {t['days_until_due']}d ({t['due_date']})"
            )

    if signals["blocked"]:
        lines.append("BLOCKED:")
        for t in signals["blocked"]:
            days = f"{t['days_blocked']}d" if t["days_blocked"] is not None else "unknown duration"
            lines.append(
                f"  - [{t['priority']}] {t['id']} {t['title']} — "
                f"blocked {days}: {t['blocked_by']}"
            )

    if signals["stalled"]:
        lines.append("STALLED (open a while, little progress):")
        for t in signals["stalled"]:
            lines.append(
                f"  - [{t['priority']}] {t['id']} {t['title']} — "
                f"open {t['days_open']}d, {t['subtasks_done']}/{t['subtasks_total']} subtasks "
                f"({int(t['progress_ratio'] * 100)}%)"
            )

    return "\n".join(lines) if lines else "(No overdue, due-soon, blocked, or stalled tasks.)"


def main():
    from tasks_parser import load_tasks

    tasks = load_tasks("./data/tasks/tasks.json")
    signals = compute_task_signals(tasks)
    print(format_task_signals(signals))


if __name__ == "__main__":
    main()
