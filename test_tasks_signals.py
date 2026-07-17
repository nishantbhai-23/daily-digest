"""
Unit tests for tasks_signals.compute_task_signals — the deterministic
overdue/due-soon/blocked/stalled computation. No LLM involved; pure
function in/out checks.

Run: python3 -m unittest test_tasks_signals -v
"""

import unittest
from datetime import date

from tasks_signals import compute_task_signals


def _task(id, status="todo", priority="P2", due_date=None, created_at=None, subtasks=None, blocked_by=None):
    task = {
        "id": id,
        "title": f"Task {id}",
        "status": status,
        "priority": priority,
        "due_date": due_date,
        "created_at": created_at,
        "subtasks": subtasks or [],
    }
    if blocked_by:
        task["blocked_by"] = blocked_by
    return task


REF = date(2026, 7, 16)


class TestOverdue(unittest.TestCase):
    def test_past_due_not_done_is_overdue(self):
        tasks = [_task("A", due_date="2026-07-01", created_at="2026-06-20")]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(len(signals["overdue"]), 1)
        self.assertEqual(signals["overdue"][0]["days_overdue"], 15)

    def test_done_task_never_overdue(self):
        tasks = [_task("A", status="done", due_date="2026-06-01", created_at="2026-05-01")]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["overdue"], [])

    def test_future_due_date_not_overdue(self):
        tasks = [_task("A", due_date="2026-08-01", created_at="2026-07-01")]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["overdue"], [])


class TestDueSoon(unittest.TestCase):
    def test_within_seven_days_is_due_soon(self):
        tasks = [_task("A", due_date="2026-07-22", created_at="2026-07-01")]  # exactly 6 days out
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(len(signals["due_soon"]), 1)
        self.assertEqual(signals["due_soon"][0]["days_until_due"], 6)

    def test_exactly_seven_days_boundary_included(self):
        tasks = [_task("A", due_date="2026-07-23", created_at="2026-07-01")]  # exactly 7 days out
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(len(signals["due_soon"]), 1)

    def test_eight_days_out_not_due_soon(self):
        tasks = [_task("A", due_date="2026-07-24", created_at="2026-07-01")]  # 8 days out
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["due_soon"], [])


class TestBlocked(unittest.TestCase):
    def test_blocked_status_captured_with_reason_and_duration(self):
        tasks = [_task("A", status="blocked", created_at="2026-06-28", blocked_by="waiting on vendor")]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(len(signals["blocked"]), 1)
        self.assertEqual(signals["blocked"][0]["days_blocked"], 18)
        self.assertEqual(signals["blocked"][0]["blocked_by"], "waiting on vendor")


class TestStalled(unittest.TestCase):
    def test_old_low_progress_task_flagged_stalled(self):
        tasks = [_task(
            "A", created_at="2026-06-22",
            subtasks=[{"done": False}, {"done": False}, {"done": False}],
        )]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(len(signals["stalled"]), 1)
        self.assertEqual(signals["stalled"][0]["progress_ratio"], 0.0)

    def test_high_progress_task_not_flagged_even_if_old(self):
        tasks = [_task(
            "A", created_at="2026-06-01",
            subtasks=[{"done": True}, {"done": True}, {"done": False}],
        )]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["stalled"], [])

    def test_recently_created_low_progress_not_yet_stalled(self):
        tasks = [_task("A", created_at="2026-07-14", subtasks=[{"done": False}])]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["stalled"], [])

    def test_done_task_never_stalled(self):
        tasks = [_task(
            "A", status="done", created_at="2026-06-01",
            subtasks=[{"done": False}],
        )]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual(signals["stalled"], [])


class TestPrioritySorting(unittest.TestCase):
    def test_higher_priority_sorts_first(self):
        tasks = [
            _task("LOW", priority="P3", due_date="2026-07-01", created_at="2026-06-01"),
            _task("HIGH", priority="P0", due_date="2026-07-01", created_at="2026-06-01"),
        ]
        signals = compute_task_signals(tasks, reference_date=REF)
        self.assertEqual([t["id"] for t in signals["overdue"]], ["HIGH", "LOW"])


if __name__ == "__main__":
    unittest.main()
