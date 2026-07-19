"""
Unit tests for calendar_agent.compute_day_stats — the deterministic
per-day calendar arithmetic (meeting load, focus-time protection,
back-to-back detection). No LLM involved; pure function in/out checks.

Run: python3 -m unittest test_calendar_stats -v
"""

import unittest
from datetime import datetime

from digest.agents.calendar_agent import compute_day_stats


def _event(summary, start, end, status="CONFIRMED", attendees=None):
    return {
        "uid": f"{summary}-{start}",
        "summary": summary,
        "status": status,
        "description": "",
        "location": "",
        "start": datetime.fromisoformat(start),
        "end": datetime.fromisoformat(end),
        "attendees": attendees or [],
    }


class TestDeepWorkConflicts(unittest.TestCase):
    def test_full_overlap_counted_as_conflict(self):
        events = [
            _event("🔒 Deep Work Block", "2026-07-07T09:00", "2026-07-07T11:00"),
            _event("Customer Call: Halberd", "2026-07-07T09:30", "2026-07-07T10:30"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(len(stats["deep_work_conflicts"]), 1)
        self.assertEqual(stats["deep_work_conflicts"][0]["overlap_hours"], 1.0)
        self.assertEqual(stats["focus_hours_eroded"], 1.0)
        self.assertEqual(stats["focus_hours_protected"], 1.0)

    def test_no_overlap_no_conflict(self):
        events = [
            _event("🔒 Deep Work Block", "2026-07-07T09:00", "2026-07-07T11:00"),
            _event("1:1 with Priya", "2026-07-07T11:30", "2026-07-07T12:00"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(stats["deep_work_conflicts"], [])
        self.assertEqual(stats["focus_hours_eroded"], 0.0)
        self.assertEqual(stats["focus_hours_protected"], 2.0)

    def test_partial_overlap_measured_correctly(self):
        events = [
            _event("🔒 Deep Work Block", "2026-07-07T09:00", "2026-07-07T11:00"),
            _event("GTM Pipeline Review", "2026-07-07T10:00", "2026-07-07T10:45"),
            _event("Wren — Pediatrician", "2026-07-07T10:15", "2026-07-07T11:00"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(len(stats["deep_work_conflicts"]), 2)
        overlaps = {c["summary"]: c["overlap_hours"] for c in stats["deep_work_conflicts"]}
        self.assertAlmostEqual(overlaps["GTM Pipeline Review"], 0.75)
        self.assertAlmostEqual(overlaps["Wren — Pediatrician"], 0.75)


class TestBackToBack(unittest.TestCase):
    def test_adjacent_meetings_counted(self):
        events = [
            _event("Standup", "2026-07-07T09:00", "2026-07-07T09:15"),
            _event("1:1 with Jordan", "2026-07-07T09:15", "2026-07-07T09:45"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(stats["back_to_back_count"], 1)

    def test_gap_not_counted(self):
        events = [
            _event("Standup", "2026-07-07T09:00", "2026-07-07T09:15"),
            _event("1:1 with Jordan", "2026-07-07T10:00", "2026-07-07T10:30"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(stats["back_to_back_count"], 0)


class TestExclusions(unittest.TestCase):
    def test_cancelled_event_excluded_from_meeting_count_but_tracked(self):
        events = [
            _event("Optional: Panel invite", "2026-07-07T10:00", "2026-07-07T11:00", status="CANCELLED"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(stats["meeting_count"], 0)
        self.assertEqual(stats["declined_or_cancelled_count"], 1)

    def test_cancelled_event_listed_by_name_not_just_counted(self):
        events = [
            _event("Optional: Panel invite", "2026-07-07T10:00", "2026-07-07T11:00", status="CANCELLED"),
        ]
        stats = compute_day_stats(events)
        self.assertEqual(len(stats["declined_or_cancelled"]), 1)
        self.assertEqual(stats["declined_or_cancelled"][0]["summary"], "Optional: Panel invite")
        self.assertEqual(stats["declined_or_cancelled"][0]["date"], "2026-07-07")

    def test_lunch_excluded_from_meeting_count(self):
        events = [_event("Lunch Break", "2026-07-07T12:00", "2026-07-07T13:00")]
        stats = compute_day_stats(events)
        self.assertEqual(stats["meeting_count"], 0)
        self.assertEqual(stats["meeting_hours"], 0.0)


class TestCategorization(unittest.TestCase):
    def test_one_on_one_and_external_categories(self):
        events = [
            _event("1:1 with Priya", "2026-07-07T11:00", "2026-07-07T11:30"),
            _event("Customer Call: Halberd", "2026-07-07T14:00", "2026-07-07T15:00"),
        ]
        stats = compute_day_stats(events)
        self.assertAlmostEqual(stats["meeting_hours_by_category"]["one_on_one"], 0.5)
        self.assertAlmostEqual(stats["meeting_hours_by_category"]["external"], 1.0)


if __name__ == "__main__":
    unittest.main()
