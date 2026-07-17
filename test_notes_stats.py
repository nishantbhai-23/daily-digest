"""
Unit tests for notes_agent.compute_note_stats — the deterministic
checklist open/done + staleness computation. No LLM involved; pure
function in/out checks.

Run: python3 -m unittest test_notes_stats -v
"""

import unittest
from datetime import date

from notes_agent import compute_note_stats


def _note(created_at, checklist_items):
    return {"created_at": created_at, "checklist_items": checklist_items}


class TestChecklistCounts(unittest.TestCase):
    def test_mixed_open_and_done(self):
        note = _note("2026-07-07", [
            {"text": "Decide multi-warehouse approach", "done": False},
            {"text": "Sign SAFE amendment", "done": False},
            {"text": "Book reference calls", "done": True},
        ])
        stats = compute_note_stats(note, reference_date=date(2026, 7, 16))
        self.assertEqual(stats["checklist_open"], 2)
        self.assertEqual(stats["checklist_done"], 1)

    def test_all_done(self):
        note = _note("2026-07-07", [{"text": "x", "done": True}])
        stats = compute_note_stats(note, reference_date=date(2026, 7, 16))
        self.assertEqual(stats["checklist_open"], 0)
        self.assertEqual(stats["stale_open_items"], [])

    def test_no_checklist_items(self):
        note = _note("2026-07-01", [])
        stats = compute_note_stats(note, reference_date=date(2026, 7, 16))
        self.assertEqual(stats["checklist_open"], 0)
        self.assertEqual(stats["checklist_done"], 0)
        self.assertEqual(stats["stale_open_items"], [])


class TestStaleness(unittest.TestCase):
    def test_days_open_computed_against_reference_date(self):
        note = _note("2026-07-07", [{"text": "Decide multi-warehouse approach", "done": False}])
        stats = compute_note_stats(note, reference_date=date(2026, 7, 16))
        self.assertEqual(len(stats["stale_open_items"]), 1)
        self.assertEqual(stats["stale_open_items"][0]["days_open"], 9)

    def test_unknown_created_at_does_not_crash(self):
        note = _note("unknown", [{"text": "x", "done": False}])
        stats = compute_note_stats(note, reference_date=date(2026, 7, 16))
        self.assertEqual(stats["checklist_open"], 1)
        self.assertEqual(stats["stale_open_items"], [])


if __name__ == "__main__":
    unittest.main()
