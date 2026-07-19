"""
Unit tests for cross_reference.build_cross_reference_index — the
deterministic Stage 0 matching engine. No LLM involved; pure function
in/out checks against synthetic fixtures, plus an integration check
against the real generated dataset's known cross-source patterns (Elena
Marsh, Diane's board update) found by hand during development.

Run: python3 -m unittest test_cross_reference -v
"""

import unittest

from digest.core.cross_reference import build_cross_reference_index, _title_keywords


def _task_signals(**buckets):
    base = {"overdue": [], "due_soon": [], "blocked": [], "stalled": []}
    base.update(buckets)
    return base


class TestTitleKeywords(unittest.TestCase):
    def test_filters_stopwords_and_short_words(self):
        keywords = _title_keywords("Decide on Elena Marsh offer (Senior Engineer)")
        self.assertIn("Elena", keywords)
        self.assertIn("Marsh", keywords)
        self.assertIn("Senior", keywords)
        self.assertIn("Engineer", keywords)
        self.assertNotIn("on", keywords)

    def test_empty_title_returns_no_keywords(self):
        self.assertEqual(_title_keywords("..."), [])


class TestMatching(unittest.TestCase):
    def test_task_mentioned_in_email_is_indexed(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-219", "title": "Decide on Elena Marsh offer", "priority": "P1"},
        ])
        email_ledger = [
            {"day": "2026-06-22", "delta": {"action_items": [{"description": "Review Elena Marsh feedback"}]}},
        ]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertIn("TESS-219", index)
        self.assertEqual(index["TESS-219"]["mentioned_in"][0]["source"], "email")
        self.assertEqual(index["TESS-219"]["mentioned_in"][0]["day"], "2026-06-22")

    def test_task_with_no_mentions_not_indexed(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-999", "title": "Renew office lease paperwork", "priority": "P3"},
        ])
        email_ledger = [{"day": "2026-06-22", "delta": {"action_items": [{"description": "Review Elena Marsh feedback"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_only_flagged_tasks_considered(self):
        # A task not present in any bucket shouldn't be indexed even if its
        # keywords would otherwise match — the function only receives
        # already-flagged tasks via task_signals.
        task_signals = _task_signals()  # nothing flagged
        email_ledger = [{"day": "2026-06-22", "delta": {"action_items": [{"description": "Elena Marsh"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_compacted_entries_skipped(self):
        task_signals = _task_signals(stalled=[
            {"id": "TESS-1", "title": "Multi-warehouse decision", "priority": "P1"},
        ])
        notes_ledger = [
            {"day": "2026-W25", "compacted": True, "delta": {"decisions": [{"description": "multi-warehouse"}]}},
        ]
        index = build_cross_reference_index([], [], notes_ledger, task_signals)
        self.assertEqual(index, {})

    def test_mentions_aggregated_across_sources(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-212", "title": "Send Diane the Q2 board update", "priority": "P2"},
        ])
        email_ledger = [{"day": "2026-06-16", "delta": {"deadlines": [{"description": "Diane wants board update"}]}}]
        notes_ledger = [{"day": "2026-07-01", "delta": {"decisions": [{"description": "Board update prep for Diane"}]}}]
        index = build_cross_reference_index(email_ledger, [], notes_ledger, task_signals)
        sources = {m["source"] for m in index["TESS-212"]["mentioned_in"]}
        self.assertEqual(sources, {"email", "notes"})


class TestAgainstRealData(unittest.TestCase):
    """Integration check against the actual generated dataset — the same
    Elena Marsh / Diane board-update patterns found by hand tonight should
    turn up automatically here.
    """

    def test_elena_marsh_or_diane_pattern_found_in_real_notes(self):
        # This doesn't require MAP-phase ledgers (which need an LLM) — it
        # exercises the matcher directly against notes ledger-shaped data
        # built from the real notes_parser output, so it's still a fast,
        # LLM-free check that the matching logic works on real content.
        from digest.parsers.notes_parser import load_notes

        notes = load_notes("data/notes/")
        fake_notes_ledger = [
            {"day": n["created_at"], "delta": {"decisions": [{"description": n["body"]}]}}
            for n in notes
        ]
        task_signals = _task_signals(overdue=[
            {"id": "TESS-212", "title": "Send Diane the Q2 board update", "priority": "P2"},
        ])
        index = build_cross_reference_index([], [], fake_notes_ledger, task_signals)
        self.assertIn("TESS-212", index)


if __name__ == "__main__":
    unittest.main()
