"""
Unit tests for ledger.compact_ledger — specifically the fix for a real bug
found during development: compaction only ever merged 'delta' and silently
dropped 'stats' (calendar/notes deterministic facts) and 'note_id' (notes
resume-tracking key) when collapsing old entries into weekly rollups.

Run: python3 -m unittest test_ledger_compact -v
"""

import unittest

from digest.core.ledger import compact_ledger


class FakeLLM:
    def chat_json(self, messages):
        return {"merged": True}


class TestStatsPreservedOnCompaction(unittest.TestCase):
    def test_stats_survive_as_stats_by_day(self):
        ledger = [
            {
                "day": "2026-06-16",
                "event_count": 3,
                "stats": {"meeting_count": 2, "deep_work_conflicts": []},
                "delta": {"notable_events": []},
            },
            {
                "day": "2026-06-17",
                "event_count": 4,
                "stats": {"meeting_count": 3, "deep_work_conflicts": [{"summary": "x"}]},
                "delta": {"notable_events": []},
            },
        ]
        result = compact_ledger(ledger, FakeLLM(), "merge these", retention_days=5, count_key="event_count")
        compacted = [e for e in result if e.get("compacted")]
        self.assertEqual(len(compacted), 1)
        stats_by_day = compacted[0]["stats_by_day"]
        self.assertEqual(stats_by_day["2026-06-16"]["meeting_count"], 2)
        self.assertEqual(stats_by_day["2026-06-17"]["deep_work_conflicts"], [{"summary": "x"}])

    def test_no_stats_key_when_source_has_no_stats(self):
        # Email ledger entries currently have no 'stats' field — compaction
        # shouldn't invent one.
        ledger = [
            {"day": "2026-06-16", "email_count": 5, "delta": {"deadlines": []}},
        ]
        result = compact_ledger(ledger, FakeLLM(), "merge these", retention_days=5, count_key="email_count")
        compacted = [e for e in result if e.get("compacted")]
        self.assertEqual(len(compacted), 1)
        self.assertNotIn("stats_by_day", compacted[0])


class TestNoteIdsPreservedOnCompaction(unittest.TestCase):
    def test_note_ids_survive(self):
        ledger = [
            {"day": "2026-06-16", "note_id": "a.md", "item_count": 2, "delta": {"decisions": []}},
            {"day": "2026-06-17", "note_id": "b.md", "item_count": 1, "delta": {"decisions": []}},
        ]
        result = compact_ledger(ledger, FakeLLM(), "merge these", retention_days=5, count_key="item_count")
        compacted = [e for e in result if e.get("compacted")]
        self.assertEqual(compacted[0]["note_ids_covered"], ["a.md", "b.md"])


class TestCountsStillConserved(unittest.TestCase):
    def test_count_sum_unaffected_by_stats_fix(self):
        ledger = [
            {"day": "2026-06-16", "event_count": 3, "stats": {"a": 1}, "delta": {}},
            {"day": "2026-06-17", "event_count": 4, "stats": {"a": 2}, "delta": {}},
        ]
        result = compact_ledger(ledger, FakeLLM(), "merge these", retention_days=5, count_key="event_count")
        compacted = [e for e in result if e.get("compacted")]
        self.assertEqual(compacted[0]["event_count"], 7)


if __name__ == "__main__":
    unittest.main()
