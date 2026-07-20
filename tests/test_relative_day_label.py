"""
Unit tests for ledger.relative_day_label and its wiring into each agent's
render_ledger_as_text — the fix for a stale calendar event (2026-07-19)
being narrated as happening "today" in a brief actually assembled on
2026-07-21, found live in the ceo-tenant-2026 tenant. Rather than trusting
a synthesis call to correctly subtract two dates it's shown separately,
the relative "how old is this" fact is computed in code and folded
directly into the ledger text the model reads.

Run: python3 -m unittest tests.test_relative_day_label -v
"""

import unittest
from datetime import date

from digest.agents import calendar_agent, notes_agent, triage_agent
from digest.core.ledger import relative_day_label


class TestRelativeDayLabel(unittest.TestCase):
    def test_today(self):
        self.assertEqual(relative_day_label("2026-07-21", reference_date=date(2026, 7, 21)), "TODAY")

    def test_yesterday(self):
        self.assertEqual(relative_day_label("2026-07-20", reference_date=date(2026, 7, 21)), "YESTERDAY")

    def test_multiple_days_ago(self):
        label = relative_day_label("2026-07-19", reference_date=date(2026, 7, 21))
        self.assertEqual(label, "2 DAYS AGO — NOT today")

    def test_future_dated_entry_passes_through(self):
        # Shouldn't normally occur, but must not raise or claim staleness
        # for a day that hasn't happened yet.
        label = relative_day_label("2026-07-22", reference_date=date(2026, 7, 21))
        self.assertEqual(label, "2026-07-22")

    def test_defaults_to_real_now_when_unspecified(self):
        # Just confirms it doesn't crash and returns TODAY for today's date.
        today_str = date.today().isoformat()
        self.assertEqual(relative_day_label(today_str), "TODAY")


class TestRenderLedgerAsTextReferenceDate(unittest.TestCase):
    """Each agent's render_ledger_as_text must fold the relative-day tag
    into its header when reference_date is given, and stay byte-identical
    to the pre-fix output when it isn't — existing REDUCE-phase callers,
    where "today" doesn't reliably mean real wall-clock today (e.g. during
    a backfill), must be unaffected by this opt-in.
    """

    def test_calendar_tags_stale_entry(self):
        ledger = [{"day": "2026-07-19", "event_count": 1, "stats": {}, "delta": {}}]
        text = calendar_agent.render_ledger_as_text(ledger, reference_date=date(2026, 7, 21))
        self.assertIn("2026-07-19 (2 DAYS AGO — NOT today)", text)

    def test_calendar_omits_tag_without_reference_date(self):
        ledger = [{"day": "2026-07-19", "event_count": 1, "stats": {}, "delta": {}}]
        text = calendar_agent.render_ledger_as_text(ledger)
        self.assertIn("### 2026-07-19 (1 events)", text)
        self.assertNotIn("DAYS AGO", text)

    def test_calendar_compacted_entry_untagged(self):
        ledger = [{"day": "2026-07-06", "compacted": True, "event_count": 5, "stats": {}, "delta": {}}]
        text = calendar_agent.render_ledger_as_text(ledger, reference_date=date(2026, 7, 21))
        self.assertIn("Week of 2026-07-06", text)
        self.assertNotIn("DAYS AGO", text)

    def test_triage_tags_today_entry(self):
        ledger = [{"day": "2026-07-21", "email_count": 2, "stats": {}, "delta": {}}]
        text = triage_agent.render_ledger_as_text(ledger, reference_date=date(2026, 7, 21))
        self.assertIn("2026-07-21 (TODAY)", text)

    def test_notes_tags_yesterday_entry(self):
        ledger = [{"day": "2026-07-20", "note_id": "0001.md", "stats": {}, "delta": {}}]
        text = notes_agent.render_ledger_as_text(ledger, reference_date=date(2026, 7, 21))
        self.assertIn("2026-07-20 (YESTERDAY)", text)


if __name__ == "__main__":
    unittest.main()
