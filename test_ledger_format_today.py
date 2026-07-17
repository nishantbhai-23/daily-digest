"""
Unit test for ledger.format_today — the fix for REDUCE/orchestrator prompts
guessing "today" from ledger content (observed picking the earliest or the
most recent ledger date interchangeably, both wrong) instead of being told
the actual date.

Run: python3 -m unittest test_ledger_format_today -v
"""

import unittest
from datetime import date

from ledger import format_today


class TestFormatToday(unittest.TestCase):
    def test_includes_iso_date_and_weekday(self):
        result = format_today(reference_date=date(2026, 7, 17))
        self.assertIn("2026-07-17", result)
        self.assertIn("Friday", result)

    def test_defaults_to_real_now_when_unspecified(self):
        # Just confirms it doesn't crash and returns a non-empty string when
        # no reference_date is passed — the real datetime.now() path.
        result = format_today()
        self.assertTrue(len(result) > 0)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2}")


if __name__ == "__main__":
    unittest.main()
