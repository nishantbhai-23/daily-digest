"""
Unit tests for ledger.check_data_freshness — deterministic staleness
detection per source, used by the orchestrator to honor the profile's
honesty rule about flagging stale data instead of silently digesting it
as current.

Run: python3 -m unittest test_ledger_freshness -v
"""

import unittest
from datetime import date

from digest.core.ledger import check_data_freshness

REF = date(2026, 7, 16)


class TestFreshness(unittest.TestCase):
    def test_recent_ledger_not_stale(self):
        ledgers = {"email": [{"day": "2026-07-15", "delta": {}}]}
        result = check_data_freshness(ledgers, reference_date=REF, stale_after_days=1)
        self.assertFalse(result["email"]["is_stale"])
        self.assertEqual(result["email"]["days_stale"], 1)

    def test_old_ledger_flagged_stale(self):
        ledgers = {"email": [{"day": "2026-07-10", "delta": {}}]}
        result = check_data_freshness(ledgers, reference_date=REF, stale_after_days=1)
        self.assertTrue(result["email"]["is_stale"])
        self.assertEqual(result["email"]["days_stale"], 6)

    def test_empty_ledger_flagged_stale(self):
        ledgers = {"calendar": []}
        result = check_data_freshness(ledgers, reference_date=REF)
        self.assertTrue(result["calendar"]["is_stale"])
        self.assertIsNone(result["calendar"]["most_recent_day"])

    def test_uses_most_recent_entry_not_first(self):
        ledgers = {"notes": [{"day": "2026-06-01", "delta": {}}, {"day": "2026-07-14", "delta": {}}]}
        result = check_data_freshness(ledgers, reference_date=REF, stale_after_days=1)
        self.assertEqual(result["notes"]["most_recent_day"], "2026-07-14")

    def test_multiple_sources_checked_independently(self):
        ledgers = {
            "email": [{"day": "2026-07-15", "delta": {}}],
            "calendar": [{"day": "2026-07-01", "delta": {}}],
        }
        result = check_data_freshness(ledgers, reference_date=REF, stale_after_days=1)
        self.assertFalse(result["email"]["is_stale"])
        self.assertTrue(result["calendar"]["is_stale"])


if __name__ == "__main__":
    unittest.main()
