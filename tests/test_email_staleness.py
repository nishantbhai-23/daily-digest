"""
Unit tests for triage_agent.compute_sender_staleness — the deterministic
"who's gone quiet" computation. No LLM involved; pure function in/out
checks.

Includes a regression test for a real bug found during development: an
earlier version filtered out senders with only 1 email, which excluded
exactly the case this function exists to catch (a single unanswered P0
ask, like an investor's one-time data-room request).

Run: python3 -m unittest test_email_staleness -v
"""

import unittest

from digest.agents.triage_agent import compute_sender_staleness


def _email(from_header, date_key):
    return {"from": from_header, "date_key": date_key}


class TestSingleEmailStillFlagged(unittest.TestCase):
    """Regression test: a sender with exactly one old email must still be
    flagged as quiet — this was the actual bug (count >= 2 filter) found
    while building the Marcus Webb quiet-investor-thread scenario.
    """

    def test_single_old_email_is_flagged(self):
        emails = [
            _email("Marcus Webb <marcus.webb@inflectionpointvc.com>", "2026-06-19"),
            _email("Ravi Chandra <ravi@tessera.io>", "2026-07-15"),
        ]
        results = compute_sender_staleness(emails, quiet_threshold_days=3)
        names = [r["name"] for r in results]
        self.assertIn("Marcus Webb", names)
        marcus = next(r for r in results if r["name"] == "Marcus Webb")
        self.assertEqual(marcus["email_count"], 1)
        self.assertEqual(marcus["days_since_last_contact"], 26)


class TestRecentSenderNotFlagged(unittest.TestCase):
    def test_recent_contact_not_flagged(self):
        emails = [
            _email("Priya Iyer <priya@tessera.io>", "2026-07-14"),
            _email("Priya Iyer <priya@tessera.io>", "2026-07-15"),
        ]
        results = compute_sender_staleness(emails, quiet_threshold_days=3)
        self.assertEqual(results, [])


class TestLastSeenIsMostRecent(unittest.TestCase):
    def test_multiple_emails_uses_max_date(self):
        emails = [
            _email("Carla Whitfield <carla.whitfield@halberdmfg.com>", "2026-06-21"),
            _email("Carla Whitfield <carla.whitfield@halberdmfg.com>", "2026-07-03"),
            _email("Ravi Chandra <ravi@tessera.io>", "2026-07-15"),
        ]
        results = compute_sender_staleness(emails, quiet_threshold_days=3)
        carla = next(r for r in results if "Carla" in r["name"])
        self.assertEqual(carla["last_seen"], "2026-07-03")
        self.assertEqual(carla["first_seen"], "2026-06-21")
        self.assertEqual(carla["email_count"], 2)
        self.assertEqual(carla["days_since_last_contact"], 12)


class TestFromHeaderParsing(unittest.TestCase):
    def test_name_and_email_extracted(self):
        emails = [
            _email("Diane Okafor <diane@bramblewoodvc.com>", "2026-06-16"),
            _email("Ravi Chandra <ravi@tessera.io>", "2026-07-15"),
        ]
        results = compute_sender_staleness(emails, quiet_threshold_days=3)
        diane = next(r for r in results if "Diane" in r["name"])
        self.assertEqual(diane["name"], "Diane Okafor")
        self.assertEqual(diane["email"], "diane@bramblewoodvc.com")


class TestSortOrder(unittest.TestCase):
    def test_most_quiet_first(self):
        emails = [
            _email("A <a@x.com>", "2026-06-16"),
            _email("B <b@x.com>", "2026-06-25"),
            _email("Ref <ref@x.com>", "2026-07-15"),
        ]
        results = compute_sender_staleness(emails, quiet_threshold_days=3)
        self.assertEqual([r["email"] for r in results], ["a@x.com", "b@x.com"])


if __name__ == "__main__":
    unittest.main()
