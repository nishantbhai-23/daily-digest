"""
Unit tests for triage_agent.compute_day_email_stats — the deterministic
per-day email stats added for envelope consistency with calendar/notes
ledger entries (both already had a 'stats' sibling to 'delta'; email
didn't). No LLM involved; pure function in/out checks.

Run: python3 -m unittest test_email_stats -v
"""

import unittest

from digest.agents.triage_agent import compute_day_email_stats


def _email(from_header, subject, body=""):
    return {"from": from_header, "subject": subject, "body": body}


class TestSenderCounting(unittest.TestCase):
    def test_unique_senders_deduplicated(self):
        batch = [
            _email("Priya Iyer <priya@tessera.io>", "PR up for review"),
            _email("Priya Iyer <priya@tessera.io>", "Re: PR up for review"),
            _email("Jordan Liu <jordan@tessera.io>", "Sprint notes"),
        ]
        stats = compute_day_email_stats(batch)
        self.assertEqual(stats["unique_senders"], 2)


class TestReplyCounting(unittest.TestCase):
    def test_re_and_fwd_prefixes_counted_as_replies(self):
        batch = [
            _email("A <a@x.com>", "Re: something"),
            _email("B <b@x.com>", "Fwd: something else"),
            _email("C <c@x.com>", "New topic"),
        ]
        stats = compute_day_email_stats(batch)
        self.assertEqual(stats["reply_count"], 2)
        self.assertEqual(stats["new_thread_count"], 1)

    def test_case_insensitive_prefix_match(self):
        batch = [_email("A <a@x.com>", "RE: something")]
        stats = compute_day_email_stats(batch)
        self.assertEqual(stats["reply_count"], 1)


class TestBodyLength(unittest.TestCase):
    def test_average_computed_correctly(self):
        batch = [
            _email("A <a@x.com>", "x", body="a" * 100),
            _email("B <b@x.com>", "y", body="b" * 300),
        ]
        stats = compute_day_email_stats(batch)
        self.assertEqual(stats["avg_body_chars"], 200)

    def test_empty_batch_does_not_crash(self):
        stats = compute_day_email_stats([])
        self.assertEqual(stats["avg_body_chars"], 0)
        self.assertEqual(stats["unique_senders"], 0)


if __name__ == "__main__":
    unittest.main()
