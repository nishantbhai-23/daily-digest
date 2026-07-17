"""
Unit tests for triage_agent.filter_blocked_senders — the deterministic
pre-MAP noise filter. No LLM involved; pure function in/out checks.

Run: python3 -m unittest test_email_noise_filter -v
"""

import unittest

from triage_agent import filter_blocked_senders


def _email(from_header):
    return {"from": from_header, "subject": "x", "body": "y", "date_key": "2026-07-01"}


class TestBlockedSenders(unittest.TestCase):
    def test_exact_sender_filtered(self):
        emails = [
            _email("Stratechery <ben@stratechery.com>"),
            _email("Priya Iyer <priya@tessera.io>"),
        ]
        config = {"map_noise_filter": {"blocked_senders": ["ben@stratechery.com"], "blocked_domains": []}}
        result = filter_blocked_senders(emails, config)
        self.assertEqual(len(result), 1)
        self.assertIn("priya@tessera.io", result[0]["from"])

    def test_case_insensitive_matching(self):
        emails = [_email("Stratechery <BEN@STRATECHERY.COM>")]
        config = {"map_noise_filter": {"blocked_senders": ["ben@stratechery.com"], "blocked_domains": []}}
        result = filter_blocked_senders(emails, config)
        self.assertEqual(result, [])


class TestBlockedDomains(unittest.TestCase):
    def test_whole_domain_filtered(self):
        emails = [
            _email("Notion <team@makenotion.com>"),
            _email("Notion <updates@makenotion.com>"),
            _email("Priya Iyer <priya@tessera.io>"),
        ]
        config = {"map_noise_filter": {"blocked_senders": [], "blocked_domains": ["makenotion.com"]}}
        result = filter_blocked_senders(emails, config)
        self.assertEqual(len(result), 1)


class TestNoFilterConfigured(unittest.TestCase):
    def test_empty_config_returns_all_emails_unchanged(self):
        emails = [_email("A <a@x.com>"), _email("B <b@y.com>")]
        config = {"map_noise_filter": {"blocked_senders": [], "blocked_domains": []}}
        result = filter_blocked_senders(emails, config)
        self.assertEqual(result, emails)

    def test_missing_map_noise_filter_key_does_not_crash(self):
        emails = [_email("A <a@x.com>")]
        result = filter_blocked_senders(emails, {})
        self.assertEqual(result, emails)


if __name__ == "__main__":
    unittest.main()
