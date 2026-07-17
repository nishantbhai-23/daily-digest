"""
Unit tests for digest_checks.py — verified against real digest output
captured during development, not synthetic examples.

The "bad" fixture is the actual local-llama3 output saved at
output/history/current_30day_summary_2026-07-16_225204.md (the
schema-describing failure this whole check exists to catch). The "good"
fixture is drawn from the real Claude Haiku 4.5 digest produced earlier in
the same project.

Run: python3 -m unittest test_digest_checks -v
"""

import os
import unittest

from digest_checks import (
    check_keywords_present,
    check_min_length,
    check_not_schema_description,
    looks_like_schema_description,
)

BAD_DIGEST_FIXTURE = "output/history/current_30day_summary_2026-07-16_225204.md"

GOOD_DIGEST_EXCERPT = """
## 1. WHAT NEEDS ME TODAY

**P0 — Series A diligence is live and moving fast.**
- **Marcus Webb** (Series A lead) wants to discuss multi-warehouse data model this week.
- **Ben Schaffer** needs intros to Derek (Northstar) and Lindsey (Veritas) for reference calls.
- **Diane Okafor** is asking for: updated cap table, SAFE conversion schedule, June financials.

## 2. WHAT AM I ABOUT TO DROP

**Quiet P0 threads — investor side.**
- Marcus Webb requested a call on multi-warehouse data model. No call scheduled yet.
"""


class TestSchemaDescriptionDetector(unittest.TestCase):
    def test_flags_real_bad_digest(self):
        if not os.path.exists(BAD_DIGEST_FIXTURE):
            self.skipTest(f"fixture not present: {BAD_DIGEST_FIXTURE}")
        with open(BAD_DIGEST_FIXTURE, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(looks_like_schema_description(text))
        with self.assertRaises(AssertionError):
            check_not_schema_description(text)

    def test_does_not_flag_real_good_digest(self):
        self.assertFalse(looks_like_schema_description(GOOD_DIGEST_EXCERPT))
        check_not_schema_description(GOOD_DIGEST_EXCERPT)  # should not raise

    def test_catches_each_known_red_flag_phrase(self):
        samples = [
            "This is a JSON file containing the profile data for Avery Chen.",
            "The file contains an array of objects, each representing a day.",
            "Here's a breakdown of the properties in each object.",
            "* `day`: The date. This is the top-level key structure.",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(looks_like_schema_description(sample))


class TestKeywordPresence(unittest.TestCase):
    def test_all_present_returns_empty(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, ["marcus", "diane"])
        self.assertEqual(missing, [])

    def test_missing_keyword_reported(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, ["marcus", "elena marsh"])
        self.assertEqual(missing, ["elena marsh"])

    def test_case_insensitive(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, ["MARCUS WEBB"])
        self.assertEqual(missing, [])


class TestMinLength(unittest.TestCase):
    def test_short_text_fails(self):
        self.assertFalse(check_min_length("too short", min_words=50))

    def test_long_text_passes(self):
        self.assertTrue(check_min_length(GOOD_DIGEST_EXCERPT, min_words=20))


if __name__ == "__main__":
    unittest.main()
