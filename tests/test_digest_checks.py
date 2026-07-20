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

from digest.eval.digest_checks import (
    check_conciseness,
    check_extraction_bloat,
    check_keywords_present,
    check_min_length,
    check_not_schema_description,
    dynamic_bloat_ceiling,
    extract_searchable_text,
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

    def test_or_set_passes_when_any_variant_present(self):
        # "Marcus Webb" (a variant) is in the text even though "marcus w."
        # and "mw" (the other variants) are not — the OR-set as a whole
        # should still count as satisfied.
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, [["marcus w.", "marcus webb", "mw"]])
        self.assertEqual(missing, [])

    def test_or_set_fails_and_reports_the_whole_set_when_no_variant_present(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, [["corrugator", "line 3", "grinding"]])
        self.assertEqual(missing, [["corrugator", "line 3", "grinding"]])

    def test_or_set_mixed_with_plain_keywords(self):
        # "diane" (plain) is present; the OR-set's only satisfied variant
        # is "webb" — both requirements clear, so nothing is missing.
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, ["diane", ["nonexistent", "webb"]])
        self.assertEqual(missing, [])

    def test_plain_keyword_still_reported_missing_alongside_a_satisfied_or_set(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, ["nonexistent-plain", ["marcus", "irrelevant"]])
        self.assertEqual(missing, ["nonexistent-plain"])

    def test_or_set_case_insensitive(self):
        missing = check_keywords_present(GOOD_DIGEST_EXCERPT, [["MARCUS WEBB", "irrelevant"]])
        self.assertEqual(missing, [])


class TestMinLength(unittest.TestCase):
    def test_short_text_fails(self):
        self.assertFalse(check_min_length("too short", min_words=50))

    def test_long_text_passes(self):
        self.assertTrue(check_min_length(GOOD_DIGEST_EXCERPT, min_words=20))


class TestExtractSearchableText(unittest.TestCase):
    def test_no_categories_searches_whole_container(self):
        delta = {"action_items": [{"description": "call Marcus about data room"}]}
        text = extract_searchable_text(delta)
        self.assertIn("Marcus", text)

    def test_single_category_scopes_search(self):
        delta = {
            "action_items": [{"description": "call Marcus"}],
            "decisions": [{"description": "renew Halberd contract"}],
        }
        text = extract_searchable_text(delta, "action_items")
        self.assertIn("Marcus", text)
        self.assertNotIn("Halberd", text)

    def test_list_of_categories_scopes_search_to_union(self):
        delta = {
            "action_items": [{"description": "call Marcus"}],
            "decisions": [{"description": "renew Halberd contract"}],
            "thread_progressions": [{"thread": "unrelated", "progression": "noise"}],
        }
        text = extract_searchable_text(delta, ["action_items", "decisions"])
        self.assertIn("Marcus", text)
        self.assertIn("Halberd", text)
        self.assertNotIn("noise", text)

    def test_missing_category_returns_empty_not_error(self):
        delta = {"action_items": [{"description": "call Marcus"}]}
        text = extract_searchable_text(delta, "decisions")
        self.assertEqual(text, "")

    def test_schema_key_names_do_not_pollute_search(self):
        # The exact bug class found and fixed in cross_reference.py — a
        # json.dumps() search would let "description" pass for free since
        # every MAP item has that field name as a key.
        delta = {"action_items": [{"description": "call Marcus about data room"}]}
        text = extract_searchable_text(delta)
        missing = check_keywords_present(text, ["description"])
        self.assertEqual(missing, ["description"])


class TestExtractionBloat(unittest.TestCase):
    def test_under_threshold_returns_none(self):
        delta = {"action_items": [{"description": "a"}, {"description": "b"}]}
        self.assertIsNone(check_extraction_bloat(delta, max_items=5))

    def test_at_threshold_returns_none(self):
        delta = {"action_items": [{"description": "a"}, {"description": "b"}]}
        self.assertIsNone(check_extraction_bloat(delta, max_items=2))

    def test_over_threshold_returns_warning(self):
        delta = {"action_items": [{"description": str(i)} for i in range(10)]}
        warning = check_extraction_bloat(delta, max_items=5)
        self.assertIsNotNone(warning)
        self.assertIn("10", warning)

    def test_sums_across_categories(self):
        delta = {
            "action_items": [{"description": str(i)} for i in range(3)],
            "decisions": [{"description": str(i)} for i in range(3)],
        }
        warning = check_extraction_bloat(delta, max_items=5)
        self.assertIsNotNone(warning)
        self.assertIn("6", warning)


class TestDynamicBloatCeiling(unittest.TestCase):
    def test_scales_with_input_count(self):
        self.assertEqual(dynamic_bloat_ceiling(10, floor=8, per_item_multiplier=4.0), 40)

    def test_floor_protects_small_batches(self):
        self.assertEqual(dynamic_bloat_ceiling(1, floor=8, per_item_multiplier=4.0), 8)

    def test_rounds_to_nearest_int(self):
        self.assertEqual(dynamic_bloat_ceiling(3, floor=4, per_item_multiplier=3.0), 9)

    def test_quiet_day_hallucination_flood_now_caught(self):
        # The exact gap a flat ceiling missed: a static EMAIL_MAX_ITEMS=40
        # (calibrated against a busy ~10-email day) let a 2-email day
        # produce 15 items without ever tripping the bloat check. A
        # dynamic ceiling scaled to the day's real batch size catches it.
        delta = {"action_items": [{"description": str(i)} for i in range(15)]}
        flat_ceiling = 40
        self.assertIsNone(check_extraction_bloat(delta, flat_ceiling))  # old behavior: misses it

        dynamic_ceiling = dynamic_bloat_ceiling(2, floor=8, per_item_multiplier=4.0)
        warning = check_extraction_bloat(delta, dynamic_ceiling)
        self.assertIsNotNone(warning)  # new behavior: catches it

    def test_busy_day_does_not_false_positive(self):
        # A genuinely busy day (12 emails) producing a proportionally
        # large but legitimate 30 items shouldn't trip the bound just
        # because 30 > some fixed small number.
        delta = {"action_items": [{"description": str(i)} for i in range(30)]}
        dynamic_ceiling = dynamic_bloat_ceiling(12, floor=8, per_item_multiplier=4.0)
        self.assertIsNone(check_extraction_bloat(delta, dynamic_ceiling))


class TestConciseness(unittest.TestCase):
    def test_summary_within_ratio_passes(self):
        source = " ".join(["word"] * 100)
        output = " ".join(["word"] * 50)  # ratio 0.5, under default 0.6
        self.assertIsNone(check_conciseness(output, source))

    def test_summary_exceeding_ratio_is_flagged(self):
        source = " ".join(["word"] * 100)
        output = " ".join(["word"] * 80)  # ratio 0.8, over default 0.6
        warning = check_conciseness(output, source)
        self.assertIsNotNone(warning)
        self.assertIn("0.80", warning)

    def test_empty_source_does_not_divide_by_zero(self):
        self.assertIsNone(check_conciseness("some output", ""))

    def test_custom_max_ratio_is_respected(self):
        source = " ".join(["word"] * 100)
        output = " ".join(["word"] * 40)  # ratio 0.4
        self.assertIsNone(check_conciseness(output, source, max_ratio=0.5))
        self.assertIsNotNone(check_conciseness(output, source, max_ratio=0.3))


if __name__ == "__main__":
    unittest.main()
