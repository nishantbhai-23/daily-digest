"""
Unit tests for eval_map.py's score_scenario — the pure, LLM-free scoring
logic separated out of eval_email_scenarios/eval_calendar_scenarios/
eval_notes_scenarios specifically so it's testable without a real model
call. Covers the core ① regression (a keyword landing in the wrong
category must fail when expected_category is set) and the ③
forbidden_keywords negative-case path.

No network calls — synthetic entry/scenario dicts only.

Run: python3 -m unittest test_eval_map_scoring -v
"""

import unittest

from digest.eval.eval_map import score_scenario


class TestScoreScenarioPositive(unittest.TestCase):
    def test_passes_when_keyword_present_unscoped(self):
        entry = {"delta": {"action_items": [{"description": "call Marcus about data room"}]}}
        scenario = {"required_keywords": ["marcus"]}
        passed, detail = score_scenario(entry, scenario)
        self.assertTrue(passed)
        self.assertEqual(detail, "OK")

    def test_fails_when_keyword_missing(self):
        entry = {"delta": {"action_items": [{"description": "nothing relevant"}]}}
        scenario = {"required_keywords": ["marcus"]}
        passed, detail = score_scenario(entry, scenario)
        self.assertFalse(passed)
        self.assertIn("marcus", detail)

    def test_passes_when_keyword_in_expected_category(self):
        entry = {
            "delta": {
                "action_items": [{"description": "call Marcus about data room"}],
                "decisions": [{"description": "unrelated"}],
            }
        }
        scenario = {"required_keywords": ["marcus"], "expected_category": "action_items"}
        passed, detail = score_scenario(entry, scenario)
        self.assertTrue(passed)

    def test_fails_when_keyword_in_wrong_category(self):
        # The core ① regression: "marcus" is present in the extraction, but
        # not in the category the scenario says it should be in — a
        # whole-delta search would incorrectly pass this.
        entry = {
            "delta": {
                "thread_progressions": [{"thread": "Marcus", "progression": "unrelated update"}],
                "action_items": [{"description": "something else entirely"}],
            }
        }
        scenario = {"required_keywords": ["marcus"], "expected_category": "action_items"}
        passed, detail = score_scenario(entry, scenario)
        self.assertFalse(passed)

    def test_list_expected_category_accepts_either(self):
        entry = {"delta": {"decisions": [{"description": "reassessing budget with Carla"}]}}
        scenario = {"required_keywords": ["carla", "budget"], "expected_category": ["thread_progressions", "decisions"]}
        passed, _ = score_scenario(entry, scenario)
        self.assertTrue(passed)

    def test_include_stats_searches_stats_too(self):
        entry = {"delta": {}, "stats": {"deep_work_conflicts": ["Halberd Quarterly Review"]}}
        scenario = {"required_keywords": ["halberd"]}
        passed, _ = score_scenario(entry, scenario, include_stats=True)
        self.assertTrue(passed)

    def test_include_stats_false_ignores_stats(self):
        entry = {"delta": {}, "stats": {"deep_work_conflicts": ["Halberd Quarterly Review"]}}
        scenario = {"required_keywords": ["halberd"]}
        passed, _ = score_scenario(entry, scenario, include_stats=False)
        self.assertFalse(passed)


class TestScoreScenarioNegative(unittest.TestCase):
    def test_passes_when_forbidden_keyword_absent(self):
        entry = {"delta": {"action_items": [{"description": "unrelated item"}]}}
        scenario = {"forbidden_keywords": ["trent", "apex"]}
        passed, detail = score_scenario(entry, scenario)
        self.assertTrue(passed)
        self.assertEqual(detail, "OK")

    def test_fails_when_forbidden_keyword_present(self):
        entry = {"delta": {"action_items": [{"description": "follow up with Trent"}]}}
        scenario = {"forbidden_keywords": ["trent", "apex"]}
        passed, detail = score_scenario(entry, scenario)
        self.assertFalse(passed)
        self.assertIn("trent", [k.lower() for k in ["Trent"]])  # sanity: case-insensitive match happened
        self.assertIn("forbidden keyword", detail)

    def test_forbidden_check_ignores_expected_category(self):
        # A false positive could land in any category, so forbidden checks
        # always search the whole delta regardless of expected_category.
        entry = {"delta": {"thread_progressions": [{"thread": "Trent", "progression": "noise"}]}}
        scenario = {"forbidden_keywords": ["trent"], "expected_category": "action_items"}
        passed, _ = score_scenario(entry, scenario)
        self.assertFalse(passed)

    def test_forbidden_check_includes_stats_when_requested(self):
        entry = {"delta": {}, "stats": {"some_field": ["mentions Trent here"]}}
        scenario = {"forbidden_keywords": ["trent"]}
        passed, _ = score_scenario(entry, scenario, include_stats=True)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
