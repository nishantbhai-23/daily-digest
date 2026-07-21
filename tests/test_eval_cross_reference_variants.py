"""
Unit tests for eval_cross_reference_variants.score_cross_reference_scenario
— a pure function, no model call, no ledger loading needed.

Run: python3 -m unittest tests.test_eval_cross_reference_variants -v
"""

import unittest

from digest.eval.eval_cross_reference_variants import (
    score_cross_reference_precision_scenario,
    score_cross_reference_scenario,
)


def _index(task_id, sources):
    return {task_id: {"title": "irrelevant", "priority": "P1", "mentioned_in": [{"source": s} for s in sources]}}


def _index_with_mentions(task_id, mentions):
    """mentions: list of (source, day) pairs."""
    return {
        task_id: {
            "title": "irrelevant", "priority": "P1",
            "mentioned_in": [{"source": s, "day": d} for s, d in mentions],
        }
    }


class TestScoreCrossReferenceScenario(unittest.TestCase):
    def test_expected_sources_all_present_passes(self):
        index = _index("ARC-102", ["email", "notes"])
        scenario = {"task_id": "ARC-102", "expected_sources": {"email", "notes"}}
        passed, detail = score_cross_reference_scenario(index, scenario)
        self.assertTrue(passed)
        self.assertEqual(detail, "OK")

    def test_superset_of_expected_still_passes(self):
        # A floor, not an exact match — extra mentions beyond what's
        # expected shouldn't fail the scenario.
        index = _index("ARC-102", ["email", "notes", "calendar"])
        scenario = {"task_id": "ARC-102", "expected_sources": {"email", "notes"}}
        passed, _ = score_cross_reference_scenario(index, scenario)
        self.assertTrue(passed)

    def test_missing_expected_source_fails(self):
        index = _index("ARC-102", ["email"])
        scenario = {"task_id": "ARC-102", "expected_sources": {"email", "notes"}}
        passed, detail = score_cross_reference_scenario(index, scenario)
        self.assertFalse(passed)
        self.assertIn("notes", detail)

    def test_task_not_in_index_at_all_fails(self):
        index = {}
        scenario = {"task_id": "ARC-102", "expected_sources": {"email"}}
        passed, detail = score_cross_reference_scenario(index, scenario)
        self.assertFalse(passed)
        self.assertIn("not found", detail)


class TestScoreCrossReferencePrecisionScenario(unittest.TestCase):
    def test_forbidden_mention_absent_passes(self):
        index = _index_with_mentions("ARC-102", [("email", "2026-07-08")])
        scenario = {"task_id": "ARC-102", "forbidden_mentions": [{"source": "email", "day": "2026-07-03"}]}
        passed, detail = score_cross_reference_precision_scenario(index, scenario)
        self.assertTrue(passed)
        self.assertEqual(detail, "OK")

    def test_forbidden_mention_present_fails(self):
        index = _index_with_mentions("ARC-102", [("email", "2026-07-03")])
        scenario = {"task_id": "ARC-102", "forbidden_mentions": [{"source": "email", "day": "2026-07-03"}]}
        passed, detail = score_cross_reference_precision_scenario(index, scenario)
        self.assertFalse(passed)
        self.assertIn("2026-07-03", detail)

    def test_same_day_different_source_does_not_false_positive(self):
        # The forbidden pair is (source, day) together — a real mention
        # on the same day but a different source shouldn't trip this.
        index = _index_with_mentions("ARC-102", [("calendar", "2026-07-03")])
        scenario = {"task_id": "ARC-102", "forbidden_mentions": [{"source": "email", "day": "2026-07-03"}]}
        passed, _ = score_cross_reference_precision_scenario(index, scenario)
        self.assertTrue(passed)

    def test_task_not_in_index_at_all_trivially_passes(self):
        # Nothing to forbid if there's no mentioned_in list at all — that's
        # score_cross_reference_scenario's (recall) concern, not this one's.
        index = {}
        scenario = {"task_id": "ARC-102", "forbidden_mentions": [{"source": "email", "day": "2026-07-03"}]}
        passed, _ = score_cross_reference_precision_scenario(index, scenario)
        self.assertTrue(passed)

    def test_multiple_forbidden_mentions_any_leak_fails(self):
        index = _index_with_mentions("ARC-102", [("email", "2026-07-08"), ("calendar", "2026-07-10")])
        scenario = {
            "task_id": "ARC-102",
            "forbidden_mentions": [
                {"source": "email", "day": "2026-07-03"},
                {"source": "calendar", "day": "2026-07-10"},
            ],
        }
        passed, detail = score_cross_reference_precision_scenario(index, scenario)
        self.assertFalse(passed)
        self.assertIn("calendar", detail)


if __name__ == "__main__":
    unittest.main()
