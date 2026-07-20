"""
Unit tests for eval_cross_reference_variants.score_cross_reference_scenario
— a pure function, no model call, no ledger loading needed.

Run: python3 -m unittest tests.test_eval_cross_reference_variants -v
"""

import unittest

from digest.eval.eval_cross_reference_variants import score_cross_reference_scenario


def _index(task_id, sources):
    return {task_id: {"title": "irrelevant", "priority": "P1", "mentioned_in": [{"source": s} for s in sources]}}


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


if __name__ == "__main__":
    unittest.main()
