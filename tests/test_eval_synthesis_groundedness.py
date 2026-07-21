"""
Unit tests for eval_synthesis_groundedness.groundedness_ratio — a pure
function given cite_brief's own stats dict, no model call, no file I/O.

Run: python3 -m unittest tests.test_eval_synthesis_groundedness -v
"""

import unittest

from digest.eval.eval_synthesis_groundedness import groundedness_ratio


class TestGroundednessRatio(unittest.TestCase):
    def test_all_bullets_cited_scores_one(self):
        stats = {"cited_keyword": 3, "cited_embedding": 1, "cited_llm": 1, "uncited": 0}
        self.assertEqual(groundedness_ratio(stats), 1.0)

    def test_all_bullets_uncited_scores_zero(self):
        stats = {"cited_keyword": 0, "cited_embedding": 0, "cited_llm": 0, "uncited": 5}
        self.assertEqual(groundedness_ratio(stats), 0.0)

    def test_partial_coverage_computed_correctly(self):
        # 3 of 4 cited -> 0.75 grounded.
        stats = {"cited_keyword": 2, "cited_embedding": 1, "cited_llm": 0, "uncited": 1}
        self.assertEqual(groundedness_ratio(stats), 0.75)

    def test_no_bullets_at_all_is_vacuously_fully_grounded(self):
        stats = {"cited_keyword": 0, "cited_embedding": 0, "cited_llm": 0, "uncited": 0}
        self.assertEqual(groundedness_ratio(stats), 1.0)


if __name__ == "__main__":
    unittest.main()
