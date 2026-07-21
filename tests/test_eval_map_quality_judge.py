"""
Unit tests for eval_map.py's quality-judge wiring —
_discretize_groundedness (pure) and _run_quality_judge's persistence
call (record_eval_run mocked, no real disk writes or model calls).

Run: python3 -m unittest tests.test_eval_map_quality_judge -v
"""

import unittest
from unittest.mock import patch

from digest.eval.eval_map import _discretize_groundedness, _run_quality_judge

_EMPTY_DIMENSIONS = {
    "completeness": {"gaps": [], "contradicted_gaps": []},
    "conciseness": {"ratio": 0.5, "verdict": "ok"},
    "coherence_tone": {"score": 4, "notes": "fine"},
}


class _FakeLLM:
    provider = "deepseek"
    model = "deepseek-chat"

    def chat_json(self, messages):
        return {}


class TestDiscretizeGroundedness(unittest.TestCase):
    def test_no_unverified_claims_is_fully_grounded(self):
        score = {"groundedness": {"score": 1.0, "unverified_claims": []}, **_EMPTY_DIMENSIONS}
        self.assertEqual(_discretize_groundedness(score), "fully_grounded")

    def test_any_unverified_claim_is_has_unverified_claims(self):
        score = {"groundedness": {"score": 0.5, "unverified_claims": ["fabricated detail"]}, **_EMPTY_DIMENSIONS}
        self.assertEqual(_discretize_groundedness(score), "has_unverified_claims")

    def test_partial_score_with_one_unverified_claim_still_flagged(self):
        # A 0.875 score (mostly grounded) must not be conflated with a
        # perfect 1.0 — any unverified claim at all is the flagged state.
        score = {"groundedness": {"score": 0.875, "unverified_claims": ["one bad claim"]}, **_EMPTY_DIMENSIONS}
        self.assertEqual(_discretize_groundedness(score), "has_unverified_claims")


class TestRunQualityJudgePersistence(unittest.TestCase):
    @patch("digest.eval.eval_map.record_eval_run")
    @patch("digest.eval.eval_map.judge_map_quality")
    def test_fully_grounded_result_is_recorded(self, mock_judge, mock_record):
        mock_judge.return_value = {
            "groundedness": {"score": 1.0, "unverified_claims": []},
            **_EMPTY_DIMENSIONS,
        }
        entry = {"delta": {"action_items": [{"description": "call Marcus"}]}}
        _run_quality_judge(_FakeLLM(), "quiet_marcus_investor_thread", [{"subject": "hi"}], entry)

        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        self.assertEqual(kwargs["eval_name"], "map_quality_judge")
        self.assertEqual(kwargs["variant"], "deepseek/deepseek-chat")
        self.assertEqual(kwargs["provider"], "deepseek")
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(
            kwargs["scenario_results"],
            {"quiet_marcus_investor_thread": {"expected": "fully_grounded", "results": ["fully_grounded"]}},
        )

    @patch("digest.eval.eval_map.record_eval_run")
    @patch("digest.eval.eval_map.judge_map_quality")
    def test_unverified_claim_result_is_recorded(self, mock_judge, mock_record):
        mock_judge.return_value = {
            "groundedness": {"score": 0.5, "unverified_claims": ["fabricated detail"]},
            **_EMPTY_DIMENSIONS,
        }
        entry = {"delta": {"action_items": [{"description": "call Marcus"}]}}
        _run_quality_judge(_FakeLLM(), "some_scenario", [{"subject": "hi"}], entry)

        _, kwargs = mock_record.call_args
        self.assertEqual(
            kwargs["scenario_results"],
            {"some_scenario": {"expected": "fully_grounded", "results": ["has_unverified_claims"]}},
        )


if __name__ == "__main__":
    unittest.main()
