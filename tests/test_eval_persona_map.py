"""
Unit tests for eval_persona_map.py's persistence helpers —
_record_coverage and _record_quality, both pure orchestration over an
already-computed results dict, with record_eval_run mocked (no real disk
writes or model calls). Mirrors tests/test_eval_map_quality_judge.py's
mocking approach for the same reason: these functions' job is shaping
scenario_results correctly per variant, not producing scores themselves.

Run: python3 -m unittest tests.test_eval_persona_map -v
"""

import unittest
from unittest.mock import patch

from digest.eval.eval_persona_map import _record_coverage, _record_quality

_GROUNDED = {
    "groundedness": {"score": 1.0, "unverified_claims": []},
    "completeness": {"gaps": [], "contradicted_gaps": []},
    "conciseness": {"ratio": 0.5, "verdict": "ok"},
    "coherence_tone": {"score": 4, "notes": "fine"},
}
_UNGROUNDED = {
    "groundedness": {"score": 0.5, "unverified_claims": ["fabricated detail"]},
    "completeness": {"gaps": [], "contradicted_gaps": []},
    "conciseness": {"ratio": 0.5, "verdict": "ok"},
    "coherence_tone": {"score": 4, "notes": "fine"},
}


class TestRecordCoverage(unittest.TestCase):
    @patch("digest.eval.eval_persona_map.record_eval_run")
    def test_pass_and_fail_shaped_correctly_per_variant(self, mock_record):
        mock_record.side_effect = lambda **kwargs: {"aggregate_accuracy": 1.0 if kwargs["variant"] == "persona" else 0.5}
        results = {
            "persona": {"email": [{"name": "scenario_a", "passed": True, "detail": "OK", "quality": None}]},
            "no_persona": {
                "email": [
                    {"name": "scenario_a", "passed": False, "detail": "missing: ['marcus']", "quality": None},
                ]
            },
        }
        records = _record_coverage(results, "deepseek", "deepseek-chat")

        self.assertEqual(mock_record.call_count, 2)
        persona_kwargs = next(c.kwargs for c in mock_record.call_args_list if c.kwargs["variant"] == "persona")
        no_persona_kwargs = next(c.kwargs for c in mock_record.call_args_list if c.kwargs["variant"] == "no_persona")

        self.assertEqual(persona_kwargs["eval_name"], "map_persona_ablation")
        self.assertEqual(
            persona_kwargs["scenario_results"],
            {"scenario_a": {"expected": "OK", "results": ["OK"]}},
        )
        self.assertEqual(
            no_persona_kwargs["scenario_results"],
            {"scenario_a": {"expected": "OK", "results": ["missing: ['marcus']"]}},
        )
        self.assertIn("persona", records)
        self.assertIn("no_persona", records)

    @patch("digest.eval.eval_persona_map.record_eval_run")
    def test_multiple_sources_merged_into_one_record_per_variant(self, mock_record):
        mock_record.return_value = {"aggregate_accuracy": 1.0}
        results = {
            "persona": {
                "email": [{"name": "e1", "passed": True, "detail": "OK", "quality": None}],
                "calendar": [{"name": "c1", "passed": True, "detail": "OK", "quality": None}],
            },
            "no_persona": {
                "email": [{"name": "e1", "passed": True, "detail": "OK", "quality": None}],
                "calendar": [{"name": "c1", "passed": True, "detail": "OK", "quality": None}],
            },
        }
        _record_coverage(results, "deepseek", "deepseek-chat")

        persona_kwargs = next(c.kwargs for c in mock_record.call_args_list if c.kwargs["variant"] == "persona")
        self.assertEqual(set(persona_kwargs["scenario_results"].keys()), {"e1", "c1"})


class TestRecordQuality(unittest.TestCase):
    @patch("digest.eval.eval_persona_map.record_eval_run")
    def test_returns_none_when_no_quality_scores_present(self, mock_record):
        results = {
            "persona": {"email": [{"name": "e1", "passed": True, "detail": "OK", "quality": None}]},
            "no_persona": {"email": [{"name": "e1", "passed": True, "detail": "OK", "quality": None}]},
        }
        records = _record_quality(results, "deepseek", "deepseek-chat")
        self.assertIsNone(records)
        mock_record.assert_not_called()

    @patch("digest.eval.eval_persona_map.record_eval_run")
    def test_discretizes_and_tags_variant_in_provider_model_string(self, mock_record):
        mock_record.return_value = {"aggregate_accuracy": 1.0}
        results = {
            "persona": {"email": [{"name": "e1", "passed": True, "detail": "OK", "quality": _GROUNDED}]},
            "no_persona": {"email": [{"name": "e1", "passed": True, "detail": "OK", "quality": _UNGROUNDED}]},
        }
        records = _record_quality(results, "deepseek", "deepseek-chat")

        self.assertIsNotNone(records)
        persona_kwargs = next(c.kwargs for c in mock_record.call_args_list if c.kwargs["variant"] == "persona/deepseek/deepseek-chat")
        no_persona_kwargs = next(c.kwargs for c in mock_record.call_args_list if c.kwargs["variant"] == "no_persona/deepseek/deepseek-chat")

        self.assertEqual(persona_kwargs["eval_name"], "map_persona_quality_judge")
        self.assertEqual(
            persona_kwargs["scenario_results"],
            {"e1": {"expected": "fully_grounded", "results": ["fully_grounded"]}},
        )
        self.assertEqual(
            no_persona_kwargs["scenario_results"],
            {"e1": {"expected": "fully_grounded", "results": ["has_unverified_claims"]}},
        )


if __name__ == "__main__":
    unittest.main()
