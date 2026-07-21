"""
Unit tests for digest/eval/quality_judge.py — groundedness quote
verification, completeness self-contradiction detection, and conciseness
delegation, all with an injected fake LLM (no network call), same
convention tests/test_citations.py already uses for judge-fallback tests.

Run: python3 -m unittest tests.test_quality_judge -v
"""

import unittest

from digest.core.llm import TerminalLLMError
from digest.eval.quality_judge import (
    _score_completeness,
    _score_conciseness,
    _score_groundedness,
    judge_map_quality,
)

_SOURCE = (
    "Samir confirmed the firmware rewrite will take 3 weeks and cost $150K. "
    "He also flagged that the vendor needs a signed PO by Friday to hold the "
    "current pricing."
)


class _FakeJudgeLLM:
    def __init__(self, response):
        self.response = response

    def chat_json(self, messages):
        return self.response


class _BrokenLLM:
    """Raises TerminalLLMError, not a plain exception, so call_with_retry
    fails fast without the real exponential-backoff sleep (2s/4s/8s) a
    plain retryable exception would trigger — keeps this test fast
    without needing production code to expose retry-tuning knobs it
    doesn't otherwise need.
    """

    def chat_json(self, messages):
        raise TerminalLLMError("model unavailable")


class TestScoreGroundedness(unittest.TestCase):
    def test_no_claims_scores_perfect(self):
        result = _score_groundedness([], _SOURCE)
        self.assertEqual(result, {"score": 1.0, "unverified_claims": []})

    def test_real_quote_is_verified(self):
        claims = [{"text": "firmware rewrite takes 3 weeks", "quote": "firmware rewrite will take 3 weeks"}]
        result = _score_groundedness(claims, _SOURCE)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["unverified_claims"], [])

    def test_fabricated_quote_is_flagged_and_named(self):
        claims = [{"text": "the deal closes next month", "quote": "the deal will close next month"}]
        result = _score_groundedness(claims, _SOURCE)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["unverified_claims"], ["the deal closes next month"])

    def test_empty_quote_is_flagged(self):
        claims = [{"text": "some claim", "quote": ""}]
        result = _score_groundedness(claims, _SOURCE)
        self.assertEqual(result["unverified_claims"], ["some claim"])

    def test_mixed_real_and_fabricated_scores_partial(self):
        claims = [
            {"text": "real one", "quote": "cost $150K"},
            {"text": "fake one", "quote": "cost $999K"},
        ]
        result = _score_groundedness(claims, _SOURCE)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["unverified_claims"], ["fake one"])

    def test_quote_matching_is_case_insensitive(self):
        claims = [{"text": "vendor PO deadline", "quote": "SIGNED PO BY FRIDAY"}]
        result = _score_groundedness(claims, _SOURCE)
        self.assertEqual(result["score"], 1.0)


class TestScoreCompleteness(unittest.TestCase):
    def test_genuine_gap_passes_through(self):
        output = "Samir confirmed the firmware rewrite will take 3 weeks."
        result = _score_completeness(["the vendor PO deadline on Friday"], output)
        self.assertEqual(result["gaps"], ["the vendor PO deadline on Friday"])
        self.assertEqual(result["contradicted_gaps"], [])

    def test_self_contradicting_claim_is_flagged_not_trusted(self):
        # The judge says the PO deadline is missing, but the output text
        # literally already contains those exact words.
        output = "Samir needs a signed PO by Friday to hold pricing."
        result = _score_completeness(["signed PO by Friday"], output)
        self.assertEqual(result["contradicted_gaps"], ["signed PO by Friday"])
        self.assertEqual(result["gaps"], [])

    def test_empty_missing_list_returns_empty(self):
        result = _score_completeness([], "some output")
        self.assertEqual(result, {"gaps": [], "contradicted_gaps": []})


class TestScoreConciseness(unittest.TestCase):
    def test_delegates_to_check_conciseness(self):
        source = " ".join(["word"] * 100)
        output = " ".join(["word"] * 40)
        result = _score_conciseness(source, output)
        self.assertEqual(result["ratio"], 0.4)
        self.assertEqual(result["verdict"], "ok")

    def test_flags_verbose_output(self):
        source = " ".join(["word"] * 100)
        output = " ".join(["word"] * 90)
        result = _score_conciseness(source, output)
        self.assertNotEqual(result["verdict"], "ok")


class TestJudgeMapQuality(unittest.TestCase):
    def test_end_to_end_with_fake_llm(self):
        llm = _FakeJudgeLLM({
            "claims": [
                {"text": "firmware rewrite takes 3 weeks", "quote": "firmware rewrite will take 3 weeks"},
                {"text": "fabricated detail", "quote": "this text is not in the source"},
            ],
            "missing": ["signed PO by Friday"],
            "coherence_tone": {"score": 4, "notes": "reads naturally"},
        })
        output_text = "Samir confirmed the firmware rewrite will take 3 weeks and cost $150K."

        result = judge_map_quality(llm, _SOURCE, output_text)

        self.assertEqual(result["groundedness"]["score"], 0.5)
        self.assertEqual(result["groundedness"]["unverified_claims"], ["fabricated detail"])
        self.assertEqual(result["completeness"]["gaps"], ["signed PO by Friday"])
        self.assertEqual(result["coherence_tone"], {"score": 4, "notes": "reads naturally"})
        self.assertIn("ratio", result["conciseness"])

    def test_llm_failure_degrades_gracefully_without_crashing(self):
        result = judge_map_quality(_BrokenLLM(), _SOURCE, "some output text")
        self.assertIsNone(result["groundedness"]["score"])
        self.assertIsNone(result["coherence_tone"]["score"])
        # Conciseness needs no LLM, so it's still computed even on failure.
        self.assertIn("ratio", result["conciseness"])


if __name__ == "__main__":
    unittest.main()
