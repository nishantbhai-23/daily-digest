"""
Unit tests for digest.eval.eval_history — the persistent prompt-registry +
results log built for the few-shot evaluation harness (Track B). Verifies
the accuracy computation, content-addressed snapshot dedup, and read-back
filtering, all against a temp directory so the real eval_history/ at the
repo root is never touched by tests.

Run: python3 -m unittest test_eval_history -v
"""

import os
import tempfile
import unittest

from digest.eval.eval_history import load_eval_history, record_eval_run


class TestRecordEvalRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results_file = os.path.join(self.tmp.name, "results.jsonl")
        self.prompts_dir = os.path.join(self.tmp.name, "prompts")
        self.addCleanup(self.tmp.cleanup)

    def _record(self, **overrides):
        kwargs = dict(
            eval_name="map_priority_calibration",
            variant="zero_shot",
            prompt_text="You are an email triage node...",
            provider="deepseek",
            model="deepseek-chat",
            trials_per_scenario=5,
            scenario_results={
                "quiet_marcus_investor_thread": {"expected": "P0", "results": ["P0", "P0", "P1", "P0", "P0"]},
                "recruiter_cold_pitch": {"expected": "P4", "results": ["P4", "P4", "P4", "P4", "P4"]},
            },
            results_file=self.results_file,
            prompts_dir=self.prompts_dir,
        )
        kwargs.update(overrides)
        return record_eval_run(**kwargs)

    def test_computes_per_scenario_accuracy(self):
        record = self._record()
        self.assertAlmostEqual(record["scenarios"]["quiet_marcus_investor_thread"]["accuracy"], 0.8)
        self.assertAlmostEqual(record["scenarios"]["recruiter_cold_pitch"]["accuracy"], 1.0)

    def test_computes_aggregate_accuracy(self):
        record = self._record()
        # 4/5 + 5/5 = 9 correct out of 10 total trials
        self.assertAlmostEqual(record["aggregate_accuracy"], 0.9)

    def test_writes_prompt_snapshot(self):
        record = self._record()
        self.assertTrue(os.path.exists(record["prompt_snapshot"]))
        with open(record["prompt_snapshot"]) as f:
            self.assertEqual(f.read(), "You are an email triage node...")

    def test_identical_prompt_text_reuses_same_snapshot(self):
        r1 = self._record()
        r2 = self._record()
        self.assertEqual(r1["prompt_snapshot"], r2["prompt_snapshot"])

    def test_different_prompt_text_gets_different_snapshot(self):
        r1 = self._record(prompt_text="prompt A")
        r2 = self._record(prompt_text="prompt B")
        self.assertNotEqual(r1["prompt_snapshot"], r2["prompt_snapshot"])

    def test_appends_not_overwrites(self):
        self._record(variant="zero_shot")
        self._record(variant="few_shot_v1")
        records = load_eval_history(results_file=self.results_file)
        self.assertEqual(len(records), 2)
        self.assertEqual({r["variant"] for r in records}, {"zero_shot", "few_shot_v1"})

    def test_empty_results_list_gives_zero_accuracy_not_error(self):
        record = self._record(scenario_results={"some_scenario": {"expected": "P0", "results": []}})
        self.assertEqual(record["scenarios"]["some_scenario"]["accuracy"], 0.0)
        self.assertEqual(record["aggregate_accuracy"], 0.0)


class TestLoadEvalHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results_file = os.path.join(self.tmp.name, "results.jsonl")
        self.prompts_dir = os.path.join(self.tmp.name, "prompts")
        self.addCleanup(self.tmp.cleanup)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_eval_history(results_file=self.results_file), [])

    def test_filters_by_eval_name(self):
        record_eval_run(
            eval_name="map_priority_calibration", variant="zero_shot", prompt_text="p1",
            provider="deepseek", model="deepseek-chat", trials_per_scenario=1,
            scenario_results={"s": {"expected": "P0", "results": ["P0"]}},
            results_file=self.results_file, prompts_dir=self.prompts_dir,
        )
        record_eval_run(
            eval_name="other_eval", variant="zero_shot", prompt_text="p2",
            provider="deepseek", model="deepseek-chat", trials_per_scenario=1,
            scenario_results={"s": {"expected": "P0", "results": ["P0"]}},
            results_file=self.results_file, prompts_dir=self.prompts_dir,
        )
        records = load_eval_history(eval_name="map_priority_calibration", results_file=self.results_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["eval_name"], "map_priority_calibration")

    def test_filters_by_variant(self):
        for variant in ("zero_shot", "few_shot_v1", "few_shot_v1"):
            record_eval_run(
                eval_name="map_priority_calibration", variant=variant, prompt_text=f"p-{variant}-{id(variant)}",
                provider="deepseek", model="deepseek-chat", trials_per_scenario=1,
                scenario_results={"s": {"expected": "P0", "results": ["P0"]}},
                results_file=self.results_file, prompts_dir=self.prompts_dir,
            )
        records = load_eval_history(variant="few_shot_v1", results_file=self.results_file)
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
