"""
Unit tests for orchestrator.py's check_priority_coverage — the deterministic
post-check applied after either Stage-2 variant (single_call or staged) to
catch a known-important (P0/P1) task silently dropped from the synthesized
brief text. Same title_keywords/2-keyword-match threshold cross_reference.py
already uses to decide "is this genuinely mentioned," reused here rather
than reinvented.

No network calls — tests the deterministic post-check directly against
synthetic synthesis/task_signals data.

Run: python3 -m unittest test_priority_coverage_check -v
"""

import unittest

from digest.orchestrator import check_priority_coverage


def _task_signals(**buckets):
    base = {"overdue": [], "due_soon": [], "blocked": [], "stalled": []}
    base.update(buckets)
    return base


class TestCheckPriorityCoverage(unittest.TestCase):
    def test_covered_task_not_flagged(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-219", "title": "Decide on Elena Marsh offer", "priority": "P0"},
        ])
        synthesis = {
            "what_matters_today": "You need to decide on the Elena Marsh offer today.",
            "what_might_be_missed": "",
            "dispatchable_items": [],
        }
        self.assertEqual(check_priority_coverage(synthesis, task_signals, embed_fn=None), [])

    def test_task_mentioned_nowhere_is_flagged(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-219", "title": "Decide on Elena Marsh offer", "priority": "P0"},
        ])
        synthesis = {
            "what_matters_today": "Nothing relevant here.",
            "what_might_be_missed": "Also nothing relevant.",
            "dispatchable_items": [],
        }
        missing = check_priority_coverage(synthesis, task_signals, embed_fn=None)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["task_id"], "TESS-219")
        self.assertEqual(missing[0]["priority"], "P0")

    def test_below_two_keyword_threshold_is_flagged(self):
        # Only one of two keywords ("Elena") present — same 2-keyword
        # threshold cross_reference.py uses, so a single coincidental word
        # match isn't treated as real coverage.
        task_signals = _task_signals(overdue=[
            {"id": "TESS-219", "title": "Decide on Elena Marsh offer", "priority": "P1"},
        ])
        synthesis = {
            "what_matters_today": "Elena joined the call today about something unrelated.",
            "what_might_be_missed": "",
            "dispatchable_items": [],
        }
        missing = check_priority_coverage(synthesis, task_signals, embed_fn=None)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["task_id"], "TESS-219")

    def test_low_priority_task_never_checked(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-999", "title": "Renew office lease paperwork", "priority": "P3"},
        ])
        synthesis = {
            "what_matters_today": "Nothing about the lease here.",
            "what_might_be_missed": "",
            "dispatchable_items": [],
        }
        self.assertEqual(check_priority_coverage(synthesis, task_signals, embed_fn=None), [])

    def test_coverage_via_dispatchable_item_context_counts(self):
        task_signals = _task_signals(due_soon=[
            {"id": "TESS-050", "title": "Renew Halberd contract", "priority": "P1"},
        ])
        synthesis = {
            "what_matters_today": "",
            "what_might_be_missed": "",
            "dispatchable_items": [
                {"id": "d1", "type": "email_reply", "summary": "Quick reply", "context": "Renew Halberd contract paperwork"},
            ],
        }
        self.assertEqual(check_priority_coverage(synthesis, task_signals, embed_fn=None), [])

    def test_multiple_buckets_all_checked(self):
        task_signals = _task_signals(
            overdue=[{"id": "TESS-1", "title": "Decide Elena Marsh offer", "priority": "P0"}],
            blocked=[{"id": "TESS-2", "title": "Renew Halberd contract", "priority": "P1"}],
        )
        synthesis = {
            "what_matters_today": "Decide Elena Marsh offer today.",
            "what_might_be_missed": "",
            "dispatchable_items": [],
        }
        missing = check_priority_coverage(synthesis, task_signals, embed_fn=None)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["task_id"], "TESS-2")

    def test_task_flagged_in_two_buckets_counted_once_not_twice(self):
        # Real bug found while investigating the eval_synthesis_variants.py
        # comparison run: task_signals legitimately flags the same task in
        # more than one bucket at once (e.g. both "overdue" and "stalled").
        # A missing task must only ever appear once in the result, not once
        # per bucket it happens to be flagged in.
        same_task = {"id": "TESS-219", "title": "Decide on Elena Marsh offer (Senior Engineer)", "priority": "P1"}
        task_signals = _task_signals(overdue=[same_task], stalled=[same_task])
        synthesis = {"what_matters_today": "Nothing relevant.", "what_might_be_missed": "", "dispatchable_items": []}
        missing = check_priority_coverage(synthesis, task_signals, embed_fn=None)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["task_id"], "TESS-219")

    def test_empty_synthesis_flags_all_flagged_tasks(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-1", "title": "Decide Elena Marsh offer", "priority": "P0"},
        ])
        synthesis = {"what_matters_today": "", "what_might_be_missed": "", "dispatchable_items": []}
        missing = check_priority_coverage(synthesis, task_signals, embed_fn=None)
        self.assertEqual(len(missing), 1)

    def test_no_flagged_tasks_returns_empty(self):
        self.assertEqual(check_priority_coverage({"what_matters_today": "", "what_might_be_missed": "", "dispatchable_items": []}, _task_signals(), embed_fn=None), [])


# 2D unit vectors with a known cosine similarity, same fixture-engineering
# approach as tests/test_citations.py — lets these tests pin an exact
# above/below-threshold (0.70) result without depending on a real
# embedding model's actual output.
_HIGH_SIM_A = [1.0, 0.0]
_HIGH_SIM_B = [0.9, 0.43589]   # cosine similarity with _HIGH_SIM_A ~= 0.9
_LOW_SIM = [0.3, 0.95394]      # cosine similarity with _HIGH_SIM_A ~= 0.3


def _dispatch_embed_fn(mapping, default=_LOW_SIM):
    def embed_fn(texts):
        return [mapping.get(t, default) for t in texts]
    return embed_fn


class TestCheckPriorityCoverageEmbeddingRescue(unittest.TestCase):
    def setUp(self):
        self.task_title = "Reach out to Andrew about the hardware slip"
        self.matching_chunk = "Called Drew regarding the delayed shipment."
        self.task_signals = _task_signals(overdue=[
            {"id": "TESS-1", "title": self.task_title, "priority": "P0"},
        ])

    def test_paraphrase_covered_task_is_rescued(self):
        # Zero keyword overlap between the task title and the chunk — the
        # keyword tier alone would flag this missing.
        synthesis = {"what_matters_today": self.matching_chunk, "what_might_be_missed": "", "dispatchable_items": []}
        embed_fn = _dispatch_embed_fn({self.task_title: _HIGH_SIM_A, self.matching_chunk: _HIGH_SIM_B})
        self.assertEqual(check_priority_coverage(synthesis, self.task_signals, embed_fn=embed_fn), [])

    def test_genuinely_uncovered_task_stays_flagged(self):
        synthesis = {"what_matters_today": self.matching_chunk, "what_might_be_missed": "", "dispatchable_items": []}
        embed_fn = _dispatch_embed_fn({self.task_title: _HIGH_SIM_A, self.matching_chunk: _LOW_SIM})
        missing = check_priority_coverage(synthesis, self.task_signals, embed_fn=embed_fn)
        self.assertEqual([m["task_id"] for m in missing], ["TESS-1"])

    def test_best_of_multiple_chunks_rescues_even_if_narrative_misses(self):
        # what_matters_today doesn't match, but a dispatchable item's
        # context does — the rescue must take the *best* similarity across
        # every chunk, not just the first one.
        synthesis = {
            "what_matters_today": "Something else entirely, unrelated.",
            "what_might_be_missed": "",
            "dispatchable_items": [{"id": "d1", "type": "email_reply", "summary": "reply", "context": self.matching_chunk}],
        }
        embed_fn = _dispatch_embed_fn({
            self.task_title: _HIGH_SIM_A,
            "Something else entirely, unrelated.": _LOW_SIM,
            f"reply {self.matching_chunk}": _HIGH_SIM_B,
        })
        self.assertEqual(check_priority_coverage(synthesis, self.task_signals, embed_fn=embed_fn), [])

    def test_embed_fn_none_disables_rescue_falls_back_to_keyword_only(self):
        synthesis = {"what_matters_today": self.matching_chunk, "what_might_be_missed": "", "dispatchable_items": []}
        missing = check_priority_coverage(synthesis, self.task_signals, embed_fn=None)
        self.assertEqual([m["task_id"] for m in missing], ["TESS-1"])

    def test_embedding_failure_degrades_gracefully_to_keyword_only(self):
        def broken_embed_fn(texts):
            raise RuntimeError("Ollama server not running")

        synthesis = {"what_matters_today": self.matching_chunk, "what_might_be_missed": "", "dispatchable_items": []}
        missing = check_priority_coverage(synthesis, self.task_signals, embed_fn=broken_embed_fn)
        self.assertEqual([m["task_id"] for m in missing], ["TESS-1"])

    def test_keyword_covered_task_never_needs_the_rescue(self):
        # A task the keyword tier already covers shouldn't even reach the
        # embedding pass — confirmed indirectly by using an embed_fn that
        # would raise if called with unexpected input.
        def embed_fn(texts):
            raise AssertionError(f"embed_fn should not have been called: {texts}")

        task_signals = _task_signals(overdue=[
            {"id": "TESS-1", "title": "Decide Elena Marsh offer", "priority": "P0"},
        ])
        synthesis = {"what_matters_today": "Decide Elena Marsh offer today.", "what_might_be_missed": "", "dispatchable_items": []}
        self.assertEqual(check_priority_coverage(synthesis, task_signals, embed_fn=embed_fn), [])


if __name__ == "__main__":
    unittest.main()
