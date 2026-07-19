"""
Unit tests for ledger.py's apply_digest_window — the shared logic behind
triage_agent.py / calendar_agent.py's --digest-days and --holdout-days flags
(hybrid cold/hot digest ingestion).

Pure dict-slicing over an already-chronologically-sorted {day: batch} dict
(as produced by email_parser.group_by_date / calendar_parser.group_by_date)
— no LLM calls, no I/O.

Run: python3 -m unittest test_digest_windowing -v
"""

import unittest

from digest.core.ledger import apply_digest_window


def _days(n, prefix="2026-06"):
    # Small helper: build a sorted {day: batch} dict with n days,
    # "01".."0n", batch content irrelevant to this logic.
    return {f"{prefix}-{i + 1:02d}": [f"item-{i}"] for i in range(n)}


class TestDigestDaysCap(unittest.TestCase):
    def test_none_means_no_cap(self):
        batches = _days(30)
        windowed, held_out = apply_digest_window(batches, digest_days=None, holdout_days=None)
        self.assertEqual(windowed, batches)
        self.assertEqual(held_out, [])

    def test_caps_to_most_recent_n_days(self):
        batches = _days(30)
        windowed, _ = apply_digest_window(batches, digest_days=7, holdout_days=None)
        self.assertEqual(list(windowed.keys()), list(batches.keys())[-7:])

    def test_cap_larger_than_available_is_a_no_op(self):
        batches = _days(5)
        windowed, _ = apply_digest_window(batches, digest_days=30, holdout_days=None)
        self.assertEqual(windowed, batches)

    def test_cap_preserves_chronological_order(self):
        batches = _days(10)
        windowed, _ = apply_digest_window(batches, digest_days=3, holdout_days=None)
        self.assertEqual(list(windowed.keys()), sorted(windowed.keys()))


class TestHoldoutDays(unittest.TestCase):
    def test_holds_out_most_recent_n_days(self):
        batches = _days(30)
        windowed, held_out = apply_digest_window(batches, digest_days=None, holdout_days=1)
        self.assertEqual(len(windowed), 29)
        self.assertEqual(held_out, ["2026-06-30"])
        self.assertNotIn("2026-06-30", windowed)

    def test_holdout_days_zero_is_a_no_op(self):
        batches = _days(30)
        windowed, held_out = apply_digest_window(batches, digest_days=None, holdout_days=0)
        self.assertEqual(windowed, batches)
        self.assertEqual(held_out, [])

    def test_holdout_larger_than_available_holds_out_nothing(self):
        # Documented behavior: rather than emptying the whole window (which
        # would silently produce an empty cold run), holding out more days
        # than exist is a no-op — the caller is expected to warn on this.
        batches = _days(5)
        windowed, held_out = apply_digest_window(batches, digest_days=None, holdout_days=5)
        self.assertEqual(windowed, batches)
        self.assertEqual(held_out, [])

    def test_two_pass_reconstructs_full_dataset(self):
        # The core hybrid cold/hot simulation: a cold pass with holdout_days
        # set, followed by a second "hot" pass without it, should together
        # cover every day exactly once — this is what lets --holdout-days
        # emulate a hot run using only the existing synthetic dataset.
        batches = _days(30)
        cold, held_out = apply_digest_window(batches, digest_days=None, holdout_days=1)
        self.assertEqual(set(cold.keys()) | set(held_out), set(batches.keys()))
        self.assertEqual(set(cold.keys()) & set(held_out), set())


class TestDigestDaysAndHoldoutCombined(unittest.TestCase):
    def test_digest_days_bounds_before_holdout_carves_tail(self):
        # --digest-days 30 --holdout-days 1 on a 30-day dataset should
        # reproduce "cold(29) + hot(1)" exactly, per the two scenarios
        # described in the original request.
        batches = _days(30)
        windowed, held_out = apply_digest_window(batches, digest_days=30, holdout_days=1)
        self.assertEqual(len(windowed), 29)
        self.assertEqual(held_out, ["2026-06-30"])

    def test_holdout_applies_within_the_digest_days_window_not_outside_it(self):
        batches = _days(30)
        windowed, held_out = apply_digest_window(batches, digest_days=10, holdout_days=2)
        # digest_days=10 -> days 21-30; holdout_days=2 -> hold out 29, 30
        self.assertEqual(list(windowed.keys()), [f"2026-06-{i:02d}" for i in range(21, 29)])
        self.assertEqual(held_out, ["2026-06-29", "2026-06-30"])


if __name__ == "__main__":
    unittest.main()
