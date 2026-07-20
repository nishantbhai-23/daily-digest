"""
Unit tests for cross_reference.build_cross_reference_index — the
deterministic Stage 0 matching engine. No LLM involved; pure function
in/out checks against synthetic fixtures, plus an integration check
against the real generated dataset's known cross-source patterns (Elena
Marsh, Diane's board update) found by hand during development.

Run: python3 -m unittest test_cross_reference -v
"""

import unittest

from digest.core.cross_reference import build_cross_reference_index, title_keywords


def _task_signals(**buckets):
    base = {"overdue": [], "due_soon": [], "blocked": [], "stalled": []}
    base.update(buckets)
    return base


class TestTitleKeywords(unittest.TestCase):
    def test_filters_stopwords_and_short_words(self):
        keywords = title_keywords("Decide on Elena Marsh offer (Senior Engineer)")
        self.assertIn("Elena", keywords)
        self.assertIn("Marsh", keywords)
        self.assertIn("Senior", keywords)
        self.assertIn("Engineer", keywords)
        self.assertNotIn("on", keywords)

    def test_empty_title_returns_no_keywords(self):
        self.assertEqual(title_keywords("..."), [])

    def test_short_proper_noun_kept_short_common_word_dropped(self):
        # min_length dropped to 3 specifically so short real names like
        # "Sam" (the persona's actual P0 contact) survive — the extended
        # _STOPWORDS list is what keeps that from also letting through
        # noise words of the same length.
        keywords = title_keywords("Fix Sam onboarding checklist")
        self.assertIn("Sam", keywords)
        self.assertNotIn("Fix", keywords)


class TestMatching(unittest.TestCase):
    def test_task_mentioned_in_email_is_indexed(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-219", "title": "Decide on Elena Marsh offer", "priority": "P1"},
        ])
        email_ledger = [
            {"day": "2026-06-22", "delta": {"action_items": [{"description": "Review Elena Marsh feedback"}]}},
        ]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertIn("TESS-219", index)
        self.assertEqual(index["TESS-219"]["mentioned_in"][0]["source"], "email")
        self.assertEqual(index["TESS-219"]["mentioned_in"][0]["day"], "2026-06-22")

    def test_task_with_no_mentions_not_indexed(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-999", "title": "Renew office lease paperwork", "priority": "P3"},
        ])
        email_ledger = [{"day": "2026-06-22", "delta": {"action_items": [{"description": "Review Elena Marsh feedback"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_only_flagged_tasks_considered(self):
        # A task not present in any bucket shouldn't be indexed even if its
        # keywords would otherwise match — the function only receives
        # already-flagged tasks via task_signals.
        task_signals = _task_signals()  # nothing flagged
        email_ledger = [{"day": "2026-06-22", "delta": {"action_items": [{"description": "Elena Marsh"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_compacted_entries_skipped(self):
        task_signals = _task_signals(stalled=[
            {"id": "TESS-1", "title": "Multi-warehouse decision", "priority": "P1"},
        ])
        notes_ledger = [
            {"day": "2026-W25", "compacted": True, "delta": {"decisions": [{"description": "multi-warehouse"}]}},
        ]
        index = build_cross_reference_index([], [], notes_ledger, task_signals)
        self.assertEqual(index, {})

    def test_mentions_aggregated_across_sources(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-212", "title": "Send Diane the Q2 board update", "priority": "P2"},
        ])
        email_ledger = [{"day": "2026-06-16", "delta": {"deadlines": [{"description": "Diane wants board update"}]}}]
        notes_ledger = [{"day": "2026-07-01", "delta": {"decisions": [{"description": "Board update prep for Diane"}]}}]
        index = build_cross_reference_index(email_ledger, [], notes_ledger, task_signals)
        sources = {m["source"] for m in index["TESS-212"]["mentioned_in"]}
        self.assertEqual(sources, {"email", "notes"})

    def test_json_schema_keys_do_not_pollute_matches(self):
        # An earlier version built searchable text via json.dumps(entry),
        # which put dict keys like "action_items"/"description" into the
        # search corpus for free — a task title containing those same
        # words would then match 100% of entries regardless of content.
        # Only leaf string *values* should ever be searched.
        task_signals = _task_signals(overdue=[
            {"id": "TESS-1", "title": "Action Items Review", "priority": "P2"},
        ])
        email_ledger = [
            {"day": "2026-06-22", "delta": {"action_items": [{"description": "Renew the office lease paperwork"}]}},
        ]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_word_boundary_not_substring(self):
        # "Marsh" must not match inside "Marshall", and "Board" must not
        # match inside "Onboarding" — both real substring false-positive
        # shapes found during review.
        task_signals = _task_signals(overdue=[
            {"id": "TESS-2", "title": "Board Marsh Renewal", "priority": "P1"},
        ])
        email_ledger = [
            {"day": "2026-06-22", "delta": {"action_items": [{"description": "Onboarding for Marshall completed"}]}},
        ]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_excerpt_centered_on_match_not_prefix(self):
        # A fixed text[:500] misses the match entirely on any entry longer
        # than 500 chars — measured as the common case against real ledger
        # data (avg entry length ~2099 chars).
        task_signals = _task_signals(overdue=[
            {"id": "TESS-3", "title": "Halberd SLA Renewal", "priority": "P1"},
        ])
        padding = "Unrelated filler text discussing quarterly numbers and routine updates. " * 10
        description = padding + "Halberd asked about the SLA renewal terms again today."
        email_ledger = [{"day": "2026-06-22", "delta": {"action_items": [{"description": description}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        excerpt = index["TESS-3"]["mentioned_in"][0]["excerpt"]
        self.assertIn("Halberd", excerpt)
        self.assertIn("SLA", excerpt)

    def test_matched_fields_reports_item_category(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-4", "title": "Halberd SLA Renewal", "priority": "P1"},
        ])
        email_ledger = [{"day": "2026-06-22", "delta": {"deadlines": [{"description": "Halberd SLA renewal due"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index["TESS-4"]["mentioned_in"][0]["matched_fields"], ["deadlines"])

    def test_item_level_matching_rejects_cross_item_contamination(self):
        # The critical bug found while verifying this rewrite against real
        # data: two DIFFERENT, unrelated items landing in the same category
        # on the same day, each contributing one keyword, used to be
        # counted as a single combined "mention" once the whole category
        # was flattened into one blob. Neither item is actually about the
        # flagged task, so this must not match.
        task_signals = _task_signals(overdue=[
            {"id": "TESS-5", "title": "Close Halberd Manufacturing Integration", "priority": "P1"},
        ])
        email_ledger = [{
            "day": "2026-06-22",
            "delta": {
                "thread_progressions": [
                    {"description": "Close out the Q2 expense receipts"},
                    {"description": "Halberd asked about a different vendor contract"},
                ]
            },
        }]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index, {})

    def test_item_level_matching_accepts_genuine_same_item_match(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-6", "title": "Close Halberd Manufacturing Integration", "priority": "P1"},
        ])
        email_ledger = [{
            "day": "2026-06-22",
            "delta": {"thread_progressions": [{"description": "Close out Halberd Manufacturing integration timeline"}]},
        }]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertIn("TESS-6", index)

    def test_days_from_due_computed_with_correct_sign(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-7", "title": "Halberd SLA Renewal", "priority": "P1", "due_date": "2026-06-20"},
        ])
        email_ledger = [{"day": "2026-06-18", "delta": {"deadlines": [{"description": "Halberd SLA renewal reminder"}]}}]
        index = build_cross_reference_index(email_ledger, [], [], task_signals)
        self.assertEqual(index["TESS-7"]["mentioned_in"][0]["days_from_due"], -2)

    def test_days_from_due_none_when_day_unparseable(self):
        task_signals = _task_signals(overdue=[
            {"id": "TESS-8", "title": "Halberd SLA Renewal", "priority": "P1", "due_date": "2026-06-20"},
        ])
        notes_ledger = [{"day": "unknown", "delta": {"decisions": [{"description": "Halberd SLA renewal discussion"}]}}]
        index = build_cross_reference_index([], [], notes_ledger, task_signals)
        self.assertIsNone(index["TESS-8"]["mentioned_in"][0]["days_from_due"])


class TestAgainstRealData(unittest.TestCase):
    """Integration check against the actual generated dataset — the same
    Elena Marsh / Diane board-update patterns found by hand tonight should
    turn up automatically here.
    """

    def test_elena_marsh_or_diane_pattern_found_in_real_notes(self):
        # This doesn't require MAP-phase ledgers (which need an LLM) — it
        # exercises the matcher directly against notes ledger-shaped data
        # built from the real notes_parser output, so it's still a fast,
        # LLM-free check that the matching logic works on real content.
        from digest.parsers.notes_parser import load_notes

        notes = load_notes("data/notes/")
        fake_notes_ledger = [
            {"day": n["created_at"], "delta": {"decisions": [{"description": n["body"]}]}}
            for n in notes
        ]
        task_signals = _task_signals(overdue=[
            {"id": "TESS-212", "title": "Send Diane the Q2 board update", "priority": "P2"},
        ])
        index = build_cross_reference_index([], [], fake_notes_ledger, task_signals)
        self.assertIn("TESS-212", index)


if __name__ == "__main__":
    unittest.main()
