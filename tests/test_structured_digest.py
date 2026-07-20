"""
Unit tests for digest/core/structured_digest.py — the deterministic layers
(date_urgency, annotate_items) and the citation-attachment step (with a fake
LLM, no network call, same _FakeJudgeLLM pattern test_citations.py uses).
build_structured_digest itself is LLM-dependent end to end and is instead
exercised live (see the plan's verification section), not unit tested here.

Run: python3 -m unittest test_structured_digest -v
"""

import unittest
from datetime import date

from digest.core.structured_digest import annotate_items, attach_citations, date_urgency


class TestDateUrgency(unittest.TestCase):
    TODAY = date(2026, 7, 20)

    def test_no_due_date_is_no_date(self):
        self.assertEqual(date_urgency(None, self.TODAY), "no_date")

    def test_unparseable_due_date_is_no_date(self):
        self.assertEqual(date_urgency("not-a-date", self.TODAY), "no_date")

    def test_past_due_date_is_overdue(self):
        self.assertEqual(date_urgency("2026-07-19", self.TODAY), "overdue")

    def test_today_is_due_today(self):
        self.assertEqual(date_urgency("2026-07-20", self.TODAY), "due_today")

    def test_within_seven_days_is_due_soon(self):
        self.assertEqual(date_urgency("2026-07-27", self.TODAY), "due_soon")

    def test_exactly_seven_days_is_still_due_soon(self):
        # Boundary — same 7-day threshold tasks_signals.compute_task_signals
        # uses for its own due_soon bucket.
        self.assertEqual(date_urgency("2026-07-27", self.TODAY), "due_soon")

    def test_eight_days_out_is_later(self):
        self.assertEqual(date_urgency("2026-07-28", self.TODAY), "later")


class TestAnnotateItems(unittest.TestCase):
    TODAY = date(2026, 7, 20)

    def test_adds_date_urgency_to_every_item(self):
        items = [{"id": "i1", "due_date": "2026-07-19", "priority": "P1"}]
        annotated = annotate_items(items, self.TODAY)
        self.assertEqual(annotated[0]["date_urgency"], "overdue")

    def test_flags_priority_disagreement_when_low_priority_but_overdue(self):
        items = [{"id": "i1", "due_date": "2026-07-19", "priority": "P4"}]
        annotated = annotate_items(items, self.TODAY)
        self.assertTrue(annotated[0]["priority_disagreement"])

    def test_no_disagreement_flag_when_priority_matches_urgency(self):
        items = [{"id": "i1", "due_date": "2026-07-19", "priority": "P0"}]
        annotated = annotate_items(items, self.TODAY)
        self.assertNotIn("priority_disagreement", annotated[0])

    def test_no_disagreement_flag_when_not_urgent(self):
        # Low priority is fine when the item isn't overdue/due today.
        items = [{"id": "i1", "due_date": "2026-08-01", "priority": "P4"}]
        annotated = annotate_items(items, self.TODAY)
        self.assertNotIn("priority_disagreement", annotated[0])

    def test_due_today_with_low_priority_also_flags(self):
        items = [{"id": "i1", "due_date": "2026-07-20", "priority": "P3"}]
        annotated = annotate_items(items, self.TODAY)
        self.assertTrue(annotated[0]["priority_disagreement"])

    def test_does_not_mutate_original_items(self):
        items = [{"id": "i1", "due_date": "2026-07-19", "priority": "P4"}]
        annotate_items(items, self.TODAY)
        self.assertNotIn("date_urgency", items[0])


class _FakeJudgeLLM:
    def __init__(self, response):
        self.response = response

    def chat_json(self, messages):
        return self.response


class TestAttachCitations(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source": "email", "ref": "0003.eml", "label": "x", "text": "Press #3 vibration readings are trending upward this week"},
            {"source": "notes", "ref": "log.md", "label": "y", "text": "Tom reported Press #2 jam cleared in standup"},
        ]

    def test_keyword_match_attaches_source_refs(self):
        items = [{"id": "i1", "summary": "Press #3 vibration readings are trending upward"}]
        annotated = attach_citations(items, self.sources)
        self.assertEqual(annotated[0]["source_refs"], ["email:0003.eml"])

    def test_no_match_gives_empty_source_refs(self):
        items = [{"id": "i1", "summary": "Something entirely unrelated about lunch plans"}]
        annotated = attach_citations(items, self.sources)
        self.assertEqual(annotated[0]["source_refs"], [])

    def test_llm_fallback_used_for_unmatched_item(self):
        items = [{"id": "i1", "summary": "Tom reported Press #2 jam cleared in standup"}]
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "log.md", "quote": "Tom reported Press #2 jam cleared in standup"},
        ]}]})
        annotated = attach_citations(items, self.sources, llm=llm)
        self.assertEqual(annotated[0]["source_refs"], ["notes:log.md"])

    def test_does_not_mutate_original_items(self):
        items = [{"id": "i1", "summary": "Press #3 vibration readings are trending upward"}]
        attach_citations(items, self.sources)
        self.assertNotIn("source_refs", items[0])


if __name__ == "__main__":
    unittest.main()
