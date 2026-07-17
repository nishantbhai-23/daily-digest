"""
Unit tests for orchestrator's never-draft-contact matching — the
generalization of the old hardcoded "sam park" string check into a
tenant_config-driven, name-or-email match (build_never_draft_patterns,
mentions_blocked_contact), plus generate_drafts' short-circuit behavior when
everything is filtered.

Run: python3 -m unittest test_orchestrator_draft_filter -v
"""

import unittest

from orchestrator import build_never_draft_patterns, generate_drafts, mentions_blocked_contact


class PoisonLLM:
    """Raises if called — proves generate_drafts short-circuits instead of
    reaching the LLM when nothing survives the never-draft-contacts filter.
    Tracks call_count explicitly since call_with_retry catches and retries
    on any Exception, so relying on the raised error alone can't distinguish
    "never called" from "called, then retry-exhausted" — both can produce
    the same final {"drafts": []} result.
    """

    def __init__(self):
        self.call_count = 0

    def chat_json(self, messages):
        self.call_count += 1
        raise AssertionError("LLM should not have been called — everything should be filtered")


SAM_CONTACT = [{"name": "Sam Park", "email": "sam.park@gmail.com"}]


class TestMatchingLogic(unittest.TestCase):
    """Direct tests of the matching functions — instant, no LLM/retry path
    involved at all.
    """

    def test_matches_full_name(self):
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertTrue(mentions_blocked_contact({"summary": "Reply to Sam Park about groceries"}, patterns))

    def test_matches_first_name_only(self):
        # The real bug this guards against: an earlier version only matched
        # the full name as one substring, so prose saying just "Sam" (very
        # plausible for an LLM to write) silently slipped through unfiltered.
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertTrue(mentions_blocked_contact({"summary": "quick note back to Sam", "context": "personal"}, patterns))

    def test_matches_email(self):
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertTrue(mentions_blocked_contact({"context": "re: sam.park@gmail.com"}, patterns))

    def test_case_insensitive(self):
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertTrue(mentions_blocked_contact({"summary": "reply to SAM PARK"}, patterns))

    def test_word_boundary_prevents_substring_false_positive(self):
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertFalse(mentions_blocked_contact({"summary": "Samsung monitor arrived"}, patterns))

    def test_unrelated_item_not_matched(self):
        patterns = build_never_draft_patterns(SAM_CONTACT)
        self.assertFalse(mentions_blocked_contact({"summary": "Reply to Derek about seats"}, patterns))

    def test_no_configured_contacts_matches_nothing(self):
        patterns = build_never_draft_patterns([])
        self.assertFalse(mentions_blocked_contact({"summary": "Reply to Sam Park"}, patterns))


class TestGenerateDraftsShortCircuit(unittest.TestCase):
    def test_all_items_filtered_short_circuits_before_llm_call(self):
        items = [{"id": "d1", "type": "email_reply", "summary": "Reply to Sam Park about groceries", "context": ""}]
        llm = PoisonLLM()
        result = generate_drafts(llm, items, "persona text", {"never_draft_contacts": SAM_CONTACT})
        self.assertEqual(result, {"drafts": []})
        self.assertEqual(llm.call_count, 0)

    def test_no_never_draft_contacts_configured_does_not_filter(self):
        class StubLLM:
            def chat_json(self, messages):
                return {"drafts": [{"id": "d1", "draft_text": "ok"}]}

        items = [{"id": "d1", "type": "email_reply", "summary": "Reply to Derek about seats", "context": ""}]
        result = generate_drafts(StubLLM(), items, "persona text", {"never_draft_contacts": []})
        self.assertEqual(len(result["drafts"]), 1)


if __name__ == "__main__":
    unittest.main()
