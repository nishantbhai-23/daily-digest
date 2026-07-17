"""
Unit tests for orchestrator.py's contradiction-grounding check — the fix
for docs/ERROR_HANDLING.md gap #4: schema validation checks that a Stage 1
contradiction is well-formed JSON, not that it's actually true. A live run
produced a well-formed but hallucinated contradiction about "Elena Marsh"
(see docs/SESSION_NOTES.md) — used here as the regression fixture.

No network calls — tests the deterministic post-filter directly against
synthetic cross-reference data.

Run: python3 -m unittest test_grounding_check -v
"""

import unittest

from orchestrator import _ground_contradictions, _resolve_entity

MULTI_SOURCE = {
    "TESS-219": {
        "title": "Decide on Elena Marsh offer (Senior Engineer)",
        "priority": "P1",
        "mentioned_in": [
            {"source": "email", "day": "2026-06-22", "matched_keywords": ["Elena", "Marsh"], "excerpt": "onsite went really well, move to offer?"},
            {"source": "notes", "day": "2026-07-03-hiring-pipeline", "matched_keywords": ["Elena", "Marsh"], "excerpt": "awaiting go/no-go from Avery"},
        ],
    },
    "TESS-050": {
        "title": "Renew Halberd contract",
        "priority": "P1",
        "mentioned_in": [
            {"source": "email", "day": "2026-07-01", "matched_keywords": ["Halberd", "renew"], "excerpt": "budget concern raised"},
        ],
    },
}


class TestResolveEntity(unittest.TestCase):
    def test_resolves_by_task_id(self):
        self.assertIs(_resolve_entity("TESS-219", MULTI_SOURCE), MULTI_SOURCE["TESS-219"])

    def test_resolves_by_title_case_insensitive(self):
        match = _resolve_entity("decide on elena marsh offer (senior engineer)", MULTI_SOURCE)
        self.assertIs(match, MULTI_SOURCE["TESS-219"])

    def test_unknown_entity_returns_none(self):
        self.assertIsNone(_resolve_entity("Some Unrelated Candidate", MULTI_SOURCE))


class TestGroundContradictions(unittest.TestCase):
    def test_keeps_contradiction_with_real_source_pairing(self):
        claims = [{"entity": "TESS-219", "sources": ["email", "notes"], "description": "email says X, notes says Y"}]
        self.assertEqual(_ground_contradictions(claims, MULTI_SOURCE), claims)

    def test_drops_contradiction_for_unknown_entity(self):
        # The Elena Marsh incident's failure mode: a claim referencing an
        # entity/story that was never one of the candidates given at all.
        claims = [{"entity": "Some Unrelated Candidate", "sources": ["email", "notes"], "description": "fabricated"}]
        self.assertEqual(_ground_contradictions(claims, MULTI_SOURCE), [])

    def test_drops_contradiction_claiming_a_source_never_presented(self):
        # TESS-050 was only ever mentioned in email — a claim asserting a
        # calendar/email conflict for it references a pairing that never
        # existed in what the model was shown.
        claims = [{"entity": "TESS-050", "sources": ["email", "calendar"], "description": "fabricated conflict"}]
        self.assertEqual(_ground_contradictions(claims, MULTI_SOURCE), [])

    def test_drops_contradiction_with_empty_sources(self):
        claims = [{"entity": "TESS-219", "sources": [], "description": "vague"}]
        self.assertEqual(_ground_contradictions(claims, MULTI_SOURCE), [])

    def test_partial_drop_keeps_the_grounded_one(self):
        good = {"entity": "TESS-219", "sources": ["email", "notes"], "description": "real"}
        bad = {"entity": "TESS-050", "sources": ["notes"], "description": "fabricated"}
        self.assertEqual(_ground_contradictions([good, bad], MULTI_SOURCE), [good])

    def test_empty_input_returns_empty(self):
        self.assertEqual(_ground_contradictions([], MULTI_SOURCE), [])


if __name__ == "__main__":
    unittest.main()
