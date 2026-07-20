"""
Unit tests for digest/core/citations.py — the post-processing per-bullet
citation pass. Covers the pure, LLM-free pieces (split_citable_lines,
keyword_match_sources, load_citable_sources, cite_brief with llm=None)
directly, and llm_match_sources' grounding logic with a fake LLM (no
network call) — the direct regression test for "never trust a claimed
source_ref that wasn't actually shown to the model," same shape
tests/test_grounding_check.py already established for Stage 1.

Run: python3 -m unittest test_citations -v
"""

import os
import tempfile
import unittest

from digest.core.citations import (
    cite_brief,
    corpus_common_keywords,
    keyword_match_sources,
    llm_match_sources,
    load_citable_sources,
    split_citable_lines,
)

_SAMPLE_EML = (
    "Subject: Delta Queen draft report\n"
    "From: Diego Reyes <diego@example.com>\n"
    "To: Catherine <catherine@example.com>\n"
    "Date: Sat, 18 Jul 2026 09:00:00 -0500\n"
    "Message-ID: <diego-1@example.com>\n"
    "\n"
    "Please review the Delta Queen draft report when you get a chance.\n"
)

_SAMPLE_ICS = (
    "BEGIN:VCALENDAR\n"
    "VERSION:2.0\n"
    "PRODID:-//Test//Test//EN\n"
    "BEGIN:VEVENT\n"
    "UID:test-event-1@example.com\n"
    "DTSTAMP:20260716T120000Z\n"
    "DTSTART:20260718T090000\n"
    "DTEND:20260718T093000\n"
    "SUMMARY:Phone call with Captain Stroud\n"
    "STATUS:CONFIRMED\n"
    "DESCRIPTION:Discuss the Ocean Pride grounding\n"
    "END:VEVENT\n"
    "END:VCALENDAR\n"
)

_SAMPLE_NOTE = "# Daily Log\n\nOcean Pride grounding update: preliminary report due Tuesday.\n"


class TestSplitCitableLines(unittest.TestCase):
    def test_excludes_headers_blockquotes_and_blank_lines(self):
        text = (
            "# Daily Brief\n\n"
            "> ⚠️ Data freshness notice\n"
            "> - email: stale\n\n"
            "## What Matters Today\n\n"
            "Review the draft report today.\n\n"
            "*(surfaced for you to handle directly — not drafted)*\n"
        )
        lines = split_citable_lines(text)
        self.assertEqual(lines, ["Review the draft report today."])

    def test_keeps_bold_summary_lines(self):
        text = "**Confirm availability for the inspection**\n> Confirmed.\n"
        lines = split_citable_lines(text)
        self.assertEqual(lines, ["**Confirm availability for the inspection**"])


class TestKeywordMatchSources(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source": "email", "ref": "0003.eml", "label": "Delta Queen", "text": "Diego sent the Delta Queen draft report for review"},
            {"source": "notes", "ref": "log.md", "label": "log", "text": "Ocean Pride grounding preliminary report due"},
        ]

    def test_matches_correct_source(self):
        matches = keyword_match_sources("Review Diego's Delta Queen draft report", self.sources)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["ref"], "0003.eml")

    def test_below_threshold_returns_empty(self):
        # Only "report" overlaps between both sources' text and this
        # bullet's other keywords don't appear anywhere — below the
        # min(2, len(keywords)) threshold for either source.
        matches = keyword_match_sources("Submit the quarterly report", self.sources)
        self.assertEqual(matches, [])

    def test_matches_multiple_sources_when_both_clear_threshold(self):
        bullet = "Diego's Delta Queen draft report ties to the Ocean Pride grounding report"
        matches = keyword_match_sources(bullet, self.sources)
        refs = {m["ref"] for m in matches}
        self.assertEqual(refs, {"0003.eml", "log.md"})


class TestCorpusCommonKeywords(unittest.TestCase):
    def test_word_in_most_sources_is_flagged_common(self):
        # Real failure found live: "Ocean"/"Pride" appeared in 6 of a
        # tenant's 10 sources because the whole scenario centered on one
        # storyline, letting a 2-keyword short bullet match 6 sources.
        sources = [
            {"source": "email", "ref": f"{i}.eml", "label": "x", "text": "Ocean Pride grounding update"}
            for i in range(4)
        ] + [{"source": "notes", "ref": "unrelated.md", "label": "y", "text": "quarterly budget planning notes"}]
        common = corpus_common_keywords(sources, threshold=0.5)
        self.assertIn("ocean", common)
        self.assertIn("pride", common)
        self.assertNotIn("budget", common)

    def test_below_minimum_source_count_returns_empty(self):
        # With very few sources, "appears in every source" is trivially
        # true for almost any word — the exact regression this guard
        # exists to prevent (it wiped out every keyword in a 1-source
        # test fixture before the guard was added).
        sources = [{"source": "email", "ref": "0001.eml", "label": "x", "text": "Delta Queen draft report"}]
        self.assertEqual(corpus_common_keywords(sources), set())

    def test_keyword_match_sources_excludes_common_keywords_when_given(self):
        sources = [
            {"source": "email", "ref": f"{i}.eml", "label": "x", "text": "Ocean Pride grounding update"}
            for i in range(4)
        ] + [{"source": "calendar", "ref": "evt-1", "label": "y", "text": "Ocean Pride inspection confirmed today"}]
        common = corpus_common_keywords(sources, threshold=0.5)
        # "Ocean Pride" alone (both common) should no longer be enough to
        # match anything once excluded.
        matches = keyword_match_sources("Confirm Ocean Pride availability", sources, common_keywords=common)
        self.assertEqual(matches, [])


class TestLoadCitableSources(unittest.TestCase):
    def test_loads_all_three_source_types_with_correct_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox_dir = os.path.join(tmp, "inbox")
            notes_dir = os.path.join(tmp, "notes")
            calendar_file = os.path.join(tmp, "calendar.ics")
            os.makedirs(inbox_dir)
            os.makedirs(notes_dir)

            with open(os.path.join(inbox_dir, "0003.eml"), "w", encoding="utf-8") as f:
                f.write(_SAMPLE_EML)
            with open(calendar_file, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_ICS)
            with open(os.path.join(notes_dir, "2026-07-18-log.md"), "w", encoding="utf-8") as f:
                f.write(_SAMPLE_NOTE)

            sources = load_citable_sources(inbox_dir, calendar_file, notes_dir)

        by_source = {s["source"]: s for s in sources}
        self.assertEqual(by_source["email"]["ref"], "0003.eml")
        self.assertEqual(by_source["calendar"]["ref"], "test-event-1@example.com")
        self.assertEqual(by_source["notes"]["ref"], "2026-07-18-log.md")
        self.assertIn("Delta Queen", by_source["email"]["text"])
        self.assertIn("Ocean Pride", by_source["calendar"]["text"])


class _FakeJudgeLLM:
    def __init__(self, response):
        self.response = response

    def chat_json(self, messages):
        return self.response


class TestLlmMatchSourcesGrounding(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source": "email", "ref": "0003.eml", "label": "x", "text": "x"},
            {"source": "notes", "ref": "log.md", "label": "y", "text": "y"},
        ]

    def test_keeps_grounded_source_ref(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "source_refs": ["0003.eml"]}]})
        result = llm_match_sources(llm, ["some claim"], self.sources)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["ref"], "0003.eml")

    def test_drops_ungrounded_source_ref(self):
        # The model claims a ref that was never in the candidate list given
        # to it — the exact hallucination shape the grounding check exists
        # to catch, mirroring test_grounding_check.py's own regression.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "source_refs": ["never-shown.eml"]}]})
        result = llm_match_sources(llm, ["some claim"], self.sources)
        self.assertEqual(result, {})

    def test_partial_grounding_keeps_only_the_real_ref(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "source_refs": ["0003.eml", "fake.eml"]}]})
        result = llm_match_sources(llm, ["some claim"], self.sources)
        self.assertEqual([s["ref"] for s in result[0]], ["0003.eml"])

    def test_empty_unmatched_list_skips_the_call_entirely(self):
        llm = _FakeJudgeLLM({"matches": []})
        result = llm_match_sources(llm, [], self.sources)
        self.assertEqual(result, {})


class TestCiteBriefKeywordOnly(unittest.TestCase):
    def test_end_to_end_no_llm(self):
        brief = (
            "# Daily Brief\n\n"
            "## What Matters Today\n\n"
            "Review Diego's Delta Queen draft report today.\n"
            "Unrelated bullet about nothing in the sources.\n"
        )
        sources = [
            {"source": "email", "ref": "0003.eml", "label": "x", "text": "Diego sent the Delta Queen draft report"},
        ]
        annotated, stats = cite_brief(brief, sources, llm=None)
        self.assertIn("Review Diego's Delta Queen draft report today. _[source: email: 0003.eml]_", annotated)
        self.assertIn("Unrelated bullet about nothing in the sources.\n", annotated)
        self.assertNotIn("Unrelated bullet about nothing in the sources. _[source:", annotated)
        self.assertEqual(stats["cited_keyword"], 1)
        self.assertEqual(stats["uncited"], 1)
        self.assertEqual(stats["cited_llm"], 0)


if __name__ == "__main__":
    unittest.main()
