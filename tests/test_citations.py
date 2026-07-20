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

import json
import os
import tempfile
import unittest

from digest.core.citations import (
    cite_brief,
    corpus_common_keywords,
    embedding_match_sources,
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


def _dissimilar_embed_fn(texts):
    """Deterministic, network-free stand-in for digest.core.embeddings.embed_texts
    — every distinct text gets its own orthogonal one-hot vector, so cosine
    similarity is always ~0 between any two different texts (and 1.0 for
    identical ones). Used everywhere llm_match_sources/cite_brief's
    embedding OR-branch would otherwise fall back to a real local Ollama
    call, which would make these tests slow, non-deterministic, and
    dependent on the test machine having Ollama running with an embedding
    model pulled — none of which this suite should require to pass.
    """
    unique = {}
    for text in texts:
        if text not in unique:
            unique[text] = len(unique)
    dim = len(unique)
    return [[1.0 if i == unique[text] else 0.0 for i in range(dim)] for text in texts]


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
        self.assertNotIn("task", by_source)  # tasks_file omitted -> no task sources

    def test_tasks_file_included_when_given(self):
        # Real gap found live: tasks were never loaded at all, so a digest
        # bullet about a task could never be cited to it, and nothing could
        # detect content duplicated from an email into a task description.
        with tempfile.TemporaryDirectory() as tmp:
            inbox_dir = os.path.join(tmp, "inbox")
            notes_dir = os.path.join(tmp, "notes")
            calendar_file = os.path.join(tmp, "calendar.ics")
            tasks_file = os.path.join(tmp, "tasks.json")
            os.makedirs(inbox_dir)
            os.makedirs(notes_dir)

            with open(os.path.join(inbox_dir, "0003.eml"), "w", encoding="utf-8") as f:
                f.write(_SAMPLE_EML)
            with open(calendar_file, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_ICS)
            with open(os.path.join(notes_dir, "2026-07-18-log.md"), "w", encoding="utf-8") as f:
                f.write(_SAMPLE_NOTE)
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump([{"id": "T-1", "title": "Review Delta Queen draft", "description": "Finish the review by Friday.", "status": "todo", "priority": "P1"}], f)

            sources = load_citable_sources(inbox_dir, calendar_file, notes_dir, tasks_file)

        by_source = {s["source"]: s for s in sources}
        self.assertEqual(by_source["task"]["ref"], "T-1")
        self.assertIn("Delta Queen", by_source["task"]["text"])


class _FakeJudgeLLM:
    def __init__(self, response):
        self.response = response

    def chat_json(self, messages):
        return self.response


class TestLlmMatchSourcesGrounding(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "source": "email", "ref": "0003.eml", "label": "x",
                "text": (
                    "Press #3 vibration readings are trending upward this week. "
                    "Also the new felt from DuroFelt arrived yesterday."
                ),
            },
            {"source": "notes", "ref": "log.md", "label": "y", "text": "Tom reported Press #2 jam cleared in standup"},
        ]
        self.vibration_claim = "Press vibration readings are trending upward"

    def test_keeps_grounded_source_ref_with_real_quote(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "vibration readings are trending upward"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["ref"], "0003.eml")

    def test_drops_ungrounded_source_ref(self):
        # The model claims a ref that was never in the candidate list given
        # to it — the exact hallucination shape the grounding check exists
        # to catch, mirroring test_grounding_check.py's own regression.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "never-shown.eml", "quote": "anything"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_partial_grounding_keeps_only_the_real_ref(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "vibration readings are trending upward"},
            {"source_ref": "fake.eml", "quote": "anything"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual([s["ref"] for s in result[0]], ["0003.eml"])

    def test_empty_unmatched_list_skips_the_call_entirely(self):
        llm = _FakeJudgeLLM({"matches": []})
        result = llm_match_sources(llm, [], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_drops_match_whose_quote_is_not_actually_in_the_source(self):
        # Real bug found live: the judge attached a genuinely valid ref
        # (0003.eml, the Press #3 vibration email) to a claim about "Press
        # #2 jam cleared" — a topic that email never mentions at all. The
        # ref existed (so ref-grounding alone passed it through), but no
        # real quote from 0003.eml could ever support that claim. This is
        # the exact case quote-existence verification exists to catch.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "Press #2 jam cleared"},
        ]}]})
        result = llm_match_sources(llm, ["Press #2 jam cleared"], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_drops_real_but_irrelevant_quote(self):
        # Second live-found bug, narrower than the first: after adding
        # quote-existence verification, the judge started passing quote
        # checks with a real-but-unrelated span from the *same* wrong
        # source — here, the email's own felt-delivery sentence, quoted as
        # "evidence" for an unrelated jam claim. The quote genuinely exists
        # in 0003.eml (layer 2 passes), but shares no real keyword with the
        # claim, so layer 3 must still reject it.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "the new felt from DuroFelt arrived yesterday"},
        ]}]})
        result = llm_match_sources(llm, ["Press #2 jam cleared in standup, Tom reported"], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_common_keyword_alone_cannot_satisfy_relevance_check(self):
        # Mirrors the actual live corpus behavior: "Press" appeared in most
        # of demo-1's sources, so corpus_common_keywords filtered it before
        # cite_brief ever calls llm_match_sources. Without that filtering,
        # sharing only a corpus-common word would wrongly look relevant.
        # Deliberately no "#N" tokens on either side, so this exercises
        # layer 3 specifically rather than being caught by layer 4 first.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "Press readings trending upward"},
        ]}]})
        result = llm_match_sources(
            llm, ["Press jam cleared in standup"], self.sources, common_keywords={"press"},
            embed_fn=_dissimilar_embed_fn,
        )
        self.assertEqual(result, {})

    def test_drops_match_naming_a_different_numbered_instance(self):
        # Third live-found bug, narrower still: layers 2 and 3 both passed
        # a real, on-topic-sounding quote — "Press #3 vibration analysis,"
        # the email's own subject line — against a claim actually about a
        # *different* press. "press" and "vibration" aren't corpus-common
        # enough in this fixture to be filtered by layer 3, so keyword
        # overlap alone lets it through. The claim names a specific
        # instance ("#2"); the quote names a different one ("#3") — layer 4
        # must catch that even though the surrounding vocabulary overlaps.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "Press #3 vibration readings are trending upward"},
        ]}]})
        result = llm_match_sources(llm, ["Press #2 jam cleared, vibration readings normal"], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_keeps_match_naming_the_same_numbered_instance(self):
        # The positive case for layer 4: claim and quote both name "#3",
        # so the identifier check shouldn't reject a genuinely correct
        # match just because a "#N" token happens to be present.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "Press #3 vibration readings are trending upward"},
        ]}]})
        result = llm_match_sources(llm, ["Press #3 vibration readings trending upward"], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["ref"], "0003.eml")

    def test_no_identifier_in_claim_skips_layer_four_entirely(self):
        # A claim with no "#N" token at all shouldn't require one in the
        # quote either — layer 4 only applies when the claim itself names
        # a specific instance.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "vibration readings are trending upward"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(len(result[0]), 1)

    def test_missing_quote_field_drops_the_match(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(result, {})

    def test_quote_matching_is_case_insensitive(self):
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "VIBRATION READINGS ARE TRENDING UPWARD"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual(len(result[0]), 1)

    def test_mixed_valid_and_unverifiable_matches_on_same_claim(self):
        # One evidence item is real, the other's quote doesn't exist in
        # its source — only the real one should survive.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0003.eml", "quote": "vibration readings are trending upward"},
            {"source_ref": "log.md", "quote": "this text is not in the note"},
        ]}]})
        result = llm_match_sources(llm, [self.vibration_claim], self.sources, embed_fn=_dissimilar_embed_fn)
        self.assertEqual([s["ref"] for s in result[0]], ["0003.eml"])


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
        annotated, stats = cite_brief(brief, sources, llm=None, embed_fn=None)
        self.assertIn("Review Diego's Delta Queen draft report today. _[source: email: 0003.eml]_", annotated)
        self.assertIn("Unrelated bullet about nothing in the sources.\n", annotated)
        self.assertNotIn("Unrelated bullet about nothing in the sources. _[source:", annotated)
        self.assertEqual(stats["cited_keyword"], 1)
        self.assertEqual(stats["uncited"], 1)
        self.assertEqual(stats["cited_llm"], 0)


# Two 2D unit vectors with a known, hand-computed cosine similarity — lets
# tests engineer an exact above/below-_EMBEDDING_SIMILARITY_THRESHOLD (0.78)
# result without depending on any real embedding model's actual output.
_HIGH_SIM_A = [1.0, 0.0]
_HIGH_SIM_B = [0.9, 0.43589]   # cosine similarity with _HIGH_SIM_A ≈ 0.9
_LOW_SIM = [0.3, 0.95394]      # cosine similarity with _HIGH_SIM_A ≈ 0.3


class TestEmbeddingMatchSources(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source": "email", "ref": "0001.eml", "label": "a", "text": "irrelevant"},
            {"source": "email", "ref": "0002.eml", "label": "b", "text": "irrelevant"},
        ]

    def test_source_at_or_above_threshold_is_matched(self):
        matches = embedding_match_sources(_HIGH_SIM_A, [_HIGH_SIM_B], [self.sources[0]])
        self.assertEqual([s["ref"] for s in matches], ["0001.eml"])

    def test_source_below_threshold_is_not_matched(self):
        matches = embedding_match_sources(_HIGH_SIM_A, [_LOW_SIM], [self.sources[0]])
        self.assertEqual(matches, [])

    def test_multiple_matches_sorted_by_similarity_descending(self):
        matches = embedding_match_sources(_HIGH_SIM_A, [_LOW_SIM, _HIGH_SIM_B], self.sources)
        self.assertEqual([s["ref"] for s in matches], ["0002.eml"])


_BULLET_TEXT = "Reached out to Andrew regarding the hardware slip."
_SOURCE_TEXT = "called Drew about the edge-compute delay"


def _dispatch_embed_fn(mapping, default=_LOW_SIM):
    """Builds an embed_fn that returns a fixed vector per exact text,
    falling back to `default` for anything unlisted — keeps each test's
    intent explicit (which exact strings are "similar") instead of
    relying on brittle substring sniffing.
    """
    def embed_fn(texts):
        return [mapping.get(t, default) for t in texts]
    return embed_fn


class TestCiteBriefEmbeddingTier(unittest.TestCase):
    def setUp(self):
        self.sources = [{"source": "email", "ref": "0001.eml", "label": "x", "text": _SOURCE_TEXT}]
        self.brief = f"# Daily Brief\n\n## What Matters Today\n\n{_BULLET_TEXT}\n"

    def test_zero_keyword_overlap_bullet_matched_via_embedding_tier(self):
        # _BULLET_TEXT shares no real keyword with _SOURCE_TEXT —
        # keyword_match_sources structurally can't place it. A high
        # embedding similarity between the two should rescue it without
        # ever reaching the (unconfigured, would-crash-if-called) LLM tier.
        embed_fn = _dispatch_embed_fn({_BULLET_TEXT: _HIGH_SIM_A, _SOURCE_TEXT: _HIGH_SIM_B})

        annotated, stats = cite_brief(self.brief, self.sources, llm=None, embed_fn=embed_fn)
        self.assertIn("(embedding-matched)", annotated)
        self.assertIn("email: 0001.eml", annotated)
        self.assertEqual(stats["cited_embedding"], 1)
        self.assertEqual(stats["cited_keyword"], 0)
        self.assertEqual(stats["uncited"], 0)

    def test_below_threshold_falls_through_to_llm_tier(self):
        # Outer tier: bullet vs. full source text scores low, so it falls
        # through. Inner layer 3 (the LLM's own quote, a *different*
        # string from the full source text) gets a high similarity to the
        # bullet, so the LLM tier successfully grounds and cites it —
        # proving the two tiers are independently wired, not conflated.
        quote = "Drew about the edge-compute delay"  # a real substring of _SOURCE_TEXT
        embed_fn = _dispatch_embed_fn({
            _BULLET_TEXT: _HIGH_SIM_A,
            _SOURCE_TEXT: _LOW_SIM,
            quote: _HIGH_SIM_B,
        })
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0001.eml", "quote": quote},
        ]}]})
        annotated, stats = cite_brief(self.brief, self.sources, llm=llm, embed_fn=embed_fn)
        self.assertEqual(stats["cited_embedding"], 0)
        self.assertEqual(stats["cited_llm"], 1)
        self.assertIn("(inferred)", annotated)

    def test_embed_fn_none_disables_embeddings_everywhere_bullet_ends_uncited(self):
        # embed_fn=None must disable embeddings both at this outer tier
        # AND inside llm_match_sources' own layer-3 rescue — with zero
        # keyword overlap and no embedding available anywhere, there's
        # genuinely no way to ground this claim, so it should correctly
        # end up uncited rather than crash or get cited on a technicality.
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0001.eml", "quote": _SOURCE_TEXT},
        ]}]})
        annotated, stats = cite_brief(self.brief, self.sources, llm=llm, embed_fn=None)
        self.assertEqual(stats["cited_embedding"], 0)
        self.assertEqual(stats["cited_llm"], 0)
        self.assertEqual(stats["uncited"], 1)

    def test_embedding_failure_degrades_gracefully_without_crashing(self):
        # A broken embed_fn (e.g. Ollama not running) must not crash
        # cite_brief — it should print a warning and fall through,
        # landing wherever the remaining tiers (keyword, then LLM) can
        # legitimately place it. With zero keyword overlap on both sides,
        # that's correctly "uncited," not a crash.
        def broken_embed_fn(texts):
            raise RuntimeError("Ollama server not running")

        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0001.eml", "quote": _SOURCE_TEXT},
        ]}]})
        annotated, stats = cite_brief(self.brief, self.sources, llm=llm, embed_fn=broken_embed_fn)
        self.assertEqual(stats["cited_keyword"] + stats["cited_embedding"] + stats["cited_llm"] + stats["uncited"], 1)
        self.assertEqual(stats["uncited"], 1)


class TestLlmMatchSourcesEmbeddingRescue(unittest.TestCase):
    def test_embedding_similarity_rescues_a_zero_keyword_overlap_quote(self):
        # Layer 3's keyword-overlap check alone would drop this — the
        # claim and quote share no real keyword. The embedding OR-branch
        # should rescue it when similarity clears threshold, exactly the
        # paraphrase gap layer 3's own docstring calls out as its weakest
        # point.
        sources = [{"source": "email", "ref": "0001.eml", "label": "x", "text": _SOURCE_TEXT}]
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0001.eml", "quote": _SOURCE_TEXT},
        ]}]})
        embed_fn = _dispatch_embed_fn({_BULLET_TEXT: _HIGH_SIM_A, _SOURCE_TEXT: _HIGH_SIM_B})

        result = llm_match_sources(llm, [_BULLET_TEXT], sources, embed_fn=embed_fn)
        self.assertEqual([s["ref"] for s in result[0]], ["0001.eml"])

    def test_low_embedding_similarity_still_drops_the_match(self):
        sources = [{"source": "email", "ref": "0001.eml", "label": "x", "text": _SOURCE_TEXT}]
        llm = _FakeJudgeLLM({"matches": [{"claim_index": 0, "evidence": [
            {"source_ref": "0001.eml", "quote": _SOURCE_TEXT},
        ]}]})
        embed_fn = _dispatch_embed_fn({_BULLET_TEXT: _HIGH_SIM_A, _SOURCE_TEXT: _LOW_SIM})

        result = llm_match_sources(llm, [_BULLET_TEXT], sources, embed_fn=embed_fn)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
