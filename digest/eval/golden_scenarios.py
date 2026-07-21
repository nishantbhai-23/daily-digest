"""
Golden Scenarios
==================
Registry of deliberately planted or identified scenarios with known-correct
expected signals, across four categories:
- EMAIL/CALENDAR/NOTES_MAP_SCENARIOS — planted in the synthetic data
  (generate_data.py). Used by eval_map.py (MAP-phase extraction) and
  reusable for REDUCE-level digest checks — both check the same underlying
  facts, just at different pipeline stages.
- CONTRADICTION_SCENARIOS — planted via --hot-input rather than
  generate_data.py, to avoid invalidating the existing 30-day corpus. Used
  by orchestrator Stage 1 verification.
- PRIORITY_CALIBRATION_SCENARIOS — not planted at all, identified in the
  existing corpus where persona.md's own rules make the correct priority
  unambiguous. Used by eval_prompt_variants.py to compare MAP prompt
  variants against a measurable target.
- CROSS_REFERENCE_SCENARIOS — not planted, observed: real tasks in the
  arclight tenant already flagged as time-sensitive with real cross-source
  mentions, confirmed by running cross_reference.build_cross_reference_index
  directly against arclight's actual ledgers. Used by
  eval_cross_reference_variants.py to compare the lexical and
  embedding-assisted Stage-0 implementations.

This turns the manual `python3 -c` / `jq` checks done throughout
development into something repeatable: every scenario here is a real bug,
design decision, or open question this project actually hit, not a
hypothetical.

Each MAP scenario's `required_keywords` check is deliberately blunt (see
digest_checks.py) — a coarse proxy for "did the signal survive," not a
substitute for reading the output. Still fully deterministic and free (no
LLM judge, no ground-truth circularity — see the module docstring on
digest_checks.check_keywords_present for why an ungrounded LLM judge was
considered and rejected here after citations.py's own judge needed four
rounds of grounding fixes to stop hallucinating matches on a structurally
similar problem): an entry in `required_keywords` can be a plain string
(must appear literally) or a nested list — an OR-set, at least one variant
must appear, e.g. `["budget", ["corrugator", "line 3", "grinding"]]` —
for a concept the model's own wording varies on run to run, without
resorting to an ungrounded judge or a brittle single literal string. Two
further optional fields narrow what the proxy actually checks:
- `expected_category` — a single MAP_SCHEMA category name, or a list where
  a scenario's own description documents genuine ambiguity between two
  categories. Scopes the keyword search to just that subtree (via
  digest_checks.extract_searchable_text) instead of the whole delta, so a
  keyword landing in the wrong category no longer passes for free. Omit
  when no single category (or small ambiguous set) is truly correct —
  forcing precision the scenario doesn't have would be worse than leaving
  it unscoped.
- `forbidden_keywords` — the inverse of `required_keywords`: this day's
  extraction must NOT contain any of these, anywhere. A scenario has
  either `required_keywords` or `forbidden_keywords`, never both — this is
  the negative-case path (testing precision: nothing should be extracted)
  as opposed to the positive path (testing recall: something should be).
"""

EMAIL_MAP_SCENARIOS = [
    {
        "name": "quiet_marcus_investor_thread",
        "day": "2026-06-19",
        "description": (
            "Marcus Webb's data-room ask, sent once and never followed up on — "
            "this day's MAP output should extract an action item about Marcus "
            "and data room access. The email-generator reply pool used to leak "
            "Marcus into unrelated threads, silently breaking this scenario; "
            "fixed by excluding him from the reply pool, not just new-email "
            "selection."
        ),
        "required_keywords": ["marcus"],
        "expected_category": "action_items",
    },
    {
        "name": "halberd_souring_signal",
        "day": "2026-07-03",
        "description": (
            "Carla Whitfield's 'reassessing budget allocations' email — a "
            "customer-health signal buried in a routine-sounding renewal "
            "thread. Should show up as a thread_progression or decision, not "
            "get treated as ordinary customer chatter."
        ),
        # "halberd", not "carla" — corrected via this eval framework's own
        # live verification (score_scenario's new category scoping
        # surfaced a failure here that the old whole-delta search masked).
        # Real MAP output consistently attributes this signal to the
        # company ("Halberd flagged potential budget reassessment...") in
        # thread_progressions, not the specific sender's name — the old
        # unscoped "carla" match was only ever passing because "Carla" also
        # appears, coincidentally, in a wholly unrelated action_item
        # ("Intro Diane to Carla at Halberd for reference call") elsewhere
        # in the same day's extraction. Requiring "carla" specifically was
        # testing a naming choice the MAP prompt doesn't reliably make, not
        # the actual signal this scenario cares about (the risk itself
        # getting surfaced, not which name it's attributed to).
        "required_keywords": ["halberd", "budget"],
        # Genuinely ambiguous per this scenario's own description above —
        # narrowed to these two plausible categories rather than one, which
        # would force precision the scenario doesn't actually have.
        "expected_category": ["thread_progressions", "decisions"],
    },
    {
        "name": "unlogged_sla_promise",
        "day": "2026-06-25",
        "description": (
            "Zoe Alvarez's reminder about the Halberd SLA one-pager — a "
            "promise made in conversation with no corresponding task. Should "
            "be extracted as an action item."
        ),
        "required_keywords": ["sla"],
        "expected_category": "action_items",
    },
    {
        "name": "stalled_elena_marsh_hiring",
        "day": "2026-06-22",
        "description": (
            "Grace Lin's Elena Marsh onsite feedback — flagged as needing a "
            "fast decision, then goes quiet. Should be extracted as a "
            "decision/action item, not a routine hiring update."
        ),
        "required_keywords": ["elena"],
        "expected_category": ["decisions", "action_items"],
    },
    {
        "name": "recruiter_noise_suppression",
        "day": "2026-06-17",
        "description": (
            "Trent Bailey (Apex Technical Recruiting) sent exactly 2 "
            "cold-pitch emails this day — below persona.md's '3+ from the "
            "same firm in a week' surfacing threshold (same fact "
            "PRIORITY_CALIBRATION_SCENARIOS' recruiter_cold_pitch already "
            "established for eval_prompt_variants.py; this generalizes it "
            "into eval_map.py's actual regression suite, which never tested "
            "it). MAP should extract nothing about him at all — the first "
            "scenario here testing precision (nothing extracted) rather than "
            "recall (something extracted)."
        ),
        "forbidden_keywords": ["trent", "apex"],
    },
]

CALENDAR_MAP_SCENARIOS = [
    {
        "name": "deep_work_violation",
        "day": "2026-06-30",
        "description": (
            "Halberd Quarterly Review scheduled directly over the protected "
            "9-11am deep-work block. The deterministic stats.deep_work_conflicts "
            "should be non-empty (this part doesn't need an LLM at all), and "
            "the LLM's qualitative delta should flag it as a violation."
        ),
        "required_keywords": ["halberd"],
        # Deliberately left without expected_category: "flag it as a
        # violation" doesn't pin down a single MAP_SCHEMA category — both
        # pattern_flags and notable_events are plausible fits, and no
        # existing precedent says which. Forcing one would be a guess, not
        # a real assertion; falls back to the whole-delta search.
    },
    {
        "name": "family_calendar_collision",
        "day": "2026-06-25",
        "description": (
            "Wren's pediatrician follow-up collides with the GTM Pipeline "
            "Review — should be flagged as a family_calendar_item, and per the "
            "profile's 'family first, always' rule, surfaced prominently."
        ),
        "required_keywords": ["wren"],
        "expected_category": "family_calendar_items",
    },
]

NOTES_MAP_SCENARIOS = [
    {
        "name": "multiwarehouse_open_item",
        "note_id": "2026-07-07-weekly-priorities.md",
        "description": (
            "The weekly-priorities note's open 'decide multi-warehouse "
            "approach' item should be flagged as still relevant — it's the "
            "same decision that appears unresolved across 4 separate notes "
            "(6/17, 6/23, 6/30, 7/07), getting older each time. Deterministic "
            "checklist staleness gives the exact day-count; the LLM's job is "
            "noticing it's the same recurring decision, not a new one."
        ),
        "required_keywords": ["multi-warehouse", "warehouse"],
        "expected_category": "open_items_still_relevant",
    },
]

# Digest-level (REDUCE) checks reuse the same keyword lists as a coarse
# proxy — if MAP correctly extracted "Marcus" + "data room", the REDUCE
# digest should still mention Marcus somewhere in its "about to drop"
# section. Not a strict equivalence, just a cheap regression signal.
EMAIL_DIGEST_SCENARIOS = EMAIL_MAP_SCENARIOS
CALENDAR_DIGEST_SCENARIOS = CALENDAR_MAP_SCENARIOS
NOTES_DIGEST_SCENARIOS = NOTES_MAP_SCENARIOS

# Orchestrator Stage 1 (contradiction detection) scenarios — a different
# category from the MAP scenarios above: these check whether
# detect_contradictions actually finds a genuine cross-source disagreement,
# not just correctly returns empty when there's nothing to find. Added after
# a design review noted Stage 1 had never been exercised against a true
# positive — only the empty case, plus one unplanned false positive (the
# Elena Marsh hallucination that motivated the grounding check in the first
# place). Planted via --hot-input rather than generate_data.py, specifically
# to avoid invalidating the existing 30-day corpus and its already-built
# ledger — see docs/LOW_LEVEL_DESIGN.md's "Planted scenarios" section for
# the full reasoning.
CONTRADICTION_SCENARIOS = [
    {
        "name": "marcus_closing_call_reschedule",
        "description": (
            "data/scenarios/contradiction_marcus_reschedule/calendar/calendar.ics "
            "(Friday 7/17, 'Call with Marcus — Series A closing') directly "
            "contradicts .../inbox/0001.eml (7/16, Marcus asking to move the "
            "call to Monday) — a genuine date conflict between two sources "
            "for the same flagged task (TESS-225: 'Confirm Series A closing "
            "call time with Marcus'), deliberately mirroring persona.md's own "
            "worked example ('calendar says I'm meeting Marcus Friday but "
            "email says we moved to Monday'). Live-verified via DeepSeek: "
            "Stage 1 correctly reported it, and the entity resolved and "
            "passed the grounding check — which surfaced and fixed a real "
            "bug in _resolve_entity along the way (the model referenced the "
            "entity as 'TESS-225: <title>', a combined format the grounding "
            "check didn't originally handle and silently dropped as "
            "'ungrounded' — a false negative in the safety check itself)."
        ),
        "task_id": "TESS-225",
        "required_keywords": ["marcus", "friday", "monday"],
    },
]

# Priority-calibration scenarios — used by eval_prompt_variants.py to compare
# MAP prompt variants (zero-shot vs. few-shot) against a measurable target,
# rather than adding few-shot examples on priors. Each entry names a real
# email in the existing corpus where persona.md's own stated rules make the
# correct P0-P4 priority unambiguous — not a judgment call this eval script
# has to make itself, just a comparison against what the profile already says.
PRIORITY_CALIBRATION_SCENARIOS = [
    {
        "name": "quiet_marcus_investor_thread",
        "day": "2026-06-19",
        "match_on": "Marcus",
        "expected_priority": "P0",
        "reasoning": "persona.md: 'Marcus Webb (Series A lead...) — P0 during the raise.'",
    },
    {
        "name": "halberd_souring_signal",
        "day": "2026-07-03",
        "match_on": "Carla",
        "expected_priority": "P1",
        "reasoning": (
            "persona.md: 'The procurement leads at Halberd Manufacturing, "
            "Northstar Foods, and Veritas Components — our three reference "
            "customers. Anything from them is P1.'"
        ),
    },
    {
        "name": "recruiter_cold_pitch",
        "day": "2026-06-17",
        "match_on": "Trent",
        # Corrected via the eval harness's own dry run — the original guess
        # here ("P4") was wrong, not the model. persona.md: "Recruiters
        # cold-emailing me — P4... Don't surface them unless three+ from the
        # same firm in a week." Trent Bailey sent exactly 2 emails on this
        # specific day (verified directly against data/inbox/), short of
        # the 3+ threshold — correct behavior is to extract nothing at all,
        # not to tag one P4. Both variants correctly returned "not_found" in
        # the dry run; kept as a scenario testing noise-suppression
        # consistency instead of numeric priority calibration.
        "expected_priority": "not_found",
        "reasoning": (
            "persona.md: 'Recruiters cold-emailing me — P4... Don't surface "
            "them unless three+ from the same firm in a week.' Only 2 emails "
            "from Trent Bailey (Apex Technical Recruiting) on this day — "
            "below the 3+ threshold, so correct MAP output extracts nothing "
            "for this sender at all."
        ),
    },
]

# Cross-reference scenarios — used by eval_cross_reference_variants.py to
# compare cross_reference.py's lexical and embedding-assisted Stage-0
# implementations. Each entry names a real, already-flagged task in the
# arclight tenant's actual data, with the sources build_cross_reference_index
# (the lexical variant) is directly verified to find today — not a planted
# fixture, an observed fact confirmed by running the lexical function
# against arclight's real ledgers. expected_sources is a floor, not a
# ceiling: a variant that finds a superset (e.g. embedding_assisted adding a
# calendar mention lexical misses) still passes.
#
# forbidden_mentions is the precision-testing counterpart — the inverse of
# expected_sources, same positive/negative-pair convention this file
# already uses for MAP scenarios (required_keywords vs. forbidden_keywords).
# A scenario has either expected_sources or forbidden_mentions, never both.
# Every entry below is a real, live-verified near-miss found while
# calibrating cross_reference.py's embedding threshold: a genuinely
# topically-related item (headcount/SRE-capacity chatter) that scored
# 0.50-0.52 cosine similarity against ARC-102's title — real, but
# correctly judged not the same mention at the current 0.70 threshold.
# Recorded as a permanent regression test rather than a one-off manual
# check, so a future threshold change or embedding model swap that starts
# pulling these in gets caught here instead of shipping a silent
# precision regression.
CROSS_REFERENCE_SCENARIOS = [
    {
        "name": "arclight_sre_req_open",
        "tenant_id": "arclight",
        "task_id": "ARC-102",
        "expected_sources": {"email", "notes"},
    },
    {
        "name": "arclight_oncall_rebalance_blocked",
        "tenant_id": "arclight",
        "task_id": "ARC-103",
        "expected_sources": {"email", "calendar", "notes"},
    },
    {
        "name": "arclight_sre_req_not_headcount_ask_email",
        "tenant_id": "arclight",
        "task_id": "ARC-102",
        # "Send Q3 eng headcount ask to Om today" — real, topically
        # adjacent (headcount/hiring), similarity 0.508. A different,
        # specific ask than the SRE req itself, not the same mention.
        "forbidden_mentions": [{"source": "email", "day": "2026-07-03"}],
    },
    {
        "name": "arclight_sre_req_not_capacity_retro_prep",
        "tenant_id": "arclight",
        "task_id": "ARC-102",
        # "Forecasting Fix Ship Retro Prep... Prep notes for SRE capacity
        # retro" — real, similarity 0.524, the closest near-miss found.
        # A retro about current team capacity/workload, not the same
        # thing as the open req/interview task.
        "forbidden_mentions": [{"source": "calendar", "day": "2026-07-10"}],
    },
    # richcross-test — generated via generate_persona.py specifically to
    # stress-test this comparison with richer, more naturally-interconnected
    # content than arclight's hand-authored fixture. Confirmed real: run
    # build_cross_reference_index directly against its ledgers, then read
    # the actual raw source files (not just the extracted snippets) before
    # asserting anything, since generated content turned out ambiguous in
    # ways arclight's didn't — one genuinely close near-miss (0.611
    # similarity, the calendar entry for the meeting where the sensor
    # decision was actually made) was deliberately NOT encoded here,
    # since a reasonable person could argue it should legitimately match;
    # only the one confidently-wrong case below was kept.
    {
        "name": "richcross_palletco_contract_open",
        "tenant_id": "richcross-test",
        "task_id": "task-001",
        "expected_sources": {"calendar", "email"},
    },
    {
        "name": "richcross_forklift_sensors_open",
        "tenant_id": "richcross-test",
        "task_id": "task-002",
        "expected_sources": {"email", "notes"},
    },
    {
        "name": "richcross_palletco_not_autovend_sla_note",
        "tenant_id": "richcross-test",
        "task_id": "task-001",
        # notes/2026-07-20-contractreviewthoughts.md is titled "Thoughts on
        # AutoVend contract draft" — a different vendor entirely, not
        # PalletCo (task-001's actual subject). Similarity 0.490 against
        # task-001's title despite being about an unrelated contract —
        # confirmed by reading the raw note content, not just the
        # embedding-matched snippet.
        "forbidden_mentions": [{"source": "notes", "day": "2026-07-20-contractreviewthoughts.md"}],
    },
]
