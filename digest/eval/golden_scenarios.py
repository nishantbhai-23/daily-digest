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

This turns the manual `python3 -c` / `jq` checks done throughout
development into something repeatable: every scenario here is a real bug,
design decision, or open question this project actually hit, not a
hypothetical.

Each MAP scenario's `required_keywords` check is deliberately blunt (see
digest_checks.py) — a coarse proxy for "did the signal survive," not a
substitute for reading the output. Two optional fields narrow what that
proxy actually checks:
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
