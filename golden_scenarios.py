"""
Golden Scenarios
==================
Registry of the deliberately planted scenarios in the synthetic data
(generate_data.py) with their expected signals. Used by eval_map.py
(MAP-phase extraction) and can be reused for REDUCE-level digest checks —
both check the same underlying facts, just at different pipeline stages.

This turns the manual `python3 -c` / `jq` checks done throughout
development into something repeatable: every scenario here is a real bug
or design decision this project actually hit, not a hypothetical.

Each scenario's `required_keywords` check is deliberately blunt (see
digest_checks.py) — a coarse proxy for "did the signal survive," not a
substitute for reading the output.
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
        "required_keywords": ["carla", "budget"],
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
