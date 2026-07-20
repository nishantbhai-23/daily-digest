"""
MAP-Phase Extraction Eval
============================
Runs MAP-phase extraction against the known planted scenarios in
golden_scenarios.py and checks whether the expected signal actually got
extracted — turns the manual ad hoc checks done throughout development
(grep-ing ledgers, one-off `python3 -c` snippets) into something
repeatable.

Scoring is done by score_scenario, a pure function separated out from the
three eval_*_scenarios functions below specifically so it's unit-testable
without a real model call (see tests/test_eval_map_scoring.py) — same split
orchestrator.check_priority_coverage uses. It reads golden_scenarios.py's
optional `expected_category` (scope the keyword search to the category the
signal should actually land in, not the whole delta — catches a keyword
landing in the wrong extraction category) and `forbidden_keywords` (a
scenario asserting nothing should be extracted at all — the negative-case
path this eval didn't have before). Both use
digest_checks.extract_searchable_text, which is leaf-string-based rather
than json.dumps-based, so schema key names never leak into the search
corpus.

Also runs digest_checks.check_extraction_bloat per entry — a coarse,
per-source-calibrated sanity bound that warns (never fails) if a day's
extraction looks like a hallucination flood rather than real content.

Optional --llm-judge flag additionally runs quality_judge.judge_map_quality
per scenario — groundedness/completeness/conciseness/coherence, a
supplement to score_scenario's keyword-coverage check (which verifies a
known signal wasn't dropped, not whether the output is otherwise a *good*
summary). Costs one extra LLM call per scenario; print-only, not
persisted to eval_history.py in this pass.

This calls a real model — it is NOT run automatically. Invoke explicitly
once a working LLM provider is available:

    python3 eval_map.py --provider anthropic --model claude-haiku-4-5
    python3 eval_map.py --provider deepseek --model deepseek-chat
    python3 eval_map.py --source email       # limit to one source
    python3 eval_map.py --provider ollama --model llama3   # slow; local
    python3 eval_map.py --llm-judge          # also score summary quality
"""

import argparse
import sys

from digest.agents import calendar_agent
from digest.agents import notes_agent
from digest.agents import triage_agent
from digest.parsers.calendar_parser import group_by_date as group_calendar_by_date
from digest.parsers.calendar_parser import load_calendar
from digest.eval.digest_checks import check_extraction_bloat, check_keywords_present, dynamic_bloat_ceiling, extract_searchable_text
from digest.parsers.email_parser import group_by_date as group_email_by_date
from digest.parsers.email_parser import load_inbox
from digest.eval.golden_scenarios import (
    CALENDAR_MAP_SCENARIOS,
    EMAIL_MAP_SCENARIOS,
    NOTES_MAP_SCENARIOS,
)
from digest.core.llm import create_llm
from digest.parsers.notes_parser import load_notes
from digest.core.persona import load_persona
from digest.eval.quality_judge import judge_map_quality

# Email/calendar bloat ceilings scale with the actual day's batch size (see
# digest_checks.dynamic_bloat_ceiling) instead of one flat number for every
# day regardless of volume — a flat ceiling calibrated against a busy day
# can't catch a hallucination flood on a quiet one. The multiplier is
# schema-motivated, not guessed: each source's MAP_SCHEMA category count is
# the ceiling on how many distinct categories a single input item could
# plausibly populate at once in the busiest realistic case (triage_agent.
# MAP_SCHEMA: deadlines/decisions/action_items/thread_progressions = 4;
# calendar_agent.MAP_SCHEMA also has 4, but pattern_flags/notable_events
# are day-level flags rather than strictly per-event, so a slightly more
# conservative multiplier is used there). Floors keep a single-item day
# from getting a falsely tight bound.
EMAIL_BLOAT_FLOOR = 8
EMAIL_BLOAT_MULTIPLIER = 4.0
CALENDAR_BLOAT_FLOOR = 4
CALENDAR_BLOAT_MULTIPLIER = 3.0

# notes_agent.py MAPs one note at a time, not a day's batch of notes — there
# is no natural per-call input count to scale a ceiling against the way
# there is for email/calendar's per-day batches (a note's own length isn't
# an equivalent signal without more calibration data than exists today), so
# this stays a flat, real-corpus-calibrated ceiling.
NOTES_MAX_ITEMS = 35   # real corpus observed max: 23 items/entry


def score_scenario(entry: dict, scenario: dict, include_stats: bool = False) -> tuple[bool, str]:
    """Pure, LLM-free scoring of one already-produced MAP entry against one
    golden_scenarios.py scenario.

    Args:
        entry: {"delta": {...}} or {"delta": {...}, "stats": {...}} as
            produced by _map_single_day/_map_single_note.
        scenario: one golden_scenarios.py entry — either `required_keywords`
            (positive case, optionally scoped by `expected_category`) or
            `forbidden_keywords` (negative case, always unscoped).
        include_stats: whether this source's deterministic stats field
            should also be searched (True for calendar/notes, False for
            email, which has no per-entry stats field in its MAP schema).

    Returns:
        (passed, detail) — detail is "OK" or a human-readable failure
        reason.
    """
    if "forbidden_keywords" in scenario:
        # expected_category doesn't apply to a forbidden check — a false
        # positive could land in any category, so this always searches the
        # whole delta (+ stats), never category-scoped.
        full = extract_searchable_text(entry.get("delta", {}))
        if include_stats:
            full = extract_searchable_text(entry.get("stats", {})) + " " + full
        present = [kw for kw in scenario["forbidden_keywords"] if kw.lower() in full.lower()]
        return (not present, "OK" if not present else f"forbidden keyword(s) found: {present}")

    delta_text = extract_searchable_text(entry.get("delta", {}), scenario.get("expected_category"))
    haystack = (extract_searchable_text(entry.get("stats", {})) + " " + delta_text) if include_stats else delta_text
    missing = check_keywords_present(haystack, scenario["required_keywords"])
    return (not missing, "OK" if not missing else f"missing: {missing}")


def _run_quality_judge(llm, scenario_name: str, source_items: list[dict], entry: dict) -> None:
    """Opt-in (see --llm-judge) supplement to score_scenario's deterministic
    keyword-coverage check — prints groundedness/completeness/conciseness/
    coherence for this scenario's MAP output. Print-only, like the rest of
    this module's output; not persisted to eval_history.py in this pass.
    """
    source_text = " ".join(extract_searchable_text(item) for item in source_items)
    output_text = extract_searchable_text(entry.get("delta", {}))
    score = judge_map_quality(llm, source_text, output_text)

    g, c, cc, ct = score["groundedness"], score["completeness"], score["conciseness"], score["coherence_tone"]
    print(f"      🔎 quality — groundedness: {g['score']}", end="")
    if g["unverified_claims"]:
        print(f" (unverified: {g['unverified_claims']})", end="")
    print(f" | completeness gaps: {c['gaps']}", end="")
    if c["contradicted_gaps"]:
        print(f" (contradicted: {c['contradicted_gaps']})", end="")
    print(f" | conciseness: {cc['verdict']} (ratio {cc['ratio']})", end="")
    print(f" | coherence/tone: {ct['score']}/5 — {ct.get('notes', '')}")


def eval_email_scenarios(llm, persona_text: str, llm_judge: bool = False) -> list[tuple]:
    prompt = triage_agent.build_map_system_prompt(persona_text)
    by_day = group_email_by_date(load_inbox(triage_agent.INBOX_DIR))

    results = []
    for scenario in EMAIL_MAP_SCENARIOS:
        batch = by_day.get(scenario["day"], [])
        if not batch:
            results.append((scenario["name"], False, "no emails found for that day — regenerate data?"))
            continue
        entry = triage_agent._map_single_day(llm, scenario["day"], batch, prompt, triage_agent.MAP_SCHEMA)
        if entry is None:
            results.append((scenario["name"], False, "MAP call failed after retries"))
            continue
        max_items = dynamic_bloat_ceiling(len(batch), EMAIL_BLOAT_FLOOR, EMAIL_BLOAT_MULTIPLIER)
        bloat = check_extraction_bloat(entry["delta"], max_items)
        if bloat:
            print(f"   ⚠️  {scenario['name']}: {bloat}")
        if llm_judge:
            _run_quality_judge(llm, scenario["name"], batch, entry)
        passed, detail = score_scenario(entry, scenario, include_stats=False)
        results.append((scenario["name"], passed, detail))
    return results


def eval_calendar_scenarios(llm, persona_text: str, llm_judge: bool = False) -> list[tuple]:
    prompt = calendar_agent.build_map_system_prompt(persona_text)
    by_day = group_calendar_by_date(load_calendar(calendar_agent.CALENDAR_FILE))

    results = []
    for scenario in CALENDAR_MAP_SCENARIOS:
        batch = by_day.get(scenario["day"], [])
        if not batch:
            results.append((scenario["name"], False, "no events found for that day — regenerate data?"))
            continue
        entry = calendar_agent._map_single_day(llm, scenario["day"], batch, prompt)
        if entry is None:
            results.append((scenario["name"], False, "MAP call failed after retries"))
            continue
        max_items = dynamic_bloat_ceiling(len(batch), CALENDAR_BLOAT_FLOOR, CALENDAR_BLOAT_MULTIPLIER)
        bloat = check_extraction_bloat(entry["delta"], max_items)
        if bloat:
            print(f"   ⚠️  {scenario['name']}: {bloat}")
        if llm_judge:
            _run_quality_judge(llm, scenario["name"], batch, entry)
        # Deterministic stats and the LLM delta both count as "extraction".
        passed, detail = score_scenario(entry, scenario, include_stats=True)
        results.append((scenario["name"], passed, detail))
    return results


def eval_notes_scenarios(llm, persona_text: str, llm_judge: bool = False) -> list[tuple]:
    prompt = notes_agent.build_map_system_prompt(persona_text)
    notes_by_id = {n["note_id"]: n for n in load_notes(notes_agent.NOTES_DIR)}

    results = []
    for scenario in NOTES_MAP_SCENARIOS:
        note = notes_by_id.get(scenario["note_id"])
        if note is None:
            results.append((scenario["name"], False, "note not found — regenerate data?"))
            continue
        entry = notes_agent._map_single_note(llm, note, prompt, notes_agent.MAP_SCHEMA)
        if entry is None:
            results.append((scenario["name"], False, "MAP call failed after retries"))
            continue
        bloat = check_extraction_bloat(entry["delta"], NOTES_MAX_ITEMS)
        if bloat:
            print(f"   ⚠️  {scenario['name']}: {bloat}")
        if llm_judge:
            _run_quality_judge(llm, scenario["name"], [note], entry)
        passed, detail = score_scenario(entry, scenario, include_stats=True)
        results.append((scenario["name"], passed, detail))
    return results


def print_results(source: str, results: list[tuple]) -> None:
    print(f"\n{source.upper()}")
    for name, passed, detail in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} — {detail}")


def main():
    parser = argparse.ArgumentParser(description="Golden-scenario MAP-phase extraction eval")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--source", choices=["email", "calendar", "notes", "all"], default="all")
    parser.add_argument("--llm-judge", action="store_true", help="Also run the in-house quality judge (groundedness/completeness/conciseness/coherence) per scenario — costs one extra LLM call each")
    args = parser.parse_args()

    persona_text = load_persona()
    llm = create_llm(provider=args.provider, model=args.model, temperature=0.0)

    all_results = []
    if args.source in ("email", "all"):
        all_results.append(("email", eval_email_scenarios(llm, persona_text, llm_judge=args.llm_judge)))
    if args.source in ("calendar", "all"):
        all_results.append(("calendar", eval_calendar_scenarios(llm, persona_text, llm_judge=args.llm_judge)))
    if args.source in ("notes", "all"):
        all_results.append(("notes", eval_notes_scenarios(llm, persona_text, llm_judge=args.llm_judge)))

    total_pass = total = 0
    for source, results in all_results:
        print_results(source, results)
        total_pass += sum(1 for _, passed, _ in results if passed)
        total += len(results)

    print(f"\n{'─' * 40}\n{total_pass}/{total} scenarios passed\n")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
