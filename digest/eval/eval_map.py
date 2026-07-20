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

This calls a real model — it is NOT run automatically. Invoke explicitly
once a working LLM provider is available:

    python3 eval_map.py --provider anthropic --model claude-haiku-4-5
    python3 eval_map.py --provider deepseek --model deepseek-chat
    python3 eval_map.py --source email       # limit to one source
    python3 eval_map.py --provider ollama --model llama3   # slow; local
"""

import argparse
import sys

from digest.agents import calendar_agent
from digest.agents import notes_agent
from digest.agents import triage_agent
from digest.parsers.calendar_parser import group_by_date as group_calendar_by_date
from digest.parsers.calendar_parser import load_calendar
from digest.eval.digest_checks import check_extraction_bloat, check_keywords_present, extract_searchable_text
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

# Per-source bloat thresholds — calibrated against this project's own real
# corpus (sampled directly from output/*_rolling_ledger.json), not guessed.
# Calendar runs much lower volume than email/notes by nature (meeting-based,
# not thread-based), so one flat threshold across all three would either be
# useless for calendar or false-positive on a legitimately busy email/notes
# day.
EMAIL_MAX_ITEMS = 40      # real corpus observed max: 31 items/day
NOTES_MAX_ITEMS = 35      # real corpus observed max: 23 items/entry
CALENDAR_MAX_ITEMS = 15   # real corpus observed max: 5 items/entry


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


def eval_email_scenarios(llm, persona_text: str) -> list[tuple]:
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
        bloat = check_extraction_bloat(entry["delta"], EMAIL_MAX_ITEMS)
        if bloat:
            print(f"   ⚠️  {scenario['name']}: {bloat}")
        passed, detail = score_scenario(entry, scenario, include_stats=False)
        results.append((scenario["name"], passed, detail))
    return results


def eval_calendar_scenarios(llm, persona_text: str) -> list[tuple]:
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
        bloat = check_extraction_bloat(entry["delta"], CALENDAR_MAX_ITEMS)
        if bloat:
            print(f"   ⚠️  {scenario['name']}: {bloat}")
        # Deterministic stats and the LLM delta both count as "extraction".
        passed, detail = score_scenario(entry, scenario, include_stats=True)
        results.append((scenario["name"], passed, detail))
    return results


def eval_notes_scenarios(llm, persona_text: str) -> list[tuple]:
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
    args = parser.parse_args()

    persona_text = load_persona()
    llm = create_llm(provider=args.provider, model=args.model, temperature=0.0)

    all_results = []
    if args.source in ("email", "all"):
        all_results.append(("email", eval_email_scenarios(llm, persona_text)))
    if args.source in ("calendar", "all"):
        all_results.append(("calendar", eval_calendar_scenarios(llm, persona_text)))
    if args.source in ("notes", "all"):
        all_results.append(("notes", eval_notes_scenarios(llm, persona_text)))

    total_pass = total = 0
    for source, results in all_results:
        print_results(source, results)
        total_pass += sum(1 for _, passed, _ in results if passed)
        total += len(results)

    print(f"\n{'─' * 40}\n{total_pass}/{total} scenarios passed\n")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
