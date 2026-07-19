"""
MAP-Phase Extraction Eval
============================
Runs MAP-phase extraction against the known planted scenarios in
golden_scenarios.py and checks whether the expected signal actually got
extracted — turns the manual ad hoc checks done throughout development
(grep-ing ledgers, one-off `python3 -c` snippets) into something
repeatable.

This calls a real model — it is NOT run automatically. Invoke explicitly
once a working LLM provider is available:

    python3 eval_map.py --provider anthropic --model claude-haiku-4-5
    python3 eval_map.py --provider deepseek --model deepseek-chat
    python3 eval_map.py --source email       # limit to one source
    python3 eval_map.py --provider ollama --model llama3   # slow; local
"""

import argparse
import json
import sys

from digest.agents import calendar_agent
from digest.agents import notes_agent
from digest.agents import triage_agent
from digest.parsers.calendar_parser import group_by_date as group_calendar_by_date
from digest.parsers.calendar_parser import load_calendar
from digest.eval.digest_checks import check_keywords_present
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
        missing = check_keywords_present(json.dumps(entry["delta"]), scenario["required_keywords"])
        results.append((scenario["name"], not missing, "OK" if not missing else f"missing: {missing}"))
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
        # Deterministic stats and the LLM delta both count as "extraction".
        haystack = json.dumps(entry["stats"]) + json.dumps(entry["delta"])
        missing = check_keywords_present(haystack, scenario["required_keywords"])
        results.append((scenario["name"], not missing, "OK" if not missing else f"missing: {missing}"))
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
        haystack = json.dumps(entry["stats"]) + json.dumps(entry["delta"])
        missing = check_keywords_present(haystack, scenario["required_keywords"])
        results.append((scenario["name"], not missing, "OK" if not missing else f"missing: {missing}"))
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
