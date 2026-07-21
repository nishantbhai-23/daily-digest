"""
Persona-Aware vs. Persona-Free MAP Prompt Eval
==================================================
Compares build_map_system_prompt(use_persona=True) against
build_map_system_prompt(use_persona=False) across every source's golden
MAP scenarios — the toggle wired into all three agents' MAP prompts
(tenant_config's "use_persona_in_map"), never actually measured before:
does injecting the operator's persona/priority list into MAP change what
gets extracted, or just how it's labeled?

Two things get compared, per scenario, per variant:
  - Keyword-coverage pass/fail (eval_map.score_scenario, reused as-is) —
    does turning persona off drop real signal, or let more noise through?
    Persona-free schemas (MAP_SCHEMA_NO_PERSONA) omit the priority field
    entirely, but none of EMAIL/CALENDAR/NOTES_MAP_SCENARIOS'
    required_keywords/forbidden_keywords reference priority literals —
    that's PRIORITY_CALIBRATION_SCENARIOS' job (a separate,
    schema-dependent question eval_prompt_variants.py already answers for
    triage's zero-shot prompt; this script doesn't attempt it since
    MAP_SCHEMA_NO_PERSONA has no priority field to calibrate at all).
  - Optional (--llm-judge): quality_judge's groundedness/completeness/
    conciseness/coherence, run on both variants' output. Persona-agnostic
    by construction (judges source_text vs. output_text only), so directly
    comparable — persisted separately from eval_map.py's own
    map_quality_judge records (variant here is "persona"/"no_persona",
    not provider/model, so the two scripts' records don't collide or
    overwrite each other in eval_history).

recruiter_noise_suppression (email, forbidden_keywords) is the single most
informative scenario here: persona=True suppresses noise "per the
profile," persona=False suppresses it via a generic "never actionable
regardless of recipient" rule — same intended behavior, different
justification. If one mode lets recruiter noise through and the other
doesn't, that's a real functional difference, not just a labeling one.

Not run automatically — calls a real model, 2 variants x 8 scenarios
(+ 1 quality-judge call per scenario per variant with --llm-judge).

Usage:
    python3 -m digest.eval.eval_persona_map --provider deepseek --model deepseek-chat
    python3 -m digest.eval.eval_persona_map --provider deepseek --model deepseek-chat --llm-judge
    python3 -m digest.eval.eval_persona_map --source email
"""

import argparse

from digest.agents import calendar_agent
from digest.agents import notes_agent
from digest.agents import triage_agent
from digest.parsers.calendar_parser import group_by_date as group_calendar_by_date
from digest.parsers.calendar_parser import load_calendar
from digest.eval.digest_checks import check_extraction_bloat, dynamic_bloat_ceiling, extract_searchable_text
from digest.parsers.email_parser import group_by_date as group_email_by_date
from digest.parsers.email_parser import load_inbox
from digest.eval.eval_history import record_eval_run
from digest.eval.eval_map import (
    CALENDAR_BLOAT_FLOOR,
    CALENDAR_BLOAT_MULTIPLIER,
    EMAIL_BLOAT_FLOOR,
    EMAIL_BLOAT_MULTIPLIER,
    NOTES_MAX_ITEMS,
    _discretize_groundedness,
    score_scenario,
)
from digest.eval.golden_scenarios import (
    CALENDAR_MAP_SCENARIOS,
    EMAIL_MAP_SCENARIOS,
    NOTES_MAP_SCENARIOS,
)
from digest.core.llm import create_llm
from digest.parsers.notes_parser import load_notes
from digest.core.persona import load_persona
from digest.eval.quality_judge import build_quality_judge_prompt, judge_map_quality

VARIANTS = ("persona", "no_persona")


def _judge(llm, source_items: list[dict], entry: dict) -> dict:
    source_text = " ".join(extract_searchable_text(item) for item in source_items)
    output_text = extract_searchable_text(entry.get("delta", {}))
    return judge_map_quality(llm, source_text, output_text)


def _run_email(llm, persona_text: str, use_persona: bool, llm_judge: bool) -> list[dict]:
    prompt = triage_agent.build_map_system_prompt(persona_text, use_persona=use_persona)
    schema = triage_agent.MAP_SCHEMA if use_persona else triage_agent.MAP_SCHEMA_NO_PERSONA
    by_day = group_email_by_date(load_inbox(triage_agent.INBOX_DIR))

    results = []
    for scenario in EMAIL_MAP_SCENARIOS:
        batch = by_day.get(scenario["day"], [])
        if not batch:
            results.append({"name": scenario["name"], "passed": False, "detail": "no emails found — regenerate data?", "quality": None})
            continue
        entry = triage_agent._map_single_day(llm, scenario["day"], batch, prompt, schema, map_variant="persona" if use_persona else "no_persona")
        if entry is None:
            results.append({"name": scenario["name"], "passed": False, "detail": "MAP call failed after retries", "quality": None})
            continue
        max_items = dynamic_bloat_ceiling(len(batch), EMAIL_BLOAT_FLOOR, EMAIL_BLOAT_MULTIPLIER)
        bloat = check_extraction_bloat(entry["delta"], max_items)
        if bloat:
            print(f"      ⚠️  {scenario['name']}: {bloat}")
        passed, detail = score_scenario(entry, scenario, include_stats=False)
        quality = _judge(llm, batch, entry) if llm_judge else None
        results.append({"name": scenario["name"], "passed": passed, "detail": detail, "quality": quality})
    return results


def _run_calendar(llm, persona_text: str, use_persona: bool, llm_judge: bool) -> list[dict]:
    prompt = calendar_agent.build_map_system_prompt(persona_text, use_persona=use_persona)
    by_day = group_calendar_by_date(load_calendar(calendar_agent.CALENDAR_FILE))

    results = []
    for scenario in CALENDAR_MAP_SCENARIOS:
        batch = by_day.get(scenario["day"], [])
        if not batch:
            results.append({"name": scenario["name"], "passed": False, "detail": "no events found — regenerate data?", "quality": None})
            continue
        entry = calendar_agent._map_single_day(llm, scenario["day"], batch, prompt, map_variant="persona" if use_persona else "no_persona")
        if entry is None:
            results.append({"name": scenario["name"], "passed": False, "detail": "MAP call failed after retries", "quality": None})
            continue
        max_items = dynamic_bloat_ceiling(len(batch), CALENDAR_BLOAT_FLOOR, CALENDAR_BLOAT_MULTIPLIER)
        bloat = check_extraction_bloat(entry["delta"], max_items)
        if bloat:
            print(f"      ⚠️  {scenario['name']}: {bloat}")
        passed, detail = score_scenario(entry, scenario, include_stats=True)
        quality = _judge(llm, batch, entry) if llm_judge else None
        results.append({"name": scenario["name"], "passed": passed, "detail": detail, "quality": quality})
    return results


def _run_notes(llm, persona_text: str, use_persona: bool, llm_judge: bool) -> list[dict]:
    prompt = notes_agent.build_map_system_prompt(persona_text, use_persona=use_persona)
    schema = notes_agent.MAP_SCHEMA if use_persona else notes_agent.MAP_SCHEMA_NO_PERSONA
    notes_by_id = {n["note_id"]: n for n in load_notes(notes_agent.NOTES_DIR)}

    results = []
    for scenario in NOTES_MAP_SCENARIOS:
        note = notes_by_id.get(scenario["note_id"])
        if note is None:
            results.append({"name": scenario["name"], "passed": False, "detail": "note not found — regenerate data?", "quality": None})
            continue
        entry = notes_agent._map_single_note(llm, note, prompt, schema, map_variant="persona" if use_persona else "no_persona")
        if entry is None:
            results.append({"name": scenario["name"], "passed": False, "detail": "MAP call failed after retries", "quality": None})
            continue
        bloat = check_extraction_bloat(entry["delta"], NOTES_MAX_ITEMS)
        if bloat:
            print(f"      ⚠️  {scenario['name']}: {bloat}")
        passed, detail = score_scenario(entry, scenario, include_stats=True)
        quality = _judge(llm, [note], entry) if llm_judge else None
        results.append({"name": scenario["name"], "passed": passed, "detail": detail, "quality": quality})
    return results


_RUNNERS = {"email": _run_email, "calendar": _run_calendar, "notes": _run_notes}


def run_comparison(llm, persona_text: str, source: str, llm_judge: bool) -> dict:
    """Runs both variants against every requested source's golden
    scenarios. Returns {variant: {source: [result_dict, ...]}}.
    """
    sources = list(_RUNNERS.keys()) if source == "all" else [source]
    results = {variant: {} for variant in VARIANTS}
    for use_persona, variant in ((True, "persona"), (False, "no_persona")):
        print(f"\n--- {variant} ---")
        for src in sources:
            print(f"   {src}:")
            src_results = _RUNNERS[src](llm, persona_text, use_persona, llm_judge)
            for r in src_results:
                icon = "✅" if r["passed"] else "❌"
                print(f"      {icon} {r['name']} — {r['detail']}")
                if r["quality"]:
                    q = r["quality"]
                    g, c, cc, ct = q["groundedness"], q["completeness"], q["conciseness"], q["coherence_tone"]
                    print(
                        f"         🔎 groundedness: {g['score']} "
                        f"({'unverified: ' + str(g['unverified_claims']) if g['unverified_claims'] else 'clean'}) "
                        f"| completeness gaps: {c['gaps']} | conciseness: {cc['verdict']} ({cc['ratio']}) "
                        f"| coherence/tone: {ct['score']}/5"
                    )
            results[variant][src] = src_results
    return results


def _record_coverage(results: dict, provider: str, model: str) -> dict:
    """Keyword-coverage pass/fail, persisted per variant across all
    sources' scenarios combined into one eval_history record each — same
    scenario_results shape eval_cross_reference_variants.py already uses
    ({"expected": "OK", "results": ["OK" or detail]}).
    """
    prompt_text = (
        "triage_agent/calendar_agent/notes_agent.build_map_system_prompt(use_persona=True/False) "
        "— see each agent module for the exact rendered text; not snapshotted as one string here "
        "since three separate prompts (one per source) are being compared as a single variant."
    )
    records = {}
    for variant, by_source in results.items():
        scenario_results = {
            r["name"]: {"expected": "OK", "results": ["OK" if r["passed"] else r["detail"]]}
            for src_results in by_source.values()
            for r in src_results
        }
        records[variant] = record_eval_run(
            eval_name="map_persona_ablation",
            variant=variant,
            prompt_text=prompt_text,
            provider=provider,
            model=model,
            trials_per_scenario=1,
            scenario_results=scenario_results,
        )
    return records


def _record_quality(results: dict, provider: str, model: str) -> dict | None:
    """Discretized groundedness, persisted per variant — mirrors
    eval_map.py's _run_quality_judge persistence shape exactly, but keyed
    so a "persona" vs "no_persona" run never collides with eval_map.py's
    own map_quality_judge records (which key variant by provider/model,
    not by persona mode).
    """
    has_quality = any(r["quality"] for by_source in results.values() for src_results in by_source.values() for r in src_results)
    if not has_quality:
        return None

    records = {}
    for variant, by_source in results.items():
        scenario_results = {
            r["name"]: {"expected": "fully_grounded", "results": [_discretize_groundedness(r["quality"])]}
            for src_results in by_source.values()
            for r in src_results
            if r["quality"]
        }
        if not scenario_results:
            continue
        records[variant] = record_eval_run(
            eval_name="map_persona_quality_judge",
            variant=f"{variant}/{provider}/{model}",
            prompt_text=build_quality_judge_prompt(),
            provider=provider,
            model=model,
            trials_per_scenario=1,
            scenario_results=scenario_results,
        )
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare persona-aware vs. persona-free MAP prompts across all three agents' golden scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.eval.eval_persona_map --provider deepseek --model deepseek-chat\n"
            "  python3 -m digest.eval.eval_persona_map --provider deepseek --model deepseek-chat --llm-judge\n"
            "  python3 -m digest.eval.eval_persona_map --source email\n"
        ),
    )
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--source", choices=["email", "calendar", "notes", "all"], default="all")
    parser.add_argument("--llm-judge", action="store_true", help="Also run the in-house quality judge on both variants' output — costs one extra LLM call per scenario per variant")
    return parser.parse_args()


def main():
    args = parse_args()
    persona_text = load_persona()
    llm = create_llm(provider=args.provider, model=args.model, temperature=0.0)

    total_scenarios = len(EMAIL_MAP_SCENARIOS) + len(CALENDAR_MAP_SCENARIOS) + len(NOTES_MAP_SCENARIOS)
    scenario_count = total_scenarios if args.source == "all" else len(
        {"email": EMAIL_MAP_SCENARIOS, "calendar": CALENDAR_MAP_SCENARIOS, "notes": NOTES_MAP_SCENARIOS}[args.source]
    )
    multiplier = 2 if args.llm_judge else 1
    print(f"🤖 {args.provider}/{args.model} — 2 variants x {scenario_count} scenario(s) x {multiplier} call(s) each")

    results = run_comparison(llm, persona_text, args.source, args.llm_judge)

    print(f"\n{'─' * 60}\n   Summary — keyword coverage\n{'─' * 60}")
    coverage_records = _record_coverage(results, args.provider, args.model)
    for variant, record in coverage_records.items():
        print(f"   {variant:12s} scenario accuracy: {record['aggregate_accuracy']:.0%}  (saved to eval_history/results.jsonl)")

    delta = coverage_records["persona"]["aggregate_accuracy"] - coverage_records["no_persona"]["aggregate_accuracy"]
    print(f"\n   persona vs no_persona coverage accuracy: {delta:+.0%}")
    if abs(delta) < 1e-9:
        print("   → No coverage difference — persona doesn't change whether the required signal survives extraction on this data.")
    elif delta > 0:
        print("   → persona-aware MAP retains more of the expected signal. Losing persona at MAP time costs real recall here.")
    else:
        print("   → persona-free MAP retained more of the expected signal — worth checking which scenario regressed and why.")

    if args.llm_judge:
        print(f"\n{'─' * 60}\n   Summary — quality judge (groundedness)\n{'─' * 60}")
        quality_records = _record_quality(results, args.provider, args.model)
        if quality_records:
            for variant, record in quality_records.items():
                print(f"   {variant:12s} fully-grounded rate: {record['aggregate_accuracy']:.0%}  (saved to eval_history/results.jsonl)")
            if "persona" in quality_records and "no_persona" in quality_records:
                g_delta = quality_records["persona"]["aggregate_accuracy"] - quality_records["no_persona"]["aggregate_accuracy"]
                print(f"\n   persona vs no_persona fully-grounded rate: {g_delta:+.0%}")
                print("   Reminder: read the per-scenario completeness/conciseness/coherence output above by hand — "
                      "groundedness is the only dimension persisted for trend-tracking, the rest is advisory.")


if __name__ == "__main__":
    main()
