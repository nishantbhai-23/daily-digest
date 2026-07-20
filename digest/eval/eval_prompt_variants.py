"""
Prompt Variant Comparison Eval — Few-Shot vs. Zero-Shot MAP
===============================================================
Generates real evidence on whether few-shot examples improve MAP-phase
priority-calibration consistency (critique #1 from a prompt design review),
rather than adding them permanently on priors. Every other fix from that
review traced to something that actually broke; this one didn't, so instead
of skipping it or adding it speculatively, this measures it.

For each PRIORITY_CALIBRATION_SCENARIOS entry, runs the same day's email
batch through two MAP prompt variants N times each:
  - zero_shot: triage_agent.build_map_system_prompt as-is.
  - few_shot_v1: the same prompt, plus one worked example (a fictional
    investor-lead email, correctly tagged P0) appended as a genuine
    user/assistant turn pair before the real data — not inlined into the
    system prompt, per the reviewed critique's own proposed pattern. The
    example is fictional and doesn't overlap with any tested scenario, so
    this measures whether the *rule* generalizes, not whether the model
    memorized an answer.

The few-shot variant is constructed here only — not added to
triage_agent.py — since whether it's worth keeping permanently is exactly
what this script determines. Every run's prompt text and per-scenario
results are persisted via eval_history.record_eval_run, not just printed.

Not run automatically — calls a real model. Costs real provider budget
(default 3 scenarios × 2 variants × 5 trials = 30 calls).

Usage:
    python3 -m digest.eval.eval_prompt_variants --provider deepseek --model deepseek-chat
    python3 -m digest.eval.eval_prompt_variants --provider deepseek --model deepseek-chat --trials 1  # cheap dry run
"""

import argparse
import json

from digest.agents import triage_agent
from digest.core.ledger import validate_schema
from digest.core.llm import call_with_retry, create_llm
from digest.core.persona import load_persona
from digest.eval.eval_history import record_eval_run
from digest.eval.golden_scenarios import PRIORITY_CALIBRATION_SCENARIOS
from digest.parsers.email_parser import group_by_date, load_inbox

# A fictional worked example — deliberately not any real contact in this
# corpus, and not the same relationship-type as more than one tested
# scenario, so this measures whether the *rule* ("read the profile's
# priority list, apply it, override on content") generalizes to the real
# held-out scenarios rather than the model having memorized a leaked answer.
_FEW_SHOT_USER = (
    "Emails for 2026-05-01:\n\n"
    "--- Email 1 ---\n"
    "From: Priya Chen <priya.chen@meridiancapital.vc>\n"
    "Subject: Term sheet redline — can we talk today?\n"
    "Body: Avery, our legal team sent back a redline on the term sheet. "
    "I'd like to walk through it with you before end of day if possible — "
    "a couple of points need your input directly.\n"
)
_FEW_SHOT_ASSISTANT = json.dumps(
    {
        "deadlines": [],
        "decisions": [],
        "action_items": [
            {
                "description": "Review term sheet redline and talk with Priya Chen before end of day",
                "owner": "Avery Chen",
                "source_subject": "Term sheet redline — can we talk today?",
                "priority": "P0",
            }
        ],
        "thread_progressions": [],
    },
    indent=2,
)


def _run_map_call(llm, day: str, batch: list[dict], system_prompt: str, extra_messages: list[dict] | None = None) -> dict | None:
    """One MAP call returning the raw delta dict, or None on failure.

    A minimal parallel to triage_agent._map_single_day — reuses the same
    formatting/validation/retry building blocks, extended only to support
    injecting few-shot turns between the system prompt and the real data
    (which _map_single_day's fixed 2-message shape doesn't support).
    """
    context = triage_agent.format_email_batch(batch)
    messages = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": f"Emails for {day}:\n\n{context}"})

    def _call():
        delta = llm.chat_json(messages=messages)
        errors = validate_schema(delta, triage_agent.MAP_SCHEMA)
        if errors:
            raise ValueError(f"Invalid MAP output: {errors}")
        return delta

    try:
        return call_with_retry(_call)
    except Exception as e:
        print(f"      ⚠️  call failed: {e}")
        return None


def _find_priority(delta: dict, match_on: str) -> str | None:
    """Search deadlines/decisions/action_items (the categories that carry a
    priority field — thread_progressions doesn't) for an item mentioning
    match_on, case-insensitive. Returns its priority, or None if nothing in
    this run's output mentioned the target at all — itself informative,
    logged as "not_found" by the caller rather than crashing.
    """
    match_lower = match_on.lower()
    for category in ("deadlines", "decisions", "action_items"):
        for item in delta.get(category, []):
            haystack = " ".join(str(v) for v in item.values()).lower()
            if match_lower in haystack:
                return item.get("priority", "not_found")
    return None


def run_comparison(llm, persona_text: str, trials: int) -> dict:
    """Runs both variants against every scenario. Returns
    {variant: {scenario_name: {"expected": str, "results": [str, ...]}}}.
    """
    zero_shot_prompt = triage_agent.build_map_system_prompt(persona_text)
    few_shot_prompt = zero_shot_prompt  # same base prompt; the example is injected as extra turns, not prompt text
    few_shot_turns = [
        {"role": "user", "content": _FEW_SHOT_USER},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
    ]

    by_day = group_by_date(load_inbox(triage_agent.INBOX_DIR))

    results = {"zero_shot": {}, "few_shot_v1": {}}
    for scenario in PRIORITY_CALIBRATION_SCENARIOS:
        batch = by_day.get(scenario["day"], [])
        if not batch:
            print(f"   ⚠️  {scenario['name']}: no emails found for {scenario['day']} — skipping")
            continue

        print(f"\n   {scenario['name']} (day {scenario['day']}, expect {scenario['expected_priority']} for '{scenario['match_on']}')")

        for variant, prompt, extra in (
            ("zero_shot", zero_shot_prompt, None),
            ("few_shot_v1", few_shot_prompt, few_shot_turns),
        ):
            outcomes = []
            for trial in range(trials):
                delta = _run_map_call(llm, scenario["day"], batch, prompt, extra)
                priority = _find_priority(delta, scenario["match_on"]) if delta is not None else "call_failed"
                outcomes.append(priority or "not_found")
            correct = sum(1 for o in outcomes if o == scenario["expected_priority"])
            print(f"      {variant:12s}: {outcomes}  ({correct}/{trials} correct)")
            results[variant][scenario["name"]] = {
                "expected": scenario["expected_priority"],
                "results": outcomes,
            }

    return results, zero_shot_prompt, few_shot_prompt + "\n\n[+ few-shot example turns — see _FEW_SHOT_USER/_FEW_SHOT_ASSISTANT in this script]"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare zero-shot vs. few-shot MAP prompts on priority-calibration accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.eval.eval_prompt_variants --provider deepseek --model deepseek-chat\n"
            "  python3 -m digest.eval.eval_prompt_variants --provider deepseek --model deepseek-chat --trials 1\n"
        ),
    )
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.7, help="Nonzero by default — this eval measures run-to-run consistency, which temperature 0 would trivially hide")
    parser.add_argument("--trials", type=int, default=5, help="Runs per scenario per variant (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    persona_text = load_persona()
    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    total_calls = len(PRIORITY_CALIBRATION_SCENARIOS) * 2 * args.trials
    print(f"🤖 {args.provider}/{args.model}, temperature={args.temperature}, {args.trials} trials/variant/scenario")
    print(f"   Up to {total_calls} real calls total.\n")

    results, zero_shot_prompt, few_shot_prompt = run_comparison(llm, persona_text, args.trials)

    print(f"\n{'─' * 60}\n   Summary\n{'─' * 60}")
    records = {}
    for variant, prompt_text in (("zero_shot", zero_shot_prompt), ("few_shot_v1", few_shot_prompt)):
        record = record_eval_run(
            eval_name="map_priority_calibration",
            variant=variant,
            prompt_text=prompt_text,
            provider=args.provider,
            model=args.model,
            trials_per_scenario=args.trials,
            scenario_results=results[variant],
        )
        records[variant] = record
        print(f"   {variant:12s} aggregate accuracy: {record['aggregate_accuracy']:.0%}  (saved to eval_history/results.jsonl)")

    delta = records["few_shot_v1"]["aggregate_accuracy"] - records["zero_shot"]["aggregate_accuracy"]
    print(f"\n   few_shot_v1 vs zero_shot: {delta:+.0%}")
    if abs(delta) < 0.10:
        print("   → No meaningful difference at this sample size. Recommendation: leave zero-shot as-is; document why not.")
    elif delta > 0:
        print("   → Few-shot measurably improved accuracy. Recommendation: consider merging into build_map_system_prompt.")
    else:
        print("   → Few-shot measurably hurt accuracy. Recommendation: do not adopt it.")


if __name__ == "__main__":
    main()
