"""
Synthesis Variant Comparison Eval — Single-Call vs. Staged Stage 2
======================================================================
Compares orchestrator.py's two Stage-2 implementations against a tenant's
real ledger data: synthesize_brief_single_call (one LLM call producing all
three outputs) vs synthesize_brief_staged (two dependent calls — narrative,
then dispatchable items informed by it). Requested directly rather than
picked on priors, same as the MAP few-shot comparison
(eval_prompt_variants.py) — both variants stay in the codebase until this
settles which one wins.

Metric: orchestrator.check_priority_coverage's own deterministic P0/P1
coverage count — not a scenario-keyed expected answer (Stage 2 operates
over a tenant's whole ledger, not one day's batch, so there's no single
"correct" digest to compare against), but "did this variant surface every
task_signals already knows is P0/P1" is a real, objective, LLM-free signal,
reused directly from the same check that runs in production. Each trial's
result is recorded via eval_history's existing scenario_results shape
unmodified (expected="0", results=[str(missing_count), ...]) — "accuracy"
there naturally reads as "fraction of trials with full P0/P1 coverage."

Also reports (not scored, just surfaced): dispatchable-item count and
wall-clock latency per variant/trial — the real cost tradeoff (staged is 2
calls, single_call is 1) that motivated splitting Stage 2 in the first
place, surfaced plainly rather than hidden behind the coverage score alone.

Stage 0 (cross-reference index) and Stage 1 (contradiction detection) run
ONCE per tenant, not once per trial — neither depends on which Stage-2
variant is under test, and re-running them per trial would make this
slower and less apples-to-apples, not more rigorous.

Not run automatically — calls a real model repeatedly. Costs real provider
budget (default trials=5 x 2 variants = 10 Stage-2 runs, each 1 or 2 calls,
plus one shared Stage-0/1 pass).

Usage:
    python3 -m digest.eval.eval_synthesis_variants --tenant default --provider deepseek --model deepseek-chat
    python3 -m digest.eval.eval_synthesis_variants --trials 1  # cheap dry run
"""

import argparse
import time

from digest import orchestrator
from digest.core import tenant_paths
from digest.core.ledger import check_data_freshness, load_ledger
from digest.core.llm import create_llm
from digest.core.persona import load_persona
from digest.core.tasks_signals import compute_task_signals
from digest.eval.eval_history import record_eval_run
from digest.parsers.tasks_parser import load_tasks

PROMPT_TEXT_BY_VARIANT = {
    "single_call": lambda persona_text: orchestrator.build_synthesis_prompt(persona_text),
    "staged": lambda persona_text: (
        orchestrator.build_narrative_prompt(persona_text)
        + "\n\n=== STAGE 2b: DISPATCHABLE ITEMS ===\n\n"
        + orchestrator.build_dispatchable_items_prompt(persona_text)
    ),
}


def run_comparison(llm, tenant_id: str, persona_text: str, trials: int) -> dict:
    """Runs both Stage-2 variants `trials` times each against one tenant's
    real ledger data. Returns {variant: {"missing_counts": [...],
    "dispatchable_counts": [...], "latencies": [...]}}.
    """
    paths = tenant_paths.for_tenant(tenant_id)
    email_ledger, _ = load_ledger(paths.email_ledger_file)
    calendar_ledger, _ = load_ledger(paths.calendar_ledger_file)
    notes_ledger, _ = load_ledger(paths.notes_ledger_file)
    tasks = load_tasks(paths.tasks_file)
    task_signals = compute_task_signals(tasks)

    if not (email_ledger or calendar_ledger or notes_ledger):
        raise SystemExit(f"No source ledgers found for tenant {tenant_id!r}. Run the agents first.")

    print("🔎 Stage 0 — deterministic cross-reference index + freshness check (once, shared across variants)...")
    cross_ref_index = orchestrator.build_cross_reference_index(email_ledger, calendar_ledger, notes_ledger, task_signals)
    freshness = check_data_freshness({"email": email_ledger, "calendar": calendar_ledger, "notes": notes_ledger})

    print("🔍 Stage 1 — contradiction detection (once, shared across variants)...")
    contradictions = orchestrator.detect_contradictions(llm, cross_ref_index, persona_text)
    print(f"   {len(contradictions.get('contradictions', []))} contradiction(s) found.\n")

    results = {}
    for variant_name, synthesize_fn in orchestrator.SYNTHESIS_VARIANTS.items():
        print(f"--- {variant_name} ---")
        missing_counts, dispatchable_counts, latencies = [], [], []
        for trial in range(trials):
            start = time.time()
            synthesis = synthesize_fn(
                llm, email_ledger, calendar_ledger, notes_ledger, task_signals,
                cross_ref_index, contradictions, freshness, persona_text,
            )
            latency = time.time() - start
            missing = orchestrator.check_priority_coverage(synthesis, task_signals)
            dispatchable_count = len(synthesis.get("dispatchable_items", []))
            missing_counts.append(len(missing))
            dispatchable_counts.append(dispatchable_count)
            latencies.append(latency)
            print(f"   trial {trial + 1}: {len(missing)} missing P0/P1 item(s), {dispatchable_count} dispatchable item(s), {latency:.1f}s")
        results[variant_name] = {
            "missing_counts": missing_counts,
            "dispatchable_counts": dispatchable_counts,
            "latencies": latencies,
        }

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare single_call vs. staged Stage-2 synthesis implementations on P0/P1 coverage, dispatchable-item count, and latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.eval.eval_synthesis_variants --provider deepseek --model deepseek-chat\n"
            "  python3 -m digest.eval.eval_synthesis_variants --trials 1  # cheap dry run\n"
        ),
    )
    parser.add_argument("--tenant", default=tenant_paths.DEFAULT_TENANT)
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.7, help="Nonzero by default — this eval measures run-to-run consistency, which temperature 0 would trivially hide")
    parser.add_argument("--trials", type=int, default=5, help="Runs per variant (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = tenant_paths.for_tenant(args.tenant)
    persona_text = load_persona(paths.persona_file)
    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature, tenant_id=paths.tenant_id)

    print(f"🤖 {args.provider}/{args.model}, temperature={args.temperature}, {args.trials} trials/variant")
    print(f"   Tenant: {args.tenant} — up to {2 * args.trials} Stage-2 runs (single_call=1 call each, staged=2 calls each) plus one shared Stage-0/1 pass.\n")

    results = run_comparison(llm, args.tenant, persona_text, args.trials)

    print(f"\n{'─' * 60}\n   Summary\n{'─' * 60}")
    records = {}
    for variant_name, data in results.items():
        missing_counts = data["missing_counts"]
        dispatchable_counts = data["dispatchable_counts"]
        latencies = data["latencies"]

        record = record_eval_run(
            eval_name="synthesis_variant_comparison",
            variant=variant_name,
            prompt_text=PROMPT_TEXT_BY_VARIANT[variant_name](persona_text),
            provider=args.provider,
            model=args.model,
            trials_per_scenario=args.trials,
            scenario_results={
                args.tenant: {"expected": "0", "results": [str(c) for c in missing_counts]},
            },
        )
        records[variant_name] = record
        avg_dispatchable = sum(dispatchable_counts) / len(dispatchable_counts) if dispatchable_counts else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        print(
            f"   {variant_name:12s} coverage accuracy: {record['aggregate_accuracy']:.0%}  "
            f"avg dispatchable items: {avg_dispatchable:.1f}  avg latency: {avg_latency:.1f}s  "
            f"(saved to eval_history/results.jsonl)"
        )

    delta = records["staged"]["aggregate_accuracy"] - records["single_call"]["aggregate_accuracy"]
    print(f"\n   staged vs single_call coverage: {delta:+.0%}")
    if abs(delta) < 0.10:
        print("   → No meaningful coverage difference at this sample size. Weigh the latency/cost tradeoff (staged = 2 calls) directly.")
    elif delta > 0:
        print("   → staged measurably improved P0/P1 coverage — weigh against the latency delta above before adopting.")
    else:
        print("   → staged measurably hurt P0/P1 coverage relative to single_call. Prefer single_call unless another concern outweighs this.")


if __name__ == "__main__":
    main()
