"""
Cross-Reference Variant Comparison Eval — Lexical vs. Embedding-Assisted Stage 0
====================================================================================
Compares cross_reference.py's two Stage-0 implementations against a
tenant's real ledger data: build_cross_reference_index (keyword matching
only) vs. build_cross_reference_index_embedding_assisted (adds a local-
embedding rescue pass for paraphrase matches the keyword pass misses).
Both variants stay in the codebase — CROSS_REFERENCE_VARIANTS — until this
comparison gives a reason to change orchestrator.py's default
(--cross-reference-variant currently defaults to "lexical").

Unlike eval_synthesis_variants.py, neither variant here makes an LLM call
— cross-referencing is keyword matching plus, optionally, local
embeddings (digest.core.embeddings, via Ollama). No --provider/--model
flags, no real API budget spent, no "trials" in the run-to-run-consistency
sense (embeddings given the same input produce the same output — nothing
here samples). The one real dependency is Ollama actually running with an
embedding model pulled for the embedding_assisted variant to do anything
beyond its own lexical floor; if it isn't, build_cross_reference_index_embedding_assisted
degrades to a lexical-only result on its own (see its docstring), and
this script's comparison will correctly show no difference between the
two variants rather than erroring.

Metric: score_cross_reference_scenario against golden_scenarios.CROSS_REFERENCE_SCENARIOS
— for each scenario's task_id, does the variant's index include at least
`expected_sources` (a floor: a variant that finds a superset still
passes). Recorded via eval_history's existing scenario_results shape.

Also surfaced (not scored): total mention count per variant across the
whole tenant (not just golden-scenario tasks) — the real tradeoff being
measured here is precision risk (does the embedding tier ever add a
mention that isn't really the same task), not $ cost the way an LLM
judge's tradeoff is. A large jump in total mentions with no corresponding
golden-scenario improvement is worth eyeballing by hand before trusting.

Usage:
    python3 -m digest.eval.eval_cross_reference_variants --tenant arclight
"""

import argparse
import time

from digest.core import tenant_paths
from digest.core.cross_reference import CROSS_REFERENCE_VARIANTS
from digest.core.ledger import load_ledger
from digest.core.tasks_signals import compute_task_signals
from digest.eval.eval_history import record_eval_run
from digest.eval.golden_scenarios import CROSS_REFERENCE_SCENARIOS
from digest.parsers.tasks_parser import load_tasks


def score_cross_reference_scenario(index: dict, scenario: dict) -> tuple[bool, str]:
    """Pure, no-model-call scoring of one already-built cross-reference
    index against one golden_scenarios.py CROSS_REFERENCE_SCENARIOS entry
    — same split score_scenario (eval_map.py) and check_priority_coverage
    (orchestrator.py) already use their own metrics with.

    expected_sources is a floor, not an exact match: a variant reporting
    a superset of the expected sources still passes.

    Returns:
        (passed, detail) — detail is "OK" or a human-readable failure
        reason.
    """
    task_id = scenario["task_id"]
    if task_id not in index:
        return (False, f"task {task_id} not found in index at all")
    actual_sources = {m["source"] for m in index[task_id]["mentioned_in"]}
    missing = scenario["expected_sources"] - actual_sources
    if missing:
        return (False, f"missing source(s): {sorted(missing)} (found: {sorted(actual_sources)})")
    return (True, "OK")


def run_comparison(tenant_id: str) -> dict:
    """Runs both cross-reference variants once each against one tenant's
    real ledger data (no trials — see module docstring for why).

    Returns:
        {variant: {"scenario_results": {name: (passed, detail)},
                    "total_mentions": int, "latency": float}}
    """
    paths = tenant_paths.for_tenant(tenant_id)
    email_ledger, _ = load_ledger(paths.email_ledger_file)
    calendar_ledger, _ = load_ledger(paths.calendar_ledger_file)
    notes_ledger, _ = load_ledger(paths.notes_ledger_file)
    tasks = load_tasks(paths.tasks_file)
    task_signals = compute_task_signals(tasks)

    if not (email_ledger or calendar_ledger or notes_ledger):
        raise SystemExit(f"No source ledgers found for tenant {tenant_id!r}. Run the agents first.")

    scenarios = [s for s in CROSS_REFERENCE_SCENARIOS if s["tenant_id"] == tenant_id]
    if not scenarios:
        print(f"   ⚠️  No CROSS_REFERENCE_SCENARIOS entries for tenant {tenant_id!r} — scoring will be empty.")

    results = {}
    for variant_name, build_fn in CROSS_REFERENCE_VARIANTS.items():
        print(f"--- {variant_name} ---")
        start = time.time()
        index = build_fn(email_ledger, calendar_ledger, notes_ledger, task_signals)
        latency = time.time() - start

        scenario_results = {}
        for scenario in scenarios:
            passed, detail = score_cross_reference_scenario(index, scenario)
            scenario_results[scenario["name"]] = (passed, detail)
            icon = "✅" if passed else "❌"
            print(f"   {icon} {scenario['name']} — {detail}")

        total_mentions = sum(len(data["mentioned_in"]) for data in index.values())
        print(f"   {len(index)} flagged task(s) indexed, {total_mentions} total mention(s), {latency:.2f}s")

        results[variant_name] = {
            "scenario_results": scenario_results,
            "total_mentions": total_mentions,
            "latency": latency,
        }

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare lexical vs. embedding_assisted cross_reference.py Stage-0 implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 -m digest.eval.eval_cross_reference_variants --tenant arclight\n",
    )
    parser.add_argument("--tenant", default=tenant_paths.DEFAULT_TENANT)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Tenant: {args.tenant} — comparing {list(CROSS_REFERENCE_VARIANTS.keys())} (no LLM calls, local only)\n")

    results = run_comparison(args.tenant)

    print(f"\n{'─' * 60}\n   Summary\n{'─' * 60}")
    records = {}
    for variant_name, data in results.items():
        record = record_eval_run(
            eval_name="cross_reference_variant_comparison",
            variant=variant_name,
            prompt_text="(no LLM prompt — deterministic/local-embedding only)",
            provider="none",
            model="none",
            trials_per_scenario=1,
            scenario_results={
                name: {"expected": "OK", "results": ["OK" if passed else detail]}
                for name, (passed, detail) in data["scenario_results"].items()
            },
        )
        records[variant_name] = record
        print(
            f"   {variant_name:18s} scenario accuracy: {record['aggregate_accuracy']:.0%}  "
            f"total mentions: {data['total_mentions']}  latency: {data['latency']:.2f}s  "
            f"(saved to eval_history/results.jsonl)"
        )

    if "lexical" in records and "embedding_assisted" in records:
        delta = records["embedding_assisted"]["aggregate_accuracy"] - records["lexical"]["aggregate_accuracy"]
        mention_delta = results["embedding_assisted"]["total_mentions"] - results["lexical"]["total_mentions"]
        print(f"\n   embedding_assisted vs lexical scenario accuracy: {delta:+.0%}")
        print(f"   embedding_assisted found {mention_delta:+d} mention(s) relative to lexical across the whole tenant.")
        if mention_delta > 0:
            print(
                "   → embedding_assisted found real additional coverage. Before adopting it as default, "
                "spot-check a few of the added mentions by hand — the tradeoff here is precision risk, "
                "not $ cost, since this index directly seeds Stage 1's contradiction-detection candidates."
            )
        else:
            print("   → No additional coverage found on this tenant's data — not yet a reason to change the default.")


if __name__ == "__main__":
    main()
