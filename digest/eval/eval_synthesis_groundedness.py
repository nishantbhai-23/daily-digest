"""
Stage-2 Synthesis Groundedness Eval
=======================================
Closes the biggest of the three hallucination-detection gaps found while
auditing this codebase's eval coverage: nothing checked the actual
synthesized daily_brief.md — the real deliverable — for groundedness at
all. quality_judge.py only runs on MAP output (one day's raw source vs.
its extraction); eval_synthesis_variants.py only measures coverage
(was a P0 task mentioned), never fabrication.

Reuses citations.py's already-proven grounding wholesale
(load_citable_sources + cite_brief) rather than building a second
dedicated LLM-judge pipeline — cheaper, no new LLM-call machinery, same
"ship the coarse deterministic signal first" pattern citations' own
grounding was itself built with.

IMPORTANT CAVEAT, read before trusting a low score: this is a
citation-coverage proxy, not a literal fact-verification score.
`groundedness = 1 - (uncited bullets / total bullets)`. An uncited bullet
is one none of citations.py's three tiers (keyword, embedding, LLM judge)
could confidently attribute to a single source — which is NOT the same
claim as "this bullet is false." A legitimate cross-source synthesis
bullet (e.g. "ARC-102 appears in 5 sources...") can end up uncited for
structural reasons — no single source attribution makes sense for a
claim that's genuinely about the aggregate — just as easily as a
fabricated one can. A low score means "go read this brief by hand," not
"this brief is definitely wrong." Deliberately blunt, same posture
digest_checks.py's other checks already take.

Usage:
    python3 -m digest.eval.eval_synthesis_groundedness --tenant arclight --provider deepseek --model deepseek-chat
    python3 -m digest.eval.eval_synthesis_groundedness --tenant arclight --reuse-existing-brief  # cheap re-score, no Stage 2 regeneration

This calls a real model — it is NOT run automatically. Regenerating the
brief (the default) costs a full orchestrator pipeline run (Stage 1-3);
--reuse-existing-brief skips that and only costs the citation-judge call,
but is only meaningful if the existing daily_brief.md is still current
relative to the ledgers being scored against.
"""

import argparse
import os

from digest.core import tenant_paths
from digest.core.citations import cite_brief, load_citable_sources
from digest.core.llm import create_llm
from digest.eval.eval_history import record_eval_run
from digest import orchestrator

# Starting point, not live-calibrated yet — a coarse sanity floor pending
# a real run's actual distribution. Half the bullets grounded is a low
# bar deliberately: this check's job is catching a brief that's mostly
# fabricated, not judging borderline cases (see module docstring's
# caveat on precision).
_GROUNDEDNESS_FLOOR = 0.5


def groundedness_ratio(cite_stats: dict) -> float:
    """Pure ratio computation given cite_brief's own stats dict
    ({"cited_keyword", "cited_embedding", "cited_llm", "uncited"}) — no
    LLM call, no file I/O, trivially unit-testable in isolation from the
    rest of this script.

    Returns 1.0 (vacuously fully grounded) for a brief with zero citable
    bullets — nothing to be wrong about.
    """
    total = sum(cite_stats.values())
    if total == 0:
        return 1.0
    return 1 - (cite_stats["uncited"] / total)


def run_groundedness_check(tenant_id: str, provider: str, model: str, reuse_existing_brief: bool = False) -> dict:
    """Regenerates the tenant's daily_brief.md via the real orchestrator
    pipeline (unless reuse_existing_brief and one already exists), then
    scores it via citations.cite_brief.

    Returns:
        {"groundedness": float, "cite_stats": {...}}
    """
    paths = tenant_paths.for_tenant(tenant_id)

    if reuse_existing_brief and os.path.exists(paths.brief_file):
        print(f"Reusing existing brief at {paths.brief_file} (not regenerating).")
    else:
        print("Generating a fresh brief via the real orchestrator pipeline (Stage 0-3)...")
        orchestrator.run_for_tenant(tenant_id, provider, model)

    with open(paths.brief_file, "r", encoding="utf-8") as f:
        brief_text = f.read()

    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    llm = create_llm(provider=provider, model=model, temperature=0.0, tenant_id=tenant_id)
    _, cite_stats = cite_brief(brief_text, sources, llm=llm)

    return {"groundedness": groundedness_ratio(cite_stats), "cite_stats": cite_stats}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a tenant's synthesized daily_brief.md for groundedness, reusing citations.py's own grounding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.eval.eval_synthesis_groundedness --tenant arclight --provider deepseek --model deepseek-chat\n"
            "  python3 -m digest.eval.eval_synthesis_groundedness --tenant arclight --reuse-existing-brief\n"
        ),
    )
    parser.add_argument("--tenant", default=tenant_paths.DEFAULT_TENANT)
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--reuse-existing-brief", action="store_true",
        help="Score the tenant's existing daily_brief.md instead of regenerating it — cheap, but only meaningful if that brief is still current relative to the ledgers scored against.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_groundedness_check(args.tenant, args.provider, args.model, args.reuse_existing_brief)
    stats = result["cite_stats"]
    total = sum(stats.values())

    print(f"\n{'─' * 60}\n   Groundedness: {result['groundedness']:.0%} ({stats['uncited']} of {total} bullets uncited)\n{'─' * 60}")
    print(
        f"   cited by keyword: {stats['cited_keyword']}  cited by embedding: {stats['cited_embedding']}  "
        f"cited by LLM: {stats['cited_llm']}  uncited: {stats['uncited']}"
    )
    print(
        "   Reminder: uncited means 'couldn't be attributed to a single source,' not 'false' — "
        "spot-check the uncited bullets by hand before treating a low score as a real hallucination finding."
    )

    outcome = "OK" if result["groundedness"] >= _GROUNDEDNESS_FLOOR else f"below floor: {result['groundedness']:.2f}"
    record_eval_run(
        eval_name="synthesis_groundedness",
        variant=f"{args.provider}/{args.model}",
        prompt_text="(reuses citations.py's own grounding — no separate judge prompt)",
        provider=args.provider,
        model=args.model,
        trials_per_scenario=1,
        scenario_results={args.tenant: {"expected": "OK", "results": [outcome]}},
    )
    print(f"   (saved to eval_history/results.jsonl)")


if __name__ == "__main__":
    main()
