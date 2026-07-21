"""
Eval History
=============
Persistent record of prompt-variant comparison runs — the "come back to
this later" store for critique #1 (few-shot) and any future prompt eval
(chain-of-thought, persona-in-user-vs-system, etc.). Same filesystem-as-
storage approach as the rest of this codebase (persona.py, ledger.py,
resilience.log_call_metrics's JSONL pattern).

Deliberately not under output/ (gitignored, regenerable) or data/
(tenant-scoped input) — eval_history/ is a durable record meant to be
committed and inspected later.

Two things persisted, not just printed and discarded:
- A human-readable snapshot of the exact rendered prompt text tested,
  since prompt-builder functions (build_map_system_prompt etc.) are code
  that keeps changing — a log entry that just says "few_shot_v1" is
  useless once that function has moved on. Content-addressed (by a short
  hash of the prompt text), so re-running the same variant never writes a
  duplicate file.
- One JSONL line per (eval run × variant), with per-scenario and aggregate
  accuracy — so "did few-shot's accuracy hold up after the model changed"
  is a query over history, not a lost one-off result.

Usage:
    from digest.eval.eval_history import record_eval_run, load_eval_history

    record_eval_run(
        eval_name="map_priority_calibration",
        variant="few_shot_v1",
        prompt_text=few_shot_prompt,
        provider="deepseek",
        model="deepseek-chat",
        trials_per_scenario=5,
        scenario_results={
            "quiet_marcus_investor_thread": {"expected": "P0", "results": ["P0", "P0", "P1", "P0", "P0"]},
        },
    )

    past_runs = load_eval_history(eval_name="map_priority_calibration", variant="few_shot_v1")
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone

EVAL_HISTORY_DIR = "eval_history"
RESULTS_FILE = os.path.join(EVAL_HISTORY_DIR, "results.jsonl")
PROMPTS_DIR = os.path.join(EVAL_HISTORY_DIR, "prompts")


def _snapshot_filename(eval_name: str, variant: str, prompt_text: str) -> str:
    """Content-addressed filename — re-running the same variant with
    identical prompt text always resolves to the same file, so
    record_eval_run never writes a duplicate snapshot.

    variant is sanitized before being used as a path component — found
    live: a caller passing "provider/model" as its variant name (a
    reasonable-looking label) turned the "/" into an unintended
    subdirectory, crashing with FileNotFoundError since that directory
    was never created. Every existing caller's variant names happened to
    already be filesystem-safe (single_call/staged/lexical/etc.), which
    is exactly why this went unnoticed until a caller broke that
    unwritten assumption.
    """
    safe_variant = re.sub(r"[^A-Za-z0-9._-]", "-", variant)
    content_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    return f"{eval_name}_{safe_variant}_{content_hash}.txt"


def record_eval_run(
    eval_name: str,
    variant: str,
    prompt_text: str,
    provider: str,
    model: str,
    trials_per_scenario: int,
    scenario_results: dict,
    results_file: str = RESULTS_FILE,
    prompts_dir: str = PROMPTS_DIR,
) -> dict:
    """Write/reuse a prompt snapshot and append one JSONL record.

    Args:
        eval_name: Which eval this run belongs to (e.g.
            "map_priority_calibration") — groups variants that should be
            compared against each other.
        variant: Which prompt variant was tested (e.g. "zero_shot",
            "few_shot_v1").
        prompt_text: The actual rendered prompt string tested — snapshotted
            verbatim so it's inspectable later even after the prompt-builder
            function's code has changed.
        provider, model: Which LLM actually produced these results.
        trials_per_scenario: How many times each scenario was run.
        scenario_results: {scenario_name: {"expected": str, "results": [str, ...]}}
            — one list of raw per-trial outcomes per scenario.
        results_file, prompts_dir: Overridable for tests; default to the
            real eval_history/ location.

    Returns:
        The full record that was appended (accuracy computed per scenario
        and in aggregate), for the caller to print immediately without a
        separate read-back.
    """
    os.makedirs(prompts_dir, exist_ok=True)
    snapshot_path = os.path.join(prompts_dir, _snapshot_filename(eval_name, variant, prompt_text))
    if not os.path.exists(snapshot_path):
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

    scenarios_scored = {}
    total_correct = 0
    total_trials = 0
    for name, data in scenario_results.items():
        expected = data["expected"]
        results = data["results"]
        correct = sum(1 for r in results if r == expected)
        scenarios_scored[name] = {
            "expected": expected,
            "results": results,
            "accuracy": (correct / len(results)) if results else 0.0,
        }
        total_correct += correct
        total_trials += len(results)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "eval_name": eval_name,
        "variant": variant,
        "prompt_snapshot": snapshot_path,
        "provider": provider,
        "model": model,
        "trials_per_scenario": trials_per_scenario,
        "scenarios": scenarios_scored,
        "aggregate_accuracy": (total_correct / total_trials) if total_trials else 0.0,
    }

    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_eval_history(eval_name: str | None = None, variant: str | None = None, results_file: str = RESULTS_FILE) -> list[dict]:
    """Read back past eval runs, optionally filtered.

    Args:
        results_file: Overridable for tests; defaults to the real
            eval_history/ location.

    Returns:
        A list of records in file order (oldest first). Empty list if
        the results file doesn't exist yet.
    """
    if not os.path.exists(results_file):
        return []

    records = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if eval_name is not None and record.get("eval_name") != eval_name:
                continue
            if variant is not None and record.get("variant") != variant:
                continue
            records.append(record)
    return records
