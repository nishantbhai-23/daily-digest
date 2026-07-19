"""
Fleet Runner — Concurrent Multi-Tenant Emulation
===================================================
Answers the question that started this whole thread: how do you emulate a
multi-tenant architecture with multi-QPS calls, without a real server or a
database? Fires orchestrator.py's pipeline for N tenant IDs concurrently, in
one process, via a ThreadPoolExecutor.

The "one process" part is deliberate, not incidental: resilience.py's
CircuitBreaker and TokenBucket registries are in-memory and module-level —
they only have fleet-wide visibility if every tenant's calls happen inside
the same process's memory. orchestrator.run_for_tenant is called directly
(not as a subprocess) specifically so that shared state stays shared, and so
the breaker/limiter/timeout/classification machinery already built into
llm.py and orchestrator.py actually gets exercised under real concurrent
contention against a real provider, instead of only ever being tested one
call at a time.

This only reads ledgers each tenant's triage_agent.py/calendar_agent.py/
notes_agent.py have already produced (same requirement as orchestrator.py
itself) — --seed-from-default is a convenience for emulation: it copies the
"default" tenant's data and already-built ledgers into each new tenant's
directory so there's something to synthesize from without running a full
live MAP pass per tenant first. A real onboarded tenant would have their own
data; this flag exists purely to make local multi-tenant testing cheap.

Usage:
    python3 run_fleet.py --tenants acme globex initech --seed-from-default
    python3 run_fleet.py --tenants acme globex --provider deepseek --model deepseek-chat
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from digest import orchestrator
from digest.core import tenant_paths
def seed_tenant_from_default(tenant_id: str) -> None:
    """Copy the default tenant's raw data and already-built ledgers into a
    new tenant's directories. Emulation-only convenience, not part of the
    normal per-tenant onboarding story.
    """
    default_paths = tenant_paths.for_tenant(tenant_paths.DEFAULT_TENANT)
    paths = tenant_paths.for_tenant(tenant_id)

    for src_dir, dst_dir in [
        (default_paths.inbox_dir, paths.inbox_dir),
        (default_paths.notes_dir, paths.notes_dir),
    ]:
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    for src_file, dst_file in [
        (default_paths.calendar_file, paths.calendar_file),
        (default_paths.persona_file, paths.persona_file),
        (default_paths.tenant_config_file, paths.tenant_config_file),
        (default_paths.email_ledger_file, paths.email_ledger_file),
        (default_paths.calendar_ledger_file, paths.calendar_ledger_file),
        (default_paths.notes_ledger_file, paths.notes_ledger_file),
    ]:
        if os.path.exists(src_file):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)


def run_fleet(tenants: list[str], provider: str, model: str, temperature: float = 0.0) -> dict:
    """Run orchestrator.run_for_tenant for every tenant concurrently.

    Returns:
        {tenant_id: "ok" or "failed: <error>"}, in completion order.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=len(tenants)) as executor:
        future_to_tenant = {
            executor.submit(orchestrator.run_for_tenant, tenant_id, provider, model, temperature): tenant_id
            for tenant_id in tenants
        }
        for future in as_completed(future_to_tenant):
            tenant_id = future_to_tenant[future]
            try:
                future.result()
                results[tenant_id] = "ok"
            except Exception as e:
                results[tenant_id] = f"failed: {type(e).__name__}: {e}"
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fleet Runner — concurrent multi-tenant orchestrator emulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 run_fleet.py --tenants acme globex --seed-from-default\n"
            "  python3 run_fleet.py --tenants acme globex initech --provider deepseek --model deepseek-chat\n"
        ),
    )
    parser.add_argument("--tenants", nargs="+", required=True, help="Tenant IDs to run concurrently")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"], help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="llama3", help="Model name (default: llama3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument(
        "--seed-from-default",
        action="store_true",
        help="Copy the default tenant's data + ledgers into each tenant before running (emulation convenience)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed_from_default:
        print(f"🌱 Seeding {len(args.tenants)} tenant(s) from 'default'...")
        for tenant_id in args.tenants:
            seed_tenant_from_default(tenant_id)

    print(f"🚀 Firing {len(args.tenants)} tenant(s) concurrently: {', '.join(args.tenants)}\n")
    start = time.time()
    results = run_fleet(args.tenants, args.provider, args.model, args.temperature)
    elapsed = time.time() - start

    print(f"\n=== Fleet run summary ({elapsed:.1f}s total) ===")
    for tenant_id, status in results.items():
        icon = "✅" if status == "ok" else "❌"
        print(f"  {icon} {tenant_id}: {status}")


if __name__ == "__main__":
    main()
