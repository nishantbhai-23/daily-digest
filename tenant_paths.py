"""
Tenant Paths
=============
Resolves every on-disk path the pipeline touches for a given tenant. Every
loader/saver function in this codebase already accepts a `path` parameter
with a module-constant default (persona.load_persona, tenant_config.
load_tenant_config, ledger.load_ledger/save_ledger/save_digest) — what's
missing is a single place that decides *which* paths to use for a given
tenant, instead of each script hardcoding its own module-level constants.

`tenant_id="default"` resolves to exactly today's existing paths (./data/...,
./output/...) — running any script with no --tenant flag is a complete
no-op change. Any other tenant_id resolves under data/tenants/<id>/ and
output/tenants/<id>/, mirroring the current single-tenant layout per tenant.

Deliberately filesystem-only, no registry/database of known tenants: the
directory listing under data/tenants/ (or just knowing the id you passed in)
is the registry. Consistent with the rest of this codebase's stdlib-only,
filesystem-as-storage approach (ledger.py, persona.py, tenant_config.py).

Usage:
    from tenant_paths import for_tenant

    paths = for_tenant(args.tenant)
    persona_text = load_persona(paths.persona_file)
    emails = load_inbox(paths.inbox_dir)
"""

import os
from dataclasses import dataclass

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class TenantPaths:
    tenant_id: str

    inbox_dir: str
    calendar_file: str
    notes_dir: str
    tasks_file: str
    persona_file: str
    tenant_config_file: str

    output_dir: str
    history_dir: str

    email_ledger_file: str
    calendar_ledger_file: str
    notes_ledger_file: str

    email_summary_file: str
    calendar_summary_file: str
    notes_summary_file: str

    brief_file: str
    metrics_file: str


def for_tenant(tenant_id: str = DEFAULT_TENANT) -> TenantPaths:
    """Resolve every path a pipeline script needs for one tenant.

    Args:
        tenant_id: Defaults to "default", which maps to today's exact
            existing paths — the no-`--tenant`-flag case stays byte-for-byte
            identical to current behavior.

    Returns:
        A fully-resolved TenantPaths.
    """
    if tenant_id == DEFAULT_TENANT:
        data_root = "./data"
        output_root = "./output"
    else:
        data_root = f"./data/tenants/{tenant_id}"
        output_root = f"./output/tenants/{tenant_id}"

    output_dir = output_root if output_root.endswith("/") else f"{output_root}/"

    return TenantPaths(
        tenant_id=tenant_id,
        inbox_dir=f"{data_root}/inbox/",
        calendar_file=f"{data_root}/calendar/calendar.ics",
        notes_dir=f"{data_root}/notes/",
        tasks_file=f"{data_root}/tasks/tasks.json",
        persona_file=f"{data_root}/persona.md",
        tenant_config_file=f"{data_root}/tenant_config.json",
        output_dir=output_dir,
        history_dir=os.path.join(output_dir, "history"),
        email_ledger_file=os.path.join(output_dir, "rolling_ledger.json"),
        calendar_ledger_file=os.path.join(output_dir, "calendar_rolling_ledger.json"),
        notes_ledger_file=os.path.join(output_dir, "notes_rolling_ledger.json"),
        email_summary_file=os.path.join(output_dir, "current_30day_summary.md"),
        calendar_summary_file=os.path.join(output_dir, "current_30day_calendar_summary.md"),
        notes_summary_file=os.path.join(output_dir, "current_30day_notes_summary.md"),
        brief_file=os.path.join(output_dir, "daily_brief.md"),
        metrics_file=os.path.join(output_dir, "metrics.jsonl"),
    )
