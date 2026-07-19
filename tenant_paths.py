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

`tenant_id` is f-string-interpolated directly into every path this module
builds — the entire cross-tenant isolation guarantee rests on this one
function, so `for_tenant` validates `tenant_id` against an allowlist before
building anything. Confirmed necessary, not speculative: an earlier version
had no check at all, and `for_tenant("../../../../tmp/evil")` resolved to a
path completely outside `data/tenants/` — a live path-traversal bug, found in
a design review, not fixed proactively.

Usage:
    from tenant_paths import for_tenant

    paths = for_tenant(args.tenant)
    persona_text = load_persona(paths.persona_file)
    emails = load_inbox(paths.inbox_dir)
"""

import os
import re
from dataclasses import dataclass

DEFAULT_TENANT = "default"

# Lowercase alphanumeric, hyphen, underscore; must start with an alphanumeric
# char; capped at 64 chars. Deliberately conservative — no ".", "/", or
# whitespace at all, so there's no path-separator or dot-segment shaped
# character to reject case-by-case. "default" satisfies this itself, so no
# special-case exemption is needed.
_VALID_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


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

    Raises:
        ValueError: If tenant_id doesn't match _VALID_TENANT_ID_RE — rejects
            path separators, ".."-shaped traversal, whitespace, and anything
            outside a conservative lowercase-alphanumeric/hyphen/underscore
            allowlist, before any path is constructed.
    """
    if not _VALID_TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r} — must match "
            f"{_VALID_TENANT_ID_RE.pattern} (lowercase alphanumeric, "
            f"hyphen, underscore; max 64 chars)"
        )

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
