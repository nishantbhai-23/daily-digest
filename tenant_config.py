"""
Tenant Config
==============
Per-tenant behavioral configuration — the subset of rules that must be
enforced deterministically in code, regardless of prompt compliance, as
opposed to persona.md's free-text tone/priority guidance which is left to
the LLM to read and follow.

Kept as a separate file from persona.md on purpose: persona.md is prose the
model is trusted to interpret; tenant_config.json holds things where prompt
compliance alone isn't good enough — e.g. "never draft for this contact" is
exactly the kind of rule that should never depend on a model remembering to
follow an instruction. Extracting these rules from persona.md's prose via an
LLM at onboarding time was considered and rejected for the same reason:
anything safety-critical enough to deserve code-level enforcement should be
explicitly authored, not inferred.

Usage:
    from tenant_config import load_tenant_config

    config = load_tenant_config()
    if config["use_persona_in_map"]:
        ...
"""

import json
import os

DEFAULT_CONFIG = {
    # Whether persona.md is injected into MAP prompts at all. REDUCE always
    # gets full persona regardless — this only affects the per-day/per-note
    # extraction pass. See triage_agent.py/calendar_agent.py/notes_agent.py's
    # build_map_system_prompt for what changes when this is False.
    "use_persona_in_map": True,
    # Contacts to never draft a reply for — surfaced instead, per the same
    # reasoning as persona.md's "for Sam: don't draft" rule, generalized to
    # any tenant's own list. List of {"name": ..., "email": ...}; matched
    # against both, since the text being filtered is LLM-generated prose
    # that's far more likely to reference someone by name than by email.
    "never_draft_contacts": [],
    # Deterministic pre-MAP email filter — senders/domains that never reach
    # an LLM call at all. Email-specific: calendar/notes don't have an
    # equivalent "spam sender" concept to filter the same way.
    "map_noise_filter": {
        "blocked_senders": [],
        "blocked_domains": [],
    },
}

CONFIG_FILE = "./data/tenant_config.json"

# Fleet-level ceilings a tenant must never be able to set for themselves —
# same reasoning that pulled never_draft_contacts out of persona.md's free
# text and into deterministic code: the highest-stakes settings shouldn't
# depend on trusting arbitrary tenant-authored content, whether that's a
# prompt or, as here, a JSON file the tenant is otherwise free to edit. Never
# stored under data/tenants/<id>/ — one file, shared across every tenant.
SYSTEM_CONFIG_FILE = "./data/system_config.json"

DEFAULT_SYSTEM_CONFIG = {
    "max_qps_per_tenant": 2.0,
    "allowed_providers": ["ollama", "anthropic", "google", "openrouter", "deepseek"],
    "circuit_breaker_threshold": 5,
}

# Keys that always come from system config, even if a tenant's own file
# also happens to set them — enforced by load_tenant_config always applying
# these last, overwriting whatever the tenant-config merge produced.
SYSTEM_ENFORCED_KEYS = frozenset(DEFAULT_SYSTEM_CONFIG.keys())


def load_system_config(path: str = SYSTEM_CONFIG_FILE) -> dict:
    """Load the fleet-level system config, falling back to defaults for
    anything unset. Missing file returns a copy of DEFAULT_SYSTEM_CONFIG.
    """
    merged = json.loads(json.dumps(DEFAULT_SYSTEM_CONFIG))  # deep copy, stdlib-only

    if not os.path.exists(path):
        return merged

    with open(path, "r", encoding="utf-8") as f:
        system_config = json.load(f)

    merged.update(system_config)
    return merged


def load_tenant_config(path: str = CONFIG_FILE, system_path: str = SYSTEM_CONFIG_FILE) -> dict:
    """Load tenant config, falling back to defaults for anything unset, then
    layer system config on top so SYSTEM_ENFORCED_KEYS always reflect the
    fleet-level file regardless of what the tenant's own file contains.

    Args:
        path: Path to the tenant's config JSON file.
        system_path: Path to the fleet-level system config — deliberately
            NOT tenant-scoped; always the same file for every tenant unless
            a caller explicitly overrides it (e.g. in tests).

    Returns:
        DEFAULT_CONFIG merged with the tenant's file, then with
        SYSTEM_ENFORCED_KEYS overwritten by system config as the final step.
    """
    merged = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy, stdlib-only

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value

    system_config = load_system_config(system_path)
    for key in SYSTEM_ENFORCED_KEYS:
        merged[key] = system_config[key]

    return merged
