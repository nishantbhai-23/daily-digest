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


def load_tenant_config(path: str = CONFIG_FILE) -> dict:
    """Load tenant config, falling back to defaults for anything unset.

    Args:
        path: Path to the tenant's config JSON file.

    Returns:
        DEFAULT_CONFIG merged with whatever the file overrides. Missing file
        returns a copy of DEFAULT_CONFIG unchanged.
    """
    merged = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy, stdlib-only

    if not os.path.exists(path):
        return merged

    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value

    return merged
