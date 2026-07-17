"""
Ledger Utilities
=================
Shared persistence, validation, digest-history, and compaction helpers
used by both triage_agent.py (email) and calendar_agent.py (calendar).

Both agents produce ledgers with the same entry shape:
    {"day": "YYYY-MM-DD", "<count_key>": int, "delta": {...}}

Keeping this logic in one place means both pipelines get the same
reliability fixes (retention/compaction, digest history, schema
validation) instead of drifting independently.

Usage:
    from ledger import load_ledger, save_ledger, save_digest, validate_schema, compact_ledger
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta


# ─── Ledger Load/Save ─────────────────────────────────────────────────────────


def load_ledger(path: str) -> tuple:
    """Load an existing ledger and the set of already-processed day keys.

    Returns:
        (ledger_entries, processed_days_set)
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        processed_days = {entry["day"] for entry in ledger}
        return ledger, processed_days
    return [], set()


def save_ledger(path: str, ledger: list[dict]) -> None:
    """Persist the ledger, sorted chronologically by day key."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ledger_sorted = sorted(ledger, key=_chronological_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger_sorted, f, indent=2)


# ─── Schema Validation ────────────────────────────────────────────────────────


def validate_schema(data, required_keys: dict) -> list[str]:
    """Lightweight, dependency-free shape check for LLM JSON output.

    Args:
        data: The parsed JSON to validate.
        required_keys: Maps top-level key -> list of required keys for each
            item in that list (or None to only check the key is a list).

    Returns:
        A list of human-readable error strings. Empty list means valid.
    """
    errors = []
    if not isinstance(data, dict):
        return [f"Expected a JSON object at top level, got {type(data).__name__}"]

    for key, item_keys in required_keys.items():
        if key not in data:
            errors.append(f"Missing top-level key: '{key}'")
            continue
        value = data[key]
        if not isinstance(value, list):
            errors.append(f"Expected '{key}' to be a list, got {type(value).__name__}")
            continue
        if not item_keys:
            continue
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"'{key}[{i}]' expected an object, got {type(item).__name__}")
                continue
            missing = [k for k in item_keys if k not in item]
            if missing:
                errors.append(f"'{key}[{i}]' missing keys: {missing}")

    return errors


# ─── Digest History ───────────────────────────────────────────────────────────


def save_digest(content: str, current_path: str, history_dir: str) -> str:
    """Write the digest to its 'current' path and a timestamped history copy.

    Overwriting a single current-summary file loses the ability to see how
    the digest evolved run over run. This keeps a permanent, timestamped
    trail in `history_dir` alongside the always-fresh current file.

    Returns:
        The path to the timestamped history copy.
    """
    os.makedirs(os.path.dirname(current_path) or ".", exist_ok=True)
    with open(current_path, "w", encoding="utf-8") as f:
        f.write(content)

    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(current_path))[0]
    history_path = os.path.join(history_dir, f"{base_name}_{timestamp}.md")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(content)

    return history_path


# ─── Ledger Compaction ─────────────────────────────────────────────────────────


def compact_ledger(
    ledger: list[dict],
    llm,
    compact_system_prompt: str,
    retention_days: int = 30,
    count_key: str = "email_count",
) -> list[dict]:
    """Collapse daily entries older than `retention_days` into weekly rollups.

    Without this, REDUCE re-sends the entire ledger every run, unbounded —
    fine at 30 days, but cost/context grows forever as history accumulates.
    Entries older than the retention window are grouped by ISO week and
    merged into a single compressed weekly entry via one LLM call per stale
    week, using the same delta schema as the daily entries. Already-compacted
    entries (marked `"compacted": True`) are left alone.

    Args:
        ledger: The full ledger (list of {"day", count_key, "delta"} entries).
        llm: A BaseLLM instance used to synthesize each stale week.
        compact_system_prompt: System prompt describing how to merge a week's
            worth of daily deltas into one weekly delta (source-specific).
        retention_days: Entries older than this (relative to today) are
            eligible for compaction.
        count_key: The per-entry count field name ("email_count",
            "event_count", ...).

    Returns:
        A new ledger list: untouched recent entries + compacted weekly
        entries for anything older than the retention window.
    """
    if not ledger:
        return ledger

    cutoff = datetime.now().date() - timedelta(days=retention_days)

    fresh, stale = [], []
    for entry in ledger:
        if entry.get("compacted"):
            fresh.append(entry)
            continue
        try:
            entry_date = datetime.strptime(entry["day"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            # Unparseable day key — don't touch it, just carry it forward.
            fresh.append(entry)
            continue
        if entry_date < cutoff:
            stale.append((entry_date, entry))
        else:
            fresh.append(entry)

    if not stale:
        return ledger

    weeks = defaultdict(list)
    for entry_date, entry in stale:
        iso_year, iso_week, _ = entry_date.isocalendar()
        weeks[f"{iso_year}-W{iso_week:02d}"].append(entry)

    compacted = []
    for week_key, entries in sorted(weeks.items()):
        entries.sort(key=lambda e: e["day"])
        deltas_context = json.dumps([e["delta"] for e in entries], indent=2)
        count_sum = sum(e.get(count_key, 0) for e in entries)

        try:
            merged_delta = llm.chat_json(
                messages=[
                    {"role": "system", "content": compact_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Daily deltas for week {week_key} "
                            f"({entries[0]['day']} to {entries[-1]['day']}):\n\n"
                            f"{deltas_context}"
                        ),
                    },
                ]
            )
        except Exception as e:
            print(f"   ⚠️  Compaction failed for week {week_key}: {e} — keeping daily entries")
            compacted.extend(entries)
            continue

        compacted_entry = {
            "day": week_key,
            count_key: count_sum,
            "delta": merged_delta,
            "compacted": True,
            "days_covered": [e["day"] for e in entries],
        }
        # Deterministic per-day facts (calendar's deep_work_conflicts,
        # notes' checklist staleness, ...) have a source-specific schema
        # ledger.py doesn't know — rather than trying to merge them, keep
        # them keyed by day so nothing is silently dropped on compaction.
        stats_by_day = {e["day"]: e["stats"] for e in entries if "stats" in e}
        if stats_by_day:
            compacted_entry["stats_by_day"] = stats_by_day
        note_ids = [e["note_id"] for e in entries if "note_id" in e]
        if note_ids:
            compacted_entry["note_ids_covered"] = note_ids

        compacted.append(compacted_entry)

    return sorted(compacted + fresh, key=_chronological_key)


def _chronological_key(entry: dict) -> str:
    """Sort key that orders daily ('YYYY-MM-DD') and compacted weekly
    ('YYYY-Www') entries chronologically by resolving weekly entries to
    their week's Monday date.
    """
    day = entry["day"]
    if entry.get("compacted") and "-W" in day:
        year, week = day.split("-W")
        monday = datetime.strptime(f"{year}-{week}-1", "%G-%V-%u").date()
        return monday.isoformat()
    return day
