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
    from digest.core.ledger import load_ledger, save_ledger, save_digest, validate_schema, compact_ledger
"""

import contextlib
import fcntl
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta


# ─── Ledger Load/Save ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def _ledger_lock(path: str, exclusive: bool = True):
    """Advisory file lock guarding one ledger file against concurrent access.

    What this actually protects, verified directly (not assumed): each
    individual load_ledger()/save_ledger() call is now atomic with respect
    to every other one on the same path — no reader ever sees a torn/
    half-written file, and no two writers' bytes ever interleave into
    invalid JSON. That's real and was worth having on its own.

    What it does NOT protect, found via a concurrent-write test written
    while verifying this: a full read-modify-write *session* (load once,
    mutate an in-memory list, save — repeated across several calls, which
    is exactly triage_agent.py/calendar_agent.py/notes_agent.py's
    run_map_phase pattern) is not atomic end-to-end. Two concurrent
    run_map_phase invocations for the same tenant can each load the same
    starting ledger, each append their own new day, and the second save()
    silently wins — lost update, not corruption. Closing that gap needs a
    lock held for the whole MAP-phase session, not per call, which isn't
    built here. Not a live risk for what currently calls save_ledger
    concurrently (orchestrator.run_for_tenant, driven by run_fleet.py, only
    *reads* ledgers — it never calls save_ledger), but a real limitation if
    someone runs the same tenant's MAP phase twice at once. Documented here
    rather than silently claimed as solved.

    Uses a sidecar `.lock` file rather than locking the data file directly,
    since save_ledger opens the data file with "w" (which truncates
    immediately on open) — a separate lock file lets the lock be acquired
    *before* any truncation happens.

    POSIX-only (fcntl) — fine for this codebase's target environment; no
    cross-platform fallback, consistent with not adding dependencies for a
    problem this system doesn't have yet.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_ledger(path: str) -> tuple:
    """Load an existing ledger and the set of already-processed day keys.

    Returns:
        (ledger_entries, processed_days_set)
    """
    with _ledger_lock(path, exclusive=False):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
            processed_days = {entry["day"] for entry in ledger}
            return ledger, processed_days
        return [], set()


def save_ledger(path: str, ledger: list[dict]) -> None:
    """Persist the ledger, sorted chronologically by day key."""
    with _ledger_lock(path, exclusive=True):
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


# ─── Data Freshness ───────────────────────────────────────────────────────────


def apply_digest_window(
    daily_batches: dict,
    digest_days: int | None = None,
    holdout_days: int | None = None,
) -> tuple[dict, list[str]]:
    """Apply the --digest-days / --holdout-days windowing shared by
    triage_agent.py and calendar_agent.py's MAP phase.

    Both agents' `group_by_date` already returns a chronologically sorted
    {day: batch} dict, which makes both flags a pure slicing problem rather
    than anything requiring new pipeline logic:

    - digest_days caps a cold run to the most recent N days present in the
      loaded data — "how big a digest" control. None means no cap (process
      everything found, today's existing default behavior).
    - holdout_days excludes the most recent N days from what this run even
      sees — they're not marked "already processed," just not included, so
      a later run without this flag picks them up as new days through the
      normal incremental ledger diff. This is what lets the existing
      synthetic dataset simulate a hot pass with no new data: cold now, hot
      later, same command minus this one flag.

    If both are set, digest_days bounds the window first and holdout_days
    carves the tail off of that result.

    Returns:
        (windowed_batches, held_out_day_keys) — the second element is purely
        for caller-side logging (which days got excluded), always [] unless
        holdout_days actually removed something.
    """
    if digest_days is not None:
        daily_batches = dict(list(daily_batches.items())[-digest_days:])

    held_out_days: list[str] = []
    if holdout_days:
        items = list(daily_batches.items())
        if holdout_days < len(items):
            held_out_days = [day for day, _ in items[-holdout_days:]]
            daily_batches = dict(items[:-holdout_days])

    return daily_batches, held_out_days


def format_today(reference_date=None) -> str:
    """Return an explicit "today" string for injection into REDUCE/synthesis
    prompts as ground truth.

    Nothing in the persona or ledgers tells the model what today's actual
    date is — persona.md describes the *concept* of "today" but can't
    contain a literal date since it's a static file reused every run. Left
    ungrounded, REDUCE calls have been observed guessing "today" from
    whatever dates happen to appear in the ledger (the first entry, or the
    most recent one) instead of the real date. Same category of fix as
    check_data_freshness: compute the fact in code, inject it as text,
    don't leave it to inference.
    """
    reference_date = reference_date or datetime.now().date()
    return f"{reference_date.isoformat()} ({reference_date.strftime('%A')})"


def check_schema_consistency(ledger: list[dict]) -> list[str]:
    """Detect a ledger whose entries were produced under different MAP
    schema/prompt configurations — surfaced as a warning instead of silently
    blended.

    Found as a real, not hypothetical, gap: `use_persona_in_map` already
    makes MAP produce structurally different `delta` shapes (with/without a
    `priority` field) depending on tenant config at the time a given day/note
    was processed. Ledgers are resumable by design (see HLD Decision 1),
    which means a config change between runs doesn't get reprocessed — it
    just accumulates alongside the old entries with no prior way to tell them
    apart. Every MAP call now stamps `"map_variant"` on its entry (see
    triage_agent.py's `_map_single_day`); this function is the deterministic
    check that makes a resulting mismatch visible.

    Compacted (weekly-rollup) entries are excluded — they're built by a
    separate compaction LLM call merging multiple original entries, not
    day/note MAP, so they don't carry a comparable map_variant and mixing
    with them is expected, not a version drift signal.

    Args:
        ledger: A loaded ledger (list of entry dicts).

    Returns:
        A list of human-readable warning strings — empty means consistent
        (including the trivial case of 0 or 1 comparable entries). Entries
        from before this field existed are treated as their own
        "unversioned" category rather than skipped, since a ledger silently
        mixing versioned and unversioned entries is exactly the kind of
        blend this check exists to catch.
    """
    comparable = [entry for entry in ledger if not entry.get("compacted")]
    counts: dict[str, int] = {}
    for entry in comparable:
        variant = entry.get("map_variant", "unversioned")
        counts[variant] = counts.get(variant, 0) + 1

    if len(counts) <= 1:
        return []

    breakdown = ", ".join(f"{variant}: {n}" for variant, n in sorted(counts.items()))
    return [
        f"Ledger mixes entries from {len(counts)} different MAP configurations "
        f"({breakdown}) — content shape/quality may differ across entries from "
        f"before vs. after a tenant_config.json change. Re-run MAP for the older "
        f"entries if consistency matters for this digest."
    ]


def check_data_freshness(ledgers: dict, reference_date=None, stale_after_days: int = 1) -> dict:
    """Check how current each source's ledger is, relative to today.

    Deterministic date math — per the profile's honesty rules ("if the
    inbox data is older than 24 hours, say so"), staleness should be
    reported directly, not left for an LLM to notice or silently ignored.

    Args:
        ledgers: {"email": email_ledger, "calendar": calendar_ledger, ...}
        reference_date: "Today" for the comparison; defaults to real now().
        stale_after_days: entries older than this (relative to today) are
            flagged as stale.

    Returns:
        {source_name: {"most_recent_day": "YYYY-MM-DD" | None,
                        "days_stale": int | None, "is_stale": bool}}
    """
    reference_date = reference_date or datetime.now().date()

    result = {}
    for name, entries in ledgers.items():
        if not entries:
            result[name] = {"most_recent_day": None, "days_stale": None, "is_stale": True}
            continue
        most_recent = max(_chronological_key(e) for e in entries)
        most_recent_date = datetime.strptime(most_recent, "%Y-%m-%d").date()
        days_stale = (reference_date - most_recent_date).days
        result[name] = {
            "most_recent_day": most_recent,
            "days_stale": days_stale,
            "is_stale": days_stale > stale_after_days,
        }
    return result


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
