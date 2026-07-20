"""
Calendar Triage Agent — Map-Reduce Pipeline
=============================================
Processes calendar events through a two-phase pipeline to build a persistent
30-day compressed memory ledger, then synthesizes actionable schedule
insights — mirroring triage_agent.py's architecture for email, kept as a
fully independent pipeline (own parser, own ledger, own digest). Any
cross-referencing with email/notes/tasks is deliberately left to a future
orchestrator, not built into this agent.

MAP phase:  Each day's events → deterministic stats (pure Python — meeting
            load, focus-time protection, back-to-back streaks) + an LLM call
            for the qualitative layer arithmetic can't do (prep needed,
            family-calendar flags, pattern drift, notable one-offs).
REDUCE phase: Full 30-day ledger → schedule-focused digest.

The ledger is persistent and supports incremental updates — re-running the
script only processes newly added days. Old entries beyond the retention
window are compacted into weekly rollups so REDUCE cost doesn't grow
unbounded.

Usage:
    python calendar_agent.py
    python calendar_agent.py --provider anthropic --model claude-sonnet-4-20250514
    python calendar_agent.py --map-only
    python calendar_agent.py --reduce-only
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from digest.core import tenant_paths
from digest.parsers.calendar_parser import load_calendar, group_by_date
from digest.core.ledger import load_ledger, save_ledger, save_digest, validate_schema, compact_ledger, format_today, apply_digest_window, check_schema_consistency
from digest.core.llm import create_llm, call_with_retry
from digest.core.persona import load_persona
from digest.core.tenant_config import load_tenant_config


# ─── Path Configuration ──────────────────────────────────────────────────────

CALENDAR_FILE = "./data/calendar/calendar.ics"
OUTPUT_DIR = "./output/"
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
LEDGER_FILE = os.path.join(OUTPUT_DIR, "calendar_rolling_ledger.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "current_30day_calendar_summary.md")

COUNT_KEY = "event_count"

MAP_SCHEMA = {
    "meetings_needing_prep": ["summary"],
    "family_calendar_items": ["summary"],
    "pattern_flags": ["description"],
    "notable_events": ["summary"],
}


def build_compact_system_prompt(persona_text: str) -> str:
    """Persona-aware weekly compaction — see triage_agent.py's
    build_compact_system_prompt for the full rationale.
    """
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are compressing a week's worth of daily calendar-triage deltas into a single "
        "weekly delta, using the exact same JSON schema. Merge duplicate/similar flags, "
        "keep genuinely distinct items, and preserve specifics (names, dates, meeting "
        "titles).\n\n"
        "When merging, preserve P0-P1 items individually even if they look similar — "
        "repeated mentions of a high-priority item across the week is a real signal "
        "(recurrence), not redundancy, and collapsing them into one entry would lose "
        "it. P3-P4 items that appear only once may be dropped if a more important "
        "item on the same topic already covers it.\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "meetings_needing_prep": [{"summary": "...", "why": "..."}],\n'
        '  "family_calendar_items": [{"summary": "...", "note": "..."}],\n'
        '  "pattern_flags": [{"description": "..."}],\n'
        '  "notable_events": [{"summary": "...", "why_notable": "..."}]\n'
        "}\n\n"
        "Do not write any markdown wrappers, conversational pleasantries, or extra text."
    )


# ─── Prompt Builders (persona-injected) ───────────────────────────────────────


def build_map_system_prompt(persona_text: str, use_persona: bool = True) -> str:
    """Build the MAP system prompt.

    Args:
        persona_text: The operator profile text.
        use_persona: When False, family-detection and notable-event weighting
            fall back to generic structural heuristics (attendees outside the
            company domain, personal-sounding titles) instead of persona-
            informed judgment (e.g. knowing a specific name is family). The
            schema stays identical either way — envelope consistency for
            downstream code (cross_reference.py, render_ledger_as_text)
            matters more than which mode produced a given ledger entry.
            REDUCE always gets full persona regardless (see
            tenant_config.py's "use_persona_in_map").
    """
    if use_persona:
        header = f"{persona_text}\n\n---\n\n"
        role = "You are a calendar triage node for the operator profiled above. "
        family_instruction = (
            "2. **Family calendar items**: Any event that looks like it was added by or "
            "concerns the operator's partner or family (per the profile above), especially "
            "ones that collide with a work meeting — these matter more than anything else "
            "on the calendar, per the profile.\n"
        )
        notable_instruction = (
            "4. **Notable events**: One-off events (investor calls, board meetings, "
            "external calls) worth a one-line flag, weighted using the people/priority "
            "guidance in the profile above.\n\n"
        )
    else:
        header = ""
        role = "You are a calendar extraction node. "
        family_instruction = (
            "2. **Family calendar items**: Any event that structurally looks personal or "
            "non-business (attendees outside the company's email domain, personal-sounding "
            "titles), especially ones that collide with a work meeting.\n"
        )
        notable_instruction = (
            "4. **Notable events**: One-off events (investor calls, board meetings, "
            "external calls) worth a one-line flag.\n\n"
        )

    return (
        f"{header}"
        f"{role}"
        "You will receive one day's calendar events, plus deterministic stats already "
        "computed in code (meeting load, focus-time protection, deep-work overrides, "
        "back-to-back streaks — do not recompute these, they're ground truth). Your job "
        "is the layer arithmetic can't do:\n\n"
        "1. **Meetings needing prep**: Based on the summary/description/attendees, "
        "which meetings need the operator to prepare something beforehand?\n"
        f"{family_instruction}"
        "3. **Pattern flags**: Anything that looks like drift from a normal pattern — "
        "e.g. a direct report's 1:1 cadence slipping, an unusual gap or cluster.\n"
        f"{notable_instruction}"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "meetings_needing_prep": [{"summary": "...", "why": "..."}],\n'
        '  "family_calendar_items": [{"summary": "...", "note": "..."}],\n'
        '  "pattern_flags": [{"description": "..."}],\n'
        '  "notable_events": [{"summary": "...", "why_notable": "..."}]\n'
        "}\n\n"
        "If a category has no entries for the day, use an empty array. Do not write any "
        "markdown wrappers, conversational pleasantries, or extra text."
    )


def build_reduce_system_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are the calendar half of the operator's daily digest, for the operator "
        "profiled above. You are given a chronological sequence of daily calendar "
        "signals (deterministic stats + qualitative flags) covering up to a 30-day "
        "window. This complements a separate email digest — focus only on schedule and "
        "time, don't try to cover email content.\n\n"
        "You will be given today's actual date — use it as ground truth for anything "
        "date-relative (\"today\", \"coming up\"). Do not infer today's date from the "
        "ledger content itself (e.g. the earliest or most recent day present) — that "
        "has produced wrong dates before.\n\n"
        "Synthesize this into a concise digest covering:\n\n"
        "## 1. TODAY'S SCHEDULE — WHAT NEEDS PREP\n"
        "Meetings today or coming up that need something prepared beforehand.\n\n"
        "## 2. DEEP WORK / FOCUS TIME\n"
        "How well protected time has actually held up — call out any overrides, using "
        "the deep_work_conflicts stats as ground truth, per the profile's stated rule "
        "about this time being protected.\n\n"
        "## 3. FAMILY & COLLISIONS\n"
        "Any family-calendar items, especially ones that collide with a work meeting. "
        "Per the profile: family first, always — surface these prominently, don't bury "
        "them under work content.\n\n"
        "## 4. PATTERNS WORTH FLAGGING\n"
        "Cadence drift (1:1s slipping, unusual meeting-load changes) across the window.\n\n"
        "Target length: 200-400 words total. Use short bullet points (2-5 per "
        "section), not paragraphs — if the full digest would take longer than 90 "
        "seconds to read, cut the least important items rather than compressing "
        "everything to fit.\n\n"
        "Follow the profile's honesty rules: say if the data looks stale, flag "
        "assumptions, and don't hide contradictions. Match the profile's tone — short, "
        "direct, no fluff. Write in markdown.\n\n"
        "Important: write the digest itself, addressed directly to the operator in "
        "second person, as their morning brief. Do NOT describe, summarize, or explain "
        "the JSON data format, field names, or structure of the ledger you were given — "
        "that is input data for you to read, not something to report on."
    )


# ─── Deterministic Stats (pure Python — no LLM needed for arithmetic) ────────


def _duration_hours(event: dict) -> float:
    if not event.get("start") or not event.get("end"):
        return 0.0
    return (event["end"] - event["start"]).total_seconds() / 3600


def _is_focus_block(event: dict) -> bool:
    return "deep work" in event["summary"].lower() or "focus time" in event["summary"].lower()


def _is_lunch(event: dict) -> bool:
    return "lunch" in event["summary"].lower()


def _categorize(event: dict) -> str:
    summary = event["summary"].lower()
    if "1:1" in summary or "one-on-one" in summary:
        return "one_on_one"
    if any(k in summary for k in ("customer", "investor", "board")):
        return "external"
    if any(k in summary for k in ("standup", "retro", "planning", "all-hands", "all hands", "pipeline review")):
        return "internal_ceremony"
    return "other"


def compute_day_stats(events: list[dict]) -> dict:
    """Pure-Python deterministic stats for a day's calendar events.

    Kept out of the LLM entirely — meeting counts, hours, and interval
    overlap are arithmetic, not judgment. The LLM layer (MAP phase) only
    reasons about what these numbers *mean* for the operator.
    """
    confirmed = [e for e in events if e.get("status", "CONFIRMED") == "CONFIRMED"]
    meetings = [e for e in confirmed if not _is_focus_block(e) and not _is_lunch(e)]
    focus_blocks = [e for e in confirmed if _is_focus_block(e)]
    declined_or_cancelled = [e for e in events if e.get("status", "CONFIRMED") != "CONFIRMED"]

    meeting_hours = sum(_duration_hours(e) for e in meetings)
    focus_hours_scheduled = sum(_duration_hours(e) for e in focus_blocks)

    deep_work_conflicts = []
    focus_hours_eroded = 0.0
    for fb in focus_blocks:
        for m in meetings:
            if not m.get("start") or not m.get("end"):
                continue
            latest_start = max(fb["start"], m["start"])
            earliest_end = min(fb["end"], m["end"])
            overlap = max(0.0, (earliest_end - latest_start).total_seconds() / 3600)
            if overlap > 0:
                focus_hours_eroded += overlap
                deep_work_conflicts.append({
                    "summary": m["summary"],
                    "start": m["start"].isoformat(),
                    "overlap_hours": round(overlap, 2),
                })
    focus_hours_protected = max(0.0, focus_hours_scheduled - focus_hours_eroded)

    meetings_sorted = sorted(
        [m for m in meetings if m.get("start") and m.get("end")],
        key=lambda e: e["start"],
    )
    back_to_back_count = sum(
        1 for a, b in zip(meetings_sorted, meetings_sorted[1:]) if b["start"] <= a["end"]
    )

    hours_by_category = defaultdict(float)
    for m in meetings:
        hours_by_category[_categorize(m)] += _duration_hours(m)

    declined_or_cancelled_list = [
        {
            "summary": e["summary"],
            "date": e["start"].date().isoformat() if e.get("start") else "unknown",
            "status": e.get("status", "CANCELLED"),
        }
        for e in declined_or_cancelled
    ]

    return {
        "meeting_count": len(meetings),
        "meeting_hours": round(meeting_hours, 2),
        "focus_hours_scheduled": round(focus_hours_scheduled, 2),
        "focus_hours_protected": round(focus_hours_protected, 2),
        "focus_hours_eroded": round(focus_hours_eroded, 2),
        "deep_work_conflicts": deep_work_conflicts,
        "back_to_back_count": back_to_back_count,
        "declined_or_cancelled_count": len(declined_or_cancelled),
        "declined_or_cancelled": declined_or_cancelled_list,
        "meeting_hours_by_category": {k: round(v, 2) for k, v in hours_by_category.items()},
    }


# ─── Helper Functions ─────────────────────────────────────────────────────────


def format_event_batch(events: list[dict]) -> str:
    """Format a batch of parsed calendar events into a context string for the LLM."""
    parts = []
    for i, ev in enumerate(events, 1):
        attendee_names = ", ".join(a["name"] for a in ev["attendees"]) or "none listed"
        parts.append(
            f"--- Event {i} ---\n"
            f"Summary: {ev['summary']}\n"
            f"Status: {ev['status']}\n"
            f"Start: {ev['start'].isoformat() if ev['start'] else 'unknown'}\n"
            f"End: {ev['end'].isoformat() if ev['end'] else 'unknown'}\n"
            f"Location: {ev['location'] or 'none'}\n"
            f"Attendees: {attendee_names}\n"
            f"Description: {ev['description'] or 'none'}\n"
        )
    return "\n".join(parts)


# ─── MAP Phase ────────────────────────────────────────────────────────────────


def _map_single_day(llm, day: str, events: list[dict], map_system_prompt: str, hot: bool = False, map_variant: str = "persona") -> dict | None:
    """Process a single day's calendar events. Thread-safe.

    Args:
        hot: Marks this entry as sourced from --hot-input — see
            triage_agent.py's `_map_single_day` for the full rationale;
            same convention, mirrored here for the calendar side.
        map_variant: Which MAP prompt configuration produced this entry —
            "persona" or "no_persona". Calendar's MAP_SCHEMA shape doesn't
            change between the two (unlike email/notes), only the
            instruction text does, but the entry is still tagged for
            consistency — see triage_agent.py's `_map_single_day` and
            ledger.check_schema_consistency.

    Returns:
        A ledger entry dict on success, None on failure.
    """
    stats = compute_day_stats(events)
    context = format_event_batch(events)
    start = time.time()

    def _call():
        delta = llm.chat_json(
            messages=[
                {"role": "system", "content": map_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Calendar events for {day}:\n\n{context}\n\n"
                        f"Precomputed stats (ground truth, do not recompute):\n"
                        f"{json.dumps(stats, indent=2)}"
                    ),
                },
            ]
        )
        errors = validate_schema(delta, MAP_SCHEMA)
        if errors:
            raise ValueError(f"Invalid MAP output: {errors}")
        return delta

    try:
        delta = call_with_retry(_call)
        elapsed = time.time() - start
        hot_tag = " 🔥" if hot else ""
        print(f"   ✅ {day} — {len(events)} events — {elapsed:.1f}s{hot_tag}")
        entry = {
            "day": day,
            COUNT_KEY: len(events),
            "stats": stats,
            "delta": delta,
            "map_variant": map_variant,
        }
        if hot:
            entry["hot"] = True
        return entry
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ⚠️  {day} — FAILED after retries ({elapsed:.1f}s): {e}")
        return None


def run_map_phase(
    llm,
    map_system_prompt: str,
    paths,
    max_workers: int = 4,
    digest_days: int | None = None,
    holdout_days: int | None = None,
    hot_input: str | None = None,
    map_variant: str = "persona",
) -> bool:
    """MAP: Process calendar events day-by-day into structured signal deltas.

    Args:
        digest_days, holdout_days, hot_input: See triage_agent.py's
            run_map_phase — same flags, same semantics, mirrored here.
            hot_input points at a single additional .ics file rather than a
            directory, since calendar data is one file, not one-file-per-item.
    """
    map_start = time.time()
    print("🚀 Starting Calendar MAP Phase...")

    t0 = time.time()
    print(f"📅 Loading calendar from {paths.calendar_file}...")
    events = load_calendar(paths.calendar_file)
    if not events:
        print("❌ No calendar events found. Check the calendar file.")
        return False
    parse_time = time.time() - t0
    print(f"   Found {len(events)} events. (parsed in {parse_time:.1f}s)")

    if hot_input:
        hot_events = load_calendar(hot_input)
        if not hot_events:
            print(f"   ⚠️  No events found under --hot-input '{hot_input}'.")
        else:
            for e in hot_events:
                e["_hot"] = True
            print(f"   🔥 Loaded {len(hot_events)} hot event(s) from {hot_input}.")
            events = events + hot_events

    daily_batches = group_by_date(events)
    print(f"   Spanning {len(daily_batches)} days.\n")

    before_days = len(daily_batches)
    daily_batches, held_out_days = apply_digest_window(daily_batches, digest_days, holdout_days)
    if digest_days is not None and digest_days < before_days:
        print(f"   ✂️  --digest-days {digest_days}: using the most recent {digest_days} day(s).")
    if holdout_days:
        if held_out_days:
            print(f"   🧊 --holdout-days {holdout_days}: holding out {held_out_days} for a later hot pass.")
        else:
            print(f"   ⚠️  --holdout-days {holdout_days} >= {len(daily_batches)} available days; holding out nothing.")

    hot_days = {day for day, batch in daily_batches.items() if any(e.get("_hot") for e in batch)}

    ledger, processed_days = load_ledger(paths.calendar_ledger_file)
    if processed_days:
        print(f"📋 Existing ledger found: {len(processed_days)} days already processed.")

    days_to_process = [
        (day, batch) for day, batch in daily_batches.items() if day not in processed_days
    ]
    skipped = len(daily_batches) - len(days_to_process)
    if skipped > 0:
        print(f"   ⏭️  Skipping {skipped} already-processed days.")

    if not days_to_process:
        print("   Nothing new to process.")
        save_ledger(paths.calendar_ledger_file, ledger)
        return True

    print(f"\n🔄 Processing {len(days_to_process)} days with {max_workers} worker(s)...\n")

    succeeded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_day = {
            executor.submit(_map_single_day, llm, day, batch, map_system_prompt, day in hot_days, map_variant): day
            for day, batch in days_to_process
        }
        for future in as_completed(future_to_day):
            result = future.result()
            if result is not None:
                ledger.append(result)
                save_ledger(paths.calendar_ledger_file, ledger)
                succeeded += 1

    total_time = time.time() - map_start
    print(f"\n✅ Calendar MAP Phase completed in {total_time:.1f}s.")
    print(
        f"   {succeeded}/{len(days_to_process)} days succeeded. "
        f"Ledger: {paths.calendar_ledger_file} ({len(ledger)} total days)\n"
    )
    return True


# ─── REDUCE Phase ─────────────────────────────────────────────────────────────


def _render_calendar_stats(stats: dict, indent: str = "") -> list[str]:
    """Render one day's deterministic calendar stats as text lines."""
    lines = [
        f"{indent}- Meetings: {stats.get('meeting_count', 0)} "
        f"({stats.get('meeting_hours', 0)}h) | "
        f"Focus time protected: {stats.get('focus_hours_protected', 0)}h, "
        f"eroded: {stats.get('focus_hours_eroded', 0)}h | "
        f"Back-to-back: {stats.get('back_to_back_count', 0)} | "
        f"Declined/cancelled: {stats.get('declined_or_cancelled_count', 0)}"
    ]
    for conflict in stats.get("deep_work_conflicts", []):
        lines.append(
            f"{indent}  - DEEP WORK CONFLICT: '{conflict['summary']}' at "
            f"{conflict['start']} ({conflict['overlap_hours']}h overlap)"
        )
    for declined in stats.get("declined_or_cancelled", []):
        lines.append(
            f"{indent}  - DECLINED/CANCELLED: '{declined['summary']}' on {declined['date']}"
        )
    return lines


def render_ledger_as_text(ledger: list[dict]) -> str:
    """Render the ledger as readable text instead of raw JSON.

    Smaller local models tend to fall into "describe this JSON" pattern-
    completion when handed a large literal JSON blob as context, regardless
    of system-prompt instructions to the contrary. Flattening it to prose
    keeps the model focused on synthesizing content instead of narrating
    the data structure.
    """
    lines = []
    for entry in ledger:
        label = f"Week of {entry['day']}" if entry.get("compacted") else entry["day"]
        hot_suffix = " [JUST ARRIVED]" if entry.get("hot") else ""
        lines.append(f"### {label} ({entry.get(COUNT_KEY, 0)} events){hot_suffix}")

        if entry.get("stats"):
            lines.extend(_render_calendar_stats(entry["stats"]))
        for day, day_stats in entry.get("stats_by_day", {}).items():
            lines.append(f"  {day}:")
            lines.extend(_render_calendar_stats(day_stats, indent="  "))

        delta = entry.get("delta", {})
        for item in delta.get("meetings_needing_prep", []):
            lines.append(f"- PREP NEEDED: {item.get('summary')} — {item.get('why', '')}")
        for item in delta.get("family_calendar_items", []):
            lines.append(f"- FAMILY: {item.get('summary')} — {item.get('note', '')}")
        for item in delta.get("pattern_flags", []):
            lines.append(f"- PATTERN: {item.get('description')}")
        for item in delta.get("notable_events", []):
            lines.append(f"- NOTABLE: {item.get('summary')} — {item.get('why_notable', '')}")

        lines.append("")
    return "\n".join(lines)


def run_reduce_phase(llm, reduce_system_prompt: str, paths, persona_text: str) -> None:
    """REDUCE: Synthesize the calendar ledger into a schedule-focused digest."""
    reduce_start = time.time()
    print("📉 Starting Calendar REDUCE Phase...")

    ledger, _ = load_ledger(paths.calendar_ledger_file)
    if not ledger:
        print(f"❌ Ledger not found or empty at '{paths.calendar_ledger_file}'. Run MAP phase first.")
        return

    for warning in check_schema_consistency(ledger):
        print(f"   ⚠️  {warning}")

    ledger = compact_ledger(ledger, llm, build_compact_system_prompt(persona_text), retention_days=30, count_key=COUNT_KEY)
    save_ledger(paths.calendar_ledger_file, ledger)

    ledger_context = render_ledger_as_text(ledger)
    print(f"🧠 Synthesizing {len(ledger)}-entry context window...")

    try:
        summary = call_with_retry(
            llm.chat,
            messages=[
                {"role": "system", "content": reduce_system_prompt},
                {"role": "user", "content": f"TODAY'S DATE: {format_today()}\n\n---\n\nChronological Calendar Ledger:\n\n{ledger_context}"},
            ],
        )
    except Exception as e:
        print(f"❌ Calendar REDUCE phase failed after retries: {e}")
        return

    history_path = save_digest(summary, paths.calendar_summary_file, paths.history_dir)

    reduce_time = time.time() - reduce_start
    print(f"✅ Calendar REDUCE Phase completed in {reduce_time:.1f}s.")
    print(f"   Summary saved to: {paths.calendar_summary_file}")
    print(f"   History copy: {history_path}")
    print(f"\n{'─' * 60}")
    print("   Current 30-Day Calendar Digest")
    print(f"{'─' * 60}\n")
    print(summary)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments for provider, model, and phase selection."""
    parser = argparse.ArgumentParser(
        description="Calendar Triage Agent — Map-Reduce Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python calendar_agent.py\n"
            "  python calendar_agent.py --provider anthropic --model claude-sonnet-4-20250514\n"
            "  python calendar_agent.py --map-only\n"
            "  python calendar_agent.py --reduce-only\n"
        ),
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"], help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="llama3", help="Model name (default: llama3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--map-only", action="store_true", help="Run only the MAP phase (extract signals from calendar)")
    parser.add_argument("--reduce-only", action="store_true", help="Run only the REDUCE phase (synthesize existing ledger)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers for MAP phase (default: 4)")
    parser.add_argument("--digest-days", type=int, default=None, help="Cap the digest to the most recent N days found in the calendar (default: all)")
    parser.add_argument("--holdout-days", type=int, default=None, help="Hold out the most recent N days from this run so a later run (without this flag) picks them up as a 'hot' pass")
    parser.add_argument("--hot-input", default=None, help="Path to an additional .ics file to process as newly-arrived 'hot' data")
    parser.add_argument(
        "--tenant",
        default=tenant_paths.DEFAULT_TENANT,
        help="Tenant ID — resolves data/output paths under data/tenants/<id>/ and "
        "output/tenants/<id>/ (default: 'default', today's existing ./data/./output/ layout)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = tenant_paths.for_tenant(args.tenant)

    print(f"🤖 Initializing LLM: {args.provider}/{args.model}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Workers: {args.workers}")
    print(f"   Tenant: {paths.tenant_id}\n")

    persona_text = load_persona(paths.persona_file)
    config = load_tenant_config(paths.tenant_config_file)
    map_system_prompt = build_map_system_prompt(persona_text, use_persona=config.get("use_persona_in_map", True))
    reduce_system_prompt = build_reduce_system_prompt(persona_text)

    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    total_start = time.time()

    map_kwargs = dict(
        max_workers=args.workers,
        digest_days=args.digest_days,
        holdout_days=args.holdout_days,
        hot_input=args.hot_input,
        map_variant="persona" if config.get("use_persona_in_map", True) else "no_persona",
    )

    if args.reduce_only:
        run_reduce_phase(llm, reduce_system_prompt, paths, persona_text)
    elif args.map_only:
        run_map_phase(llm, map_system_prompt, paths, **map_kwargs)
    else:
        map_success = run_map_phase(llm, map_system_prompt, paths, **map_kwargs)
        if map_success:
            run_reduce_phase(llm, reduce_system_prompt, paths, persona_text)

    total_time = time.time() - total_start
    print(f"\n⏱️  Total elapsed: {total_time:.1f}s")


if __name__ == "__main__":
    main()
