"""
Notes Triage Agent — Per-Note Map, Single Reduce
==================================================
Processes notes into a persistent ledger, then synthesizes actionable
insights — same reliability posture as triage_agent.py (email) and
calendar_agent.py (calendar): persona injection, deterministic-where-
possible + LLM-where-judgment, retry, schema validation, digest history,
ledger compaction. Fully independent — own parser, own ledger, own digest.
Cross-referencing with email/calendar/tasks is deliberately left to a
future orchestrator, not built into this agent.

Unlike email (a flood, ~500/month) or calendar (a flood, ~75 events/month),
notes are sparse — a handful of documents a month, each already a complete,
information-dense unit (an RFC, a decision log, a call summary). Day-batched
MAP would mean mostly-empty days; the natural MAP unit here is one note.

MAP phase:  Each note → deterministic checklist stats (open/done counts,
            staleness of open items — pure Python, no LLM) + an LLM call
            for the qualitative layer: note type, decisions made, which
            open items still matter, key people mentioned.
REDUCE phase: All notes → a notes-focused digest (decisions on record,
            still-open items worth revisiting, relationship context).

Usage:
    python notes_agent.py
    python notes_agent.py --provider anthropic --model claude-sonnet-4-20250514
    python notes_agent.py --map-only
    python notes_agent.py --reduce-only
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ledger import load_ledger, save_ledger, save_digest, validate_schema, compact_ledger
from llm import create_llm, call_with_retry
from notes_parser import load_notes
from persona import load_persona


# ─── Path Configuration ──────────────────────────────────────────────────────

NOTES_DIR = "./data/notes/"
OUTPUT_DIR = "./output/"
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
LEDGER_FILE = os.path.join(OUTPUT_DIR, "notes_rolling_ledger.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "current_30day_notes_summary.md")

COUNT_KEY = "item_count"

# Only the list-shaped fields are schema-validated; note_type is a free-form
# string with a safe default, not worth failing a whole note's MAP over.
MAP_SCHEMA = {
    "decisions": ["description", "priority"],
    "open_items_still_relevant": ["text", "priority"],
    "key_people": ["name"],
}

COMPACT_SYSTEM_PROMPT = (
    "You are compressing a week's worth of per-note triage deltas into a single "
    "weekly delta, using the exact same JSON schema. Merge duplicate/similar items, "
    "keep genuinely distinct ones, and preserve specifics (names, dates, priorities).\n\n"
    "Output strictly valid JSON matching this schema:\n"
    "{\n"
    '  "note_type": "mixed",\n'
    '  "decisions": [{"description": "...", "priority": "P0-P4"}],\n'
    '  "open_items_still_relevant": [{"text": "...", "why": "...", "priority": "P0-P4"}],\n'
    '  "key_people": [{"name": "...", "context": "..."}]\n'
    "}\n\n"
    "Do not write any markdown wrappers, conversational pleasantries, or extra text."
)


# ─── Prompt Builders (persona-injected) ───────────────────────────────────────


def build_map_system_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are a notes triage node for the operator profiled above. You will "
        "receive one note (a meeting note, RFC, decision log, checklist, or call "
        "summary), plus deterministic checklist stats already computed in code "
        "(open/done counts, days each open item has gone unchecked — do not "
        "recompute these, they're ground truth). Your job is the layer arithmetic "
        "can't do:\n\n"
        "1. **Note type**: classify as one of standup, one_on_one, rfc_design_doc, "
        "decision_log, checklist, call_notes, or other.\n"
        "2. **Decisions**: decisions actually recorded as made in this note (not "
        "open questions), weighted using the profile's people-priority list.\n"
        "3. **Open items still relevant**: of the open checklist items given to you, "
        "which ones still matter given how much time has passed and the note's "
        "content — an old item can be superseded or no longer relevant; use "
        "judgment, don't just restate every open item.\n"
        "4. **Key people**: names mentioned in this note and the context they're "
        "mentioned in (e.g. a promotion timeline, a blocker they own).\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "note_type": "...",\n'
        '  "decisions": [{"description": "...", "priority": "P0-P4"}],\n'
        '  "open_items_still_relevant": [{"text": "...", "why": "...", "priority": "P0-P4"}],\n'
        '  "key_people": [{"name": "...", "context": "..."}]\n'
        "}\n\n"
        "If a category has no entries, use an empty array. Do not write any markdown "
        "wrappers, conversational pleasantries, or extra text."
    )


def build_reduce_system_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are the notes half of the operator's digest, for the operator profiled "
        "above. You are given a chronological sequence of per-note triage signals "
        "covering up to a 30-day window. This complements separate email and "
        "calendar digests — focus only on what's recorded in notes, don't try to "
        "cover inbox or schedule content.\n\n"
        "Synthesize this into a concise digest covering:\n\n"
        "## 1. DECISIONS ON RECORD\n"
        "What's actually been decided, per the notes — not proposed, decided.\n\n"
        "## 2. STILL-OPEN ITEMS WORTH REVISITING\n"
        "Open checklist items that have gone unchecked for a while and still "
        "matter, weighted by the profile's priority list. Use the deterministic "
        "staleness data as ground truth for how long something's been open — "
        "don't guess.\n\n"
        "## 3. RELATIONSHIP CONTEXT\n"
        "Notable people-related threads across notes (career growth conversations, "
        "team health flags, recurring blockers someone owns).\n\n"
        "Follow the profile's honesty rules: say if the data looks stale or sparse, "
        "flag assumptions, and don't hide contradictions. Write a concise digest in "
        "markdown.\n\n"
        "Important: write the digest itself, addressed directly to the operator in "
        "second person. Do NOT describe, summarize, or explain the JSON data format "
        "or structure of what you were given — that is input data to read, not "
        "something to report on."
    )


# ─── Deterministic Checklist Stats (pure Python — no LLM for arithmetic) ─────


def compute_note_stats(note: dict, reference_date=None) -> dict:
    """Deterministic checklist stats for a single note.

    Staleness is anchored on the note's creation date, since this dataset
    doesn't track last-edited separately — a known simplification (a real
    platform adapter, e.g. Notion, would use last_edited_time instead, since
    a note revisited yesterday isn't stale even if first created weeks ago).

    Args:
        note: Parsed note dict (from notes_parser.load_notes).
        reference_date: "Today" for staleness math; defaults to real now().

    Returns:
        {"checklist_open", "checklist_done", "stale_open_items"}.
    """
    reference_date = reference_date or datetime.now().date()
    open_items = [c for c in note["checklist_items"] if not c["done"]]
    done_items = [c for c in note["checklist_items"] if c["done"]]

    stale_open_items = []
    if note.get("created_at") and note["created_at"] != "unknown":
        created = datetime.strptime(note["created_at"], "%Y-%m-%d").date()
        days_open = (reference_date - created).days
        stale_open_items = [
            {"text": item["text"], "days_open": days_open} for item in open_items
        ]

    return {
        "checklist_open": len(open_items),
        "checklist_done": len(done_items),
        "stale_open_items": stale_open_items,
    }


# ─── Helper Functions ─────────────────────────────────────────────────────────


def format_note(note: dict, stats: dict) -> str:
    """Format a single note plus its precomputed stats for the LLM."""
    return (
        f"Title: {note['title']}\n"
        f"Created: {note['created_at']}\n"
        f"Body:\n{note['body']}\n\n"
        f"Precomputed checklist stats (ground truth, do not recompute):\n"
        f"{json.dumps(stats, indent=2)}"
    )


# ─── MAP Phase ────────────────────────────────────────────────────────────────


def _map_single_note(llm, note: dict, map_system_prompt: str) -> dict | None:
    """Process a single note through the LLM. Thread-safe.

    Returns:
        A ledger entry dict on success, None on failure.
    """
    stats = compute_note_stats(note)
    context = format_note(note, stats)
    start = time.time()

    def _call():
        delta = llm.chat_json(
            messages=[
                {"role": "system", "content": map_system_prompt},
                {"role": "user", "content": context},
            ]
        )
        errors = validate_schema(delta, MAP_SCHEMA)
        if errors:
            raise ValueError(f"Invalid MAP output: {errors}")
        return delta

    try:
        delta = call_with_retry(_call)
        elapsed = time.time() - start
        print(f"   ✅ {note['note_id']} — {elapsed:.1f}s")
        return {
            "day": note["created_at"],
            "note_id": note["note_id"],
            COUNT_KEY: stats["checklist_open"] + stats["checklist_done"],
            "stats": stats,
            "delta": delta,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ⚠️  {note['note_id']} — FAILED after retries ({elapsed:.1f}s): {e}")
        return None


def run_map_phase(llm, map_system_prompt: str, max_workers: int = 4) -> bool:
    """MAP: Process notes one-by-one into structured signal deltas."""
    map_start = time.time()
    print("🚀 Starting Notes MAP Phase...")

    t0 = time.time()
    print(f"📝 Loading notes from {NOTES_DIR}...")
    notes = load_notes(NOTES_DIR)
    if not notes:
        print("❌ No notes found. Check the notes directory.")
        return False
    parse_time = time.time() - t0
    print(f"   Found {len(notes)} notes. (parsed in {parse_time:.1f}s)\n")

    # Resume tracking is note_id-based, not day-based — unlike email/calendar,
    # two notes could share a creation date, and "day" here is a chronological
    # sort/compaction key, not a uniqueness guarantee.
    ledger, _ = load_ledger(LEDGER_FILE)
    processed_note_ids = {entry["note_id"] for entry in ledger if "note_id" in entry}
    if processed_note_ids:
        print(f"📋 Existing ledger found: {len(processed_note_ids)} notes already processed.")

    notes_to_process = [n for n in notes if n["note_id"] not in processed_note_ids]
    skipped = len(notes) - len(notes_to_process)
    if skipped > 0:
        print(f"   ⏭️  Skipping {skipped} already-processed notes.")

    if not notes_to_process:
        print("   Nothing new to process.")
        save_ledger(LEDGER_FILE, ledger)
        return True

    print(f"\n🔄 Processing {len(notes_to_process)} notes with {max_workers} worker(s)...\n")

    succeeded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_note = {
            executor.submit(_map_single_note, llm, note, map_system_prompt): note
            for note in notes_to_process
        }
        for future in as_completed(future_to_note):
            result = future.result()
            if result is not None:
                ledger.append(result)
                save_ledger(LEDGER_FILE, ledger)
                succeeded += 1

    total_time = time.time() - map_start
    print(f"\n✅ Notes MAP Phase completed in {total_time:.1f}s.")
    print(
        f"   {succeeded}/{len(notes_to_process)} notes succeeded. "
        f"Ledger: {LEDGER_FILE} ({len(ledger)} total entries)\n"
    )
    return True


# ─── REDUCE Phase ─────────────────────────────────────────────────────────────


def _render_notes_stats(stats: dict, indent: str = "") -> list[str]:
    """Render one note's deterministic checklist stats as text lines."""
    lines = [
        f"{indent}- Checklist: {stats.get('checklist_open', 0)} open, "
        f"{stats.get('checklist_done', 0)} done"
    ]
    for item in stats.get("stale_open_items", []):
        lines.append(f"{indent}  - OPEN {item['days_open']}d: {item['text']}")
    return lines


def render_ledger_as_text(ledger: list[dict]) -> str:
    """Render the ledger as readable text instead of raw JSON.

    Smaller models tend to fall into "describe this JSON" pattern-completion
    when handed a large literal JSON blob as context — flattening it to
    prose keeps the model focused on synthesizing content.
    """
    lines = []
    for entry in ledger:
        label = f"Week of {entry['day']}" if entry.get("compacted") else entry["day"]
        note_ref = "" if entry.get("compacted") else f" — {entry.get('note_id', '')}"
        lines.append(f"### {label}{note_ref}")

        if entry.get("stats"):
            lines.extend(_render_notes_stats(entry["stats"]))
        note_ids_covered = entry.get("note_ids_covered", [])
        if note_ids_covered:
            lines.append(f"  Notes covered: {', '.join(note_ids_covered)}")
        for day, day_stats in entry.get("stats_by_day", {}).items():
            lines.append(f"  {day}:")
            lines.extend(_render_notes_stats(day_stats, indent="  "))

        delta = entry.get("delta", {})
        if delta.get("note_type"):
            lines.append(f"- Type: {delta['note_type']}")
        for item in delta.get("decisions", []):
            lines.append(f"- DECISION [{item.get('priority', '?')}]: {item.get('description')}")
        for item in delta.get("open_items_still_relevant", []):
            lines.append(
                f"- STILL RELEVANT [{item.get('priority', '?')}]: {item.get('text')} — {item.get('why', '')}"
            )
        for item in delta.get("key_people", []):
            lines.append(f"- PEOPLE: {item.get('name')} — {item.get('context', '')}")

        lines.append("")
    return "\n".join(lines)


def run_reduce_phase(llm, reduce_system_prompt: str) -> None:
    """REDUCE: Synthesize the notes ledger into a notes-focused digest."""
    reduce_start = time.time()
    print("📉 Starting Notes REDUCE Phase...")

    ledger, _ = load_ledger(LEDGER_FILE)
    if not ledger:
        print(f"❌ Ledger not found or empty at '{LEDGER_FILE}'. Run MAP phase first.")
        return

    ledger = compact_ledger(ledger, llm, COMPACT_SYSTEM_PROMPT, retention_days=30, count_key=COUNT_KEY)
    save_ledger(LEDGER_FILE, ledger)

    ledger_context = render_ledger_as_text(ledger)
    print(f"🧠 Synthesizing {len(ledger)}-entry context window...")

    try:
        summary = call_with_retry(
            llm.chat,
            messages=[
                {"role": "system", "content": reduce_system_prompt},
                {"role": "user", "content": f"Chronological Notes Ledger:\n\n{ledger_context}"},
            ],
        )
    except Exception as e:
        print(f"❌ Notes REDUCE phase failed after retries: {e}")
        return

    history_path = save_digest(summary, SUMMARY_FILE, HISTORY_DIR)

    reduce_time = time.time() - reduce_start
    print(f"✅ Notes REDUCE Phase completed in {reduce_time:.1f}s.")
    print(f"   Summary saved to: {SUMMARY_FILE}")
    print(f"   History copy: {history_path}")
    print(f"\n{'─' * 60}")
    print("   Current 30-Day Notes Digest")
    print(f"{'─' * 60}\n")
    print(summary)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments for provider, model, and phase selection."""
    parser = argparse.ArgumentParser(
        description="Notes Triage Agent — Per-Note Map, Single Reduce",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python notes_agent.py\n"
            "  python notes_agent.py --provider anthropic --model claude-sonnet-4-20250514\n"
            "  python notes_agent.py --map-only\n"
            "  python notes_agent.py --reduce-only\n"
        ),
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"], help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="llama3", help="Model name (default: llama3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--map-only", action="store_true", help="Run only the MAP phase (extract signals from notes)")
    parser.add_argument("--reduce-only", action="store_true", help="Run only the REDUCE phase (synthesize existing ledger)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers for MAP phase (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🤖 Initializing LLM: {args.provider}/{args.model}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Workers: {args.workers}\n")

    persona_text = load_persona()
    map_system_prompt = build_map_system_prompt(persona_text)
    reduce_system_prompt = build_reduce_system_prompt(persona_text)

    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    total_start = time.time()

    if args.reduce_only:
        run_reduce_phase(llm, reduce_system_prompt)
    elif args.map_only:
        run_map_phase(llm, map_system_prompt, max_workers=args.workers)
    else:
        map_success = run_map_phase(llm, map_system_prompt, max_workers=args.workers)
        if map_success:
            run_reduce_phase(llm, reduce_system_prompt)

    total_time = time.time() - total_start
    print(f"\n⏱️  Total elapsed: {total_time:.1f}s")


if __name__ == "__main__":
    main()
