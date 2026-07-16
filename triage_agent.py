"""
Email Triage Agent — Map-Reduce Pipeline
=========================================
Processes emails through a two-phase pipeline to build a persistent
30-day compressed memory ledger, then synthesizes actionable insights.

MAP phase:  Each day's emails → structured signals
            (deadlines, decisions, action items, thread progressions)
REDUCE phase: Full 30-day ledger → executive digest with trajectory tracking

The ledger is persistent and supports incremental updates — re-running
the script only processes newly added emails.

Usage:
    python triage_agent.py
    python triage_agent.py --provider anthropic --model claude-sonnet-4-20250514
    python triage_agent.py --provider google --model gemini-2.5-flash
    python triage_agent.py --map-only
    python triage_agent.py --reduce-only
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from email_parser import load_inbox, group_by_date
from llm import create_llm


# ─── Path Configuration ──────────────────────────────────────────────────────

INBOX_DIR = "./data/inbox/"
OUTPUT_DIR = "./output/"
LEDGER_FILE = os.path.join(OUTPUT_DIR, "rolling_ledger.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "current_30day_summary.md")


# ─── System Prompts ──────────────────────────────────────────────────────────

MAP_SYSTEM_PROMPT = (
    "You are a strict data compression node. You will receive a batch of emails "
    "from a single day. Extract the following signals:\n"
    "1. **Deadlines**: Any hard deadlines mentioned or implied (include dates).\n"
    "2. **Decisions**: Critical decisions that were reached or proposed.\n"
    "3. **Action items**: Tasks assigned to or promised by anyone (include who).\n"
    "4. **Thread progressions**: For ongoing conversation threads, note how "
    "the thread advanced (e.g., 'Aurora caching discussion moved from RFC "
    "review to implementation approval').\n\n"
    "Output strictly valid JSON matching this schema:\n"
    "{\n"
    '  "deadlines": [{"description": "...", "date": "...", "source_subject": "..."}],\n'
    '  "decisions": [{"description": "...", "source_subject": "..."}],\n'
    '  "action_items": [{"description": "...", "owner": "...", "source_subject": "..."}],\n'
    '  "thread_progressions": [{"thread": "...", "progression": "..."}]\n'
    "}\n\n"
    "If a category has no entries for the day, use an empty array.\n"
    "Do not write any markdown wrappers, conversational pleasantries, or extra text."
)

REDUCE_SYSTEM_PROMPT = (
    "You are an expert executive analyst assistant. You are given a chronological "
    "sequence of highly compressed daily email summaries covering a 30-day window.\n\n"
    "Synthesize this timeline to produce an executive digest covering:\n\n"
    "## 1. OVERARCHING TRAJECTORY\n"
    "What major objectives or projects dominated this month? How have key "
    "conversations and initiatives evolved over time?\n\n"
    "## 2. PROACTIVE DEADLINE ANALYSIS\n"
    "Identify upcoming deliverables due in the next week or two. List specific "
    "actions that MUST be addressed TODAY to prevent slipping on those timelines.\n\n"
    "## 3. CONVERSATION PROGRESSION\n"
    "Track how important email threads have evolved — what started as a question, "
    "became a decision, became an action item. Highlight threads that stalled.\n\n"
    "## 4. HIGH PRIORITY RADAR\n"
    "Open threads or action items that appear neglected across multiple days. "
    "Flag items that were mentioned early in the window but never resolved.\n\n"
    "Write a concise, high-impact executive digest in markdown format. "
    "Be specific — name people, dates, and project names."
)


# ─── Helper Functions ─────────────────────────────────────────────────────────


def format_email_batch(emails: list[dict]) -> str:
    """Format a batch of parsed emails into a context string for the LLM.

    Each email is rendered with its headers and decoded body, separated
    by dividers for clear delineation.
    """
    parts = []
    for i, em in enumerate(emails, 1):
        parts.append(
            f"--- Email {i} ---\n"
            f"From: {em['from']}\n"
            f"To: {em['to']}\n"
            f"Subject: {em['subject']}\n"
            f"Date: {em['date']}\n"
            f"Thread: {em['thread_id']}\n"
            f"Body:\n{em['body']}\n"
        )
    return "\n".join(parts)


def load_existing_ledger() -> tuple:
    """Load existing ledger and extract the set of already-processed days.

    Returns:
        (ledger_entries, processed_days_set)
    """
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        processed_days = {entry["day"] for entry in ledger}
        return ledger, processed_days
    return [], set()


# ─── MAP Phase ────────────────────────────────────────────────────────────────


def _map_single_day(llm, day: str, batch: list[dict]) -> dict | None:
    """Process a single day's emails through the LLM. Thread-safe.

    Returns:
        A ledger entry dict on success, None on failure.
    """
    context = format_email_batch(batch)
    start = time.time()

    try:
        structured_delta = llm.chat_json(
            messages=[
                {"role": "system", "content": MAP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Emails for {day}:\n\n{context}"},
            ]
        )
        elapsed = time.time() - start
        print(f"   ✅ {day} — {len(batch)} emails — {elapsed:.1f}s")
        return {
            "day": day,
            "email_count": len(batch),
            "delta": structured_delta,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ⚠️  {day} — FAILED ({elapsed:.1f}s): {e}")
        return None


def run_map_phase(llm, max_workers: int = 4):
    """MAP: Process emails day-by-day into structured signal deltas.

    Uses ThreadPoolExecutor for concurrent processing — each day's
    LLM call is independent. For Ollama, requests are queued server-side
    but overlapping I/O is still beneficial. For API providers (Anthropic,
    Google), this gives true parallel speedup.

    Args:
        llm: A BaseLLM instance from the abstraction layer.
        max_workers: Number of concurrent threads (default: 4).

    Returns:
        True if processing succeeded, False otherwise.
    """
    map_start = time.time()
    print("🚀 Starting MAP Phase...")

    # Load and parse all emails from inbox
    t0 = time.time()
    print(f"📬 Loading emails from {INBOX_DIR}...")
    emails = load_inbox(INBOX_DIR)
    if not emails:
        print("❌ No emails found. Check the inbox directory.")
        return False
    parse_time = time.time() - t0
    print(f"   Found {len(emails)} emails. (parsed in {parse_time:.1f}s)")

    # Group by date for day-by-day processing
    daily_batches = group_by_date(emails)
    print(f"   Spanning {len(daily_batches)} days.\n")

    # Load existing ledger for incremental updates
    ledger, processed_days = load_existing_ledger()
    if processed_days:
        print(
            f"📋 Existing ledger found: {len(processed_days)} days already processed."
        )

    # Filter to only unprocessed days
    days_to_process = [
        (day, batch)
        for day, batch in daily_batches.items()
        if day not in processed_days
    ]
    skipped = len(daily_batches) - len(days_to_process)
    if skipped > 0:
        print(f"   ⏭️  Skipping {skipped} already-processed days.")

    if not days_to_process:
        print("   Nothing new to process.")
        # Still save ledger (idempotent)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
        return True

    print(
        f"\n🔄 Processing {len(days_to_process)} days "
        f"with {max_workers} worker(s)...\n"
    )

    # Concurrent MAP: each day is an independent task
    # Save ledger incrementally as each future completes for resilience
    succeeded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_day = {
            executor.submit(_map_single_day, llm, day, batch): day
            for day, batch in days_to_process
        }
        for future in as_completed(future_to_day):
            result = future.result()
            if result is not None:
                ledger.append(result)
                # Keep ledger sorted chronologically
                ledger.sort(key=lambda r: r["day"])
                # Save immediately
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(LEDGER_FILE, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, indent=2)
                succeeded += 1

    total_time = time.time() - map_start
    print(f"\n✅ MAP Phase completed in {total_time:.1f}s.")
    print(
        f"   {succeeded}/{len(days_to_process)} days succeeded. "
        f"Ledger: {LEDGER_FILE} ({len(ledger)} total days)\n"
    )
    return True


# ─── REDUCE Phase ─────────────────────────────────────────────────────────────


def run_reduce_phase(llm):
    """REDUCE: Synthesize the 30-day ledger into an executive digest.

    Reads the full rolling ledger and passes it to the LLM for synthesis.
    Produces a markdown summary tracking trajectories, deadlines, and
    conversation progressions.

    Args:
        llm: A BaseLLM instance from the abstraction layer.
    """
    reduce_start = time.time()
    print("📉 Starting REDUCE Phase...")

    if not os.path.exists(LEDGER_FILE):
        print(f"❌ Ledger not found at '{LEDGER_FILE}'. Run MAP phase first.")
        return

    with open(LEDGER_FILE, "r", encoding="utf-8") as f:
        ledger_data = json.load(f)

    ledger_context = json.dumps(ledger_data, indent=2)

    print(f"🧠 Synthesizing {len(ledger_data)}-day context window...")

    try:
        summary = llm.chat(
            messages=[
                {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Chronological 30-Day Email Ledger:\n\n{ledger_context}"
                    ),
                },
            ]
        )
    except Exception as e:
        print(f"❌ REDUCE phase failed: {e}")
        return

    # Save the summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    reduce_time = time.time() - reduce_start
    print(f"✅ REDUCE Phase completed in {reduce_time:.1f}s.")
    print(f"   Summary saved to: {SUMMARY_FILE}")
    print(f"\n{'─' * 60}")
    print("   Current 30-Day Executive Digest")
    print(f"{'─' * 60}\n")
    print(summary)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments for provider, model, and phase selection."""
    parser = argparse.ArgumentParser(
        description="Email Triage Agent — Map-Reduce Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python triage_agent.py\n"
            "  python triage_agent.py --provider anthropic --model claude-sonnet-4-20250514\n"
            "  python triage_agent.py --provider google --model gemini-2.5-flash\n"
            "  python triage_agent.py --map-only\n"
            "  python triage_agent.py --reduce-only\n"
        ),
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "anthropic", "google"],
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Model name (default: llama3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Run only the MAP phase (extract signals from emails)",
    )
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Run only the REDUCE phase (synthesize existing ledger)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers for MAP phase (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🤖 Initializing LLM: {args.provider}/{args.model}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Workers: {args.workers}\n")

    llm = create_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    total_start = time.time()

    if args.reduce_only:
        run_reduce_phase(llm)
    elif args.map_only:
        run_map_phase(llm, max_workers=args.workers)
    else:
        # Full pipeline: MAP → REDUCE
        map_success = run_map_phase(llm, max_workers=args.workers)
        if map_success:
            run_reduce_phase(llm)

    total_time = time.time() - total_start
    print(f"\n⏱️  Total elapsed: {total_time:.1f}s")


if __name__ == "__main__":
    main()
