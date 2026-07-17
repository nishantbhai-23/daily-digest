"""
Email Triage Agent — Map-Reduce Pipeline
=========================================
Processes emails through a two-phase pipeline to build a persistent
30-day compressed memory ledger, then synthesizes actionable insights.

MAP phase:  Each day's emails → structured signals
            (deadlines, decisions, action items, thread progressions)
REDUCE phase: Full 30-day ledger → executive digest with trajectory tracking

The ledger is persistent and supports incremental updates — re-running
the script only processes newly added emails. Old entries beyond the
retention window are compacted into weekly rollups so REDUCE cost doesn't
grow unbounded. Both phases are personalized via data/persona.md — a
profile of the operator this digest serves — so triage priority and
digest structure reflect their actual stated priorities, not generic
defaults.

Usage:
    python triage_agent.py
    python triage_agent.py --provider anthropic --model claude-sonnet-4-20250514
    python triage_agent.py --provider google --model gemini-2.5-flash
    python triage_agent.py --map-only
    python triage_agent.py --reduce-only
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from email_parser import load_inbox, group_by_date
from ledger import load_ledger, save_ledger, save_digest, validate_schema, compact_ledger, format_today
from llm import create_llm, call_with_retry
from persona import load_persona
from tenant_config import load_tenant_config


# ─── Path Configuration ──────────────────────────────────────────────────────

INBOX_DIR = "./data/inbox/"
OUTPUT_DIR = "./output/"
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
LEDGER_FILE = os.path.join(OUTPUT_DIR, "rolling_ledger.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "current_30day_summary.md")

COUNT_KEY = "email_count"

MAP_SCHEMA = {
    "deadlines": ["description", "priority"],
    "decisions": ["description", "priority"],
    "action_items": ["description", "priority"],
    "thread_progressions": ["thread", "progression"],
}

# Used when tenant_config's "use_persona_in_map" is False — no priority field,
# since assigning P0-P4 requires the persona's people-priority list.
MAP_SCHEMA_NO_PERSONA = {
    "deadlines": ["description"],
    "decisions": ["description"],
    "action_items": ["description"],
    "thread_progressions": ["thread", "progression"],
}

COMPACT_SYSTEM_PROMPT = (
    "You are compressing a week's worth of daily email-triage deltas into a single "
    "weekly delta, using the exact same JSON schema. Merge duplicate/similar items, "
    "keep genuinely distinct ones, and preserve specifics (names, dates, subjects, "
    "priority levels).\n\n"
    "Output strictly valid JSON matching this schema:\n"
    "{\n"
    '  "deadlines": [{"description": "...", "date": "...", "source_subject": "...", "priority": "P0-P4"}],\n'
    '  "decisions": [{"description": "...", "source_subject": "...", "priority": "P0-P4"}],\n'
    '  "action_items": [{"description": "...", "owner": "...", "source_subject": "...", "priority": "P0-P4"}],\n'
    '  "thread_progressions": [{"thread": "...", "progression": "..."}]\n'
    "}\n\n"
    "Do not write any markdown wrappers, conversational pleasantries, or extra text."
)


# ─── Prompt Builders (persona-injected) ───────────────────────────────────────


def build_map_system_prompt(persona_text: str, use_persona: bool = True) -> str:
    """Build the MAP system prompt.

    Args:
        persona_text: The operator profile text.
        use_persona: When False, produces a persona-free extraction-only
            prompt — no priority tagging, no persona-driven noise judgment.
            REDUCE always gets full persona regardless and does its own
            priority-weighting reasoning over the resulting ledger, so this
            only changes what MAP itself is asked to judge (see
            tenant_config.py's "use_persona_in_map").
    """
    priority_field = ', "priority": "P0-P4"' if use_persona else ""
    schema = (
        "{\n"
        f'  "deadlines": [{{"description": "...", "date": "...", "source_subject": "..."{priority_field}}}],\n'
        f'  "decisions": [{{"description": "...", "source_subject": "..."{priority_field}}}],\n'
        f'  "action_items": [{{"description": "...", "owner": "...", "source_subject": "..."{priority_field}}}],\n'
        '  "thread_progressions": [{"thread": "...", "progression": "..."}]\n'
        "}\n\n"
    )

    if use_persona:
        header = f"{persona_text}\n\n---\n\n"
        role = (
            "You are an email triage node for the operator profiled above. You will "
            "receive a batch of emails from a single day. Extract the following signals, "
            "using the profile's \"People who matter\" priority weighting (P0-P4) as your "
            "default for each item — override based on content when the email itself "
            "signals otherwise:\n\n"
        )
        noise_instruction = (
            "Do NOT extract signals from newsletters, marketing email, automated "
            "notifications, or threads whose only content is 'FYI' — per the profile, "
            "these should not be surfaced at all.\n\n"
        )
    else:
        header = ""
        role = (
            "You are an email extraction node. You will receive a batch of emails from a "
            "single day. Extract the following signals — structural extraction only, not "
            "judgment about importance or priority:\n\n"
        )
        noise_instruction = (
            "Do NOT extract signals from newsletters, marketing email, automated "
            "notifications, or threads whose only content is 'FYI' — these are never "
            "actionable signal regardless of recipient.\n\n"
        )

    return (
        f"{header}"
        f"{role}"
        "1. **Deadlines**: Any hard deadlines mentioned or implied (include dates).\n"
        "2. **Decisions**: Critical decisions that were reached or proposed.\n"
        "3. **Action items**: Tasks assigned to or promised by anyone (include who).\n"
        "4. **Thread progressions**: For ongoing conversation threads, note how "
        "the thread advanced (e.g., 'a vendor renewal conversation moved from routine "
        "check-in to a budget concern').\n\n"
        f"{noise_instruction}"
        "Output strictly valid JSON matching this schema:\n"
        f"{schema}"
        "If a category has no entries for the day, use an empty array. Do not write any "
        "markdown wrappers, conversational pleasantries, or extra text."
    )


def build_reduce_system_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are the email half of the operator's daily digest, for the operator "
        "profiled above. You are given a chronological sequence of highly compressed "
        "daily email summaries covering up to a 30-day window. This complements a "
        "separate calendar digest — focus on email content and threads, not the "
        "operator's schedule.\n\n"
        "You will be given today's actual date — use it as ground truth for anything "
        "date-relative (\"today\", \"this week\", deadlines). Do not infer today's date "
        "from the ledger content itself (e.g. the earliest or most recent day present) "
        "— that has produced wrong dates before.\n\n"
        "Per the profile, a great digest answers three questions in under 90 seconds:\n\n"
        "## 1. WHAT NEEDS ME TODAY\n"
        "Not what's interesting — what requires the operator's judgment, signature, or "
        "reply before end of day. Weight by the profile's people-priority list "
        "(P0-P4).\n\n"
        "## 2. WHAT AM I ABOUT TO DROP\n"
        "Threads that have gone quiet, decisions deferred and forgotten, promises made "
        "in email with no corresponding follow-through visible in the data. Per the "
        "profile: flag quiet P0/P1 threads (e.g. an investor gone quiet several "
        "business days), not just loud ones. You will be given a DETERMINISTIC SENDER "
        "STALENESS list computed in code — this is ground truth for who has actually "
        "gone quiet and for how long. Use it directly for this section, weighted by the "
        "profile's people-priority list; do not try to independently guess staleness "
        "from the ledger narrative, and do not omit a quiet P0/P1 sender that appears in "
        "that list.\n\n"
        "## 3. WHAT CAN I DISPATCH RIGHT NOW\n"
        "Replies under 3 sentences, quick yes/no approvals, anything low-effort to "
        "clear. Match the profile's tone rules for any drafted language (short, "
        "direct, no 'circling back', no fluff) — except don't draft anything for the "
        "profile's P0 personal contact, just surface it per the profile's own rule.\n\n"
        "Do not surface newsletters, already-accepted calendar invites, marketing "
        "email, long threads where the operator already had the last word, or anything "
        "whose only content is FYI — per the profile, these are noise, not signal.\n\n"
        "Follow the profile's honesty rules: say if the data looks stale, flag any "
        "assumptions behind a suggested action, and don't hide contradictions between "
        "sources. Write a concise, high-impact digest in markdown. Be specific — name "
        "people, dates, and threads.\n\n"
        "Important: write the digest itself, addressed directly to the operator in "
        "second person, as their morning brief. Do NOT describe, summarize, or explain "
        "the JSON data format, field names, or structure of the ledger you were given — "
        "that is input data for you to read, not something to report on."
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


# ─── Deterministic Sender Staleness ───────────────────────────────────────────
# "Who's gone quiet" is a hard fact (last-contact date, computed from parsed
# headers) — not something the LLM should have to infer by scanning 30 days
# of rendered ledger text and noticing an absence. That's a needle-in-haystack
# task language models are bad at, and it's exactly the failure mode observed
# where a genuinely-stale P0 thread got diluted in favor of a more recent but
# less important mention. Compute it in code, hand REDUCE the answer.

_FROM_HEADER_RE = re.compile(r'^"?([^"<]*)"?\s*<?([\w.+-]+@[\w.-]+)?>?$')


def _parse_from_header(from_header: str) -> tuple[str, str]:
    """Parse a "Name <email>" From header into (name, lowercased email).

    Falls back to the raw header string for both if it doesn't match the
    expected shape, rather than raising — callers already treat "unknown"
    senders as a valid (if degenerate) case.
    """
    from_header = (from_header or "").strip()
    match = _FROM_HEADER_RE.match(from_header)
    if match and match.group(2):
        name = (match.group(1) or "").strip() or match.group(2)
        return name, match.group(2).lower()
    return from_header or "unknown", (from_header or "unknown").lower()


def compute_sender_staleness(emails: list[dict], quiet_threshold_days: int = 3) -> list[dict]:
    """Track last-contact date per sender and flag who's gone quiet.

    Surfaces every sender whose most recent email is at least
    `quiet_threshold_days` before the most recent day present in the
    dataset — including senders with only one email, since a single
    unanswered P0 ask (e.g. an investor's one unreplied request) is exactly
    the kind of quiet thread this is meant to catch. This deliberately does
    not try to filter out low-relevance senders (newsletters, recruiters,
    notifications) by email count — that's a relevance judgment, and the
    REDUCE prompt already instructs the LLM to disregard that category of
    noise using the profile, the same way it does everywhere else.

    Args:
        emails: Parsed email dicts (must have 'from' and 'date_key').
        quiet_threshold_days: Minimum days since last contact to surface.

    Returns:
        List of {"name", "email", "email_count", "first_seen", "last_seen",
        "days_since_last_contact"}, sorted most-quiet first.
    """
    if not emails:
        return []

    activity = {}
    for em in emails:
        day = em.get("date_key", "unknown")
        if day == "unknown":
            continue

        name, email_addr = _parse_from_header(em.get("from", ""))

        entry = activity.setdefault(
            email_addr, {"name": name, "count": 0, "first_seen": day, "last_seen": day}
        )
        entry["count"] += 1
        entry["first_seen"] = min(entry["first_seen"], day)
        entry["last_seen"] = max(entry["last_seen"], day)

    reference_day = max(em["date_key"] for em in emails if em.get("date_key") != "unknown")
    ref_date = datetime.strptime(reference_day, "%Y-%m-%d").date()

    results = []
    for email_addr, entry in activity.items():
        last_date = datetime.strptime(entry["last_seen"], "%Y-%m-%d").date()
        days_since = (ref_date - last_date).days
        if days_since < quiet_threshold_days:
            continue
        results.append({
            "name": entry["name"],
            "email": email_addr,
            "email_count": entry["count"],
            "first_seen": entry["first_seen"],
            "last_seen": entry["last_seen"],
            "days_since_last_contact": days_since,
        })

    results.sort(key=lambda r: -r["days_since_last_contact"])
    return results


def format_staleness_report(staleness: list[dict]) -> str:
    """Render the staleness list as ground-truth text for the REDUCE prompt."""
    if not staleness:
        return "(No senders with an established thread have gone quiet.)"
    lines = []
    for s in staleness:
        lines.append(
            f"- {s['name']} <{s['email']}>: last heard from {s['last_seen']} "
            f"({s['days_since_last_contact']} days ago), {s['email_count']} emails total "
            f"in this window (first: {s['first_seen']})"
        )
    return "\n".join(lines)


# ─── Deterministic Per-Day Email Stats ────────────────────────────────────────
# Thinner than calendar's structured events or notes' checkboxes — free-text
# email doesn't carry much that's reliably extractable without judgment —
# but the envelope should still carry a 'stats' sibling to 'delta' the same
# way calendar/notes ledger entries do, both for consistency and because
# even a little deterministic signal (reply ratio, sender diversity) is
# still cheaper and more trustworthy than asking the LLM to compute it.


def compute_day_email_stats(batch: list[dict]) -> dict:
    """Pure-Python deterministic stats for a day's email batch."""
    senders = set()
    reply_count = 0
    for em in batch:
        _, email_addr = _parse_from_header(em.get("from", ""))
        senders.add(email_addr)
        if em.get("subject", "").strip().lower().startswith(("re:", "fwd:", "fw:")):
            reply_count += 1

    total_chars = sum(len(em.get("body", "")) for em in batch)

    return {
        "unique_senders": len(senders),
        "reply_count": reply_count,
        "new_thread_count": len(batch) - reply_count,
        "avg_body_chars": round(total_chars / len(batch)) if batch else 0,
    }


# ─── Pre-MAP Noise Filter ──────────────────────────────────────────────────────
# Tenant-configured senders/domains never reach an LLM call at all — strictly
# better than asking the model to notice and discard them per day, which costs
# the same tokens either way. The MAP prompt's "don't extract newsletters"
# instruction stays as a secondary defense for content-based noise a sender
# list can't catch (a legitimate sender's FYI-only thread); the two are
# complementary, not redundant.


def filter_blocked_senders(emails: list[dict], config: dict) -> list[dict]:
    """Drop emails from tenant-configured blocked senders/domains.

    Args:
        emails: Parsed email dicts (must have 'from').
        config: Tenant config dict (see tenant_config.py) — reads
            config["map_noise_filter"]["blocked_senders"/"blocked_domains"].

    Returns:
        Emails with blocked senders/domains removed.
    """
    noise_filter = config.get("map_noise_filter", {})
    blocked_senders = {s.lower() for s in noise_filter.get("blocked_senders", [])}
    blocked_domains = {d.lower() for d in noise_filter.get("blocked_domains", [])}

    if not blocked_senders and not blocked_domains:
        return emails

    kept = []
    for em in emails:
        _, email_addr = _parse_from_header(em.get("from", ""))
        domain = email_addr.rsplit("@", 1)[-1] if "@" in email_addr else ""
        if email_addr in blocked_senders or domain in blocked_domains:
            continue
        kept.append(em)
    return kept


# ─── MAP Phase ────────────────────────────────────────────────────────────────


def _map_single_day(llm, day: str, batch: list[dict], map_system_prompt: str, map_schema: dict) -> dict | None:
    """Process a single day's emails through the LLM. Thread-safe.

    Returns:
        A ledger entry dict on success, None on failure.
    """
    stats = compute_day_email_stats(batch)
    context = format_email_batch(batch)
    start = time.time()

    def _call():
        delta = llm.chat_json(
            messages=[
                {"role": "system", "content": map_system_prompt},
                {"role": "user", "content": f"Emails for {day}:\n\n{context}"},
            ]
        )
        errors = validate_schema(delta, map_schema)
        if errors:
            raise ValueError(f"Invalid MAP output: {errors}")
        return delta

    try:
        structured_delta = call_with_retry(_call)
        elapsed = time.time() - start
        print(f"   ✅ {day} — {len(batch)} emails — {elapsed:.1f}s")
        return {
            "day": day,
            COUNT_KEY: len(batch),
            "stats": stats,
            "delta": structured_delta,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ⚠️  {day} — FAILED after retries ({elapsed:.1f}s): {e}")
        return None


def run_map_phase(llm, map_system_prompt: str, map_schema: dict, config: dict, max_workers: int = 4) -> bool:
    """MAP: Process emails day-by-day into structured signal deltas.

    Uses ThreadPoolExecutor for concurrent processing — each day's
    LLM call is independent. For Ollama, requests are queued server-side
    but overlapping I/O is still beneficial. For API providers (Anthropic,
    Google), this gives true parallel speedup.

    Args:
        llm: A BaseLLM instance from the abstraction layer.
        map_system_prompt: The MAP system prompt (persona-injected or not,
            per config["use_persona_in_map"] — selected by the caller).
        map_schema: Schema to validate MAP output against — varies with
            map_system_prompt, since the persona-off prompt doesn't ask for
            a priority field (see build_map_system_prompt).
        config: Tenant config (see tenant_config.py) — used here for the
            pre-MAP noise filter.
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

    # Deterministic pre-MAP filter — blocked senders never reach an LLM call.
    before_count = len(emails)
    emails = filter_blocked_senders(emails, config)
    if before_count != len(emails):
        print(f"   🧹 Filtered {before_count - len(emails)} emails from blocked senders.")

    # Group by date for day-by-day processing
    daily_batches = group_by_date(emails)
    print(f"   Spanning {len(daily_batches)} days.\n")

    # Load existing ledger for incremental updates
    ledger, processed_days = load_ledger(LEDGER_FILE)
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
        save_ledger(LEDGER_FILE, ledger)
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
            executor.submit(_map_single_day, llm, day, batch, map_system_prompt, map_schema): day
            for day, batch in days_to_process
        }
        for future in as_completed(future_to_day):
            result = future.result()
            if result is not None:
                ledger.append(result)
                # Save immediately
                save_ledger(LEDGER_FILE, ledger)
                succeeded += 1

    total_time = time.time() - map_start
    print(f"\n✅ MAP Phase completed in {total_time:.1f}s.")
    print(
        f"   {succeeded}/{len(days_to_process)} days succeeded. "
        f"Ledger: {LEDGER_FILE} ({len(ledger)} total days)\n"
    )
    return True


# ─── REDUCE Phase ─────────────────────────────────────────────────────────────


def _render_email_stats(stats: dict, indent: str = "") -> list[str]:
    """Render one day's deterministic email stats as text lines."""
    return [
        f"{indent}- Senders: {stats.get('unique_senders', 0)} unique | "
        f"Replies: {stats.get('reply_count', 0)} | "
        f"New threads: {stats.get('new_thread_count', 0)} | "
        f"Avg body length: {stats.get('avg_body_chars', 0)} chars"
    ]


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
        lines.append(f"### {label} ({entry.get(COUNT_KEY, 0)} emails)")

        if entry.get("stats"):
            lines.extend(_render_email_stats(entry["stats"]))
        for day, day_stats in entry.get("stats_by_day", {}).items():
            lines.append(f"  {day}:")
            lines.extend(_render_email_stats(day_stats, indent="  "))

        delta = entry.get("delta", {})
        for item in delta.get("deadlines", []):
            lines.append(
                f"- DEADLINE [{item.get('priority', '?')}]: {item.get('description')} "
                f"(date: {item.get('date') or 'unspecified'}, from: {item.get('source_subject', '')})"
            )
        for item in delta.get("decisions", []):
            lines.append(
                f"- DECISION [{item.get('priority', '?')}]: {item.get('description')} "
                f"(from: {item.get('source_subject', '')})"
            )
        for item in delta.get("action_items", []):
            lines.append(
                f"- ACTION [{item.get('priority', '?')}]: {item.get('description')} "
                f"(owner: {item.get('owner') or 'unspecified'}, from: {item.get('source_subject', '')})"
            )
        for item in delta.get("thread_progressions", []):
            lines.append(f"- THREAD '{item.get('thread')}': {item.get('progression')}")

        lines.append("")
    return "\n".join(lines)


def run_reduce_phase(llm, reduce_system_prompt: str) -> None:
    """REDUCE: Synthesize the 30-day ledger into an executive digest.

    Reads the full rolling ledger and passes it to the LLM for synthesis.
    Produces a markdown summary tracking trajectories, deadlines, and
    conversation progressions.

    Args:
        llm: A BaseLLM instance from the abstraction layer.
        reduce_system_prompt: The persona-injected REDUCE system prompt.
    """
    reduce_start = time.time()
    print("📉 Starting REDUCE Phase...")

    ledger, _ = load_ledger(LEDGER_FILE)
    if not ledger:
        print(f"❌ Ledger not found or empty at '{LEDGER_FILE}'. Run MAP phase first.")
        return

    # Collapse anything older than the retention window into weekly rollups
    # before synthesizing, so REDUCE cost doesn't grow unbounded over time.
    ledger = compact_ledger(ledger, llm, COMPACT_SYSTEM_PROMPT, retention_days=30, count_key=COUNT_KEY)
    save_ledger(LEDGER_FILE, ledger)

    ledger_context = render_ledger_as_text(ledger)

    # Deterministic ground truth for "who's gone quiet" — computed from raw
    # inbox headers, not inferred by the LLM from the rendered ledger.
    emails = load_inbox(INBOX_DIR)
    staleness = compute_sender_staleness(emails)
    staleness_report = format_staleness_report(staleness)

    print(f"🧠 Synthesizing {len(ledger)}-entry context window...")

    try:
        summary = call_with_retry(
            llm.chat,
            messages=[
                {"role": "system", "content": reduce_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"TODAY'S DATE: {format_today()}\n\n"
                        f"---\n\n"
                        f"DETERMINISTIC SENDER STALENESS (ground truth — computed in code, "
                        f"not inferred; use this directly for 'what am I about to drop', "
                        f"don't try to re-derive quietness from the ledger below):\n\n"
                        f"{staleness_report}\n\n"
                        f"---\n\n"
                        f"Chronological Email Ledger:\n\n{ledger_context}"
                    ),
                },
            ],
        )
    except Exception as e:
        print(f"❌ REDUCE phase failed after retries: {e}")
        return

    # Save current + a timestamped history copy
    history_path = save_digest(summary, SUMMARY_FILE, HISTORY_DIR)

    reduce_time = time.time() - reduce_start
    print(f"✅ REDUCE Phase completed in {reduce_time:.1f}s.")
    print(f"   Summary saved to: {SUMMARY_FILE}")
    print(f"   History copy: {history_path}")
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
        choices=["ollama", "anthropic", "google", "openrouter", "deepseek"],
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

    persona_text = load_persona()
    config = load_tenant_config()
    use_persona = config.get("use_persona_in_map", True)

    map_system_prompt = build_map_system_prompt(persona_text, use_persona=use_persona)
    map_schema = MAP_SCHEMA if use_persona else MAP_SCHEMA_NO_PERSONA
    reduce_system_prompt = build_reduce_system_prompt(persona_text)

    llm = create_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    total_start = time.time()

    if args.reduce_only:
        run_reduce_phase(llm, reduce_system_prompt)
    elif args.map_only:
        run_map_phase(llm, map_system_prompt, map_schema, config, max_workers=args.workers)
    else:
        # Full pipeline: MAP → REDUCE
        map_success = run_map_phase(llm, map_system_prompt, map_schema, config, max_workers=args.workers)
        if map_success:
            run_reduce_phase(llm, reduce_system_prompt)

    total_time = time.time() - total_start
    print(f"\n⏱️  Total elapsed: {total_time:.1f}s")


if __name__ == "__main__":
    main()
