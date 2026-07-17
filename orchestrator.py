"""
Orchestrator — Unified Chief-of-Staff Daily Brief
====================================================
Reads all four independent source pipelines (email, calendar, notes,
tasks) and produces the actual top-level deliverable: a single daily brief
that highlights what matters, surfaces what's being missed via cross-source
patterns, drafts anything dispatchable in under a minute, and is honest
about data quality — per data/persona.md's own stated rubric.

Staged pipeline, not one mega-prompt, continuing the deterministic-where-
possible / narrow-LLM-calls-where-judgment principle used throughout the
other four pipelines:

Stage 0 (no LLM):   Deterministic cross-reference index — which flagged
                     tasks show up in email/calendar/notes too
                     (cross_reference.py), plus a data-freshness check
                     (ledger.check_data_freshness).
Stage 1 (LLM):      Contradiction detection — only for tasks Stage 0 found
                     in 2+ sources; skipped entirely if there are none.
Stage 2 (LLM):      Unified synthesis — "what matters today" +
                     "what might be missed" + a list of dispatchable items,
                     as structured JSON (not free prose, since Stage 3
                     needs a clean list to draft against).
Stage 3 (LLM):      Draft generation for dispatchable items, tone-locked to
                     the persona, with Sam Park items filtered out in code
                     first (persona's own rule: never draft for Sam).

Assembly is pure Python — no LLM involved in stitching the final brief
together.

Usage:
    python3 orchestrator.py
    python3 orchestrator.py --provider anthropic --model claude-haiku-4-5
"""

import argparse
import json
import re
import os
import time

import calendar_agent
import notes_agent
import triage_agent
from cross_reference import build_cross_reference_index
from ledger import check_data_freshness, format_today, load_ledger, save_digest, validate_schema
from llm import call_with_retry, create_llm
from persona import load_persona
from tenant_config import load_tenant_config
from tasks_parser import load_tasks
from tasks_signals import compute_task_signals, format_task_signals

# ─── Path Configuration ──────────────────────────────────────────────────────

TASKS_FILE = "./data/tasks/tasks.json"
OUTPUT_DIR = "./output/"
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
BRIEF_FILE = os.path.join(OUTPUT_DIR, "daily_brief.md")

CONTRADICTION_SCHEMA = {"contradictions": ["entity", "sources", "description"]}
SYNTHESIS_LIST_SCHEMA = {"dispatchable_items": ["id", "type", "summary"]}
DRAFT_SCHEMA = {"drafts": ["id", "draft_text"]}


# ─── Prompt Builders (persona-injected) ───────────────────────────────────────


def build_contradiction_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are checking for contradictions between sources, for the operator "
        "profiled above. You will receive a list of tasks that show up in more "
        "than one data source (email, calendar, notes), with excerpts of what "
        "each source actually says about them. Per the profile's honesty rules: "
        "if two sources disagree, that must be surfaced, not silently resolved "
        "by picking one.\n\n"
        "For each task, check whether the excerpts actually conflict (different "
        "dates, different owners, different decisions) — not just that they "
        "both mention the same thing, which is normal and not a contradiction "
        "by itself.\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "contradictions": [{"entity": "...", "sources": ["email", "notes"], "description": "..."}]\n'
        "}\n\n"
        "If nothing actually conflicts, return an empty array. Do not write any "
        "markdown wrappers, conversational pleasantries, or extra text."
    )


def build_synthesis_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are producing the unified daily brief for the operator profiled "
        "above, synthesizing across email, calendar, notes, and tasks — this is "
        "the actual top-level deliverable, not a per-source summary. You are "
        "given: today's actual date, the three source ledgers, live task signals "
        "(overdue/due-soon/blocked/stalled), a deterministic cross-reference index "
        "showing which flagged tasks appear in multiple sources, any detected "
        "contradictions, and a data-freshness report.\n\n"
        "Use the provided date as ground truth for anything date-relative — do not "
        "infer today's date from the ledger content itself (e.g. the earliest or "
        "most recent entry present).\n\n"
        "Produce three things:\n\n"
        "1. **what_matters_today**: what genuinely requires the operator's "
        "judgment, signature, or reply today — weighted by the profile's "
        "people-priority list. Not what's interesting, what's urgent.\n"
        "2. **what_might_be_missed**: forgotten threads, declined/cancelled "
        "meetings, stale follow-ups, and — critically — anything the "
        "cross-reference index or contradiction report surfaced. A task "
        "flagged as stalled AND mentioned in an old email is a stronger signal "
        "than either alone; say so.\n"
        "3. **dispatchable_items**: things genuinely answerable in under a "
        "minute per the profile's own bar — short replies, yes/no approvals, "
        "quick task follow-ups. Do not include anything that needs real "
        "judgment (e.g. a hiring offer decision is not dispatchable, a "
        "one-line acknowledgment is).\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "what_matters_today": "markdown text",\n'
        '  "what_might_be_missed": "markdown text",\n'
        '  "dispatchable_items": [{"id": "...", "type": "email_reply|calendar_rsvp|task_followup", "summary": "...", "context": "..."}]\n'
        "}\n\n"
        "Important: what_matters_today and what_might_be_missed must be the "
        "actual digest content, addressed directly to the operator in second "
        "person. Do NOT describe, summarize, or explain the JSON data format "
        "you were given — that is input data to read, not something to report "
        "on. Follow the profile's honesty rules: say if data looks stale "
        "(a freshness report is provided — use it, don't guess), flag "
        "assumptions, and don't hide contradictions."
    )


def build_draft_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are drafting quick replies for the operator profiled above, for "
        "items already identified as dispatchable in under a minute. Follow the "
        "profile's tone rules exactly, including greeting and sign-off style, as "
        "specified in the profile above — don't invent your own phrasing for "
        "these. Warmth comes from specificity, not adjectives.\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "drafts": [{"id": "...", "draft_text": "..."}]\n'
        "}\n\n"
        "Do not write any markdown wrappers, conversational pleasantries, or "
        "extra text."
    )


# ─── Stage 1: Contradiction Detection ─────────────────────────────────────────


def detect_contradictions(llm, cross_ref_index: dict, persona_text: str) -> dict:
    """Only checks tasks that appear in 2+ distinct sources — a task
    mentioned twice within the same source has nothing to contradict
    against. Skips the LLM call entirely if there are no such candidates.
    """
    multi_source = {
        task_id: data
        for task_id, data in cross_ref_index.items()
        if len({m["source"] for m in data["mentioned_in"]}) >= 2
    }

    if not multi_source:
        print("   ℹ️  No multi-source task mentions found — skipping contradiction check.")
        return {"contradictions": []}

    prompt = build_contradiction_prompt(persona_text)
    context = json.dumps(multi_source, indent=2)

    def _call():
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Cross-source task mentions:\n\n{context}"},
            ]
        )
        errors = validate_schema(result, CONTRADICTION_SCHEMA)
        if errors:
            raise ValueError(f"Invalid contradiction-check output: {errors}")
        return result

    try:
        return call_with_retry(_call)
    except Exception as e:
        print(f"   ⚠️  Contradiction detection failed after retries: {e}")
        return {"contradictions": []}


# ─── Stage 2: Unified Synthesis ───────────────────────────────────────────────


def synthesize_brief(
    llm,
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
    persona_text: str,
) -> dict:
    prompt = build_synthesis_prompt(persona_text)

    context = (
        f"TODAY'S DATE: {format_today()}\n\n"
        f"---\n\n"
        f"EMAIL LEDGER:\n{triage_agent.render_ledger_as_text(email_ledger)}\n\n"
        f"CALENDAR LEDGER:\n{calendar_agent.render_ledger_as_text(calendar_ledger)}\n\n"
        f"NOTES LEDGER:\n{notes_agent.render_ledger_as_text(notes_ledger)}\n\n"
        f"TASK SIGNALS:\n{format_task_signals(task_signals)}\n\n"
        f"CROSS-REFERENCE INDEX (tasks found in multiple sources):\n"
        f"{json.dumps(cross_ref_index, indent=2)}\n\n"
        f"CONTRADICTIONS DETECTED:\n{json.dumps(contradictions, indent=2)}\n\n"
        f"DATA FRESHNESS:\n{json.dumps(freshness, indent=2)}"
    )

    def _call():
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ]
        )
        if "what_matters_today" not in result or "what_might_be_missed" not in result:
            raise ValueError("Missing what_matters_today or what_might_be_missed")
        errors = validate_schema(result, SYNTHESIS_LIST_SCHEMA)
        if errors:
            raise ValueError(f"Invalid synthesis output: {errors}")
        return result

    return call_with_retry(_call)


# ─── Stage 3: Draft Generation ────────────────────────────────────────────────


def build_never_draft_patterns(never_draft_contacts: list[dict]) -> list[re.Pattern]:
    """Compile match patterns for tenant-configured "never draft" contacts.

    Matches on individual name parts (first/last), not just the full name as
    one substring — LLM-generated summary/context prose is far more likely
    to say just "Sam" than "Sam Park" or the literal email address. Word-
    boundary regex avoids a short name part false-positiving inside an
    unrelated word (e.g. "sam" inside "samsung").
    """
    patterns = []
    for contact in never_draft_contacts:
        for part in contact.get("name", "").split():
            if len(part) >= 3:
                patterns.append(re.compile(r"\b" + re.escape(part.lower()) + r"\b"))
        email = contact.get("email", "")
        if email:
            patterns.append(re.compile(re.escape(email.lower())))
    return patterns


def mentions_blocked_contact(item: dict, patterns: list[re.Pattern]) -> bool:
    """Check whether a dispatchable item's summary/context matches any
    never-draft-contact pattern (see build_never_draft_patterns)."""
    text = (item.get("summary", "") + " " + item.get("context", "")).lower()
    return any(pattern.search(text) for pattern in patterns)


def generate_drafts(llm, dispatchable_items: list[dict], persona_text: str, config: dict) -> dict:
    """Filters out anything sourced from a tenant-configured "never draft"
    contact in code before the call is even made — enforced deterministically
    rather than trusted to the prompt alone. Generalized from a Sam-Park-
    specific hardcoded check to config["never_draft_contacts"] (see
    tenant_config.py) so this works for any tenant's own such rule, not just
    this persona's.
    """
    patterns = build_never_draft_patterns(config.get("never_draft_contacts", []))
    draftable = [item for item in dispatchable_items if not mentions_blocked_contact(item, patterns)]

    if not draftable:
        print("   ℹ️  No draftable items (after filtering never-draft contacts) — skipping draft generation.")
        return {"drafts": []}

    prompt = build_draft_prompt(persona_text)
    context = json.dumps(draftable, indent=2)

    def _call():
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Dispatchable items:\n\n{context}"},
            ]
        )
        errors = validate_schema(result, DRAFT_SCHEMA)
        if errors:
            raise ValueError(f"Invalid draft output: {errors}")
        return result

    try:
        return call_with_retry(_call)
    except Exception as e:
        print(f"   ⚠️  Draft generation failed after retries: {e}")
        return {"drafts": []}


# ─── Assembly (pure Python, no LLM) ───────────────────────────────────────────


def assemble_brief(synthesis: dict, drafts: dict, freshness: dict) -> str:
    lines = ["# Daily Brief\n"]

    stale_sources = [name for name, info in freshness.items() if info["is_stale"]]
    if stale_sources:
        lines.append("> ⚠️ **Data freshness notice**")
        for name in stale_sources:
            info = freshness[name]
            if info["most_recent_day"] is None:
                lines.append(f"> - {name}: no data available at all.")
            else:
                lines.append(
                    f"> - {name}: most recent data is from {info['most_recent_day']} "
                    f"({info['days_stale']} days ago)."
                )
        lines.append("")

    lines.append("## What Matters Today\n")
    lines.append(synthesis.get("what_matters_today", "").strip())
    lines.append("")

    lines.append("## What You Might Be Missing\n")
    lines.append(synthesis.get("what_might_be_missed", "").strip())
    lines.append("")

    draft_by_id = {d["id"]: d["draft_text"] for d in drafts.get("drafts", [])}
    dispatchable = synthesis.get("dispatchable_items", [])
    if dispatchable:
        lines.append("## Quick Dispatches\n")
        for item in dispatchable:
            lines.append(f"**{item.get('summary', item.get('id'))}**")
            draft_text = draft_by_id.get(item["id"])
            if draft_text:
                lines.append(f"> {draft_text}")
            else:
                lines.append("*(surfaced for you to handle directly — not drafted)*")
            lines.append("")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrator — Unified Chief-of-Staff Daily Brief",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 orchestrator.py\n"
            "  python3 orchestrator.py --provider anthropic --model claude-haiku-4-5\n"
        ),
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"], help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="llama3", help="Model name (default: llama3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🤖 Initializing LLM: {args.provider}/{args.model}\n")
    persona_text = load_persona()
    config = load_tenant_config()
    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature)

    total_start = time.time()

    print("📚 Loading source ledgers and live task signals...")
    email_ledger, _ = load_ledger(triage_agent.LEDGER_FILE)
    calendar_ledger, _ = load_ledger(calendar_agent.LEDGER_FILE)
    notes_ledger, _ = load_ledger(notes_agent.LEDGER_FILE)
    tasks = load_tasks(TASKS_FILE)
    task_signals = compute_task_signals(tasks)

    if not (email_ledger or calendar_ledger or notes_ledger):
        print("❌ No source ledgers found. Run triage_agent.py / calendar_agent.py / notes_agent.py first.")
        return

    print("🔎 Stage 0 — deterministic cross-reference index + freshness check...")
    cross_ref_index = build_cross_reference_index(email_ledger, calendar_ledger, notes_ledger, task_signals)
    freshness = check_data_freshness({
        "email": email_ledger, "calendar": calendar_ledger, "notes": notes_ledger,
    })
    print(f"   {len(cross_ref_index)} flagged task(s) found in multiple sources.")

    print("🔍 Stage 1 — contradiction detection...")
    contradictions = detect_contradictions(llm, cross_ref_index, persona_text)
    print(f"   {len(contradictions.get('contradictions', []))} contradiction(s) found.")

    print("🧠 Stage 2 — unified synthesis...")
    synthesis = synthesize_brief(
        llm, email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness, persona_text,
    )
    print(f"   {len(synthesis.get('dispatchable_items', []))} dispatchable item(s) identified.")

    print("✍️  Stage 3 — draft generation...")
    drafts = generate_drafts(llm, synthesis.get("dispatchable_items", []), persona_text, config)
    print(f"   {len(drafts.get('drafts', []))} draft(s) written.")

    brief = assemble_brief(synthesis, drafts, freshness)
    history_path = save_digest(brief, BRIEF_FILE, HISTORY_DIR)

    total_time = time.time() - total_start
    print(f"\n✅ Daily brief completed in {total_time:.1f}s.")
    print(f"   Saved to: {BRIEF_FILE}")
    print(f"   History copy: {history_path}")
    print(f"\n{'─' * 60}\n   Daily Brief\n{'─' * 60}\n")
    print(brief)


if __name__ == "__main__":
    main()
