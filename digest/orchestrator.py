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

from digest.agents import calendar_agent
from digest.agents import notes_agent
from digest.core import resilience
from digest.core import tenant_paths
from digest.agents import triage_agent
from digest.core.cross_reference import build_cross_reference_index, title_keywords
from digest.core.ledger import check_data_freshness, format_today, load_ledger, save_digest, validate_schema
from digest.core.llm import call_with_retry, create_llm
from digest.core.persona import load_persona
from digest.core.tenant_config import load_tenant_config
from digest.parsers.tasks_parser import load_tasks
from digest.core.tasks_signals import compute_task_signals, format_task_signals

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
        "Only report a contradiction you can point to directly in the excerpts "
        "given for that specific task — never infer one from thematic "
        "similarity to something else you recall, and never combine an excerpt "
        "from one task with your general knowledge of a similar situation. If "
        "the excerpts for a task don't actually contain enough to say two "
        "sources conflict, leave it out rather than guessing.\n\n"
        "You will receive a JSON object whose keys are task IDs, each with a "
        "'title' field. Only report contradictions for entities that appear as "
        "a key or title in that object — you may refer to an entity by its "
        "task ID, its title, or both together, but never reference any other "
        "entity, person, or project, even one you recognize from general "
        "knowledge.\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "contradictions": [{"entity": "...", "sources": ["email", "notes"], "description": "..."}]\n'
        "}\n\n"
        "If nothing actually conflicts, return an empty array. Do not write any "
        "markdown wrappers, conversational pleasantries, or extra text."
    )


def build_synthesis_prompt(persona_text: str) -> str:
    """Header-structured, matching the pattern build_reduce_system_prompt
    already uses successfully (## 1. WHAT NEEDS ME TODAY etc. in
    triage_agent.py) — extended here to the one prompt that didn't have it
    yet, not a new technique introduced on priors.
    """
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "## YOUR ROLE\n"
        "You are producing the unified daily brief for the operator profiled "
        "above, synthesizing across email, calendar, notes, and tasks — this is "
        "the actual top-level deliverable, not a per-source summary.\n\n"
        "## INPUT SECTIONS YOU WILL RECEIVE\n"
        "- TODAY'S DATE — ground truth for anything date-relative; do not infer "
        "it from the ledger content itself (e.g. the earliest or most recent "
        "entry present).\n"
        "- EMAIL / CALENDAR / NOTES LEDGERS — the three source digests.\n"
        "- TASK SIGNALS — live overdue/due-soon/blocked/stalled tasks.\n"
        "- CROSS-REFERENCE INDEX — which flagged tasks appear in multiple "
        "sources.\n"
        "- CONTRADICTIONS — conflicts already detected between sources.\n"
        "- DATA FRESHNESS — how current each source's data actually is.\n\n"
        "## OUTPUT 1: what_matters_today\n"
        "What genuinely requires the operator's judgment, signature, or reply "
        "today — weighted by the profile's people-priority list. Not what's "
        "interesting, what's urgent.\n\n"
        "## OUTPUT 2: what_might_be_missed\n"
        "Forgotten threads, declined/cancelled meetings, stale follow-ups, and "
        "— critically — anything the cross-reference index or contradiction "
        "report surfaced. A task flagged as stalled AND mentioned in an old "
        "email is a stronger signal than either alone; say so.\n\n"
        "## OUTPUT 3: dispatchable_items\n"
        "Things genuinely answerable in under a minute per the profile's own "
        "bar — short replies, yes/no approvals, quick task follow-ups. Do not "
        "include anything that needs real judgment (e.g. a hiring offer "
        "decision is not dispatchable, a one-line acknowledgment is).\n\n"
        "## OUTPUT FORMAT\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "what_matters_today": "markdown text",\n'
        '  "what_might_be_missed": "markdown text",\n'
        '  "dispatchable_items": [{"id": "...", "type": "email_reply|calendar_rsvp|task_followup", "summary": "...", "context": "..."}]\n'
        "}\n\n"
        "## RULES\n"
        "- what_matters_today and what_might_be_missed must be the actual "
        "digest content, addressed directly to the operator in second person.\n"
        "- Do NOT describe, summarize, or explain the JSON data format you "
        "were given — that is input data to read, not something to report on.\n"
        "- Follow the profile's honesty rules: say if data looks stale (a "
        "freshness report is provided — use it, don't guess), flag "
        "assumptions, and don't hide contradictions."
    )


def build_narrative_prompt(persona_text: str) -> str:
    """Staged Stage-2 variant (see SYNTHESIS_VARIANTS): the same prompt as
    build_synthesis_prompt, minus the dispatchable_items output — kept
    together in one prompt because what_matters_today and
    what_might_be_missed need to be written with awareness of each other to
    avoid the same item being duplicated into both, or dropped because each
    section assumed the other covered it.
    """
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "## YOUR ROLE\n"
        "You are producing the unified daily brief for the operator profiled "
        "above, synthesizing across email, calendar, notes, and tasks — this is "
        "the actual top-level deliverable, not a per-source summary.\n\n"
        "## INPUT SECTIONS YOU WILL RECEIVE\n"
        "- TODAY'S DATE — ground truth for anything date-relative; do not infer "
        "it from the ledger content itself (e.g. the earliest or most recent "
        "entry present).\n"
        "- EMAIL / CALENDAR / NOTES LEDGERS — the three source digests.\n"
        "- TASK SIGNALS — live overdue/due-soon/blocked/stalled tasks.\n"
        "- CROSS-REFERENCE INDEX — which flagged tasks appear in multiple "
        "sources.\n"
        "- CONTRADICTIONS — conflicts already detected between sources.\n"
        "- DATA FRESHNESS — how current each source's data actually is.\n\n"
        "## OUTPUT 1: what_matters_today\n"
        "What genuinely requires the operator's judgment, signature, or reply "
        "today — weighted by the profile's people-priority list. Not what's "
        "interesting, what's urgent.\n\n"
        "## OUTPUT 2: what_might_be_missed\n"
        "Forgotten threads, declined/cancelled meetings, stale follow-ups, and "
        "— critically — anything the cross-reference index or contradiction "
        "report surfaced. A task flagged as stalled AND mentioned in an old "
        "email is a stronger signal than either alone; say so.\n\n"
        "## OUTPUT FORMAT\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "what_matters_today": "markdown text",\n'
        '  "what_might_be_missed": "markdown text"\n'
        "}\n\n"
        "## RULES\n"
        "- Both outputs must be the actual digest content, addressed directly "
        "to the operator in second person.\n"
        "- Do NOT describe, summarize, or explain the JSON data format you "
        "were given — that is input data to read, not something to report on.\n"
        "- Follow the profile's honesty rules: say if data looks stale (a "
        "freshness report is provided — use it, don't guess), flag "
        "assumptions, and don't hide contradictions."
    )


def build_dispatchable_items_prompt(persona_text: str) -> str:
    """Staged Stage-2 variant's second call — pulled out of the combined
    prompt because "is this genuinely answerable in under a minute" is a
    narrower classification task than the narrative outputs, and doesn't
    need to be written in the same breath as them. Receives the narrative
    this same brief already produced (see synthesize_dispatchable_items) so
    it can avoid re-describing something at length rather than re-deriving
    priority from scratch.
    """
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "## YOUR ROLE\n"
        "You are extracting dispatchable items for the daily brief being "
        "assembled for the operator profiled above — the same brief whose "
        "narrative sections (what matters today, what might be missed) you "
        "will also be shown below, for context only.\n\n"
        "## INPUT SECTIONS YOU WILL RECEIVE\n"
        "- TODAY'S DATE, EMAIL / CALENDAR / NOTES LEDGERS, TASK SIGNALS, "
        "CROSS-REFERENCE INDEX, CONTRADICTIONS, DATA FRESHNESS — the same "
        "source material used to write the narrative.\n"
        "- NARRATIVE ALREADY WRITTEN — what_matters_today and "
        "what_might_be_missed, already finalized for this brief. Use it to "
        "avoid wholesale duplication: referencing the same underlying item "
        "here is fine if it's independently a quick reply, but don't "
        "re-describe it at length.\n\n"
        "## WHAT COUNTS AS DISPATCHABLE\n"
        "Things genuinely answerable in under a minute per the profile's own "
        "bar — short replies, yes/no approvals, quick task follow-ups. Do not "
        "include anything that needs real judgment (e.g. a hiring offer "
        "decision is not dispatchable, a one-line acknowledgment is).\n\n"
        "## OUTPUT FORMAT\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "dispatchable_items": [{"id": "...", "type": "email_reply|calendar_rsvp|task_followup", "summary": "...", "context": "..."}]\n'
        "}\n\n"
        "## RULES\n"
        "- If nothing genuinely qualifies, return an empty array — do not "
        "stretch to fill it.\n"
        "- Do NOT describe, summarize, or explain the JSON data format you "
        "were given — that is input data to read, not something to report on."
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
        "Use the `context` field on each item to infer who it's for and "
        "calibrate tone per the profile's own rules — e.g. investors/board "
        "members during a raise get slightly more polish (still short), team "
        "members get direct and warm-through-specificity, everyone else gets "
        "short and neutral.\n\n"
        "Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "drafts": [{"id": "...", "draft_text": "..."}]\n'
        "}\n\n"
        "Do not write any markdown wrappers, conversational pleasantries, or "
        "extra text."
    )


# ─── Stage 1: Contradiction Detection ─────────────────────────────────────────


def _resolve_entity(entity: str, multi_source: dict) -> dict | None:
    """Match a claimed contradiction's 'entity' field against multi_source.

    The prompt doesn't dictate whether the model should echo the task_id
    (the dict key it was shown) or the task's title field back — both are
    visible to it in the JSON context. Accepts either, case-insensitively,
    rather than silently failing to resolve legitimate claims just because
    the model picked the other valid label.

    Also accepts the model combining both into one string — e.g.
    "TESS-225: Confirm Series A closing call time with Marcus" — since that's
    a normal, legitimate way to reference an item, not a malformed one.
    Found live, not hypothetically: the first real true-positive contradiction
    this grounding check ever saw (a genuine planted date conflict) was
    dropped as "ungrounded" purely because of this formatting mismatch,
    before this fix — a false negative in the safety check itself.
    """
    if entity in multi_source:
        return multi_source[entity]
    entity_lower = entity.strip().lower()
    for task_id, data in multi_source.items():
        title_lower = data.get("title", "").strip().lower()
        if title_lower == entity_lower:
            return data
        if entity_lower.startswith(task_id.lower()):
            return data
        if title_lower and title_lower in entity_lower:
            return data
    return None


def _ground_contradictions(contradictions: list, multi_source: dict) -> list:
    """Drop contradiction claims whose entity/sources were never actually
    presented together — the deterministic half of closing gap #4 in
    docs/ERROR_HANDLING.md (schema validation checks shape, not truth).

    This does not fact-check a claim's *content* — that would mean
    re-deriving the same judgment call the LLM was asked to make. It only
    checks the structural precondition for the claim to be possible at all:
    the entity must be one of the candidates actually given to the model,
    and the claimed sources must be a subset of where that entity was
    actually seen (per cross_reference.py's mentioned_in). A claim that
    fails this check referenced a source/entity pairing that was never
    shown together, which is either a hallucination or, defensively, worth
    treating the same as one either way.
    """
    grounded = []
    for c in contradictions:
        entity = c.get("entity", "")
        claimed_sources = set(c.get("sources", []))
        match = _resolve_entity(entity, multi_source)
        if match is None:
            print(f"   ⚠️  Dropping ungrounded contradiction — unknown entity {entity!r}")
            continue
        actual_sources = {m["source"] for m in match["mentioned_in"]}
        if not claimed_sources or not claimed_sources.issubset(actual_sources):
            print(
                f"   ⚠️  Dropping ungrounded contradiction for {entity!r} — "
                f"claimed sources {sorted(claimed_sources)} not all present "
                f"in actual mentions {sorted(actual_sources)}"
            )
            continue
        grounded.append(c)
    return grounded


def detect_contradictions(llm, cross_ref_index: dict, persona_text: str, breaker=None, metrics_path=None) -> dict:
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
        result = call_with_retry(_call, breaker=breaker, llm=llm, metrics_path=metrics_path)
    except Exception as e:
        print(f"   ⚠️  Contradiction detection failed after retries: {e}")
        return {"contradictions": []}

    grounded = _ground_contradictions(result.get("contradictions", []), multi_source)
    dropped = len(result.get("contradictions", [])) - len(grounded)
    if dropped:
        print(f"   ℹ️  Grounding check dropped {dropped} ungrounded contradiction(s).")
    return {"contradictions": grounded}


# ─── Stage 2: Unified Synthesis ───────────────────────────────────────────────


def render_cross_ref_index_as_text(cross_ref_index: dict) -> str:
    """Flatten the cross-reference index into prose instead of raw JSON —
    same reasoning as render_ledger_as_text in each agent: dumping raw JSON
    into a prompt has reliably produced schema-narration output elsewhere in
    this system (see HLD's "what was avoided"), and this section was the one
    place synthesize_brief's context still did it.
    """
    if not cross_ref_index:
        return "(none — no flagged tasks found mentioned in multiple sources)"
    lines = []
    for task_id, data in cross_ref_index.items():
        lines.append(f"- {task_id}: {data['title']} (priority {data.get('priority', '?')})")
        for mention in data.get("mentioned_in", []):
            excerpt = mention.get("excerpt", "")[:200]
            lines.append(f"    [{mention.get('source')}, {mention.get('day')}] {excerpt}")
    return "\n".join(lines)


def render_contradictions_as_text(contradictions: dict) -> str:
    """Flatten Stage 1's contradiction report into prose — same reasoning
    as render_cross_ref_index_as_text above.
    """
    items = contradictions.get("contradictions", [])
    if not items:
        return "(none detected)"
    lines = []
    for c in items:
        sources = ", ".join(c.get("sources", []))
        lines.append(f"- {c.get('entity')} ({sources}): {c.get('description')}")
    return "\n".join(lines)


def _build_synthesis_context(
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
) -> str:
    """Shared context block for every Stage-2 variant (single-call and
    staged) — extracted so both build it identically instead of each
    re-assembling the same rendered ledgers/task signals/cross-ref
    index/contradictions/freshness text independently.
    """
    return (
        f"TODAY'S DATE: {format_today()}\n\n"
        f"---\n\n"
        f"EMAIL LEDGER:\n{triage_agent.render_ledger_as_text(email_ledger)}\n\n"
        f"CALENDAR LEDGER:\n{calendar_agent.render_ledger_as_text(calendar_ledger)}\n\n"
        f"NOTES LEDGER:\n{notes_agent.render_ledger_as_text(notes_ledger)}\n\n"
        f"TASK SIGNALS:\n{format_task_signals(task_signals)}\n\n"
        f"CROSS-REFERENCE INDEX (tasks found in multiple sources):\n"
        f"{render_cross_ref_index_as_text(cross_ref_index)}\n\n"
        f"CONTRADICTIONS DETECTED:\n{render_contradictions_as_text(contradictions)}\n\n"
        f"DATA FRESHNESS:\n{json.dumps(freshness, indent=2)}"
    )


def synthesize_brief_single_call(
    llm,
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
    persona_text: str,
    breaker=None,
    metrics_path=None,
) -> dict:
    """Stage-2 variant: one call producing all three outputs. See
    SYNTHESIS_VARIANTS and synthesize_brief_staged for the alternative being
    compared against this via eval_synthesis_variants.py.
    """
    prompt = build_synthesis_prompt(persona_text)
    context = _build_synthesis_context(
        email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness,
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

    return call_with_retry(_call, breaker=breaker, llm=llm, metrics_path=metrics_path)


def synthesize_narrative(
    llm,
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
    persona_text: str,
    breaker=None,
    metrics_path=None,
) -> dict:
    """Staged Stage-2 variant, call 1 of 2: what_matters_today +
    what_might_be_missed only. No try/except — a failure here propagates
    and aborts the run, same as synthesize_brief_single_call does today:
    there's no meaningful digest without the narrative, so degrading to
    empty would be worse than failing loudly.
    """
    prompt = build_narrative_prompt(persona_text)
    context = _build_synthesis_context(
        email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness,
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
        return result

    return call_with_retry(_call, breaker=breaker, llm=llm, metrics_path=metrics_path)


def synthesize_dispatchable_items(
    llm,
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
    narrative: dict,
    persona_text: str,
    breaker=None,
    metrics_path=None,
) -> dict:
    """Staged Stage-2 variant, call 2 of 2: dispatchable_items only, given
    the narrative synthesize_narrative already produced as extra context.
    Wrapped in try/except, degrading to an empty list on failure — this one
    is additive/optional, same graceful-degradation treatment contradiction
    detection and draft generation already get.
    """
    prompt = build_dispatchable_items_prompt(persona_text)
    context = _build_synthesis_context(
        email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness,
    )
    context += (
        "\n\n---\n\n"
        "NARRATIVE ALREADY WRITTEN (for context — do not duplicate wholesale):\n"
        f"what_matters_today:\n{narrative.get('what_matters_today', '')}\n\n"
        f"what_might_be_missed:\n{narrative.get('what_might_be_missed', '')}"
    )

    def _call():
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ]
        )
        errors = validate_schema(result, SYNTHESIS_LIST_SCHEMA)
        if errors:
            raise ValueError(f"Invalid dispatchable-items output: {errors}")
        return result

    try:
        return call_with_retry(_call, breaker=breaker, llm=llm, metrics_path=metrics_path)
    except Exception as e:
        print(f"   ⚠️  Dispatchable-item extraction failed after retries: {e}")
        return {"dispatchable_items": []}


def synthesize_brief_staged(
    llm,
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    cross_ref_index: dict,
    contradictions: dict,
    freshness: dict,
    persona_text: str,
    breaker=None,
    metrics_path=None,
) -> dict:
    """Stage-2 variant: narrative and dispatchable_items as two dependent
    calls instead of one — see SYNTHESIS_VARIANTS. Same call signature and
    return shape as synthesize_brief_single_call, so either is a drop-in
    replacement for the other to any caller.
    """
    print("   🧠 Stage 2a — narrative synthesis (what matters, what's missed)...")
    narrative = synthesize_narrative(
        llm, email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness, persona_text,
        breaker=breaker, metrics_path=metrics_path,
    )

    print("   📋 Stage 2b — dispatchable-item extraction...")
    dispatchable = synthesize_dispatchable_items(
        llm, email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness, narrative, persona_text,
        breaker=breaker, metrics_path=metrics_path,
    )

    return {**narrative, **dispatchable}


SYNTHESIS_VARIANTS = {
    "single_call": synthesize_brief_single_call,
    "staged": synthesize_brief_staged,
}


def check_priority_coverage(synthesis: dict, task_signals: dict, priorities=("P0", "P1")) -> list[dict]:
    """Deterministic post-check, applies after either Stage-2 variant:
    verify every task_signals task at P0/P1 is referenced somewhere in the
    synthesized brief text, using the same title_keywords/2-keyword-match
    threshold cross_reference.py already uses to decide "is this genuinely
    mentioned" — reused here, not reinvented, so "mentioned" means the same
    thing in both places.

    This is a coverage check, not a truth check: it can't verify the brief
    is *correct*, only that a known-important item wasn't silently dropped
    from the output entirely. Deliberately blunt (keyword-substring, not
    semantic) — same accepted tradeoff as digest_checks.py and
    cross_reference.py. The caller logs this as a warning, not a hard
    failure: a false positive here (an item covered via paraphrase, with no
    literal keyword overlap) shouldn't block shipping a brief that may in
    fact be fine.

    Also doubles as eval_synthesis_variants.py's scoring metric for
    comparing the single_call and staged Stage-2 implementations — the same
    deterministic signal serves both a production safety net and an
    objective, LLM-free eval score.

    Returns:
        [{"task_id", "title", "priority"}] for every flagged task that
        didn't clear the match threshold — deduplicated by task_id, since a
        task can legitimately be flagged in more than one task_signals
        bucket at once (e.g. both "overdue" and "stalled"); each such task
        must only ever count once here, not once per bucket it happens to
        appear in. Empty list means full coverage.
    """
    combined_text = " ".join([
        synthesis.get("what_matters_today", ""),
        synthesis.get("what_might_be_missed", ""),
        " ".join(
            f"{item.get('summary', '')} {item.get('context', '')}"
            for item in synthesis.get("dispatchable_items", [])
        ),
    ]).lower()

    candidates = {}
    for bucket in ("overdue", "due_soon", "blocked", "stalled"):
        for task in task_signals.get(bucket, []):
            if task.get("priority") not in priorities:
                continue
            candidates[task["id"]] = task

    missing = []
    for task_id, task in candidates.items():
        keywords = title_keywords(task["title"])
        if not keywords:
            continue
        matched = sum(1 for kw in keywords if kw.lower() in combined_text)
        if matched < min(2, len(keywords)):
            missing.append({"task_id": task_id, "title": task["title"], "priority": task["priority"]})
    return missing


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


def generate_drafts(llm, dispatchable_items: list[dict], persona_text: str, config: dict, breaker=None, metrics_path=None) -> dict:
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
        return call_with_retry(_call, breaker=breaker, llm=llm, metrics_path=metrics_path)
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
    parser.add_argument(
        "--tenant",
        default=tenant_paths.DEFAULT_TENANT,
        help="Tenant ID — reads/writes under data/tenants/<id>/ and output/tenants/<id>/ "
        "(default: 'default', today's existing ./data/./output/ layout). Must match the "
        "--tenant used when running triage_agent.py/calendar_agent.py/notes_agent.py, "
        "since this only reads ledgers those scripts already produced.",
    )
    parser.add_argument(
        "--synthesis-variant",
        default="single_call",
        choices=list(SYNTHESIS_VARIANTS.keys()),
        help="Stage 2 implementation to use: 'single_call' (one LLM call for all "
        "three outputs) or 'staged' (narrative, then dispatchable_items as a "
        "dependent second call). Default 'single_call' preserves current "
        "production behavior — see eval_synthesis_variants.py for the comparison "
        "this default is pending on.",
    )
    return parser.parse_args()


def run_for_tenant(tenant_id: str, provider: str, model: str, temperature: float = 0.0, synthesis_variant: str = "single_call") -> None:
    """Run the full orchestrator pipeline for one tenant — everything
    main() used to do inline, extracted so run_fleet.py can call this
    directly, once per tenant, inside a shared ThreadPoolExecutor. Keeping
    this an in-process function call (not a subprocess) is what lets
    resilience.py's breaker/token-bucket registries actually be shared
    across concurrent tenants — the entire point of the fleet runner.
    """
    paths = tenant_paths.for_tenant(tenant_id)

    print(f"🤖 Initializing LLM: {provider}/{model}")
    print(f"   Tenant: {paths.tenant_id}\n")
    persona_text = load_persona(paths.persona_file)
    config = load_tenant_config(paths.tenant_config_file)
    llm = create_llm(provider=provider, model=model, temperature=temperature, tenant_id=paths.tenant_id)

    # Shared across every tenant process calling this (provider, model) —
    # see resilience.py. Meaningful once run_fleet.py drives several
    # tenants concurrently in one process; a no-op-in-practice single
    # instance otherwise (never opens on a lone successful run).
    breaker = resilience.get_breaker(
        provider, model,
        failure_threshold=config.get("circuit_breaker_threshold", 5),
    )
    metrics_path = paths.metrics_file

    total_start = time.time()

    print("📚 Loading source ledgers and live task signals...")
    email_ledger, _ = load_ledger(paths.email_ledger_file)
    calendar_ledger, _ = load_ledger(paths.calendar_ledger_file)
    notes_ledger, _ = load_ledger(paths.notes_ledger_file)
    tasks = load_tasks(paths.tasks_file)
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
    contradictions = detect_contradictions(llm, cross_ref_index, persona_text, breaker=breaker, metrics_path=metrics_path)
    print(f"   {len(contradictions.get('contradictions', []))} contradiction(s) found.")

    print(f"🧠 Stage 2 — unified synthesis (variant: {synthesis_variant})...")
    synthesis = SYNTHESIS_VARIANTS[synthesis_variant](
        llm, email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness, persona_text,
        breaker=breaker, metrics_path=metrics_path,
    )
    print(f"   {len(synthesis.get('dispatchable_items', []))} dispatchable item(s) identified.")

    missing_priority = check_priority_coverage(synthesis, task_signals)
    if missing_priority:
        print(f"   ⚠️  {len(missing_priority)} high-priority item(s) may be missing from the brief:")
        for m in missing_priority:
            print(f"      - {m['task_id']}: {m['title']} ({m['priority']})")

    print("✍️  Stage 3 — draft generation...")
    drafts = generate_drafts(llm, synthesis.get("dispatchable_items", []), persona_text, config, breaker=breaker, metrics_path=metrics_path)
    print(f"   {len(drafts.get('drafts', []))} draft(s) written.")

    brief = assemble_brief(synthesis, drafts, freshness)
    history_path = save_digest(brief, paths.brief_file, paths.history_dir)

    total_time = time.time() - total_start
    print(f"\n✅ Daily brief completed in {total_time:.1f}s.")
    print(f"   Saved to: {paths.brief_file}")
    print(f"   History copy: {history_path}")
    print(f"\n{'─' * 60}\n   Daily Brief\n{'─' * 60}\n")
    print(brief)


def main():
    args = parse_args()
    run_for_tenant(args.tenant, args.provider, args.model, args.temperature, args.synthesis_variant)


if __name__ == "__main__":
    main()
