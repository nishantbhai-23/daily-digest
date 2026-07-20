"""
Structured Digest — Machine-Readable Alternative to the Markdown Brief
=========================================================================
A separate, optional pipeline stage that produces a structured, per-item
JSON representation of the daily digest instead of markdown prose — every
item explicitly typed (action/information/conflict), carrying the model's
own priority judgment plus a deterministic date-urgency cross-check, in a
shape close to tasks.json's own fields.

Deliberately NOT a synthesis variant registered in
orchestrator.SYNTHESIS_VARIANTS, and doesn't touch orchestrator.py,
assemble_brief, or daily_brief.md at all — run standalone, any time after
the three MAP agents have populated ledgers, the same way citations.py
already runs standalone after the orchestrator. The existing markdown
pipeline keeps working exactly as it does today; this is additive.

Reuses Stage 0/1 verbatim from orchestrator.py (cross-reference index,
contradiction detection, freshness check, the shared synthesis context
builder) rather than reimplementing them — only Stage 2 is new: one LLM
call producing a flat list of typed items instead of three separate
prose/list outputs.

Two deterministic layers on top of the model's own output, same "don't
just ask nicely, verify it" posture as the rest of this codebase:
- date_urgency: overdue/due_today/due_soon/later/no_date, computed in
  Python from each item's due_date — never trusted from the model. Same
  7-day due_soon threshold tasks_signals.compute_task_signals already
  uses, so "urgent" means the same thing in both places.
- priority_disagreement: flagged (not corrected) when a model-assigned
  P3/P4 priority contradicts an overdue/due-today date_urgency — mirrors
  orchestrator.check_priority_coverage's "flag, don't block" posture.

Citations attach directly to each item's `summary` field via citations.py's
existing keyword_match_sources/llm_match_sources — no markdown bullet-
splitting needed, since there's no markdown to parse here at all.

Effort ("how much work remains") is explicitly out of scope: nothing in
the current data model tracks it, and inferring it from prose would just
reintroduce the LLM-judgment problem this is trying to get a deterministic
handle on.

Usage:
    python3 -m digest.core.structured_digest --tenant demo-1 --provider deepseek --model deepseek-chat
"""

import argparse
import json
import os
from datetime import date, datetime

from digest.core.citations import corpus_common_keywords, keyword_match_sources, llm_match_sources, load_citable_sources
from digest.core.cross_reference import build_cross_reference_index
from digest.core.ledger import check_data_freshness, load_ledger, validate_schema
from digest.core.llm import call_with_retry, create_llm
from digest.core.persona import load_persona
from digest.core.tenant_paths import for_tenant
from digest.core.tasks_signals import compute_task_signals
from digest.orchestrator import _build_synthesis_context, detect_contradictions
from digest.parsers.tasks_parser import load_tasks

STRUCTURED_DIGEST_SCHEMA = {"items": ["id", "type", "title", "summary", "priority"]}

# Same threshold tasks_signals.compute_task_signals uses for its own
# due_soon bucket — "urgent" means the same thing in both places.
_DUE_SOON_DAYS = 7
_LOW_PRIORITIES = {"P3", "P4"}
_URGENT_BUCKETS = {"overdue", "due_today"}


def build_structured_digest_prompt(persona_text: str) -> str:
    return (
        f"{persona_text}\n\n"
        "---\n\n"
        "You are the structured half of the operator's daily digest, for "
        "the operator profiled above. You'll be given the same source "
        "material (email/calendar/notes ledgers, task signals, "
        "cross-reference index, contradictions, freshness) a narrative "
        "digest would use — instead of writing prose, extract every "
        "distinct thing worth surfacing as one item in a flat list.\n\n"
        "For each item, decide:\n"
        "- type: \"action\" (something the operator needs to do or "
        "decide), \"information\" (worth knowing, nothing to do), or "
        "\"conflict\" (competing claims, a scheduling collision, or a "
        "contradiction).\n"
        "- priority: P0-P4, your holistic judgment — who it's from, what "
        "it's about, and urgency together, not just whether a date is "
        "close.\n"
        "- due_date: YYYY-MM-DD if the item genuinely has one, else null.\n"
        "- related_task_id: if this item corresponds to a task named in "
        "TASK SIGNALS, that task's id, else null.\n\n"
        "Output strictly valid JSON:\n"
        '{"items": [{"id": "item-001", '
        '"type": "action|information|conflict", "title": "...", '
        '"summary": "...", "priority": "P0-P4", '
        '"due_date": "YYYY-MM-DD or null", "related_task_id": "... or null"}]}\n\n'
        "Do not write any markdown wrappers, conversational pleasantries, "
        "or extra text."
    )


def date_urgency(due_date: str | None, reference_date: date) -> str:
    """"overdue" | "due_today" | "due_soon" (<= _DUE_SOON_DAYS) | "later" |
    "no_date" — computed in Python, never trusted from the model.
    """
    if not due_date:
        return "no_date"
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
    except ValueError:
        return "no_date"

    delta = (due - reference_date).days
    if delta < 0:
        return "overdue"
    if delta == 0:
        return "due_today"
    if delta <= _DUE_SOON_DAYS:
        return "due_soon"
    return "later"


def annotate_items(items: list[dict], reference_date: date) -> list[dict]:
    """Adds date_urgency (always) and priority_disagreement (only when it
    trips) to every item. Returns new dicts — never mutates the model's
    own fields.
    """
    annotated = []
    for item in items:
        urgency = date_urgency(item.get("due_date"), reference_date)
        disagreement = urgency in _URGENT_BUCKETS and item.get("priority") in _LOW_PRIORITIES
        out = dict(item)
        out["date_urgency"] = urgency
        if disagreement:
            out["priority_disagreement"] = True
        annotated.append(out)
    return annotated


def attach_citations(items: list[dict], sources: list[dict], llm=None) -> list[dict]:
    """Same two-pass matching citations.cite_brief uses (keyword pass
    first, batched LLM-judge fallback for anything unmatched), applied
    directly to each item's `summary` field instead of a markdown bullet
    — no split_citable_lines step needed, since there's no markdown here.
    """
    common_keywords = corpus_common_keywords(sources)

    keyword_matches: dict[int, list[dict]] = {}
    unmatched_indices: list[int] = []
    unmatched_texts: list[str] = []
    for i, item in enumerate(items):
        matched = keyword_match_sources(item["summary"], sources, common_keywords=common_keywords)
        if matched:
            keyword_matches[i] = matched
        else:
            unmatched_indices.append(i)
            unmatched_texts.append(item["summary"])

    llm_matches: dict[int, list[dict]] = {}
    if llm is not None and unmatched_texts:
        raw = llm_match_sources(llm, unmatched_texts, sources, common_keywords=common_keywords)
        for local_idx, matched_sources in raw.items():
            llm_matches[unmatched_indices[local_idx]] = matched_sources

    annotated = []
    for i, item in enumerate(items):
        out = dict(item)
        if i in keyword_matches:
            out["source_refs"] = [f"{s['source']}:{s['ref']}" for s in keyword_matches[i]]
        elif i in llm_matches:
            out["source_refs"] = [f"{s['source']}:{s['ref']}" for s in llm_matches[i]]
        else:
            out["source_refs"] = []
        annotated.append(out)
    return annotated


def build_structured_digest(llm, paths, persona_text: str) -> dict:
    """Runs Stage 0/1 reused verbatim from orchestrator.py, plus the new
    structured Stage 2, then both deterministic layers. Never writes or
    reads daily_brief.md or anything orchestrator.py owns.
    """
    email_ledger, _ = load_ledger(paths.email_ledger_file)
    calendar_ledger, _ = load_ledger(paths.calendar_ledger_file)
    notes_ledger, _ = load_ledger(paths.notes_ledger_file)
    tasks = load_tasks(paths.tasks_file)
    task_signals = compute_task_signals(tasks)

    if not (email_ledger or calendar_ledger or notes_ledger):
        raise SystemExit(
            "No source ledgers found. Run triage_agent.py / calendar_agent.py / notes_agent.py first."
        )

    cross_ref_index = build_cross_reference_index(email_ledger, calendar_ledger, notes_ledger, task_signals)
    freshness = check_data_freshness({"email": email_ledger, "calendar": calendar_ledger, "notes": notes_ledger})
    contradictions = detect_contradictions(llm, cross_ref_index, persona_text)

    prompt = build_structured_digest_prompt(persona_text)
    context = _build_synthesis_context(
        email_ledger, calendar_ledger, notes_ledger, task_signals,
        cross_ref_index, contradictions, freshness,
    )

    def _call():
        result = llm.chat_json(messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ])
        errors = validate_schema(result, STRUCTURED_DIGEST_SCHEMA)
        if errors:
            raise ValueError(f"Invalid structured-digest output: {errors}")
        return result

    result = call_with_retry(_call)
    items = annotate_items(result.get("items", []), date.today())

    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    items = attach_citations(items, sources, llm=llm)

    return {"items": items}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce a structured, per-item JSON digest alongside the existing markdown brief",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.core.structured_digest --tenant demo-1 --provider deepseek --model deepseek-chat\n"
        ),
    )
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = for_tenant(args.tenant)
    persona_text = load_persona(paths.persona_file)
    llm = create_llm(provider=args.provider, model=args.model, temperature=0.0, tenant_id=args.tenant)

    print(f"🧩 Building structured digest for tenant '{args.tenant}'...")
    digest = build_structured_digest(llm, paths, persona_text)

    disagreements = [i for i in digest["items"] if i.get("priority_disagreement")]
    if disagreements:
        print(f"   ⚠️  {len(disagreements)} item(s) flagged priority_disagreement (low priority, but overdue/due today):")
        for i in disagreements:
            print(f"      - {i['id']}: {i['title']} ({i['priority']}, {i['date_urgency']})")

    output_path = os.path.join(paths.output_dir, "daily_brief_structured.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2)

    print(f"   {len(digest['items'])} item(s) written.")
    print(f"   Wrote {output_path}")


if __name__ == "__main__":
    main()
