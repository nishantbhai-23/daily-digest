"""
Cross-Reference Index
=======================
Stage 0 of the orchestrator: a deterministic (no LLM) index of where each
time-sensitive task shows up outside tasks.json — email, calendar, and
notes ledger entries that mention the same task by title keywords.

Same category of thing as calendar_agent.compute_day_stats or
triage_agent.compute_sender_staleness: finding that two sources mention the
same thing is mechanical string matching, not judgment. What that overlap
*means* — is it worth a special callout, is it evidence something's
stalled — belongs to the LLM synthesis stage downstream, not this module.

Only tasks already flagged as time-sensitive by tasks_signals
(overdue/due_soon/blocked/stalled) get indexed — an on-track task doesn't
need cross-referencing, since it isn't "at risk" in the first place.

Known precision limit: keyword matching is deliberately blunt (same
philosophy as digest_checks.py) and requires 2+ distinct title keywords to
match, which rules out the worst false positives (a single generic word
like "offer" coincidentally appearing elsewhere) but not all of them — two
generic words together (e.g. "Senior"+"Engineer" matching a different
hiring thread than the one intended) can still produce a false positive.
This is an accepted tradeoff, not a silent gap: the module documents it,
and the LLM stages downstream (contradiction detection, synthesis) are
specifically designed to apply judgment on top of these candidates rather
than trust every match as confirmed fact.

Usage:
    from digest.core.cross_reference import build_cross_reference_index

    index = build_cross_reference_index(email_ledger, calendar_ledger, notes_ledger, task_signals)
"""

import json
import re

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "on", "in", "to", "with",
    "is", "are", "at", "by", "from", "this", "that",
}


def _title_keywords(title: str, min_length: int = 4) -> list[str]:
    """Extract meaningful keywords from a task title for matching.

    Filters short/common words to avoid false-positive matches (e.g. "the",
    "a") while keeping the specific nouns that make a match meaningful.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
    return [w for w in words if len(w) >= min_length and w.lower() not in _STOPWORDS]


def _entry_searchable_text(entry: dict) -> str:
    """Flatten a ledger entry's delta and stats into one searchable string.

    Deliberately includes 'stats' too — e.g. calendar's notable_events and
    declined-event lists carry real names worth matching against, not just
    the LLM-derived delta.
    """
    parts = []
    if "delta" in entry:
        parts.append(json.dumps(entry["delta"]))
    if "stats" in entry:
        parts.append(json.dumps(entry["stats"]))
    return " ".join(parts)


def _find_mentions(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of keywords that appear (case-insensitive) in text."""
    lowered = text.lower()
    return [kw for kw in keywords if kw.lower() in lowered]


def _scan_ledger(source_name: str, ledger: list[dict], keywords: list[str]) -> list[dict]:
    """Scan one source's ledger for entries mentioning a task's keywords.

    Requires at least 2 distinct keyword matches (or all of them, for
    short titles) — a single generic word like "offer" or "Engineer"
    coincidentally appearing elsewhere isn't a real mention of the task,
    and requiring just one match produced exactly that kind of false
    positive when checked against real ledger data during development.
    """
    min_matches = min(2, len(keywords))
    mentions = []
    for entry in ledger:
        if entry.get("compacted"):
            # Compacted weekly entries lose per-day granularity by design
            # (see ledger.compact_ledger) — skipped rather than reported at
            # week granularity; a known scope cut, not a silent gap.
            continue
        searchable = _entry_searchable_text(entry)
        matched = _find_mentions(searchable, keywords)
        if len(matched) >= min_matches:
            mentions.append({
                "source": source_name,
                "day": entry.get("note_id") or entry.get("day", "unknown"),
                "matched_keywords": matched,
                # Bounded excerpt of the entry's actual content — downstream
                # contradiction-detection needs real text to reason over,
                # not just which keywords matched.
                "excerpt": searchable[:500],
            })
    return mentions


def build_cross_reference_index(
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
) -> dict:
    """Build a deterministic index of time-sensitive-task-to-source mentions.

    Args:
        email_ledger, calendar_ledger, notes_ledger: the three source
            ledgers (from ledger.load_ledger).
        task_signals: output of tasks_signals.compute_task_signals — only
            tasks already flagged as overdue/due_soon/blocked/stalled are
            indexed, not every task in tasks.json.

    Returns:
        {task_id: {"title": ..., "priority": ..., "mentioned_in": [...]}}
        for every flagged task with at least one cross-source mention.
    """
    flagged_tasks = {}
    for bucket in ("overdue", "due_soon", "blocked", "stalled"):
        for task in task_signals.get(bucket, []):
            flagged_tasks[task["id"]] = task

    index = {}
    for task_id, task in flagged_tasks.items():
        keywords = _title_keywords(task["title"])
        if not keywords:
            continue

        mentions = (
            _scan_ledger("email", email_ledger, keywords)
            + _scan_ledger("calendar", calendar_ledger, keywords)
            + _scan_ledger("notes", notes_ledger, keywords)
        )

        if mentions:
            index[task_id] = {
                "title": task["title"],
                "priority": task.get("priority", "?"),
                "mentioned_in": mentions,
            }

    return index
