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
match **within one individual item** (one deadline, one action_item, one
key_person — see _entry_searchable_items), not just anywhere in the
day's/note's whole entry. That item-level requirement was added after
verifying a real false positive against live ledger data: an earlier
version matched 2 keywords found in two *different, unrelated* items that
happened to land in the same day's batch (e.g. one keyword in a receipts
reminder, the other in an unrelated customer-renewal thread), which is not
a real mention of anything. Item-level matching closes that, but leaves a
narrower residual: two keywords can still both land within one item's own
description+source_subject+owner text where the subject line drifted from
the body's actual topic (verified as occasional, not systemic) — an
accepted tradeoff, not a silent gap. The LLM stages downstream
(contradiction detection, synthesis) are specifically designed to apply
judgment on top of these candidates rather than trust every match as
confirmed fact.

build_cross_reference_index above stays purely lexical by design — this
index directly seeds Stage 1's contradiction-detection candidate set, so
a false match here has a bigger blast radius than a false positive
anywhere else this system does keyword matching (it could feed a
hallucinated contradiction about two things that were never actually the
same task). An embedding-assisted alternative exists as a genuinely
separate implementation (build_cross_reference_index_embedding_assisted,
see CROSS_REFERENCE_VARIANTS below) rather than a modification of this
one, specifically so it can be compared against real data
(eval_cross_reference_variants.py) before any decision to change what
Stage 0 actually runs by default — see docs/HIGH_LEVEL_DESIGN.md's Stage
0 section for the current status of that comparison.

Usage:
    from digest.core.cross_reference import build_cross_reference_index

    index = build_cross_reference_index(email_ledger, calendar_ledger, notes_ledger, task_signals)
"""

import re
from datetime import date

from digest.core.embeddings import cosine_similarity, embed_texts

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "on", "in", "to", "with",
    "is", "are", "at", "by", "from", "this", "that",
    # Common short (3-letter) English words that would otherwise pass the
    # lowered min_length below and pollute matches — not exhaustive, same
    # accepted-tradeoff reasoning as the 2-keyword-match threshold below.
    # Needed once min_length dropped from 4 to 3 to stop legitimate short
    # proper nouns (e.g. "Sam" — the persona's actual P0 personal contact)
    # from being silently unindexable.
    "fix", "get", "now", "due", "new", "add", "run", "set", "put", "use",
    "our", "you", "all", "not", "but", "yet", "off", "out", "via",
}

_LEADING_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def title_keywords(title: str, min_length: int = 3) -> list[str]:
    """Extract meaningful keywords from a task title for matching.

    Filters short/common words to avoid false-positive matches (e.g. "the",
    "a") while keeping the specific nouns that make a match meaningful.
    min_length is 3, not 4 — short proper nouns ("Sam", "Kim", "Raj") are
    real and matter; the extra noise that lets in is handled by extending
    _STOPWORDS above instead.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
    return [w for w in words if len(w) >= min_length and w.lower() not in _STOPWORDS]


def leaf_strings(obj) -> list[str]:
    """Recursively collect leaf string values only — never dict keys.

    This is what keeps JSON schema field names out of the search corpus.
    Every MAP delta item has a "description" field, so naively including
    dict keys (e.g. via json.dumps) means any task title containing the
    word "Description" would match 100% of ledger entries for free —
    verified directly against real ledger data, not hypothetical.
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(leaf_strings(v))
        return result
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(leaf_strings(item))
        return result
    return []


def _entry_searchable_items(entry: dict) -> list[tuple[str, str]]:
    """Yield (category, text) pairs — one per individual *item* within each
    delta/stats category, not merged across sibling items.

    This is the granularity a genuine mention needs to be judged at, found
    necessary while verifying the field-level version of this function
    against real data: a day's `action_items` (or `thread_progressions`,
    etc.) is a *list*, often containing several unrelated items from
    several unrelated emails that just happened to land the same day.
    Flattening a whole category into one string meant two keywords landing
    in two different, unrelated items within the same category (e.g. one
    matching in an unrelated receipt-reminder deadline, the other in an
    unrelated Halberd-renewal thread) still counted as one "mention" —
    verified as a live false positive against real ledger data, not
    hypothetical. Requiring both keywords within one *item's* own text
    closes that gap, at the cost of missing a genuine mention split across
    two different items/categories (e.g. a name in notes' key_people and a
    separate decision about them) — an accepted precision-over-recall
    tradeoff, consistent with this module's existing stance.
    """
    items = []
    for section_name in ("delta", "stats"):
        section = entry.get(section_name)
        if not isinstance(section, dict):
            continue
        for category, value in section.items():
            entries = value if isinstance(value, list) else [value]
            for item in entries:
                text = " ".join(leaf_strings(item))
                if text:
                    items.append((category, text))
    return items


def find_mentions(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of keywords that appear (case-insensitive, on a
    word boundary) in text.

    Word-boundary, not substring: a plain substring check would match
    "Marsh" against "Marshall" or "board" against "onboarding" — both
    real, verified false-positive shapes, not hypothetical.
    """
    lowered = text.lower()
    return [kw for kw in keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lowered)]


def _excerpt_around_match(text: str, keywords: list[str], window: int = 250) -> str:
    """Return a window of text centered on the earliest keyword match,
    instead of an unconditional prefix slice.

    A fixed text[:500] silently misses the actual match on any entry
    longer than 500 characters — measured directly against real ledger
    data: most entries exceed that, so the old approach routinely handed
    downstream LLM stages an excerpt that didn't contain the thing that
    matched at all.
    """
    lowered = text.lower()
    earliest = None
    for kw in keywords:
        pos = lowered.find(kw.lower())
        if pos != -1 and (earliest is None or pos < earliest):
            earliest = pos
    if earliest is None:
        return text[:window]
    start = max(0, earliest - window // 2)
    return text[start:start + window]


def _parse_leading_date(s: str) -> date | None:
    """Extract a YYYY-MM-DD date prefix from a day/note_id string.

    Returns None for anything unparseable ("unknown", compacted week keys
    like "2026-W25") rather than raising — same "skip what can't be
    handled" posture this module already takes toward compacted entries.
    """
    match = _LEADING_DATE_RE.match(s)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


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
        # Best-matching individual item wins — not an aggregate across
        # every item in the entry (see _entry_searchable_items for why).
        best_category, best_matched, best_text = None, set(), ""
        for category, text in _entry_searchable_items(entry):
            matched = set(find_mentions(text, keywords))
            if len(matched) > len(best_matched):
                best_category, best_matched, best_text = category, matched, text
        if len(best_matched) >= min_matches:
            mentions.append({
                "source": source_name,
                "day": entry.get("note_id") or entry.get("day", "unknown"),
                "matched_keywords": sorted(best_matched),
                "matched_fields": [best_category],
                # Context window around the actual match — downstream
                # contradiction-detection needs real text to reason over,
                # not just which keywords matched.
                "excerpt": _excerpt_around_match(best_text, sorted(best_matched)),
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
        for every flagged task with at least one cross-source mention. Each
        mention also carries "days_from_due" (mention date minus the
        task's due date; negative means the mention predates the deadline)
        when both dates are parseable, else None.
    """
    flagged_tasks = {}
    for bucket in ("overdue", "due_soon", "blocked", "stalled"):
        for task in task_signals.get(bucket, []):
            flagged_tasks[task["id"]] = task

    index = {}
    for task_id, task in flagged_tasks.items():
        keywords = title_keywords(task["title"])
        if not keywords:
            continue

        mentions = (
            _scan_ledger("email", email_ledger, keywords)
            + _scan_ledger("calendar", calendar_ledger, keywords)
            + _scan_ledger("notes", notes_ledger, keywords)
        )

        if not mentions:
            continue

        due = _parse_leading_date(task["due_date"]) if task.get("due_date") else None
        for mention in mentions:
            mention_date = _parse_leading_date(mention["day"])
            mention["days_from_due"] = (mention_date - due).days if (due and mention_date) else None

        index[task_id] = {
            "title": task["title"],
            "priority": task.get("priority", "?"),
            "mentioned_in": mentions,
        }

    return index


# ─── Embedding-Assisted Variant (separate implementation, not a replacement) ──
#
# build_cross_reference_index above is untouched by everything below — this
# is a genuinely parallel implementation, not a modification of it, exposed
# alongside it via CROSS_REFERENCE_VARIANTS for orchestrator.py's
# --cross-reference-variant flag and eval_cross_reference_variants.py to
# compare against real data before any decision to change the default.
#
# Higher stakes than citations.py's embedding tier: this index directly
# seeds Stage 1's contradiction-detection candidate set, so a false
# cross-reference here could feed a hallucinated contradiction about two
# things that were never actually the same task — not just mislabel one
# citation, as a false positive in citations.py would. That's the reason
# this stays a separate, evaluated variant rather than folding straight
# into the lexical function the way citations.py's tier did.

# Live-checked, not just guessed and left alone — and deliberately kept
# higher than the other two embedding thresholds in this codebase
# (citations.py's 0.70, orchestrator.check_priority_coverage's 0.65).
# Scanned every non-lexically-covered item in arclight's real data for
# ARC-102 ("Open senior SRE req and run interviews"): the highest
# similarity found among genuinely uncovered items was 0.524 ("SRE
# capacity tight this sprint", "Send Q3 eng headcount ask to Om today") —
# real, topically related, but not confidently "the same mention," and
# well short of qualifying at any of this codebase's other calibrated
# thresholds either. Left at 0.70 rather than lowered to catch these:
# this index directly seeds Stage 1's contradiction-detection candidates,
# so the higher precision bar is a deliberate asymmetry with the lower-
# stakes, advisory-only checks elsewhere, not an oversight.
_EMBEDDING_SIMILARITY_THRESHOLD = 0.70


def _embed_ledger_entries(ledger: list[dict], embed_fn) -> list[tuple]:
    """Pre-embeds every non-compacted entry's searchable items once —
    same 'embed the corpus once, reuse across every comparison'
    discipline as citations.py's cite_brief, since this gets reused
    across every flagged task's comparison, not recomputed per task.

    Returns [(entry, category, text, vector), ...]. Compacted entries are
    skipped, same reasoning _scan_ledger already documents (per-day
    granularity lost by design — see ledger.compact_ledger).
    """
    meta = []
    texts = []
    for entry in ledger:
        if entry.get("compacted"):
            continue
        for category, text in _entry_searchable_items(entry):
            meta.append((entry, category, text))
            texts.append(text)

    if not texts:
        return []
    vectors = embed_fn(texts)
    return [(entry, category, text, vector) for (entry, category, text), vector in zip(meta, vectors)]


def _scan_ledger_embedding(
    source_name: str, embedded_items: list[tuple], task_vector: list[float],
    already_covered_days: set, threshold: float = _EMBEDDING_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """OR-rescue pass: adds a mention for an entry's best-matching item
    when it clears embedding similarity — but only for days
    `already_covered_days` (the lexical _scan_ledger pass's own results)
    didn't already cover for this task. Lexical always wins when both
    would match on the same day; this only fills gaps the lexical pass
    left, mirroring _scan_ledger's own "best-matching item per entry
    wins" posture.

    Mentions are tagged "matched_via": "embedding" (lexical mentions have
    no such key) so provenance stays visible downstream — Stage 1's
    prompt, and eval_cross_reference_variants.py's scoring.
    """
    best_by_entry = {}
    for entry, category, text, vector in embedded_items:
        day = entry.get("note_id") or entry.get("day", "unknown")
        if day in already_covered_days:
            continue
        sim = cosine_similarity(task_vector, vector)
        key = id(entry)
        if key not in best_by_entry or sim > best_by_entry[key][0]:
            best_by_entry[key] = (sim, category, text, day)

    mentions = []
    for sim, category, text, day in best_by_entry.values():
        if sim >= threshold:
            mentions.append({
                "source": source_name,
                "day": day,
                "matched_keywords": [],
                "matched_fields": [category],
                "excerpt": text[:250],
                "matched_via": "embedding",
                "similarity": round(sim, 3),
            })
    return mentions


def build_cross_reference_index_embedding_assisted(
    email_ledger: list[dict],
    calendar_ledger: list[dict],
    notes_ledger: list[dict],
    task_signals: dict,
    embed_fn=embed_texts,
) -> dict:
    """Same output shape as build_cross_reference_index (see its
    docstring for the field-by-field contract) — a separate
    implementation, not a modification of it. Runs the identical lexical
    _scan_ledger pass first for every flagged task (the existing
    precision floor never regresses), then adds embedding-matched
    mentions only for whatever the lexical pass missed on a given
    day/source.

    Batch-embeds all flagged task titles once upfront (not once per
    task) plus each ledger once (via _embed_ledger_entries), reusing both
    across every task's comparison — the same "embed the corpus once"
    discipline as citations.py. Degrades to the pure-lexical result
    (prints a warning) if embed_fn raises, e.g. Ollama isn't running.
    """
    flagged_tasks = {}
    for bucket in ("overdue", "due_soon", "blocked", "stalled"):
        for task in task_signals.get(bucket, []):
            flagged_tasks[task["id"]] = task

    lexical_by_task = {}
    for task_id, task in flagged_tasks.items():
        keywords = title_keywords(task["title"])
        if not keywords:
            continue
        lexical_by_task[task_id] = (
            _scan_ledger("email", email_ledger, keywords)
            + _scan_ledger("calendar", calendar_ledger, keywords)
            + _scan_ledger("notes", notes_ledger, keywords)
        )

    embedded_by_source = {}
    task_vectors = {}
    if embed_fn is not None and lexical_by_task:
        try:
            embedded_by_source = {
                "email": _embed_ledger_entries(email_ledger, embed_fn),
                "calendar": _embed_ledger_entries(calendar_ledger, embed_fn),
                "notes": _embed_ledger_entries(notes_ledger, embed_fn),
            }
            task_ids = list(lexical_by_task.keys())
            vectors = embed_fn([flagged_tasks[tid]["title"] for tid in task_ids])
            task_vectors = dict(zip(task_ids, vectors))
        except Exception as e:
            print(f"   ⚠️  Embedding-assisted cross-referencing unavailable ({e}) — using lexical-only result.")
            embedded_by_source, task_vectors = {}, {}

    index = {}
    for task_id, lexical_mentions in lexical_by_task.items():
        task = flagged_tasks[task_id]
        covered_days = {m["day"] for m in lexical_mentions}

        embedding_mentions = []
        if task_id in task_vectors:
            task_vector = task_vectors[task_id]
            for source_name in ("email", "calendar", "notes"):
                embedding_mentions += _scan_ledger_embedding(
                    source_name, embedded_by_source.get(source_name, []), task_vector, covered_days,
                )

        mentions = lexical_mentions + embedding_mentions
        if not mentions:
            continue

        due = _parse_leading_date(task["due_date"]) if task.get("due_date") else None
        for mention in mentions:
            mention_date = _parse_leading_date(mention["day"])
            mention["days_from_due"] = (mention_date - due).days if (due and mention_date) else None

        index[task_id] = {
            "title": task["title"],
            "priority": task.get("priority", "?"),
            "mentioned_in": mentions,
        }

    return index


CROSS_REFERENCE_VARIANTS = {
    "lexical": build_cross_reference_index,
    "embedding_assisted": build_cross_reference_index_embedding_assisted,
}
