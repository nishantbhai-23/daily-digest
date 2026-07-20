"""
Post-Processing Citations
============================
Attaches per-bullet source references (which email/calendar-event/note a
digest claim actually came from) to an already-generated brief, entirely
after the fact — a separate, optional pass that touches no MAP or
synthesis prompt, schema, or existing pipeline stage. Not wired into
orchestrator.py; run it standalone against a brief that already exists.

Deliberately post-hoc, not generation-time citation, even though citation
research generally favors generation-time faithfulness for open-corpus
RAG. That doesn't apply the same way here: every email/event/note that
could have produced a given digest line was already in the LLM's context
that day — a small, fully-known, closed candidate set (5-30 items), not
open retrieval. The matching problem shrinks to "which of these known
items does this claim match," a narrower, safer version of what that
research warns about.

Hybrid matching, not keyword-only: a digest bullet is the LLM's
*abstractive summary* of a source, not a literal quote, so keyword overlap
alone under-matches it in a way cross_reference.py's original use case
(matching a stable task title against mentions) doesn't hit. Keyword
matching (cross_reference.title_keywords/find_mentions) runs first as a
free, deterministic pass; bullets it can't place fall back to one batched
LLM-judge call, grounded the same way orchestrator._ground_contradictions
grounds Stage 1 — a claimed source_ref that isn't actually one of the refs
the model was shown gets dropped, not trusted. A bullet with no match from
either pass stays visibly uncited rather than forcing a guess.

The keyword pass needed its own tuning, found live rather than guessed: a
real brief's first bullet matched 9 of 10 available sources before this
fix. Two compounding causes. First, cross_reference.py's min(2, len(keywords))
threshold was tuned for short task titles (3-8 words); a digest bullet can
run 20-40 words and produce 15-25 extracted keywords, so requiring only 2
matches out of that many is a trivial bar — _MIN_MATCH_RATIO scales the
floor with the bullet's own keyword count instead. Second, title_keywords'
stopword list is tuned for title vocabulary and doesn't cover the pronouns/
auxiliary verbs ordinary prose is full of ("your", "have", "did", "need",
"today") — _PROSE_STOPWORDS layers a second filter on top of
title_keywords' own output rather than editing title_keywords itself,
since that function's behavior is already tested and correct for its
original (title-matching) use case. Erring toward filtering more, not
less: a false negative here just falls through to the LLM-judge fallback,
which is exactly its job; a false positive here has no correction
mechanism and ships a wrong citation directly.

LLM-judge grounding vs. correctness (found live, fixed three times in a
row against the same real bug before it actually closed): the first
version of llm_match_sources only checked that a claimed source_ref
existed among the candidates shown to the model — "faithfulness," not
"correctness" in the citation-literature sense. That let through a real
misattribution: a bullet about "Press #2 jam cleared" got cited to an
email that was actually about unrelated "Press #3 vibration," a real ref
with zero actual support.

Fix 1: require the model to supply a verbatim quote from the source and
verify it's a real substring (layer 2). Re-testing live, the *same* claim
still mis-cited — the model quoted the email's own subject line ("Press #3
vibration analysis") as "evidence," a genuine substring but irrelevant.

Fix 2: also require the quote and the claim to share a real,
corpus-filtered keyword (layer 3). Re-testing live *again*, the same claim
still mis-cited, because this specific corpus (14 sources, "press"
appearing in 5 of them) sat right under corpus_common_keywords' 0.5
threshold — "press" alone was enough shared vocabulary to pass layer 3
even though "Press #2" and "Press #3" are different machines.

Fix 3: if the claim names a specific numbered/lettered instance (a
"#N"-shaped token), require the quote to name that same instance, not
just the surrounding category noun (layer 4). This is what finally closed
the concrete bug: "#2" vs "#3" never overlap, regardless of how much
surrounding vocabulary ("press", "vibration") the two machines share.

Each fix in this chain was verified by literally re-running the citation
pass against the same live tenant and reading the output, not assumed —
same "don't just ask nicely, verify it" discipline as the rest of this
codebase, applied to the fix itself, not just the original feature.

Known, accepted limitation: layers 1-4 close every concrete failure mode
found so far, including the one that survived two earlier fixes, but none
of them verify semantic correctness the way a human re-reading the source
would. A model could in principle still find a real, keyword-overlapping
quote from the wrong source when neither claim nor source happens to
contain a "#N"-shaped identifier to disambiguate by. Closing that fully
would require another LLM call to judge the judge, which just relocates
the same trust problem — not pursued, consistent with this file's
"post-hoc, closed candidate set" scope rather than a fully
adversarial-input citation system.

Usage:
    python3 -m digest.core.citations --tenant demo-1 --provider deepseek --model deepseek-chat
    python3 -m digest.core.citations --tenant demo-1 --keyword-only
"""

import argparse
import os
import re

from digest.core.cross_reference import find_mentions, leaf_strings, title_keywords
from digest.core.ledger import validate_schema
from digest.core.llm import call_with_retry, create_llm
from digest.core.tenant_paths import for_tenant
from digest.parsers.calendar_parser import load_calendar
from digest.parsers.email_parser import load_inbox
from digest.parsers.notes_parser import load_notes
from digest.parsers.tasks_parser import load_tasks

CITATION_JUDGE_SCHEMA = {"matches": ["claim_index", "evidence"]}

_UNDRAFTED_LINE = "*(surfaced for you to handle directly — not drafted)*"

# Layered on top of title_keywords' own stopword filtering — see module
# docstring. Common pronouns, auxiliary/modal verbs, and relative-time
# words that pollute prose keyword extraction but rarely appear in a task
# title, so title_keywords' own list doesn't cover them. Not exhaustive,
# same "blunt but documented, erring toward over-filtering" tradeoff used
# throughout this codebase's other keyword-matching code.
_PROSE_STOPWORDS = {
    "you", "your", "yours", "have", "has", "had", "do", "does", "did",
    "already", "need", "needs", "needed", "today", "yesterday", "tomorrow",
    "top", "yet", "was", "were", "will", "would", "could", "should",
    "also", "just", "still", "get", "gets", "got", "make", "makes",
    "made", "may", "might", "can", "cannot", "into", "onto", "any",
    "some", "when", "then", "than", "more", "most", "much", "many",
    "over", "before", "after", "again", "here", "there", "what", "which",
    "who", "whom",
}

# Matches a specific numbered/lettered instance within a shared category
# noun — "Press #2" vs "Press #3", "Invoice #4521", "Room #12". Found live
# in llm_match_sources: two sources can share every ordinary keyword
# ("press", "vibration"/"jam" aside) while actually describing different
# specific instances, and neither quote-existence nor plain keyword
# overlap catches that — both quotes are real, and "press" alone isn't
# always common enough across a brief's sources to get filtered by
# corpus_common_keywords. An identifier token is unambiguous by
# construction: if the claim names one, a genuine supporting quote from
# the right source should name the same one, not a different one.
_IDENTIFIER_TOKEN_RE = re.compile(r"#\d+")


def _identifier_tokens(text: str) -> set[str]:
    return set(_IDENTIFIER_TOKEN_RE.findall(text))

# A flat min(2, len(keywords)) works for cross_reference.py's short task
# titles but floods on long prose bullets — see module docstring. Scales
# the match floor with the bullet's own keyword count instead.
_MIN_MATCH_RATIO = 0.4

# A word appearing in more than this fraction of a brief's own candidate
# sources isn't distinguishing for that brief, even if it isn't a generic
# English stopword — found live: "Ocean"/"Pride" appeared in 6 of 10
# sources because the whole tenant's planted scenario centers on one
# storyline, which let a short bullet ("Confirm availability for Ocean
# Pride inspection") clear the match threshold against 6 sources on two
# words alone. No static stopword list can anticipate a dataset-specific
# frequent term like this, so corpus_common_keywords computes it fresh per
# brief instead.
_CORPUS_COMMON_THRESHOLD = 0.5

# With very few sources, "appears in every source" is trivially true for
# almost any word (one source's whole vocabulary is "common" by
# definition) — this would wipe out every keyword rather than filtering
# genuinely over-represented ones. Below this many sources, corpus-
# frequency filtering isn't a meaningful signal, so it's skipped entirely.
_MIN_SOURCES_FOR_CORPUS_FILTER = 4


def corpus_common_keywords(sources: list[dict], threshold: float = _CORPUS_COMMON_THRESHOLD) -> set[str]:
    """Lowercased keywords appearing in more than `threshold` fraction of
    the given sources — computed once per brief (see cite_brief) and
    layered into keyword_match_sources on top of _PROSE_STOPWORDS, the
    same "erring toward over-filtering" reasoning: a keyword excluded here
    just means that bullet falls through to the LLM fallback instead,
    which is a safe outcome, not a lost one.
    """
    if len(sources) < _MIN_SOURCES_FOR_CORPUS_FILTER:
        return set()

    doc_counts: dict[str, int] = {}
    for source in sources:
        words_in_source = {kw.lower() for kw in title_keywords(source["text"])}
        for word in words_in_source:
            doc_counts[word] = doc_counts.get(word, 0) + 1

    n = len(sources)
    return {word for word, count in doc_counts.items() if count / n > threshold}


def load_citable_sources(inbox_dir: str, calendar_file: str, notes_dir: str, tasks_file: str | None = None) -> list[dict]:
    """Re-parses raw source files independently of the MAP/ledger pipeline
    — the ledger's MAP deltas don't carry file-level references
    (source_subject is a weak, ambiguous stand-in that collides across a
    reply thread's multiple emails), so this goes straight to the same
    parsers MAP already uses and keeps each format's real reference:
    filename for email, uid for calendar, note_id for notes, task id for
    tasks.

    tasks_file is optional (default None, meaning tasks aren't included) —
    tasks.json isn't loaded through the same MAP pipeline as the other
    three sources, so a caller only interested in email/calendar/notes
    doesn't need to supply it.

    Returns:
        [{"source": "email"|"calendar"|"notes"|"task", "ref": str,
        "label": str, "text": str}] — text is leaf-extracted searchable
        content; label is a short human-readable description; ref is what
        gets rendered as the citation tag.
    """
    sources = []

    for email in load_inbox(inbox_dir):
        text = " ".join(leaf_strings({"subject": email.get("subject", ""), "body": email.get("body", "")}))
        sources.append({
            "source": "email",
            "ref": email.get("filename") or email.get("message_id") or "unknown",
            "label": email.get("subject", "(no subject)"),
            "text": text,
        })

    for event in load_calendar(calendar_file):
        text = " ".join(leaf_strings({
            "summary": event.get("summary", ""),
            "description": event.get("description", ""),
            "location": event.get("location", ""),
            "attendees": event.get("attendees", []),
        }))
        sources.append({
            "source": "calendar",
            "ref": event.get("uid") or "unknown",
            "label": event.get("summary", "(no summary)"),
            "text": text,
        })

    for note in load_notes(notes_dir):
        text = " ".join(leaf_strings({"title": note.get("title", ""), "body": note.get("body", "")}))
        sources.append({
            "source": "notes",
            "ref": note.get("note_id") or "unknown",
            "label": note.get("title") or note.get("note_id") or "(note)",
            "text": text,
        })

    if tasks_file:
        for task in load_tasks(tasks_file):
            text = " ".join(leaf_strings({"title": task.get("title", ""), "description": task.get("description", "")}))
            sources.append({
                "source": "task",
                "ref": task.get("id") or "unknown",
                "label": task.get("title") or task.get("id") or "(task)",
                "text": text,
            })

    return sources


def split_citable_lines(markdown_text: str) -> list[str]:
    """Splits a rendered brief into citable lines.

    Skips '#'-prefixed headers, '>'-prefixed lines (the freshness notice
    AND drafted-reply blockquotes — a draft answering an item doesn't need
    its own citation separate from the item's own summary line above it),
    the fixed "not drafted" placeholder line, and blank lines. Works
    uniformly across all three of assemble_brief's sections (prose bullets
    in What Matters/What Might Be Missing, and the **bold summary** line
    per Quick Dispatches item) without section-specific parsing, since
    assemble_brief's header/blockquote conventions are fixed, code-authored
    strings, not LLM-generated text.
    """
    lines = []
    for raw_line in markdown_text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith(">"):
            continue
        if stripped == _UNDRAFTED_LINE:
            continue
        lines.append(stripped)
    return lines


def keyword_match_sources(bullet_text: str, sources: list[dict], common_keywords: set[str] | None = None) -> list[dict]:
    """cross_reference's title_keywords/find_mentions, run in the opposite
    direction (bullet -> source instead of task title -> mentions), with
    three adjustments for prose rather than titles — see module docstring:
    _PROSE_STOPWORDS layered on top of title_keywords' own filtering, a
    match floor that scales with the bullet's own keyword count
    (_MIN_MATCH_RATIO) instead of a flat min(2, len(keywords)), and
    optionally `common_keywords` (see corpus_common_keywords) to exclude
    words too frequent across this specific brief's own sources to be
    distinguishing. `common_keywords` defaults to none excluded — callers
    matching against a full brief's source set should pass
    corpus_common_keywords(sources) (see cite_brief); direct/test calls
    against a small, deliberately distinct fixture set can omit it.

    Returns every source clearing the threshold, sorted by match strength
    (not just the single best) — a bullet synthesizing an email+note
    cross-reference should cite both. Empty list means keyword matching
    found nothing — not necessarily uncited, the LLM fallback still gets a
    chance at it.
    """
    common = common_keywords or set()
    keywords = [
        kw for kw in title_keywords(bullet_text)
        if kw.lower() not in _PROSE_STOPWORDS and kw.lower() not in common
    ]
    if not keywords:
        return []

    min_matches = max(2, round(len(keywords) * _MIN_MATCH_RATIO))
    scored = []
    for source in sources:
        matched = find_mentions(source["text"], keywords)
        if len(matched) >= min_matches:
            scored.append((len(matched), source))

    scored.sort(key=lambda pair: -pair[0])
    return [source for _, source in scored]


def build_citation_judge_prompt() -> str:
    return (
        "You are matching claims in a summary to the source documents they "
        "came from. You will receive a numbered list of claims and a "
        "numbered list of candidate source documents (each an email, "
        "calendar event, note, or task, shown as 'N. [source_type:ref] "
        "label — text').\n\n"
        "For each claim, decide which source document(s), if any, "
        "genuinely support or describe that claim's content — not just "
        "topically related, the actual source the claim was drawn from. A "
        "claim can match zero, one, or multiple sources.\n\n"
        "For every source you match, you must include a short verbatim "
        "quote — a contiguous span copied character-for-character from "
        "that source's own text above — that supports the claim. Do not "
        "paraphrase, summarize, or invent the quote. If you cannot find "
        "an exact span in a source that supports the claim, do not match "
        "that source at all, even if it seems topically related.\n\n"
        "Output strictly valid JSON:\n"
        '{"matches": [{"claim_index": 0, "evidence": '
        '[{"source_ref": "ref-shown-in-brackets", "quote": "exact text copied from that source"}]}]}\n\n'
        "source_ref must be exactly the ref value shown in the source "
        "list's brackets (e.g. '0003.eml' or an event's uid) — never the "
        "source's list number. Include exactly one entry per claim index "
        "given, even when evidence is an empty list. Do not write any "
        "markdown wrappers, conversational pleasantries, or extra text."
    )


def llm_match_sources(llm, unmatched: list[str], sources: list[dict], common_keywords: set[str] | None = None) -> dict[int, list[dict]]:
    """One batched call covering every keyword-unmatched bullet at once,
    not one call per bullet. Grounds the response before trusting it in
    four layers, not one — each added after the previous layer(s) still
    let the same real bug through on live re-testing (see module
    docstring for the full story):

    1. Ref grounding (same shape as orchestrator._ground_contradictions):
       any claimed source_ref that isn't actually one of the refs in
       `sources` gets dropped — catches an invented/hallucinated ref.
    2. Quote existence: the model must also supply a verbatim quote copied
       from that source's text. If the quote isn't actually a substring of
       the source (case-insensitive), the match is dropped — a real, valid
       ref can still be a *wrong* match (plausible-sounding but actually
       unrelated), which ref-grounding alone can't catch since it only
       checks the ref existed, not that it supports the claim.
    3. Quote relevance: a real quote can still be a red herring — the
       model quoted an email's own subject line as "evidence" for an
       unrelated claim, a real, verbatim, but irrelevant span. This layer
       requires the quote and the claim to share at least one real
       keyword (same title_keywords/_PROSE_STOPWORDS primitives as
       keyword_match_sources, with the same corpus-common filtering so a
       frequent word can't carry a match by itself).
    4. Identifier disambiguation: two different specific instances of the
       same category noun ("Press #2" vs "Press #3") can share every
       ordinary keyword and still be wrong, especially when the shared
       noun isn't frequent enough across the corpus for layer 3's
       corpus-common filtering to exclude it. If the claim names a
       "#N"-shaped token, the quote must name the same one.

    None of this proves semantic correctness the way a human re-reading
    the source would — a model could still find a real, keyword-
    overlapping quote from the wrong source when neither side has a
    "#N"-shaped identifier to disambiguate by. That residual risk doesn't
    have a deterministic fix and is a known, documented limitation (see
    module docstring), not something layers 1-4 claim to close entirely.

    Returns:
        {claim_index: [source, ...]} — only for indices the model actually
        returned a grounded, quote-verified, relevant, disambiguated match
        for.
    """
    if not unmatched:
        return {}
    common = common_keywords or set()

    claims_text = "\n".join(f"{i}. {text}" for i, text in enumerate(unmatched))
    sources_text = "\n".join(
        f"{i}. [{s['source']}:{s['ref']}] {s['label']} — {s['text'][:300]}"
        for i, s in enumerate(sources)
    )
    prompt = build_citation_judge_prompt()
    context = f"Claims:\n{claims_text}\n\nCandidate sources:\n{sources_text}"

    def _call():
        result = llm.chat_json(messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ])
        errors = validate_schema(result, CITATION_JUDGE_SCHEMA)
        if errors:
            raise ValueError(f"Invalid citation-judge output: {errors}")
        return result

    try:
        result = call_with_retry(_call)
    except Exception as e:
        print(f"   ⚠️  Citation LLM fallback failed after retries: {e}")
        return {}

    valid_refs = {s["ref"] for s in sources}
    by_ref = {s["ref"]: s for s in sources}

    matches = {}
    for claim in result.get("matches", []):
        claim_index = claim.get("claim_index")
        if not isinstance(claim_index, int) or not (0 <= claim_index < len(unmatched)):
            continue

        claim_text = unmatched[claim_index]
        claim_keywords = {
            kw.lower() for kw in title_keywords(claim_text)
            if kw.lower() not in _PROSE_STOPWORDS and kw.lower() not in common
        }
        claim_ids = _identifier_tokens(claim_text)

        grounded_sources = []
        dropped_ref = 0
        dropped_quote = 0
        dropped_irrelevant = 0
        for item in claim.get("evidence", []):
            if not isinstance(item, dict):
                continue
            ref = item.get("source_ref")
            quote = (item.get("quote") or "").strip()
            if ref not in valid_refs:
                dropped_ref += 1
                continue
            source = by_ref[ref]
            if not quote or quote.lower() not in source["text"].lower():
                dropped_quote += 1
                continue
            quote_keywords = {
                kw.lower() for kw in title_keywords(quote)
                if kw.lower() not in _PROSE_STOPWORDS and kw.lower() not in common
            }
            # If the claim names a specific numbered/lettered instance
            # ("Press #2"), a genuine quote must name the same one — two
            # different instances of the same category noun ("Press #2"
            # vs "Press #3") can otherwise share every ordinary keyword
            # and still be wrong. See module docstring.
            if claim_ids and not (claim_ids & _identifier_tokens(quote)):
                dropped_irrelevant += 1
                continue
            if not (quote_keywords & claim_keywords):
                dropped_irrelevant += 1
                continue
            grounded_sources.append(source)

        if dropped_ref or dropped_quote or dropped_irrelevant:
            print(
                f"   ⚠️  Dropping {dropped_ref} ungrounded ref(s), "
                f"{dropped_quote} unverifiable-quote, and "
                f"{dropped_irrelevant} irrelevant-quote match(es) for claim {claim_index}"
            )
        if grounded_sources:
            matches[claim_index] = grounded_sources

    return matches


def _format_refs(sources: list[dict]) -> str:
    return ", ".join(f"{s['source']}: {s['ref']}" for s in sources)


def cite_brief(markdown_text: str, sources: list[dict], llm=None) -> tuple[str, dict]:
    """Runs split_citable_lines, then keyword_match_sources per line. Lines
    with no keyword match are batched into one llm_match_sources call if
    llm is given (None means keyword-only). Appends ' _[source: X]_' for a
    keyword match or ' _[source: X (inferred)]_' for an LLM match — purely
    additive, never rewrites the original wording.

    Returns:
        (annotated_text, stats) where stats =
        {"cited_keyword": N, "cited_llm": M, "uncited": K}.
    """
    bullets = split_citable_lines(markdown_text)
    common_keywords = corpus_common_keywords(sources)

    keyword_matches: dict[str, list[dict]] = {}
    unmatched_bullets: list[str] = []
    for bullet in bullets:
        matched = keyword_match_sources(bullet, sources, common_keywords=common_keywords)
        if matched:
            keyword_matches[bullet] = matched
        else:
            unmatched_bullets.append(bullet)

    llm_matches: dict[str, list[dict]] = {}
    if llm is not None and unmatched_bullets:
        raw = llm_match_sources(llm, unmatched_bullets, sources, common_keywords=common_keywords)
        for idx, matched_sources in raw.items():
            llm_matches[unmatched_bullets[idx]] = matched_sources

    stats = {"cited_keyword": 0, "cited_llm": 0, "uncited": 0}
    out_lines = []
    for raw_line in markdown_text.split("\n"):
        stripped = raw_line.strip()
        if stripped in keyword_matches:
            out_lines.append(f"{raw_line} _[source: {_format_refs(keyword_matches[stripped])}]_")
            stats["cited_keyword"] += 1
        elif stripped in llm_matches:
            out_lines.append(f"{raw_line} _[source: {_format_refs(llm_matches[stripped])} (inferred)]_")
            stats["cited_llm"] += 1
        elif stripped in unmatched_bullets:
            out_lines.append(raw_line)
            stats["uncited"] += 1
        else:
            out_lines.append(raw_line)

    return "\n".join(out_lines), stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach per-bullet source citations to an already-generated daily brief",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m digest.core.citations --tenant demo-1 --provider deepseek --model deepseek-chat\n"
            "  python3 -m digest.core.citations --tenant demo-1 --keyword-only\n"
        ),
    )
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--keyword-only", action="store_true", help="Skip the LLM fallback pass entirely — free, but under-cites paraphrased claims")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = for_tenant(args.tenant)

    if not os.path.exists(paths.brief_file):
        raise SystemExit(f"No brief found at {paths.brief_file} — run the orchestrator for this tenant first.")

    with open(paths.brief_file, "r", encoding="utf-8") as f:
        brief_text = f.read()

    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    print(f"📎 Loaded {len(sources)} candidate source item(s) for tenant '{args.tenant}'.")

    llm = None
    if not args.keyword_only:
        llm = create_llm(provider=args.provider, model=args.model, temperature=0.0, tenant_id=args.tenant)

    annotated, stats = cite_brief(brief_text, sources, llm=llm)

    output_path = os.path.join(os.path.dirname(paths.brief_file), "daily_brief_cited.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(annotated)

    print(f"   {stats['cited_keyword']} cited by keyword, {stats['cited_llm']} cited by LLM fallback, {stats['uncited']} uncited.")
    print(f"   Wrote {output_path}")


if __name__ == "__main__":
    main()
