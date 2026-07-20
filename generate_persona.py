#!/usr/bin/env python3
"""
Persona Generator — LLM-Backed Small-Scale Review Tenants
=============================================================
Generates a fresh, fictional persona (persona.md + tenant_config.json) and
a small, hand-reviewable dataset (5-10 emails, 2-3 calendar events with a
genuine conflict, 2-3 notes, 2 tasks, across 3 consecutive days) for the
multi-tenant pipeline — a repeatable alternative to hand-authoring a fixed
demo tenant (like data/tenants/arclight/) every time someone wants a small
dataset to sanity-check the pipeline against before trusting it at real
30-day scale.

One LLM call produces structured JSON (see GENERATION_SCHEMA below) — never
raw .eml/.ics/.md text directly, which would be fragile to parse and
impossible to validate. Deterministic Python code (render_email,
render_calendar, etc.) turns that JSON into the actual files, reusing
generate_data.py's ics_event for VEVENT formatting rather than
reimplementing iCalendar output.

LLMs are unreliable at exact HH:MM overlap arithmetic, and the whole point
of this dataset is a genuine, verifiable calendar conflict — so this
doesn't just trust the model's claim that two events conflict. After
rendering, it reloads the calendar through the real parser
(calendar_parser.load_calendar) and runs the actual
calendar_agent.compute_day_stats deterministic check against it. If no day
shows a real deep_work_conflicts entry, it retries the generation once
before failing loudly — never silently ships a "conflict" that wouldn't
actually trigger the pipeline's own detection.

Usage:
    python3 generate_persona.py --tenant-id demo-1 --provider deepseek --model deepseek-chat
    python3 generate_persona.py --tenant-id demo-2 --hint "boutique law firm, managing partner"
"""

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta, timezone

from digest.agents.calendar_agent import compute_day_stats
from digest.core.citations import corpus_common_keywords, keyword_match_sources, load_citable_sources
from digest.core.ledger import validate_schema
from digest.core.llm import call_with_retry, create_llm
from digest.core.tenant_paths import for_tenant
from digest.parsers.calendar_parser import group_by_date, load_calendar
from generate_data import ics_event

GENERATION_SCHEMA = {
    "days": ["date", "emails", "calendar_events"],
    "notes": ["date", "filename_hint", "body"],
    "tasks": ["id", "title", "status", "priority"],
}

_TZ = timezone(timedelta(hours=-5))


# ─── Generation (LLM) ──────────────────────────────────────────────────────


_GENERATION_SHAPE = {
    "company_name": "...",
    "protagonist_name": "...",
    "protagonist_email": "...",
    "persona_markdown": "...(see PERSONA_MARKDOWN REQUIREMENTS below — do not copy this placeholder text into the value)",
    "tenant_config": {
        "use_persona_in_map": True,
        "never_draft_contacts": [{"name": "...", "email": "..."}],
        "map_noise_filter": {"blocked_senders": ["..."], "blocked_domains": []},
    },
    "days": [
        {
            "date": "YYYY-MM-DD",
            "emails": [
                {
                    "filename_hint": "...",
                    "from_name": "...",
                    "from_email": "...",
                    "subject": "...",
                    "body": "...",
                    "time": "HH:MM",
                }
            ],
            "calendar_events": [
                {
                    "summary": "...",
                    "start_time": "HH:MM",
                    "end_time": "HH:MM",
                    "description": "...",
                    "location": "...",
                    "attendees": ["..."],
                    "is_focus_block": True,
                }
            ],
        }
    ],
    "notes": [{"date": "YYYY-MM-DD", "filename_hint": "...", "body": "# markdown note content"}],
    "tasks": [
        {
            "id": "...",
            "title": "...",
            "description": "...",
            "status": "todo|in-progress|blocked|done",
            "priority": "P0-P4",
            "due_date": "YYYY-MM-DD or null",
            "tags": ["..."],
        }
    ],
    "answer_key_markdown": "# What a reviewer should expect to see at each review step (see requirements below)",
}


def default_days() -> list[str]:
    """3 consecutive real dates ending today, as ISO strings — injected into
    the prompt as facts rather than left for the model to invent. Same
    reasoning as orchestrator.py's TODAY'S DATE fact: models guess dates
    unreliably (a live run of this generator produced 2024 dates,
    2+ years stale relative to the real today, before this fix), so this
    computes them in code and the model just fills in content for them.
    """
    today = date.today()
    return [(today - timedelta(days=2 - i)).isoformat() for i in range(3)]


def build_generation_prompt(hint: str | None, days: list[str]) -> str:
    if hint:
        hint_line = f"Loose steer on flavor, don't follow it rigidly: {hint}\n\n"
    else:
        hint_line = (
            "No steer given — invent a fresh, specific company/industry/role "
            "yourself. Avoid generic 'tech startup' framing; pick something "
            "concrete and different from whatever you'd default to.\n\n"
        )

    days_line = (
        f"Use exactly these 3 dates, in this order, for days[0].date, "
        f"days[1].date, and days[2].date: {days[0]}, {days[1]}, {days[2]}. "
        "Do not invent different dates — these are real calendar dates the "
        "surrounding system will treat as recent, and using anything else "
        "will make the generated data look stale.\n\n"
    )

    requirements = [
        f"days[0].date = {days[0]}, days[1].date = {days[1]}, days[2].date = {days[2]} — exactly, as given above.",
        "5-10 emails total across all 3 days combined (not per day).",
        "2-3 calendar events total across all 3 days combined. Among them, at "
        "least one genuine conflict: one event with is_focus_block true whose "
        "summary contains the literal words Deep Work (a protected block), and "
        "a separate meeting the same day whose start_time/end_time range "
        "genuinely overlaps it — not adjacent, not close, an actual time "
        "overlap you have checked by hand. At least one calendar event must be "
        f"dated {days[2]} (days[2], the most recent day) — the pipeline treats "
        "the calendar's latest event day as its freshness signal, so a "
        "calendar with nothing scheduled on the final day will look "
        "artificially stale immediately after generation. The conflict pair "
        "may still be placed on any day; this just requires at least one "
        f"event (the conflict pair itself, or a third event) dated {days[2]}.",
        "2-3 notes total, each a short markdown note in the protagonist's own voice.",
        "At least one note must be written as the protagonist's own minutes "
        "or draft from one of the generated calendar meetings — naturally "
        "referencing that meeting (who was there, what was decided or "
        "discussed) — and must capture at least one detail, decision, or "
        "follow-up that is NOT mentioned in any email. Referencing the "
        "meeting is expected and correct here, that's what makes it "
        "minutes; the constraint is specifically that the content never "
        "got emailed to anyone, not that it's isolated from everything.",
        "Exactly 2 tasks.",
        "At least one email from a sender that should clearly be suppressed by "
        "the noise filter (a recruiter or vendor-sales-shaped cold pitch) — its "
        "sender email must appear in tenant_config.map_noise_filter.blocked_senders.",
        "At least half of the emails (rounded up) must cover a topic that "
        "appears nowhere else in the dataset — no note, calendar event, or "
        "task may reference it. This is the majority case, not the "
        "exception: most emails should stand entirely on their own.",
        "At least one, but no more than two, signals should show up "
        "independently in 2+ different sources (e.g. the same concern "
        "mentioned in both an email and a note, or an email and a task) — "
        "something a cross-referencing check could connect, not just "
        "extract once. When a fact does legitimately appear in more than "
        "one source, each mention must add something the others don't (a "
        "decision, a new detail, a next step) — never restate the same "
        "sentence in different words. A note that just re-narrates what an "
        "email already said is wrong; a note capturing what the "
        "protagonist decided to do about it, in their own voice, is right.",
        "persona_markdown's People section must give every named sender/attendee "
        "used anywhere in days/notes/tasks an explicit priority — nobody should "
        "appear in the data without being named in the persona.",
        "answer_key_markdown must have three sections (Step 1 / Step 2 / Step "
        "3) describing what should appear in the digest after each of days 1, "
        "1+2, and 1+2+3 are processed respectively — specific enough that a "
        "reviewer can check the real output against it (name the actual "
        "signal, not just 'the email should be extracted'). IMPORTANT: only "
        "EMAIL is revealed incrementally across these 3 steps — calendar "
        "events, notes, and tasks are all visible in full starting at Step "
        "1 (same as a real calendar shows next week's meetings today). So "
        "the calendar conflict, every note, and both tasks should appear "
        "identically in all three answer-key sections; only which emails "
        "are visible should change step to step. Do not write an answer key "
        "that describes the calendar conflict as a Step 2 or Step 3 reveal.",
        "If an email, note, or the persona mentions a specific meeting by "
        "name/attendee (e.g. 'our site visit Monday') and that same meeting "
        "also appears as a calendar_events entry, the calendar event's own "
        "day (the days[].date it's nested under) must be the actual date "
        "that weekday name falls on within the 3 generated days — never "
        "place a meeting's calendar event on one day while an email or note "
        "describing that same meeting names a different day for it.",
    ]
    requirements_text = "\n".join(f"- {r}" for r in requirements)

    persona_markdown_requirements = (
        "The persona_markdown VALUE must be the actual persona document itself "
        "— first-person, starting with a top-level # heading naming the "
        "protagonist, and NOT a description of what the document should "
        "contain. Use exactly these ## section headings, in this order: "
        "Who I am / What a great morning digest looks like to me / How I work "
        "/ People who matter / What I don't want surfaced / What I might miss "
        "without help / Tone for drafted replies / Things you should be "
        "honest about / What today means. Under 'People who matter', give "
        "every person a name, role, and an explicit P0-P4 priority with a "
        "one-line reason."
    )

    return (
        "You are generating a small, fictional test dataset for a daily-digest "
        "tool — a persona profile plus a realistic 3-day slice of their email, "
        "calendar, notes, and tasks. This is used by a human reviewer to sanity-"
        "check the tool at a small, fully-readable scale before trusting it on "
        "a real 30-day inbox, so realism and internal consistency matter more "
        "than volume.\n\n"
        f"{hint_line}"
        f"{days_line}"
        "## Required output shape (strict JSON, no markdown fences, no extra text)\n"
        "The example values below (e.g. \"...\") are placeholders showing "
        "shape only — never copy them literally into your output.\n"
        f"{json.dumps(_GENERATION_SHAPE, indent=2)}\n\n"
        "## PERSONA_MARKDOWN REQUIREMENTS\n"
        f"{persona_markdown_requirements}\n\n"
        "## Hard requirements\n"
        f"{requirements_text}"
    )


def _validate_generation(data: dict, expected_days: list[str] | None = None) -> list[str]:
    errors = validate_schema(data, GENERATION_SCHEMA)
    for key in ("persona_markdown", "answer_key_markdown", "company_name", "protagonist_name", "protagonist_email"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            errors.append(f"Missing or empty top-level string key: '{key}'")
    if not isinstance(data.get("tenant_config"), dict):
        errors.append("Missing or invalid top-level dict key: 'tenant_config'")

    if expected_days is not None and isinstance(data.get("days"), list):
        actual_days = [d.get("date") for d in data["days"] if isinstance(d, dict)]
        if actual_days != expected_days:
            errors.append(f"days[].date must be exactly {expected_days} in order, got {actual_days}")

        calendar_days = {
            d.get("date") for d in data["days"]
            if isinstance(d, dict) and d.get("calendar_events")
        }
        if calendar_days and expected_days[-1] not in calendar_days:
            errors.append(
                f"No calendar_events on the final day ({expected_days[-1]}) — "
                f"got calendar events on: {sorted(calendar_days)}"
            )

    persona_markdown = data.get("persona_markdown", "")
    if isinstance(persona_markdown, str) and persona_markdown.strip():
        # Regression guard: an earlier prompt version had the model echo the
        # JSON shape's placeholder description verbatim as persona_markdown's
        # first line instead of writing the actual document — catch that
        # failure mode structurally rather than trusting prose instructions
        # alone to prevent it.
        if "## Who I am" not in persona_markdown:
            errors.append("persona_markdown is missing the required '## Who I am' section — looks malformed")
        first_line = persona_markdown.strip().splitlines()[0]
        if "placeholder" in first_line.lower() or "section structure" in first_line.lower():
            errors.append(f"persona_markdown's first line looks like echoed prompt instructions, not real content: {first_line!r}")

    return errors


def generate_persona_data(llm, hint: str | None = None, days: list[str] | None = None) -> dict:
    """One LLM call producing the full generation JSON, validated against
    GENERATION_SCHEMA (plus the persona_markdown and date-fidelity checks
    in _validate_generation). Retried via call_with_retry on invalid output.

    Args:
        days: 3 ISO date strings to anchor the generated data to. Defaults
            to default_days() (3 real consecutive dates ending today) — a
            caller only needs to override this for testing.
    """
    days = days or default_days()
    prompt = build_generation_prompt(hint, days)

    def _call():
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the persona and dataset now."},
            ]
        )
        errors = _validate_generation(result, expected_days=days)
        if errors:
            raise ValueError(f"Invalid persona generation output: {errors}")
        return result

    return call_with_retry(_call)


# ─── Rendering (pure Python, no LLM) ───────────────────────────────────────


def _day_time(date: str, time_str: str | None, default: str) -> datetime:
    year, month, day = (int(x) for x in date.split("-"))
    hh, mm = (int(x) for x in (time_str or default).split(":"))
    return datetime(year, month, day, hh, mm, tzinfo=_TZ)


def render_email(email: dict, date: str, protagonist_name: str, protagonist_email: str, msg_id: str) -> str:
    """Render one email dict into .eml text — same Subject/From/To/Date/
    Message-ID header shape data/tenants/arclight/'s emails already use.
    """
    from email.utils import format_datetime

    dt = _day_time(date, email.get("time"), "09:00")
    from_name = email.get("from_name", "Unknown Sender")
    from_email = email.get("from_email", "unknown@example.com")
    subject = email.get("subject", "(no subject)")
    body = email.get("body", "").strip()
    return (
        f"Subject: {subject}\n"
        f"From: {from_name} <{from_email}>\n"
        f"To: {protagonist_name} <{protagonist_email}>\n"
        f"Date: {format_datetime(dt)}\n"
        f"Message-ID: {msg_id}\n"
        "\n"
        f"{body}\n"
    )


def render_calendar(days: list[dict], company_slug: str) -> str:
    """Render every day's calendar_events into one .ics file, reusing
    generate_data.ics_event for VEVENT formatting.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Generated Persona//Daily Digest Generator//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Generated Persona's Calendar",
        "X-WR-TIMEZONE:America/Chicago",
    ]
    for day in days:
        date = day["date"]
        for i, event in enumerate(day.get("calendar_events", [])):
            start = _day_time(date, event.get("start_time"), "09:00")
            end = _day_time(date, event.get("end_time"), "09:30")
            attendees = [
                {"name": name, "email": f"{re.sub(r'[^a-z]', '', name.lower())}@{company_slug}.example.com"}
                for name in event.get("attendees", []) or []
            ] or None
            lines.append(
                ics_event(
                    uid=f"generated-{date}-{i}@{company_slug}.example.com",
                    summary=event.get("summary", "Meeting"),
                    dtstart=start,
                    dtend=end,
                    description=event.get("description", ""),
                    location=event.get("location", ""),
                    attendees=attendees,
                )
            )
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


def note_filename(note: dict) -> str:
    date = note.get("date", "2026-01-01")
    hint = re.sub(r"[^a-z0-9-]", "", note.get("filename_hint", "note").strip().lower().replace(" ", "-")) or "note"
    return f"{date}-{hint}.md"


def render_note(note: dict) -> str:
    return note.get("body", "").strip() + "\n"


def render_tasks(tasks: list[dict]) -> str:
    return json.dumps(tasks, indent=2)


def write_tenant_files(data: dict, tenant_id: str) -> "TenantPaths":  # noqa: F821 (type hint only)
    """Renders and writes every file for one generated tenant, matching
    tenant_paths.for_tenant's exact expected layout.
    """
    paths = for_tenant(tenant_id)  # raises ValueError on an invalid tenant_id, before touching disk
    company_slug = re.sub(r"[^a-z0-9]", "", data["company_name"].lower()) or "example"

    os.makedirs(paths.inbox_dir, exist_ok=True)
    os.makedirs(os.path.dirname(paths.calendar_file), exist_ok=True)
    os.makedirs(paths.notes_dir, exist_ok=True)
    os.makedirs(os.path.dirname(paths.tasks_file), exist_ok=True)

    msg_index = 0
    for day in data["days"]:
        for email in day.get("emails", []):
            msg_index += 1
            msg_id = f"<generated-{day['date']}-{msg_index:04d}@{company_slug}.example.com>"
            filename = os.path.join(paths.inbox_dir, f"{msg_index:04d}.eml")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(render_email(email, day["date"], data["protagonist_name"], data["protagonist_email"], msg_id))

    with open(paths.calendar_file, "w", encoding="utf-8") as f:
        f.write(render_calendar(data["days"], company_slug))

    for note in data.get("notes", []):
        with open(os.path.join(paths.notes_dir, note_filename(note)), "w", encoding="utf-8") as f:
            f.write(render_note(note))

    with open(paths.tasks_file, "w", encoding="utf-8") as f:
        f.write(render_tasks(data.get("tasks", [])))

    with open(paths.persona_file, "w", encoding="utf-8") as f:
        f.write(data["persona_markdown"].strip() + "\n")

    with open(paths.tenant_config_file, "w", encoding="utf-8") as f:
        json.dump(data["tenant_config"], f, indent=2)

    answer_key_path = os.path.join(os.path.dirname(paths.persona_file), "ANSWER_KEY.md")
    with open(answer_key_path, "w", encoding="utf-8") as f:
        f.write(data["answer_key_markdown"].strip() + "\n")

    return paths


# ─── Conflict verification ─────────────────────────────────────────────────


def has_real_deep_work_conflict(calendar_file: str) -> bool:
    """Reloads the rendered .ics through the real parser and runs the real
    deterministic conflict check (calendar_agent.compute_day_stats) against
    it — verifies the LLM's claimed conflict actually triggers the
    pipeline's own detection, rather than trusting its self-reported
    is_focus_block/time-overlap claims.
    """
    events = load_calendar(calendar_file)
    by_day = group_by_date(events)
    return any(compute_day_stats(day_events)["deep_work_conflicts"] for day_events in by_day.values())


def count_exclusive_emails(paths) -> tuple[int, int]:
    """Reloads the rendered tenant through citations.load_citable_sources —
    the same leaf-extracted, per-source text every citation match already
    uses — and, for each email, checks whether keyword_match_sources finds
    it anywhere else in the dataset (notes/calendar/tasks). Reuses the
    exact matching logic and corpus-frequency filtering already fixed and
    verified for citations.py's own precision bugs; this just checks for
    the *absence* of a match instead of its presence.

    Found live, not hypothetical: two separate generations both produced
    datasets where nearly every email's content was near-verbatim restated
    in a note, calendar event, and/or task — this is the deterministic
    backstop for the prompt's "most emails should stand on their own"
    requirement, same "don't just ask nicely, verify it" posture as
    has_real_deep_work_conflict above.

    Returns:
        (exclusive_count, total_email_count).
    """
    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    emails = [s for s in sources if s["source"] == "email"]
    others = [s for s in sources if s["source"] != "email"]
    common = corpus_common_keywords(others)
    exclusive = sum(1 for e in emails if not keyword_match_sources(e["text"], others, common_keywords=common))
    return exclusive, len(emails)


def count_email_exclusive_notes(paths) -> tuple[int, int]:
    """Same matching primitives as count_exclusive_emails, checked in the
    opposite direction: for each note, does its content overlap with any
    EMAIL specifically — not calendar, not tasks. A note referencing its
    own meeting (i.e. also overlapping a calendar event) is expected and
    correct, that's what makes it "minutes"; only email-duplication is the
    problem this checks for.

    Found live: every note in two separate generations had its action
    items also duplicated into a matching task, so the case notes exist
    for in the first place — meeting-only content never emailed to
    anyone — had never actually been exercised.

    Returns:
        (email_exclusive_count, total_note_count).
    """
    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    notes = [s for s in sources if s["source"] == "notes"]
    emails = [s for s in sources if s["source"] == "email"]
    common = corpus_common_keywords(emails)
    exclusive = sum(1 for n in notes if not keyword_match_sources(n["text"], emails, common_keywords=common))
    return exclusive, len(notes)


_WEEKDAY_RE = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)


def _weekday_dates(days: list[str]) -> dict[str, str]:
    """Maps each generated day's actual weekday name (lowercase) to its ISO
    date, e.g. {"saturday": "2026-07-18", "sunday": "2026-07-19",
    "monday": "2026-07-20"} — the 3-day window is short enough that a
    weekday name inside it names exactly one unambiguous date, so this
    needs no date-parsing library, just this lookup built from `days`.
    """
    return {datetime.strptime(d, "%Y-%m-%d").strftime("%A").lower(): d for d in days}


def find_calendar_date_inconsistencies(paths, days: list[str]) -> list[dict]:
    """Cross-checks each calendar event's own placement date against any
    weekday named in an email/note that's actually discussing that same
    event — reuses citations.keyword_match_sources (the same "is this text
    about the same thing" matching already tuned and tested for citations)
    to decide which sources are describing a given event at all, then
    checks whether any of those sources names a weekday that resolves to a
    different date than the event's own.

    Found live, not hypothetical: a "Site Visit with Sarah Chen" calendar
    event was placed on Saturday, while the emails and note describing
    that exact meeting (same location, same "walk the east parcel"
    description) all said Monday — has_real_deep_work_conflict only checks
    that *a* conflict exists, not that the conflicting event's own date is
    the one the rest of the dataset actually agrees on, so this had never
    been caught before.

    Returns:
        A list of {event_summary, event_date, source_ref, claimed_weekday,
        claimed_date} dicts, one per inconsistency found. Empty if none.
    """
    weekday_dates = _weekday_dates(days)
    events = load_calendar(paths.calendar_file)
    sources = load_citable_sources(paths.inbox_dir, paths.calendar_file, paths.notes_dir, paths.tasks_file)
    narrative_sources = [s for s in sources if s["source"] in ("email", "notes")]
    common = corpus_common_keywords(narrative_sources)

    inconsistencies = []
    for event in events:
        if not event.get("start"):
            continue
        event_date = event["start"].date().isoformat()
        event_text = f"{event.get('summary', '')} {event.get('description', '')}"
        for source in keyword_match_sources(event_text, narrative_sources, common_keywords=common):
            for weekday_match in _WEEKDAY_RE.finditer(source["text"]):
                weekday = weekday_match.group(1).lower()
                claimed_date = weekday_dates.get(weekday)
                if claimed_date and claimed_date != event_date:
                    inconsistencies.append({
                        "event_summary": event.get("summary", ""),
                        "event_date": event_date,
                        "source_ref": source["ref"],
                        "claimed_weekday": weekday,
                        "claimed_date": claimed_date,
                    })
    return inconsistencies


def verify_generated_tenant(paths, days: list[str]) -> dict:
    """Combines the conflict check, the email-exclusivity check, the
    notes-email-exclusivity check, and the calendar-date-consistency check
    into one result — main() retries generation at most once total if any
    check fails, rather than running independent retry cycles per check.
    """
    exclusive, total = count_exclusive_emails(paths)
    min_required = -(-total // 2)  # ceil(total / 2)
    notes_exclusive, notes_total = count_email_exclusive_notes(paths)
    date_inconsistencies = find_calendar_date_inconsistencies(paths, days)
    return {
        "has_conflict": has_real_deep_work_conflict(paths.calendar_file),
        "exclusive": exclusive,
        "total": total,
        "min_required": min_required,
        "exclusive_ok": exclusive >= min_required,
        "notes_exclusive": notes_exclusive,
        "notes_total": notes_total,
        "notes_exclusive_ok": notes_exclusive >= 1,
        "date_inconsistencies": date_inconsistencies,
        "date_consistent": len(date_inconsistencies) == 0,
    }


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a fresh fictional persona + small reviewable dataset for a new tenant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 generate_persona.py --tenant-id demo-1 --provider deepseek --model deepseek-chat\n"
            "  python3 generate_persona.py --tenant-id demo-2 --hint \"boutique law firm, managing partner\"\n"
        ),
    )
    parser.add_argument("--tenant-id", required=True, help="New tenant id — written to data/tenants/<id>/")
    parser.add_argument("--hint", default=None, help="Optional freeform steer on industry/role (omit for a fully random persona)")
    parser.add_argument("--provider", default="deepseek", choices=["ollama", "anthropic", "google", "openrouter", "deepseek"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.8, help="Nonzero by default — this generates varied fictional content, not a factual extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    llm = create_llm(provider=args.provider, model=args.model, temperature=args.temperature, tenant_id=args.tenant_id)

    print(f"🎭 Generating persona for tenant '{args.tenant_id}' ({args.provider}/{args.model})...")
    data = generate_persona_data(llm, args.hint)
    print(f"   Company: {data['company_name']} — protagonist: {data['protagonist_name']}")

    paths = write_tenant_files(data, args.tenant_id)
    print(f"   Wrote persona + dataset to data/tenants/{args.tenant_id}/")

    days = [d["date"] for d in data["days"]]
    result = verify_generated_tenant(paths, days)
    if not result["has_conflict"] or not result["exclusive_ok"] or not result["notes_exclusive_ok"] or not result["date_consistent"]:
        problems = []
        if not result["has_conflict"]:
            problems.append("no genuine deep-work conflict")
        if not result["exclusive_ok"]:
            problems.append(
                f"only {result['exclusive']}/{result['total']} emails have "
                f"exclusive content (need {result['min_required']})"
            )
        if not result["notes_exclusive_ok"]:
            problems.append("no note has content that isn't also in an email")
        if not result["date_consistent"]:
            problems.append(f"{len(result['date_inconsistencies'])} calendar-date inconsistency(ies)")
        print(f"   ⚠️  {'; '.join(problems)} — regenerating once...")
        data = generate_persona_data(llm, args.hint)
        paths = write_tenant_files(data, args.tenant_id)
        days = [d["date"] for d in data["days"]]
        result = verify_generated_tenant(paths, days)

    if result["has_conflict"]:
        print("   ✅ Verified: a genuine deep-work-block conflict is present.")
    else:
        print(
            "   ❌ Still no genuine conflict after one retry. The generated "
            "tenant is otherwise usable, but doesn't demonstrate the "
            "deep-work-conflict check — inspect data/tenants/"
            f"{args.tenant_id}/calendar/calendar.ics by hand or re-run this "
            "command."
        )

    if result["exclusive_ok"]:
        print(f"   ✅ Verified: {result['exclusive']}/{result['total']} emails have exclusive content.")
    else:
        print(
            f"   ❌ Still only {result['exclusive']}/{result['total']} emails "
            f"have exclusive content after one retry (wanted "
            f"{result['min_required']}). The generated tenant is otherwise "
            "usable, but has more cross-source duplication than intended — "
            "inspect by hand or re-run this command."
        )

    if result["notes_exclusive_ok"]:
        print(f"   ✅ Verified: {result['notes_exclusive']}/{result['notes_total']} notes have content not in any email.")
    else:
        print(
            f"   ❌ Still no note has content that isn't also in an email, "
            "after one retry. The generated tenant is otherwise usable, but "
            "doesn't exercise genuine meeting-only content — inspect by "
            "hand or re-run this command."
        )

    if result["date_consistent"]:
        print("   ✅ Verified: calendar event dates agree with what emails/notes say about them.")
    else:
        for inc in result["date_inconsistencies"]:
            print(
                f"   ❌ '{inc['event_summary']}' is on the calendar under "
                f"{inc['event_date']}, but {inc['source_ref']} calls it "
                f"'{inc['claimed_weekday'].capitalize()}' ({inc['claimed_date']}), "
                "after one retry. The generated tenant is otherwise usable, "
                "but this meeting's date disagrees across sources — inspect "
                "by hand or re-run this command."
            )

    print(f"\n   Answer key: data/tenants/{args.tenant_id}/ANSWER_KEY.md")
    print(f"   Review with: ./scripts/run_review_step.sh {args.tenant_id} 1")


if __name__ == "__main__":
    main()
