"""
Calendar Parser
================
Parses .ics calendar files into structured dicts using Python stdlib only.
Hand-rolled RFC5545 VEVENT parser — un-folds continuation lines, decodes
DESCRIPTION newline escapes, and extracts ATTENDEE participant info. Mirrors
email_parser.py's contract so calendar_agent.py can follow the same
MAP-REDUCE shape as triage_agent.py.

Usage:
    from digest.parsers.calendar_parser import parse_ics, load_calendar, group_by_date

    events = parse_ics("data/calendar/calendar.ics")
    events = load_calendar("data/calendar/calendar.ics")
    daily_batches = group_by_date(events)
"""

import os
from collections import defaultdict
from datetime import datetime


def _unfold_lines(raw: str) -> list[str]:
    """Un-fold RFC5545 continuation lines: a line starting with a space or
    tab is a continuation of the previous line, not a new property. Real
    ICS exports (Google Calendar, Outlook, etc.) wrap long lines this way
    even though our own generator doesn't emit folded lines.
    """
    lines = raw.replace("\r\n", "\n").split("\n")
    unfolded = []
    for line in lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _parse_dt(value: str) -> datetime:
    """Parse an ICS DATE-TIME value (YYYYMMDDTHHMMSS[Z]) into a datetime."""
    value = value.strip()
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def _parse_attendee(line: str) -> dict:
    """Parse an ATTENDEE line's CN/PARTSTAT parameters and mailto value.

    Example: ATTENDEE;CN=Jordan Liu;PARTSTAT=ACCEPTED:mailto:jordan@tessera.io
    """
    if ":" not in line:
        return {"name": "", "email": "", "partstat": "NEEDS-ACTION"}

    params_part, _, value = line.partition(":")
    params = {}
    for param in params_part.split(";")[1:]:
        if "=" in param:
            key, _, val = param.partition("=")
            params[key.upper()] = val

    email = value.strip()
    if email.lower().startswith("mailto:"):
        email = email[len("mailto:"):]

    return {
        "name": params.get("CN", "").strip() or email,
        "email": email,
        "partstat": params.get("PARTSTAT", "NEEDS-ACTION"),
    }


def _finalize_event(event: dict) -> dict:
    """Fill in defaults and derive date_key for a parsed event."""
    start = event.get("start")
    return {
        "uid": event.get("uid", ""),
        "summary": event.get("summary", ""),
        "status": event.get("status", "CONFIRMED"),
        "description": event.get("description", ""),
        "location": event.get("location", ""),
        "start": start,
        "end": event.get("end"),
        "date_key": start.date().isoformat() if start else "unknown",
        "attendees": event.get("attendees", []),
    }


def parse_ics(filepath: str) -> list[dict]:
    """Parse a single .ics file into a list of structured event dicts.

    Returns:
        List of dicts with keys: uid, summary, status, description, location,
        start (datetime|None), end (datetime|None), date_key ("YYYY-MM-DD" of
        start, or "unknown"), attendees (list of {"name","email","partstat"}).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    events = []
    current = None
    for line in _unfold_lines(raw):
        if line == "BEGIN:VEVENT":
            current = {"attendees": []}
            continue
        if line == "END:VEVENT":
            if current is not None:
                events.append(_finalize_event(current))
            current = None
            continue
        if current is None or ":" not in line:
            continue

        key_part, _, value = line.partition(":")
        key = key_part.split(";")[0].upper()

        if key == "UID":
            current["uid"] = value.strip()
        elif key == "SUMMARY":
            current["summary"] = value.strip()
        elif key == "STATUS":
            current["status"] = value.strip()
        elif key == "DESCRIPTION":
            current["description"] = value.replace("\\n", "\n").replace("\\,", ",").strip()
        elif key == "LOCATION":
            current["location"] = value.strip()
        elif key == "DTSTART":
            current["start"] = _parse_dt(value)
        elif key == "DTEND":
            current["end"] = _parse_dt(value)
        elif key == "ATTENDEE":
            current["attendees"].append(_parse_attendee(line))

    return events


def load_calendar(path: str) -> list[dict]:
    """Load and parse a calendar .ics file, sorted chronologically by start.

    Args:
        path: Path to the .ics file.

    Returns:
        List of parsed event dicts, sorted by start time.
    """
    if not os.path.exists(path):
        print(f"❌ Error: Calendar file '{path}' not found.")
        return []

    try:
        events = parse_ics(path)
    except Exception as e:
        print(f"⚠️  Failed to parse '{path}': {e}")
        return []

    events.sort(key=lambda e: e["start"] or datetime.min)
    return events


def group_by_date(events: list[dict]) -> dict:
    """Group parsed events by date (YYYY-MM-DD) for day-by-day MAP processing.

    Args:
        events: List of parsed event dicts (must have 'date_key' field).

    Returns:
        Dict mapping date strings to lists of events, sorted chronologically.
    """
    groups = defaultdict(list)
    for event in events:
        groups[event.get("date_key", "unknown")].append(event)
    return dict(sorted(groups.items()))
