"""
Notes Parser
=============
Parses markdown notes into a normalized, platform-agnostic dict shape.

There's no universal "notes format" the way MIME is for email or ICS is for
calendar — Notion, Confluence, Google Docs, and flat markdown files all have
different native schemas. This module is the markdown adapter: it produces
the same normalized shape (title, created_at, checklist_items, ...) that a
different adapter (e.g. a Notion API adapter) would populate from a
different source, so notes_agent.py doesn't need to know which platform the
notes came from.

`linked_entity_ids` is included but always empty here — richer platforms
(Notion relations, Confluence-Jira links) give structured cross-references
for free; flat markdown doesn't, so there's nothing to extract yet. It's a
placeholder for a richer adapter, not dead code: keeping the field in the
normalized shape now means the orchestrator's schema doesn't have to change
later when a richer source is added.

Usage:
    from digest.parsers.notes_parser import parse_note, load_notes

    note = parse_note("data/notes/2026-07-07-weekly-priorities.md")
    notes = load_notes("data/notes/")
"""

import os
import re

_FILENAME_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.md$")
_CHECKBOX_RE = re.compile(r"^\s*-\s*\[( |x|X)\]\s*(.+)$", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def parse_note(filepath: str) -> dict:
    """Parse a single markdown note into a normalized dict.

    Args:
        filepath: Path to the .md file. Expected filename convention is
            "YYYY-MM-DD-slug.md" (falls back to "unknown" date otherwise).

    Returns:
        Dict with keys: note_id, title, created_at, date_key, body,
        checklist_items (list of {"text", "done"}), linked_entity_ids.
    """
    filename = os.path.basename(filepath)
    match = _FILENAME_DATE_RE.match(filename)
    created_at = match.group(1) if match else "unknown"

    with open(filepath, "r", encoding="utf-8") as f:
        body = f.read()

    title_match = _TITLE_RE.search(body)
    title = title_match.group(1).strip() if title_match else filename

    checklist_items = [
        {"text": text.strip(), "done": marker.lower() == "x"}
        for marker, text in _CHECKBOX_RE.findall(body)
    ]

    return {
        "note_id": filename,
        "title": title,
        "created_at": created_at,
        "date_key": created_at,
        "body": body,
        "checklist_items": checklist_items,
        "linked_entity_ids": [],
    }


def load_notes(notes_dir: str) -> list[dict]:
    """Load and parse all markdown notes from a directory.

    Args:
        notes_dir: Path to directory containing .md files.

    Returns:
        List of parsed note dicts, sorted chronologically by created_at.
    """
    notes = []

    if not os.path.exists(notes_dir):
        print(f"❌ Error: Notes directory '{notes_dir}' not found.")
        return notes

    for filename in sorted(os.listdir(notes_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(notes_dir, filename)
        try:
            notes.append(parse_note(filepath))
        except Exception as e:
            print(f"⚠️  Failed to parse {filename}: {e}")

    notes.sort(key=lambda n: n["created_at"])
    return notes
