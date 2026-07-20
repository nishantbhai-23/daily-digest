"""
Unit tests for generate_persona.py — the LLM-backed small-scale persona
generator. Covers the pure, LLM-free pieces: rendering (JSON -> .eml/.ics/
.md/.json text), schema validation, and the deep-work-conflict verification
that runs the real calendar_agent.compute_day_stats check against
generated data.

write_tenant_files/has_real_deep_work_conflict tests write to a real
data/tenants/<test-id>/ directory (tenant_paths.for_tenant has no
injectable root), cleaned up via addCleanup.

Run: python3 -m unittest test_generate_persona -v
"""

import shutil
import unittest

from digest.core.tenant_paths import for_tenant
from generate_persona import (
    _validate_generation,
    has_real_deep_work_conflict,
    note_filename,
    render_calendar,
    render_email,
    render_note,
    render_tasks,
    write_tenant_files,
)

_VALID_SAMPLE = {
    "company_name": "TestCo",
    "protagonist_name": "Jamie Rivera",
    "protagonist_email": "jamie@testco.example.com",
    "persona_markdown": "# Jamie Rivera — Profile\n\n## Who I am\nI run things.\n",
    "tenant_config": {
        "use_persona_in_map": True,
        "never_draft_contacts": [],
        "map_noise_filter": {"blocked_senders": ["spammer@vendor.com"], "blocked_domains": []},
    },
    "days": [
        {
            "date": "2026-07-16",
            "emails": [
                {"filename_hint": "greeting", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Hi", "body": "Hello there", "time": "09:15"},
            ],
            "calendar_events": [
                {"summary": "🔒 Deep Work Block", "start_time": "08:00", "end_time": "10:00", "description": "focus", "location": "", "attendees": [], "is_focus_block": True},
                {"summary": "Team Sync", "start_time": "09:00", "end_time": "09:30", "description": "sync", "location": "", "attendees": ["Bob"], "is_focus_block": False},
            ],
        },
    ],
    "notes": [{"date": "2026-07-16", "filename_hint": "weekly notes", "body": "# Notes\nStuff."}],
    "tasks": [{"id": "T-1", "title": "Do a thing", "status": "todo", "priority": "P1"}],
    "answer_key_markdown": "# Answer Key\n## Step 1\nStuff.",
}


class TestValidateGeneration(unittest.TestCase):
    def test_valid_sample_has_no_errors(self):
        self.assertEqual(_validate_generation(_VALID_SAMPLE), [])

    def test_matching_expected_days_has_no_error(self):
        errors = _validate_generation(_VALID_SAMPLE, expected_days=["2026-07-16"])
        self.assertEqual(errors, [])

    def test_wrong_dates_are_caught(self):
        # Real failure found live: without an anchor, the model invented
        # dates 2+ years stale relative to the real today.
        errors = _validate_generation(_VALID_SAMPLE, expected_days=["2024-06-10"])
        self.assertTrue(any("days[].date" in e for e in errors))

    def test_missing_days_key_is_caught(self):
        bad = {k: v for k, v in _VALID_SAMPLE.items() if k != "days"}
        errors = _validate_generation(bad)
        self.assertTrue(any("days" in e for e in errors))

    def test_missing_persona_markdown_is_caught(self):
        bad = dict(_VALID_SAMPLE)
        bad["persona_markdown"] = ""
        errors = _validate_generation(bad)
        self.assertTrue(any("persona_markdown" in e for e in errors))

    def test_missing_tenant_config_is_caught(self):
        bad = dict(_VALID_SAMPLE)
        bad["tenant_config"] = "not a dict"
        errors = _validate_generation(bad)
        self.assertTrue(any("tenant_config" in e for e in errors))

    def test_echoed_schema_instructions_are_caught(self):
        # Real failure found live: the model echoed the JSON shape's
        # placeholder description verbatim as persona_markdown's first line
        # instead of writing the actual document.
        bad = dict(_VALID_SAMPLE)
        bad["persona_markdown"] = (
            "# Full persona.md content, first person, following this exact "
            "section structure: ## Who I am / ...\n\n## Who I am\nI run things.\n"
        )
        errors = _validate_generation(bad)
        self.assertTrue(any("echoed prompt instructions" in e for e in errors))

    def test_persona_markdown_missing_who_i_am_section_is_caught(self):
        bad = dict(_VALID_SAMPLE)
        bad["persona_markdown"] = "# Jamie Rivera\n\n## Some other section\nStuff.\n"
        errors = _validate_generation(bad)
        self.assertTrue(any("Who I am" in e for e in errors))

    def test_task_missing_priority_is_caught(self):
        bad = dict(_VALID_SAMPLE)
        bad["tasks"] = [{"id": "T-1", "title": "Do a thing", "status": "todo"}]
        errors = _validate_generation(bad)
        self.assertTrue(any("priority" in e for e in errors))


class TestRenderEmail(unittest.TestCase):
    def test_produces_expected_headers_and_body(self):
        email = _VALID_SAMPLE["days"][0]["emails"][0]
        text = render_email(email, "2026-07-16", "Jamie Rivera", "jamie@testco.example.com", "<msg-1@testco.example.com>")
        self.assertIn("Subject: Hi", text)
        self.assertIn("From: Bob <bob@testco.example.com>", text)
        self.assertIn("To: Jamie Rivera <jamie@testco.example.com>", text)
        self.assertIn("Message-ID: <msg-1@testco.example.com>", text)
        self.assertIn("Hello there", text)
        self.assertIn("2026", text)


class TestRenderCalendar(unittest.TestCase):
    def test_contains_both_events(self):
        ics = render_calendar(_VALID_SAMPLE["days"], "testco")
        self.assertIn("BEGIN:VCALENDAR", ics)
        self.assertIn("END:VCALENDAR", ics)
        self.assertEqual(ics.count("BEGIN:VEVENT"), 2)
        self.assertIn("Deep Work Block", ics)
        self.assertIn("Team Sync", ics)

    def test_uses_correct_day_and_times(self):
        ics = render_calendar(_VALID_SAMPLE["days"], "testco")
        self.assertIn("DTSTART:20260716T080000", ics)
        self.assertIn("DTSTART:20260716T090000", ics)


class TestRenderNoteAndTasks(unittest.TestCase):
    def test_note_filename_uses_date_and_slug(self):
        note = {"date": "2026-07-16", "filename_hint": "Weekly Notes!"}
        self.assertEqual(note_filename(note), "2026-07-16-weekly-notes.md")

    def test_render_note_returns_body(self):
        note = {"body": "# Notes\nStuff."}
        self.assertIn("Stuff.", render_note(note))

    def test_render_tasks_is_valid_json_list(self):
        import json
        text = render_tasks(_VALID_SAMPLE["tasks"])
        parsed = json.loads(text)
        self.assertEqual(parsed[0]["id"], "T-1")


class TestWriteTenantFilesAndConflictCheck(unittest.TestCase):
    TEST_TENANT = "generate-persona-test-tenant"

    def setUp(self):
        self.addCleanup(self._cleanup)
        self._cleanup()  # in case a prior failed run left files behind

    def _cleanup(self):
        paths = for_tenant(self.TEST_TENANT)
        shutil.rmtree(f"./data/tenants/{self.TEST_TENANT}", ignore_errors=True)
        shutil.rmtree(paths.output_dir, ignore_errors=True)

    def test_write_tenant_files_and_detects_real_conflict(self):
        paths = write_tenant_files(_VALID_SAMPLE, self.TEST_TENANT)
        self.assertTrue(has_real_deep_work_conflict(paths.calendar_file))

    def test_no_conflict_when_events_dont_overlap(self):
        no_conflict_sample = dict(_VALID_SAMPLE)
        no_conflict_sample["days"] = [
            {
                "date": "2026-07-16",
                "emails": [],
                "calendar_events": [
                    {"summary": "🔒 Deep Work Block", "start_time": "08:00", "end_time": "10:00", "description": "", "location": "", "attendees": [], "is_focus_block": True},
                    {"summary": "Team Sync", "start_time": "11:00", "end_time": "11:30", "description": "", "location": "", "attendees": [], "is_focus_block": False},
                ],
            },
        ]
        paths = write_tenant_files(no_conflict_sample, self.TEST_TENANT)
        self.assertFalse(has_real_deep_work_conflict(paths.calendar_file))

    def test_writes_answer_key_file(self):
        import os
        paths = write_tenant_files(_VALID_SAMPLE, self.TEST_TENANT)
        answer_key_path = os.path.join(os.path.dirname(paths.persona_file), "ANSWER_KEY.md")
        self.assertTrue(os.path.exists(answer_key_path))
        with open(answer_key_path) as f:
            self.assertIn("Answer Key", f.read())


if __name__ == "__main__":
    unittest.main()
