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
    count_email_exclusive_notes,
    count_exclusive_emails,
    find_calendar_date_inconsistencies,
    has_real_deep_work_conflict,
    note_filename,
    render_calendar,
    render_email,
    render_note,
    render_tasks,
    verify_generated_tenant,
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

    def test_no_calendar_events_on_final_day_is_caught(self):
        # Real failure found live (ceo-tenant-2026): the model clustered
        # both calendar events on days[0] to satisfy the same-day-conflict
        # requirement, leaving nothing on days[2] — so the calendar ledger
        # ends up 2 days behind "today" by construction on every run.
        bad = dict(_VALID_SAMPLE)
        bad["days"] = [
            {"date": "2026-07-16", "emails": [], "calendar_events": [
                {"summary": "Deep Work Block", "start_time": "08:00", "end_time": "10:00", "description": "", "location": "", "attendees": [], "is_focus_block": True},
                {"summary": "Team Sync", "start_time": "09:00", "end_time": "09:30", "description": "", "location": "", "attendees": [], "is_focus_block": False},
            ]},
            {"date": "2026-07-17", "emails": [], "calendar_events": []},
            {"date": "2026-07-18", "emails": [], "calendar_events": []},
        ]
        errors = _validate_generation(bad, expected_days=["2026-07-16", "2026-07-17", "2026-07-18"])
        self.assertTrue(any("No calendar_events on the final day" in e for e in errors))

    def test_calendar_event_on_final_day_has_no_error(self):
        good = dict(_VALID_SAMPLE)
        good["days"] = [
            {"date": "2026-07-16", "emails": [], "calendar_events": [
                {"summary": "Deep Work Block", "start_time": "08:00", "end_time": "10:00", "description": "", "location": "", "attendees": [], "is_focus_block": True},
                {"summary": "Team Sync", "start_time": "09:00", "end_time": "09:30", "description": "", "location": "", "attendees": [], "is_focus_block": False},
            ]},
            {"date": "2026-07-17", "emails": [], "calendar_events": []},
            {"date": "2026-07-18", "emails": [], "calendar_events": [
                {"summary": "Board Call", "start_time": "14:00", "end_time": "15:00", "description": "", "location": "", "attendees": [], "is_focus_block": False},
            ]},
        ]
        errors = _validate_generation(good, expected_days=["2026-07-16", "2026-07-17", "2026-07-18"])
        self.assertFalse(any("No calendar_events on the final day" in e for e in errors))

    def test_no_calendar_events_at_all_does_not_trigger_final_day_check(self):
        # An empty-calendar generation is a different (already-covered)
        # problem; this check should only fire when there ARE calendar
        # events somewhere, just not on the final day.
        bad = dict(_VALID_SAMPLE)
        bad["days"] = [
            {"date": "2026-07-16", "emails": [], "calendar_events": []},
            {"date": "2026-07-17", "emails": [], "calendar_events": []},
        ]
        errors = _validate_generation(bad, expected_days=["2026-07-16", "2026-07-17"])
        self.assertFalse(any("No calendar_events on the final day" in e for e in errors))


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


class TestExclusiveEmailCheck(unittest.TestCase):
    TEST_TENANT = "generate-persona-exclusivity-test-tenant"

    def setUp(self):
        self.addCleanup(self._cleanup)
        self._cleanup()

    def _cleanup(self):
        paths = for_tenant(self.TEST_TENANT)
        shutil.rmtree(f"./data/tenants/{self.TEST_TENANT}", ignore_errors=True)
        shutil.rmtree(paths.output_dir, ignore_errors=True)

    def _sample_with_emails(self, emails, notes, tasks):
        sample = dict(_VALID_SAMPLE)
        sample["days"] = [{"date": "2026-07-16", "emails": emails, "calendar_events": _VALID_SAMPLE["days"][0]["calendar_events"]}]
        sample["notes"] = notes
        sample["tasks"] = tasks
        return sample

    def test_email_duplicated_in_note_is_not_exclusive(self):
        emails = [
            {"filename_hint": "a", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Widget spec review", "body": "Please review the widget spec for the new product launch.", "time": "09:00"},
            {"filename_hint": "b", "from_name": "Ana", "from_email": "ana@testco.example.com", "subject": "Payments cluster outage", "body": "We had a server outage on the payments cluster last night.", "time": "10:00"},
        ]
        notes = [{"date": "2026-07-16", "filename_hint": "outage-followup", "body": "# Outage follow-up\nPayments cluster outage handled, root cause found."}]
        tasks = _VALID_SAMPLE["tasks"]
        sample = self._sample_with_emails(emails, notes, tasks)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        exclusive, total = count_exclusive_emails(paths)

        self.assertEqual(total, 2)
        self.assertEqual(exclusive, 1)  # widget-spec email is exclusive; outage email is not

    def test_email_duplicated_in_task_is_not_exclusive(self):
        emails = [
            {"filename_hint": "a", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Client demo request", "body": "Client wants a demo scheduled for next Tuesday.", "time": "09:00"},
            {"filename_hint": "b", "from_name": "Ana", "from_email": "ana@testco.example.com", "subject": "Login bug", "body": "Need someone to fix the login bug before Friday.", "time": "10:00"},
        ]
        notes = _VALID_SAMPLE["notes"]
        tasks = [{"id": "T-9", "title": "Fix login bug before Friday deadline", "status": "todo", "priority": "P1"}]
        sample = self._sample_with_emails(emails, notes, tasks)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        exclusive, total = count_exclusive_emails(paths)

        self.assertEqual(total, 2)
        self.assertEqual(exclusive, 1)  # demo-request email is exclusive; login-bug email is not

    def test_verify_generated_tenant_min_required_is_ceil_division(self):
        # 4 emails, 1 exclusive -> min_required = ceil(4/2) = 2 -> not ok.
        emails = [
            {"filename_hint": f"e{i}", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Login bug", "body": "Need someone to fix the login bug before Friday.", "time": "09:00"}
            for i in range(3)
        ] + [
            {"filename_hint": "e4", "from_name": "Ana", "from_email": "ana@testco.example.com", "subject": "Widget spec review", "body": "Please review the widget spec for the new product launch.", "time": "10:00"},
        ]
        notes = _VALID_SAMPLE["notes"]
        tasks = [{"id": "T-9", "title": "Fix login bug before Friday deadline", "status": "todo", "priority": "P1"}]
        sample = self._sample_with_emails(emails, notes, tasks)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        result = verify_generated_tenant(paths, ["2026-07-16"])

        self.assertEqual(result["total"], 4)
        self.assertEqual(result["min_required"], 2)
        self.assertEqual(result["exclusive"], 1)
        self.assertFalse(result["exclusive_ok"])


class TestNotesEmailExclusiveCheck(unittest.TestCase):
    TEST_TENANT = "generate-persona-notes-exclusivity-test-tenant"

    def setUp(self):
        self.addCleanup(self._cleanup)
        self._cleanup()

    def _cleanup(self):
        paths = for_tenant(self.TEST_TENANT)
        shutil.rmtree(f"./data/tenants/{self.TEST_TENANT}", ignore_errors=True)
        shutil.rmtree(paths.output_dir, ignore_errors=True)

    def _sample_with(self, emails, notes):
        sample = dict(_VALID_SAMPLE)
        sample["days"] = [{"date": "2026-07-16", "emails": emails, "calendar_events": _VALID_SAMPLE["days"][0]["calendar_events"]}]
        sample["notes"] = notes
        sample["tasks"] = _VALID_SAMPLE["tasks"]
        return sample

    def test_note_duplicated_in_email_is_not_exclusive(self):
        emails = [
            {"filename_hint": "a", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Payments cluster outage", "body": "We had a server outage on the payments cluster last night.", "time": "09:00"},
        ]
        notes = [
            {"date": "2026-07-16", "filename_hint": "outage-followup", "body": "# Outage follow-up\nPayments cluster outage handled, root cause found."},
            {"date": "2026-07-16", "filename_hint": "standup-minutes", "body": "# Team standup minutes\nDiscussed Q3 roadmap priorities and headcount plan for design team."},
        ]
        sample = self._sample_with(emails, notes)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        exclusive, total = count_email_exclusive_notes(paths)

        self.assertEqual(total, 2)
        self.assertEqual(exclusive, 1)  # standup-minutes note is email-exclusive; outage note is not

    def test_note_referencing_only_calendar_meeting_is_still_exclusive(self):
        # A note tied to a calendar meeting (not to any email) is exactly
        # the intended "minutes" shape and must still count as exclusive —
        # this check only cares about email overlap, not calendar overlap.
        emails = [
            {"filename_hint": "a", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Unrelated topic", "body": "Completely unrelated content about shipping logistics.", "time": "09:00"},
        ]
        notes = [
            {"date": "2026-07-16", "filename_hint": "deep-work-minutes", "body": "# Deep Work Block notes\nDecided to postpone the propagation schedule review until soil tests come back."},
        ]
        sample = self._sample_with(emails, notes)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        exclusive, total = count_email_exclusive_notes(paths)

        self.assertEqual(total, 1)
        self.assertEqual(exclusive, 1)

    def test_verify_generated_tenant_includes_notes_exclusive_ok(self):
        emails = [
            {"filename_hint": "a", "from_name": "Bob", "from_email": "bob@testco.example.com", "subject": "Payments cluster outage", "body": "We had a server outage on the payments cluster last night.", "time": "09:00"},
        ]
        notes = [
            {"date": "2026-07-16", "filename_hint": "outage-followup", "body": "# Outage follow-up\nPayments cluster outage handled, root cause found."},
        ]
        sample = self._sample_with(emails, notes)

        paths = write_tenant_files(sample, self.TEST_TENANT)
        result = verify_generated_tenant(paths, ["2026-07-16"])

        self.assertEqual(result["notes_total"], 1)
        self.assertEqual(result["notes_exclusive"], 0)
        self.assertFalse(result["notes_exclusive_ok"])


class TestCalendarDateConsistencyCheck(unittest.TestCase):
    TEST_TENANT = "generate-persona-date-consistency-test-tenant"
    DAYS = ["2026-07-18", "2026-07-19", "2026-07-20"]  # Saturday, Sunday, Monday

    def setUp(self):
        self.addCleanup(self._cleanup)
        self._cleanup()

    def _cleanup(self):
        paths = for_tenant(self.TEST_TENANT)
        shutil.rmtree(f"./data/tenants/{self.TEST_TENANT}", ignore_errors=True)
        shutil.rmtree(paths.output_dir, ignore_errors=True)

    def _sample_with(self, event_day, note_body):
        event = {
            "summary": "Site Visit with Sarah", "start_time": "10:00", "end_time": "11:00",
            "description": "Walk the east parcel", "location": "", "attendees": [], "is_focus_block": False,
        }
        sample = dict(_VALID_SAMPLE)
        sample["days"] = [
            {"date": d, "emails": [], "calendar_events": [event] if d == event_day else []}
            for d in self.DAYS
        ]
        sample["notes"] = [{"date": self.DAYS[0], "filename_hint": "minutes", "body": note_body}]
        sample["tasks"] = _VALID_SAMPLE["tasks"]
        return sample

    def test_flags_event_dated_differently_than_a_note_describing_it(self):
        # Real bug found live: a "Site Visit with Sarah Chen" calendar
        # event placed on Saturday while the emails/notes describing that
        # same meeting (matching location/description) all said Monday.
        note = "# Meeting minutes\nSite visit with Sarah on Monday — we'll walk the east parcel together."
        sample = self._sample_with(event_day="2026-07-18", note_body=note)
        paths = write_tenant_files(sample, self.TEST_TENANT)

        inconsistencies = find_calendar_date_inconsistencies(paths, self.DAYS)

        self.assertEqual(len(inconsistencies), 1)
        self.assertEqual(inconsistencies[0]["event_date"], "2026-07-18")
        self.assertEqual(inconsistencies[0]["claimed_weekday"], "monday")
        self.assertEqual(inconsistencies[0]["claimed_date"], "2026-07-20")

    def test_no_inconsistency_when_note_agrees_with_the_event_date(self):
        note = "# Meeting minutes\nSite visit with Sarah on Saturday — we'll walk the east parcel together."
        sample = self._sample_with(event_day="2026-07-18", note_body=note)
        paths = write_tenant_files(sample, self.TEST_TENANT)

        inconsistencies = find_calendar_date_inconsistencies(paths, self.DAYS)

        self.assertEqual(inconsistencies, [])

    def test_no_inconsistency_when_no_source_mentions_a_weekday(self):
        note = "# Meeting minutes\nSite visit with Sarah — we'll walk the east parcel together."
        sample = self._sample_with(event_day="2026-07-18", note_body=note)
        paths = write_tenant_files(sample, self.TEST_TENANT)

        inconsistencies = find_calendar_date_inconsistencies(paths, self.DAYS)

        self.assertEqual(inconsistencies, [])

    def test_unrelated_note_mentioning_a_different_weekday_is_not_flagged(self):
        # The note doesn't share enough keywords with the event to be
        # considered "about" it at all, so its weekday mention shouldn't
        # be cross-checked against this event.
        note = "# Meeting minutes\nBudget review on Wednesday with finance."
        sample = self._sample_with(event_day="2026-07-18", note_body=note)
        paths = write_tenant_files(sample, self.TEST_TENANT)

        inconsistencies = find_calendar_date_inconsistencies(paths, self.DAYS)

        self.assertEqual(inconsistencies, [])

    def test_verify_generated_tenant_includes_date_consistent(self):
        note = "# Meeting minutes\nSite visit with Sarah on Monday — we'll walk the east parcel together."
        sample = self._sample_with(event_day="2026-07-18", note_body=note)
        paths = write_tenant_files(sample, self.TEST_TENANT)

        result = verify_generated_tenant(paths, self.DAYS)

        self.assertFalse(result["date_consistent"])
        self.assertEqual(len(result["date_inconsistencies"]), 1)


if __name__ == "__main__":
    unittest.main()
