"""
Unit tests for the --hot-input side of hybrid cold/hot digest ingestion in
triage_agent.py and calendar_agent.py.

Two layers tested, both deterministic (no LLM calls):
1. Parsing + tagging + merging: a small "hot" batch of new .eml/.ics data,
   loaded via the same load_inbox/load_calendar parsers already used for the
   cold path, correctly lands in its own day bucket after group_by_date and
   carries the internal "_hot" marker used to decide which ledger entries
   get tagged.
2. _map_single_day's hot-tagging: given hot=True, the returned ledger entry
   carries "hot": True; given hot=False (default), the key is absent
   entirely (mirrors the existing "compacted" key convention in ledger.py —
   present-when-true, not present-and-False).

Run: python3 -m unittest test_hot_input -v
"""

import os
import tempfile
import unittest

from calendar_parser import group_by_date as calendar_group_by_date
from calendar_parser import load_calendar
from calendar_agent import _map_single_day as calendar_map_single_day
from email_parser import group_by_date as email_group_by_date
from email_parser import load_inbox
from triage_agent import _map_single_day as email_map_single_day


COLD_EML = """\
Subject: Weekly sync notes
From: Grace Lin <grace@tessera.io>
To: Avery Chen <avery@tessera.io>
Date: Mon, 15 Jun 2026 09:00:00 -0700
Message-ID: <cold-1@tessera.io>

Nothing urgent this week.
"""

HOT_EML = """\
Subject: Just landed — need a decision today
From: Grace Lin <grace@tessera.io>
To: Avery Chen <avery@tessera.io>
Date: Mon, 20 Jul 2026 09:00:00 -0700
Message-ID: <hot-1@tessera.io>

This just came in.
"""

COLD_ICS = """\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:cold-event-0@tessera.io
DTSTAMP:20260615T090000Z
DTSTART:20260615T090000
DTEND:20260615T100000
SUMMARY:Cold sync
END:VEVENT
END:VCALENDAR
"""

HOT_ICS = """\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:hot-event-0@tessera.io
DTSTAMP:20260720T090000Z
DTSTART:20260720T090000
DTEND:20260720T100000
SUMMARY:Just-scheduled hot meeting
END:VEVENT
END:VCALENDAR
"""


class TestEmailHotMerge(unittest.TestCase):
    def setUp(self):
        self.cold_dir = tempfile.TemporaryDirectory()
        self.hot_dir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.cold_dir.name, "0001.eml"), "w") as f:
            f.write(COLD_EML)
        with open(os.path.join(self.hot_dir.name, "0001.eml"), "w") as f:
            f.write(HOT_EML)

    def tearDown(self):
        self.cold_dir.cleanup()
        self.hot_dir.cleanup()

    def test_hot_input_merges_into_its_own_day_bucket(self):
        cold_emails = load_inbox(self.cold_dir.name)
        hot_emails = load_inbox(self.hot_dir.name)
        for e in hot_emails:
            e["_hot"] = True

        merged = cold_emails + hot_emails
        daily_batches = email_group_by_date(merged)

        self.assertIn("2026-06-15", daily_batches)
        self.assertIn("2026-07-20", daily_batches)
        self.assertNotIn("_hot", daily_batches["2026-06-15"][0])
        self.assertTrue(daily_batches["2026-07-20"][0]["_hot"])

    def test_hot_days_detection_matches_only_tagged_days(self):
        cold_emails = load_inbox(self.cold_dir.name)
        hot_emails = load_inbox(self.hot_dir.name)
        for e in hot_emails:
            e["_hot"] = True
        daily_batches = email_group_by_date(cold_emails + hot_emails)

        hot_days = {day for day, batch in daily_batches.items() if any(e.get("_hot") for e in batch)}
        self.assertEqual(hot_days, {"2026-07-20"})


class TestCalendarHotMerge(unittest.TestCase):
    def setUp(self):
        self.cold_dir = tempfile.TemporaryDirectory()
        self.hot_dir = tempfile.TemporaryDirectory()
        self.cold_path = os.path.join(self.cold_dir.name, "cold.ics")
        self.hot_path = os.path.join(self.hot_dir.name, "hot.ics")
        with open(self.cold_path, "w") as f:
            f.write(COLD_ICS)
        with open(self.hot_path, "w") as f:
            f.write(HOT_ICS)

    def tearDown(self):
        self.cold_dir.cleanup()
        self.hot_dir.cleanup()

    def test_hot_input_merges_into_its_own_day_bucket(self):
        cold_events = load_calendar(self.cold_path)
        hot_events = load_calendar(self.hot_path)
        for e in hot_events:
            e["_hot"] = True

        daily_batches = calendar_group_by_date(cold_events + hot_events)

        self.assertIn("2026-06-15", daily_batches)
        self.assertIn("2026-07-20", daily_batches)
        self.assertTrue(daily_batches["2026-07-20"][0]["_hot"])


class _StubLLM:
    def chat_json(self, messages):
        return {"deadlines": [], "decisions": [], "action_items": [], "thread_progressions": []}


class _StubCalendarLLM:
    def chat_json(self, messages):
        return {"meetings_needing_prep": [], "family_calendar_items": [], "pattern_flags": [], "notable_events": []}


_EMAIL_MAP_SCHEMA = {"deadlines": [], "decisions": [], "action_items": [], "thread_progressions": []}


class TestMapSingleDayHotTag(unittest.TestCase):
    """Uses real parsed fixtures (not hand-rolled dicts) as the batch input,
    since format_email_batch/format_event_batch expect the full field set
    parse_eml/parse_ics produce — only the hot=True/False flag under test
    should vary, not the batch shape.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _one_email(self, content):
        path = os.path.join(self.tmp.name, "0001.eml")
        with open(path, "w") as f:
            f.write(content)
        return load_inbox(self.tmp.name)

    def _one_event(self, content):
        path = os.path.join(self.tmp.name, "cal.ics")
        with open(path, "w") as f:
            f.write(content)
        return load_calendar(path)

    def test_email_entry_tagged_hot_when_true(self):
        batch = self._one_email(HOT_EML)
        entry = email_map_single_day(_StubLLM(), "2026-07-20", batch, "sys prompt", _EMAIL_MAP_SCHEMA, hot=True)
        self.assertTrue(entry["hot"])

    def test_email_entry_not_tagged_when_false(self):
        batch = self._one_email(COLD_EML)
        entry = email_map_single_day(_StubLLM(), "2026-06-15", batch, "sys prompt", _EMAIL_MAP_SCHEMA)
        self.assertNotIn("hot", entry)

    def test_calendar_entry_tagged_hot_when_true(self):
        batch = self._one_event(HOT_ICS)
        entry = calendar_map_single_day(_StubCalendarLLM(), "2026-07-20", batch, "sys prompt", hot=True)
        self.assertTrue(entry["hot"])

    def test_calendar_entry_not_tagged_when_false(self):
        batch = self._one_event(COLD_ICS)
        entry = calendar_map_single_day(_StubCalendarLLM(), "2026-06-15", batch, "sys prompt")
        self.assertNotIn("hot", entry)


if __name__ == "__main__":
    unittest.main()
