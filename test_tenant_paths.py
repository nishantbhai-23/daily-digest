"""
Unit tests for tenant_paths.for_tenant — the "default" tenant must resolve
to today's exact existing paths (zero behavior change for the no-`--tenant`-
flag case), and any other tenant_id must resolve under its own isolated
data/tenants/<id>/ and output/tenants/<id>/ tree with no overlap with the
default tenant or with each other.

No I/O — for_tenant only builds strings, doesn't touch the filesystem.

Run: python3 -m unittest test_tenant_paths -v
"""

import unittest

import calendar_agent
import notes_agent
import orchestrator
import triage_agent
from persona import PERSONA_FILE
from tenant_config import CONFIG_FILE
from tenant_paths import DEFAULT_TENANT, for_tenant


class TestDefaultTenantMatchesExistingConstants(unittest.TestCase):
    """The most important regression check in this whole module: running
    any script with no --tenant flag at all must be byte-for-byte identical
    to today's hardcoded paths — this is the guarantee that let every other
    script's --tenant wiring ship with zero behavior change by default.
    """

    def setUp(self):
        self.paths = for_tenant()

    def test_default_tenant_id(self):
        self.assertEqual(self.paths.tenant_id, DEFAULT_TENANT)

    def test_matches_triage_agent_constants(self):
        self.assertEqual(self.paths.inbox_dir, triage_agent.INBOX_DIR)
        self.assertEqual(self.paths.email_ledger_file, triage_agent.LEDGER_FILE)
        self.assertEqual(self.paths.email_summary_file, triage_agent.SUMMARY_FILE)

    def test_matches_calendar_agent_constants(self):
        self.assertEqual(self.paths.calendar_file, calendar_agent.CALENDAR_FILE)
        self.assertEqual(self.paths.calendar_ledger_file, calendar_agent.LEDGER_FILE)
        self.assertEqual(self.paths.calendar_summary_file, calendar_agent.SUMMARY_FILE)

    def test_matches_notes_agent_constants(self):
        self.assertEqual(self.paths.notes_dir, notes_agent.NOTES_DIR)
        self.assertEqual(self.paths.notes_ledger_file, notes_agent.LEDGER_FILE)
        self.assertEqual(self.paths.notes_summary_file, notes_agent.SUMMARY_FILE)

    def test_matches_orchestrator_constants(self):
        self.assertEqual(self.paths.tasks_file, orchestrator.TASKS_FILE)
        self.assertEqual(self.paths.brief_file, orchestrator.BRIEF_FILE)
        self.assertEqual(self.paths.history_dir, orchestrator.HISTORY_DIR)

    def test_matches_persona_and_tenant_config_constants(self):
        self.assertEqual(self.paths.persona_file, PERSONA_FILE)
        self.assertEqual(self.paths.tenant_config_file, CONFIG_FILE)

    def test_shared_history_dir_across_all_four_scripts(self):
        # All four scripts write history copies into one shared directory
        # (save_digest distinguishes them by filename, not by directory).
        self.assertEqual(triage_agent.HISTORY_DIR, calendar_agent.HISTORY_DIR)
        self.assertEqual(calendar_agent.HISTORY_DIR, notes_agent.HISTORY_DIR)
        self.assertEqual(notes_agent.HISTORY_DIR, orchestrator.HISTORY_DIR)


class TestNonDefaultTenantIsolation(unittest.TestCase):
    def test_resolves_under_tenants_subdirectory(self):
        paths = for_tenant("acme")
        self.assertTrue(paths.inbox_dir.startswith("./data/tenants/acme/"))
        self.assertTrue(paths.output_dir.startswith("./output/tenants/acme/"))

    def test_two_tenants_never_share_a_path(self):
        a = for_tenant("acme")
        b = for_tenant("globex")
        a_paths = {v for k, v in vars(a).items() if k != "tenant_id"}
        b_paths = {v for k, v in vars(b).items() if k != "tenant_id"}
        self.assertEqual(a_paths & b_paths, set())

    def test_non_default_tenant_never_touches_default_paths(self):
        default = for_tenant()
        acme = for_tenant("acme")
        default_paths = {v for k, v in vars(default).items() if k != "tenant_id"}
        acme_paths = {v for k, v in vars(acme).items() if k != "tenant_id"}
        self.assertEqual(default_paths & acme_paths, set())

    def test_same_tenant_id_is_idempotent(self):
        self.assertEqual(for_tenant("acme"), for_tenant("acme"))


if __name__ == "__main__":
    unittest.main()
