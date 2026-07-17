"""
Unit tests for tenant_config.load_tenant_config — default fallback and
merge behavior. No LLM involved; pure function in/out checks.

Run: python3 -m unittest test_tenant_config -v
"""

import json
import os
import tempfile
import unittest

from tenant_config import DEFAULT_CONFIG, load_tenant_config


class TestMissingFile(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        result = load_tenant_config(path="/nonexistent/path/tenant_config.json")
        self.assertEqual(result, DEFAULT_CONFIG)

    def test_missing_file_returns_a_copy_not_the_same_object(self):
        # Mutating the result shouldn't corrupt DEFAULT_CONFIG for later callers.
        result = load_tenant_config(path="/nonexistent/path/tenant_config.json")
        result["never_draft_contacts"].append("someone@example.com")
        self.assertEqual(DEFAULT_CONFIG["never_draft_contacts"], [])


class TestMergeBehavior(unittest.TestCase):
    def _write_temp_config(self, content: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(content, f)
        self.addCleanup(os.remove, path)
        return path

    def test_partial_override_keeps_other_defaults(self):
        path = self._write_temp_config({"use_persona_in_map": False})
        result = load_tenant_config(path=path)
        self.assertFalse(result["use_persona_in_map"])
        self.assertEqual(result["never_draft_contacts"], [])  # untouched default

    def test_nested_dict_merges_not_replaces(self):
        path = self._write_temp_config({
            "map_noise_filter": {"blocked_senders": ["a@x.com"]}
        })
        result = load_tenant_config(path=path)
        self.assertEqual(result["map_noise_filter"]["blocked_senders"], ["a@x.com"])
        self.assertEqual(result["map_noise_filter"]["blocked_domains"], [])  # merged, not dropped

    def test_full_override(self):
        path = self._write_temp_config({
            "never_draft_contacts": ["a@x.com", "b@y.com"],
        })
        result = load_tenant_config(path=path)
        self.assertEqual(result["never_draft_contacts"], ["a@x.com", "b@y.com"])


class TestRealConfigFile(unittest.TestCase):
    def test_real_tenant_config_loads_and_has_sam_park(self):
        result = load_tenant_config()
        emails = [c["email"] for c in result["never_draft_contacts"]]
        self.assertIn("sam.park@gmail.com", emails)
        self.assertTrue(len(result["map_noise_filter"]["blocked_senders"]) > 0)


if __name__ == "__main__":
    unittest.main()
