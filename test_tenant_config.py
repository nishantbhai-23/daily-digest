"""
Unit tests for tenant_config.load_tenant_config — default fallback and
merge behavior. No LLM involved; pure function in/out checks.

Run: python3 -m unittest test_tenant_config -v
"""

import json
import os
import tempfile
import unittest

from tenant_config import DEFAULT_CONFIG, DEFAULT_SYSTEM_CONFIG, SYSTEM_ENFORCED_KEYS, load_tenant_config

_NONEXISTENT_SYSTEM_PATH = "/nonexistent/path/system_config.json"


class TestMissingFile(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        # Both tenant and system config files missing -> DEFAULT_CONFIG plus
        # DEFAULT_SYSTEM_CONFIG's keys layered on top (system layering always
        # runs, even against its own defaults — see TestSystemConfigLayering).
        result = load_tenant_config(path="/nonexistent/path/tenant_config.json", system_path=_NONEXISTENT_SYSTEM_PATH)
        expected = dict(DEFAULT_CONFIG, **DEFAULT_SYSTEM_CONFIG)
        self.assertEqual(result, expected)

    def test_missing_file_returns_a_copy_not_the_same_object(self):
        # Mutating the result shouldn't corrupt DEFAULT_CONFIG for later callers.
        result = load_tenant_config(path="/nonexistent/path/tenant_config.json", system_path=_NONEXISTENT_SYSTEM_PATH)
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
        result = load_tenant_config(path=path, system_path=_NONEXISTENT_SYSTEM_PATH)
        self.assertFalse(result["use_persona_in_map"])
        self.assertEqual(result["never_draft_contacts"], [])  # untouched default

    def test_nested_dict_merges_not_replaces(self):
        path = self._write_temp_config({
            "map_noise_filter": {"blocked_senders": ["a@x.com"]}
        })
        result = load_tenant_config(path=path, system_path=_NONEXISTENT_SYSTEM_PATH)
        self.assertEqual(result["map_noise_filter"]["blocked_senders"], ["a@x.com"])
        self.assertEqual(result["map_noise_filter"]["blocked_domains"], [])  # merged, not dropped

    def test_full_override(self):
        path = self._write_temp_config({
            "never_draft_contacts": ["a@x.com", "b@y.com"],
        })
        result = load_tenant_config(path=path, system_path=_NONEXISTENT_SYSTEM_PATH)
        self.assertEqual(result["never_draft_contacts"], ["a@x.com", "b@y.com"])


class TestSystemConfigLayering(unittest.TestCase):
    """SYSTEM_ENFORCED_KEYS must always reflect system_config.json, never a
    tenant's own file — the same reasoning already applied to
    never_draft_contacts (safety-critical settings shouldn't depend on
    trusting arbitrary tenant-authored content).
    """

    def _write_temp_config(self, content: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(content, f)
        self.addCleanup(os.remove, path)
        return path

    def test_missing_system_file_uses_system_defaults(self):
        result = load_tenant_config(path="/nonexistent/tenant.json", system_path=_NONEXISTENT_SYSTEM_PATH)
        for key in SYSTEM_ENFORCED_KEYS:
            self.assertEqual(result[key], DEFAULT_SYSTEM_CONFIG[key])

    def test_system_file_values_are_applied(self):
        system_path = self._write_temp_config({"max_qps_per_tenant": 7.5})
        result = load_tenant_config(path="/nonexistent/tenant.json", system_path=system_path)
        self.assertEqual(result["max_qps_per_tenant"], 7.5)

    def test_tenant_file_cannot_override_system_enforced_keys(self):
        # Even if a tenant's own file sets a system-owned key, system config
        # must win — this is the entire point of the layering.
        tenant_path = self._write_temp_config({"max_qps_per_tenant": 9999.0})
        system_path = self._write_temp_config({"max_qps_per_tenant": 1.0})
        result = load_tenant_config(path=tenant_path, system_path=system_path)
        self.assertEqual(result["max_qps_per_tenant"], 1.0)


class TestRealConfigFile(unittest.TestCase):
    def test_real_tenant_config_loads_and_has_sam_park(self):
        result = load_tenant_config()
        emails = [c["email"] for c in result["never_draft_contacts"]]
        self.assertIn("sam.park@gmail.com", emails)
        self.assertTrue(len(result["map_noise_filter"]["blocked_senders"]) > 0)

    def test_real_system_config_loads(self):
        result = load_tenant_config()
        self.assertEqual(result["max_qps_per_tenant"], 2.0)
        self.assertIn("deepseek", result["allowed_providers"])


if __name__ == "__main__":
    unittest.main()
