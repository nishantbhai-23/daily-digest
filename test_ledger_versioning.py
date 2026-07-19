"""
Unit tests for ledger.check_schema_consistency — the fix for a design-review
finding: MAP_SCHEMA vs MAP_SCHEMA_NO_PERSONA (tenant_config's
use_persona_in_map) already produce structurally different `delta` shapes,
and ledgers are resumable by design, so a config change between runs used to
mix old and new entries with nothing recording which was which. Every MAP
call now stamps "map_variant" on its entry; this file tests the deterministic
check that surfaces a resulting mismatch instead of leaving it silent.

Run: python3 -m unittest test_ledger_versioning -v
"""

import unittest

from ledger import check_schema_consistency


def _entry(day, map_variant=None, compacted=False):
    entry = {"day": day, "email_count": 1, "stats": {}, "delta": {}}
    if map_variant is not None:
        entry["map_variant"] = map_variant
    if compacted:
        entry["compacted"] = True
    return entry


class TestCheckSchemaConsistency(unittest.TestCase):
    def test_empty_ledger_no_warning(self):
        self.assertEqual(check_schema_consistency([]), [])

    def test_single_entry_no_warning(self):
        ledger = [_entry("2026-07-01", "persona")]
        self.assertEqual(check_schema_consistency(ledger), [])

    def test_all_same_variant_no_warning(self):
        ledger = [_entry("2026-07-01", "persona"), _entry("2026-07-02", "persona")]
        self.assertEqual(check_schema_consistency(ledger), [])

    def test_mixed_variants_warns(self):
        ledger = [_entry("2026-07-01", "persona"), _entry("2026-07-02", "no_persona")]
        warnings = check_schema_consistency(ledger)
        self.assertEqual(len(warnings), 1)
        self.assertIn("persona", warnings[0])
        self.assertIn("no_persona", warnings[0])

    def test_unversioned_entries_treated_as_own_category(self):
        # Pre-existing entries from before map_variant existed shouldn't be
        # silently ignored — a ledger mixing tagged and untagged entries is
        # exactly the kind of drift this check exists to catch.
        ledger = [_entry("2026-07-01", "persona"), _entry("2026-07-02", map_variant=None)]
        warnings = check_schema_consistency(ledger)
        self.assertEqual(len(warnings), 1)
        self.assertIn("unversioned", warnings[0])

    def test_all_unversioned_no_warning(self):
        # An old ledger, entirely pre-dating this field, is internally
        # consistent with itself even though it has no version info at all.
        ledger = [_entry("2026-07-01"), _entry("2026-07-02")]
        self.assertEqual(check_schema_consistency(ledger), [])

    def test_compacted_entries_excluded_from_check(self):
        # Compacted (weekly-rollup) entries come from a different LLM call
        # entirely (compact_ledger's merge), not day/note MAP — mixing with
        # them is expected, not a version-drift signal.
        ledger = [
            _entry("2026-07-01", "persona"),
            _entry("2026-W25", compacted=True),  # no map_variant at all
        ]
        self.assertEqual(check_schema_consistency(ledger), [])

    def test_compacted_entries_dont_mask_a_real_mismatch(self):
        ledger = [
            _entry("2026-07-01", "persona"),
            _entry("2026-07-02", "no_persona"),
            _entry("2026-W25", compacted=True),
        ]
        warnings = check_schema_consistency(ledger)
        self.assertEqual(len(warnings), 1)

    def test_three_way_mismatch_reports_all(self):
        ledger = [
            _entry("2026-07-01", "persona"),
            _entry("2026-07-02", "no_persona"),
            _entry("2026-07-03", map_variant=None),
        ]
        warnings = check_schema_consistency(ledger)
        self.assertEqual(len(warnings), 1)
        self.assertIn("3 different MAP configurations", warnings[0])


if __name__ == "__main__":
    unittest.main()
