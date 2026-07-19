"""
Unit tests for ledger.py's flock-based locking (_ledger_lock, used by
load_ledger/save_ledger) — added for multi-tenant readiness, since two
concurrent runs of the same tenant would otherwise race on that tenant's
ledger file.

Verified directly while writing this (not assumed): the lock guarantees no
torn/corrupted JSON under heavy concurrent load_ledger+save_ledger calls —
real and worth having. It does NOT make a full read-modify-write *session*
(load once, mutate in memory, save — triage_agent.py's run_map_phase
pattern) atomic end-to-end; two concurrent sessions can still lose an
update. See ledger.py's _ledger_lock docstring for the full explanation.
This file tests both: what's actually guaranteed, and documents the known
gap with a test that demonstrates it rather than leaving it unverified.

Run: python3 -m unittest test_ledger_lock -v
"""

import json
import os
import tempfile
import threading
import unittest

from digest.core.ledger import load_ledger, save_ledger


class TestNoCorruptionUnderConcurrentAccess(unittest.TestCase):
    """The actual, verified guarantee: every individual load/save call is
    atomic with respect to every other one on the same path.
    """

    def test_heavy_concurrent_load_and_save_never_produces_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            save_ledger(path, [{"day": "2026-06-01", "email_count": 1, "stats": {}, "delta": {}}])

            errors = []

            def hammer():
                try:
                    for _ in range(50):
                        ledger, _ = load_ledger(path)
                        save_ledger(path, ledger)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=hammer) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            with open(path) as f:
                raw = f.read()
            # Must not raise -- this is the property the lock actually
            # guarantees: no reader/writer ever sees a torn file.
            parsed = json.loads(raw)
            self.assertIsInstance(parsed, list)


class TestKnownReadModifyWriteLimitation(unittest.TestCase):
    """Documents (via a passing test that demonstrates the behavior, not a
    skipped/xfail one) that per-call locking does not make a multi-call
    read-modify-write session atomic. Not a live risk for anything that
    currently ships (orchestrator.run_for_tenant only reads ledgers, never
    writes them), but real if two run_map_phase invocations for the same
    tenant ever run concurrently.
    """

    def test_two_concurrent_sessions_can_lose_an_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            save_ledger(path, [])

            barrier = threading.Barrier(2)

            def session(day: str):
                ledger, _ = load_ledger(path)
                barrier.wait()  # force both sessions to load before either saves
                ledger.append({"day": day, "email_count": 1, "stats": {}, "delta": {}})
                save_ledger(path, ledger)

            t1 = threading.Thread(target=session, args=("2026-06-01",))
            t2 = threading.Thread(target=session, args=("2026-06-02",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            final, _ = load_ledger(path)
            # Demonstrates the gap: with true session-level atomicity this
            # would be 2. Documenting the actual (lesser) guarantee rather
            # than asserting behavior the lock doesn't provide.
            self.assertEqual(len(final), 1)


if __name__ == "__main__":
    unittest.main()
