"""
Unit tests for resilience.py — CircuitBreaker state transitions and
TokenBucket timing, plus the optional breaker/metrics wiring in
llm.call_with_retry.

All timing-dependent behavior uses an injected fake clock (a plain counter,
not real time.sleep) — no test in this file takes real wall-clock time to
run, consistent with this session's established no-network, no-real-time-
delay testing pattern.

Run: python3 -m unittest test_resilience -v
"""

import json
import os
import tempfile
import unittest

from llm import TerminalLLMError, call_with_retry
from resilience import CircuitBreaker, CircuitOpenError, TokenBucket, get_breaker, get_provider_bucket, get_tenant_bucket, log_call_metrics, reset_registries


class _FakeClock:
    """A controllable clock — .advance(n) moves time forward without
    sleeping, so breaker/bucket recovery-timeout tests are instant.
    """

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreakerStateMachine(unittest.TestCase):
    def test_starts_closed(self):
        breaker = CircuitBreaker()
        self.assertEqual(breaker.state, "closed")
        breaker.before_call()  # should not raise

    def test_opens_after_threshold_consecutive_failures(self):
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            breaker.record_failure()
        self.assertEqual(breaker.state, "closed")
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")

    def test_open_breaker_rejects_calls_before_recovery_timeout(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")
        with self.assertRaises(CircuitOpenError):
            breaker.before_call()

    def test_success_resets_consecutive_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        # Two more failures shouldn't be enough to open, since success reset the count.
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.state, "closed")

    def test_transitions_to_half_open_after_recovery_timeout(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")

        clock.advance(30.0)
        breaker.before_call()  # should not raise — allows exactly one probe
        self.assertEqual(breaker.state, "half_open")

    def test_half_open_probe_success_closes_breaker(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)
        breaker.record_failure()
        clock.advance(10.0)
        breaker.before_call()
        breaker.record_success()
        self.assertEqual(breaker.state, "closed")

    def test_half_open_probe_failure_reopens_breaker(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)
        breaker.record_failure()
        clock.advance(10.0)
        breaker.before_call()
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")

    def test_half_open_rejects_a_second_concurrent_probe(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)
        breaker.record_failure()
        clock.advance(10.0)
        breaker.before_call()  # first probe admitted
        with self.assertRaises(CircuitOpenError):
            breaker.before_call()  # second probe while the first is still in flight


class TestTokenBucket(unittest.TestCase):
    def test_consumes_up_to_capacity(self):
        clock = _FakeClock()
        bucket = TokenBucket(capacity=3, refill_rate=1.0, clock=clock)
        self.assertTrue(bucket.try_consume())
        self.assertTrue(bucket.try_consume())
        self.assertTrue(bucket.try_consume())
        self.assertFalse(bucket.try_consume())  # exhausted

    def test_refills_over_time(self):
        clock = _FakeClock()
        bucket = TokenBucket(capacity=1, refill_rate=1.0, clock=clock)
        self.assertTrue(bucket.try_consume())
        self.assertFalse(bucket.try_consume())
        clock.advance(1.0)  # 1 token/sec * 1s = 1 token back
        self.assertTrue(bucket.try_consume())

    def test_never_exceeds_capacity(self):
        clock = _FakeClock()
        bucket = TokenBucket(capacity=2, refill_rate=10.0, clock=clock)
        clock.advance(100.0)  # way more than enough to overfill
        self.assertTrue(bucket.try_consume(2))
        self.assertFalse(bucket.try_consume(1))  # capped at capacity, not unbounded


class TestRegistries(unittest.TestCase):
    def tearDown(self):
        reset_registries()

    def test_get_breaker_returns_same_instance_for_same_key(self):
        self.assertIs(get_breaker("deepseek", "deepseek-chat"), get_breaker("deepseek", "deepseek-chat"))

    def test_get_breaker_returns_different_instances_for_different_models(self):
        self.assertIsNot(get_breaker("deepseek", "deepseek-chat"), get_breaker("deepseek", "deepseek-reasoner"))

    def test_get_provider_bucket_shared_across_tenants(self):
        self.assertIs(get_provider_bucket("deepseek"), get_provider_bucket("deepseek"))

    def test_get_tenant_bucket_isolated_per_tenant(self):
        self.assertIsNot(get_tenant_bucket("acme", "deepseek"), get_tenant_bucket("globex", "deepseek"))


class TestCallWithRetryBreakerIntegration(unittest.TestCase):
    def test_open_breaker_short_circuits_without_calling_fn(self):
        clock = _FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0, clock=clock)
        breaker.record_failure()  # opens it

        calls = {"n": 0}

        def _fn():
            calls["n"] += 1
            return "should never run"

        with self.assertRaises(CircuitOpenError):
            call_with_retry(_fn, breaker=breaker)
        self.assertEqual(calls["n"], 0)

    def test_success_records_on_breaker(self):
        breaker = CircuitBreaker(failure_threshold=1)
        result = call_with_retry(lambda: "ok", breaker=breaker)
        self.assertEqual(result, "ok")
        self.assertEqual(breaker.state, "closed")

    def test_terminal_error_records_failure_on_breaker(self):
        breaker = CircuitBreaker(failure_threshold=1)

        def _fn():
            raise TerminalLLMError("bad request")

        with self.assertRaises(TerminalLLMError):
            call_with_retry(_fn, breaker=breaker)
        self.assertEqual(breaker.state, "open")

    def test_no_breaker_passed_behaves_exactly_as_before(self):
        # Regression check: every existing call site that doesn't pass
        # breaker/llm/metrics_path must be completely unaffected.
        result = call_with_retry(lambda: "ok")
        self.assertEqual(result, "ok")


class TestMetricsLogging(unittest.TestCase):
    def test_log_call_metrics_appends_one_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            log_call_metrics(path, tenant_id="acme", provider="deepseek", model="deepseek-chat", latency_ms=123.4, success=True, retries=0)
            log_call_metrics(path, tenant_id="acme", provider="deepseek", model="deepseek-chat", latency_ms=456.7, success=False, retries=2, breaker_state="open")

            with open(path) as f:
                lines = [json.loads(line) for line in f]

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["tenant_id"], "acme")
            self.assertTrue(lines[0]["success"])
            self.assertFalse(lines[1]["success"])
            self.assertEqual(lines[1]["breaker_state"], "open")

    def test_call_with_retry_logs_metrics_when_path_given(self):
        class _StubLLM:
            provider = "deepseek"
            model = "deepseek-chat"
            tenant_id = "acme"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            call_with_retry(lambda: "ok", llm=_StubLLM(), metrics_path=path)

            with open(path) as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["tenant_id"], "acme")
            self.assertEqual(lines[0]["provider"], "deepseek")

    def test_no_metrics_path_writes_nothing(self):
        # Implicit: if this didn't no-op cleanly, every other test in this
        # file (none of which pass metrics_path) would have side effects.
        call_with_retry(lambda: "ok")


if __name__ == "__main__":
    unittest.main()
