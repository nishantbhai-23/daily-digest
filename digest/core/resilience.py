"""
Resilience
===========
Fleet-level policy that sits *above* a single LLM call — split out of
llm.py deliberately: llm.py owns provider adapters and per-call retry
mechanics (call_with_retry, TerminalLLMError, timeouts); this module owns
policy that only makes sense once multiple tenants share one process's
calls to a given provider. Same reasoning as why tenant_config.py was split
from digest.core.persona import.py — different layer, different module.

Both pieces here are in-memory, module-level, process-local:
- CircuitBreaker: stops every concurrent tenant caller from independently
  retrying into a provider that's already degraded (the thundering-herd
  scenario in docs/ERROR_HANDLING.md gap #3).
- TokenBucket: caps QPS against a provider (and optionally per-tenant, for
  fairness) before a call is even attempted.

Deliberately in-memory rather than backed by Redis/a database: state only
needs fleet-wide visibility *within one process*, and this codebase's
concurrent multi-tenant runner (run_fleet.py) uses a ThreadPoolExecutor
specifically so this state is naturally shared for free. This doesn't
survive a process restart and doesn't share across multiple machines — the
actual trigger for needing something like Redis is "multiple independent
server processes," not "multiple tenants," and that's not this system's
shape today.

Usage:
    from digest.core.resilience import get_breaker, get_provider_bucket, CircuitOpenError

    breaker = get_breaker("deepseek", "deepseek-chat")
    breaker.before_call()  # raises CircuitOpenError if open
    try:
        result = do_the_call()
    except Exception:
        breaker.record_failure()
        raise
    else:
        breaker.record_success()
"""

import json
import os
import threading
import time


class CircuitOpenError(Exception):
    """Raised by CircuitBreaker.before_call() when the breaker is open —
    fails faster than even a TerminalLLMError, since it never touches the
    network at all.
    """


class CircuitBreaker:
    """Closed -> open -> half-open state machine for one (provider, model).

    - closed: calls proceed normally; consecutive failures are counted.
    - open: calls fail immediately via CircuitOpenError until
      recovery_timeout has elapsed since opening.
    - half-open: exactly one probe call is allowed through; success closes
      the breaker (resets failure count), failure reopens it (and resets
      the recovery clock).

    Thread-safe — state read/written for calls from many threads at once
    (the whole point of this class), so every method holds a lock.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()

        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def before_call(self) -> None:
        """Call before attempting the LLM call. Raises CircuitOpenError if
        the call should not be attempted at all.
        """
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "open":
                if self._clock() - self._opened_at >= self.recovery_timeout:
                    self._state = "half_open"
                    self._probe_in_flight = True
                    return
                raise CircuitOpenError(
                    f"Circuit open — provider unavailable, retrying in "
                    f"{self.recovery_timeout - (self._clock() - self._opened_at):.1f}s"
                )
            if self._state == "half_open":
                if self._probe_in_flight:
                    raise CircuitOpenError("Circuit half-open — a probe call is already in flight")
                self._probe_in_flight = True
                return

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self._state == "half_open":
                # Probe failed — reopen and reset the recovery clock.
                self._state = "open"
                self._opened_at = self._clock()
                self._probe_in_flight = False
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()


class TokenBucket:
    """Simple in-memory token bucket, threading.Lock-guarded.

    capacity: max tokens held (also the max instantaneous burst).
    refill_rate: tokens added per second.
    """

    def __init__(self, capacity: float, refill_rate: float, clock=time.monotonic):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = capacity
        self._last_refill = clock()

    def try_consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = self._clock()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


# ─── Module-level registries ───────────────────────────────────────────────
# Keyed lazily so callers don't need to pre-register every (provider, model)
# or tenant up front — first call to get_* creates the entry.

_breakers: dict[tuple, CircuitBreaker] = {}
_breakers_lock = threading.Lock()

_provider_buckets: dict[str, TokenBucket] = {}
_tenant_buckets: dict[tuple, TokenBucket] = {}
_buckets_lock = threading.Lock()


def get_breaker(provider: str, model: str, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> CircuitBreaker:
    """Fetch (or lazily create) the shared breaker for one (provider, model).

    failure_threshold/recovery_timeout only take effect on first creation —
    later calls just fetch the existing instance, so all callers share one
    breaker's state per key regardless of who configured it first.
    """
    key = (provider, model)
    with _breakers_lock:
        if key not in _breakers:
            _breakers[key] = CircuitBreaker(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
        return _breakers[key]


def get_provider_bucket(provider: str, capacity: float = 10.0, refill_rate: float = 2.0) -> TokenBucket:
    """Fetch (or lazily create) the shared token bucket protecting one
    provider's real rate limit, across every tenant calling it.
    """
    with _buckets_lock:
        if provider not in _provider_buckets:
            _provider_buckets[provider] = TokenBucket(capacity=capacity, refill_rate=refill_rate)
        return _provider_buckets[provider]


def get_tenant_bucket(tenant_id: str, provider: str, capacity: float = 5.0, refill_rate: float = 1.0) -> TokenBucket:
    """Fetch (or lazily create) a per-tenant token bucket — fairness on top
    of the provider-wide bucket, so one tenant's heavy run can't starve
    others sharing the same provider.
    """
    key = (tenant_id, provider)
    with _buckets_lock:
        if key not in _tenant_buckets:
            _tenant_buckets[key] = TokenBucket(capacity=capacity, refill_rate=refill_rate)
        return _tenant_buckets[key]


def log_call_metrics(
    metrics_path: str,
    *,
    tenant_id: str,
    provider: str,
    model: str,
    latency_ms: float,
    success: bool,
    retries: int,
    breaker_state: str | None = None,
) -> None:
    """Append one JSONL line describing an LLM call — "poor man's
    observability": no time-series DB, just enough structured data to
    compute QPS/error-rate/cost-per-tenant with a short script after the
    fact. Appends, never truncates — callers from multiple threads/tenants
    share this file, so each write must be a single atomic line (a single
    f.write() of one JSON-encoded line plus newline is atomic on POSIX for
    writes under PIPE_BUF, which every realistic metrics line is well
    under).
    """
    record = {
        "ts": time.time(),
        "tenant_id": tenant_id,
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "retries": retries,
        "breaker_state": breaker_state,
    }
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def reset_registries() -> None:
    """Clear all breaker/bucket state. Test-only — production code never
    needs to reset the shared registries mid-run.
    """
    with _breakers_lock:
        _breakers.clear()
    with _buckets_lock:
        _provider_buckets.clear()
        _tenant_buckets.clear()
