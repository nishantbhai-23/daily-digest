"""
Unit tests for llm.py's retryable/terminal error classification — the fix
for docs/ERROR_HANDLING.md gap #2: retrying a 402 (insufficient balance,
hit directly on both OpenRouter and DeepSeek this session) or other
confidently-non-retryable error wastes time and, on a paid API, money on
every attempt, since retrying can never fix it.

No network calls — tests the classification and retry-short-circuit logic
directly.

Run: python3 -m unittest test_llm_error_classification -v
"""

import time
import unittest

from llm import TerminalLLMError, _raise_classified, call_with_retry


class TestRaiseClassified(unittest.TestCase):
    def test_terminal_codes_raise_terminal_error(self):
        for code in (400, 401, 402, 403):
            with self.subTest(code=code):
                cause = ValueError("original")
                with self.assertRaises(TerminalLLMError) as ctx:
                    _raise_classified(f"error {code}", code, cause)
                self.assertIs(ctx.exception.__cause__, cause)

    def test_non_terminal_codes_raise_runtime_error_not_terminal(self):
        for code in (429, 500, 503, None):
            with self.subTest(code=code):
                cause = ValueError("original")
                with self.assertRaises(RuntimeError) as ctx:
                    _raise_classified("error", code, cause)
                self.assertNotIsInstance(ctx.exception, TerminalLLMError)
                self.assertIs(ctx.exception.__cause__, cause)


class TestCallWithRetryShortCircuit(unittest.TestCase):
    def test_terminal_error_fails_immediately_no_retry(self):
        attempts = {"n": 0}

        def _call():
            attempts["n"] += 1
            raise TerminalLLMError("insufficient balance")

        start = time.time()
        with self.assertRaises(TerminalLLMError):
            call_with_retry(_call, max_retries=3, base_delay=2.0)
        elapsed = time.time() - start

        self.assertEqual(attempts["n"], 1)  # never retried
        self.assertLess(elapsed, 0.5)  # never slept waiting to retry

    def test_retryable_error_still_retries_as_before(self):
        # Regression check: the TerminalLLMError short-circuit must not
        # break the existing retry behavior for ordinary errors.
        attempts = {"n": 0}

        def _call():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "success"

        result = call_with_retry(_call, max_retries=3, base_delay=0.01)
        self.assertEqual(result, "success")
        self.assertEqual(attempts["n"], 3)


if __name__ == "__main__":
    unittest.main()
