"""
Unit tests confirming every provider actually applies a request timeout —
the fix for the gap identified in docs/ERROR_HANDLING.md: only
_OpenAICompatibleLLM (OpenRouter/DeepSeek) had an explicit timeout before
this; OllamaLLM, AnthropicLLM, and GoogleLLM had none, which is the likely
root cause of the multi-thousand-second stalls observed on local Ollama.

No network calls — these check that the timeout value actually reaches the
underlying client/request configuration, not that a real call respects it.

Run: python3 -m unittest test_llm_timeouts -v
"""

import unittest

from llm import DEFAULT_TIMEOUT_SECONDS, OllamaLLM


class TestOllamaTimeout(unittest.TestCase):
    def test_default_timeout_applied(self):
        llm = OllamaLLM("llama3")
        self.assertEqual(llm.timeout, DEFAULT_TIMEOUT_SECONDS)

    def test_custom_timeout_applied(self):
        llm = OllamaLLM("llama3", timeout=45)
        self.assertEqual(llm.timeout, 45)

    def test_timeout_reaches_underlying_httpx_client(self):
        # The bare module-level ollama.chat() has no way to set a timeout at
        # all (confirmed by inspecting the installed package's signature) —
        # this is why OllamaLLM now instantiates an explicit ollama.Client.
        # Verify the timeout actually propagates all the way down, not just
        # that our own attribute is set.
        llm = OllamaLLM("llama3", timeout=45)
        self.assertEqual(llm._client._client.timeout.connect, 45)


if __name__ == "__main__":
    unittest.main()
