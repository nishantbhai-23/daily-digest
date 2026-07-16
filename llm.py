"""
LLM Abstraction Layer
=====================
Provider-agnostic interface for language model interactions.
Business logic calls chat() or chat_json() without knowing
which provider is behind it.

Supported providers:
    - ollama:     Local models via Ollama (llama3.1, mistral, qwen, etc.)
    - anthropic:  Claude models via Anthropic API
    - google:     Gemini models via Google Generative AI API
    - openrouter: Any model available via OpenRouter (openrouter.ai) — useful
                  for fast, large-context, cheap cloud models when a local
                  model's context window or single-slot concurrency is a
                  bottleneck.
    - deepseek:   DeepSeek models (deepseek-chat, deepseek-reasoner) via a
                  direct DeepSeek API connection — very low cost, dedicated
                  rate limits (not a shared aggregator pool).

Usage:
    from llm import create_llm

    llm = create_llm("ollama", "llama3.1:8b", temperature=0.0)
    response = llm.chat([{"role": "user", "content": "Hello"}])
    data = llm.chat_json([{"role": "user", "content": "Extract..."}])
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


def call_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 2.0, retry_on=(Exception,), **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff retry.

    A single transient LLM failure (timeout, malformed JSON, a schema
    validation error raised by the caller) shouldn't silently drop a day's
    data — this retries before giving up. Callers that need schema
    validation on top of a raw chat_json() call should wrap both in a local
    closure and pass that in, so a validation failure also triggers a retry:

        def _call():
            result = llm.chat_json(messages=...)
            errors = validate_schema(result, SCHEMA)
            if errors:
                raise ValueError(f"Invalid MAP output: {errors}")
            return result

        delta = call_with_retry(_call)

    Args:
        fn: Callable to invoke.
        max_retries: Total attempts before giving up (>= 1).
        base_delay: Seconds to wait before the first retry; doubles each
            subsequent attempt.
        retry_on: Exception types that trigger a retry.

    Returns:
        Whatever fn(*args, **kwargs) returns.

    Raises:
        The last exception encountered, if every attempt fails.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except retry_on as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_error


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement chat(). The chat_json() method has a default
    implementation that calls chat() and parses JSON from the response,
    but subclasses can override it for native JSON mode support.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages and get a text response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Supported roles: 'system', 'user', 'assistant'.

        Returns:
            The model's text response.
        """
        pass

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Send messages and get a parsed JSON response.

        Default implementation calls chat() and parses the result.
        Subclasses can override for native JSON mode support.

        Returns:
            Parsed JSON as a dict.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
        """
        response = self.chat(messages, **kwargs)
        cleaned = self._extract_json(response)
        return json.loads(cleaned)

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple:
        """Separate system message from conversation messages.

        Some providers (Anthropic, Google) require system prompts
        to be passed separately from the message array.

        Returns:
            (system_content_or_none, non_system_messages)
        """
        system_content = None
        other_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                other_messages.append(msg)
        return system_content, other_messages

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a response that may be wrapped in markdown fences."""
        # Try to find JSON inside ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try to find a raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text


# ─── Ollama Provider ──────────────────────────────────────────────────────────


class OllamaLLM(BaseLLM):
    """LLM provider for local models via Ollama.

    Requires: pip install ollama
    Requires: Ollama server running locally (ollama serve)
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import ollama

            self._client = ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for the Ollama provider.\n"
                "Install with: pip install ollama"
            )

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = self._client.chat(
            model=self.model,
            options={"temperature": self.temperature},
            messages=messages,
        )
        return response["message"]["content"]

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Uses Ollama's native JSON mode (format='json')."""
        response = self._client.chat(
            model=self.model,
            format="json",
            options={"temperature": self.temperature},
            messages=messages,
        )
        content = response["message"]["content"]
        return json.loads(content)


# ─── Anthropic Provider ───────────────────────────────────────────────────────


class AnthropicLLM(BaseLLM):
    """LLM provider for Claude models via the Anthropic API.

    Requires: pip install anthropic
    Requires: ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider.\n"
                "Install with: pip install anthropic"
            )

    def chat(self, messages: list[dict], **kwargs) -> str:
        system_content, chat_messages = self._split_system(messages)

        api_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": chat_messages,
        }
        if system_content:
            api_kwargs["system"] = system_content

        response = self._client.messages.create(**api_kwargs)
        return response.content[0].text

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Injects JSON instruction into system prompt for structured output."""
        json_suffix = (
            "\n\nYou MUST respond with valid JSON only. "
            "No markdown wrappers, no explanation, no extra text."
        )
        modified = []
        for msg in messages:
            if msg["role"] == "system":
                modified.append(
                    {"role": "system", "content": msg["content"] + json_suffix}
                )
            else:
                modified.append(msg)

        response_text = self.chat(modified, **kwargs)
        cleaned = self._extract_json(response_text)
        return json.loads(cleaned)


# ─── Google Provider ──────────────────────────────────────────────────────────


class GoogleLLM(BaseLLM):
    """LLM provider for Gemini models via Google Generative AI API.

    Requires: pip install google-generativeai
    Requires: GOOGLE_API_KEY environment variable
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import google.generativeai as genai

            self._genai = genai
        except ImportError:
            raise ImportError(
                "The 'google-generativeai' package is required for the Google provider.\n"
                "Install with: pip install google-generativeai"
            )
        import os

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set.\n"
                "Get a key at: https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)

    def _create_model(self, system_content=None, json_mode=False):
        """Create a GenerativeModel with optional system instruction and JSON mode."""
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model_kwargs = {
            "model_name": self.model,
            "generation_config": generation_config,
        }
        if system_content:
            model_kwargs["system_instruction"] = system_content

        return self._genai.GenerativeModel(**model_kwargs)

    def chat(self, messages: list[dict], **kwargs) -> str:
        system_content, chat_messages = self._split_system(messages)
        model = self._create_model(system_content=system_content)

        # Convert to Gemini message format
        gemini_history = []
        for msg in chat_messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        response = model.generate_content(gemini_history)
        return response.text

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Uses Gemini's native JSON mode via response_mime_type."""
        system_content, chat_messages = self._split_system(messages)
        model = self._create_model(system_content=system_content, json_mode=True)

        gemini_history = []
        for msg in chat_messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        response = model.generate_content(gemini_history)
        return json.loads(response.text)


# ─── OpenAI-Compatible REST Providers (OpenRouter, DeepSeek, ...) ────────────

try:
    import ssl
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # certifi not installed — fall back to Python's default SSL context.
    # (On some macOS Python.org installs, this default lacks a working CA
    # bundle; installing certifi is the portable fix, no system changes needed.)
    _SSL_CONTEXT = None


class _OpenAICompatibleLLM(BaseLLM):
    """Base class for providers exposing an OpenAI-style /chat/completions
    endpoint (OpenRouter, DeepSeek, and similar). Implemented with stdlib
    urllib — no extra package dependency. Uses certifi's CA bundle for TLS
    verification when available, since some Python.org macOS installs don't
    wire up to the system trust store.

    Subclasses set API_URL, API_KEY_ENV, and PROVIDER_NAME/KEY_URL (used only
    in the missing-key error message).
    """

    API_URL: str = ""
    API_KEY_ENV: str = ""
    PROVIDER_NAME: str = ""
    KEY_URL: str = ""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        api_key = os.environ.get(self.API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_ENV} environment variable is not set.\n"
                f"Get a key at: {self.KEY_URL}"
            )
        self._api_key = api_key

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120, context=_SSL_CONTEXT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.PROVIDER_NAME} API error {e.code}: {body}") from e

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = self._post({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return response["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Requests the provider's JSON object response format when supported."""
        response = self._post({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        })
        content = response["choices"][0]["message"]["content"]
        cleaned = self._extract_json(content)
        return json.loads(cleaned)


class OpenRouterLLM(_OpenAICompatibleLLM):
    """LLM provider for models routed through OpenRouter (openrouter.ai).

    Requires: OPENROUTER_API_KEY environment variable.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    PROVIDER_NAME = "OpenRouter"
    KEY_URL = "https://openrouter.ai/keys"


class DeepSeekLLM(_OpenAICompatibleLLM):
    """LLM provider for DeepSeek models via the DeepSeek API (deepseek.com).

    Requires: DEEPSEEK_API_KEY environment variable.
    Model names: "deepseek-chat" (DeepSeek-V3, general purpose) or
    "deepseek-reasoner" (DeepSeek-R1, reasoning model). A direct, dedicated
    connection — not routed through any third-party aggregator's shared
    rate limits or credit pool.
    """

    API_URL = "https://api.deepseek.com/chat/completions"
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "DeepSeek"
    KEY_URL = "https://platform.deepseek.com/api_keys"


# ─── Factory ──────────────────────────────────────────────────────────────────

PROVIDERS = {
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
    "google": GoogleLLM,
    "openrouter": OpenRouterLLM,
    "deepseek": DeepSeekLLM,
}


def create_llm(provider: str, model: str, **kwargs) -> BaseLLM:
    """Factory function to create an LLM instance.

    Args:
        provider: One of "ollama", "anthropic", "google".
        model: Model name (e.g. "llama3.1:8b", "claude-sonnet-4-20250514",
               "gemini-2.5-flash").
        **kwargs: temperature (float), max_tokens (int).

    Returns:
        A configured BaseLLM instance.

    Raises:
        ValueError: If the provider is not recognized.
        ImportError: If the required package for the provider is not installed.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {', '.join(PROVIDERS.keys())}"
        )

    return PROVIDERS[provider](model, **kwargs)
