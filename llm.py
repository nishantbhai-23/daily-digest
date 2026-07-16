"""
LLM Abstraction Layer
=====================
Provider-agnostic interface for language model interactions.
Business logic calls chat() or chat_json() without knowing
which provider is behind it.

Supported providers:
    - ollama:    Local models via Ollama (llama3.1, mistral, qwen, etc.)
    - anthropic: Claude models via Anthropic API
    - google:    Gemini models via Google Generative AI API

Usage:
    from llm import create_llm

    llm = create_llm("ollama", "llama3.1:8b", temperature=0.0)
    response = llm.chat([{"role": "user", "content": "Hello"}])
    data = llm.chat_json([{"role": "user", "content": "Extract..."}])
"""

import json
import re
from abc import ABC, abstractmethod


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


# ─── Factory ──────────────────────────────────────────────────────────────────

PROVIDERS = {
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
    "google": GoogleLLM,
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
