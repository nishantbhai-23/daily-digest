"""
Persona Loader
===============
Loads the operator profile (data/persona.md) that both the email and
calendar triage agents inject into their prompts so triage and digest
synthesis reflect the operator's stated priorities, tone, and blind spots
instead of generic defaults.

Usage:
    from digest.core.persona import load_persona

    persona_text = load_persona()
"""

import os

PERSONA_FILE = "./data/persona.md"


def load_persona(path: str = PERSONA_FILE) -> str:
    """Load the persona markdown file as raw text.

    Args:
        path: Path to the persona markdown file.

    Returns:
        The persona file contents as a string.

    Raises:
        FileNotFoundError: If the persona file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Persona file not found at '{path}'. "
            "The triage agents rely on this to personalize triage and digests."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
