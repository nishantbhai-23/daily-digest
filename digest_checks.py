"""
Digest Quality Checks
=======================
Pure-Python (no LLM) checks for REDUCE-phase digest output — a cheap first
line of defense against the failure modes actually hit during development:
the model describing the JSON data format instead of writing a digest, and
known planted scenarios silently getting dropped or diluted during
synthesis.

These are deliberately blunt (keyword/substring checks), not semantic
evaluation — they're meant to catch obvious regressions fast and for free,
not to replace an actual read of the digest. Pair with golden_scenarios.py
(what's being checked) and eval_map.py (the MAP-phase equivalent).

Usage:
    from digest_checks import check_not_schema_description, check_keywords_present

    text = open("output/current_30day_summary.md").read()
    check_not_schema_description(text)  # raises AssertionError if it looks like schema narration
    missing = check_keywords_present(text, ["marcus"])
"""

import re

# Phrases that show up when a model narrates the JSON structure instead of
# writing the actual digest — the exact failure mode hit repeatedly with a
# local 8B model, even after explicit prompt instructions forbade it.
_SCHEMA_DESCRIPTION_RED_FLAGS = [
    r"this (?:is|file) (?:a|an) (?:json|profile)",
    r"contains an array of objects",
    r"each (?:object|entry|record) represents",
    r"the file (?:is in|contains|appears to be)",
    r"breakdown of (?:the|this) (?:properties|information|data)",
    r"here'?s a breakdown",
    r"top-level key",
]


def looks_like_schema_description(text: str) -> bool:
    """Return True if the text looks like it's narrating a data format
    instead of writing an actual digest.
    """
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _SCHEMA_DESCRIPTION_RED_FLAGS)


def check_not_schema_description(text: str) -> None:
    """Raise AssertionError if the digest looks like schema narration."""
    if looks_like_schema_description(text):
        raise AssertionError(
            "Digest looks like it's describing the JSON data format instead of "
            "writing an actual digest — this is the schema-describing failure mode."
        )


def check_keywords_present(text: str, required_keywords: list[str]) -> list[str]:
    """Check that all required keywords/phrases appear (case-insensitive).

    Returns:
        List of missing keywords — empty list means all were found.
    """
    lowered = text.lower()
    return [kw for kw in required_keywords if kw.lower() not in lowered]


def check_min_length(text: str, min_words: int = 50) -> bool:
    """Sanity check that the digest isn't empty or truncated."""
    return len(text.split()) >= min_words
