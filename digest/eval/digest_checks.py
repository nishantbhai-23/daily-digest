"""
Digest Quality Checks
=======================
Pure-Python (no LLM) checks used across both MAP-phase extraction
(eval_map.py) and REDUCE-phase digest output — a cheap first line of
defense against the failure modes actually hit during development: the
model describing the JSON data format instead of writing a digest, known
planted scenarios silently getting dropped or diluted during synthesis,
and (via extract_searchable_text/check_extraction_bloat) a keyword landing
in the wrong extraction category or the model hallucinating far more items
than a day's real content could support.

These are deliberately blunt (keyword/substring checks), not semantic
evaluation — they're meant to catch obvious regressions fast and for free,
not to replace an actual read of the digest. Pair with golden_scenarios.py
(what's being checked) and eval_map.py (the MAP-phase caller).

Usage:
    from digest.eval.digest_checks import check_not_schema_description, check_keywords_present

    text = open("output/current_30day_summary.md").read()
    check_not_schema_description(text)  # raises AssertionError if it looks like schema narration
    missing = check_keywords_present(text, ["marcus"])
"""

import re

from digest.core.cross_reference import leaf_strings

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


def extract_searchable_text(container: dict, categories: str | list[str] | None = None) -> str:
    """Leaf-string-only text extraction for keyword checks — never includes
    schema key names, unlike a raw json.dumps() search (the same bug class
    found and fixed in cross_reference.py's own matching, via leaf_strings
    there too — every MAP delta item has a "description" field, so a
    json.dumps() search would let any scenario whose keywords happen to
    collide with a schema key name pass for free, regardless of content).

    If `categories` is given (a single category name, or a list — for
    scenarios where the correct category is genuinely ambiguous, see
    golden_scenarios.py), scopes the search to just those subtrees instead
    of the whole container. This is what makes a keyword match structurally
    meaningful ("landed in the right slot"), not just present somewhere in
    the extraction.
    """
    if categories is None:
        target = container
    elif isinstance(categories, str):
        target = {categories: container.get(categories, [])}
    else:
        target = {c: container.get(c, []) for c in categories}
    return " ".join(leaf_strings(target))


def check_extraction_bloat(delta: dict, max_items: int) -> str | None:
    """Coarse sanity bound on total extracted items across every category
    in one MAP delta — catches a hallucination-flood failure mode (the
    model inventing dozens of spurious items), not everyday volume
    variation. Deliberately generous: callers pass a per-source max_items
    calibrated against this project's own real corpus (see eval_map.py's
    *_MAX_ITEMS constants and the real observed maxima that set them), not
    a tight precision bound.

    Returns:
        None if under the bound, else a human-readable warning string.
    """
    total = sum(len(v) for v in delta.values() if isinstance(v, list))
    if total > max_items:
        return f"{total} total items extracted (>{max_items}) — possible extraction bloat"
    return None
