"""
Local Embeddings (Ollama)
============================
Semantic similarity as a middle tier between literal keyword matching and
an LLM judge — see citations.py for where this actually gets used. Ollama
rather than a hosted embeddings API or a heavier local ML runtime: it's
already an integrated provider (digest.core.llm.OllamaLLM), genuinely
local/offline once a model is pulled, and needs no new API key.

Requires: pip install ollama (already a dependency of the Ollama chat
provider), Ollama server running locally, and an embedding model pulled:
    ollama pull nomic-embed-text

Usage:
    from digest.core.embeddings import embed_texts, cosine_similarity, best_matches

    vectors = embed_texts(["call Drew about the delay", "reach out to Andrew"])
    similarity = cosine_similarity(vectors[0], vectors[1])
"""

DEFAULT_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL, client=None) -> list[list[float]]:
    """Embeds a batch of texts in one call.

    `client` is injectable so tests can mock it without a real Ollama call
    — same testability discipline as reference_date being injectable
    everywhere date logic appears in this codebase.

    Raises a clear, actionable error if the ollama package isn't installed
    or the server/model isn't reachable, rather than letting a raw
    exception bubble up uncaught — OllamaLLM's chat calls currently have
    zero error classification (a gap already flagged during this
    project's error-handling review); this doesn't repeat it here.
    """
    if not texts:
        return []

    if client is None:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for local embeddings.\n"
                "Install with: pip install ollama"
            )
        client = ollama.Client()

    try:
        response = client.embed(model=model, input=texts)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get embeddings from Ollama (model='{model}'). "
            f"Is the Ollama server running, and has the model been pulled "
            f"(`ollama pull {model}`)? Original error: {e}"
        ) from e

    return response["embeddings"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure stdlib math — no numpy. Embedding dimensionality (hundreds to
    low thousands) and candidate counts (tens of sources) are small enough
    that this is a non-issue at this scale.

    Returns 0.0 for a zero-vector input rather than raising a
    division-by-zero error.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def best_matches(
    query_embedding: list[float], candidate_embeddings: list[list[float]], threshold: float
) -> list[tuple[int, float]]:
    """Indices + similarity scores for every candidate at or above
    threshold, sorted by score descending — keeps call sites from
    repeating the same scan-and-sort loop.
    """
    scored = [
        (i, cosine_similarity(query_embedding, candidate))
        for i, candidate in enumerate(candidate_embeddings)
    ]
    matches = [(i, score) for i, score in scored if score >= threshold]
    matches.sort(key=lambda pair: -pair[1])
    return matches
