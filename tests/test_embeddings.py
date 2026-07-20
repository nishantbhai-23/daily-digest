"""
Unit tests for digest.core.embeddings — pure math (cosine_similarity,
best_matches) needs no live Ollama call; embed_texts is tested via an
injected fake client so the suite stays fast and network-free.

Run: python3 -m unittest tests.test_embeddings -v
"""

import unittest

from digest.core.embeddings import best_matches, cosine_similarity, embed_texts


class _FakeEmbedResponse(dict):
    """Mimics ollama's EmbedResponse enough for embed_texts: dict-style
    __getitem__ access to "embeddings"."""


class _FakeClient:
    def __init__(self, embeddings: list[list[float]]):
        self._embeddings = embeddings
        self.last_call = None

    def embed(self, model, input):
        self.last_call = {"model": model, "input": input}
        return _FakeEmbedResponse(embeddings=self._embeddings)


class _BrokenClient:
    def embed(self, model, input):
        raise ConnectionError("server not running")


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_are_similarity_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_vectors_are_similarity_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_are_similarity_negative_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero_not_a_crash(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)


class TestBestMatches(unittest.TestCase):
    def test_returns_only_matches_at_or_above_threshold_sorted_descending(self):
        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],   # similarity 1.0
            [0.0, 1.0],   # similarity 0.0
            [0.9, 0.1],   # similarity ~0.994
        ]
        matches = best_matches(query, candidates, threshold=0.5)
        self.assertEqual([i for i, _ in matches], [0, 2])
        self.assertGreater(matches[0][1], matches[1][1])

    def test_no_matches_above_threshold_returns_empty(self):
        matches = best_matches([1.0, 0.0], [[0.0, 1.0]], threshold=0.9)
        self.assertEqual(matches, [])


class TestEmbedTexts(unittest.TestCase):
    def test_empty_input_returns_empty_without_calling_client(self):
        client = _FakeClient(embeddings=[])
        self.assertEqual(embed_texts([], client=client), [])
        self.assertIsNone(client.last_call)

    def test_injected_client_receives_model_and_input(self):
        client = _FakeClient(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        result = embed_texts(["a", "b"], model="nomic-embed-text", client=client)
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(client.last_call, {"model": "nomic-embed-text", "input": ["a", "b"]})

    def test_client_failure_raises_actionable_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            embed_texts(["a"], client=_BrokenClient())
        self.assertIn("Ollama server running", str(ctx.exception))
        self.assertIn("ollama pull", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
