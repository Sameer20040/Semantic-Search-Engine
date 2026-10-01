"""Integration tests against the real ``all-MiniLM-L6-v2`` checkpoint.

Marked ``slow`` and excluded from the default run because they load a real
262 MB model. Run them with::

    pytest -m slow tests/integration/test_encoder_real.py

What is actually being checked here is *semantic behaviour*, not just that the
call returns an array: related sentences must land closer together than
unrelated ones. A model that returns noise would pass every shape assertion in
the unit tests while being useless for search.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_search.core.config import EmbeddingConfig
from semantic_search.data.loader import Document
from semantic_search.embeddings.encoder import EmbeddingEncoder

pytestmark = pytest.mark.slow

#: The checkpoint is in the local HF cache on the build machine. These tests are
#: skipped elsewhere rather than silently downloading 262 MB during CI.
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def real_encoder() -> EmbeddingEncoder:
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(MODEL, device="cpu")
    except Exception as exc:
        pytest.skip(f"Model {MODEL} unavailable: {exc}")
    return EmbeddingEncoder(EmbeddingConfig(model_name=MODEL, device="cpu"))


def cosine(matrix: np.ndarray, i: int, j: int) -> float:
    return float(matrix[i] @ matrix[j])


class TestRealEncoder:
    def test_dimension_is_384(self, real_encoder: EmbeddingEncoder) -> None:
        assert real_encoder.dimension == 384

    def test_vectors_are_normalised(self, real_encoder: EmbeddingEncoder) -> None:
        matrix = real_encoder.encode(["a sentence", "another sentence"])
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_related_sentences_are_closer_than_unrelated(
        self, real_encoder: EmbeddingEncoder
    ) -> None:
        matrix = real_encoder.encode(
            [
                "The Moon is a natural satellite of the Earth.",
                "The Earth's moon orbits the planet roughly every 27 days.",
                "Quantum chromodynamics describes the strong nuclear force.",
            ]
        )
        assert cosine(matrix, 0, 1) > cosine(matrix, 0, 2)

    def test_word_overlap_is_not_required(self, real_encoder: EmbeddingEncoder) -> None:
        # The property that separates dense retrieval from BM25: zero shared
        # content words, high semantic similarity.
        matrix = real_encoder.encode(
            [
                "how do I stop my car from freezing in winter",
                "ways to prevent your vehicle from ice over during cold months",
            ]
        )
        assert cosine(matrix, 0, 1) > 0.5

    def test_encoding_is_deterministic(self, real_encoder: EmbeddingEncoder) -> None:
        first = real_encoder.encode(["stability check"])
        second = real_encoder.encode(["stability check"])
        assert np.allclose(first, second, atol=1e-6)

    def test_long_text_is_truncated_not_failed(self, real_encoder: EmbeddingEncoder) -> None:
        # Documents far exceed the 256 word-piece window; they must still encode.
        matrix = real_encoder.encode(["word " * 5000])
        assert matrix.shape == (1, 384)
        assert np.isfinite(matrix).all()

    def test_title_prefix_changes_the_vector(self, real_encoder: EmbeddingEncoder) -> None:
        docs = [
            Document(id="1", title="Astronomy", text="A body that reflects light."),
            Document(id="2", title="Cooking", text="A body that reflects light."),
        ]
        matrix = real_encoder.encode_documents(docs)
        assert not np.allclose(matrix[0], matrix[1])

    def test_batching_does_not_change_vectors(self, real_encoder: EmbeddingEncoder) -> None:
        texts = [f"document number {i} about assorted topics" for i in range(8)]
        batched = real_encoder.encode(texts, batch_size=8)
        one_at_a_time = np.vstack([real_encoder.encode([t], batch_size=1) for t in texts])
        assert np.allclose(batched, one_at_a_time, atol=1e-5)
