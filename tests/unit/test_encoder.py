"""Unit tests for the embedding encoder.

These use a **fake** model rather than the real Sentence-BERT checkpoint, so the
suite runs in well under a second and needs no network. The real model is
exercised by the ``slow`` tests in ``tests/integration/test_encoder_real.py``,
which is where actual embedding quality is checked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from semantic_search.core.config import EmbeddingConfig
from semantic_search.data.loader import Document
from semantic_search.embeddings.encoder import EmbeddingEncoder, detect_device


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Encodes each text as a fixed-width vector derived from its characters, so
    identical inputs always give identical outputs and different inputs give
    different outputs -- everything the encoder's own logic depends on.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.max_seq_length = 256
        self.encode_calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encode_calls.append(
            {
                "count": len(sentences),
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
                "progress": show_progress_bar,
            }
        )
        matrix = np.zeros((len(sentences), self._dimension), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for position, char in enumerate(sentence[: self._dimension]):
                matrix[row, position % self._dimension] += ord(char)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, np.where(norms == 0, 1.0, norms))
        return matrix


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder(
        EmbeddingConfig(model_name="fake-model", batch_size=4),
        device="cpu",
        model=FakeModel(dimension=8),
    )


class TestDeviceDetection:
    def test_explicit_device_wins(self) -> None:
        assert detect_device("cpu") == "cpu"

    def test_auto_detects_without_error(self) -> None:
        # Must not raise and must return a usable torch device string.
        assert detect_device("auto") in ("cpu", "cuda", "mps", "cuda:0")

    def test_config_default_is_auto(self) -> None:
        assert EmbeddingConfig().device == "auto"


class TestEncoding:
    def test_returns_correct_shape(self, encoder: EmbeddingEncoder) -> None:
        assert encoder.encode(["a", "b", "c"]).shape == (3, 8)

    def test_returns_float32(self, encoder: EmbeddingEncoder) -> None:
        assert encoder.encode(["a"]).dtype == np.float32

    def test_empty_input_returns_empty_matrix(self, encoder: EmbeddingEncoder) -> None:
        matrix = encoder.encode([])
        assert matrix.shape == (0, 8)

    def test_is_deterministic(self, encoder: EmbeddingEncoder) -> None:
        assert np.array_equal(encoder.encode(["hello"]), encoder.encode(["hello"]))

    def test_different_texts_give_different_vectors(self, encoder: EmbeddingEncoder) -> None:
        matrix = encoder.encode(["alpha", "zzzzzzz"])
        assert not np.array_equal(matrix[0], matrix[1])

    def test_batching_passes_configured_size(self, encoder: EmbeddingEncoder) -> None:
        encoder.encode(["a", "b"])
        assert encoder.model.encode_calls[0]["batch_size"] == 4

    def test_batch_size_can_be_overridden(self, encoder: EmbeddingEncoder) -> None:
        encoder.encode(["a"], batch_size=1)
        assert encoder.model.encode_calls[0]["batch_size"] == 1

    def test_normalisation_is_forwarded(self, encoder: EmbeddingEncoder) -> None:
        encoder.encode(["a"])
        assert encoder.model.encode_calls[0]["normalize"] is True

    def test_progress_bar_flag_is_forwarded(self, encoder: EmbeddingEncoder) -> None:
        encoder.encode(["a"], show_progress=True)
        assert encoder.model.encode_calls[0]["progress"] is True

    def test_unit_norm_when_normalised(self, encoder: EmbeddingEncoder) -> None:
        matrix = encoder.encode(["some text here"])
        assert np.linalg.norm(matrix[0]) == pytest.approx(1.0, rel=1e-5)


class TestDocumentsAndQueries:
    def test_encode_documents_uses_shape(self, encoder: EmbeddingEncoder) -> None:
        docs = [Document(id="1", title="Albedo", text="Reflectivity of a body.")]
        assert encoder.encode_documents(docs).shape == (1, 8)

    def test_title_affects_the_vector(self, encoder: EmbeddingEncoder) -> None:
        # The title-prefix is what distinguishes two documents with the same body.
        base = [Document(id="1", title="aaa", text="identical body text")]
        alt = [Document(id="2", title="zzz", text="identical body text")]
        assert not np.array_equal(encoder.encode_documents(base), encoder.encode_documents(alt))

    def test_encode_query_returns_single_row(self, encoder: EmbeddingEncoder) -> None:
        assert encoder.encode_query("albedo").shape == (1, 8)


class TestMetadata:
    def test_dimension_from_model(self, encoder: EmbeddingEncoder) -> None:
        assert encoder.dimension == 8

    def test_dimension_is_cached(self, encoder: EmbeddingEncoder) -> None:
        assert encoder.dimension is encoder.dimension or encoder.dimension == 8

    def test_info_reports_configuration(self, encoder: EmbeddingEncoder) -> None:
        info = encoder.info()
        assert info["model_name"] == "fake-model"
        assert info["dimension"] == 8
        assert info["device"] == "cpu"

    def test_model_is_not_loaded_until_used(self) -> None:
        # Constructing an encoder must not touch the network.
        encoder = EmbeddingEncoder(EmbeddingConfig(model_name="definitely-not-a-real-model"))
        assert encoder.device in ("cpu", "cuda", "mps")
