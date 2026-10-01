"""Unit tests for the embedding cache.

The cache's job is to make a *stale* cache impossible to use by accident. Most of
these tests therefore assert that an incompatible cache is *refused*, not that a
compatible one loads. A cache that silently returns wrong vectors is the worst
failure this subsystem can have.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semantic_search.core.exceptions import EmbeddingCompatibilityError
from semantic_search.embeddings.cache import CACHE_VERSION, EmbeddingCache, EmbeddingMetadata


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings.npy")


def seed(cache: EmbeddingCache, rows: int = 4, dim: int = 3, **overrides: object) -> None:
    matrix = np.arange(rows * dim, dtype=np.float32).reshape(rows, dim)
    metadata = EmbeddingMetadata(
        model_name="all-MiniLM-L6-v2",
        dimension=dim,
        document_count=rows,
        completed_count=rows,
        corpus_fingerprint="abc123",
        created_at="2026-01-01T00:00:00+00:00",
    )
    for key, value in overrides.items():
        setattr(metadata, key, value)
    cache.save(matrix, metadata)


class TestSaveLoad:
    def test_roundtrip_preserves_values(self, cache: EmbeddingCache) -> None:
        seed(cache)
        loaded = cache.load(mmap=False)
        assert loaded.shape == (4, 3)
        assert np.allclose(loaded, np.arange(12, dtype=np.float32).reshape(4, 3))

    def test_stores_float32(self, cache: EmbeddingCache) -> None:
        seed(cache)
        assert cache.load(mmap=False).dtype == np.float32

    def test_writes_metadata_sidecar(self, cache: EmbeddingCache) -> None:
        seed(cache)
        payload = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
        assert payload["model_name"] == "all-MiniLM-L6-v2"
        assert payload["dimension"] == 3

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "deep" / "nested" / "embeddings.npy")
        seed(cache)
        assert cache.exists()

    def test_exists_reflects_both_files(self, cache: EmbeddingCache) -> None:
        assert not cache.exists()
        seed(cache)
        assert cache.exists()
        cache.metadata_path.unlink()
        assert not cache.exists()

    def test_load_without_matrix_raises(self, cache: EmbeddingCache) -> None:
        with pytest.raises(EmbeddingCompatibilityError, match="No embedding matrix"):
            cache.load()

    def test_metadata_without_file_raises(self, cache: EmbeddingCache) -> None:
        with pytest.raises(EmbeddingCompatibilityError, match="No embedding metadata"):
            _ = cache.metadata

    def test_size_bytes(self, cache: EmbeddingCache) -> None:
        assert cache.size_bytes() == 0
        seed(cache)
        # .npy carries a 128-byte header alongside the raw float32 payload.
        assert cache.size_bytes() == 4 * 3 * 4 + 128


class TestCompleteness:
    def test_complete_when_all_rows_done(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4)
        assert cache.is_complete() is True

    def test_incomplete_when_rows_remain(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4)
        payload = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
        payload["completed_count"] = 2
        cache.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        assert cache.is_complete() is False

    def test_incomplete_when_absent(self, cache: EmbeddingCache) -> None:
        assert cache.is_complete() is False

    def test_load_complete_refuses_partial(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4)
        payload = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
        payload["completed_count"] = 1
        cache.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EmbeddingCompatibilityError, match="incomplete"):
            cache.load_complete(model_name="all-MiniLM-L6-v2", dimension=3, document_count=4)

    def test_load_complete_accepts_full(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4)
        matrix = cache.load_complete(model_name="all-MiniLM-L6-v2", dimension=3, document_count=4)
        assert matrix.shape == (4, 3)


class TestCompatibility:
    def test_accepts_matching(self, cache: EmbeddingCache) -> None:
        seed(cache)
        cache.check_compatible(
            model_name="all-MiniLM-L6-v2",
            dimension=3,
            document_count=4,
            corpus_fingerprint="abc123",
        )

    def test_rejects_different_model(self, cache: EmbeddingCache) -> None:
        seed(cache)
        with pytest.raises(EmbeddingCompatibilityError, match="model"):
            cache.check_compatible(model_name="gte-base", dimension=3, document_count=4)

    def test_rejects_different_dimension(self, cache: EmbeddingCache) -> None:
        seed(cache)
        with pytest.raises(EmbeddingCompatibilityError, match="dimension"):
            cache.check_compatible(model_name="all-MiniLM-L6-v2", dimension=768, document_count=4)

    def test_rejects_different_document_count(self, cache: EmbeddingCache) -> None:
        seed(cache)
        with pytest.raises(EmbeddingCompatibilityError, match="document_count"):
            cache.check_compatible(model_name="all-MiniLM-L6-v2", dimension=3, document_count=99)

    def test_rejects_different_fingerprint_when_strict(self, cache: EmbeddingCache) -> None:
        seed(cache)
        with pytest.raises(EmbeddingCompatibilityError, match="fingerprint"):
            cache.check_compatible(
                model_name="all-MiniLM-L6-v2",
                dimension=3,
                document_count=4,
                corpus_fingerprint="different",
                strict_fingerprint=True,
            )

    def test_tolerates_fingerprint_change_when_not_strict(self, cache: EmbeddingCache) -> None:
        # A search-only restart re-reads the corpus but adds nothing; refusing
        # the cache there would be needlessly hostile.
        seed(cache)
        cache.check_compatible(
            model_name="all-MiniLM-L6-v2",
            dimension=3,
            document_count=4,
            corpus_fingerprint="different",
            strict_fingerprint=False,
        )

    def test_rejects_future_version(self, cache: EmbeddingCache) -> None:
        seed(cache, version=CACHE_VERSION + 1)
        with pytest.raises(EmbeddingCompatibilityError, match="cache version"):
            cache.check_compatible(model_name="all-MiniLM-L6-v2", dimension=3, document_count=4)

    def test_absent_cache_is_always_compatible(self, cache: EmbeddingCache) -> None:
        cache.check_compatible(model_name="anything", dimension=1, document_count=1)

    def test_error_lists_every_mismatch(self, cache: EmbeddingCache) -> None:
        seed(cache)
        with pytest.raises(EmbeddingCompatibilityError) as excinfo:
            cache.check_compatible(model_name="gte", dimension=768, document_count=99)
        message = str(excinfo.value)
        assert "model" in message and "dimension" in message and "document_count" in message


class TestResumableBuild:
    def test_fresh_build_allocates_zeros(self, cache: EmbeddingCache) -> None:
        matrix, resume_from = cache.open_for_build(
            model_name="all-MiniLM-L6-v2", dimension=3, document_count=5
        )
        assert matrix.shape == (5, 3)
        assert resume_from == 0
        assert not matrix.any()

    def test_resumes_partial_build(self, cache: EmbeddingCache) -> None:
        # Rows 0-1 were filled before the run was interrupted; the rest is the
        # zero placeholder a fresh allocation would leave.
        partial = np.zeros((5, 3), dtype=np.float32)
        partial[:2] = 1.0
        cache.save(
            partial,
            EmbeddingMetadata(
                model_name="all-MiniLM-L6-v2",
                dimension=3,
                document_count=5,
                completed_count=2,
                corpus_fingerprint="",
            ),
        )
        matrix, resume_from = cache.open_for_build(
            model_name="all-MiniLM-L6-v2", dimension=3, document_count=5
        )
        assert resume_from == 2
        assert matrix[:2].all()
        assert not matrix[2:].any()

    def test_discards_incompatible_cache(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4, dim=3)
        matrix, resume_from = cache.open_for_build(
            model_name="a-completely-different-model", dimension=3, document_count=4
        )
        assert resume_from == 0
        assert not matrix.any()

    def test_discards_wrong_shaped_matrix(self, cache: EmbeddingCache) -> None:
        seed(cache, rows=4, dim=3)
        matrix, resume_from = cache.open_for_build(
            model_name="all-MiniLM-L6-v2", dimension=3, document_count=9
        )
        assert matrix.shape == (9, 3)
        assert resume_from == 0

    def test_flush_records_progress(self, cache: EmbeddingCache) -> None:
        matrix = np.ones((10, 4), dtype=np.float32)
        cache.flush(
            matrix,
            model_name="all-MiniLM-L6-v2",
            dimension=4,
            document_count=10,
            completed_count=6,
        )
        assert cache.is_complete() is False
        assert cache.metadata.completed_count == 6

    def test_flush_then_resume_roundtrip(self, cache: EmbeddingCache) -> None:
        cache.flush(
            np.arange(40, dtype=np.float32).reshape(10, 4),
            model_name="all-MiniLM-L6-v2",
            dimension=4,
            document_count=10,
            completed_count=10,
        )
        assert cache.is_complete() is True


class TestDelete:
    def test_delete_removes_both_files(self, cache: EmbeddingCache) -> None:
        seed(cache)
        cache.delete()
        assert not cache.exists()

    def test_delete_is_idempotent(self, cache: EmbeddingCache) -> None:
        cache.delete()
        cache.delete()
        assert not cache.exists()


class TestMetadata:
    def test_roundtrip_through_dict(self) -> None:
        original = EmbeddingMetadata(
            model_name="m", dimension=8, document_count=3, completed_count=3
        )
        assert EmbeddingMetadata.from_dict(original.to_dict()) == original

    def test_from_dict_tolerates_extra_keys(self) -> None:
        assert EmbeddingMetadata.from_dict({"model_name": "m", "unknown": 1}).model_name == "m"
