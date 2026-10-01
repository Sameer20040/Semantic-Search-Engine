"""Unit tests for the FAISS vector index.

Two failure modes matter more than anything else here, and most of these tests
target one of them:

* **Wrong results that look right.** A row/score mix-up, or an off-by-one in the
  rank sequence, produces plausible rankings. Ranks are checked for contiguity
  because RRF consumes them positionally.
* **A stale index silently reused.** A FAISS file built by a different model, or
  from a different corpus, is just a list of meaningless numbers. `load()` must
  refuse it rather than serve nonsense.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from semantic_search.core.config import FaissConfig
from semantic_search.core.exceptions import CorruptIndexError, SearchIndexError
from semantic_search.index.faiss_index import (
    FAISS_INDEX_VERSION,
    FaissHit,
    FaissIndex,
    normalize_rows,
)

faiss = pytest.importorskip("faiss")

DIM = 8


@pytest.fixture
def matrix() -> np.ndarray:
    """Well-separated unit vectors so rankings are unambiguous."""
    rng = np.random.default_rng(1234)
    return normalize_rows(rng.standard_normal((20, DIM), dtype=np.float32))


@pytest.fixture
def index(matrix: np.ndarray) -> FaissIndex:
    return FaissIndex.build(matrix, FaissConfig(index_type="flat_ip"), embedding_model="test-model")


class TestBuild:
    def test_reports_size_and_dimension(self, index: FaissIndex) -> None:
        assert len(index) == 20
        assert index.dimension == DIM
        assert index.index_type == "flat_ip"

    def test_result_ranks_itself_first(self, index: FaissIndex, matrix: np.ndarray) -> None:
        # Every unit vector is its own nearest neighbour, which is the one
        # ranking guarantee an exact inner-product index must always give.
        for row in range(len(index)):
            assert index.search(matrix[row], top_k=1)[0].row == row

    def test_ranks_are_contiguous_from_zero(self, index: FaissIndex, matrix: np.ndarray) -> None:
        hits = index.search(matrix[3], top_k=10)
        assert [hit.rank for hit in hits] == list(range(len(hits)))

    def test_scores_descend(self, index: FaissIndex, matrix: np.ndarray) -> None:
        scores = [hit.score for hit in index.search(matrix[1], top_k=15)]
        assert scores == sorted(scores, reverse=True)

    def test_hits_are_unique(self, index: FaissIndex, matrix: np.ndarray) -> None:
        rows = [hit.row for hit in index.search(matrix[7], top_k=10)]
        assert len(rows) == len(set(rows))

    def test_self_similarity_is_about_one(self, index: FaissIndex, matrix: np.ndarray) -> None:
        # Confirms the vectors really are normalised, which is what makes the
        # inner product a cosine similarity.
        assert index.search(matrix[0], top_k=1)[0].score == pytest.approx(1.0, abs=1e-5)

    def test_ranks_rows_consistently_across_queries(
        self, index: FaissIndex, matrix: np.ndarray
    ) -> None:
        # The same query must always produce the same ordering.
        first = [hit.row for hit in index.search(matrix[5], top_k=8)]
        second = [hit.row for hit in index.search(matrix[5], top_k=8)]
        assert first == second

    def test_build_ignores_nonsquare_slicing(self, matrix: np.ndarray) -> None:
        # A transposed matrix is a 2-D array too; the shape guard must catch it
        # via the dimension it records, not by luck.
        built = FaissIndex.build(np.ascontiguousarray(matrix.T))
        assert built.dimension == 20
        assert len(built) == DIM

    def test_coerces_float64_input_to_float32(self, matrix: np.ndarray) -> None:
        # FAISS only stores float32, and a caller handing over a float64 matrix
        # (easy to get from a sklearn pipeline) must not hit a cryptic error.
        built = FaissIndex.build(matrix.astype(np.float64))
        assert built.dimension == DIM
        assert len(built) == 20
        assert built.search(matrix[0].astype(np.float64), top_k=1)[0].row == 0

    def test_records_build_metadata(self, index: FaissIndex) -> None:
        assert index.metadata.version == FAISS_INDEX_VERSION
        assert index.metadata.embedding_model == "test-model"
        assert index.metadata.document_count == 20
        assert index.metadata.build_seconds >= 0.0
        assert index.metadata.created_at


class TestBuildGuards:
    def test_config_rejects_unknown_index_type(self) -> None:
        # pydantic validates the literal before build() ever runs, so an
        # operator typo in the YAML fails at config load, not mid-build.
        with pytest.raises(ValidationError):
            FaissConfig(index_type="nonsense")  # type: ignore[arg-type]

    def test_build_rejects_unknown_index_type_from_an_unvalidated_config(
        self, matrix: np.ndarray
    ) -> None:
        # Defence in depth: a config object that bypassed validation still must
        # not reach faiss.IndexFlatIP with a name it does not recognise.
        from pydantic import ValidationError as _ValidationError

        try:
            bad = FaissConfig(index_type="flat_ip")
        except _ValidationError:  # pragma: no cover - config shape changed
            pytest.skip("FaissConfig no longer accepts a flat_ip literal")
        object.__setattr__(bad, "index_type", "nonsense")
        with pytest.raises(SearchIndexError, match="Unknown FAISS index type"):
            FaissIndex.build(matrix, bad)

    def test_rejects_one_dimensional_input(self) -> None:
        with pytest.raises(SearchIndexError, match="2-D"):
            FaissIndex.build(np.zeros(10, dtype=np.float32))

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(SearchIndexError, match="zero documents"):
            FaissIndex.build(np.zeros((0, DIM), dtype=np.float32))

    def test_rejects_nan_vectors(self, matrix: np.ndarray) -> None:
        matrix = matrix.copy()
        matrix[4, 0] = np.nan
        with pytest.raises(SearchIndexError, match="NaN or infinity"):
            FaissIndex.build(matrix)

    def test_rejects_infinite_vectors(self, matrix: np.ndarray) -> None:
        matrix = matrix.copy()
        matrix[2, 3] = np.inf
        with pytest.raises(SearchIndexError, match="NaN or infinity"):
            FaissIndex.build(matrix)

    def test_normalises_when_caller_forgets_to(self, matrix: np.ndarray) -> None:
        # Un-normalised input built as flat_ip must still be searchable by a
        # unit query, otherwise the score silently becomes a dot product.
        raw = matrix * 7.0
        built = FaissIndex.build(raw, FaissConfig(index_type="flat_ip"), normalized=False)
        assert built.search(matrix[0], top_k=1)[0].score == pytest.approx(1.0, abs=1e-4)

    def test_ivf_reports_why_it_is_not_available(self, matrix: np.ndarray) -> None:
        with pytest.raises(SearchIndexError, match="requires a training sample"):
            FaissIndex.build(matrix, FaissConfig(index_type="ivf_flat"))


class TestSearch:
    def test_top_k_larger_than_index_is_clamped(self, index: FaissIndex) -> None:
        # Asking for 1000 neighbours from a 20-vector index is a user error, not
        # an FAISS crash.
        assert len(index.search(np.ones(DIM, dtype=np.float32), top_k=1000)) == 20

    def test_top_k_zero_returns_nothing(self, index: FaissIndex) -> None:
        assert index.search(np.ones(DIM, dtype=np.float32), top_k=0) == []

    def test_negative_top_k_returns_nothing(self, index: FaissIndex) -> None:
        assert index.search(np.ones(DIM, dtype=np.float32), top_k=-5) == []

    def test_rejects_wrong_query_dimension(self, index: FaissIndex) -> None:
        with pytest.raises(SearchIndexError, match="different models"):
            index.search(np.ones(384, dtype=np.float32))

    def test_accepts_row_vector_or_matrix(self, index: FaissIndex, matrix: np.ndarray) -> None:
        as_row = index.search(matrix[2], top_k=5)
        as_batch = index.search(matrix[2].reshape(1, -1), top_k=5)
        assert [hit.row for hit in as_row] == [hit.row for hit in as_batch]

    def test_search_batch_returns_one_result_list_per_query(
        self, index: FaissIndex, matrix: np.ndarray
    ) -> None:
        results = index.search_batch(matrix[:4], top_k=3)
        assert len(results) == 4
        for offset, hits in enumerate(results):
            assert [hit.row for hit in hits] == [
                hit.row for hit in index.search(matrix[offset], top_k=3)
            ]

    def test_search_batch_over_empty_input(self, index: FaissIndex) -> None:
        assert index.search_batch(np.zeros((0, DIM), dtype=np.float32)) == []

    def test_search_batch_rejects_1d_input(self, index: FaissIndex) -> None:
        with pytest.raises(SearchIndexError, match="2-D"):
            index.search_batch(np.zeros(DIM, dtype=np.float32))

    def test_flat_l2_orders_nearest_first(self, matrix: np.ndarray) -> None:
        # For an L2 index, score is a distance: smaller is better, and the hits
        # must still come back best-first. Fusion flips the sign downstream.
        built = FaissIndex.build(matrix, FaissConfig(index_type="flat_l2"), normalized=True)
        hits = built.search(matrix[6], top_k=6)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores)
        assert hits[0].row == 6

    def test_hnsw_finds_the_exact_nearest_neighbour(self, matrix: np.ndarray) -> None:
        # HNSW is approximate, but at n=20 it should not lose the self-match.
        built = FaissIndex.build(matrix, FaissConfig(index_type="hnsw"), normalized=True)
        assert built.search(matrix[6], top_k=1)[0].row == 6

    def test_hnsw_agrees_with_exact_search_on_this_corpus(self, matrix: np.ndarray) -> None:
        exact = FaissIndex.build(matrix, FaissConfig(index_type="flat_ip"), normalized=True)
        approx = FaissIndex.build(matrix, FaissConfig(index_type="hnsw"), normalized=True)
        for row in range(20):
            assert (
                approx.search(matrix[row], top_k=1)[0].row
                == exact.search(matrix[row], top_k=1)[0].row
            )


class TestFaissHit:
    def test_rejects_negative_row(self) -> None:
        # FAISS pads with -1; if that ever reached a FaissHit it would index
        # from the end of the document store.
        with pytest.raises(SearchIndexError, match="invalid row id"):
            FaissHit(row=-1, score=0.5)

    def test_accepts_zero_row(self) -> None:
        assert FaissHit(row=0, score=1.0).row == 0

    def test_is_frozen(self) -> None:
        hit = FaissHit(row=1, score=0.5)
        with pytest.raises((AttributeError, TypeError)):
            hit.score = 0.9  # type: ignore[misc]


class TestPersistence:
    def test_roundtrip_preserves_exact_ranking(
        self, index: FaissIndex, matrix: np.ndarray, tmp_path: Path
    ) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        reloaded = FaissIndex.load(path)
        for row in range(20):
            assert [hit.row for hit in reloaded.search(matrix[row], top_k=5)] == [
                hit.row for hit in index.search(matrix[row], top_k=5)
            ]

    def test_roundtrip_preserves_scores(self, index: FaissIndex, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        reloaded = FaissIndex.load(path)
        query = np.ones(DIM, dtype=np.float32)
        assert [hit.score for hit in reloaded.search(query, top_k=5)] == pytest.approx(
            [hit.score for hit in index.search(query, top_k=5)], abs=1e-6
        )

    def test_save_creates_parent_directory(self, index: FaissIndex, tmp_path: Path) -> None:
        path = index.save(tmp_path / "deep" / "nested" / "vectors.faiss")
        assert path.is_file()

    def test_save_writes_metadata_sidecar(self, index: FaissIndex, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        payload = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert payload["version"] == FAISS_INDEX_VERSION
        assert payload["dimension"] == DIM
        assert payload["document_count"] == 20
        assert payload["embedding_model"] == "test-model"

    def test_load_missing_file_says_how_to_fix_it(self, tmp_path: Path) -> None:
        with pytest.raises(SearchIndexError, match="make index"):
            FaissIndex.load(tmp_path / "absent.faiss")

    def test_load_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.faiss"
        path.write_bytes(b"this is not a faiss index")
        with pytest.raises(CorruptIndexError, match="Could not read"):
            FaissIndex.load(path)

    def test_rejects_index_built_with_another_model(
        self, index: FaissIndex, tmp_path: Path
    ) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        with pytest.raises(SearchIndexError, match="Rebuild the index"):
            FaissIndex.load(path, expected_model="gte-base")

    def test_accepts_index_built_with_the_expected_model(
        self, index: FaissIndex, tmp_path: Path
    ) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        assert len(FaissIndex.load(path, expected_model="test-model")) == 20

    def test_rejects_document_count_mismatch(self, index: FaissIndex, tmp_path: Path) -> None:
        # The index and the document store coming from different corpora is the
        # worst realistic way to get wrong-but-confident results.
        path = index.save(tmp_path / "vectors.faiss")
        with pytest.raises(SearchIndexError, match="document store has"):
            FaissIndex.load(path, expected_documents=999)

    def test_rejects_future_version(self, index: FaissIndex, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        meta = path.with_suffix(".meta.json")
        payload = json.loads(meta.read_text(encoding="utf-8"))
        payload["version"] = FAISS_INDEX_VERSION + 1
        meta.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SearchIndexError, match="version"):
            FaissIndex.load(path)

    def test_rejects_unreadable_metadata(self, index: FaissIndex, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        path.with_suffix(".meta.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CorruptIndexError, match="unreadable"):
            FaissIndex.load(path)

    def test_missing_sidecar_is_used_but_flagged(
        self, index: FaissIndex, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An index with no provenance is still usable, but the operator should
        # be told their safety check is not running.
        path = index.save(tmp_path / "vectors.faiss")
        path.with_suffix(".meta.json").unlink()
        with caplog.at_level("WARNING", logger="semantic_search.index.faiss_index"):
            reloaded = FaissIndex.load(path)
        assert len(reloaded) == 20
        assert reloaded.metadata.embedding_model == ""
        assert "faiss.metadata_missing" in caplog.text

    def test_persisted_index_refuses_mismatched_query(self, tmp_path: Path) -> None:
        # A reloaded index must enforce the same dimension rule as a fresh one;
        # otherwise a query embedded by a different model is silently truncated
        # or errors deep inside FAISS.
        index = FaissIndex.build(normalize_rows(np.ones((3, DIM), dtype=np.float32)))
        path = index.save(tmp_path / "vectors.faiss")
        reloaded = FaissIndex.load(path)
        with pytest.raises(SearchIndexError, match="different models"):
            reloaded.search(np.ones(DIM + 1, dtype=np.float32))


class TestStats:
    def test_reports_index_shape(self, index: FaissIndex) -> None:
        stats = index.stats()
        assert stats["documents"] == 20
        assert stats["dimension"] == DIM
        assert stats["index_type"] == "flat_ip"
        assert stats["embedding_model"] == "test-model"
        assert stats["trained"] is True

    def test_memory_figure_matches_vector_payload(self, index: FaissIndex) -> None:
        assert index.stats()["memory_bytes"] == 20 * DIM * 4

    def test_faiss_index_property_exposes_underlying_object(self, index: FaissIndex) -> None:
        assert index.faiss_index.ntotal == 20


class TestNormalizeRows:
    def test_produces_unit_rows(self) -> None:
        out = normalize_rows(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

    def test_leaves_a_zero_row_intact(self) -> None:
        # A zero vector has no direction; normalising would divide by zero and
        # produce NaN that then poisons every search.
        out = normalize_rows(np.array([[0.0, 0.0]], dtype=np.float32))
        assert np.isfinite(out).all()
        assert not out.any()

    def test_does_not_mutate_the_input(self) -> None:
        original = np.array([[3.0, 4.0]], dtype=np.float32)
        normalize_rows(original)
        assert original.tolist() == [[3.0, 4.0]]

    def test_preserves_direction(self) -> None:
        out = normalize_rows(np.array([[3.0, 4.0]], dtype=np.float32))
        assert out[0][1] / out[0][0] == pytest.approx(4 / 3, rel=1e-5)


class TestTrain:
    def test_train_is_a_noop_on_a_trained_index(self, matrix: np.ndarray) -> None:
        index = FaissIndex.build(matrix, FaissConfig(index_type="hnsw"), normalized=True)
        FaissIndex.train(index.faiss_index, matrix)
        assert len(index) == 20
