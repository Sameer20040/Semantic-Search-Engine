"""Unit tests for the semantic search service.

These run on small in-memory fixtures with a fake model, so no network and no
262 MB checkpoint. The real model is covered separately by
``tests/integration/test_encoder_real.py``.

What is deliberately *not* stubbed is FAISS and the document store: the bugs
worth catching here are exactly the seams between them -- a row resolving to the
wrong document, a rank sequence with a hole, a score whose sign is wrong -- and
stubbing those out would remove the thing under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from semantic_search.core.config import EmbeddingConfig, FaissConfig, SearchConfig
from semantic_search.core.exceptions import (
    DataError,
    EmptyQueryError,
    IndexCompatibilityError,
    IndexNotFoundError,
    QueryTooLongError,
    SearchError,
    SearchIndexError,
)
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.data.preprocessor import build_embedding_text
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.search.models import SearchResult
from semantic_search.search.semantic import SemanticSearchService

DIM = 16


class FakeModel:
    """A deterministic stand-in for SentenceTransformer.

    Embedding is a hash of the text into ``DIM`` dimensions, so "semantic"
    similarity here is really lexical similarity -- which is fine, because these
    tests assert plumbing (rows, ranks, scores, ordering), not relevance. The
    real model's semantic behaviour is asserted in the integration tests.
    """

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.max_seq_length = 0

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            # blake2b, not the builtin hash(): CPython salts string hashing per
            # process, so a hash()-based fake gives a different ranking on every
            # run. See the note in tests/unit/test_hybrid_service.py.
            digest = int.from_bytes(
                hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(), "big"
            )
            vectors[row] = [((digest >> (3 * i)) % 17) / 17.0 - 0.5 for i in range(self.dimension)]
        # Normalised, so that an inner-product search over these vectors is a
        # genuine cosine similarity and the score-direction tests below are
        # asserting something true about the implementation rather than about
        # the stand-in.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, np.where(norms == 0, 1.0, norms)).astype(np.float32)


@pytest.fixture
def store() -> DocumentStore:
    return DocumentStore(
        [
            Document(id="0", title="Astronomy", text="Stars and their orbits."),
            Document(id="1", title="Geology", text="Rocks, plates, and erosion."),
            Document(id="2", title="Biology", text="Cells, tissues, and organs."),
            Document(id="3", title="Computing", text="Algorithms and data structures."),
            Document(id="4", title="Music", text="Melody, rhythm, and harmony."),
        ]
    )


@pytest.fixture
def vectors() -> np.ndarray:
    return FakeModel().encode([f"document {i}" for i in range(5)])


@pytest.fixture
def index(vectors: np.ndarray) -> FaissIndex:
    return FaissIndex.build(
        vectors,
        FaissConfig(index_type="flat_ip"),
        embedding_model="fake-model",
    )


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder(
        EmbeddingConfig(model_name="fake-model", device="cpu"),
        device="cpu",
        model=FakeModel(),
    )


@pytest.fixture
def service(encoder: EmbeddingEncoder, index: FaissIndex, store: DocumentStore):
    return SemanticSearchService(encoder, index, store)


class TestSearch:
    def test_returns_results_for_a_normal_query(self, service: SemanticSearchService) -> None:
        results = service.search("orbital mechanics")
        assert len(results) == 5  # default_top_k
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.title for r in results)

    def test_every_result_resolves_to_a_real_document(
        self, service: SemanticSearchService, store: DocumentStore
    ) -> None:
        # The row -> document mapping is the classic silent-corruption bug.
        for result in service.search("stars", top_k=5):
            document = store[result.row]
            assert document.id == result.doc_id
            assert document.title == result.title

    def test_ranks_are_contiguous_from_zero(self, service: SemanticSearchService) -> None:
        for top_k in (1, 2, 3, 5):
            results = service.search("cells", top_k=top_k)
            assert [r.rank for r in results] == list(range(len(results)))

    def test_preserves_index_order(self, service: SemanticSearchService) -> None:
        # FAISS returns best-first for an exact index; the service must not
        # re-sort, and the scores must come out non-increasing.
        scores = [r.score for r in service.search("algorithms", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_documents_are_unique_across_results(self, service: SemanticSearchService) -> None:
        ids = [r.doc_id for r in service.search("rocks and algorithms", top_k=5)]
        assert len(ids) == len(set(ids))

    def test_deterministic_for_the_same_query(self, service: SemanticSearchService) -> None:
        first = [r.doc_id for r in service.search("melody", top_k=5)]
        second = [r.doc_id for r in service.search("melody", top_k=5)]
        assert first == second

    def test_result_scores_carry_the_retriever_name(self, service: SemanticSearchService) -> None:
        result = service.search("cells", top_k=1)[0]
        assert result.scores == {"semantic": pytest.approx(result.score)}

    def test_snippet_is_populated(self, service: SemanticSearchService) -> None:
        assert service.search("cells", top_k=1)[0].snippet

    def test_default_top_k_is_used_when_unspecified(self, service: SemanticSearchService) -> None:
        # The default is 10 but the fixture corpus has 5 documents, so the
        # observable result is "everything available".
        assert service.config.default_top_k == 10
        assert len(service.search("anything")) == service.corpus_size


class TestTopK:
    def test_top_k_one(self, service: SemanticSearchService) -> None:
        results = service.search("stars", top_k=1)
        assert len(results) == 1
        assert results[0].rank == 0

    def test_top_k_ten_on_a_five_document_corpus(self, service: SemanticSearchService) -> None:
        # Asking for more than exists is answered, not rejected.
        assert len(service.search("stars", top_k=10)) == 5

    def test_top_k_far_beyond_corpus_size(self, service: SemanticSearchService) -> None:
        assert len(service.search("stars", top_k=100_000)) == 5

    def test_top_k_is_capped_by_max_top_k(
        self, service: SemanticSearchService, encoder, index, store
    ) -> None:
        capped = SemanticSearchService(
            encoder, index, store, search_config=SearchConfig(max_top_k=2)
        )
        assert len(capped.search("stars", top_k=50)) == 2

    def test_top_k_zero_is_rejected(self, service: SemanticSearchService) -> None:
        with pytest.raises(SearchError, match="at least 1"):
            service.search("stars", top_k=0)

    def test_negative_top_k_is_rejected(self, service: SemanticSearchService) -> None:
        with pytest.raises(SearchError, match="at least 1"):
            service.search("stars", top_k=-3)

    def test_non_integer_top_k_is_rejected(self, service: SemanticSearchService) -> None:
        with pytest.raises(SearchError, match="must be an integer"):
            service.search("stars", top_k=2.5)  # type: ignore[arg-type]

    def test_bool_top_k_is_rejected(self, service: SemanticSearchService) -> None:
        # bool is an int subclass, so True would silently mean "1 result".
        with pytest.raises(SearchError, match="must be an integer"):
            service.search("stars", top_k=True)  # type: ignore[arg-type]


class TestQueryValidation:
    def test_empty_query_is_rejected(self, service: SemanticSearchService) -> None:
        with pytest.raises(EmptyQueryError):
            service.search("")

    def test_whitespace_only_query_is_rejected(self, service: SemanticSearchService) -> None:
        for blank in ("   ", "\t", "\n", "  \n\t  "):
            with pytest.raises(EmptyQueryError):
                service.search(blank)

    def test_non_string_query_is_rejected(self, service: SemanticSearchService) -> None:
        with pytest.raises(SearchError, match="must be a string"):
            service.search(None)  # type: ignore[arg-type]

    def test_overlong_query_is_rejected(self, encoder, index, store) -> None:
        service = SemanticSearchService(encoder, index, store, query_max_length=20)
        with pytest.raises(QueryTooLongError, match="maximum is 20"):
            service.search("x" * 21)

    def test_query_of_exactly_max_length_is_accepted(self, encoder, index, store) -> None:
        service = SemanticSearchService(encoder, index, store, query_max_length=20)
        assert len(service.search("x" * 20, top_k=1)) == 1

    def test_length_check_applies_after_whitespace_collapse(self, encoder, index, store) -> None:
        # 20 characters of content padded out to 5000 with spaces is not an
        # overlong query, and must not be rejected as one.
        service = SemanticSearchService(encoder, index, store, query_max_length=20)
        assert len(service.search("x" * 20 + " " * 5000, top_k=1)) == 1

    def test_internal_whitespace_is_collapsed(self, service: SemanticSearchService) -> None:
        # Otherwise "a  b" and "a b" cache separately and embed differently.
        assert service.search("orbital   mechanics", top_k=3) == service.search(
            "orbital mechanics", top_k=3
        )


class PaddedHit:
    """Stands in for a raw FAISS row that ``FaissHit`` would refuse to build.

    ``FaissHit`` deliberately rejects a negative row, so the padding case has to
    be simulated at the boundary -- which is exactly the point: the service must
    not depend on a collaborator having already filtered.
    """

    def __init__(self, row: int, score: float) -> None:
        self.row = row
        self.score = score
        self.rank = 0


class TestNegativeRowHandling:
    def test_padded_rows_never_reach_a_result(
        self, service: SemanticSearchService, store: DocumentStore
    ) -> None:
        # A -1 row would index store[-1], silently returning the last document.
        for result in service.search("anything", top_k=100):
            assert result.row >= 0
            assert 0 <= result.row < len(store)

    def test_a_retriever_returning_padding_is_filtered(
        self, service: SemanticSearchService, store: DocumentStore
    ) -> None:
        # Simulate an index type that forgets to filter FAISS's -1 padding.
        padded = [PaddedHit(-1, 0.9), PaddedHit(2, 0.4)]
        results = service._to_results(padded, query="cells")
        assert len(results) == 1
        assert results[0].row == 2
        assert store[results[0].row].id == results[0].doc_id

    def test_padded_row_does_not_displace_a_real_one(self, service: SemanticSearchService) -> None:
        # A high-scoring pad must not push a genuine match out of the list.
        padded = [PaddedHit(-1, 9.9), PaddedHit(4, 0.1)]
        assert [r.row for r in service._to_results(padded, query="cells")] == [4]

    def test_ranks_stay_contiguous_after_filtering(self, service: SemanticSearchService) -> None:
        padded = [
            PaddedHit(-1, 0.9),
            PaddedHit(1, 0.5),
            PaddedHit(-1, 0.4),
            PaddedHit(3, 0.2),
        ]
        results = service._to_results(padded, query="cells")
        assert [r.rank for r in results] == [0, 1]

    def test_all_padding_yields_no_results(self, service: SemanticSearchService) -> None:
        assert service._to_results([PaddedHit(-1, 0.1), PaddedHit(-1, 0.2)], query="x") == []


class TestScoreDirection:
    @pytest.fixture
    def document_text(self) -> str:
        # Documents are embedded via build_embedding_text, which prefixes the
        # title. Querying the bare body would not reproduce the document vector,
        # so these tests use the same representation the index was built from.
        return build_embedding_text("Astronomy", "Stars and their orbits.")

    def test_flat_ip_scores_are_cosine_similarities(
        self, encoder, store, document_text: str
    ) -> None:
        vectors = encoder.encode_documents(store)
        index = FaissIndex.build(
            vectors, FaissConfig(index_type="flat_ip"), embedding_model="fake-model"
        )
        service = SemanticSearchService(encoder, index, store)
        results = service.search(document_text, top_k=5)
        # The query is the first document's exact embedding input, so it must be
        # its own nearest neighbour at a cosine of 1.
        assert results[0].doc_id == "0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_scores_stay_within_cosine_bounds(self, service: SemanticSearchService) -> None:
        for result in service.search("cells", top_k=5):
            assert -1.0 <= result.score <= 1.0

    def test_flat_l2_scores_are_negated_to_higher_is_better(
        self, encoder, store, document_text: str
    ) -> None:
        # An L2 index reports a distance (smaller better). The service contract
        # is higher-is-better everywhere, so the sign must flip.
        vectors = encoder.encode_documents(store)
        index = FaissIndex.build(
            vectors,
            FaissConfig(index_type="flat_l2"),
            embedding_model="fake-model",
            normalized=True,
        )
        service = SemanticSearchService(encoder, index, store)
        results = service.search(document_text, top_k=5)
        # Squared L2 distance between two identical unit vectors is 0, so the
        # negated score is 0 -- the point is the ordering and the sign, not 1.0.
        assert results[0].doc_id == "0"
        assert results[0].score == pytest.approx(0.0, abs=1e-5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(score <= 0.0 for score in scores)

    def test_both_index_types_rank_identically(self, encoder, store) -> None:
        # A sanity check that the sign flip did not reorder anything.
        vectors = encoder.encode_documents(store)
        ip = FaissIndex.build(
            vectors, FaissConfig(index_type="flat_ip"), embedding_model="fake-model"
        )
        l2 = FaissIndex.build(
            vectors,
            FaissConfig(index_type="flat_l2"),
            embedding_model="fake-model",
            normalized=True,
        )
        ip_ids = [r.doc_id for r in SemanticSearchService(encoder, ip, store).search("stars")]
        l2_ids = [r.doc_id for r in SemanticSearchService(encoder, l2, store).search("stars")]
        assert ip_ids == l2_ids


class TestConstruction:
    def test_empty_store_is_reported_as_a_missing_index(self, encoder, index) -> None:
        with pytest.raises(IndexNotFoundError, match="document store is empty"):
            SemanticSearchService(encoder, index, DocumentStore([]))

    def test_store_index_size_mismatch_is_rejected(self, encoder, index, store) -> None:
        with pytest.raises(DataError, match="Rebuild the index"):
            SemanticSearchService(encoder, index, DocumentStore(list(store)[:3]))

    def test_model_mismatch_with_index_is_rejected(self, index, store) -> None:
        # A different model with the same dimension would still "work" in FAISS
        # and return nonsense. The service refuses instead.
        wrong = EmbeddingEncoder(
            EmbeddingConfig(model_name="some-other-model", device="cpu"),
            device="cpu",
            model=FakeModel(),
        )
        with pytest.raises(DataError, match="does not match the model"):
            SemanticSearchService(wrong, index, store)

    def test_dimension_mismatch_is_rejected(self, index, store) -> None:
        wrong = EmbeddingEncoder(
            EmbeddingConfig(model_name="fake-model", device="cpu"),
            device="cpu",
            model=FakeModel(dimension=DIM + 8),
        )
        with pytest.raises(DataError, match="dimension"):
            SemanticSearchService(wrong, index, store)

    def test_alignment_check_can_be_disabled(self, encoder, index, store) -> None:
        # Not safe in production, but useful when slicing a corpus for a demo.
        service = SemanticSearchService(
            encoder, index, DocumentStore(list(store)[:3]), strict_alignment=False
        )
        assert service.corpus_size == 3

    def test_info_reports_both_collaborators(self, service: SemanticSearchService) -> None:
        info = service.info()
        assert info["documents"] == 5
        assert info["index"]["dimension"] == DIM
        assert info["encoder"]["model_name"] == "fake-model"

    def test_no_model_load_at_construction(self, index, store) -> None:
        # A production service must not pay a 262 MB model load to start up;
        # only the first query should trigger it.
        lazy = EmbeddingEncoder(EmbeddingConfig(model_name="a-model-that-does-not-exist"))
        with pytest.raises(DataError, match="does not match the model"):
            SemanticSearchService(lazy, index, store)


class TestSearchResultModel:
    def test_to_dict_has_the_expected_keys(self, service: SemanticSearchService) -> None:
        payload = service.search("cells", top_k=1)[0].to_dict()
        assert set(payload) == {"doc_id", "title", "snippet", "score", "rank", "scores"}

    def test_to_dict_rounds_scores(self, service: SemanticSearchService) -> None:
        payload = service.search("cells", top_k=1)[0].to_dict()
        assert payload["score"] == round(payload["score"], 6)

    def test_from_document_builds_a_rank(self, store: DocumentStore) -> None:
        result = SearchResult.from_document(store[2], score=0.5, rank=1, row=2)
        assert (result.doc_id, result.rank, result.row) == ("2", 1, 2)

    def test_doc_index_alias_matches_row(self, store: DocumentStore) -> None:
        result = SearchResult.from_document(store[3], score=0.1, rank=0, row=3)
        assert result.doc_index == result.row

    def test_is_frozen(self, store: DocumentStore) -> None:
        result = SearchResult.from_document(store[0], score=0.1, rank=0, row=0)
        with pytest.raises((AttributeError, TypeError)):
            result.score = 9.9  # type: ignore[misc]


class TestArtifactLoading:
    def test_missing_index_raises_index_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError, match="make index"):
            FaissIndex.load(tmp_path / "nope.faiss")

    def test_index_from_another_model_is_refused_on_load(self, index, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        with pytest.raises(IndexCompatibilityError, match="Rebuild the index"):
            FaissIndex.load(path, expected_model="a-different-model")

    def test_service_survives_a_save_load_roundtrip(
        self, encoder, index, store, tmp_path: Path
    ) -> None:
        # The API loads from disk at startup; that path must give the same
        # results as the in-memory one.
        path = index.save(tmp_path / "vectors.faiss")
        reloaded = FaissIndex.load(path, expected_model="fake-model")
        before = SemanticSearchService(encoder, index, store).search("stars", top_k=5)
        after = SemanticSearchService(encoder, reloaded, store).search("stars", top_k=5)
        assert [r.doc_id for r in before] == [r.doc_id for r in after]
        assert [r.score for r in before] == pytest.approx([r.score for r in after], abs=1e-6)

    def test_metadata_sidecar_records_the_building_model(self, index, tmp_path: Path) -> None:
        path = index.save(tmp_path / "vectors.faiss")
        payload = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert payload["embedding_model"] == "fake-model"

    def test_index_type_is_preserved_across_reload(self, encoder, store, tmp_path: Path) -> None:
        vectors = encoder.encode_documents(store)
        index = FaissIndex.build(
            vectors, FaissConfig(index_type="hnsw"), embedding_model="fake-model"
        )
        path = index.save(tmp_path / "graph.faiss")
        reloaded = FaissIndex.load(path, expected_model="fake-model")
        assert reloaded.index_type == "hnsw"
        assert len(SemanticSearchService(encoder, reloaded, store).search("stars", top_k=3)) == 3

    def test_generic_index_error_is_still_raised_for_unknown_paths(self, tmp_path: Path) -> None:
        with pytest.raises(SearchIndexError):
            FaissIndex.load(tmp_path / "missing.faiss")
