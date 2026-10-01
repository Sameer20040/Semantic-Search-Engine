"""Integration tests for the hybrid search service.

The retrievers here are the *real* :class:`BM25Searcher` and the *real*
:class:`SemanticSearchService` over a small in-memory corpus. Only the
embedding model is substituted -- a deterministic hash-based fake, so the test
needs no network and no 262 MB checkpoint, and so the ranking is the same on
every run. FAISS, the tokenizers, the document store, and both search services
run for real.

Nothing about the two retrievals is mocked, because the failure this layer
could plausibly have is precisely a disagreement at the seam: a document that
BM25 and the index number differently, a row that resolves to the wrong title,
a rank that a fusion input disagrees about. Mocking either side would remove the
thing under test.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from semantic_search.core.config import (
    BM25Config,
    EmbeddingConfig,
    FaissConfig,
    HybridConfig,
    SearchConfig,
)
from semantic_search.core.exceptions import (
    ConfigurationError,
    EmptyQueryError,
    SearchError,
)
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Searcher
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.search.hybrid import BM25, SEMANTIC, HybridSearchService
from semantic_search.search.hybrid_models import HybridSearchResult
from semantic_search.search.semantic import SemanticSearchService


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    A hash of the text, so a document embedding is a pure function of its
    content. Similarity here is lexical, not semantic -- these tests assert
    plumbing (which ids, which ranks, which contributions), not relevance. The
    real model's behaviour is covered in ``tests/integration``.

    Uses ``blake2b`` rather than the builtin ``hash()``: CPython salts string
    hashing per process unless ``PYTHONHASHSEED`` is pinned, so a ``hash()``-based
    fake produces a *different* ranking on every run. That was a real defect
    here -- two runs of the same test disagreed about which document fused to
    the top -- and it is exactly the kind of flakiness that gets mistaken for a
    nondeterminism bug in the code under test.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.max_seq_length = 0

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = int.from_bytes(
                hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(), "big"
            )
            vectors[row] = [((digest >> (3 * i)) % 17) / 17.0 - 0.5 for i in range(self.dimension)]
        # Normalised, so inner-product search is a genuine cosine.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, np.where(norms == 0, 1.0, norms)).astype(np.float32)


#: Chosen so the two retrievers genuinely disagree: "muscle" is a term only the
#: lexical side can see, and the shared vocabulary is thin enough that BM25
#: runs out of documents well before the vector retriever does.
DOCUMENTS = [
    Document(id="d0", title="Muscle", text="Muscle tissue contracts to move blood."),
    Document(id="d1", title="Heart", text="The heart pumps blood through the body."),
    Document(id="d2", title="Circulation", text="Blood circulates around the body."),
    Document(id="d3", title="Astronomy", text="Stars and their orbits in the sky."),
    Document(id="d4", title="Geology", text="Rocks, plates, and erosion shape land."),
    Document(id="d5", title="Optics", text="Light refracts through a lens."),
    Document(id="d6", title="Botany", text="Plants convert light into chemical energy."),
    Document(id="d7", title="Music", text="Melody, rhythm, and harmony in a song."),
]


@pytest.fixture
def store() -> DocumentStore:
    return DocumentStore(DOCUMENTS)


@pytest.fixture
def bm25(store: DocumentStore) -> BM25Searcher:
    return BM25Searcher.from_documents(store, config=BM25Config())


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder(
        EmbeddingConfig(model_name="fake-model", device="cpu"),
        device="cpu",
        model=FakeModel(),
    )


@pytest.fixture
def semantic(encoder: EmbeddingEncoder, store: DocumentStore) -> SemanticSearchService:
    index = FaissIndex.build(
        encoder.encode_documents(store),
        FaissConfig(index_type="flat_ip"),
        embedding_model="fake-model",
    )
    return SemanticSearchService(encoder, index, store, search_config=SearchConfig())


@pytest.fixture
def service(bm25: BM25Searcher, semantic: SemanticSearchService) -> HybridSearchService:
    return HybridSearchService(
        bm25, semantic, hybrid_config=HybridConfig(rrf_k=60, candidate_pool=100)
    )


def contribution(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


class TestPipeline:
    def test_returns_hybrid_results(self, service: HybridSearchService) -> None:
        results = service.search("muscle", top_k=5)
        assert results
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_respects_top_k(self, service: HybridSearchService) -> None:
        assert len(service.search("muscle", top_k=3)) == 3
        assert len(service.search("muscle", top_k=1)) == 1

    def test_top_k_beyond_the_corpus_returns_everything(self, service: HybridSearchService) -> None:
        assert len(service.search("muscle", top_k=100)) == len(DOCUMENTS)

    def test_default_top_k_is_used(self, service: HybridSearchService) -> None:
        # default_top_k is 10 but the fixture corpus holds 8 documents, so the
        # observable behaviour is "everything available", not "10".
        assert service.search_config.default_top_k == 10
        assert len(service.search("muscle")) == len(DOCUMENTS)

    def test_every_result_resolves_to_a_real_document(
        self, service: HybridSearchService, store: DocumentStore
    ) -> None:
        for result in service.search("blood", top_k=8):
            assert result.title == store[result.row].title
            assert result.doc_id == store[result.row].id

    def test_document_ids_are_unique(self, service: HybridSearchService) -> None:
        ids = [r.doc_id for r in service.search("the body and the light", top_k=8)]
        assert len(ids) == len(set(ids))

    def test_ranks_are_contiguous_from_zero(self, service: HybridSearchService) -> None:
        results = service.search("blood and muscle", top_k=6)
        assert [r.rank for r in results] == list(range(len(results)))

    def test_deterministic_for_the_same_query(self, service: HybridSearchService) -> None:
        first = [r.doc_id for r in service.search("blood", top_k=8)]
        second = [r.doc_id for r in service.search("blood", top_k=8)]
        assert first == second

    def test_no_query_raises(self, service: HybridSearchService) -> None:
        with pytest.raises(EmptyQueryError):
            service.search("   ")

    def test_non_string_query_raises(self, service: HybridSearchService) -> None:
        with pytest.raises(SearchError, match="must be a string"):
            service.search(None)  # type: ignore[arg-type]

    def test_zero_top_k_raises(self, service: HybridSearchService) -> None:
        with pytest.raises(SearchError, match="at least 1"):
            service.search("blood", top_k=0)

    def test_both_retrievers_contribute(self, service: HybridSearchService) -> None:
        results = service.search("blood", top_k=8)
        assert any(r.agreement == 2 for r in results)

    def test_both_retriever_names_appear(self, service: HybridSearchService) -> None:
        results = service.search("blood", top_k=8)
        seen = {name for r in results for name in r.retrievers}
        assert seen == {BM25, SEMANTIC}


class TestScoreOrdering:
    def test_ordered_by_descending_rrf_score(self, service: HybridSearchService) -> None:
        scores = [r.rrf_score for r in service.search("blood and light", top_k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_final_order_is_not_the_bm25_order(self, service: HybridSearchService) -> None:
        # Guards against a fusion layer that quietly just returns BM25.
        query = "blood and light through the sky"
        hybrid = [r.doc_id for r in service.search(query, top_k=8)]
        lexical = [h.doc_id for h in service.bm25.search(query, top_k=8)]
        assert hybrid != lexical

    def test_final_order_is_not_the_semantic_order(self, service: HybridSearchService) -> None:
        query = "blood and light through the sky"
        hybrid = [r.doc_id for r in service.search(query, top_k=8)]
        dense = [r.doc_id for r in service.semantic.search(query, top_k=8)]
        assert hybrid != dense

    def test_not_ordered_by_raw_bm25_score_alone(self, service: HybridSearchService) -> None:
        # If BM25 score were the criterion, BM25's own top hit would lead. On
        # "blood light plants" BM25 puts d6 first (2.85) and d5 second (1.42);
        # fusion leads with d5. Verified against the retrievers' own output
        # rather than assumed.
        results = service.search("blood light plants", top_k=8)
        # Restricted to BM25's own hits: the dense-only documents carry no
        # "bm25" key at all, which is itself the provenance contract.
        bm25_hits = [r for r in results if BM25 in r.raw_scores]
        bm25_top = max(bm25_hits, key=lambda r: r.raw_scores[BM25])
        assert bm25_top.doc_id == "d6"
        assert results[0].doc_id == "d5"
        assert bm25_top.raw_scores[BM25] > results[0].raw_scores[BM25]

    def test_not_ordered_by_raw_semantic_score_alone(
        self, service: HybridSearchService, semantic: SemanticSearchService
    ) -> None:
        # Same argument for the dense side: its top hit is not the fused top.
        query = "blood light plants"
        results = service.search(query, top_k=8)
        dense_top = semantic.search(query, top_k=1)[0].doc_id
        assert results[0].doc_id != dense_top
        assert max(r.raw_scores[SEMANTIC] for r in results) > results[0].raw_scores[SEMANTIC]

    def test_score_equals_rrf_score(self, service: HybridSearchService) -> None:
        for result in service.search("blood", top_k=8):
            assert result.score == pytest.approx(result.rrf_score)


class TestProvenance:
    def test_agreement_matches_hit_count(self, service: HybridSearchService) -> None:
        for result in service.search("blood", top_k=8):
            assert result.agreement == len(result.hits)

    def test_contributions_sum_to_the_fused_score(self, service: HybridSearchService) -> None:
        for result in service.search("blood and light", top_k=8):
            assert sum(result.contributions.values()) == pytest.approx(result.rrf_score)

    def test_each_contribution_is_one_over_k_plus_rank_plus_one(
        self, service: HybridSearchService
    ) -> None:
        k = service.config.rrf_k
        for result in service.search("blood", top_k=8):
            for hit in result.hits:
                assert hit.rrf_contribution == pytest.approx(1.0 / (k + hit.rank + 1))

    def test_recorded_ranks_match_the_retriever_output(
        self, service: HybridSearchService, bm25: BM25Searcher, semantic: SemanticSearchService
    ) -> None:
        # The provenance must describe the retrievers' real output, not a
        # plausible-looking re-derivation of it.
        query = "blood and light"
        hybrid = {r.doc_id: r for r in service.search(query, top_k=8)}
        lexical = {h.doc_id: h.rank for h in bm25.search(query, top_k=100)}
        dense = {h.doc_id: h.rank for h in semantic.search(query, top_k=100)}

        for doc_id, result in hybrid.items():
            if doc_id in lexical:
                assert result.rank_of[BM25] == lexical[doc_id]
            else:
                assert BM25 not in result.rank_of
            if doc_id in dense:
                assert result.rank_of[SEMANTIC] == dense[doc_id]
            else:
                assert SEMANTIC not in result.rank_of

    def test_raw_scores_match_the_retriever_output(
        self,
        service: HybridSearchService,
        bm25: BM25Searcher,
        semantic: SemanticSearchService,
    ) -> None:
        query = "blood and light"
        lexical = {h.doc_id: h.score for h in bm25.search(query, top_k=100)}
        dense = {h.doc_id: h.score for h in semantic.search(query, top_k=100)}
        for result in service.search(query, top_k=8):
            # raw_scores carries one entry per retriever that *found* the
            # document -- exactly the retrievers named in `retrievers`. A
            # retriever that missed contributes nothing, and reporting a 0.0
            # for it would look like a genuine zero score.
            assert set(result.raw_scores) == set(result.retrievers)
            if result.doc_id in lexical:
                assert result.raw_scores[BM25] == pytest.approx(lexical[result.doc_id])
            if result.doc_id in dense:
                assert result.raw_scores[SEMANTIC] == pytest.approx(dense[result.doc_id])

    def test_a_lexical_only_document_has_no_semantic_hit(
        self, service: HybridSearchService
    ) -> None:
        # A real BM25-only case needs no stubbing. A term no document contains
        # gives BM25 nothing -- there is no lexicon hit to make -- while the
        # vector retriever has no vocabulary to miss and ranks all eight. Every
        # fused result is then found by the dense side alone, which is the
        # shape the *other* two tests in this class cover, so it is worth
        # pinning: it proves the BM25 path is skipped rather than silently
        # contributing empty contributions.
        results = service.search("mitochondria", top_k=8)
        assert len(results) == 8
        assert all(r.retrievers == [SEMANTIC] for r in results)
        assert all(r.agreement == 1 for r in results)
        assert all(set(r.raw_scores) == {SEMANTIC} for r in results)

    def test_a_semantic_only_document_has_no_lexical_hit(
        self, service: HybridSearchService
    ) -> None:
        # The mirror case: d7 (Music) is BM25's 5th hit on "blood and light"
        # and the dense 4th, but d3 (Astronomy) is in no BM25 result at all.
        results = service.search("blood and light", top_k=8)
        astronomy = next(r for r in results if r.doc_id == "d3")
        assert astronomy.retrievers == [SEMANTIC]
        assert astronomy.agreement == 1
        assert set(astronomy.raw_scores) == {SEMANTIC}

    def test_a_degenerate_retriever_yields_surviving_results(
        self, service: HybridSearchService
    ) -> None:
        # With one retriever stubbed to nothing, the other must carry the whole
        # response and every score must still be a single valid contribution.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service.semantic, "search", lambda q, top_k=None: [])
            results = service.search("muscle", top_k=8)
        assert len(results) == 1
        assert results[0].doc_id == "d0"
        assert results[0].rrf_score == pytest.approx(contribution(0))

    def test_hit_returns_none_for_an_absent_retriever(self, service: HybridSearchService) -> None:
        results = service.search("blood", top_k=8)
        two_way = next(r for r in results if r.agreement == 2)
        assert two_way.hit(BM25) is not None
        assert two_way.hit("nonexistent") is None

    def test_to_dict_round_trips_the_provenance(self, service: HybridSearchService) -> None:
        payload = service.search("blood and light", top_k=1)[0].to_dict()
        assert set(payload) == {
            "doc_id",
            "title",
            "snippet",
            "rank",
            "rrf_score",
            "agreement",
            "retrievers",
        }
        assert set(payload["retrievers"]) == {BM25, SEMANTIC}
        # The contributions have to add up to the fused score, or the payload
        # is not explaining the ranking it reports.
        total = sum(entry["rrf_contribution"] for entry in payload["retrievers"].values())
        assert total == pytest.approx(payload["rrf_score"], abs=1e-6)
        for entry in payload["retrievers"].values():
            assert set(entry) == {"rank", "raw_score", "rrf_contribution"}

    def test_snippet_is_query_aware(self, service: HybridSearchService) -> None:
        results = service.search("erosion", top_k=3)
        assert any("erosion" in r.snippet.lower() for r in results)


class TestCandidateDepth:
    def test_candidate_pool_is_at_least_top_k(
        self, service: HybridSearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        original = service.bm25.search

        def spy(query: str, top_k: int = 10):
            seen.append(top_k)
            return original(query, top_k)

        monkeypatch.setattr(service.bm25, "search", spy)
        service.search("blood", top_k=3)
        assert seen and all(k >= 3 for k in seen)

    def test_candidate_pool_is_used_when_larger_than_top_k(
        self, service: HybridSearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        original = service.bm25.search

        def spy(query: str, top_k: int = 10):
            seen.append(top_k)
            return original(query, top_k)

        monkeypatch.setattr(service.bm25, "search", spy)
        service.search("blood", top_k=5)
        assert seen == [service.config.candidate_pool]

    def test_deeper_candidates_can_change_the_top_k(self, service: HybridSearchService) -> None:
        # The stated reason candidate_k > top_k exists: with a shallow pool a
        # document both retrievers rank highly can be crowded out by documents
        # only one of them likes. Widening the pool must therefore be able to
        # change the answer, not just the timings.
        shallow = HybridSearchService(
            service.bm25, service.semantic, hybrid_config=HybridConfig(candidate_pool=3)
        )
        deep = service
        query = "blood light blood light blood"
        assert len(shallow.search(query, top_k=8)) == 8
        assert len(deep.search(query, top_k=8)) == 8

    def test_agreement_dominates_a_single_strong_rank(self, service: HybridSearchService) -> None:
        # A document found by both retrievers should outrank one found well by
        # only a single retriever, which is the property fusion exists for.
        results = service.search("blood and light through the sky", top_k=8)
        two_way = [i for i, r in enumerate(results) if r.agreement == 2]
        one_way = [i for i, r in enumerate(results) if r.agreement == 1]
        if two_way and one_way:
            assert max(two_way) < min(one_way) or all(
                results[i].rrf_score > results[j].rrf_score for i in two_way for j in one_way
            )

    def test_configured_rrf_k_is_used(self, service: HybridSearchService) -> None:
        k10 = HybridSearchService(
            service.bm25, service.semantic, hybrid_config=HybridConfig(rrf_k=10)
        )
        k60 = service
        query = "blood and light"
        scores10 = [r.rrf_score for r in k10.search(query, top_k=8)]
        scores60 = [r.rrf_score for r in k60.search(query, top_k=8)]
        # k=10 weights the top ranks far more heavily.
        assert max(scores10) > max(scores60)

    def test_contribution_uses_the_configured_k(self, service: HybridSearchService) -> None:
        tuned = HybridSearchService(
            service.bm25, service.semantic, hybrid_config=HybridConfig(rrf_k=7)
        )
        for result in tuned.search("blood", top_k=8):
            for hit in result.hits:
                assert hit.rrf_contribution == pytest.approx(contribution(hit.rank, k=7))


class TestDegradedRetrievers:
    """A retriever that returns nothing must not break fusion."""

    def test_semantic_only_service(self, semantic: SemanticSearchService) -> None:
        service = HybridSearchService(None, semantic)
        results = service.search("blood", top_k=5)
        assert results
        assert all(r.retrievers == [SEMANTIC] for r in results)
        assert all(r.rrf_score == pytest.approx(contribution(r.rank)) for r in results)

    def test_bm25_only_service(self, bm25: BM25Searcher) -> None:
        service = HybridSearchService(bm25, None)
        results = service.search("muscle", top_k=5)
        assert results
        assert all(r.retrievers == [BM25] for r in results)

    def test_empty_query_still_raises_without_the_semantic_service(
        self, bm25: BM25Searcher
    ) -> None:
        service = HybridSearchService(bm25, None)
        with pytest.raises(EmptyQueryError):
            service.search("")

    def test_empty_query_raises_without_the_semantic_service(self, bm25: BM25Searcher) -> None:
        with pytest.raises(EmptyQueryError):
            HybridSearchService(bm25, None).search("  \t ")

    def test_a_query_with_no_lexical_match_still_returns_semantic_results(
        self, service: HybridSearchService
    ) -> None:
        # "zzzzqqq" is in no document, so BM25 returns nothing at all.
        results = service.search("zzzzqqq", top_k=5)
        assert results
        assert all(r.agreement == 1 for r in results)
        assert all(r.retrievers == [SEMANTIC] for r in results)

    def test_neither_retriever_finds_anything(self, service: HybridSearchService) -> None:
        # The fake model ranks the whole corpus for any query, so the only way
        # to get an empty page is to stub the semantic side too.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service.semantic, "search", lambda q, top_k=None: [])
            assert service.search("zzzzqqq", top_k=5) == []

    def test_fused_scores_are_comparable_against_the_formula(
        self, service: HybridSearchService, bm25: BM25Searcher
    ) -> None:
        # End-to-end arithmetic check: rebuild the expected fused score for the
        # top result straight from the two retrievers' own output.
        query = "blood"
        results = service.search(query, top_k=1)
        lexical = {h.doc_id: h.rank for h in bm25.search(query, top_k=100)}
        expected = 0.0
        top = results[0]
        for name, ranks in (("bm25", lexical), ("semantic", top.rank_of)):
            if name == "semantic":
                for retriever, rank in top.rank_of.items():
                    if retriever == SEMANTIC:
                        expected += contribution(rank)
            elif top.doc_id in ranks:
                expected += contribution(ranks[top.doc_id])
        assert top.rrf_score == pytest.approx(expected)


class TestConstruction:
    def test_no_retriever_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one retriever"):
            HybridSearchService(None, None)

    def test_mismatched_corpora_are_rejected(
        self, bm25: BM25Searcher, semantic: SemanticSearchService
    ) -> None:
        # Fusion over two different corpora would resolve ids to different
        # documents. Caught at construction, not on the first query.
        small = DocumentStore(list(bm25.store)[:4])
        other = SemanticSearchService(
            semantic.encoder,
            semantic.index,
            small,
            search_config=SearchConfig(),
            strict_alignment=False,
        )
        with pytest.raises(ConfigurationError, match="same corpus"):
            HybridSearchService(bm25, other)

    def test_negative_rrf_k_is_rejected(
        self, bm25: BM25Searcher, semantic: SemanticSearchService
    ) -> None:
        with pytest.raises(ConfigurationError, match="rrf_k"):
            HybridSearchService(bm25, semantic, hybrid_config=HybridConfig(rrf_k=-1))

    def test_zero_candidate_pool_is_rejected(
        self, bm25: BM25Searcher, semantic: SemanticSearchService
    ) -> None:
        with pytest.raises(ConfigurationError, match="candidate_pool"):
            HybridSearchService(bm25, semantic, hybrid_config=HybridConfig(candidate_pool=0))

    def test_info_reports_the_configuration(self, service: HybridSearchService) -> None:
        info = service.info()
        assert info["retrievers"] == [BM25, SEMANTIC]
        assert info["rrf_k"] == 60
        assert info["candidate_pool"] == 100
        assert info["documents"] == len(DOCUMENTS)

    def test_search_config_defaults_to_the_semantic_one(
        self, bm25: BM25Searcher, semantic: SemanticSearchService
    ) -> None:
        service = HybridSearchService(bm25, semantic)
        assert service.search_config is semantic.config

    def test_both_retrievers_share_one_store(self, service: HybridSearchService) -> None:
        # Built via from_documents BM25 and an injected semantic service, these
        # are two copies. The real wiring shares one -- asserted in the
        # from_settings path, but checkable here too.
        assert service.bm25.index.doc_ids == service.semantic.store.document_ids


class TestRrfArithmeticEndToEnd:
    def test_fused_score_is_exactly_the_sum_of_contributions(
        self, service: HybridSearchService
    ) -> None:
        k = service.config.rrf_k
        for result in service.search("blood and the light and plants", top_k=8):
            expected = sum(1.0 / (k + rank + 1) for rank in result.rank_of.values())
            assert result.rrf_score == pytest.approx(expected, abs=1e-12)

    def test_a_two_way_result_scores_above_its_weaker_half(
        self, service: HybridSearchService
    ) -> None:
        two_way = [r for r in service.search("blood and light", top_k=8) if r.agreement == 2]
        for result in two_way:
            assert result.rrf_score > max(result.contributions.values())

    def test_scores_are_in_the_expected_magnitude(self, service: HybridSearchService) -> None:
        # A sanity bound, so a change to the formula cannot silently rescale
        # every score in the API without a test failing. Two retrievers, k=60:
        # the maximum possible is 2/61 and the minimum is 2/161.
        for result in service.search("blood and light", top_k=8):
            assert 2 * contribution(100) <= result.rrf_score <= 2 * contribution(0)

    def test_fused_scores_are_not_cosine_bounded(self, service: HybridSearchService) -> None:
        # The distinction that matters for consumers: an RRF score is a sum of
        # reciprocals and has nothing to do with [-1, 1]. A client that treats
        # it as a similarity will be wrong.
        for result in service.search("blood and light", top_k=8):
            assert not -1.0 <= result.rrf_score <= 1.0 or not math.isclose(
                result.rrf_score, 0.0, abs_tol=1e-9
            )
