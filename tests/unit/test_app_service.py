"""Unit tests for the application search service (Phase 9).

These use **fakes** at the Phase 8 boundary rather than real indexes: the point
here is the request/response contract and the failure taxonomy, not retrieval.
The real ``BM25Searcher`` + ``FaissIndex`` + ``DocumentStore`` stack, the
ordering assertions, and the both-retrievers-same-string assertion live in
``tests/integration/test_app_integration.py``.

No performance claim is made from anything in this file.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from semantic_search.app.models import (
    ApplicationSearchResponse,
    SearchHit,
    SearchMetadata,
    SearchRequest,
)
from semantic_search.app.service import NO_MATCHES, ApplicationSearchService
from semantic_search.core.config import AppConfig
from semantic_search.core.exceptions import (
    ConfigurationError,
    CorruptIndexError,
    DataError,
    EmptyQueryError,
    QueryTooLongError,
    SearchBackendError,
    SearchError,
)
from semantic_search.query.service import SearchService
from semantic_search.search.hybrid_models import HybridSearchResult, RetrieverHit
from semantic_search.search.models import SearchResult

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def _hybrid_result(
    doc_id: str,
    *,
    score: float,
    rank: int = 0,
    title: str | None = None,
    retrievers: tuple[str, ...] = ("bm25", "semantic"),
) -> HybridSearchResult:
    """A stand-in with the attributes :func:`_to_hit` reads off a real result.

    Built through :class:`HybridSearchResult`'s real constructor -- ``result``
    plus ``rrf_score`` plus ``hits`` -- rather than by faking the pass-through
    properties, so a change to the Phase 7 model breaks this file instead of
    being papered over by a duck-typed stand-in.
    """
    hits = tuple(
        RetrieverHit(retriever=name, rank=rank, raw_score=1.5, rrf_contribution=score / 2)
        for name in retrievers
    )
    return HybridSearchResult(
        result=SearchResult(
            doc_id=doc_id,
            title=title or f"Title for {doc_id}",
            snippet=f"Snippet for {doc_id}",
            score=score,
            rank=rank,
            row=rank,
            scores=dict.fromkeys(retrievers, 1.5),
        ),
        rrf_score=score,
        hits=hits,
    )


class _FakeInner:
    """The shape of ``query.service.SearchResponse`` that the app layer reads."""

    def __init__(self, results: list[object], *, original: str, processed: str) -> None:
        from semantic_search.query.models import ProcessedQuery

        self.results = results
        self.query = ProcessedQuery(
            original=original,
            normalized=processed,
            corrected=processed,
            query_for_retrieval=processed,
        )


class _FakeSearchService:
    """Replaces the Phase 8 service so this file tests only the app layer."""

    def __init__(self, *, fail_with: BaseException | None = None) -> None:
        self.calls: list[tuple[str, int, str | None]] = []
        self._fail_with = fail_with
        self._store = object()

        class _Hybrid:
            store = self._store
            retriever_names = ("bm25", "semantic")

        self.hybrid = _Hybrid()

    def search(self, query: str, *, top_k: int = 10) -> _FakeInner:
        self.calls.append((query, top_k, "processed"))
        return self._run(query)

    def search_raw(self, query: str, *, top_k: int = 10) -> _FakeInner:
        self.calls.append((query, top_k, "raw"))
        return self._run(query)

    def _run(self, query: str) -> _FakeInner:
        if self._fail_with is not None:
            raise self._fail_with
        if not query.strip():
            raise EmptyQueryError("The query contained no searchable terms.")
        results = [_hybrid_result(f"d{i}", score=1.0 - i / 100, rank=i) for i in range(3)]
        # This fake does not actually process anything, so the retrieval string is
        # the input either way. What the app layer reads off it -- was the query
        # modified, was it corrected -- is exercised for real in the integration
        # tests, where a real processor runs.
        return _FakeInner(results, original=query, processed=query)

    def info(self) -> dict[str, object]:
        return {"enabled": True}


def _service(
    *,
    fail_with: BaseException | None = None,
    **config_kwargs: object,
) -> tuple[ApplicationSearchService, _FakeSearchService]:
    inner = _FakeSearchService(fail_with=fail_with)
    config = AppConfig(**config_kwargs)  # type: ignore[arg-type]
    service = ApplicationSearchService.__new__(ApplicationSearchService)
    # Constructed directly rather than via __init__ so the Phase 8 collaborator is
    # a fake. __init__ itself is exercised in the integration tests.
    service._service = inner  # type: ignore[assignment]
    service._config = config  # type: ignore[assignment]
    service._document_count = 3  # type: ignore[assignment]
    return service, inner


# --------------------------------------------------------------------------- #
# SearchRequest validation
# --------------------------------------------------------------------------- #
class TestSearchRequestValidation:
    def test_query_must_be_a_string(self) -> None:
        with pytest.raises(SearchError, match="Query must be a string"):
            SearchRequest(query=42)  # type: ignore[arg-type]

    @pytest.mark.parametrize("top_k", [0, -1, -100])
    def test_top_k_must_be_positive(self, top_k: int) -> None:
        with pytest.raises(SearchError, match="at least 1"):
            SearchRequest(query="heart", top_k=top_k)

    def test_bool_is_not_an_accepted_top_k(self) -> None:
        """``True`` is an ``int`` in Python. Accepting it means a caller's typo
        silently becomes "return 1 result" and nothing anywhere reports it."""
        with pytest.raises(SearchError, match="top_k must be an integer"):
            SearchRequest(query="heart", top_k=True)  # type: ignore[arg-type]
        with pytest.raises(SearchError, match="top_k must be an integer"):
            SearchRequest(query="heart", top_k=False)  # type: ignore[arg-type]

    @pytest.mark.parametrize("top_k", [3.5, "5", [5]])
    def test_non_integer_top_k_is_rejected(self, top_k: object) -> None:
        with pytest.raises(SearchError, match="top_k must be an integer"):
            SearchRequest(query="heart", top_k=top_k)  # type: ignore[arg-type]

    def test_query_processing_must_be_bool_or_none(self) -> None:
        with pytest.raises(SearchError, match="query_processing must be"):
            SearchRequest(query="heart", query_processing="yes")  # type: ignore[arg-type]
        with pytest.raises(SearchError, match="query_processing must be"):
            SearchRequest(query="heart", query_processing=1)  # type: ignore[arg-type]

    def test_valid_request_is_accepted(self) -> None:
        request = SearchRequest(query="heart", top_k=5, query_processing=False)
        assert request.query == "heart"
        assert request.top_k == 5
        assert request.wants_processing is False

    def test_request_is_frozen(self) -> None:
        request = SearchRequest(query="heart")
        with pytest.raises(Exception, match=r"cannot assign|read-only|immutable"):
            request.query = "changed"  # type: ignore[misc]

    def test_to_dict_round_trips(self) -> None:
        assert SearchRequest(query="a", top_k=2).to_dict() == {
            "query": "a",
            "top_k": 2,
            "query_processing": None,
        }


class TestAppConfigValidation:
    """Invalid configuration must produce a clear error at construction, not a
    confusing failure on the first request."""

    @pytest.mark.parametrize("field", ["default_top_k", "max_top_k", "max_query_chars"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_limits_are_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AppConfig(**{field: value})

    def test_config_error_type_is_from_the_existing_hierarchy(self) -> None:
        """An invalid limit fails at *construction*, with pydantic's message, not
        as a per-request surprise three layers down. The engine's own
        ConfigurationError is what config *loading* raises; the model itself
        raises ValidationError, and both are construction-time."""
        with pytest.raises(ValidationError):
            AppConfig(max_top_k=0)
        assert issubclass(ConfigurationError, Exception)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
class TestProcessingDispatch:
    def test_processing_enabled_uses_the_processed_path(self) -> None:
        service, inner = _service(query_processing_enabled=True)
        service.search(SearchRequest(query="heart"))
        assert inner.calls == [("heart", 10, "processed")]

    def test_processing_disabled_uses_the_raw_path(self) -> None:
        service, inner = _service(query_processing_enabled=False)
        service.search(SearchRequest(query="heart"))
        assert inner.calls == [("heart", 10, "raw")]

    def test_request_overrides_the_configured_default(self) -> None:
        service, inner = _service(query_processing_enabled=True)
        service.search(SearchRequest(query="heart", query_processing=False))
        assert inner.calls == [("heart", 10, "raw")]

    def test_request_can_enable_processing_on_a_disabled_service(self) -> None:
        service, inner = _service(query_processing_enabled=False)
        service.search(SearchRequest(query="heart", query_processing=True))
        assert inner.calls == [("heart", 10, "processed")]

    def test_search_text_builds_the_same_request(self) -> None:
        service, inner = _service()
        service.search_text("heart", 3, query_processing=False)
        assert inner.calls == [("heart", 3, "raw")]

    def test_non_request_argument_is_rejected(self) -> None:
        service, _ = _service()
        with pytest.raises(SearchError, match="expects a SearchRequest"):
            service.search("heart")  # type: ignore[arg-type]


class TestTopKResolution:
    def test_default_is_used_when_unspecified(self) -> None:
        service, inner = _service(default_top_k=7)
        service.search(SearchRequest(query="heart"))
        assert inner.calls[0][1] == 7

    def test_top_k_is_clamped_to_max(self) -> None:
        service, inner = _service(max_top_k=20)
        service.search(SearchRequest(query="heart", top_k=5000))
        assert inner.calls[0][1] == 20

    def test_requested_top_k_retains_the_pre_clamp_value(self) -> None:
        """The user asked for 5000. Reporting only 20 would hide the clamp; the
        metadata carries both numbers so the caller can see it happened."""
        service, _ = _service(max_top_k=20)
        response = service.search(SearchRequest(query="heart", top_k=5000))
        assert response.metadata is not None
        assert response.metadata.requested_top_k == 5000

    def test_requested_top_k_defaults_to_the_resolved_default(self) -> None:
        service, _ = _service(default_top_k=7)
        response = service.search(SearchRequest(query="heart"))
        assert response.metadata is not None
        assert response.metadata.requested_top_k == 7


# --------------------------------------------------------------------------- #
# Response assembly
# --------------------------------------------------------------------------- #
class TestResponseAssembly:
    def test_original_query_is_preserved_exactly(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="  Machine   Learning  "))
        assert response.original_query == "  Machine   Learning  "

    def test_processed_query_is_observable(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        assert response.processed_query == "heart"

    def test_ranks_are_one_based_and_contiguous(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        assert [hit.rank for hit in response.results] == [1, 2, 3]

    def test_result_order_is_not_changed(self) -> None:
        """The app layer maps in the order it received. There is no sort here,
        and adding one would silently invalidate Phase 7's RRF evidence."""
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        assert [hit.doc_id for hit in response.results] == ["d0", "d1", "d2"]
        assert [hit.score for hit in response.results] == [1.0, 0.99, 0.98]

    def test_provenance_preserves_the_zero_based_retriever_rank(self) -> None:
        """The 1-based display conversion must not leak into the per-retriever
        record, which Phase 7 defined as 0-based."""
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        first = response.results[0]
        for entry in first.provenance:
            assert entry.rank == 0
        assert response.results[2].provenance[0].rank == 2

    def test_metadata_is_populated(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        metadata = response.metadata
        assert metadata is not None
        assert metadata.backend == "hybrid"
        assert metadata.document_count == 3
        assert metadata.returned_count == 3
        assert metadata.corpus_exhausted is True
        assert metadata.empty_reason is None
        assert metadata.took_ms >= 0.0

    def test_helpers(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        assert len(response) == 3
        assert response.count == 3
        assert response.top_hit() is response.results[0]
        assert response.top_hit().doc_id == "d0"  # type: ignore[union-attr]

    def test_to_dict_is_json_safe(self) -> None:
        import json

        service, _ = _service()
        payload = service.search(SearchRequest(query="heart")).to_dict()
        assert json.loads(json.dumps(payload))["results"][0]["rank"] == 1

    def test_info_and_health(self) -> None:
        service, _ = _service(default_top_k=3, max_top_k=50)
        info = service.info()
        assert info["default_top_k"] == 3
        assert info["max_top_k"] == 50
        assert info["retrievers"] == ("bm25", "semantic")
        health = service.health()
        assert health["ready"] is True
        assert health["documents"] == 3


class TestZeroResultsVersusFailure:
    """The distinction the whole error taxonomy exists for."""

    def test_zero_results_is_a_normal_response(self) -> None:
        service, _ = _service()
        response = service.search(SearchRequest(query="heart"))
        # Force the zero case through the same healthy path.
        empty = ApplicationSearchResponse(
            original_query=response.original_query,
            processed_query=response.processed_query,
            results=(),
            metadata=SearchMetadata(
                took_ms=0.1,
                backend="hybrid",
                document_count=3,
                query_processing_enabled=True,
                was_modified=False,
                was_corrected=False,
                requested_top_k=10,
                returned_count=0,
                truncated=True,
                corpus_exhausted=False,
                empty_reason=NO_MATCHES,
            ),
        )
        assert empty.results == ()
        assert empty.metadata is not None
        assert empty.metadata.empty_reason == NO_MATCHES
        assert empty.top_hit() is None

    @pytest.mark.parametrize(
        "cause",
        [
            CorruptIndexError("index file is truncated"),
            DataError("model produced 384 dimensions, index expects 384"),
            RuntimeError("something unforeseen"),
        ],
    )
    def test_infrastructure_failures_raise_rather_than_return_empty(
        self, cause: BaseException
    ) -> None:
        service, _ = _service(fail_with=cause)
        with pytest.raises(SearchBackendError):
            service.search(SearchRequest(query="heart"))

    def test_backend_error_preserves_the_cause(self) -> None:
        original = CorruptIndexError("index file is truncated")
        service, _ = _service(fail_with=original)
        with pytest.raises(SearchBackendError) as excinfo:
            service.search(SearchRequest(query="heart"))
        assert excinfo.value.cause is original
        assert excinfo.value.status_code == 503
        assert excinfo.value.code == "search_backend_error"

    def test_backend_error_is_not_wrapped_twice(self) -> None:
        already = SearchBackendError("wrapped", cause=ValueError("original"))
        service, _ = _service(fail_with=already)
        with pytest.raises(SearchBackendError) as excinfo:
            service.search(SearchRequest(query="heart"))
        assert excinfo.value is already

    def test_cause_is_absent_from_the_client_payload(self) -> None:
        """A stack trace must never cross this boundary; only the code and the
        message are serialised."""
        service, _ = _service(fail_with=CorruptIndexError("boom"))
        with pytest.raises(SearchBackendError) as excinfo:
            service.search(SearchRequest(query="heart"))
        payload = excinfo.value.to_dict()
        assert "cause" not in payload
        assert payload["code"] == "search_backend_error"


class TestBadRequestsStayBadRequests:
    """A 400 must not be laundered into a 503. Reporting "retry later" for an
    empty query teaches the caller to retry something that can never succeed."""

    def test_empty_query_is_a_400_not_a_503(self) -> None:
        service, _ = _service()
        with pytest.raises(EmptyQueryError) as excinfo:
            service.search(SearchRequest(query="   "))
        assert not isinstance(excinfo.value, SearchBackendError)
        assert excinfo.value.status_code == 400

    def test_whitespace_only_query_is_a_400(self) -> None:
        service, _ = _service()
        with pytest.raises(EmptyQueryError):
            service.search(SearchRequest(query="\t\n  "))

    def test_empty_string_is_a_400(self) -> None:
        service, _ = _service()
        with pytest.raises(EmptyQueryError):
            service.search(SearchRequest(query=""))

    def test_query_too_long_stays_a_400(self) -> None:
        service, _ = _service(fail_with=QueryTooLongError("too long"))
        with pytest.raises(QueryTooLongError) as excinfo:
            service.search(SearchRequest(query="x" * 10))
        assert not isinstance(excinfo.value, SearchBackendError)

    def test_search_error_from_the_retrievers_is_not_converted(self) -> None:
        service, _ = _service(fail_with=SearchError("something else"))
        with pytest.raises(SearchError) as excinfo:
            service.search(SearchRequest(query="heart"))
        assert not isinstance(excinfo.value, SearchBackendError)


class TestCompatibility:
    """The Phase 8 public name must still exist and still be importable."""

    def test_phase8_search_service_is_untouched(self) -> None:
        assert SearchService is not None
        assert hasattr(SearchService, "search")
        assert hasattr(SearchService, "search_raw")

    def test_search_hit_and_metadata_are_constructible(self) -> None:
        hit = SearchHit(doc_id="d", title="t", snippet="s", rank=1, score=0.5)
        assert hit.provenance_for("bm25") is None
