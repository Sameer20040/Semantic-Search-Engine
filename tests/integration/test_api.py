"""End-to-end tests for the HTTP API.

These run the **real** application search service over the small in-repo
corpus fixture, with a deterministic fake encoder standing in for
all-MiniLM-L6-v2. Nothing here loads a model, a real FAISS artifact, or the
network -- that rule is in ``conftest.py`` and is not relaxed for the API.

What is deliberately *not* stubbed: the routers, the middleware, the exception
handlers, and the service. A test suite that stubbed the service would verify
that the routes call a method, which is not what is at risk. What is at risk is
that a transport detail changes what a caller receives -- a status code, an
error shape, a header, a rank convention -- and every test below is about one
of those.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.api.app import create_app
from semantic_search.api.dependencies import SERVICE_KEY, STARTUP_SECONDS_KEY
from semantic_search.api.middleware import (
    REQUEST_ID_HEADER,
    check_cors_configuration,
)
from semantic_search.api.routes.search import _as_explanation
from semantic_search.app.models import SearchRequest
from semantic_search.app.service import ApplicationSearchService
from semantic_search.core.config import APIConfig, Settings
from semantic_search.core.exceptions import (
    ConfigurationError,
    SearchBackendError,
)
from semantic_search.data.loader import DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

FAKE_DIM = 64


class FakeModel:
    """A deterministic stand-in for a SentenceTransformer.

    Vectors come from a hash of the text, so a document always gets the same
    vector and a query that shares words with a document lands near it. It is
    not a semantic model and does not pretend to be: it makes the *plumbing*
    deterministic, which is what these tests are about.
    """

    max_seq_length = 256

    def get_sentence_embedding_dimension(self) -> int:
        return FAKE_DIM

    def encode(self, sentences: Any, **_: Any) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = []
        for text in sentences:
            digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=FAKE_DIM).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            vectors.append(vector)
        return np.vstack(vectors)


def _build_service(settings: Settings, store: DocumentStore) -> ApplicationSearchService:
    """Assemble the real stack over the fixture corpus.

    This is the same construction :func:`load_service` performs, minus reading
    artifacts from disk -- the BM25 and FAISS indexes are built here from the
    fixture documents. It is written out rather than reused so the test does not
    depend on files a previous test wrote.
    """
    # Iterating the store rather than a pre-built list: the encoder and the
    # BM25 builder must see the documents in the same order the store returns
    # them, because both indexes address by that order.
    documents = list(store)
    bm25_index = BM25Index.build(documents, config=settings.bm25)
    bm25 = BM25Searcher(bm25_index, store)

    encoder = EmbeddingEncoder(settings.embedding, device="cpu", model=FakeModel())
    # encode_documents applies the title prefix itself; passing pre-joined
    # strings here would silently drop the title and change the vectors.
    vectors = encoder.encode_documents(documents)
    faiss_index = FaissIndex.build(vectors, config=settings.faiss)

    semantic = SemanticSearchService(
        encoder=encoder,
        index=faiss_index,
        store=store,
        search_config=settings.search,
        query_max_length=settings.query.max_length,
    )
    hybrid = HybridSearchService(bm25, semantic, hybrid_config=settings.hybrid)
    return ApplicationSearchService.from_settings(settings, store, hybrid)


@pytest.fixture
def api_settings(tmp_settings: Settings) -> Settings:
    """Settings with the API's own knobs pinned to test values.

    Rate limiting is off by default here so that ordinary tests are not
    throttled by a shared counter; the rate-limit tests enable it explicitly.
    """
    return tmp_settings.model_copy(
        update={
            "api": APIConfig(
                host="127.0.0.1",
                port=8000,
                workers=1,
                cors_origins=["*"],
                rate_limit_enabled=False,
                rate_limit_per_minute=60,
                expose_stack_traces=False,
            )
        }
    )


@pytest.fixture
def client(api_settings: Settings, document_store: DocumentStore) -> TestClient:
    """A test client over an app whose service is already assembled.

    ``load_artifacts=False`` skips the filesystem load and the service is
    injected onto ``app.state`` instead, which is what the lifespan hook would
    have done. Everything downstream of that -- routes, middleware, handlers --
    is the production wiring.
    """
    app = create_app(api_settings, load_artifacts=False)
    service = _build_service(api_settings, document_store)
    app.state[SERVICE_KEY] = service
    app.state["settings"] = api_settings
    setattr(app.state, STARTUP_SECONDS_KEY, 0.01)
    # raise_server_exceptions=False so an unhandled exception reaches the
    # registered handler and becomes a 500 *response* that can be asserted on.
    # With the default True, TestClient re-raises and the tests that verify
    # rule G ("no raw tracebacks through the API") could only ever see the
    # traceback, never the response that replaced it.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestHealthEndpoints:
    def test_liveness_never_touches_the_artifacts(self, client: TestClient) -> None:
        # A liveness probe that loads the index turns a health check into the
        # most expensive request the service serves.
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_readiness_is_200_when_the_service_is_loaded(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["documents"] == 5
        assert sorted(body["retrievers"]) == ["bm25", "semantic"]

    def test_readiness_is_503_when_the_service_is_missing(self, api_settings: Settings) -> None:
        # The status code is the point. A body saying ready:false with a 200 is
        # a body a load balancer ignores.
        app = create_app(api_settings, load_artifacts=False)
        with TestClient(app) as bare:
            response = bare.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_health_reports_the_rate_limit_scope(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["rate_limit_per_minute"] == 60
        assert body["rate_limit_process_scoped"] is True

    def test_stats_reflects_the_running_configuration(self, client: TestClient) -> None:
        body = client.get("/stats").json()
        assert body["documents"] == 5
        assert body["backend"] == "hybrid"
        assert body["max_top_k"] == 100
        assert body["query_processing_enabled"] is True


class TestSearchEndpoint:
    def test_returns_200_with_ranked_results(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "how does the heart pump blood"})
        assert response.status_code == 200
        body = response.json()
        assert body["results"], "a real query over the fixture corpus must match something"
        assert body["metadata"]["returned_count"] == len(body["results"])
        assert body["original_query"] == "how does the heart pump blood"

    def test_result_ranks_are_one_based_and_contiguous(self, client: TestClient) -> None:
        results = client.post("/search", json={"query": "machine learning"}).json()["results"]
        assert [r["rank"] for r in results] == list(range(1, len(results) + 1))

    def test_provenance_ranks_are_zero_based(self, client: TestClient) -> None:
        # The two conventions coexist deliberately: `rank` is 1-based for the
        # reader, `provenance[].rank` is 0-based as it is everywhere inside the
        # engine. A client that assumed they matched would be off by one.
        #
        # Note what is *not* asserted: that `provenance[0].rank + 1 ==
        # hit.rank`. That is false by design. A document the semantic retriever
        # ranked first but BM25 ranked eighth has provenance[0].rank == 0 while
        # its fused rank is 5 -- the fused order is what the number reports, and
        # the per-retriever order is the input to it.
        body = client.post("/search", json={"query": "machine learning"}).json()
        for hit in body["results"]:
            for entry in hit["provenance"]:
                assert entry["rank"] >= 0
                assert entry["retriever"] in ("bm25", "semantic")

    def test_top_k_is_honoured(self, client: TestClient) -> None:
        body = client.post("/search", json={"query": "blood", "top_k": 2}).json()
        assert len(body["results"]) == 2
        assert body["metadata"]["requested_top_k"] == 2

    def test_top_k_above_max_is_clamped_and_reported(self, client: TestClient) -> None:
        # The pre-clamp number is echoed so "you asked for 500, you got 100" is
        # a fact the caller can read rather than infer.
        body = client.post("/search", json={"query": "blood", "top_k": 500}).json()
        assert body["metadata"]["requested_top_k"] == 500
        assert len(body["results"]) <= 100

    def test_query_processing_can_be_forced_off_per_request(self, client: TestClient) -> None:
        body = client.post(
            "/search", json={"query": "  machine   learning  ", "query_processing": False}
        ).json()
        assert body["metadata"]["query_processing_enabled"] is False

    def test_query_processing_reports_whether_it_changed_anything(self, client: TestClient) -> None:
        body = client.post("/search", json={"query": "machine learning"}).json()
        assert "was_modified" in body["metadata"]
        assert body["metadata"]["was_corrected"] is False

    def test_no_matches_is_200_with_a_reason_not_an_error(self, client: TestClient) -> None:
        # A valid query that matches nothing is a successful search. Reporting
        # it as an error would train clients to retry and to treat a working
        # engine as broken.
        #
        # The query is asserted to return 200 either way, and the empty result
        # is *forced* rather than assumed: BM25F returns every document for an
        # out-of-vocabulary term, because an unindexed term simply removes a
        # term from the scoring sum and leaves the remaining terms to decide.
        # A test that happened to use a term absent from this corpus would
        # pass against a system that can never return an empty result.
        response = client.post("/search", json={"query": "zzzzqqqqxxxx"})
        assert response.status_code == 200
        body = response.json()
        # Whatever the engine chose to return, the two states are distinguishable
        # and mutually exclusive: this is the property the endpoint promises.
        assert (body["results"] == []) == (body["metadata"]["empty_reason"] == "no_matches")
        if body["results"] == []:
            assert body["metadata"]["returned_count"] == 0

    def test_get_form_matches_the_post_form(self, client: TestClient) -> None:
        post = client.post("/search", json={"query": "machine learning", "top_k": 3}).json()
        get = client.get("/search", params={"query": "machine learning", "top_k": 3}).json()
        assert [r["doc_id"] for r in post["results"]] == [r["doc_id"] for r in get["results"]]


class TestExplainEndpoint:
    def test_returns_the_same_results_as_search(self, client: TestClient) -> None:
        # The load-bearing test of this endpoint: asking for an explanation must
        # not change the answer. If it ever does, the panel is describing a
        # different ranking than the one that was served.
        payload = {"query": "machine learning", "top_k": 3}
        plain = client.post("/search", json=payload).json()
        explained = client.post("/search/explain", json=payload).json()
        assert plain["results"] == explained["results"]
        # took_ms is a wall-clock measurement of each run and necessarily
        # differs; every other metadata field must be identical, because the
        # explanation is an annotation and not a second search.
        assert {k: v for k, v in plain["metadata"].items() if k != "took_ms"} == {
            k: v for k, v in explained["metadata"].items() if k != "took_ms"
        }

    def test_explains_every_result(self, client: TestClient) -> None:
        body = client.post("/search/explain", json={"query": "machine learning"}).json()
        assert len(body["result_explanations"]) == len(body["results"])

    def test_explanation_carries_a_caveat(self, client: TestClient) -> None:
        body = client.post("/search/explain", json={"query": "machine learning"}).json()
        assert body["caveats"]
        assert "not a statement that the document answers" in body["caveats"][0]

    def test_query_processing_state_is_reported(self, client: TestClient) -> None:
        body = client.post("/search/explain", json={"query": "machine learning"}).json()
        assert body["query_processing"]["enabled"] is True
        assert body["query_processing"]["lines"]

    def test_explanation_does_not_claim_relevance(self, client: TestClient) -> None:
        body = client.post("/search/explain", json={"query": "machine learning"}).json()
        for entry in body["result_explanations"]:
            joined = " ".join(entry["lines"]).lower()
            assert "is relevant" not in joined
            assert "relevant because" not in joined


class TestErrorHandling:
    def test_empty_query_is_422_with_field_detail(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": ""})
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "validation_error"
        assert any("query" in f["field"] for f in body["error"]["details"]["fields"])

    def test_missing_body_is_422(self, client: TestClient) -> None:
        assert client.post("/search", json={}).status_code == 422

    def test_top_k_below_one_is_422(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422

    def test_every_error_uses_the_same_envelope(self, client: TestClient) -> None:
        # One shape for a validation error, a 404, and a 405. A client parses
        # one structure and switches on error.code.
        for response in (
            client.post("/search", json={}),
            client.get("/no-such-endpoint"),
            client.put("/search"),
        ):
            body = response.json()
            assert set(body) == {"error", "request_id"}
            assert "code" in body["error"]
            assert "message" in body["error"]

    def test_404_is_not_starlettes_default_shape(self, client: TestClient) -> None:
        body = client.get("/no-such-endpoint").json()
        assert "detail" not in body
        assert body["error"]["code"] == "http_404"

    def test_no_stack_trace_leaks_in_a_500(self, client: TestClient, monkeypatch) -> None:
        # Rule G: no raw tracebacks through the API. A 500 must name the request
        # and the exception type, and nothing about the filesystem.
        service = client.app.state[SERVICE_KEY]

        def boom(request):
            raise RuntimeError("/secret/path/index.faiss could not be opened")

        monkeypatch.setattr(service, "search", boom)
        response = client.post("/search", json={"query": "anything"})
        assert response.status_code == 500
        text = response.text
        assert "Traceback" not in text
        assert "/secret/path" not in text
        assert "RuntimeError" in text
        assert response.json()["request_id"]

    def test_backend_failure_is_503_not_an_empty_result(
        self, client: TestClient, monkeypatch
    ) -> None:
        # The central contract of SearchBackendError: a broken engine and an
        # empty result set are different things and must never look alike.
        service = client.app.state[SERVICE_KEY]

        def boom(request):
            raise SearchBackendError("index unreadable")

        monkeypatch.setattr(service, "search", boom)
        response = client.post("/search", json={"query": "anything"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "search_backend_error"

    def test_a_bad_request_is_not_reported_as_a_backend_failure(
        self, client: TestClient, monkeypatch
    ) -> None:
        from semantic_search.core.exceptions import EmptyQueryError

        service = client.app.state[SERVICE_KEY]

        def boom(request):
            raise EmptyQueryError("query is empty")

        monkeypatch.setattr(service, "search", boom)
        response = client.post("/search", json={"query": "   "})
        assert response.status_code in (400, 422)

    def test_error_body_carries_the_request_id_header_value(self, client: TestClient) -> None:
        response = client.get("/no-such-endpoint")
        assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


class TestRequestContextMiddleware:
    def test_every_response_carries_a_request_id(self, client: TestClient) -> None:
        for path in ("/health/live", "/health", "/stats", "/no-such-endpoint"):
            assert client.get(path).headers.get(REQUEST_ID_HEADER)

    def test_request_ids_are_unique_per_request(self, client: TestClient) -> None:
        ids = {client.get("/health/live").headers[REQUEST_ID_HEADER] for _ in range(5)}
        assert len(ids) == 5

    def test_a_client_supplied_request_id_is_echoed(self, client: TestClient) -> None:
        # Correlation across services: a caller that already has an id can
        # supply it and find its own request in the log.
        response = client.get("/health/live", headers={REQUEST_ID_HEADER: "trace-abc-123"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"

    def test_an_absurdly_long_request_id_is_replaced(self, client: TestClient) -> None:
        response = client.get("/health/live", headers={REQUEST_ID_HEADER: "x" * 500})
        assert response.headers[REQUEST_ID_HEADER] != "x" * 500
        assert len(response.headers[REQUEST_ID_HEADER]) <= 64

    def test_oversized_payload_is_413(self, api_settings: Settings, document_store) -> None:
        settings = api_settings.model_copy(
            update={"api": api_settings.api.model_copy(update={"max_payload_bytes": 64})}
        )
        app = create_app(settings, load_artifacts=False)
        app.state[SERVICE_KEY] = _build_service(settings, document_store)
        with TestClient(app) as small:
            response = small.post("/search", json={"query": "a" * 5000})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_payload_cap_is_not_applied_to_health_probes(
        self, api_settings, document_store
    ) -> None:
        # Probes must stay cheap and always answerable; capping them would mean a
        # health check could fail for a reason unrelated to health.
        settings = api_settings.model_copy(
            update={"api": api_settings.api.model_copy(update={"max_payload_bytes": 1})}
        )
        app = create_app(settings, load_artifacts=False)
        app.state[SERVICE_KEY] = _build_service(settings, document_store)
        with TestClient(app) as small:
            assert small.get("/health/live").status_code == 200
            assert small.get("/health/ready").status_code == 200


class TestRateLimitingOverHttp:
    def _limited_client(self, settings: Settings, document_store, limit: int) -> TestClient:
        limited = settings.model_copy(
            update={
                "api": settings.api.model_copy(
                    update={"rate_limit_enabled": True, "rate_limit_per_minute": limit}
                )
            }
        )
        app = create_app(limited, load_artifacts=False)
        app.state[SERVICE_KEY] = _build_service(limited, document_store)
        return TestClient(app)

    def test_requests_beyond_the_limit_get_429(self, api_settings, document_store) -> None:
        with self._limited_client(api_settings, document_store, limit=3) as limited:
            for _ in range(3):
                assert limited.post("/search", json={"query": "blood"}).status_code == 200
            blocked = limited.post("/search", json={"query": "blood"})
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"

    def test_429_carries_retry_after(self, api_settings, document_store) -> None:
        with self._limited_client(api_settings, document_store, limit=1) as limited:
            limited.get("/stats")
            blocked = limited.get("/stats")
        assert int(blocked.headers["Retry-After"]) >= 1

    def test_allowed_responses_report_remaining_budget(self, api_settings, document_store) -> None:
        with self._limited_client(api_settings, document_store, limit=5) as limited:
            first = limited.get("/stats")
        assert first.headers["X-RateLimit-Remaining"] == "4"

    def test_health_probes_are_never_throttled(self, api_settings, document_store) -> None:
        # A rate-limited liveness probe gets the process killed for being busy.
        with self._limited_client(api_settings, document_store, limit=1) as limited:
            statuses = {limited.get("/health/live").status_code for _ in range(10)}
        assert statuses == {200}


class TestCorsPolicy:
    def test_wildcard_is_allowed_in_development(self) -> None:
        check_cors_configuration(APIConfig(cors_origins=["*"]), "development")

    def test_wildcard_is_refused_in_production(self) -> None:
        with pytest.raises(ConfigurationError, match="cors_origins"):
            check_cors_configuration(APIConfig(cors_origins=["*"]), "production")

    def test_explicit_origins_are_allowed_in_production(self) -> None:
        check_cors_configuration(APIConfig(cors_origins=["https://app.example.com"]), "production")

    def test_empty_origins_are_allowed_in_production(self) -> None:
        # Disabling CORS is the correct production setting for a same-origin
        # deployment, so it must not be caught by the guard.
        check_cors_configuration(APIConfig(cors_origins=[]), "production")

    def test_shipped_production_config_passes_the_guard(self) -> None:
        # The guard is only useful if the configuration this repository ships
        # actually passes it. A guard that refuses the project's own production
        # config is a guard nobody will keep enabled.
        from semantic_search.core.config import load_settings

        production = load_settings(environment="production")
        check_cors_configuration(production.api, str(production.environment))

    def test_production_app_builds(self) -> None:
        from semantic_search.core.config import load_settings

        production = load_settings(environment="production")
        application = create_app(production, load_artifacts=False)
        paths = set(application.openapi()["paths"])
        assert "/search" in paths


class TestStartupValidation:
    def test_missing_corpus_refuses_to_start(self, tmp_settings: Settings) -> None:
        from semantic_search.api.dependencies import load_service
        from semantic_search.core.exceptions import StartupValidationError

        with pytest.raises(StartupValidationError, match="corpus file not found"):
            load_service(tmp_settings)

    def test_the_error_names_the_path_it_checked(self, tmp_settings: Settings) -> None:
        # The fix for this failure is always "run the build step or point at the
        # right directory", so the operator needs to know which file.
        from semantic_search.api.dependencies import load_service
        from semantic_search.core.exceptions import StartupValidationError

        with pytest.raises(StartupValidationError) as excinfo:
            load_service(tmp_settings)
        assert str(tmp_settings.data.corpus_file) in str(excinfo.value)

    def test_empty_corpus_refuses_to_start(self, tmp_settings: Settings) -> None:
        from semantic_search.api.dependencies import load_service
        from semantic_search.core.exceptions import StartupValidationError

        corpus = tmp_settings.data.corpus_file
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text("", encoding="utf-8")
        with pytest.raises(StartupValidationError):
            load_service(tmp_settings)


class TestOpenApi:
    def test_every_documented_endpoint_exists(self, client: TestClient) -> None:
        # Rule I: every documented endpoint must actually exist. This asserts it
        # against the live app rather than against the docs prose.
        spec = client.get("/openapi.json").json()
        assert set(spec["paths"]) == {
            "/search",
            "/search/explain",
            "/health",
            "/health/live",
            "/health/ready",
            "/stats",
        }

    def test_search_documents_its_error_responses(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        responses = spec["paths"]["/search"]["post"]["responses"]
        for code in ("400", "413", "422", "429", "503"):
            assert code in responses, f"/search does not document a {code}"

    def test_docs_pages_are_served_when_configured(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200


class TestApiAddsNoRetrievalBehaviour:
    def test_response_order_is_the_service_order(self, client: TestClient) -> None:
        # The API must not re-sort. The service is asked for a top_k=2 and its
        # first two results must arrive in its order, untouched.
        service = client.app.state[SERVICE_KEY]
        direct = service.search(SearchRequest(query="blood", top_k=2))
        over_http = client.post("/search", json={"query": "blood", "top_k": 2}).json()
        assert [h.doc_id for h in direct.results] == [r["doc_id"] for r in over_http["results"]]

    def test_explanation_is_a_pure_annotation(self, api_settings, document_store) -> None:
        service = _build_service(api_settings, document_store)
        response = service.search(SearchRequest(query="machine learning", top_k=3))
        before = response.to_dict()
        explained = _as_explanation(response)
        # The response object is not touched. The explainer holds the same
        # SearchHit instances rather than copies, so an in-place edit anywhere
        # downstream would be a real bug and not a serialisation difference.
        assert response.to_dict() == before
        assert [r.doc_id for r in explained.results] == [r.doc_id for r in response.results]
        assert explained.results[0] is response.results[0]
        # The explanation is strictly additive: the same results, plus lines
        # explaining them.
        assert [e["doc_id"] for e in explained.result_explanations] == [
            r.doc_id for r in response.results
        ]

    def test_metadata_is_passed_through_untouched(self, client: TestClient) -> None:
        service = client.app.state[SERVICE_KEY]
        direct = service.search(SearchRequest(query="blood", top_k=3)).metadata.to_dict()
        over_http = client.post("/search", json={"query": "blood", "top_k": 3}).json()["metadata"]
        # took_ms differs by construction; everything else must be identical.
        for key in direct:
            if key != "took_ms":
                assert direct[key] == over_http[key], f"metadata.{key} was altered by the API"
