"""Tests for the Streamlit frontend's business logic.

What is and is not tested here
------------------------------
Everything under ``app/utils/api.py``: URL construction, error normalisation,
and the pure view-model builders. Those hold the frontend's decisions -- when a
response is an error, what a retriever's provenance means, what the page shows
a user who got nothing back -- and they are testable without a browser.

What is **not** tested: ``app/Home.py`` and ``app/components/render.py``. Those
call ``st.*`` on every statement and assert nothing; a test of them would either
be a mock-assertion ceremony or a browser test this environment cannot run.
Their behaviour is instead verified by the import/startup smoke check in
``tests/unit/test_app_frontend_smoke.py`` and by the integration procedure in
``docs/deployment.md``.

The fixtures below reproduce the real response shape field for field. That is
deliberate: a fixture invented to suit the parser tests the parser against the
fixture, not against the API. The shape was taken from ``api/models.py`` and
``app/models.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# app/ is a Streamlit app, not an installed package, so the repository root has
# to be importable. This mirrors the sys.path.insert that app/Home.py performs
# and the one every script/ file performs.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils.api import (  # noqa: E402
    DEFAULT_BASE_URL,
    ApiClient,
    ApiError,
    ResultView,
    build_health_view,
    build_result_view,
    build_search_view,
    default_base_url,
    describe_error,
    normalise_base_url,
)

# --------------------------------------------------------------------------- #
# A transport double. Not a mock library -- just the two attributes the client
# touches, so the tests assert on the client's behaviour and not on calls.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        *,
        raise_on_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._raise_on_json = raise_on_json

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        if self._raise_on_json:
            raise ValueError("not json")
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class _FakeSession:
    """Records the request and replays a queued response."""

    def __init__(self, response: Any = None, *, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self._error is not None:
            raise self._error
        return self._response


# --------------------------------------------------------------------------- #
# Real response shapes, copied from the API models.
# --------------------------------------------------------------------------- #


def _hit(
    rank: int, doc_id: str, *, provenance: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "rank": rank,
        "title": f"Article {doc_id}",
        "snippet": f"snippet for {doc_id}",
        "score": 0.05 - 0.001 * rank,
        "sources": [p["retriever"] for p in (provenance or [])],
        "provenance": provenance if provenance is not None else [],
    }


PROVENANCE_BOTH = [
    {"retriever": "bm25", "rank": 0, "raw_score": 21.4, "rrf_contribution": 0.01639},
    {"retriever": "semantic", "rank": 2, "raw_score": 0.71, "rrf_contribution": 0.01613},
]

PROVENANCE_ONE = [
    {"retriever": "semantic", "rank": 1, "raw_score": 0.83, "rrf_contribution": 0.01639},
]


@pytest.fixture
def explained_payload() -> dict[str, Any]:
    """A ``/search/explain`` body with two results, one found by both retrievers."""
    return {
        "original_query": "how does a muscle pump move blood",
        "processed_query": "how does a muscle pump move blood",
        "results": [
            _hit(0, "muscle_pump", provenance=PROVENANCE_BOTH),
            _hit(1, "skeletal_muscle", provenance=PROVENANCE_ONE),
        ],
        "metadata": {
            "query": "how does a muscle pump move blood",
            "processed_query": "how does a muscle pump move blood",
            "top_k": 2,
            "requested_top_k": 2,
            "returned_count": 2,
            "document_count": 300,
            "took_ms": 41.5,
            "empty_reason": None,
            "query_processing_enabled": True,
            "was_modified": False,
            "was_corrected": False,
            "corrections": [],
            "expansions": [],
        },
        "query_processing": {
            "enabled": True,
            "lines": ["normalization applied (no change)"],
        },
        "result_explanations": [
            {"doc_id": "muscle_pump", "lines": ["ranked 1st by BM25", "ranked 3rd by semantic"]},
        ],
        "caveats": ["This ranking has not been evaluated for retrieval quality."],
    }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfiguration:
    def test_default_is_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STREAMLIT_API_BASE_URL", raising=False)
        monkeypatch.delenv("SSE_API_BASE_URL", raising=False)
        assert default_base_url() == DEFAULT_BASE_URL

    def test_environment_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://api.internal:9000")
        assert default_base_url() == "http://api.internal:9000"

    def test_alias_is_accepted_for_backwards_compatibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("STREAMLIT_API_BASE_URL", raising=False)
        monkeypatch.setenv("SSE_API_BASE_URL", "http://elsewhere:1234")
        assert default_base_url() == "http://elsewhere:1234"

    def test_documented_name_wins_over_the_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two names for one setting is a defect; when both are present the
        # documented one decides, so the behaviour does not depend on which
        # shell exported what first.
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://primary:1")
        monkeypatch.setenv("SSE_API_BASE_URL", "http://secondary:2")
        assert default_base_url() == "http://primary:1"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("http://localhost:8000", "http://localhost:8000"),
            ("http://localhost:8000/", "http://localhost:8000"),
            ("http://localhost:8000///", "http://localhost:8000"),
            ("  http://localhost:8000/  ", "http://localhost:8000"),
        ],
    )
    def test_trailing_slashes_are_stripped(self, raw: str, expected: str) -> None:
        assert normalise_base_url(raw) == expected


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #


class TestUrlConstruction:
    def test_a_trailing_slash_does_not_produce_a_double_slash(self) -> None:
        session = _FakeSession(_FakeResponse(200, {}))
        client = ApiClient("http://localhost:8000/", session=session)
        client.health()
        assert session.calls[0]["url"] == "http://localhost:8000/health"

    def test_base_url_is_readable_for_the_sidebar(self) -> None:
        client = ApiClient("http://example.test:1234/")
        assert client.base_url == "http://example.test:1234"


# --------------------------------------------------------------------------- #
# Client behaviour
# --------------------------------------------------------------------------- #


class TestClientRequests:
    def test_health_is_a_get(self) -> None:
        session = _FakeSession(_FakeResponse(200, {"ready": True}))
        ApiClient(session=session).health()
        assert session.calls[0]["method"] == "GET"
        assert session.calls[0]["url"].endswith("/health")

    def test_explain_search_posts_to_the_explained_endpoint(self) -> None:
        session = _FakeSession(_FakeResponse(200, {}))
        ApiClient(session=session).search("q", top_k=3, explain=True)
        assert session.calls[0]["method"] == "POST"
        assert session.calls[0]["url"].endswith("/search/explain")
        assert session.calls[0]["json"] == {"query": "q", "top_k": 3}

    def test_plain_search_posts_to_the_plain_endpoint(self) -> None:
        session = _FakeSession(_FakeResponse(200, {}))
        ApiClient(session=session).search("q", explain=False)
        assert session.calls[0]["url"].endswith("/search")

    def test_optional_fields_are_omitted_not_sent_as_null(self) -> None:
        # top_k=None means "use the server default". Sending {"top_k": null}
        # is a different request and this project rejects it as a validation
        # error, so the field must be absent rather than null.
        session = _FakeSession(_FakeResponse(200, {}))
        ApiClient(session=session).search("q", top_k=None, query_processing=None)
        assert session.calls[0]["json"] == {"query": "q"}

    def test_query_processing_can_be_forced_off_per_request(self) -> None:
        session = _FakeSession(_FakeResponse(200, {}))
        ApiClient(session=session).search("q", query_processing=False)
        assert session.calls[0]["json"] == {"query": "q", "query_processing": False}

    def test_the_configured_timeout_is_passed_through(self) -> None:
        session = _FakeSession(_FakeResponse(200, {}))
        ApiClient(session=session, timeout=12.5).health()
        assert session.calls[0]["timeout"] == 12.5


class TestClientErrors:
    def test_a_503_raises_rather_than_returning_ready_false(self) -> None:
        # The API carries readiness in the status code by design. A client
        # that returned the body instead would have to re-derive readiness,
        # and a caller that forgot to would treat a broken backend as healthy.
        session = _FakeSession(
            _FakeResponse(
                503,
                {"error": {"code": "startup_validation_error", "message": "no index"}},
                headers={"X-Request-ID": "abc123"},
            )
        )
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.status_code == 503
        assert excinfo.value.error_code == "startup_validation_error"
        assert excinfo.value.request_id == "abc123"
        assert excinfo.value.transport_failure is False

    def test_a_connection_refused_is_a_transport_failure(self) -> None:
        import requests

        session = _FakeSession(error=requests.exceptions.ConnectionError("connection refused"))
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.transport_failure is True
        assert excinfo.value.status_code is None

    def test_a_timeout_is_a_transport_failure(self) -> None:
        import requests

        session = _FakeSession(error=requests.exceptions.Timeout("too slow"))
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.transport_failure is True
        assert excinfo.value.error_code == "timeout"

    def test_a_200_that_is_not_json_is_an_error_not_a_result(self) -> None:
        # The dangerous case: an HTML page from a proxy on port 8000. Returning
        # an empty result set here would render "no documents matched" for what
        # is actually a misconfigured URL.
        session = _FakeSession(_FakeResponse(200, raise_on_json=True))
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.error_code == "invalid_json"

    def test_an_error_body_without_the_envelope_falls_back_to_the_status(self) -> None:
        session = _FakeSession(_FakeResponse(500, {"detail": "nginx was unhappy"}))
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.status_code == 500
        assert excinfo.value.message == "HTTP 500"
        assert excinfo.value.error_code is None

    def test_an_error_body_that_is_not_json_falls_back_to_the_status(self) -> None:
        session = _FakeSession(_FakeResponse(502, raise_on_json=True))
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert excinfo.value.message == "HTTP 502"

    def test_the_api_error_never_carries_a_traceback(self) -> None:
        session = _FakeSession(
            _FakeResponse(500, {"error": {"code": "internal_error", "message": "boom"}})
        )
        with pytest.raises(ApiError) as excinfo:
            ApiClient(session=session).health()
        assert "Traceback" not in excinfo.value.message
        assert json.dumps(excinfo.value.to_dict())


# --------------------------------------------------------------------------- #
# describe_error
# --------------------------------------------------------------------------- #


class TestDescribeError:
    def test_a_503_is_never_rendered_as_an_empty_result(self) -> None:
        described = describe_error(
            ApiError(message="artifacts missing", status_code=503, error_code="index_not_found")
        )
        assert described["heading"] == "Search backend unavailable"
        assert "not an empty result set" in described["hint"]

    def test_a_transport_failure_names_the_url_it_tried(self) -> None:
        described = describe_error(
            ApiError(
                message="refused",
                error_code="http://localhost:8000",
                transport_failure=True,
            )
        )
        assert described["heading"] == "Backend unreachable"
        assert "http://localhost:8000" in described["hint"]

    def test_a_timeout_is_distinguished_from_a_refusal(self) -> None:
        described = describe_error(
            ApiError(message="the request timed out", error_code="timeout", transport_failure=True)
        )
        assert "did not respond in time" in described["detail"]

    def test_the_request_id_is_surfaced_for_the_operator(self) -> None:
        described = describe_error(ApiError(message="boom", status_code=500, request_id="req-42"))
        assert "req-42" in described["hint"]

    def test_a_non_api_exception_is_labelled_client_side(self) -> None:
        # If this path silently claimed a backend problem, an operator would go
        # looking at a server that was working fine.
        described = describe_error(ValueError("bad widget key"))
        assert described["heading"] == "Unexpected error"
        assert "client-side" in described["hint"]

    def test_a_rate_limit_explains_the_wait(self) -> None:
        described = describe_error(ApiError(message="slow down", status_code=429))
        assert described["heading"] == "Rate limited"
        assert "reset" in described["hint"]

    def test_an_unmapped_status_still_produces_something_showable(self) -> None:
        described = describe_error(ApiError(message="odd", status_code=418))
        assert described["heading"] == "Request failed"
        assert described["detail"] == "odd"


# --------------------------------------------------------------------------- #
# View models
# --------------------------------------------------------------------------- #


class TestProvenance:
    def test_the_list_form_the_api_actually_returns_is_read(self) -> None:
        result = build_result_view(_hit(0, "d1", provenance=PROVENANCE_BOTH))
        assert [p.retriever for p in result.provenance] == ["bm25", "semantic"]
        assert result.provenance[0].rank == 0
        assert result.provenance[0].rrf_contribution == pytest.approx(0.01639)

    def test_the_dict_form_degrades_rather_than_raising(self) -> None:
        # Not emitted by any current response. Handled so a future reshaping
        # produces a partial panel instead of a page that crashes.
        result = build_result_view(
            {"doc_id": "d1", "rank": 0, "provenance": {"bm25": {"rank": 3, "raw_score": 1.0}}}
        )
        assert result.provenance[0].retriever == "bm25"
        assert result.provenance[0].rank == 3

    def test_unrecognised_provenance_is_dropped_not_guessed(self) -> None:
        result = build_result_view(
            {"doc_id": "d1", "rank": 0, "provenance": ["nonsense", {"no_retriever": 1}]}
        )
        assert result.provenance == []

    def test_missing_provenance_is_not_an_error(self) -> None:
        result = build_result_view({"doc_id": "d1", "rank": 0})
        assert result.provenance == []
        assert result.found_by_both is False

    def test_the_fused_rank_is_one_based_and_the_provenance_rank_is_not(self) -> None:
        """Regression test for the off-by-one this project has shipped twice.

        Two different conventions meet in this one object, and neither is a
        mistake:

        * ``SearchHit.rank`` -- what ``ApplicationSearchService._to_hit``
          assigns -- is ``position + 1``, so the API's ``rank`` already starts
          at 1 and the UI must use it unchanged.
        * ``HitProvenance.rank`` -- each retriever's own rank -- is 0-based,
          as fusion consumes it, and the UI converts only when drawing it.

        A test that applied one convention to both would render the first
        result as "#2". This asserts both values against the shapes the API
        actually sends.
        """
        result = build_result_view(_hit(1, "d1", provenance=PROVENANCE_BOTH))
        # The API sent 1; the UI must still hold 1, not 0 and not 2.
        assert result.rank == 1
        # The API sent 0 for bm25's rank; the view model must not have shifted
        # it, because the renderer needs the 0-based form to add 1 itself.
        assert result.provenance[0].rank == 0
        assert result.provenance[1].rank == 2


class TestFoundByBoth:
    def test_two_provenance_entries_means_both_retrievers(self) -> None:
        assert build_result_view(_hit(0, "d", provenance=PROVENANCE_BOTH)).found_by_both is True

    def test_one_provenance_entry_means_one_retriever(self) -> None:
        assert build_result_view(_hit(0, "d", provenance=PROVENANCE_ONE)).found_by_both is False

    def test_agreement_is_derived_from_provenance_not_from_sources(self) -> None:
        # sources and provenance describe the same fact, but provenance is the
        # authoritative record. If they disagreed, provenance decides.
        raw = _hit(0, "d", provenance=PROVENANCE_ONE)
        raw["sources"] = ["bm25", "semantic"]
        assert build_result_view(raw).found_by_both is False


class TestBuildSearchView:
    def test_a_full_explained_payload_maps_onto_the_view(self, explained_payload: dict) -> None:
        view = build_search_view(explained_payload, explain=True)
        assert view.original_query == "how does a muscle pump move blood"
        assert view.document_count == 300
        assert view.returned_count == 2
        assert view.requested_top_k == 2
        assert view.took_ms == pytest.approx(41.5)
        assert view.is_empty is False
        assert len(view.results) == 2

    def test_explanations_are_attached_to_the_right_result(self, explained_payload: dict) -> None:
        view = build_search_view(explained_payload, explain=True)
        assert view.results[0].explanation_lines == [
            "ranked 1st by BM25",
            "ranked 3rd by semantic",
        ]
        assert view.results[1].explanation_lines == []

    def test_caveats_are_passed_through_unmodified(self, explained_payload: dict) -> None:
        view = build_search_view(explained_payload, explain=True)
        assert view.caveats == ["This ranking has not been evaluated for retrieval quality."]

    def test_a_plain_payload_never_crashes_the_explanation_reader(
        self, explained_payload: dict
    ) -> None:
        # /search returns no result_explanations. build_search_view with
        # explain=True on such a payload must degrade to empty lists rather
        # than raise, so the caller need not track which endpoint it used.
        del explained_payload["result_explanations"]
        del explained_payload["caveats"]
        view = build_search_view(explained_payload, explain=True)
        assert view.caveats == []
        assert all(r.explanation_lines == [] for r in view.results)

    def test_explain_false_ignores_explanations_entirely(self, explained_payload: dict) -> None:
        view = build_search_view(explained_payload, explain=False)
        assert view.caveats == []
        assert view.results[0].explanation_lines == []

    def test_an_empty_result_set_is_an_empty_view_not_an_error(self) -> None:
        view = build_search_view(
            {
                "results": [],
                "metadata": {
                    "empty_reason": "no_matches",
                    "returned_count": 0,
                    "document_count": 300,
                },
            }
        )
        assert view.is_empty is True
        assert view.empty_reason == "no_matches"

    def test_the_empty_reason_survives_being_none(self) -> None:
        view = build_search_view({"results": [], "metadata": {"empty_reason": None}})
        assert view.empty_reason is None

    def test_query_processing_lines_are_read(self) -> None:
        view = build_search_view(
            {
                "results": [],
                "metadata": {},
                "query_processing": {"lines": ["spelling correction: tectonics -> tectonic"]},
            },
            explain=True,
        )
        assert view.query_processing_lines == ["spelling correction: tectonics -> tectonic"]

    def test_was_modified_drives_the_expander_default(self) -> None:
        view = build_search_view(
            {
                "results": [],
                "metadata": {
                    "query_processing_enabled": True,
                    "was_modified": True,
                    "was_corrected": True,
                    "corrections": [{"from": "tectonics", "to": "tectonic"}],
                },
            }
        )
        assert view.was_modified is True
        assert view.was_corrected is True
        assert view.corrections == [{"from": "tectonics", "to": "tectonic"}]

    def test_missing_metadata_does_not_raise(self) -> None:
        view = build_search_view({"results": [{"doc_id": "d", "rank": 0}]})
        assert view.took_ms == 0.0
        assert view.document_count == 0

    def test_non_dict_results_are_skipped(self) -> None:
        # A malformed entry should cost that entry, not the whole page.
        view = build_search_view({"results": [None, "junk", {"doc_id": "d", "rank": 0}]})
        assert [r.doc_id for r in view.results] == ["d"]

    def test_the_metadata_query_is_not_mistaken_for_the_request_query(self) -> None:
        # The response carries the query in two places. original_query is the
        # one the user typed; the empty state quotes it, so it must be the top
        # level field and not something reconstructed from metadata.
        view = build_search_view(
            {
                "original_query": "typed query",
                "processed_query": "processed query",
                "results": [],
                "metadata": {"query": "metadata query"},
            }
        )
        assert view.original_query == "typed query"
        assert view.processed_query == "processed query"


class TestBuildHealthView:
    def test_the_expected_fields_are_flattened(self) -> None:
        view = build_health_view(
            {
                "status": "ok",
                "ready": True,
                "backend": "hybrid",
                "documents": 300,
                "retrievers": ["bm25", "semantic"],
                "startup_seconds": 12.5,
                "rate_limit_per_minute": 120,
            }
        )
        assert view["ready"] is True
        assert view["documents"] == 300
        assert view["retrievers"] == ["bm25", "semantic"]

    def test_a_missing_ready_field_is_false_not_a_crash(self) -> None:
        # Readiness defaults to False. Defaulting it to True would let a
        # malformed health payload talk the UI into rendering a search box.
        assert build_health_view({})["ready"] is False

    def test_an_unexpected_extra_field_is_not_forwarded(self) -> None:
        # Fixed keys, so a future server field cannot silently appear in the UI.
        assert "secret_internal_detail" not in build_health_view({"secret_internal_detail": "x"})


class TestResultViewDefaults:
    def test_a_sparse_hit_still_builds(self) -> None:
        result = build_result_view({})
        assert isinstance(result, ResultView)
        assert result.title == "(untitled)"
        assert result.retrievers == []
        assert result.provenance == []


# --------------------------------------------------------------------------- #
# The frontend must not contain retrieval logic
# --------------------------------------------------------------------------- #


class TestNoRetrievalLogicInTheFrontend:
    """A structural check, stated as one.

    The frontend is required to present the backend's answer, not to produce
    one. That is enforceable by grep: none of the frontend may import the
    engine, and none of it may define a scoring or fusion function.
    """

    FORBIDDEN_IMPORTS = (
        "semantic_search",
        "sentence_transformers",
        "faiss",
        "rank_bm25",
    )

    def _frontend_sources(self) -> list[Path]:
        return sorted((_ROOT / "app").rglob("*.py"))

    def test_the_frontend_does_not_import_the_engine(self) -> None:
        offenders: list[str] = []
        for path in self._frontend_sources():
            text = path.read_text(encoding="utf-8")
            for name in self.FORBIDDEN_IMPORTS:
                # Only lines that look like imports are checked, so a mention
                # inside a docstring explaining *why* the frontend avoids the
                # engine does not trip the check.
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")) and name in stripped:
                        offenders.append(f"{path.name}: {stripped}")
        assert offenders == [], f"the frontend must not import retrieval code: {offenders}"

    def test_the_frontend_defines_no_scoring_function(self) -> None:
        offenders: list[str] = []
        for path in self._frontend_sources():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("def ") and any(
                    token in line for token in ("rrf", "reciprocal_rank", "tokenize", "embed")
                ):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"the frontend must not implement retrieval: {offenders}"
