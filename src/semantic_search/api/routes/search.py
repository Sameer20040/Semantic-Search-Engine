"""The search endpoints.

Three of them, and the difference between them is the point:

``POST /search``
    Results. No reasoning, no extra rendering. This is the endpoint a caller
    scripts against.
``POST /search/explain``
    The same results, plus a per-result explanation of *how* each was ordered
    and *what the query pipeline did*. The results are identical -- asserted in
    the tests -- so the explanation can be read as commentary on a known answer
    rather than as a different answer.
``GET /search``
    A convenience wrapper for a browser address bar. Same body, same path,
    ``GET`` instead of ``POST``, which is why the query string carries the
    parameters. It exists so the endpoint can be opened and clicked during a
    demo; a client that cares about caching or request bodies should use POST.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from semantic_search.api.dependencies import ServiceDep
from semantic_search.api.models import (
    ErrorResponse,
    ExplanationResponse,
    SearchRequestBody,
)
from semantic_search.app.models import ApplicationSearchResponse, SearchRequest
from semantic_search.explainability import explain_response

router = APIRouter(tags=["search"])

#: Attached to every search route so the generated OpenAPI documents the error
#: shapes. Without these the docs show a 200 and nothing else, and a client has
#: no way to discover the error envelope.
_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "The query was empty, too long, or unusable."},
    413: {"model": ErrorResponse, "description": "The request body exceeded the configured limit."},
    422: {"model": ErrorResponse, "description": "The request body failed validation."},
    429: {"model": ErrorResponse, "description": "The rate limit was exceeded."},
    503: {
        "model": ErrorResponse,
        "description": "The retrieval stack is unavailable. The query may still be valid.",
    },
}


@router.post(
    "/search",
    response_model=ApplicationSearchResponse,
    responses=_ERRORS,
    summary="Search the corpus",
    status_code=status.HTTP_200_OK,
)
def search(request_body: SearchRequestBody, service: ServiceDep) -> ApplicationSearchResponse:
    """Run one search and return the ranked results with their metadata.

    Zero results is a 200 with ``results: []`` and
    ``metadata.empty_reason == "no_matches"``. A 503 means the engine itself
    failed and the query may still be fine -- the two are deliberately
    distinguishable, because a client that renders a 503 as "no matches" is
    reporting a broken engine as a working one.
    """
    response = service.search(
        SearchRequest(
            query=request_body.query,
            top_k=request_body.top_k,
            query_processing=request_body.query_processing,
        )
    )
    return response


@router.post(
    "/search/explain",
    response_model=ExplanationResponse,
    responses=_ERRORS,
    summary="Search and explain how the results were ordered",
)
def search_explain(request_body: SearchRequestBody, service: ServiceDep) -> ExplanationResponse:
    """Return the ranked results together with a checkable account of the ranking.

    The explanation states which retrievers returned each document, what each
    one's rank contributed under reciprocal rank fusion, and what the query
    pipeline did or did not change. It does not state that any document is
    relevant -- the system has no evidence for that, and the caveat travels with
    every response saying so.
    """
    response = service.search(
        SearchRequest(
            query=request_body.query,
            top_k=request_body.top_k,
            query_processing=request_body.query_processing,
        )
    )
    return _as_explanation(response)


@router.get(
    "/search",
    response_model=ApplicationSearchResponse,
    responses=_ERRORS,
    summary="Search (GET convenience form)",
)
def search_get(
    service: ServiceDep,
    query: Annotated[str, Query(min_length=1, max_length=2000, description="The search query.")],
    top_k: Annotated[int | None, Query(ge=1, le=1000)] = None,
    query_processing: Annotated[bool | None, Query()] = None,
) -> ApplicationSearchResponse:
    """The same search as ``POST /search``, for a browser or a curl one-liner.

    Identical behaviour, identical response. Provided so the API can be opened
    without a request body; a client that cares about request size or caching
    should prefer the POST form.
    """
    return service.search(
        SearchRequest(query=query, top_k=top_k, query_processing=query_processing)
    )


def _as_explanation(response: ApplicationSearchResponse) -> ExplanationResponse:
    """Attach the explanation to a response without altering the response."""
    rendered = explain_response(response)
    return ExplanationResponse(
        original_query=response.original_query,
        processed_query=response.processed_query,
        results=list(response.results),
        metadata=response.metadata,
        query_processing=rendered.query_explanation,
        result_explanations=[e.to_dict() for e in rendered.results],
        caveats=list(rendered.caveats),
    )


__all__ = ["router"]
