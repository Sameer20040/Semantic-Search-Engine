"""Request and response schemas for the HTTP API.

These are deliberately separate from ``app/models.py``. The application models
describe what retrieval produced; these describe what goes over the wire, and
they differ in ways that matter:

* a field may be **excluded from responses** (``response_model_exclude_none``)
  so a payload does not carry a long tail of nulls that mean "not applicable";
* ``explanation`` is an opt-in query parameter, because rendering provenance for
  every result of every request is work most callers do not want and none of
  them should have to pay for;
* the error envelope is defined here, once, so every failure -- raised,
  validated, or produced by middleware -- has the same shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semantic_search.app.models import SearchHit, SearchMetadata


class SearchRequestBody(BaseModel):
    """Body of ``POST /search``.

    Validation here is the HTTP-shaped duplicate of checks the search service
    already makes. That is not redundant: the service's rules are the engine's
    contract and apply however it is called, while these produce a 422 with a
    field-level message a client can act on. A client that sends an empty query
    should be told *which field* was empty, not receive the engine's prose.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "how does a muscle pump move blood", "top_k": 5},
            ]
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search query.",
        examples=["how does a muscle pump move blood"],
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=1000,
        description=(
            "Number of results. Defaults to app.default_top_k and is clamped to "
            "app.max_top_k; the pre-clamp value is reported in "
            "metadata.requested_top_k."
        ),
        examples=[5],
    )
    query_processing: bool | None = Field(
        None,
        description=(
            "Force the query pipeline on or off for this request. Omit to use "
            "the server's configured default. The ON/OFF comparison depends on "
            "this being settable per request."
        ),
    )


class ExplanationResponse(BaseModel):
    """The ``/search/explain`` payload: the same results, plus the reasoning.

    The results are not a summary of the explanation or a subset of it. They are
    the identical objects ``/search`` returns, byte for byte, so a caller can
    diff the two endpoints and confirm that asking for an explanation did not
    change the answer. If it ever does, that is a bug worth catching.
    """

    original_query: str
    processed_query: str
    results: list[SearchHit]
    #: Optional, mirroring ``ApplicationSearchResponse.metadata``. The service
    #: always populates it, so in practice it is present; it is typed optional
    #: rather than asserted non-null because a hand-constructed response is a
    #: legitimate input to the explainer, which handles its absence explicitly
    #: rather than crashing on it.
    metadata: SearchMetadata | None
    query_processing: dict[str, Any]
    result_explanations: list[dict[str, Any]]
    caveats: list[str]


class HealthResponse(BaseModel):
    """Readiness of the process.

    ``ready`` is false when the artifacts are missing, which is the one state a
    load balancer must be able to see. A server that is listening but has no
    index answers searches with 503s; without this field there is no way to tell
    that apart from "the index is slow".
    """

    status: str
    ready: bool
    backend: str
    documents: int
    retrievers: list[str]
    startup_seconds: float
    rate_limit_per_minute: int
    rate_limit_process_scoped: bool = Field(
        description=(
            "True when the rate limit counts per process. With more than one "
            "worker the effective limit is the configured value times the worker "
            "count."
        )
    )


class StatsResponse(BaseModel):
    """The configuration snapshot from ``ApplicationSearchService.info()``."""

    backend: str
    documents: int
    retrievers: list[str]
    default_top_k: int
    max_top_k: int
    max_query_chars: int
    query_processing_enabled: bool
    query_processing: dict[str, Any]


class ErrorDetail(BaseModel):
    """The body of every error this API returns."""

    code: str = Field(
        description=(
            "A stable machine-readable identifier. The message is for humans and "
            "may change; this is what a client should branch on."
        )
    )
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """The uniform error envelope.

    Every failure -- a validation error, an empty query, a missing index, a rate
    limit -- arrives in this shape. A client parses one structure and switches
    on ``error.code``.
    """

    error: ErrorDetail
    request_id: str = Field(
        description="Matches the X-Request-ID response header; quote it in bug reports."
    )


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "ExplanationResponse",
    "HealthResponse",
    "SearchRequestBody",
    "StatsResponse",
]
