"""The HTTP client the UI uses, and the view-models it renders.

Two things live here, deliberately separated:

``ApiClient``
    A thin transport wrapper over :mod:`requests`. It knows URLs, timeouts, and
    the error envelope. It does not know what a Streamlit widget is.

``build_result_view`` / ``describe_error`` / ``build_health_view``
    **Pure functions** from a decoded JSON payload to something renderable.
    These are where the frontend's business logic actually is, which is why
    they are the part covered by tests: they decide what a retriever's
    provenance means, when a response is an error, and what to tell a user who
    typed a query that matched nothing.

Why the UI talks to the API at all
----------------------------------
Because :class:`~semantic_search.app.service.ApplicationSearchService` re-runs
retrieval and rebuilds its own result objects. If the UI imported it directly,
the browser session would load a second copy of the encoder, the FAISS index
and the BM25 index, and the page would be answering from a corpus the probes
never measured. The UI therefore holds no retrieval code at all -- no tokeniser,
no fusion, no scoring, no query processing. Every number it displays came out
of a real search performed by the real service.

The one thing this module must not do is *interpret* the numbers more strongly
than the backend did. ``describe_error`` says what the server said; it does not
translate a 503 into "nothing found". That distinction is the whole point of
the API's error taxonomy, and a client that undoes it makes the backend's
care pointless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

#: Environment variable naming the backend. Read by ``default_base_url`` and
#: surfaced in the sidebar so the address the UI is talking to is always
#: visible rather than buried in a launch command. This is the name already
#: documented in ``.env.example``, so it is the primary one here rather than a
#: second spelling of the same setting.
API_BASE_URL_ENV = "STREAMLIT_API_BASE_URL"

#: Accepted as an alternative. Having two names for one setting is a defect,
#: so the alternate is documented as such rather than left to be discovered:
#: if both are set the primary wins.
API_BASE_URL_ENV_ALIAS = "SSE_API_BASE_URL"

#: Fallback when the environment variable is unset. Loopback, not a hostname:
#: a UI silently pointed at some other host is a debugging session nobody
#: enjoys.
DEFAULT_BASE_URL = "http://localhost:8000"

#: Seconds before a request is abandoned. Generous, because the first request
#: against a freshly-started backend includes a 30 s model load; too short a
#: timeout would report that as a failure when the server is merely warming up.
DEFAULT_TIMEOUT = 60.0


def default_base_url() -> str:
    """The backend URL from the environment, or the loopback default.

    Reads the primary name first and the alias second, so a machine that
    already has the alternate exported keeps working while the documented name
    stays authoritative.
    """
    for name in (API_BASE_URL_ENV, API_BASE_URL_ENV_ALIAS):
        value = os.environ.get(name)
        if value:
            return value
    return DEFAULT_BASE_URL


def normalise_base_url(raw: str) -> str:
    """Strip trailing slashes so ``urljoin``-style concatenation is safe.

    A trailing slash in a configured base URL is the single most common cause
    of a ``//search`` path that a strict proxy answers with 404, and it is
    invisible in the configuration file.
    """
    return raw.strip().rstrip("/")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ApiError(Exception):
    """A failure to get an answer out of the backend.

    Carries enough context to render a useful message: the status code, the
    server's own error code, its message, and the request id an operator can
    look up in the server logs. It never carries a traceback -- there is no
    traceback to have, and the server does not send one.
    """

    message: str
    status_code: int | None = None
    error_code: str | None = None
    request_id: str | None = None
    #: True when the request never reached the server at all (DNS, connection
    #: refused, timeout). Distinct from a server that answered with 5xx: one is
    #: "start the backend", the other is "the backend has a problem".
    transport_failure: bool = False

    def __post_init__(self) -> None:
        # Not super().__init__: with slots=True the dataclass decorator builds a
        # *new* class object, so the implicit __class__ cell captured by a
        # zero-argument super() refers to the class that no longer exists and
        # raises "obj must be an instance or subtype of type" on every
        # instantiation. Naming the base explicitly is immune to that.
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "transport_failure": self.transport_failure,
        }


#: HTTP statuses this client can explain rather than merely report. A 500 with
#: an empty body still needs *something* to show, and the fallback path below
#: handles that without inventing a cause.
_TRANSPORT_MESSAGES = {
    "connection": "Could not reach the search API. Is the backend running?",
    "timeout": "The search API did not respond in time.",
}


def describe_error(exc: Exception) -> dict[str, str]:
    """Turn an exception into what the page should show.

    Returns a dict rather than a string so a caller can render a heading, a
    body, and an optional hint separately, and so the wording lives in one
    place that is testable.

    The rule this function enforces: it reports what the server said, and when
    the server said nothing it says *that* rather than guessing. A UI that
    renders "no results found" on a 503 is reporting a broken backend as a
    working one, which is the single most misleading thing this page could do.
    """
    if not isinstance(exc, ApiError):
        return {
            "heading": "Unexpected error",
            "detail": str(exc) or type(exc).__name__,
            "hint": "This is a client-side failure, not a response from the search API.",
        }

    if exc.transport_failure:
        reason = _TRANSPORT_MESSAGES.get(
            "timeout" if exc.error_code == "timeout" else "connection",
            exc.message,
        )
        return {
            "heading": "Backend unreachable",
            "detail": reason,
            "hint": f"Expected an API at {exc.error_code or 'the configured URL'}.",
        }

    heading = _headings.get(exc.status_code or 0, "Request failed")
    detail = exc.message
    hints = {
        400: "The query was rejected before retrieval ran. Check the query text.",
        413: "The request body exceeded the server's configured size limit.",
        422: "A field failed validation. The message names the field.",
        429: "The server's rate limit was reached. Wait for the window to reset.",
        503: (
            "The retrieval stack is unavailable. This is not an empty result set, "
            "and retrying the same query will not help until the server's "
            "artifacts are loaded."
        ),
    }
    result = {"heading": heading, "detail": detail, "hint": hints.get(exc.status_code or 0, "")}
    if exc.request_id:
        result["hint"] = f"{result['hint']} Request id: {exc.request_id}.".strip()
    return result


_headings = {
    400: "Query rejected",
    404: "Endpoint not found",
    413: "Request too large",
    422: "Validation failed",
    429: "Rate limited",
    500: "Server error",
    503: "Search backend unavailable",
}


# --------------------------------------------------------------------------- #
# View models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RetrieverDetail:
    """One retriever's contribution to one result, as the UI renders it.

    Every field here comes straight from ``provenance`` in the response. Nothing
    is inferred: a retriever that did not return this document is simply absent
    from the list, and that absence is the fact worth showing.
    """

    retriever: str
    #: 0-based, as the retriever produced it. This is a *different* rank from
    #: ``ResultView.rank``: the fused rank is already 1-based (the application
    #: layer converts once, at ``_to_hit``), while every retriever's own rank
    #: stays 0-based. The +1 happens here, at display time, so both values in
    #: this object remain the server's own.
    rank: int
    raw_score: float
    rrf_contribution: float


@dataclass(slots=True)
class ResultView:
    """One result, ready to render."""

    #: **1-based**, and already so on the wire. ``ApplicationSearchService``
    #: converts the engine's 0-based rank once, at the boundary, so the API's
    #: ``SearchHit.rank`` starts at 1. Adding 1 here would make the first
    #: result display as #2.
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    retrievers: list[str] = field(default_factory=list)
    provenance: list[RetrieverDetail] = field(default_factory=list)
    explanation_lines: list[str] = field(default_factory=list)

    @property
    def found_by_both(self) -> bool:
        """True when both retrievers returned this document.

        Computed from the provenance list rather than from a server field,
        because provenance is the authoritative record of which retrievers
        contributed. A document found by one retriever is not a lesser result,
        so the UI labels the count rather than ranking on it.
        """
        return len(self.provenance) > 1


@dataclass(slots=True)
class SearchView:
    """A whole search response, ready to render."""

    results: list[ResultView]
    original_query: str
    processed_query: str
    took_ms: float
    document_count: int
    returned_count: int
    requested_top_k: int
    empty_reason: str | None
    query_processing_enabled: bool
    was_modified: bool
    was_corrected: bool
    corrections: list[dict[str, Any]] = field(default_factory=list)
    expansions: list[dict[str, Any]] = field(default_factory=list)
    query_processing_lines: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.results


def _provenance_from(raw: Any) -> list[RetrieverDetail]:
    """Read the provenance list, tolerating either shape it can arrive in.

    ``SearchHit.to_dict()`` emits provenance as a **list** of entries that each
    carry their own ``retriever`` field, and that is what the API returns. The
    dict-keyed form appears in no current response; it is handled here only so a
    future reshaping of the payload degrades to a partial panel rather than a
    page that raises. Anything unrecognised is dropped, not guessed at.
    """
    if isinstance(raw, dict):
        return [
            RetrieverDetail(
                retriever=name,
                rank=int(entry.get("rank", 0)),
                raw_score=float(entry.get("raw_score", 0.0)),
                rrf_contribution=float(entry.get("rrf_contribution", 0.0)),
            )
            for name, entry in raw.items()
            if isinstance(entry, dict)
        ]
    if not isinstance(raw, list):
        return []
    out: list[RetrieverDetail] = []
    for entry in raw:
        if not isinstance(entry, dict) or "retriever" not in entry:
            continue
        out.append(
            RetrieverDetail(
                retriever=str(entry["retriever"]),
                rank=int(entry.get("rank", 0)),
                raw_score=float(entry.get("raw_score", 0.0)),
                rrf_contribution=float(entry.get("rrf_contribution", 0.0)),
            )
        )
    return out


def build_result_view(raw: Any, explanations: dict[str, list[str]] | None = None) -> ResultView:
    """Build one :class:`ResultView` from a single result object.

    Args:
        raw: One entry of ``results`` from the response.
        explanations: ``doc_id -> lines`` from ``result_explanations``, present
            only on ``/search/explain``. The backend wrote those lines; this
            passes them through unchanged rather than writing its own, because
            the guarantees about what an explanation is allowed to claim are
            enforced and tested on the server side.
    """
    doc_id = str(raw.get("doc_id", ""))
    lines = (explanations or {}).get(doc_id, [])
    sources = raw.get("sources") or []
    return ResultView(
        rank=int(raw.get("rank", 0)),
        doc_id=doc_id,
        title=str(raw.get("title", "(untitled)")),
        snippet=str(raw.get("snippet", "")),
        score=float(raw.get("score", 0.0)),
        retrievers=[str(s) for s in sources],
        provenance=_provenance_from(raw.get("provenance")),
        explanation_lines=list(lines),
    )


def build_search_view(payload: dict[str, Any], *, explain: bool = False) -> SearchView:
    """Build a :class:`SearchView` from a decoded response.

    Args:
        payload: The JSON body from ``/search`` or ``/search/explain``.
        explain: Whether to pick up ``result_explanations`` and ``caveats``.
            Passing them on a plain ``/search`` response yields an empty list
            rather than an error, so the caller may not need to know which
            endpoint it used.
    """
    explanations: dict[str, list[str]] = {}
    caveats: list[str] = []
    qp_lines: list[str] = []
    if explain:
        for entry in payload.get("result_explanations") or []:
            if isinstance(entry, dict) and "doc_id" in entry:
                explanations[str(entry["doc_id"])] = [
                    str(line) for line in entry.get("lines") or []
                ]
        caveats = [str(c) for c in payload.get("caveats") or []]
        qp = payload.get("query_processing") or {}
        if isinstance(qp, dict):
            qp_lines = [str(line) for line in qp.get("lines") or []]

    metadata = payload.get("metadata") or {}
    return SearchView(
        results=[
            build_result_view(raw, explanations)
            for raw in payload.get("results") or []
            if isinstance(raw, dict)
        ],
        original_query=str(payload.get("original_query", "")),
        processed_query=str(payload.get("processed_query", "")),
        took_ms=float(metadata.get("took_ms", 0.0)),
        document_count=int(metadata.get("document_count", 0)),
        returned_count=int(metadata.get("returned_count", 0)),
        requested_top_k=int(metadata.get("requested_top_k", 0)),
        empty_reason=metadata.get("empty_reason"),
        query_processing_enabled=bool(metadata.get("query_processing_enabled", False)),
        was_modified=bool(metadata.get("was_modified", False)),
        was_corrected=bool(metadata.get("was_corrected", False)),
        corrections=list(metadata.get("corrections") or []),
        expansions=list(metadata.get("expansions") or []),
        query_processing_lines=qp_lines,
        caveats=caveats,
    )


def build_health_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``/health`` or ``/health/ready`` into the fields the page shows.

    Returns a dict with fixed keys rather than the raw payload so the renderer
    cannot accidentally display a field the backend did not send, and so a
    missing field reads as ``None`` in one place instead of a ``KeyError``.
    """
    return {
        "ready": bool(payload.get("ready", False)),
        "status": str(payload.get("status", "unknown")),
        "backend": payload.get("backend"),
        "documents": payload.get("documents"),
        "retrievers": payload.get("retrievers") or [],
        "startup_seconds": payload.get("startup_seconds"),
        "rate_limit_per_minute": payload.get("rate_limit_per_minute"),
    }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class ApiClient:
    """A thin HTTP client for the search API.

    Args:
        base_url: The backend root. Trailing slashes are stripped.
        timeout: Per-request timeout in seconds.
        session: An injected :class:`requests.Session`, so tests can supply a
            transport double without monkeypatching the module.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = normalise_base_url(base_url)
        self.timeout = timeout
        self._session = session or requests.Session()

    # -- plumbing ---------------------------------------------------------- #

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Perform one request and return the decoded body.

        Every failure mode is normalised into :class:`ApiError` so callers have
        exactly one exception type to handle, and so no failure can escape as a
        raw ``requests`` exception with a URL the user did not type.
        """
        try:
            response = self._session.request(
                method, self._url(path), timeout=self.timeout, **kwargs
            )
        except requests.exceptions.Timeout as exc:
            raise ApiError(
                message="the request timed out",
                error_code="timeout",
                transport_failure=True,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(
                message=str(exc),
                error_code=self.base_url,
                transport_failure=True,
            ) from exc

        request_id = response.headers.get("X-Request-ID")
        if not response.ok:
            raise ApiError(**_error_fields(response, request_id))
        try:
            return response.json()
        except ValueError as exc:
            # A 200 with a body that is not JSON is a misconfigured or wrong
            # server, not a result. Saying "0 results" here would be a lie.
            raise ApiError(
                message=f"the server returned a {response.status_code} that was not JSON",
                status_code=response.status_code,
                error_code="invalid_json",
                request_id=request_id,
            ) from exc

    # -- endpoints --------------------------------------------------------- #

    def health(self) -> dict[str, Any]:
        """``GET /health``. Returns the payload; a 503 raises.

        Readiness is carried by the status code by design, so a not-ready
        backend raises rather than returning a dict with ``ready: false``. The
        caller distinguishes the two cases by exception type.
        """
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        """``GET /stats``."""
        return self._request("GET", "/stats")

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        query_processing: bool | None = None,
        explain: bool = True,
    ) -> dict[str, Any]:
        """Run a search.

        Args:
            query: The text the user typed, sent verbatim.
            top_k: Result count, or ``None`` for the server's default.
            query_processing: Force the query pipeline on or off for this
                request, or ``None`` for the server's configured default.
            explain: Use ``/search/explain`` rather than ``/search``. The
                results are identical either way; the explained endpoint also
                returns per-result provenance lines and caveats, which is why
                the UI asks for it by default.
        """
        body: dict[str, Any] = {"query": query}
        if top_k is not None:
            body["top_k"] = top_k
        if query_processing is not None:
            body["query_processing"] = query_processing
        path = "/search/explain" if explain else "/search"
        return self._request("POST", path, json=body)


def _error_fields(response: Any, request_id: str | None) -> dict[str, Any]:
    """Extract error fields from a failed response, tolerating any body.

    The API's envelope is ``{"error": {...}, "request_id": ...}``, but a
    reverse proxy in front of it will not produce that shape. Falling back to a
    status-derived message keeps a proxy's HTML error page from being displayed
    as if it were the application's own answer.
    """
    fallback = f"HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return {"message": fallback, "status_code": response.status_code, "request_id": request_id}
    if not isinstance(payload, dict):
        return {"message": fallback, "status_code": response.status_code, "request_id": request_id}
    error = payload.get("error")
    if not isinstance(error, dict):
        return {"message": fallback, "status_code": response.status_code, "request_id": request_id}
    return {
        "message": str(error.get("message") or fallback),
        "status_code": response.status_code,
        "error_code": str(error["code"]) if error.get("code") else None,
        "request_id": (
            str(payload.get("request_id") or request_id) if request_id is None else request_id
        ),
    }


__all__ = [
    "API_BASE_URL_ENV",
    "API_BASE_URL_ENV_ALIAS",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "ApiClient",
    "ApiError",
    "ResultView",
    "RetrieverDetail",
    "SearchView",
    "build_health_view",
    "build_result_view",
    "build_search_view",
    "default_base_url",
    "describe_error",
    "normalise_base_url",
]
