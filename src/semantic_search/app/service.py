"""The application-facing search service.

    query
      -> ApplicationSearchService   validate the request, choose the path
      -> SearchService               the Phase 8 orchestrator
      -> QueryProcessor              normalize / correct / expand
      -> HybridSearchService         BM25 + semantic, one string
      -> RRF
      -> ApplicationSearchResponse

What this layer adds, and what it refuses to add
------------------------------------------------
It adds exactly three things the layers below cannot supply to a caller: a
**request object** with validated inputs, a **response object** that does not
mention FAISS rows or ``rrf_k``, and **metadata** that makes "nothing matched"
distinguishable from "something is broken".

It deliberately adds no retrieval behaviour. It does not re-sort, does not
re-score, does not deduplicate, and does not drop or promote a result. The order
a caller sees is the order
:class:`~semantic_search.search.hybrid.HybridSearchService` produced, and the
one transformation applied to each result -- the 0-based rank becoming 1-based --
is a display conversion, not a re-ranking. A layer that reordered results
"for the application" would make the Phase 7 evidence describe a system that no
longer ships.

Why it wraps ``query.SearchService`` rather than replacing it
------------------------------------------------------------
Phase 8's ``SearchService`` is the tested ON/OFF mechanism: it owns processor
construction, the injected validator, and the raw baseline path. Replacing it
with a parallel implementation would leave two orchestrators to keep in step,
and the Phase 8 tests would be testing something the application no longer
runs. So this service *delegates*, and the Phase 8 type is still what executes.

Why the package is ``semantic_search.app`` and not ``semantic_search.search``
-------------------------------------------------------------------------
Because ``search/`` is the hybrid *retrieval* layer, and putting this file there
would place the application layer beside retrieval rather than above it --
inverting the dependency arrow that Phase 8 established and that
``query/processor.py``'s docstring states as the reason it imports nothing from
``semantic_search.search``. A test asserts the direction so the claim stays
checkable.

Failures are never disguised as empty results
---------------------------------------------
:func:`_as_backend_error` converts the infrastructure exceptions raised deep in
the stack -- a missing index, a corrupt artifact, a store/index disagreement, a
model mismatch -- into a single :class:`SearchBackendError` that propagates. A
valid query with no matches returns ``results=()`` with
``metadata.empty_reason == "no_matches"``. A broken engine raises. It never
returns an empty list.
"""

from __future__ import annotations

import time
from typing import Any

from semantic_search.app.models import (
    HYBRID,
    ApplicationSearchResponse,
    HitProvenance,
    SearchHit,
    SearchMetadata,
    SearchRequest,
)
from semantic_search.core.config import AppConfig, Settings
from semantic_search.core.exceptions import (
    DataError,
    SearchBackendError,
    SearchError,
    SearchIndexError,
)
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.query.service import SearchService
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.hybrid_models import HybridSearchResult

logger = get_logger(__name__)

#: Infrastructure failures that must not be reported as "no results". Every one
#: of these is a 5xx: retrying the same request later is reasonable, fixing the
#: request is not.
BACKEND_CAUSES: tuple[type[BaseException], ...] = (
    SearchIndexError,
    DataError,
)

#: Why an otherwise valid query returned nothing.
NO_MATCHES = "no_matches"


class ApplicationSearchService:
    """A stable, retrieval-agnostic search interface.

    Construct it **once**, at startup. The expensive parts -- the embedding
    model, the indexes, and the spelling vocabulary -- all belong to the
    collaborators it is given, and constructing one per request would turn the
    cheapest layer in the stack into the most expensive one.
    """

    def __init__(
        self,
        search_service: SearchService,
        app_config: AppConfig,
        *,
        document_count: int | None = None,
    ) -> None:
        """Wrap an existing Phase 8 :class:`SearchService`.

        Args:
            search_service: The Phase 8 orchestrator, already wired to a corpus
                and a hybrid service.
            app_config: Application-level limits and defaults.
            document_count: Corpus size, reported in metadata. Taken from the
                hybrid service when not given.
        """
        self._service = search_service
        self._config = app_config
        self._document_count = (
            document_count if document_count is not None else len(search_service.hybrid.store)
        )

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        store: DocumentStore,
        hybrid: HybridSearchService,
    ) -> ApplicationSearchService:
        """Build the application service over an already-assembled hybrid stack.

        The artifacts are loaded by the caller rather than here, matching the
        wiring every script and the future API layer already uses. Loading them
        in two places is how the probes end up measuring a different index from
        the one the server serves.
        """
        return cls(
            SearchService.from_settings(settings, store, hybrid),
            settings.app,
            document_count=len(store),
        )

    # ----------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------- #
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def search_service(self) -> SearchService:
        """The wrapped Phase 8 service, for callers that need its raw record."""
        return self._service

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #
    def search(self, request: SearchRequest) -> ApplicationSearchResponse:
        """Run one request end to end and return the application response.

        Args:
            request: A :class:`SearchRequest`. Its type-level fields were
                already checked at construction; the emptiness and length rules
                are enforced by the layers below, which own them.

        Returns:
            An :class:`ApplicationSearchResponse`. Zero results is a valid
            return, not an error.

        Raises:
            SearchError: The query was empty, too long, or otherwise unusable.
            SearchBackendError: The retrieval stack failed. Never returned as an
                empty result list.
        """
        if not isinstance(request, SearchRequest):
            raise SearchError(f"search() expects a SearchRequest, got {type(request).__name__}.")

        use_processing = self._resolve_processing(request)
        limit = self._resolve_top_k(request)
        started = time.perf_counter()

        try:
            if use_processing:
                inner = self._service.search(request.query, top_k=limit)
            else:
                inner = self._service.search_raw(request.query, top_k=limit)
        except SearchError:
            # A bad request is the caller's problem and stays a 400. It is
            # deliberately re-raised before the backend wrapper below, which
            # would otherwise report "retry later" for a request that can
            # never succeed however many times it is sent.
            raise
        except Exception as exc:
            # The one place this layer touches control flow. A retrieval failure
            # is re-raised as a single typed 503 so that every caller -- the
            # API, the UI, the evaluation harness -- can tell "your query matched
            # nothing" from "the engine is broken" without inspecting exception
            # types, and so no future handler can accidentally render a broken
            # engine as an empty result page.
            raise _as_backend_error(exc) from exc

        hits = tuple(_to_hit(result, position) for position, result in enumerate(inner.results))
        took_ms = (time.perf_counter() - started) * 1000.0

        return ApplicationSearchResponse(
            original_query=inner.query.original,
            processed_query=inner.query.query_for_retrieval,
            results=hits,
            metadata=self._metadata(
                inner,
                hits=hits,
                took_ms=took_ms,
                requested_top_k=request.top_k if request.top_k is not None else limit,
                limit=limit,
                processing_ran=use_processing,
            ),
        )

    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        *,
        query_processing: bool | None = None,
    ) -> ApplicationSearchResponse:
        """Convenience wrapper for callers with a bare query string.

        Builds a :class:`SearchRequest` and delegates. The convenience is real
        but it is also the easiest way to lose the request record, so it is
        offered alongside :meth:`search` rather than instead of it.
        """
        return self.search(
            SearchRequest(query=query, top_k=top_k, query_processing=query_processing)
        )

    # ----------------------------------------------------------------- #
    # Request resolution
    # ----------------------------------------------------------------- #
    def _resolve_processing(self, request: SearchRequest) -> bool:
        """Decide whether the Phase 8 pipeline runs for this request.

        Three-way, deliberately: ``True``/``False`` are the caller's explicit
        choice, and ``None`` means "whatever the deployment configured". The
        ON/OFF experiment in Phase 8 depends on being able to ask for the raw
        path on a service that has processing available, which a plain boolean
        config flag could not express per request.
        """
        if request.query_processing is None:
            return self._config.query_processing_enabled
        return request.query_processing

    def _resolve_top_k(self, request: SearchRequest) -> int:
        """Resolve and clamp ``top_k``.

        Clamping, not erroring, for the over-maximum case: a user asking for
        500 results from a 300-document corpus is asking a reasonable question.
        The clamp is still *visible* -- :attr:`SearchMetadata.requested_top_k`
        records the pre-clamp number, so "you asked for 500 and got 100" is a
        fact the caller can read rather than something it has to guess.
        """
        requested = self._config.default_top_k if request.top_k is None else request.top_k
        return min(requested, self._config.max_top_k)

    # ----------------------------------------------------------------- #
    # Response assembly
    # ----------------------------------------------------------------- #
    def _metadata(
        self,
        inner: Any,
        *,
        hits: tuple[SearchHit, ...],
        took_ms: float,
        requested_top_k: int,
        limit: int,
        processing_ran: bool,
    ) -> SearchMetadata:
        """Assemble the record that makes the response interpretable.

        Corrections and expansions are read straight off the Phase 8
        ``ProcessedQuery``. Nothing here decides whether a correction was *good*
        -- the report shows what happened, and the quality of it is unmeasured.
        """
        record = inner.query
        payload = record.to_dict()
        returned = len(hits)
        return SearchMetadata(
            took_ms=took_ms,
            backend=HYBRID,
            document_count=self._document_count,
            query_processing_enabled=processing_ran,
            was_modified=record.was_modified,
            was_corrected=record.was_corrected,
            requested_top_k=requested_top_k,
            returned_count=returned,
            truncated=returned < limit,
            corpus_exhausted=self._document_count > 0 and returned >= self._document_count,
            empty_reason=NO_MATCHES if returned == 0 else None,
            corrections=tuple(payload["corrections"]),
            expansions=tuple(payload["expansions"]),
        )

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    def info(self) -> dict[str, Any]:
        """Configuration snapshot, for a ``/stats`` endpoint or a probe."""
        return {
            "backend": HYBRID,
            "documents": self._document_count,
            "retrievers": self._service.hybrid.retriever_names,
            "default_top_k": self._config.default_top_k,
            "max_top_k": self._config.max_top_k,
            "max_query_chars": self._config.max_query_chars,
            "query_processing_enabled": self._config.query_processing_enabled,
            "query_processing": self._service.info(),
        }

    def health(self) -> dict[str, Any]:
        """A cheap readiness summary.

        Deliberately does *not* run a query. A health check that performs the
        full retrieval path turns a liveness probe into load, and the answer it
        wants -- "are the artifacts loaded and the two retrievers the same size"
        -- is already in memory by the time construction succeeded.
        """
        hybrid = self._service.hybrid
        return {
            "ready": True,
            "backend": HYBRID,
            "documents": self._document_count,
            "retrievers": hybrid.retriever_names,
            "hybrid_configured": bool(hybrid.retriever_names),
        }


# ----------------------------------------------------------------------- #
# Module-level helpers
# ----------------------------------------------------------------------- #
def _to_hit(result: HybridSearchResult, position: int) -> SearchHit:
    """Map one fused result onto the application contract.

    ``position`` is the position in the list the hybrid service returned, and
    the 1-based rank is derived from it. Deriving from position rather than
    from ``result.rank`` is deliberate: the two are equal in practice, and if
    they ever diverge the list order is the one a caller is actually looking at.
    """
    return SearchHit(
        doc_id=result.doc_id,
        title=result.title,
        snippet=result.snippet,
        rank=position + 1,
        score=result.score,
        sources=tuple(result.retrievers),
        provenance=tuple(
            HitProvenance(
                retriever=hit.retriever,
                rank=hit.rank,
                raw_score=hit.raw_score,
                rrf_contribution=hit.rrf_contribution,
            )
            for hit in result.hits
        ),
    )


def _as_backend_error(exc: BaseException) -> SearchBackendError:
    """Wrap a retrieval failure in a typed 503, keeping the original for the log.

    Only called for exceptions that are **not** :class:`SearchError` -- the
    caller filters those out first, because a bad request must stay a 400. If an
    empty query were reported as ``search_backend_error`` the caller would learn
    to retry a request that can never succeed. Only genuine infrastructure
    failures -- missing, corrupt, mismatched, or otherwise unusable artifacts --
    become a 503.

    The :class:`SearchBackendError` passthrough keeps the wrapper idempotent: an
    already-wrapped failure is not wrapped twice, so ``error.cause`` is always
    the *original* exception rather than an intermediate wrapper.
    """
    if isinstance(exc, SearchBackendError):
        return exc
    if isinstance(exc, BACKEND_CAUSES):
        return SearchBackendError(
            f"The search backend failed while serving a valid request: {exc}",
            details={"cause_type": type(exc).__name__},
            cause=exc,
        )
    return SearchBackendError(
        f"Unexpected failure in the search backend: {type(exc).__name__}: {exc}",
        details={"cause_type": type(exc).__name__},
        cause=exc,
    )


__all__ = ["NO_MATCHES", "ApplicationSearchService", "SearchBackendError"]
