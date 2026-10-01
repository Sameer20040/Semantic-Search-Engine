"""The application layer: a stable, retrieval-agnostic search contract.

This package is the **top** of the engine. Everything else exists to serve it.

    app          ApplicationSearchService, SearchRequest, ApplicationSearchResponse
      -> query   QueryProcessor, the Phase 8 orchestrator
      -> search  HybridSearchService (BM25 + semantic, fused with RRF)
      -> index   BM25Index, FaissIndex
      -> data    DocumentStore

The arrow only points downward. ``query/``, ``search/``, ``index/``, and
``data/`` must never import this package: a retriever that knew about
``SearchRequest`` could no longer be tested or reused without the application
layer, and the ON/OFF experiments Phase 8 relies on would need the app's
configuration to run. A test enforces this by parsing the import graph rather
than trusting the docstring.

Public surface
--------------
``ApplicationSearchService``
    The service to construct once at startup and call per request.
``SearchRequest``
    The validated request object.
``ApplicationSearchResponse``, ``SearchHit``, ``SearchMetadata``,
``HitProvenance``
    The response types. None of them mentions FAISS, BM25, or RRF; a caller
    using this package cannot tell which retrievers are behind it, which is the
    point -- a later Phase 7/8 change must not ripple into the UI.
``SearchBackendError``
    Re-exported from :mod:`semantic_search.core.exceptions` so a caller can
    distinguish "your query matched nothing" from "the engine is broken" by
    importing one module.

Not exported: ``HybridSearchService`` and friends. They are reachable from
``semantic_search.search`` by anyone who genuinely needs them.
"""

from semantic_search.app.models import (
    HYBRID,
    ApplicationSearchResponse,
    HitProvenance,
    SearchHit,
    SearchMetadata,
    SearchRequest,
)
from semantic_search.app.service import NO_MATCHES, ApplicationSearchService
from semantic_search.core.exceptions import SearchBackendError

__all__ = [
    "HYBRID",
    "NO_MATCHES",
    "ApplicationSearchResponse",
    "ApplicationSearchService",
    "HitProvenance",
    "SearchBackendError",
    "SearchHit",
    "SearchMetadata",
    "SearchRequest",
]
