"""The application-facing request and response contract.

This is the boundary the future FastAPI layer, the Streamlit UI, the evaluation
tooling, and the explainability panel will all consume. It is deliberately
**retrieval-agnostic**: nothing here is a FAISS row, a BM25 term score, a
``RetrievalRun``, or a mention of ``rrf_k``. A caller that can name those
concepts is already coupled to the retrieval layer, and the point of an
application service is that it does not have to be.

What is deliberately *not* here
-------------------------------
``filters``. The only filter-ish field in the codebase is
``SearchConfig.default_filters``, which nothing reads and no retriever
implements. Accepting a ``filters`` argument that is then ignored is worse than
not having one: a caller would reasonably believe a filter had been applied to
the results, and the silent-wrong-answer it produces is precisely the failure
this layer is supposed to make impossible. Filtering is a real requirement that
needs a real implementation, and it does not have one yet.

Ranks are 1-based here and 0-based everywhere below
--------------------------------------------------
``SearchResult.rank``, ``BM25Result.rank`` and the RRF arithmetic are all 0-based
on purpose: fusion consumes ranks, and Phase 7 established the ``+1`` in
exactly one place. This layer converts once, at the boundary, because a rank a
human reads should start at 1. The 0-based rank is not discarded -- it is
preserved per retriever in :attr:`SearchHit.provenance` -- so nothing is lost
and no re-ranking is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_search.core.exceptions import SearchError

#: Backend label used when the hybrid service is the one answering. Reported in
#: ``metadata.backend`` so a client can tell how a result was produced without
#: inspecting the engine.
HYBRID = "hybrid"


@dataclass(slots=True, frozen=True)
class SearchRequest:
    """One application-level search request.

    Attributes:
        query: The user's text, exactly as submitted. Validated for emptiness
            and length when the request is *executed*, not here, so that a
            request object is a faithful record of what was asked and the
            error a caller sees names the query they actually sent.
        top_k: How many results to return. ``None`` means "use the configured
            default" and is distinguishable from ``0``, which is an error.
        query_processing: ``None`` means "use the configured default"; ``True``
            or ``False`` select the Phase 8 pipeline or the raw baseline
            explicitly. This is what makes ON/OFF a runtime choice rather than a
            different deployment.
    """

    query: str
    top_k: int | None = None
    query_processing: bool | None = None

    def __post_init__(self) -> None:
        """Reject the inputs that are wrong in *type*, before any work happens.

        Emptiness and length are deliberately **not** checked here. Both belong
        to validators that already exist one layer down
        (:meth:`SemanticSearchService.validate_query` and
        :meth:`QueryProcessor._enforce_max_length`), and duplicating them here
        would create a second definition that could drift from the first. What
        is left is the part no lower layer owns: the shape of the request.
        """
        if not isinstance(self.query, str):
            raise SearchError(f"Query must be a string, got {type(self.query).__name__}.")
        if self.top_k is not None:
            # bool is a subclass of int, and `top_k=True` silently meaning 1
            # is the kind of coercion that hides a client's bug for months.
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise SearchError(f"top_k must be an integer, got {type(self.top_k).__name__}.")
            if self.top_k < 1:
                raise SearchError(f"top_k must be at least 1, got {self.top_k}.")
        if self.query_processing is not None and not isinstance(self.query_processing, bool):
            # A truthy non-bool here would silently pick a path; refusing it
            # turns "why did the typo get fixed when I disabled it?" into an
            # error at the boundary instead.
            raise SearchError(
                "query_processing must be True, False, or None, got "
                f"{type(self.query_processing).__name__}."
            )

    @property
    def wants_processing(self) -> bool | None:
        """The caller's explicit choice, if any. ``None`` means "default"."""
        return self.query_processing

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "query_processing": self.query_processing,
        }


@dataclass(slots=True, frozen=True)
class HitProvenance:
    """What one retriever contributed to one result.

    Carried through unchanged from
    :class:`~semantic_search.search.hybrid_models.RetrieverHit`. The raw scores
    are on different scales per retriever and must never be compared with each
    other; ``rrf_contribution`` is the part that sums, and
    :attr:`SearchHit.score` is the sum.
    """

    retriever: str
    #: 0-based, exactly as the retriever produced it. Not the 1-based
    #: application rank -- mixing the two is how an off-by-one gets shipped.
    rank: int
    raw_score: float
    rrf_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "raw_score": round(self.raw_score, 6),
            "rrf_contribution": round(self.rrf_contribution, 8),
        }


@dataclass(slots=True, frozen=True)
class SearchHit:
    """One application-level result.

    Attributes:
        doc_id: Stable document identifier from the store.
        title: Document title, copied from the store at assembly time.
        snippet: Query-aware excerpt, built by the same
            :meth:`Document.snippet` every other layer uses, so the text shown
            to a user is identical to the text an explainability panel would
            quote.
        rank: **1-based** display rank, contiguous from 1.
        score: The fused score, higher is better. Copied, never recomputed --
            recomputing it would be re-ranking by another name.
        sources: Which retrievers returned this document. Empty is impossible
            for a fused result; a candidate came from somewhere.
        provenance: Per-retriever detail, for the explainability panel.
    """

    doc_id: str
    title: str
    snippet: str
    rank: int
    score: float
    sources: tuple[str, ...] = ()
    provenance: tuple[HitProvenance, ...] = ()

    def provenance_for(self, retriever: str) -> HitProvenance | None:
        """Provenance for one named retriever, or ``None`` if it missed."""
        for entry in self.provenance:
            if entry.retriever == retriever:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "rank": self.rank,
            "score": round(self.score, 8),
            "sources": list(self.sources),
            "provenance": {entry.retriever: entry.to_dict() for entry in self.provenance},
        }


@dataclass(slots=True, frozen=True)
class SearchMetadata:
    """Everything a caller needs to interpret the response, and nothing more.

    The fields here exist so that a caller never has to *infer* state:

    * ``empty_reason`` distinguishes "your query matched nothing" from
      "something is broken". The second case never reaches this object -- it
      raises :class:`SearchBackendError` -- so ``empty_reason == "no_matches"``
      is a statement about a healthy system.
    * ``requested_top_k`` is the count **as asked for**, before clamping, so a
      caller that asked for 5000 and got 100 can tell that the limit was the
      application cap and not the corpus.
    * ``was_modified`` / ``was_corrected`` come straight from the Phase 8
      record, so "your query was unchanged" is representable as unchanged
      rather than as an empty corrections list pretending to be a success.
    """

    #: Wall-clock time for the whole request, in milliseconds.
    took_ms: float
    #: Which stack answered: ``"hybrid"`` today, single-retriever labels later.
    backend: str
    #: Documents in the corpus, so ``returned_count`` is interpretable.
    document_count: int
    #: Whether the query-processing pipeline ran for this request.
    query_processing_enabled: bool
    #: Did processing change the string that was searched?
    was_modified: bool
    #: Were any terms actually rewritten? Distinct from ``was_modified``, which
    #: is also true for a whitespace-only change.
    was_corrected: bool
    #: ``top_k`` as the caller asked for it, before clamping.
    requested_top_k: int
    returned_count: int
    #: Fewer results returned than asked for.
    truncated: bool
    #: The corpus was exhausted -- every document is in this response.
    corpus_exhausted: bool
    #: ``"no_matches"`` for a valid query with zero hits on a healthy backend;
    #: ``None`` whenever anything was returned.
    empty_reason: str | None
    #: Corrections and expansions exactly as the processor recorded them.
    corrections: tuple[dict[str, Any], ...] = ()
    expansions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "took_ms": round(self.took_ms, 3),
            "backend": self.backend,
            "document_count": self.document_count,
            "query_processing_enabled": self.query_processing_enabled,
            "was_modified": self.was_modified,
            "was_corrected": self.was_corrected,
            "requested_top_k": self.requested_top_k,
            "returned_count": self.returned_count,
            "truncated": self.truncated,
            "corpus_exhausted": self.corpus_exhausted,
            "empty_reason": self.empty_reason,
            "corrections": [dict(item) for item in self.corrections],
            "expansions": [dict(item) for item in self.expansions],
        }


@dataclass(slots=True, frozen=True)
class ApplicationSearchResponse:
    """A complete, serialisable answer to one :class:`SearchRequest`.

    ``original_query`` is the exact string the caller sent, never the normalized
    one. That is the contract a support conversation depends on: "I searched for
    X" has to be answerable with the X they typed.
    """

    original_query: str
    processed_query: str
    results: tuple[SearchHit, ...] = ()
    metadata: SearchMetadata | None = None

    def __len__(self) -> int:
        return len(self.results)

    @property
    def count(self) -> int:
        return len(self.results)

    def top_hit(self) -> SearchHit | None:
        """The first result, or ``None`` for an empty response."""
        return self.results[0] if self.results else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "results": [hit.to_dict() for hit in self.results],
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


#: Re-exported so callers can catch the error without importing two modules.
__all__ = [
    "HYBRID",
    "ApplicationSearchResponse",
    "HitProvenance",
    "SearchHit",
    "SearchMetadata",
    "SearchRequest",
]
