"""Semantic (dense vector) search service.

Responsibility
--------------
This module is deliberately *only* orchestration. The encoder lives in
:mod:`semantic_search.embeddings.encoder`, the vector search in
:mod:`semantic_search.index.faiss_index`, and the documents in
:mod:`semantic_search.data.loader`. Nothing here re-implements any of them. What
this adds is the part none of them can own alone:

    query text -> validated -> embedded -> searched -> rows resolved to
    documents -> ranked SearchResult objects

Why the validation here
-----------------------
The service is the boundary where untrusted text arrives, so this is where an
empty query, an over-long query, or a `top_k` of zero is rejected -- once,
clearly, with a typed error the API can map to a 400. Downstream code may then
assume a non-empty query and a `1 <= top_k <= max_top_k`.

Why the model/index compatibility check matters
-----------------------------------------------
A FAISS index built by `all-MiniLM-L6-v2` cannot be searched by a query embedded
by any other model. The dimensions may even match (both 384-d), so FAISS will
happily return a ranking -- a confident, entirely meaningless one. The check
here is against the *index's recorded* model, reusing the sidecar written in
Phase 5 rather than a second, independently-maintained mechanism.

Score direction
----------------
``SearchResult.score`` is always higher-is-better. ``flat_l2`` reports a squared
Euclidean distance, so its scores are negated on the way out. For the default
``flat_ip`` over unit vectors nothing is transformed, and the score is a cosine
similarity.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from semantic_search.core.config import SearchConfig
from semantic_search.core.exceptions import (
    DataError,
    EmptyQueryError,
    IndexNotFoundError,
    QueryTooLongError,
    SearchError,
)
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.search.models import SearchResult

logger = get_logger(__name__)


class SemanticSearchService:
    """Dense-retrieval search over a FAISS index and its document store."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: FaissIndex,
        store: DocumentStore,
        *,
        search_config: SearchConfig | None = None,
        query_max_length: int = 512,
        strict_alignment: bool = True,
    ) -> None:
        """Join the three collaborators and verify they belong together.

        Args:
            encoder: Must be the model the index was built with. Checked.
            index: A built or loaded :class:`FaissIndex`.
            store: Documents in the same order as the index's rows.
            search_config: Result shaping (``default_top_k``, ``max_top_k``,
                ``snippet_length``).
            query_max_length: Reject longer queries. Mirrors the ingestion limit;
                a query far longer than a document can be is almost always a
                mistake or an attempt to smuggle a payload into the log.
            strict_alignment: Require the store to match the index size exactly.

        Raises:
            IndexNotFoundError: No documents loaded, so there is nothing to
                search. (The index-side equivalent is raised by ``FaissIndex``.)
            DataError: Store and index disagree in size, or the active
                embedding model does not match the one recorded in the index.
        """
        self.encoder = encoder
        self.index = index
        self.store = store
        self.config = search_config or SearchConfig()
        self.query_max_length = query_max_length
        self.strict_alignment = strict_alignment

        self._validate_components()

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    def _validate_components(self) -> None:
        """Fail at construction, not on the first user query.

        A misconfigured deployment should be caught at startup, where the log
        says what is wrong, rather than as a 500 on the first request an hour
        later.
        """
        if len(self.store) == 0:
            raise IndexNotFoundError(
                "The document store is empty, so there is nothing to search. "
                "Run `python scripts/download_data.py` then `make index`."
            )
        if self.strict_alignment:
            # Raises DataError with a "Rebuild the index" message on mismatch.
            self.store.validate_alignment(len(self.index))

        active_model = self.encoder.config.model_name
        index_model = self.index.metadata.embedding_model
        if index_model and index_model != active_model:
            raise DataError(
                f"The embedding model {active_model!r} does not match the model "
                f"{index_model!r} the index was built with. Rebuild the index, or "
                f"switch the active model, before serving search."
            )

        if self.index.dimension != self.encoder.dimension:
            raise DataError(
                f"The encoder produces {self.encoder.dimension}-dimensional vectors "
                f"but the index has dimension {self.index.dimension}."
            )

        logger.info(
            "semantic.ready",
            extra={
                "event": "semantic.ready",
                "documents": len(self.store),
                "dimension": self.index.dimension,
                "index_type": self.index.index_type,
                "embedding_model": active_model,
            },
        )

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        store: DocumentStore | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> SemanticSearchService:
        """Build a service from :class:`~semantic_search.core.config.Settings`.

        This is the wiring the API, the CLI, and the benchmark script all use, so
        they cannot drift apart in which artifacts they load. The encoder is
        created lazily by :class:`EmbeddingEncoder`, so nothing is loaded until
        the first query unless one is injected.
        """
        data_config = settings.data
        store = store or DocumentStore.load_jsonl(data_config.corpus_file)
        index = FaissIndex.load(
            data_config.index_dir / settings.faiss.index_filename,
            settings.faiss,
            expected_model=settings.embedding.model_name,
            expected_documents=len(store),
        )
        return cls(
            encoder or EmbeddingEncoder(settings.embedding),
            index,
            store,
            search_config=settings.search,
            query_max_length=settings.query.max_length,
        )

    # ----------------------------------------------------------------- #
    # Query handling
    # ----------------------------------------------------------------- #
    def validate_query(self, query: str) -> str:
        """Normalise and reject unusable queries.

        Collapsing internal whitespace matters for more than tidiness: without
        it, ``"orbital  mechanics"`` and ``"orbital mechanics"`` cache as two
        different keys and produce two FAISS passes for one user intent.

        Public because :class:`~semantic_search.search.hybrid.HybridSearchService`
        reuses it. Two validators that differ in whitespace handling would let
        one retriever search a different string from the other, which is a
        fusion bug wearing a validation bug's clothes.
        """
        if not isinstance(query, str):
            raise SearchError(f"Query must be a string, got {type(query).__name__}.")
        cleaned = " ".join(query.split())
        if not cleaned:
            raise EmptyQueryError("Query must contain at least one non-whitespace character.")
        if len(cleaned) > self.query_max_length:
            raise QueryTooLongError(
                f"Query is {len(cleaned)} characters; the maximum is " f"{self.query_max_length}."
            )
        return cleaned

    def validate_top_k(self, top_k: int) -> int:
        """Clamp ``top_k`` into a sane, non-empty range.

        Requesting more results than the corpus holds is a reasonable thing for
        a user to do and is answered by returning everything, not by erroring.
        An upper cap still applies so a single request cannot ask the process to
        materialise the whole corpus in a response body.

        Public for the same reason as :meth:`validate_query`.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise SearchError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise SearchError(f"top_k must be at least 1, got {top_k}.")
        return min(top_k, self.config.max_top_k)

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the ``top_k`` documents most similar to ``query``.

        The FAISS rank order is preserved exactly; results are never re-sorted
        here, because for an exact index the index order *is* the similarity
        order and a re-sort could only get it wrong.
        """
        cleaned = self.validate_query(query)
        limit = self.validate_top_k(self.config.default_top_k if top_k is None else top_k)

        started = time.perf_counter()
        vector = self.encoder.encode_query(cleaned)
        encode_seconds = time.perf_counter() - started

        started = time.perf_counter()
        hits = self.index.search(vector, top_k=limit)
        search_seconds = time.perf_counter() - started

        started = time.perf_counter()
        results = self._to_results(hits, query=cleaned)
        retrieve_seconds = time.perf_counter() - started

        logger.debug(
            "semantic.search",
            extra={
                "event": "semantic.search",
                "query_length": len(cleaned),
                "results": len(results),
                "encode_ms": round(encode_seconds * 1000, 3),
                "faiss_ms": round(search_seconds * 1000, 3),
                "retrieve_ms": round(retrieve_seconds * 1000, 3),
            },
        )
        return results

    def _to_results(self, hits: Sequence[Any], *, query: str) -> list[SearchResult]:
        """Turn FAISS hits into ranked :class:`SearchResult` objects.

        FAISS pads short result sets with ``row = -1``; :class:`FaissIndex`
        already drops those, and this method asserts the invariant rather than
        trusting it, because a ``-1`` reaching ``store[-1]`` would silently
        return the *last* document in the corpus -- a wrong result that looks
        perfectly normal.
        """
        negate = self.index.index_type == "flat_l2"
        results: list[SearchResult] = []
        for hit in hits:
            if hit.row < 0:
                # Unreachable via FaissIndex.search; cheap insurance against a
                # future index type that forgets to filter its padding.
                logger.warning("semantic.negative_row_skipped", extra={"row": hit.row})
                continue
            document = self.store[hit.row]
            score = -hit.score if negate else hit.score
            results.append(
                SearchResult.from_document(
                    document,
                    score=score,
                    rank=len(results),
                    row=hit.row,
                    snippet_length=self.config.snippet_length,
                    scores={"semantic": score},
                )
            )
        return results

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    def info(self) -> dict[str, Any]:
        """A snapshot for the ``/stats`` endpoint."""
        return {
            "documents": len(self.store),
            "index": self.index.stats(),
            "encoder": self.encoder.info(),
        }

    @property
    def corpus_size(self) -> int:
        return len(self.store)
