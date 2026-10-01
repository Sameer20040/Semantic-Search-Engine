"""Hybrid retrieval: fuse lexical and dense retrieval with RRF.

    query
      ├── BM25Searcher.search()          → ranked doc ids
      ├── SemanticSearchService.search() → ranked doc ids
      └── fuse()                         → ranked HybridSearchResult

This module is orchestration only. It contains no BM25, no embeddings, no FAISS,
and no second document lookup: the two collaborators already do each of those
and are independently tested. What is added here is the thing neither can do
alone -- combine two orderings that are each meaningful and jointly
incomparable.

On candidate depth
------------------
Asking each retriever for exactly ``top_k`` candidates and fusing those is
wrong, and silently so. RRF rewards a document that *both* retrievers rank
highly, and a document ranked 11th by both is a stronger fusion signal than one
ranked 1st by one retriever and unseen by the other. With
``candidate_k == top_k`` that consensus is structurally unavailable: the two
lists can only ever overlap on the same 10 documents, and the pool encodes
"best in one list" rather than "best in both".

So each retriever is asked for ``candidate_k`` (default 100, from
``hybrid.candidate_pool``) and fusion picks the best ``top_k``. The cost is
bounded and small: BM25 already scores the whole corpus and truncates, and the
vector index already searches and truncates, so a larger ``k`` changes only how
many results are *materialised*, not the work done. Measured in Phase 7 --
the extra depth costs well under a millisecond against a ~17 ms query encode.

``candidate_k`` is never below ``top_k``: asking a retriever for fewer
candidates than the number of results requested is a contradiction, and would
silently cap the response at ``candidate_k`` instead of failing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from semantic_search.core.config import BM25Config, HybridConfig, SearchConfig
from semantic_search.core.exceptions import ConfigurationError, EmptyQueryError, SearchError
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid_models import HybridSearchResult, build_hybrid_result
from semantic_search.search.models import SearchResult
from semantic_search.search.rrf import FusedCandidate, RetrievalRun, fuse
from semantic_search.search.semantic import SemanticSearchService

logger = get_logger(__name__)

#: Retriever names. These are the keys in every provenance dict, so they are
#: constants rather than string literals sprinkled through the module.
BM25 = "bm25"
SEMANTIC = "semantic"

#: What a retriever contributed to one query: a fusion input and the raw
#: scores kept alongside it for provenance only.
RetrievalOutput = tuple[RetrievalRun, dict[str, float]]


class HybridSearchService:
    """Rank-fuse a lexical and a dense retriever into one ranked list."""

    def __init__(
        self,
        bm25: BM25Searcher | None,
        semantic: SemanticSearchService | None,
        *,
        hybrid_config: HybridConfig | None = None,
        search_config: SearchConfig | None = None,
    ) -> None:
        """Bind the retrievers.

        Args:
            bm25: Lexical retriever, or ``None`` for a semantic-only service.
            semantic: Dense retriever, or ``None`` for a lexical-only service.
                At least one must be present; a service with neither has no
                ranking to produce.
            hybrid_config: ``rrf_k`` and ``candidate_pool``.
            search_config: Result shaping. Defaults to the semantic service's
                config when one is attached, so the two cannot disagree about
                ``default_top_k``.

        Raises:
            ConfigurationError: No retriever, retrievers covering different
                corpora, or a negative ``rrf_k`` / empty ``candidate_pool``.
        """
        if bm25 is None and semantic is None:
            raise ConfigurationError(
                "HybridSearchService needs at least one retriever; both were None."
            )

        self.bm25 = bm25
        self.semantic = semantic
        self.config = hybrid_config or HybridConfig()
        self.search_config = search_config or (semantic.config if semantic else SearchConfig())

        # The two retrievers index the same corpus. If they do not, fusion
        # would be combining rows that mean different documents and every
        # result would be quietly wrong.
        if bm25 is not None and semantic is not None and len(bm25.store) != len(semantic.store):
            raise ConfigurationError(
                f"BM25 covers {len(bm25.store)} documents but the semantic index covers "
                f"{len(semantic.store)}. Fusion assumes both index the same corpus; "
                f"rebuild them together with `python scripts/build_index.py`."
            )
        if self.config.rrf_k < 0:
            raise ConfigurationError(f"hybrid.rrf_k must be non-negative, got {self.config.rrf_k}.")
        if self.config.candidate_pool < 1:
            raise ConfigurationError(
                f"hybrid.candidate_pool must be at least 1, got {self.config.candidate_pool}."
            )

        logger.info(
            "hybrid.ready",
            extra={
                "event": "hybrid.ready",
                "retrievers": self.retriever_names,
                "rrf_k": self.config.rrf_k,
                "candidate_pool": self.config.candidate_pool,
            },
        )

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        bm25: BM25Searcher | None = None,
        semantic: SemanticSearchService | None = None,
    ) -> HybridSearchService:
        """Assemble a hybrid service from configuration.

        Both retrievers are loaded here so the API, the CLI, the probe, and the
        benchmark all end up with the same artifact set. ``BM25Searcher`` is
        built against an *injected* index and store rather than via
        :meth:`~semantic_search.index.bm25.BM25Searcher.from_documents`, so the
        two retrievers share one :class:`DocumentStore` and one in-memory copy
        of the corpus instead of two.
        """
        if bm25 is None and semantic is None:
            store = DocumentStore.load_jsonl(settings.data.corpus_file)
            bm25 = cls._load_bm25(settings, store)
            semantic = SemanticSearchService.from_settings(settings, store=store)
        elif bm25 is None:
            semantic = semantic
            assert semantic is not None
            bm25 = cls._load_bm25(settings, semantic.store)
        elif semantic is None:
            bm25 = bm25
            semantic = SemanticSearchService.from_settings(settings, store=bm25.store)

        return cls(bm25, semantic, hybrid_config=settings.hybrid, search_config=settings.search)

    @staticmethod
    def _load_bm25(settings: Any, store: DocumentStore) -> BM25Searcher:
        """Load the persisted BM25 index against an existing document store."""
        config = BM25Config(**settings.bm25.model_dump())
        index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, config)
        return BM25Searcher(index, store)

    # ----------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------- #
    @property
    def retriever_names(self) -> list[str]:
        names: list[str] = []
        if self.bm25 is not None:
            names.append(BM25)
        if self.semantic is not None:
            names.append(SEMANTIC)
        return names

    @property
    def store(self) -> DocumentStore:
        """The shared document store, for callers that need the corpus."""
        if self.semantic is not None:
            return self.semantic.store
        assert self.bm25 is not None
        return self.bm25.store

    def info(self) -> dict[str, Any]:
        """A snapshot for the ``/stats`` endpoint."""
        return {
            "retrievers": self.retriever_names,
            "rrf_k": self.config.rrf_k,
            "candidate_pool": self.config.candidate_pool,
            "documents": len(self.store),
        }

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #
    def search(self, query: str, top_k: int | None = None) -> list[HybridSearchResult]:
        """Fuse both retrievers and return the best ``top_k`` documents.

        An empty query is rejected here by the *same* validator the semantic
        service uses, called once. Validating twice would mean two places to
        keep in step, and the failure mode of them drifting apart -- one
        retriever searching a normalised string while the other searches the
        raw one -- is a fusion bug wearing a validation bug's clothes.
        """
        if self.semantic is not None:
            cleaned = self.semantic.validate_query(query)
            limit = self.semantic.validate_top_k(
                self.search_config.default_top_k if top_k is None else top_k
            )
        else:
            assert self.bm25 is not None
            cleaned = self.semanticless_validate(query)
            limit = self.search_config.default_top_k if top_k is None else top_k
        return self.search_validated(cleaned, limit)

    def semanticless_validate(self, query: str) -> str:
        """Query normalisation for a BM25-only service.

        A lexical-only service has no :class:`SemanticSearchService` to borrow
        validation from, and :class:`BM25Index` deliberately returns ``[]``
        rather than raising for a blank query. This keeps the hybrid contract
        consistent: an empty query is an error, not an empty result page, so
        the API can map it to a 400 without a per-mode special case.
        """
        if not isinstance(query, str):
            raise SearchError(f"Query must be a string, got {type(query).__name__}.")
        cleaned = " ".join(query.split())
        if not cleaned:
            raise EmptyQueryError("Query must contain at least one non-whitespace character.")
        return cleaned

    def search_validated(self, query: str, top_k: int) -> list[HybridSearchResult]:
        """Fuse an already-validated query.

        Split from :meth:`search` so the benchmark harness can time the fusion
        stage without re-paying validation, and so a caller that validates once
        can reuse the result across both retrievers.
        """
        candidate_k = max(top_k, self.config.candidate_pool)

        started = time.perf_counter()
        bm25_output = self._run_bm25(query, candidate_k)
        after_bm25 = time.perf_counter()

        semantic_output = self._run_semantic(query, candidate_k)
        after_semantic = time.perf_counter()

        runs: list[RetrievalRun] = []
        raw_scores: dict[str, dict[str, float]] = {}
        for output in (bm25_output, semantic_output):
            if output is not None:
                run, scores = output
                runs.append(run)
                raw_scores[run.name] = scores

        candidates = fuse(runs, k=self.config.rrf_k)
        after_fusion = time.perf_counter()

        results = self._to_results(candidates[:top_k], raw_scores=raw_scores, query=query)
        after_results = time.perf_counter()

        logger.debug(
            "hybrid.search",
            extra={
                "event": "hybrid.search",
                "query_length": len(query),
                "candidate_k": candidate_k,
                "candidates": len(candidates),
                "results": len(results),
                "bm25_ms": round((after_bm25 - started) * 1000, 3),
                "semantic_ms": round((after_semantic - after_bm25) * 1000, 3),
                "fusion_ms": round((after_fusion - after_semantic) * 1000, 3),
                "resolve_ms": round((after_results - after_fusion) * 1000, 3),
            },
        )
        return results

    # ----------------------------------------------------------------- #
    # Retriever adapters
    # ----------------------------------------------------------------- #
    def _run_bm25(self, query: str, top_k: int) -> RetrievalOutput | None:
        """Run BM25 and strip it to the two things fusion consumes.

        ``BM25Result`` carries per-term scores for the explainability panel;
        fusion needs only the order and the total, so the per-term dict is
        dropped here rather than carried through the whole pipeline.
        """
        if self.bm25 is None:
            return None
        hits = self.bm25.search(query, top_k=top_k)
        scores = {hit.doc_id: hit.score for hit in hits}
        return (
            RetrievalRun(
                name=BM25,
                ranked_doc_ids=[hit.doc_id for hit in hits],
                scores=scores,
            ),
            scores,
        )

    def _run_semantic(self, query: str, top_k: int) -> RetrievalOutput | None:
        """Run the dense retriever and strip it to ids and scores."""
        if self.semantic is None:
            return None
        hits = self.semantic.search(query, top_k=top_k)
        scores = {hit.doc_id: hit.score for hit in hits}
        return (
            RetrievalRun(
                name=SEMANTIC,
                ranked_doc_ids=[hit.doc_id for hit in hits],
                scores=scores,
            ),
            scores,
        )

    # ----------------------------------------------------------------- #
    # Result assembly
    # ----------------------------------------------------------------- #
    def _to_results(
        self,
        candidates: Sequence[FusedCandidate],
        *,
        raw_scores: dict[str, dict[str, float]],
        query: str,
    ) -> list[HybridSearchResult]:
        """Attach documents and provenance to fused candidates.

        Ranks are assigned from the position in this list, so they are
        contiguous from zero -- the same invariant single-retriever results
        carry, and the one any future re-fusion or explainability panel relies
        on. When a candidate is dropped (see below) the remaining ranks stay
        contiguous because the counter is ``len(results)``, not ``enumerate``.
        """
        store = self.store
        query_terms = self._query_terms(query)
        results: list[HybridSearchResult] = []

        for candidate in candidates:
            row = store.row_of(candidate.doc_id)
            if row is None:
                # Both retrievers index the store, so this means the store was
                # swapped after construction. Skipping keeps the rest of the
                # response correct and says so loudly, rather than resolving a
                # wrong document against a valid rank.
                logger.error(
                    "hybrid.document_missing",
                    extra={"event": "hybrid.document_missing", "doc_id": candidate.doc_id},
                )
                continue
            document = store[row]

            base = SearchResult.from_document(
                document,
                score=candidate.rrf_score,
                rank=len(results),
                row=row,
                snippet_length=self.search_config.snippet_length,
                query_terms=query_terms,
            )
            results.append(
                build_hybrid_result(
                    candidate,
                    base,
                    {
                        name: scores.get(candidate.doc_id, 0.0)
                        for name, scores in raw_scores.items()
                    },
                    rank=len(results),
                )
            )
        return results

    def _query_terms(self, query: str) -> list[str] | None:
        """Tokens used to centre the snippet, from the BM25 tokenizer.

        Reusing the index's own tokenizer is the point: the excerpt then lands
        on the terms BM25 actually scored, rather than on a second analysis that
        might disagree. ``None`` when BM25 is not attached, which makes
        :meth:`Document.snippet` fall back to the document head.
        """
        if self.bm25 is None:
            return None
        return self.bm25.index.tokenizer.tokenize(query)
