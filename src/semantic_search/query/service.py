"""The orchestrator: query processing in front of hybrid retrieval.

    User Query
      -> QueryProcessor            normalize, correct, expand
      -> HybridSearchService       BM25 + semantic, one string
      -> RRF
      -> results

:class:`SearchService` is the thing that makes the architectural ordering
visible. It is deliberately thin: it calls the processor, calls the hybrid
service with whatever single string the processor returned, and packages the
two together. It contains no retrieval logic, no ranking, no scoring, and no
opinion about the results.

The two paths
-------------
``search`` and ``search_raw`` differ in exactly one thing -- whether
``QueryProcessor`` runs. Both hand their query to the same
``HybridSearchService.search``, which validates it and fans it out to both
retrievers. That is the whole comparison an ON/OFF experiment needs, and
because the raw path is always present, "processing is off" is a runtime flag
rather than a different deployment.

One string, two retrievers
--------------------------
The processed string is passed once. ``HybridSearchService`` already fans a
single query out to both retrievers internally (established in Phase 7), so
there is no seam here for the two to diverge, and the ON/OFF experiment in
``scripts/probe_query_processing.py`` can assert it directly: capture what each
retriever receives and compare the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semantic_search.core.config import Settings
from semantic_search.data.loader import DocumentStore
from semantic_search.query.models import ProcessedQuery
from semantic_search.query.processor import QueryProcessor
from semantic_search.query.spelling import SpellCorrector, build_vocabulary
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.hybrid_models import HybridSearchResult


@dataclass(slots=True, frozen=True)
class SearchResponse:
    """Results plus the record of how the query that produced them was built.

    Every field is present whether or not processing did anything.
    ``query.to_dict()`` on an untouched query reports the same string four
    times and two empty lists, which is the honest answer and lets a client
    render "your query" without a special case for the unchanged case.
    """

    query: ProcessedQuery
    results: list[HybridSearchResult] = field(default_factory=list)

    @property
    def top_k(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query.to_dict(), "results": [r.to_dict() for r in self.results]}


class SearchService:
    """Query processing in front of hybrid retrieval.

    Construction is where the expensive part happens: the spelling vocabulary is
    built from the corpus, which is O(corpus text). Build this once at startup
    and share it; constructing one per request would turn a free feature into
    the most expensive thing in the request path.
    """

    def __init__(
        self,
        hybrid: HybridSearchService,
        processor: QueryProcessor,
        settings: Settings | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.processor = processor
        self.settings = settings

    @classmethod
    def from_settings(
        cls, settings: Settings, store: DocumentStore, hybrid: HybridSearchService
    ) -> SearchService:
        """Build a service with its corrector wired to the real corpus.

        The validator is taken from the hybrid service's own semantic retriever
        rather than reimplemented, so the definition of "a valid query" has one
        home. When there is no semantic retriever -- a BM25-only deployment --
        the hybrid service's ``semanticless_validate`` is used instead, which
        enforces the same empty-query rule.
        """
        if hybrid.semantic is not None:
            validate = hybrid.semantic.validate_query
        else:
            validate = hybrid.semanticless_validate

        corrector: SpellCorrector | None = None
        if settings.query.spell_check and len(store):
            vocabulary = build_vocabulary(store)
            corrector = SpellCorrector(
                vocabulary,
                enabled=settings.query.spell_check,
                preserve_proper_nouns=settings.query.preserve_proper_nouns,
                protect_stopwords=settings.query.protect_stopwords,
                min_correction_score=settings.query.min_correction_score,
                max_edit_distance=settings.query.min_edit_distance,
                min_candidate_df=settings.query.min_candidate_df,
            )

        processor = QueryProcessor(settings.query, corrector=corrector, validate=validate)
        return cls(hybrid, processor, settings)

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Process the query, then retrieve with the processed string.

        This is the ON path. With ``query.enabled = False`` inside the
        processor it degrades to exactly :meth:`search_raw`, which is asserted
        by the tests rather than merely intended.
        """
        processed = self.processor.process(query)
        results = self.hybrid.search(processed.query_for_retrieval, top_k=top_k)
        return SearchResponse(query=processed, results=results)

    def search_raw(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Retrieve with the query exactly as typed. The always-available baseline.

        The string goes straight to ``HybridSearchService.search``, which
        validates it -- so this is not an unvalidated shortcut, it is the same
        call the processed path makes, minus the processing. The returned
        ``ProcessedQuery`` is a faithful record of that: every field equals the
        input, ``enabled`` is False, and the collections are empty.

        This path is not a debug convenience. It is what makes the ON/OFF
        comparison a controlled experiment rather than two deployments.
        """
        results = self.hybrid.search(query, top_k=top_k)
        return SearchResponse(
            query=ProcessedQuery(
                original=query,
                normalized=query,
                corrected=query,
                query_for_retrieval=query,
                enabled=False,
            ),
            results=results,
        )

    def info(self) -> dict[str, Any]:
        """Configuration report, for a ``/stats``-style endpoint or a probe."""
        processor = self.processor
        corrector = processor.corrector
        config = processor.config
        return {
            "enabled": config.enabled,
            "stages": [
                "normalize",
                "spell_correct" if config.spell_check else None,
                "expand",
            ],
            "spell_check": config.spell_check,
            "preserve_proper_nouns": config.preserve_proper_nouns,
            "protect_stopwords": config.protect_stopwords,
            "min_correction_score": config.min_correction_score,
            "max_edit_distance": config.min_edit_distance,
            "min_candidate_df": config.min_candidate_df,
            "max_length": config.max_length,
            # Recorded always; applied only when the flag is on.
            "synonym_expansion_recorded": True,
            "synonym_expansion_applied": config.synonym_expansion,
            "vocabulary_size": len(corrector.vocabulary) if corrector else 0,
        }
