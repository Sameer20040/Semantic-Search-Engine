"""Query processing, layered above hybrid retrieval.

The pipeline is::

    User Query
      -> QueryProcessor      normalize, spell-correct, expand (recorded)
      -> HybridSearchService BM25 + semantic, receiving ONE string
      -> RRF
      -> results

Nothing in this package retrieves, ranks, embeds, or fuses. The dependency
arrow points one way -- this package imports from ``data`` and ``core``, and
never from ``search`` or ``index`` -- which is what keeps the processing layer
genuinely above the retrieval layer rather than tangled into it.

Public surface::

    from semantic_search.query import SearchService, QueryProcessor
    response = SearchService.from_settings(settings, store, hybrid).search("machne learning")
    response.query.corrections   # what changed, or ()
"""

from semantic_search.query.models import Correction, Expansion, ProcessedQuery, Vocabulary
from semantic_search.query.normalizer import normalize
from semantic_search.query.processor import QueryProcessor
from semantic_search.query.service import SearchResponse, SearchService
from semantic_search.query.spelling import SpellCorrector, build_vocabulary
from semantic_search.query.synonyms import SYNONYMS, apply_expansions, find_expansions

__all__ = [
    "SYNONYMS",
    "Correction",
    "Expansion",
    "ProcessedQuery",
    "QueryProcessor",
    "SearchResponse",
    "SearchService",
    "SpellCorrector",
    "Vocabulary",
    "apply_expansions",
    "build_vocabulary",
    "find_expansions",
    "normalize",
]
