"""Search orchestration: the layer that turns a query into ranked results."""

from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.hybrid_models import HybridSearchResult, RetrieverHit
from semantic_search.search.models import SearchResult
from semantic_search.search.rrf import (
    FusedCandidate,
    RetrievalRun,
    fuse,
    reciprocal_rank,
)
from semantic_search.search.semantic import SemanticSearchService

__all__ = [
    "FusedCandidate",
    "HybridSearchResult",
    "HybridSearchService",
    "RetrievalRun",
    "RetrieverHit",
    "SearchResult",
    "SemanticSearchService",
    "fuse",
    "reciprocal_rank",
]
