"""The result type returned by hybrid search.

Why not reuse :class:`~semantic_search.search.models.SearchResult`
-----------------------------------------------------------------
``SearchResult`` is the right shape for a single retriever: one score, one
rank, and a ``scores`` dict that happens to have one key. A fused result is
genuinely different -- it has a fused score whose meaning is *not* comparable to
either input score, up to two 0-based input ranks, up to two raw input scores
that are **not** on a common scale, and up to two per-retriever RRF
contributions that *do* sum to the fused score. Flattening that into
``SearchResult`` would mean overloading ``score`` with a third meaning and
hiding the inputs in a dict, which is exactly the ambiguity the typed model
was introduced to remove.

So this is a separate type, not a second copy: it holds one ``SearchResult``
for the winning document (so title and snippet are built once, by the same
code, with the same query-aware excerpt) and adds the provenance on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_search.search.models import SearchResult
from semantic_search.search.rrf import FusedCandidate


@dataclass(slots=True, frozen=True)
class RetrieverHit:
    """What one retriever contributed for one document.

    Attributes:
        retriever: ``"bm25"`` or ``"semantic"``.
        rank: 0-based position in that retriever's output.
        raw_score: The retriever's own score, at its own scale. Unbounded and
            query-dependent for BM25; a cosine in ``[-1, 1]`` for the vector
            retriever. Present for display and debugging, and explicitly
            **not** comparable with another retriever's ``raw_score``.
        rrf_contribution: ``1 / (k + rank + 1)`` -- the part of the fused score
            this retriever is responsible for. Sums with the other
            contributions to give :attr:`HybridSearchResult.rrf_score`.
    """

    retriever: str
    rank: int
    raw_score: float
    rrf_contribution: float


@dataclass(slots=True, frozen=True)
class HybridSearchResult:
    """One fused result with full provenance.

    Attributes:
        result: The underlying :class:`SearchResult` for this document, carrying
            ``doc_id``, ``title``, ``snippet``, and the store ``row``. Its
            ``score`` is set to the fused RRF score so the field means the same
            thing as everywhere else: higher is better. Its ``rank`` is the
            0-based position in the fused list.
        rrf_score: The fused score, i.e. the sum of every
            :attr:`RetrieverHit.rrf_contribution`. Present at the top level as
            well as derivable, because it is the thing callers rank on.
        hits: One entry per retriever that returned this document, sorted by
            retriever name. Empty is impossible -- a candidate always came from
            somewhere.
    """

    result: SearchResult
    rrf_score: float
    hits: tuple[RetrieverHit, ...] = ()

    # ----------------------------------------------------------------- #
    # Pass-through accessors
    # ----------------------------------------------------------------- #
    @property
    def doc_id(self) -> str:
        return self.result.doc_id

    @property
    def title(self) -> str:
        return self.result.title

    @property
    def snippet(self) -> str:
        return self.result.snippet

    @property
    def rank(self) -> int:
        return self.result.rank

    @property
    def row(self) -> int:
        return self.result.row

    @property
    def score(self) -> float:
        """The fused score. See :attr:`rrf_score`."""
        return self.result.score

    # ----------------------------------------------------------------- #
    # Provenance
    # ----------------------------------------------------------------- #
    @property
    def retrievers(self) -> list[str]:
        return [hit.retriever for hit in self.hits]

    @property
    def agreement(self) -> int:
        """How many retrievers found this document."""
        return len(self.hits)

    def hit(self, retriever: str) -> RetrieverHit | None:
        """Provenance for one named retriever, or ``None`` if it missed."""
        for candidate in self.hits:
            if candidate.retriever == retriever:
                return candidate
        return None

    @property
    def rank_of(self) -> dict[str, int]:
        """0-based rank per retriever that found this document."""
        return {hit.retriever: hit.rank for hit in self.hits}

    @property
    def raw_scores(self) -> dict[str, float]:
        return {hit.retriever: hit.raw_score for hit in self.hits}

    @property
    def contributions(self) -> dict[str, float]:
        """Per-retriever RRF contributions. Sums to :attr:`rrf_score`."""
        return {hit.retriever: hit.rrf_contribution for hit in self.hits}

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API.

        Rounded consistently with :meth:`SearchResult.to_dict`, and the raw
        scores are nested under ``retrievers`` rather than flattened, so that a
        client cannot read them as a common scale.
        """
        return {
            "doc_id": self.result.doc_id,
            "title": self.result.title,
            "snippet": self.result.snippet,
            "rank": self.result.rank,
            "rrf_score": round(self.rrf_score, 8),
            "agreement": self.agreement,
            "retrievers": {
                hit.retriever: {
                    "rank": hit.rank,
                    "raw_score": round(hit.raw_score, 6),
                    "rrf_contribution": round(hit.rrf_contribution, 8),
                }
                for hit in self.hits
            },
        }


def build_hybrid_result(
    candidate: FusedCandidate,
    base: SearchResult,
    raw_scores: dict[str, float],
    *,
    rank: int,
) -> HybridSearchResult:
    """Attach fusion provenance to an already-built :class:`SearchResult`.

    ``base`` is reused rather than rebuilt: its snippet was produced with the
    query terms, and rebuilding it here would mean a second pass over the same
    document text for every query. Only ``score`` and ``rank`` are replaced,
    because those are the fields fusion actually determines.
    """
    hits = tuple(
        RetrieverHit(
            retriever=name,
            rank=candidate.ranks[name],
            raw_score=raw_scores.get(name, 0.0),
            rrf_contribution=contribution,
        )
        for name, contribution in sorted(candidate.contributions.items())
    )
    return HybridSearchResult(
        result=SearchResult(
            doc_id=base.doc_id,
            title=base.title,
            snippet=base.snippet,
            score=candidate.rrf_score,
            rank=rank,
            row=base.row,
            scores=base.scores,
        ),
        rrf_score=candidate.rrf_score,
        hits=hits,
    )
