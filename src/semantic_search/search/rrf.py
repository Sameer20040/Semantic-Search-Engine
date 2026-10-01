"""Reciprocal Rank Fusion.

The formula
-----------
For a document ``d`` and a set of retrievers ``R``::

    RRF(d) = Σ_{i ∈ R}  1 / (k + rank_i(d))

``rank_i(d)`` is the document's **1-based** position in retriever ``i``, and a
retriever that did not return the document contributes nothing -- it is not a
zero, it is an absence, and the two are not the same. A document found by one
retriever at rank 1 and another at rank 1 scores twice as much as either alone;
that is the entire mechanism, and it is the reason fusion rewards *agreement*
rather than *magnitude*.

Why ranks and not scores
------------------------
BM25 scores are unbounded and are not comparable across queries. Cosine
similarities are bounded, but they are not comparable across retrievers either
-- 0.62 cosine is a strong match in a dense-only ranking and a mediocre one in
a hybrid list, because the retriever that put it there saw a different
neighbourhood. A weighted sum of the two would be a number with no
interpretation, and its meaning would shift every time the corpus or the
models changed.

RRF sidesteps this by consuming only the ordinal information each retriever is
confident about. The cost is that it discards score *margins*: a retriever
that is certain and one that is barely sure look identical at rank 1. That is a
real, deliberate trade -- standard RRF has a tunable answer for it (a larger
``k`` flattens the curve further, weighting consensus over individual
placements) and the parameter is exposed rather than hard-coded.

1-based, and why the caller is responsible for it
-------------------------------------------------
``SearchResult.rank`` and ``BM25Result.rank`` are both **0-based** in this
project, because 0-based ranks are what Python enumerates and what the
explainability panel displays. RRF's constant ``k = 60`` is defined against
1-based ranks: the original paper, and every implementation that reports the
score, uses ``1/(k + 1)`` for the top hit.

Rather than let each caller remember that, :func:`fuse` takes **0-based** ranks
-- the convention the rest of this codebase uses -- and does the ``+1`` in one
place. ``reciprocal_rank(0, k) == 1/(k+1)``, which is the value you would get by
writing ``1/(k+1)`` in a test by hand. The conversion is the single most likely
place for a silent off-by-one in a fusion layer, so it is made in one function
and tested against hand arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from semantic_search.core.config import HybridConfig

#: The paper's constant. Ranks below this point contribute very little, which
#: is the "do not over-trust a retriever's #1" behaviour.
DEFAULT_RRF_K = 60


def reciprocal_rank(rank: int, k: int = DEFAULT_RRF_K) -> float:
    """The fusion contribution of a **0-based** rank.

    Args:
        rank: 0-based position of the document in one retriever's output.
        k: The RRF constant. Must be non-negative.

    Returns:
        ``1 / (k + rank + 1)``. For ``rank=0, k=60`` this is ``1/61``.

    Raises:
        ValueError: ``rank`` is negative, or ``k`` is negative.
    """
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}.")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    return 1.0 / (k + rank + 1)


@dataclass(slots=True, frozen=True)
class RetrievalRun:
    """One retriever's contribution to a single query.

    The retriever is identified by name alone (``"bm25"``, ``"semantic"``), and
    carries ranked ``doc_id`` strings in best-first order. Fusion needs nothing
    else from a retriever, which is what keeps this layer independent of
    :class:`~semantic_search.index.bm25.BM25Result` and
    :class:`~semantic_search.search.models.SearchResult` -- and what makes the
    RRF unit tests runnable without a corpus, a model, or FAISS.

    ``scores`` is optional and used only for provenance. It is deliberately not
    an input to the arithmetic: see the module docstring.
    """

    name: str
    ranked_doc_ids: Sequence[str]
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A RetrievalRun must have a name.")
        # Duplicate detection is deliberately *not* here. With a score dict
        # keyed by id, a repeated id shows up here first as a length mismatch,
        # and the message "2 ranked ids but 1 scores" is a puzzle rather than a
        # diagnosis. :func:`fuse` detects it instead, where it can say which
        # retriever emitted the duplicate and why it matters.
        if len(self.scores) > len(self.ranked_doc_ids):
            raise ValueError(
                f"RetrievalRun {self.name!r} has {len(self.ranked_doc_ids)} ranked ids but "
                f"{len(self.scores)} scores; every score must belong to a ranked id."
            )


@dataclass(slots=True, frozen=True)
class FusedCandidate:
    """One document's fused standing, with every term still visible.

    Attributes:
        doc_id: The document.
        rrf_score: Sum of the contributions from the retrievers that found it.
        best_rank: The best 0-based rank it achieved in any retriever. Ties are
            broken on this, see :func:`fuse`.
        contributions: Per-retriever ``{name: 1/(k+rank+1)}``, only for the
            retrievers that actually returned this document.
        ranks: Per-retriever 0-based rank, same keys as ``contributions``.
        retrievers: Names of the retrievers that found it, sorted.
    """

    doc_id: str
    rrf_score: float
    best_rank: int
    contributions: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def retrievers(self) -> list[str]:
        return sorted(self.contributions)

    @property
    def agreement(self) -> int:
        """How many retrievers returned this document. The reason fusion works."""
        return len(self.contributions)


def fuse(runs: Sequence[RetrievalRun], *, k: int = DEFAULT_RRF_K) -> list[FusedCandidate]:
    """Fuse ranked runs into one list, best first.

    Ordering is by ``rrf_score`` descending. Ties -- which are common, because
    documents with the same rank *pattern* across retrievers score identically
    by construction -- are broken deterministically by:

    1. the best single rank achieved (``best_rank`` ascending), then
    2. the retriever-count (agreement) descending, then
    3. ``doc_id`` ascending.

    The third key is not cosmetic. Two documents found at exactly the same ranks
    by the same retrievers are genuinely indistinguishable to RRF, and without a
    final key the sort would fall back to input order -- which depends on dict
    iteration and would let an unrelated change reorder results. ``doc_id`` is
    stable across runs, processes, and Python versions, so the same inputs
    always produce the same output.

    Args:
        runs: Ranked output from one or more retrievers, best-first.
        k: The RRF constant.

    Returns:
        Candidates sorted best-first, ranked by ``rrf_score``.

    Raises:
        ValueError: ``k`` is negative, or a run contains a duplicate ``doc_id``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")

    total: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}

    for run in runs:
        seen: set[str] = set()
        for rank, doc_id in enumerate(run.ranked_doc_ids):
            if doc_id in seen:
                # Silently keeping the first would let a buggy retriever
                # inflate a document's fused score by listing it twice, and
                # keeping the last would make the outcome depend on iteration
                # detail. Refusing is the only option a caller can act on.
                raise ValueError(
                    f"Retriever {run.name!r} returned document {doc_id!r} twice. "
                    f"Duplicate ids within one retriever would double-count its "
                    f"fusion contribution."
                )
            seen.add(doc_id)

            contribution = reciprocal_rank(rank, k)
            total[doc_id] = total.get(doc_id, 0.0) + contribution
            contributions.setdefault(doc_id, {})[run.name] = contribution
            ranks.setdefault(doc_id, {})[run.name] = rank

    candidates = [
        FusedCandidate(
            doc_id=doc_id,
            rrf_score=score,
            best_rank=min(ranks[doc_id].values()),
            contributions=contributions[doc_id],
            ranks=ranks[doc_id],
        )
        for doc_id, score in total.items()
    ]
    candidates.sort(key=lambda c: (-c.rrf_score, c.best_rank, -c.agreement, c.doc_id))
    return candidates


def fuse_top_k(
    runs: Sequence[RetrievalRun],
    top_k: int,
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    """:func:`fuse` truncated to ``top_k``.

    The truncation happens *after* the full sort, so a document at position
    ``top_k + 1`` cannot displace one at ``top_k``; there is no cheap pre-filter
    that would be equivalent, and the sort over a few hundred candidates is not
    the bottleneck. Measured at 1.06 ms mean for two retrievers at depth 100
    (Phase 7) -- see the measurements section of ``BUILD_STATUS.md``.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}.")
    return fuse(runs, k=k)[:top_k]


def fuse_to_runs(
    runs: Iterable[RetrievalRun],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    """Convenience wrapper accepting any iterable of runs."""
    return fuse(list(runs), k=k)


def rrf_settings(config: HybridConfig | None = None) -> dict[str, Any]:
    """Report the active fusion parameters, for the ``/stats`` endpoint.

    The RRF constant is part of the observable behaviour of a search response,
    so it belongs in the same place as the index type and the model name.
    """
    config = config or HybridConfig()
    return {
        "method": "reciprocal_rank_fusion",
        "rrf_k": config.rrf_k,
        "candidate_pool": config.candidate_pool,
        "formula": "RRF(d) = sum over retrievers of 1 / (k + rank + 1), rank 1-based",
    }
