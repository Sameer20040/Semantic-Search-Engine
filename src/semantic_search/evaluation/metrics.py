"""Relevance metrics. Pure functions, and no imports from this package.

Every function here is a total function of its arguments: it needs no corpus,
no index, no model, no I/O. That is what makes the metrics testable against
hand-computed values at all, and it is the same constraint
:mod:`semantic_search.search.rrf` works under. ``tests/unit/test_metrics.py``
can therefore check a number against arithmetic done by hand on paper without
building anything first.

Ranks are 1-based, and this is stated once
------------------------------------------
Metric definitions here use the **1-based** position of a result, which is what
the IR literature means. The retrievers in this project return **0-based**
ranks -- ``SearchResult.rank``, ``BM25Result.rank`` and the fused
``HybridSearchResult.rank`` are all 0-based, and that convention is deliberate
and load-bearing. The conversion between the two lives in exactly one function,
:func:`hits_to_ranked_ids`, and nowhere else. This project has already shipped
two off-by-one rank bugs (Phase 7), so the conversion is centralised and tested
rather than left to each call site to remember.

Why these edge cases return 0.0 instead of raising
--------------------------------------------------
An empty ranking, or a ranking shorter than ``k``, is a *bad system*, not an
*error in the evaluator*. If ``precision_at_k`` raised on those inputs, a run
would abort partway through and silently lose exactly the queries that
discriminate most between systems -- the ones every retriever failed. The
metrics degrade to 0.0 so that a failure is scored as a failure and the run
completes. A missing judgment is likewise 0, not a crash.

Judgments and grades
--------------------
A judgment mapping is ``{doc_id: grade}``. Grade 0 (or an absent key) means
"not relevant". Binary relevance and graded relevance are the same code path:
set grades above 0 to get graded NDCG, set them all to 1 to get the binary
special case, which falls out of the general formula without a branch.

The project has no documented protocol for graded relevance, so nothing here
invents one. :func:`ndcg_at_k` and :func:`dcg_at_k` accept grades because the
formula is standard and useful, not because any judgment in this repository
uses a scale above binary. :func:`average_precision` is the binary form and
says so in its name.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

#: A judgment mapping: doc_id -> grade. A missing key is grade 0.
Judgments = Mapping[str, int]


def _dedupe(ranked_ids: Sequence[str]) -> list[str]:
    """Ordered de-duplication, keeping the first occurrence.

    Applied inside every metric rather than only by the caller. Duplicates are
    not hypothetical: a fused candidate list assembled from several retrievers,
    or a ranking re-assembled from several sources, can name a document twice,
    and a duplicate silently occupies two ranks. The consequences are not
    cosmetic -- a duplicated gold document raises NDCG above 1.0 and inflates
    average precision above 1.0, so a metric that can return a value outside
    [0, 1] is not reporting quality, it is reporting a bug in its input.

    Keeping the *first* occurrence is the conservative choice: a retriever that
    ranks a document twice has still placed it where it placed it.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in ranked_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def hits_to_ranked_ids(hits: Iterable[object]) -> list[str]:
    """Extract an ordered, de-duplicated list of document ids from results.

    The one place in this module where a 0-based rank becomes a 1-based
    position. Results are taken in *the order given*, not sorted by their
    ``rank`` attribute: the retrievers already emit them in rank order, and
    sorting here would re-derive something the caller asserted. Ordering by
    position is what the metrics consume.

    De-duplication keeps the **first** occurrence; the metrics do the same thing
    again on their own input, so a caller that skips this function still gets
    correct numbers.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        doc_id = getattr(hit, "doc_id", None)
        if doc_id is None and isinstance(hit, str):
            doc_id = hit
        if not isinstance(doc_id, str) or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def _grade(judgments: Judgments, doc_id: str) -> int:
    """Grade of ``doc_id``, treating an unjudged document as not relevant.

    The unjudged-is-zero choice is what makes the metrics *conservative*: a
    retriever is never rewarded for a document nobody looked at. It is also why
    the report has to state the fraction of each system's top-k that went
    unjudged -- a retriever that returns mostly unjudged documents is not being
    penalised for being wrong, it is being neither rewarded nor punished, and
    hiding that would read as a quality difference.
    """
    return judgments.get(doc_id, 0)


def relevant_count(judgments: Judgments) -> int:
    """How many documents are judged relevant (grade > 0)."""
    return sum(1 for grade in judgments.values() if grade > 0)


def precision_at_k(ranked_ids: Sequence[str], judgments: Judgments, k: int) -> float:
    """Fraction of the first ``k`` results that are judged relevant.

    Returns 0.0 for an empty ranking or ``k <= 0``. Note the denominator is
    ``k``, not ``len(ranked_ids)``: a system that returns 3 results for a
    ``k = 10`` question has not achieved precision 1.0, and letting the
    denominator float would reward short answers.
    """
    if k <= 0:
        return 0.0
    window = _dedupe(ranked_ids)[:k]
    if not window:
        return 0.0
    hits = sum(1 for doc_id in window if _grade(judgments, doc_id) > 0)
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], judgments: Judgments, k: int) -> float:
    """Fraction of the judged-relevant documents found in the first ``k``.

    Returns 0.0 when nothing is judged relevant. That case is not an error but
    it is also not a 1.0: with no relevant documents in the collection there is
    nothing to recall, and scoring it as perfect would let a query with an
    empty judgment inflate a system's mean. A run containing such a query should
    exclude it from the aggregate, and :mod:`.report` reports how many it
    excluded.
    """
    total = relevant_count(judgments)
    if total == 0 or k <= 0:
        return 0.0
    window = set(_dedupe(ranked_ids)[:k])
    found = sum(1 for doc_id, grade in judgments.items() if grade > 0 and doc_id in window)
    return found / total


def reciprocal_rank(ranked_ids: Sequence[str], judgments: Judgments) -> float:
    """``1 / rank`` of the first relevant result, or 0.0 if there is none.

    Takes no ``k``: reciprocal rank has no cutoff. It is the natural summary for
    a single-judge question with one relevant document, where precision and
    recall are near-redundant with it.
    """
    for position, doc_id in enumerate(_dedupe(ranked_ids), start=1):
        if _grade(judgments, doc_id) > 0:
            return 1.0 / position
    return 0.0


def average_precision(ranked_ids: Sequence[str], judgments: Judgments) -> float:
    """Mean of the precision values at each relevant position (binary AP).

    The denominator is the number of *judged-relevant* documents, not ``k`` and
    not ``len(ranked_ids)``. A query with three relevant documents where the
    system found one at rank 1 scores 1/3, not 1.0 -- that is the difference
    between AP and precision@1, and it is why a system cannot score 1.0 here
    without finding everything.

    With exactly one relevant document this reduces to ``1 / rank``, i.e. the
    reciprocal rank. That is a real degeneracy, not a bug, and it is why AP and
    NDCG are reported only on the stratum with multiple judgments.
    """
    total = relevant_count(judgments)
    if total == 0 or not ranked_ids:
        return 0.0
    running = 0.0
    found = 0
    for position, doc_id in enumerate(_dedupe(ranked_ids), start=1):
        if _grade(judgments, doc_id) > 0:
            found += 1
            running += found / position
    return running / total


def dcg_at_k(ranked_ids: Sequence[str], judgments: Judgments, k: int) -> float:
    """Discounted cumulative gain over the first ``k``, using gain ``2^g - 1``.

    Positions are 1-based, so the first result is discounted by
    ``log2(1 + 1) = 1`` and the tenth by ``log2(11)``. The ``2^g - 1`` gain is
    the standard form: it makes a grade-2 document worth more than two grade-1
    documents, which is the point of having grades. With every grade equal to
    1 -- the binary case this project actually uses -- it reduces to a plain
    count discounted by position.
    """
    if k <= 0:
        return 0.0
    total = 0.0
    for position, doc_id in enumerate(_dedupe(ranked_ids)[:k], start=1):
        grade = _grade(judgments, doc_id)
        if grade > 0:
            total += (2**grade - 1) / math.log2(position + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], judgments: Judgments, k: int) -> float:
    """DCG normalised by the ideal DCG over the same ``k``.

    The ideal is built by sorting the *judged* grades descending, not by sorting
    the results -- normalising by the system's own output would make any
    ranking score 1.0. The ideal runs over ``min(k, n_relevant)`` items, so a
    collection with 2 relevant documents and ``k = 10`` has an ideal of 2 terms.
    A system that finds both at the top scores exactly 1.0.

    Returns 0.0 when nothing is judged relevant (no ideal exists) or when the
    ranking is empty.
    """
    if k <= 0:
        return 0.0
    ideal_grades = sorted((g for g in judgments.values() if g > 0), reverse=True)[:k]
    if not ideal_grades:
        return 0.0
    ideal = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))
    if ideal <= 0.0:
        return 0.0
    return float(dcg_at_k(ranked_ids, judgments, k) / ideal)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, 0.0 for an empty input.

    An empty input is a legitimate state (a stratum with no queries after
    filtering) and the alternative -- dividing by zero -- turns a reporting gap
    into a crash that loses the run.
    """
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def mean_precision_at_k(per_query: Sequence[tuple[Sequence[str], Judgments]], k: int) -> float:
    """Mean ``precision_at_k`` over ``(ranking, judgments)`` pairs."""
    return mean(precision_at_k(ids, j, k) for ids, j in per_query)


def mean_recall_at_k(per_query: Sequence[tuple[Sequence[str], Judgments]], k: int) -> float:
    """Mean ``recall_at_k`` over ``(ranking, judgments)`` pairs."""
    return mean(recall_at_k(ids, j, k) for ids, j in per_query)


def mean_reciprocal_rank(per_query: Sequence[tuple[Sequence[str], Judgments]]) -> float:
    """Mean reciprocal rank over ``(ranking, judgments)`` pairs."""
    return mean(reciprocal_rank(ids, j) for ids, j in per_query)


def mean_average_precision(per_query: Sequence[tuple[Sequence[str], Judgments]]) -> float:
    """Mean average precision over ``(ranking, judgments)`` pairs."""
    return mean(average_precision(ids, j) for ids, j in per_query)


def mean_ndcg_at_k(per_query: Sequence[tuple[Sequence[str], Judgments]], k: int) -> float:
    """Mean ``ndcg_at_k`` over ``(ranking, judgments)`` pairs."""
    return mean(ndcg_at_k(ids, j, k) for ids, j in per_query)
