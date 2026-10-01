"""Metric correctness, against hand arithmetic and against a second implementation.

Two independent kinds of check, because either alone is insufficient:

1. **Hand-computed values.** Small rankings with the arithmetic written out in
   the test, so a wrong formula fails with a number a reader can check by eye.
2. **An independent implementation** (:func:`_naive_precision`, and friends
   below) written with deliberately different index conventions -- it builds
   explicit rank tables and sorts, where the real code slices and enumerates.
   Asserted equal over randomised cases. A hand-computed expectation only proves
   the code matches the author's mental model; a second implementation catches a
   convention error the author made in *both* places.
"""

from __future__ import annotations

import math
import random

import pytest

from semantic_search.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    hits_to_ranked_ids,
    mean,
    mean_average_precision,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_count,
)

# The example the spec names: retrieved [A,B,C,D], relevant {A,C}.
A, B, C, D, E, F, G, H = "A", "B", "C", "D", "E", "F", "G", "H"


class _Hit:
    """Minimal stand-in for a retriever result, to exercise hits_to_ranked_ids."""

    def __init__(self, doc_id: str, rank: int = 0) -> None:
        self.doc_id = doc_id
        self.rank = rank


# --------------------------------------------------------------------------- #
# hits_to_ranked_ids
# --------------------------------------------------------------------------- #
class TestHitsToRankedIds:
    def test_preserves_given_order(self) -> None:
        assert hits_to_ranked_ids([_Hit(C, 0), _Hit(A, 1), _Hit(B, 2)]) == [C, A, B]

    def test_does_not_resort_by_rank_attribute(self) -> None:
        # Deliberately inverted ranks. The caller asserts the order; re-sorting
        # would silently substitute the evaluator's opinion for the retriever's.
        assert hits_to_ranked_ids([_Hit("x", 5), _Hit("y", 0)]) == ["x", "y"]

    def test_deduplicates_keeping_first(self) -> None:
        assert hits_to_ranked_ids([_Hit(A, 0), _Hit(B, 1), _Hit(A, 2)]) == [A, B]

    def test_accepts_plain_strings(self) -> None:
        assert hits_to_ranked_ids([A, B, A]) == [A, B]

    def test_empty_input(self) -> None:
        assert hits_to_ranked_ids([]) == []

    def test_skips_objects_without_doc_id(self) -> None:
        assert hits_to_ranked_ids([_Hit(A, 0), object(), _Hit(B, 1)]) == [A, B]

    def test_returns_list_of_str(self) -> None:
        out = hits_to_ranked_ids([_Hit(A, 0)])
        assert isinstance(out, list) and all(isinstance(x, str) for x in out)


# --------------------------------------------------------------------------- #
# precision_at_k
# --------------------------------------------------------------------------- #
class TestPrecision:
    def test_two_relevant_in_top_four(self) -> None:
        # [A,B,C,D] with {A,C}: window k=4 has 2 hits -> 2/4.
        assert precision_at_k([A, B, C, D], {A: 1, C: 1}, 4) == 0.5

    def test_relevant_at_rank_one(self) -> None:
        assert precision_at_k([A, B, C, D], {A: 1, C: 1}, 1) == 1.0

    def test_denominator_is_k_not_returned_length(self) -> None:
        # One hit, one result returned, k=4 -> 1/4. A floating denominator
        # would report a perfect 1.0 and reward refusing to answer.
        assert precision_at_k([A], {A: 1}, 4) == pytest.approx(0.25)

    def test_relevant_outside_cutoff(self) -> None:
        assert precision_at_k([A, B, C, D], {D: 1}, 2) == 0.0

    def test_empty_ranking_is_zero_not_error(self) -> None:
        assert precision_at_k([], {A: 1}, 10) == 0.0

    def test_no_relevant_documents(self) -> None:
        assert precision_at_k([A, B], {}, 2) == 0.0

    def test_all_returned_relevant(self) -> None:
        assert precision_at_k([A, B], {A: 1, B: 1}, 2) == 1.0

    def test_k_larger_than_corpus(self) -> None:
        # k=10, corpus of 3, 2 relevant found: 2/10, not 2/3.
        assert precision_at_k([A, B, C], {A: 1, B: 1}, 10) == pytest.approx(0.2)

    def test_k_zero(self) -> None:
        assert precision_at_k([A], {A: 1}, 0) == 0.0

    def test_negative_k(self) -> None:
        assert precision_at_k([A], {A: 1}, -3) == 0.0

    def test_unjudged_document_counts_as_irrelevant(self) -> None:
        assert precision_at_k([A, B], {A: 1}, 2) == 0.5

    def test_grade_zero_is_not_relevant(self) -> None:
        assert precision_at_k([A, B], {A: 0, B: 0}, 2) == 0.0

    def test_duplicates_do_not_double_count(self) -> None:
        # A repeated must not occupy two of the k slots.
        assert precision_at_k([A, A, A, A], {A: 1}, 4) == 0.25

    def test_k_one_worst_case(self) -> None:
        assert precision_at_k([B, A, C, D], {A: 1}, 1) == 0.0

    def test_k_one_best_case(self) -> None:
        assert precision_at_k([A, B, C, D], {A: 1}, 1) == 1.0

    def test_many_relevant(self) -> None:
        ids = [A, B, C, D, E]
        j = dict.fromkeys(ids, 1)
        assert precision_at_k(ids, j, 5) == 1.0

    def test_perfect_system_scores_exactly_one(self) -> None:
        assert precision_at_k([A, B, C], {A: 1, B: 1, C: 1}, 3) == 1.0

    def test_reversed_system_places_gold_last(self) -> None:
        # A is at position 3 of 3, so 1/3 -- not 0. Precision@k has no reward
        # for burying a hit, which is what recall and MRR are for.
        assert precision_at_k([C, B, A], {A: 1}, 3) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
class TestRecall:
    def test_finds_one_of_two(self) -> None:
        # C is at position 3, outside a cutoff of 2: one of two found.
        assert recall_at_k([A, B, C, D], {A: 1, C: 1}, 2) == 0.5

    def test_finds_all(self) -> None:
        assert recall_at_k([A, B, C, D], {A: 1, C: 1}, 4) == 1.0
        # A at 1 and C at 3: both inside k=3.
        assert recall_at_k([A, B, C, D], {A: 1, C: 1}, 3) == 1.0

    def test_finds_none(self) -> None:
        assert recall_at_k([B, D], {A: 1, C: 1}, 2) == 0.0

    def test_zero_relevant_is_zero_not_perfect(self) -> None:
        # The important one: an empty judgment must not read as recall 1.0.
        assert recall_at_k([A, B], {}, 2) == 0.0

    def test_only_grade_zero_is_zero_relevant(self) -> None:
        assert recall_at_k([A, B], {A: 0}, 2) == 0.0

    def test_short_result_list(self) -> None:
        assert recall_at_k([A], {A: 1, C: 1}, 10) == 0.5

    def test_k_larger_than_corpus(self) -> None:
        assert recall_at_k([A, B, C], {A: 1, B: 1}, 10) == 1.0

    def test_relevant_beyond_cutoff(self) -> None:
        assert recall_at_k([A, B, C, D], {D: 1}, 2) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k([A], {A: 1}, 0) == 0.0

    def test_empty_ranking(self) -> None:
        assert recall_at_k([], {A: 1}, 10) == 0.0

    def test_extra_relevant_beyond_corpus_is_still_counted_in_denominator(self) -> None:
        # Three judged relevant, two of them in the returned three: 2/3. A
        # denominator built from len(ranked_ids) would also give 2/3 here, so
        # the test that isolates the rule is the one below.
        assert recall_at_k([A, B, C], {A: 1, C: 1, D: 1}, 3) == pytest.approx(2 / 3)

    def test_denominator_is_judgments_not_returned_length(self) -> None:
        # One of three relevant returned. A float denominator would say 1/1.
        assert recall_at_k([A, B], {A: 1, C: 1, D: 1}, 2) == pytest.approx(1 / 3)

    def test_duplicates_do_not_inflate(self) -> None:
        assert recall_at_k([A, A, A], {A: 1, C: 1}, 3) == 0.5

    def test_perfect_system(self) -> None:
        # Both relevant documents inside the first two slots.
        assert recall_at_k([A, C, B, D], {A: 1, C: 1}, 2) == 1.0


# --------------------------------------------------------------------------- #
# reciprocal_rank
# --------------------------------------------------------------------------- #
class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank([A, B, C, D], {A: 1, C: 1}) == 1.0

    def test_second_position(self) -> None:
        assert reciprocal_rank([B, A, C], {A: 1}) == 0.5

    def test_third_position(self) -> None:
        assert reciprocal_rank([B, D, A], {A: 1}) == pytest.approx(1 / 3)

    def test_no_relevant_in_list(self) -> None:
        assert reciprocal_rank([B, D], {A: 1}) == 0.0

    def test_empty_ranking(self) -> None:
        assert reciprocal_rank([], {A: 1}) == 0.0

    def test_no_relevant_documents(self) -> None:
        assert reciprocal_rank([A, B], {}) == 0.0

    def test_uses_first_relevant_not_best_grade(self) -> None:
        # C has a higher grade but A comes first; RR reports the first hit.
        assert reciprocal_rank([A, C], {A: 1, C: 2}) == 1.0

    def test_grade_zero_is_not_a_hit(self) -> None:
        assert reciprocal_rank([A, B], {A: 0, B: 1}) == 0.5

    def test_last_of_ten(self) -> None:
        ids = [B, D, E, F, G, H, "x1", "y1", "z1", A]
        assert reciprocal_rank(ids, {A: 1}) == pytest.approx(0.1)

    def test_duplicates_do_not_advance_rank(self) -> None:
        assert reciprocal_rank([A, A, B], {B: 1}) == pytest.approx(1 / 2)

    def test_reversed_system_is_zero(self) -> None:
        assert reciprocal_rank([C, B, A], {A: 1}) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# average_precision
# --------------------------------------------------------------------------- #
class TestAveragePrecision:
    def test_single_relevant_at_rank_one(self) -> None:
        assert average_precision([A, B, C, D], {A: 1}) == 1.0

    def test_single_relevant_at_rank_two(self) -> None:
        assert average_precision([B, A, C, D], {A: 1}) == 0.5

    def test_two_relevant_first_two(self) -> None:
        # hits at 1 and 2: precisions 1/1 and 2/2, mean 1.0.
        assert average_precision([A, C, B, D], {A: 1, C: 1}) == 1.0

    def test_two_relevant_at_ranks_one_and_three(self) -> None:
        # precisions 1/1 and 2/3 -> mean 5/6.
        assert average_precision([A, B, C, D], {A: 1, C: 1}) == pytest.approx(5 / 6)

    def test_two_relevant_at_ranks_two_and_four(self) -> None:
        # Hits at 2 and 4: precisions 1/2 and 2/4 -> sum 1.0, mean 0.5.
        assert average_precision([B, A, D, C], {A: 1, C: 1}) == pytest.approx(0.5)

    def test_partial_credit_when_one_relevant_missing(self) -> None:
        # A at rank 1, C never found: 1/1, divided by 2 relevant -> 0.5.
        assert average_precision([A, B, D, E], {A: 1, C: 1}) == 0.5

    def test_no_relevant_found(self) -> None:
        assert average_precision([B, D], {A: 1, C: 1}) == 0.0

    def test_zero_relevant_documents(self) -> None:
        assert average_precision([A, B], {}) == 0.0

    def test_empty_ranking(self) -> None:
        assert average_precision([], {A: 1}) == 0.0

    def test_reduces_to_reciprocal_rank_for_single_relevant(self) -> None:
        # The degeneracy that motivates reporting AP only on multi-judged strata.
        ids = [B, D, A, C]
        assert average_precision(ids, {A: 1}) == reciprocal_rank(ids, {A: 1})

    def test_grade_zero_not_counted(self) -> None:
        assert average_precision([A, B], {A: 1, B: 0}) == 1.0

    def test_duplicates_do_not_double_count_hits(self) -> None:
        # A twice: the second must not produce a second hit at precision 2/2.
        assert average_precision([A, A, C], {A: 1, C: 1}) == 1.0

    def test_reversed_system_with_two_relevant(self) -> None:
        # Hits at 2 and 4: precisions 1/2 and 2/4 -> mean 0.5.
        assert average_precision([D, A, B, C], {A: 1, C: 1}) == pytest.approx(0.5)

    def test_all_relevant_at_top(self) -> None:
        assert average_precision([A, C, B, D], {A: 1, C: 1}) == 1.0


# --------------------------------------------------------------------------- #
# DCG / NDCG
# --------------------------------------------------------------------------- #
class TestDcg:
    def test_binary_single_hit_at_rank_one(self) -> None:
        # gain 2^1-1 = 1, discount log2(2) = 1.
        assert dcg_at_k([A, B, C], {A: 1}, 3) == pytest.approx(1.0)

    def test_binary_hit_at_rank_two(self) -> None:
        assert dcg_at_k([B, A], {A: 1}, 2) == pytest.approx(1 / math.log2(3))

    def test_graded_gain_dominates_two_binary(self) -> None:
        # One grade-2 document must be worth more than two grade-1 ones.
        graded = dcg_at_k([A], {A: 2}, 1)
        binary = dcg_at_k([A, B], {A: 1, B: 1}, 2)
        assert graded > binary

    def test_grade_three_gain(self) -> None:
        assert dcg_at_k([A], {A: 3}, 1) == pytest.approx(7.0)

    def test_no_hits_is_zero(self) -> None:
        assert dcg_at_k([B, D], {A: 1}, 2) == 0.0

    def test_empty(self) -> None:
        assert dcg_at_k([], {A: 1}, 5) == 0.0

    def test_k_zero(self) -> None:
        assert dcg_at_k([A], {A: 1}, 0) == 0.0

    def test_cutoff_excludes_later_hits(self) -> None:
        assert dcg_at_k([B, A], {A: 1}, 1) == 0.0

    def test_unjudged_contributes_nothing(self) -> None:
        assert dcg_at_k(["x1", "y1", "z1"], {}, 3) == 0.0


class TestNdcg:
    def test_perfect_ranking_is_exactly_one(self) -> None:
        assert ndcg_at_k([A, C, B, D], {A: 1, C: 1}, 4) == pytest.approx(1.0)

    def test_single_relevant_at_rank_one(self) -> None:
        assert ndcg_at_k([A, B, C], {A: 1}, 3) == pytest.approx(1.0)

    def test_single_relevant_at_rank_two(self) -> None:
        # ideal = 1/log2(2) = 1; actual = 1/log2(3).
        assert ndcg_at_k([B, A], {A: 1}, 2) == pytest.approx(1 / math.log2(3))

    def test_single_relevant_at_rank_three(self) -> None:
        assert ndcg_at_k([B, D, A], {A: 1}, 3) == pytest.approx(1 / math.log2(4))

    def test_reversed_single_relevant_is_not_zero(self) -> None:
        # Worth stating: NDCG with one relevant document never reaches 0, so a
        # system that buried the gold doc at rank 10 still scores 0.289.
        assert ndcg_at_k([B, D, A], {A: 1}, 3) == pytest.approx(0.5)

    def test_no_relevant_documents_is_zero(self) -> None:
        assert ndcg_at_k([A, B], {}, 2) == 0.0

    def test_only_grade_zero_is_zero(self) -> None:
        assert ndcg_at_k([A, B], {A: 0}, 2) == 0.0

    def test_empty_ranking(self) -> None:
        assert ndcg_at_k([], {A: 1}, 5) == 0.0

    def test_k_zero(self) -> None:
        assert ndcg_at_k([A], {A: 1}, 0) == 0.0

    def test_ideal_uses_min_k_and_n_relevant(self) -> None:
        # 2 relevant, k=10: ideal has 2 terms, not 10.
        assert ndcg_at_k([A, C], {A: 1, C: 1}, 10) == pytest.approx(1.0)

    def test_graded_perfect_ranking(self) -> None:
        assert ndcg_at_k([A, B, C], {A: 3, B: 2, C: 1}, 3) == pytest.approx(1.0)

    def test_graded_wrong_order_below_one(self) -> None:
        assert ndcg_at_k([C, B, A], {A: 3, B: 2, C: 1}, 3) < 1.0

    def test_graded_ordering_matters(self) -> None:
        good = ndcg_at_k([A, C, B], {A: 3, C: 2, B: 1}, 3)
        bad = ndcg_at_k([B, C, A], {A: 3, C: 2, B: 1}, 3)
        assert good > bad

    def test_cutoff_can_make_perfect_ranking_less_than_one(self) -> None:
        # k=1: only the first item counts, ideal is a single gain of 3 -> 3.
        assert ndcg_at_k([A, B, C], {A: 3, B: 2, C: 1}, 1) == pytest.approx(1.0)

    def test_bounded_by_one(self) -> None:
        rng = random.Random(11)
        for _ in range(200):
            ranked = rng.sample([A, B, C, D, E, F], 4)
            judgments = {x: rng.choice([0, 1, 2, 3]) for x in [A, B, C, D]}
            value = ndcg_at_k(ranked, judgments, 5)
            assert 0.0 <= value <= 1.0 + 1e-9

    def test_never_exceeds_one_when_all_returned_are_relevant(self) -> None:
        assert ndcg_at_k([A, C], {A: 1, C: 1}, 5) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_mean_of_values(self) -> None:
        assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)

    def test_mean_of_empty_is_zero(self) -> None:
        assert mean([]) == 0.0

    def test_mean_of_single(self) -> None:
        assert mean([0.25]) == 0.25

    def test_mean_precision(self) -> None:
        data = [([A, B], {A: 1}), ([B, C], {C: 1})]
        # 0.5 and 0.5 -> 0.5
        assert mean_precision_at_k(data, 2) == pytest.approx(0.5)

    def test_mean_precision_mixed(self) -> None:
        data = [([A], {A: 1}), ([B], {B: 1}), ([B], {A: 1})]
        # P@1: 1.0, 1.0, 0.0 -> 2/3
        assert mean_precision_at_k(data, 1) == pytest.approx(2 / 3)

    def test_mean_recall(self) -> None:
        data = [([A, B], {A: 1, C: 1}), ([A, C], {A: 1, C: 1})]
        # 0.5, 1.0 -> 0.75
        assert mean_recall_at_k(data, 2) == pytest.approx(0.75)

    def test_mean_reciprocal_rank(self) -> None:
        data = [([A], {A: 1}), ([B, A], {A: 1}), ([B], {A: 1})]
        # 1.0, 0.5, 0.0 -> 0.5
        assert mean_reciprocal_rank(data) == pytest.approx(0.5)

    def test_mean_average_precision(self) -> None:
        data = [([A, B], {A: 1, C: 1}), ([A, C], {A: 1, C: 1})]
        # 0.5, 1.0 -> 0.75
        assert mean_average_precision(data) == pytest.approx(0.75)

    def test_mean_ndcg(self) -> None:
        data = [([A, C], {A: 1, C: 1}), ([C, A], {A: 1, C: 1})]
        assert mean_ndcg_at_k(data, 2) == pytest.approx(1.0)

    def test_all_aggregators_on_empty_input(self) -> None:
        assert mean_precision_at_k([], 5) == 0.0
        assert mean_recall_at_k([], 5) == 0.0
        assert mean_reciprocal_rank([]) == 0.0
        assert mean_average_precision([]) == 0.0
        assert mean_ndcg_at_k([], 5) == 0.0

    def test_perfect_system_averages_exactly_one(self) -> None:
        data = [([A, C, B], {A: 1, C: 1}) for _ in range(5)]
        # 2 of 3 slots are relevant, so P@3 is 2/3 while recall@3 is 1.0 --
        # the two metrics disagreeing here is the point of reporting both.
        assert mean_precision_at_k(data, 3) == pytest.approx(2 / 3)
        assert mean_recall_at_k(data, 3) == 1.0
        assert mean_reciprocal_rank(data) == 1.0
        assert mean_ndcg_at_k(data, 3) == pytest.approx(1.0)

    def test_failing_system_averages_zero(self) -> None:
        data = [([B, D, E], {A: 1}) for _ in range(3)]
        assert mean_precision_at_k(data, 3) == 0.0
        assert mean_reciprocal_rank(data) == 0.0
        assert mean_average_precision(data) == 0.0


# --------------------------------------------------------------------------- #
# Independent implementations, for cross-checking
# --------------------------------------------------------------------------- #
def _naive_rank_table(ranked: list[str], judgments: dict[str, int]) -> dict[str, int]:
    """doc_id -> 1-based position, built by explicit counting rather than slicing."""
    table: dict[str, int] = {}
    position = 0
    for doc_id in ranked:
        if doc_id in table:
            continue
        position += 1
        table[doc_id] = position
    assert len(table) <= len(ranked)
    del judgments
    return table


def _naive_precision(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    table = _naive_rank_table(ranked, judgments)
    inside = [d for d, r in table.items() if r <= k]
    hits = len([d for d in inside if judgments.get(d, 0) > 0])
    return hits / k


def _naive_recall(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    positives = sorted(d for d, g in judgments.items() if g > 0)
    if not positives or k <= 0:
        return 0.0
    table = _naive_rank_table(ranked, judgments)
    found = [d for d in positives if table.get(d, 10**9) <= k]
    return len(found) / len(positives)


def _naive_reciprocal_rank(ranked: list[str], judgments: dict[str, int]) -> float:
    table = _naive_rank_table(ranked, judgments)
    ranks = [table[d] for d in sorted(table) if judgments.get(d, 0) > 0]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def _naive_average_precision(ranked: list[str], judgments: dict[str, int]) -> float:
    positives = [d for d, g in judgments.items() if g > 0]
    if not positives:
        return 0.0
    table = _naive_rank_table(ranked, judgments)
    ordered = sorted(table, key=lambda d: table[d])
    total = 0.0
    hits = 0
    for doc_id in ordered:
        if judgments.get(doc_id, 0) > 0:
            hits += 1
            total += hits / table[doc_id]
    return total / len(positives)


def _naive_dcg(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    table = _naive_rank_table(ranked, judgments)
    ordered = sorted(table, key=lambda d: table[d])
    total = 0.0
    for doc_id in ordered:
        if table[doc_id] > k:
            break
        grade = judgments.get(doc_id, 0)
        if grade > 0:
            total += (2**grade - 1) / math.log2(table[doc_id] + 1)
    return total


def _naive_ndcg(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    positives = [g for g in judgments.values() if g > 0]
    if not positives or k <= 0:
        return 0.0
    positives.sort(reverse=True)
    ideal_grades = positives[:k]
    ideal = 0.0
    for i, grade in enumerate(ideal_grades, start=1):
        ideal += (2**grade - 1) / math.log2(i + 1)
    if ideal <= 0.0:
        return 0.0
    return _naive_dcg(ranked, judgments, k) / ideal


class TestCrossCheckAgainstIndependentImplementation:
    """Randomised agreement with a second implementation of the same definitions."""

    @staticmethod
    def _cases(n: int = 400) -> list[tuple[list[str], dict[str, int]]]:
        rng = random.Random(20250930)
        universe = [A, B, C, D, E, F, G, H]
        cases: list[tuple[list[str], dict[str, int]]] = []
        for _ in range(n):
            length = rng.randint(0, 8)
            ranked = [rng.choice(universe) for _ in range(length)]
            judged = rng.sample(universe, rng.randint(0, 5))
            grades = {d: rng.choice([0, 0, 1, 1, 2, 3]) for d in judged}
            cases.append((ranked, grades))
        return cases

    def test_precision_agrees(self) -> None:
        for ranked, judgments in self._cases():
            for k in (1, 3, 5, 10):
                assert precision_at_k(ranked, judgments, k) == pytest.approx(
                    _naive_precision(ranked, judgments, k)
                )

    def test_recall_agrees(self) -> None:
        for ranked, judgments in self._cases():
            for k in (1, 3, 5, 10):
                assert recall_at_k(ranked, judgments, k) == pytest.approx(
                    _naive_recall(ranked, judgments, k)
                )

    def test_reciprocal_rank_agrees(self) -> None:
        for ranked, judgments in self._cases():
            assert reciprocal_rank(ranked, judgments) == pytest.approx(
                _naive_reciprocal_rank(ranked, judgments)
            )

    def test_average_precision_agrees(self) -> None:
        for ranked, judgments in self._cases():
            assert average_precision(ranked, judgments) == pytest.approx(
                _naive_average_precision(ranked, judgments)
            )

    def test_dcg_agrees(self) -> None:
        for ranked, judgments in self._cases():
            for k in (1, 3, 5, 10):
                assert dcg_at_k(ranked, judgments, k) == pytest.approx(
                    _naive_dcg(ranked, judgments, k)
                )

    def test_ndcg_agrees(self) -> None:
        for ranked, judgments in self._cases():
            for k in (1, 3, 5, 10):
                assert ndcg_at_k(ranked, judgments, k) == pytest.approx(
                    _naive_ndcg(ranked, judgments, k), abs=1e-9
                )


class TestInvariants:
    """Properties that must hold whatever the input."""

    def test_all_metrics_bounded_zero_to_one(self) -> None:
        for ranked, judgments in TestCrossCheckAgainstIndependentImplementation._cases(200):
            assert 0.0 <= precision_at_k(ranked, judgments, 5) <= 1.0
            assert 0.0 <= recall_at_k(ranked, judgments, 5) <= 1.0
            assert 0.0 <= reciprocal_rank(ranked, judgments) <= 1.0
            assert 0.0 <= average_precision(ranked, judgments) <= 1.0
            assert 0.0 <= ndcg_at_k(ranked, judgments, 5) <= 1.0

    def test_perfect_ordering_maximises_every_metric(self) -> None:
        judgments = {A: 1, C: 1}
        assert precision_at_k([A, C], judgments, 2) == 1.0
        assert recall_at_k([A, C], judgments, 2) == 1.0
        assert reciprocal_rank([A, C], judgments) == 1.0
        assert average_precision([A, C], judgments) == 1.0
        assert ndcg_at_k([A, C], judgments, 2) == 1.0

    def test_swapping_two_results_cannot_improve_ndcg(self) -> None:
        # NDCG is not monotone in rank under arbitrary swaps, but putting the
        # higher-graded document later must not raise it.
        good = ndcg_at_k([A, B], {A: 3, B: 1}, 2)
        bad = ndcg_at_k([B, A], {A: 3, B: 1}, 2)
        assert good > bad

    def test_relevant_count_ignores_zero_grades(self) -> None:
        assert relevant_count({A: 1, B: 0, C: 2}) == 2

    def test_relevant_count_of_empty(self) -> None:
        assert relevant_count({}) == 0
