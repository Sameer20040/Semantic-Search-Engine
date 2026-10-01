"""Unit tests for Reciprocal Rank Fusion.

These test the *arithmetic*, by hand, with no corpus, no model, and no FAISS.
That is the point of keeping :func:`~semantic_search.search.rrf.fuse` dependent
on nothing but ranked id strings: the fusion formula is the one piece of this
phase that can be verified exactly, so it is verified exactly rather than
approximately.

Every expected value below is written out as a fraction with k=60 so the
arithmetic is checkable by reading the test, not by trusting the
implementation.
"""

from __future__ import annotations

import pytest

from semantic_search.search.rrf import (
    DEFAULT_RRF_K,
    FusedCandidate,
    RetrievalRun,
    fuse,
    fuse_to_runs,
    fuse_top_k,
    reciprocal_rank,
    rrf_settings,
)


def run(name: str, doc_ids: list[str], scores: dict[str, float] | None = None) -> RetrievalRun:
    """A ranked run; scores default to something arbitrary but distinct.

    The scores are deliberately meaningless -- fusion must not read them, which
    is what :class:`TestRawScoresAreNeverUsed` checks. A dict is keyed by id, so
    a duplicated id shares one entry; :class:`TestCaseEDuplicateIds` builds the
    malformed runs itself.
    """
    ids = list(doc_ids)
    return RetrievalRun(
        name=name,
        ranked_doc_ids=ids,
        scores=scores or {doc_id: float(len(ids) - i) for i, doc_id in enumerate(ids)},
    )


def scored_run(name: str, pairs: list[tuple[str, float]]) -> RetrievalRun:
    """A run built from explicit ``(doc_id, score)`` pairs.

    Needed where the same id appears twice: :class:`RetrievalRun` wants a score
    dict, so a malformed run cannot be expressed through :func:`run` at all.
    """
    return RetrievalRun(
        name=name,
        ranked_doc_ids=[doc_id for doc_id, _ in pairs],
        scores=dict(pairs),
    )


def by_id(candidates: list[FusedCandidate]) -> dict[str, FusedCandidate]:
    return {c.doc_id: c for c in candidates}


class TestReciprocalRank:
    def test_top_rank_is_one_over_k_plus_one(self) -> None:
        # 1/(k+1) with a 1-based rank of 1. This is the value the paper defines.
        assert reciprocal_rank(0, k=60) == pytest.approx(1 / 61)

    def test_second_rank_is_one_over_k_plus_two(self) -> None:
        assert reciprocal_rank(1, k=60) == pytest.approx(1 / 62)

    def test_ranks_use_one_based_numbering(self) -> None:
        # The whole point of the conversion: a 0-based rank of 0 must produce
        # the contribution a 1-based rank of 1 would.
        assert reciprocal_rank(0, k=60) == pytest.approx(1 / (60 + 1))
        assert reciprocal_rank(0, k=60) != pytest.approx(1 / 60)

    def test_decreases_with_rank(self) -> None:
        values = [reciprocal_rank(r, k=60) for r in range(20)]
        assert values == sorted(values, reverse=True)

    def test_k_zero_gives_the_harmonic_weights(self) -> None:
        # 1/(0+1), 1/(0+2), ... -- the un-damped form, kept honest as a test.
        assert reciprocal_rank(0, k=0) == pytest.approx(1.0)
        assert reciprocal_rank(2, k=0) == pytest.approx(1 / 3)

    def test_larger_k_flattens_the_curve(self) -> None:
        # The documented trade-off: a bigger k weights consensus over
        # individual placements, because the gap between rank 1 and rank 10
        # shrinks.
        small = reciprocal_rank(0, k=1) - reciprocal_rank(9, k=1)
        large = reciprocal_rank(0, k=60) - reciprocal_rank(9, k=60)
        assert large < small

    def test_negative_rank_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank(-1, k=60)

    def test_negative_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank(0, k=-1)

    def test_default_k_is_sixty(self) -> None:
        assert DEFAULT_RRF_K == 60
        assert reciprocal_rank(0) == pytest.approx(1 / 61)


class TestCaseASameRanking:
    """A: A B C / A B C -- agreement, so the shared order must survive."""

    def test_order_is_preserved(self) -> None:
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["A", "B", "C"])], k=60)
        assert [c.doc_id for c in fused] == ["A", "B", "C"]

    def test_exact_arithmetic(self) -> None:
        # A: 1/61 + 1/61 = 2/61
        # B: 1/62 + 1/62 = 2/62
        # C: 1/63 + 1/63 = 2/63
        fused = by_id(fuse([run("A", ["A", "B", "C"]), run("B", ["A", "B", "C"])], k=60))
        assert fused["A"].rrf_score == pytest.approx(2 / 61)
        assert fused["B"].rrf_score == pytest.approx(2 / 62)
        assert fused["C"].rrf_score == pytest.approx(2 / 63)

    def test_strictly_decreasing_scores(self) -> None:
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["A", "B", "C"])], k=60)
        assert fused[0].rrf_score > fused[1].rrf_score > fused[2].rrf_score

    def test_every_document_agrees(self) -> None:
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["A", "B", "C"])], k=60)
        assert all(c.agreement == 2 for c in fused)


class TestCaseBOppositeRankings:
    """B: A B C / C B A -- total disagreement, so rank is a near-tie."""

    def test_exact_scores(self) -> None:
        # A: 1/61 + 1/63   (1st in A, 3rd in B)
        # B: 1/62 + 1/62 = 2/62
        # C: 1/63 + 1/61
        fused = by_id(fuse([run("A", ["A", "B", "C"]), run("B", ["C", "B", "A"])], k=60))
        assert fused["A"].rrf_score == pytest.approx(1 / 61 + 1 / 63)
        assert fused["B"].rrf_score == pytest.approx(2 / 62)
        assert fused["C"].rrf_score == pytest.approx(1 / 63 + 1 / 61)

    def test_all_three_scores_are_very_close_but_not_equal(self) -> None:
        # The near-tie is the point. 2/62 sits between 1/61+1/63 and itself:
        #   1/61 + 1/63 = 0.032266458
        #   2/62        = 0.032258064
        #   1/63 + 1/61 = 0.032266458
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["C", "B", "A"])], k=60)
        scores = [c.rrf_score for c in fused]
        assert max(scores) - min(scores) < 1e-4

    def test_rrf_is_not_rank_consistent(self) -> None:
        # A property of the formula, pinned deliberately so it is a decision and
        # not an accident: 1/x is convex, so under a *symmetric* reversal the
        # extreme pair beats the middle by ~0.026%. Rank-consistency (treating
        # reversal as a tie) would need a rank- or score-based variant such as
        # RRFr, which this is not. Documented rather than silently inherited.
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["C", "B", "A"])], k=60)
        assert [c.doc_id for c in fused] == ["A", "C", "B"]
        assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)
        assert fused[2].rrf_score < fused[0].rrf_score

    def test_a_larger_k_damps_the_disagreement_penalty(self) -> None:
        # The documented mitigation, and the reason rrf_k is configurable.
        def spread(k: int) -> float:
            scores = [
                c.rrf_score
                for c in fuse([run("A", ["A", "B", "C"]), run("B", ["C", "B", "A"])], k=k)
            ]
            return max(scores) - min(scores)

        assert spread(60) < spread(10)
        assert spread(10) < spread(1)


class TestCaseCSingleRetrieverOnly:
    """C: A B C / D E F -- disjoint, so each keeps only its own contribution."""

    def test_all_six_survive(self) -> None:
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["D", "E", "F"])], k=60)
        assert len(fused) == 6

    def test_each_gets_exactly_one_contribution(self) -> None:
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["D", "E", "F"])], k=60)
        assert all(c.agreement == 1 for c in fused)
        assert all(len(c.contributions) == 1 for c in fused)

    def test_exact_scores(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C"]), run("B", ["D", "E", "F"])], k=60))
        assert fused["A"].rrf_score == pytest.approx(1 / 61)
        assert fused["B"].rrf_score == pytest.approx(1 / 62)
        assert fused["C"].rrf_score == pytest.approx(1 / 63)
        assert fused["D"].rrf_score == pytest.approx(1 / 61)
        assert fused["E"].rrf_score == pytest.approx(1 / 62)
        assert fused["F"].rrf_score == pytest.approx(1 / 63)

    def test_ties_are_broken_by_best_rank(self) -> None:
        # A, B, C and D, E, F have identical scores pairwise; the documented
        # tie-break prefers the better single rank, which within each list is
        # the earlier one, so the A-list precedes the D-list.
        fused = fuse([run("A", ["A", "B", "C"]), run("B", ["D", "E", "F"])], k=60)
        assert [c.doc_id for c in fused] == ["A", "D", "B", "E", "C", "F"]


class TestCaseDPartialOverlap:
    """D: A B C D / C D E F -- C and D are found by both."""

    def test_both_retrievers_credited_for_the_overlap(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60))
        assert fused["C"].agreement == 2
        assert fused["D"].agreement == 2
        assert fused["A"].agreement == 1
        assert fused["F"].agreement == 1

    def test_overlap_exact_scores(self) -> None:
        # C: 3rd in A (1/63), 1st in B (1/61)
        # D: 4th in A (1/64), 2nd in B (1/62)
        fused = by_id(fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60))
        assert fused["C"].rrf_score == pytest.approx(1 / 63 + 1 / 61)
        assert fused["D"].rrf_score == pytest.approx(1 / 64 + 1 / 62)

    def test_overlap_outranks_either_single_result(self) -> None:
        fused = fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60)
        top = [c.doc_id for c in fused[:2]]
        assert set(top) == {"C", "D"}

    def test_single_result_contributions(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60))
        assert fused["A"].contributions == {"A": pytest.approx(1 / 61)}
        assert fused["B"].contributions == {"A": pytest.approx(1 / 62)}
        assert fused["E"].contributions == {"B": pytest.approx(1 / 63)}
        assert fused["F"].contributions == {"B": pytest.approx(1 / 64)}

    def test_ranks_are_recorded_per_retriever(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60))
        assert fused["C"].ranks == {"A": 2, "B": 0}
        assert fused["D"].ranks == {"A": 3, "B": 1}

    def test_contributions_sum_to_the_fused_score(self) -> None:
        for candidate in fuse(
            [run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60
        ):
            assert sum(candidate.contributions.values()) == pytest.approx(candidate.rrf_score)

    def test_retriever_names_are_reported(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C", "D"]), run("B", ["C", "D", "E", "F"])], k=60))
        assert fused["C"].retrievers == ["A", "B"]
        assert fused["A"].retrievers == ["A"]


class TestCaseEDuplicateIds:
    """E: a retriever that lists the same document twice."""

    def test_duplicate_within_one_retriever_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            fuse([scored_run("A", [("A", 1.0), ("B", 0.5), ("A", 0.1)])])

    def test_the_error_names_the_retriever_and_document(self) -> None:
        # Raw string, not re.escape: the point of the match is that the message
        # names the offending retriever *and* document, which is exactly what
        # the ".*" asserts. Escaping it would assert the literal text ".*".
        with pytest.raises(ValueError, match=r"'bm25'.*'doc-7'"):
            fuse([scored_run("bm25", [("doc-7", 1.0), ("doc-7", 0.5)])])

    def test_the_duplicate_check_runs_before_any_scoring(self) -> None:
        # The check has to fire even when the two entries are adjacent, which
        # is the case a real retriever bug would produce (a document emitted
        # once per matched field, say).
        with pytest.raises(ValueError, match="twice"):
            fuse([scored_run("A", [("A", 1.0), ("A", 1.0)])])

    def test_duplicates_across_retrievers_are_fine(self) -> None:
        # The same document in two different retrievers is the normal case,
        # not an error: that is exactly what fusion is for.
        fused = fuse([run("A", ["X"]), run("B", ["X"])], k=60)
        assert len(fused) == 1
        assert fused[0].agreement == 2

    def test_a_run_cannot_have_a_score_without_an_id(self) -> None:
        with pytest.raises(ValueError, match="every score must belong"):
            RetrievalRun(name="A", ranked_doc_ids=["A"], scores={"A": 1.0, "B": 2.0})

    def test_a_duplicate_is_reported_by_fuse_not_by_construction(self) -> None:
        # The construction-time check would say "2 ids, 1 score", which reads
        # as a missing score rather than a duplicated document. Documenting
        # where the error actually comes from, so the next person looking at a
        # failure knows which layer to blame.
        duplicate = RetrievalRun(name="A", ranked_doc_ids=["A", "A"], scores={"A": 1.0})
        with pytest.raises(ValueError, match="returned document 'A' twice"):
            fuse([duplicate])

    def test_a_run_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            RetrievalRun(name="", ranked_doc_ids=["A"])

    def test_an_empty_run_is_allowed(self) -> None:
        fused = fuse([run("A", []), run("B", ["X"])], k=60)
        assert [c.doc_id for c in fused] == ["X"]
        assert fused[0].agreement == 1


class TestCaseFAndGEmptyRuns:
    """F/G: one or both retrievers returning nothing."""

    def test_one_empty_retriever_leaves_the_other_valid(self) -> None:
        fused = fuse([run("A", []), run("B", ["X", "Y"])], k=60)
        assert [c.doc_id for c in fused] == ["X", "Y"]
        assert all(c.agreement == 1 for c in fused)

    def test_scores_are_the_survivors_own_contributions(self) -> None:
        fused = by_id(fuse([run("A", []), run("B", ["X", "Y"])], k=60))
        assert fused["X"].rrf_score == pytest.approx(1 / 61)
        assert fused["Y"].rrf_score == pytest.approx(1 / 62)

    def test_semantic_only_when_bm25_finds_nothing(self) -> None:
        fused = fuse([run("bm25", []), run("semantic", ["S1", "S2", "S3"])], k=60)
        assert [c.doc_id for c in fused] == ["S1", "S2", "S3"]
        assert all(c.retrievers == ["semantic"] for c in fused)

    def test_both_empty_yields_no_candidates(self) -> None:
        assert fuse([run("A", []), run("B", [])], k=60) == []

    def test_no_runs_at_all_yields_no_candidates(self) -> None:
        assert fuse([], k=60) == []


class TestOrdering:
    def test_sorted_by_descending_score(self) -> None:
        fused = fuse([run("A", ["A", "B", "C", "D"]), run("B", ["D", "C", "B", "A"])], k=60)
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_deterministically_by_doc_id(self) -> None:
        # A tie that survives both earlier keys: every document has the same
        # score AND the same best_rank, so only the final doc_id key can order
        # them. A and B sit at 0 and 1 in one list and 1 and 0 in the other, so
        # both have best_rank 0; the "Z" is the odd one out.
        fused = fuse([run("A", ["A", "B", "Z"]), run("B", ["B", "A", "Z"])], k=60)
        assert [c.doc_id for c in fused] == ["A", "B", "Z"]

    def test_identical_input_gives_identical_output(self) -> None:
        runs = [run("A", ["A", "B", "C"]), run("B", ["C", "B", "A"])]
        assert fuse(runs, k=60) == fuse(runs, k=60)

    def test_retriever_order_does_not_change_the_result(self) -> None:
        # Passing the runs in the other order must not reorder anything: the
        # sort keys are score, best rank, agreement, and doc_id -- none of them
        # mention which retriever produced what.
        forward = fuse([run("bm25", ["A", "B"]), run("semantic", ["B", "A"])], k=60)
        backward = fuse([run("semantic", ["B", "A"]), run("bm25", ["A", "B"])], k=60)
        assert [c.doc_id for c in forward] == [c.doc_id for c in backward]
        assert [c.rrf_score for c in forward] == [c.rrf_score for c in backward]

    def test_retriever_names_do_not_affect_ordering(self) -> None:
        # Renaming the retrievers changes the keys in `contributions` but not
        # the fused scores, so the ranking must be identical.
        a = fuse([run("x", ["A", "B", "C"]), run("y", ["C", "B", "A"])], k=60)
        b = fuse([run("p", ["A", "B", "C"]), run("q", ["C", "B", "A"])], k=60)
        assert [c.doc_id for c in a] == [c.doc_id for c in b]

    def test_negative_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fuse([run("A", ["A"])], k=-5)

    def test_more_retrievers_shift_scores_upward(self) -> None:
        # Three agreeing retrievers must out-score one, for the same document.
        one = fuse([run("A", ["X"])], k=60)[0]
        three = fuse([run("A", ["X"]), run("B", ["X"]), run("C", ["X"])], k=60)[0]
        assert three.rrf_score == pytest.approx(3 * one.rrf_score)
        assert three.rrf_score > one.rrf_score


class TestRawScoresAreNeverUsed:
    """The reason this phase uses ranks: raw scores are not comparable."""

    def test_identical_rankings_score_identically_whatever_the_raw_scores(self) -> None:
        high = fuse(
            [
                run("A", ["A", "B"], {"A": 900.0, "B": 850.0}),
                run("B", ["A", "B"], {"A": 0.9, "B": 0.8}),
            ],
            k=60,
        )
        low = fuse(
            [
                run("A", ["A", "B"], {"A": 1.0, "B": 0.5}),
                run("B", ["A", "B"], {"A": 0.1, "B": 0.0}),
            ],
            k=60,
        )
        assert [c.rrf_score for c in high] == [c.rrf_score for c in low]

    def test_inverting_the_raw_scores_does_not_reorder(self) -> None:
        # The decisive check: a retriever whose scores are exactly reversed
        # still contributes by rank alone, so the fused order cannot move.
        normal = fuse([run("A", ["A", "B", "C"]), run("B", ["B", "A", "C"])], k=60)
        inverted = fuse(
            [
                run("A", ["A", "B", "C"], {"A": 0.1, "B": 0.9, "C": 0.5}),
                run("B", ["B", "A", "C"], {"B": 0.1, "A": 0.9, "C": 0.5}),
            ],
            k=60,
        )
        assert [c.doc_id for c in normal] == [c.doc_id for c in inverted]


class TestFuseTopK:
    def test_truncates_after_the_full_sort(self) -> None:
        fused = fuse_top_k([run("A", ["A", "B", "C", "D", "E"])], 3, k=60)
        assert [c.doc_id for c in fused] == ["A", "B", "C"]

    def test_truncating_cannot_reorder(self) -> None:
        runs = [run("A", ["A", "B", "C", "D", "E"]), run("B", ["E", "D", "C", "B", "A"])]
        full = [c.doc_id for c in fuse(runs, k=60)]
        assert [c.doc_id for c in fuse_top_k(runs, 3, k=60)] == full[:3]

    def test_top_k_beyond_the_candidate_count(self) -> None:
        assert len(fuse_top_k([run("A", ["A"])], 100, k=60)) == 1

    def test_zero_top_k(self) -> None:
        assert fuse_top_k([run("A", ["A", "B"])], 0, k=60) == []

    def test_negative_top_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fuse_top_k([run("A", ["A"])], -1, k=60)

    def test_iterable_wrapper_matches(self) -> None:
        runs = (r for r in [run("A", ["A", "B"]), run("B", ["B", "A"])])
        assert [c.doc_id for c in fuse_to_runs(runs, k=60)] == ["A", "B"]


class TestFusedCandidateModel:
    def test_agreement_counts_contributions(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B"]), run("B", ["A"])], k=60))
        assert fused["A"].agreement == 2
        assert fused["B"].agreement == 1

    def test_best_rank_is_the_minimum(self) -> None:
        fused = by_id(fuse([run("A", ["A", "B", "C"]), run("B", ["C", "A", "B"])], k=60))
        assert fused["C"].best_rank == 0
        assert fused["B"].best_rank == 1
        assert fused["A"].best_rank == 0

    def test_is_frozen(self) -> None:
        candidate = fuse([run("A", ["A"])], k=60)[0]
        with pytest.raises((AttributeError, TypeError)):
            candidate.rrf_score = 9.9  # type: ignore[misc]


class TestSettings:
    def test_reports_the_active_parameters(self) -> None:
        from semantic_search.core.config import HybridConfig

        settings = rrf_settings(HybridConfig(rrf_k=30, candidate_pool=50))
        assert settings["rrf_k"] == 30
        assert settings["candidate_pool"] == 50
        assert settings["method"] == "reciprocal_rank_fusion"

    def test_defaults_match_the_published_constant(self) -> None:
        assert rrf_settings()["rrf_k"] == 60
