"""Bootstrap intervals and paired comparisons, checked against known behaviour.

The tests that matter most here are the ones that would catch the specific ways
a small-n comparison goes wrong: an independent bootstrap used where a paired
one was required, a single seed reported as if it were exact, and an
all-differences-zero case reported as "no significant difference" rather than
"no test was run".
"""

from __future__ import annotations

import random
import statistics

import pytest

from semantic_search.evaluation.stats import (
    DEFAULT_SEEDS,
    Interval,
    bootstrap_ci,
    bootstrap_ci_multi_seed,
    compare,
    paired_difference_ci,
    sign_test,
    wilcoxon_signed_rank,
)

RNG = random.Random(20250930)


def _sample(n: int, mean: float, spread: float = 0.3) -> list[float]:
    return [max(0.0, min(1.0, mean + RNG.gauss(0.0, spread))) for _ in range(n)]


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
class TestBootstrapCI:
    def test_point_equals_sample_mean(self) -> None:
        values = _sample(40, 0.5)
        interval = bootstrap_ci(values, resamples=2000)
        assert interval.point == pytest.approx(statistics.fmean(values))

    def test_interval_contains_the_point_estimate(self) -> None:
        interval = bootstrap_ci(_sample(40, 0.5), resamples=2000)
        assert interval.low <= interval.point <= interval.high

    def test_empty_input(self) -> None:
        interval = bootstrap_ci([], resamples=100)
        assert (interval.point, interval.low, interval.high, interval.n) == (0.0, 0.0, 0.0, 0.0)

    def test_single_observation_is_zero_width(self) -> None:
        interval = bootstrap_ci([0.42], resamples=100)
        assert interval.low == interval.high == 0.42
        assert interval.n == 1

    def test_constant_input_is_zero_width(self) -> None:
        interval = bootstrap_ci([0.5] * 20, resamples=500)
        assert interval.width == pytest.approx(0.0, abs=1e-12)

    def test_zero_variance_narrows_with_more_queries(self) -> None:
        # Sampling noise must shrink as queries are added -- the whole reason a
        # confidence interval exists.
        small = bootstrap_ci(_sample(10, 0.5), resamples=4000)
        large = bootstrap_ci(_sample(200, 0.5), resamples=4000)
        assert large.width < small.width

    def test_deterministic_under_a_fixed_seed(self) -> None:
        values = _sample(30, 0.6)
        first = bootstrap_ci(values, resamples=3000, seed=99)
        second = bootstrap_ci(values, resamples=3000, seed=99)
        assert (first.low, first.high) == (second.low, second.high)

    def test_different_seeds_move_the_endpoints(self) -> None:
        values = _sample(30, 0.6)
        a = bootstrap_ci(values, resamples=3000, seed=1)
        b = bootstrap_ci(values, resamples=3000, seed=2)
        assert (a.low, a.high) != (b.low, b.high)

    def test_bounded_by_the_data(self) -> None:
        # A bootstrap mean cannot leave the range of the observed values.
        values = _sample(50, 0.5, spread=0.2)
        interval = bootstrap_ci(values, resamples=3000)
        assert min(values) <= interval.low <= interval.high <= max(values)

    def test_contains_zero_for_a_mean_below_its_own_resolution(self) -> None:
        # A mean of 0.25 over eight queries, two of them perfect and six of
        # them zero. The interval runs 0.000 to 0.625: this is the case the
        # report has to state as "inconclusive", and it is only distinguishable
        # from a real effect by the interval. Note the lower endpoint lands
        # exactly on 0.0 rather than below it -- the bootstrap cannot resample
        # its way to a mean below the observed one when two successes is the
        # floor.
        interval = bootstrap_ci([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], resamples=5000)
        assert interval.point == pytest.approx(0.25)
        assert interval.contains_zero()
        assert interval.low == pytest.approx(0.0)

    def test_noise_alone_does_not_make_a_mean_unclear(self) -> None:
        # The companion to the test above, and the reason the two are separate.
        # A high-variance mean at small n can still exclude zero; reporting a
        # confidence interval rather than eyeballing the spread is the only
        # way to tell the two apart.
        interval = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.5, 0.2], resamples=5000)
        assert not interval.contains_zero()
        assert interval.width > 0.5

    def test_excludes_zero_for_a_consistently_high_series(self) -> None:
        interval = bootstrap_ci([0.95, 0.97, 0.93, 0.99, 0.96], resamples=3000)
        assert not interval.contains_zero()
        assert interval.low > 0.0

    def test_to_dict_round_trip_fields(self) -> None:
        payload = bootstrap_ci(_sample(10, 0.5), resamples=200).to_dict()
        assert set(payload) == {"name", "point", "low", "high", "n", "resamples", "contains_zero"}

    def test_str_is_readable(self) -> None:
        text = str(Interval(name="p@10", point=0.5, low=0.4, high=0.6, n=75, resamples=100))
        assert "0.500" in text and "0.400" in text and "0.600" in text


class TestBootstrapMultiSeed:
    def test_envelope_contains_every_single_seed_interval(self) -> None:
        values = _sample(30, 0.55)
        envelope, _ = bootstrap_ci_multi_seed(values, resamples=2000, seeds=(1, 2, 3, 4, 5))
        for seed in (1, 2, 3, 4, 5):
            single = bootstrap_ci(values, resamples=2000, seed=seed)
            assert envelope.low <= single.low
            assert envelope.high >= single.high

    def test_envelope_is_at_least_as_wide_as_one_seed(self) -> None:
        values = _sample(30, 0.55)
        envelope, _ = bootstrap_ci_multi_seed(values, resamples=2000, seeds=(1, 2, 3))
        single = bootstrap_ci(values, resamples=2000, seed=1)
        assert envelope.width >= single.width - 1e-12

    def test_point_spread_is_tiny(self) -> None:
        # The point estimate does not move with the seed at all -- only the
        # endpoints do. Reporting a spread here is about Monte-Carlo error in
        # the interval, not about the estimate.
        values = _sample(40, 0.5)
        _, spread = bootstrap_ci_multi_seed(values, resamples=2000)
        assert spread == pytest.approx(0.0, abs=1e-12)

    def test_default_seeds_is_a_tuple_of_ints(self) -> None:
        assert isinstance(DEFAULT_SEEDS, tuple)
        assert all(isinstance(s, int) for s in DEFAULT_SEEDS)
        assert len(DEFAULT_SEEDS) >= 2


# --------------------------------------------------------------------------- #
# paired_difference_ci
# --------------------------------------------------------------------------- #
class TestPairedDifference:
    def test_point_equals_mean_of_differences(self) -> None:
        a = [0.9, 0.8, 0.7, 0.6]
        b = [0.5, 0.5, 0.5, 0.5]
        interval = paired_difference_ci(a, b, resamples=2000)
        expected = statistics.fmean([x - y for x, y in zip(a, b, strict=True)])
        assert interval.point == pytest.approx(expected)

    def test_identical_series_gives_an_interval_containing_zero(self) -> None:
        values = _sample(30, 0.5)
        interval = paired_difference_ci(values, list(values), resamples=3000)
        assert interval.contains_zero()
        assert interval.point == pytest.approx(0.0)

    def test_constant_offset_gives_zero_width_around_the_offset(self) -> None:
        a = [0.7] * 20
        b = [0.5] * 20
        interval = paired_difference_ci(a, b, resamples=1000)
        assert interval.point == pytest.approx(0.2)
        assert interval.width == pytest.approx(0.0, abs=1e-12)
        assert not interval.contains_zero()

    def test_is_tighter_than_an_independent_bootstrap_when_correlated(self) -> None:
        # The property that motivates the whole paired design. Two systems on
        # the same corpus produce highly correlated scores; resampling the
        # differences keeps that correlation, resampling each series separately
        # throws it away and reports an interval the data does not support.
        a = _sample(60, 0.6, spread=0.25)
        b = [min(1.0, max(0.0, x - 0.05)) for x in a]
        correlation = statistics.correlation(a, b)
        assert correlation > 0.8, "test fixture is not correlated enough to be meaningful"

        paired = paired_difference_ci(a, b, resamples=4000)
        independent_low, independent_high = _independent_difference_ci(a, b, resamples=4000)
        assert paired.width < (independent_high - independent_low)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_difference_ci([0.1, 0.2], [0.1])

    def test_empty_series(self) -> None:
        interval = paired_difference_ci([], [], resamples=100)
        assert interval.n == 0 and interval.point == 0.0

    def test_single_pair(self) -> None:
        interval = paired_difference_ci([0.8], [0.3], resamples=100)
        assert interval.point == pytest.approx(0.5)
        assert interval.low == interval.high == pytest.approx(0.5)

    def test_deterministic_under_a_fixed_seed(self) -> None:
        a, b = _sample(25, 0.6), _sample(25, 0.5)
        first = paired_difference_ci(a, b, resamples=3000, seed=7)
        second = paired_difference_ci(a, b, resamples=3000, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_differences_may_be_negative(self) -> None:
        interval = paired_difference_ci([0.1, 0.2, 0.3], [0.5, 0.5, 0.5], resamples=2000)
        assert interval.point < 0
        assert interval.high < 0


def _independent_difference_ci(
    a: list[float], b: list[float], *, resamples: int, seed: int = 12345
) -> tuple[float, float]:
    """Bootstrap each series separately, then subtract -- the wrong method.

    Present so the test above can demonstrate that the paired interval is
    genuinely narrower on correlated inputs. This is the comparison most
    small-n IR papers actually run.
    """
    rng = random.Random(seed)
    means_a: list[float] = []
    means_b: list[float] = []
    for _ in range(resamples):
        means_a.append(statistics.fmean(a[rng.randrange(len(a))] for _ in range(len(a))))
        means_b.append(statistics.fmean(b[rng.randrange(len(b))] for _ in range(len(b))))
    diffs = sorted(x - y for x, y in zip(means_a, means_b, strict=False))
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


# --------------------------------------------------------------------------- #
# sign_test
# --------------------------------------------------------------------------- #
class TestSignTest:
    def test_all_ties_gives_one(self) -> None:
        assert sign_test([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_empty_gives_one(self) -> None:
        assert sign_test([]) == pytest.approx(1.0)

    def test_all_positive_is_significant(self) -> None:
        assert sign_test([0.1] * 12) < 0.01

    def test_all_negative_is_significant(self) -> None:
        assert sign_test([-0.1] * 12) < 0.01

    def test_balanced_is_not_significant(self) -> None:
        assert sign_test([0.1, -0.1] * 6) > 0.5

    def test_never_exceeds_one(self) -> None:
        rng = random.Random(3)
        for _ in range(200):
            diffs = [rng.choice([-0.1, 0.0, 0.2]) for _ in range(rng.randint(1, 30))]
            assert 0.0 <= sign_test(diffs) <= 1.0

    def test_p_value_is_two_sided(self) -> None:
        # 10 of 10 positive is as extreme as 0 of 10 negative.
        assert sign_test([1.0] * 10) == pytest.approx(sign_test([-1.0] * 10))

    def test_ignores_magnitude(self) -> None:
        # A difference of 0.001 and a difference of 5.0 count the same: the
        # sign test says nothing about effect size, which is why the median
        # paired difference is reported alongside it.
        assert sign_test([0.001] * 10) == pytest.approx(sign_test([5.0] * 10))


# --------------------------------------------------------------------------- #
# wilcoxon_signed_rank
# --------------------------------------------------------------------------- #
class TestWilcoxon:
    def test_returns_none_for_too_few_differences(self) -> None:
        assert wilcoxon_signed_rank([0.1, -0.1, 0.2]) is None

    def test_returns_none_for_all_zeros(self) -> None:
        assert wilcoxon_signed_rank([0.0] * 20) is None

    def test_significant_for_a_consistent_shift(self) -> None:
        diffs = [RNG.uniform(0.05, 0.2) for _ in range(25)]
        p = wilcoxon_signed_rank(diffs)
        assert p is not None and p < 0.01

    def test_not_significant_for_noise(self) -> None:
        diffs = [RNG.gauss(0.0, 0.1) for _ in range(40)]
        p = wilcoxon_signed_rank(diffs)
        assert p is not None and p > 0.05

    def test_never_exceeds_one(self) -> None:
        for _ in range(200):
            diffs = [RNG.choice([-0.2, 0.0, 0.3]) for _ in range(RNG.randint(10, 40))]
            p = wilcoxon_signed_rank(diffs)
            if p is not None:
                assert 0.0 <= p <= 1.0

    def test_uses_ranked_magnitudes_not_just_signs(self) -> None:
        # Two samples with the *same* sign pattern -- nine wins, one loss -- so
        # the sign test returns an identical p for both and is structurally
        # blind to what separates them. Wilcoxon ranks by absolute magnitude, so
        # it does see it. The interesting case is not "large differences win":
        # an all-equal sample separates *perfectly* and is significant, which is
        # a real property of a rank test and a reason the p-value is reported as
        # secondary. What Wilcoxon buys is the *ordering* within the positives.
        # Same one huge loss in both; in one the nine positives are the smallest
        # magnitudes available so the loss outranks all of them, in the other
        # they are large enough to outrank it. The second is the significant
        # one, and no sign pattern distinguishes them.
        wins_large = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -5.0]
        wins_small = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.001]

        assert sign_test(wins_large) == pytest.approx(sign_test(wins_small))
        strong = wilcoxon_signed_rank(wins_small)
        weak = wilcoxon_signed_rank(wins_large)
        assert strong is not None and weak is not None
        assert strong < weak

    def test_perfectly_equal_magnitudes_are_maximally_separated(self) -> None:
        # Documented rather than assumed: nine +1s and one -1 is the most
        # separated arrangement a rank test can see, so it produces the
        # smallest p-value. It is also the case most like a real retrieval
        # comparison where one system moved every query by the same amount --
        # which is why an effect size is reported next to every p-value.
        diffs = [1.0] * 9 + [-1.0]
        p = wilcoxon_signed_rank(diffs)
        assert p is not None
        assert p < wilcoxon_signed_rank([1.0] * 5 + [0.5] * 4 + [-1.0])

    def test_is_invariant_to_uniform_rescaling(self) -> None:
        # A known property of a rank test, asserted so it is not mistaken for
        # a bug later: doubling every difference doubles the rank sum and the
        # mean, which cancel. The report therefore has to give the median paired
        # difference alongside the p-value -- the p-value alone says nothing
        # about how big the effect was.
        diffs = [0.9, 0.4, -0.2, 0.7, 0.3, 0.1, -0.4, 0.2, 0.5, 0.6, 0.8, 0.05]
        assert wilcoxon_signed_rank(diffs) == pytest.approx(
            wilcoxon_signed_rank([d * 1000.0 for d in diffs])
        )


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
class TestCompare:
    def test_reports_wins_losses_ties(self) -> None:
        a = [0.9, 0.9, 0.5, 0.4]
        b = [0.8, 0.7, 0.5, 0.6]
        result = compare(a, b, system_a="a", system_b="b", resamples=1000)
        assert (result.wins, result.losses, result.ties) == (2, 1, 1)

    def test_median_difference(self) -> None:
        result = compare(
            [0.9, 0.8, 0.7, 0.6], [0.5, 0.5, 0.5, 0.5], system_a="a", system_b="b", resamples=500
        )
        # Four differences: 0.4, 0.3, 0.2, 0.1. Even length, so the median is the
        # mean of the two middle ones -- 0.25, not 0.2 and not the mean (0.25
        # again here, which is a coincidence of this fixture, not a rule).
        assert result.median_difference == pytest.approx(0.25)
        assert result.mean_difference == pytest.approx(0.25)

    def test_median_of_even_length_series(self) -> None:
        result = compare([0.9, 0.7], [0.5, 0.5], system_a="a", system_b="b", resamples=200)
        assert result.median_difference == pytest.approx(0.3)

    def test_means_are_reported_for_both_systems(self) -> None:
        a, b = [0.9, 0.7], [0.5, 0.3]
        result = compare(a, b, system_a="a", system_b="b", resamples=500)
        assert result.mean_a == pytest.approx(0.8)
        assert result.mean_b == pytest.approx(0.4)
        assert result.mean_difference == pytest.approx(0.4)

    def test_all_ties_is_reported_as_inconclusive_not_significant(self) -> None:
        # The distinction the report has to preserve: a zero difference with no
        # evidence is not the same claim as a demonstrated null.
        values = _sample(20, 0.5)
        result = compare(values, list(values), system_a="a", system_b="b", resamples=2000)
        assert result.wins == 0 and result.losses == 0 and result.ties == 20
        assert result.difference.contains_zero()
        assert result.wilcoxon_p is None

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            compare([0.1, 0.2, 0.3], [0.1], system_a="a", system_b="b")

    def test_to_dict_is_serialisable(self) -> None:
        import json

        result = compare([0.9, 0.1], [0.5, 0.5], system_a="a", system_b="b", resamples=200)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["system_a"] == "a"
        assert "difference_ci" in payload
        assert payload["wins"] == 1

    def test_empty_comparison_does_not_crash(self) -> None:
        # A stratum with no queries in it reaches here, and computing the
        # median of an empty difference list used to raise IndexError -- a
        # crash in the one place that is supposed to survive a gap in the data.
        result = compare([], [], system_a="a", system_b="b", resamples=100)
        assert result.difference.n == 0
        assert result.sign_test_p == 1.0
        assert result.wilcoxon_p is None
        assert result.median_difference == 0.0
        assert (result.wins, result.losses, result.ties) == (0, 0, 0)

    def test_single_query_comparison(self) -> None:
        # One query is not a comparison, but it must not be an error either:
        # the interval is exactly the difference and the sign test is undefined
        # in the only way that is still reportable (p = 1.0).
        result = compare([0.9], [0.4], system_a="a", system_b="b", resamples=50)
        assert result.median_difference == pytest.approx(0.5)
        assert result.difference.low == result.difference.high == pytest.approx(0.5)
        assert result.wins == 1 and result.losses == 0 and result.ties == 0
