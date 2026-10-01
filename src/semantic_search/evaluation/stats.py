"""Bootstrap confidence intervals and paired comparisons for per-query metrics.

What this module does and does not quantify
-------------------------------------------
Every number here is a statement about **one thing only: the sampling
distribution of the mean over the queries in this dataset.** Resample the
queries, recompute the mean, and the spread of those means is the interval.

It is *not* a statement about:

* whether these 75 queries represent real user traffic (they do not; they were
  written by the person who wrote the code),
* whether one judge agrees with another (there is one judge),
* whether the corpus is representative (it is 300 Wikipedia articles chosen by
  a size setting, not a traffic sample),
* whether pooling at depth 50 caught every relevant document (it did not).

Those are all real sources of error, all of them plausibly larger than the
intervals computed here, and none of them are reduced by running more
resamples. :func:`bootstrap_ci` is a narrow instrument and the report says so
wherever its output appears.

Paired, not independent
-----------------------
Comparing two systems means resampling the **per-query differences**, applying
the same resampled query indices to both systems in each replicate. The
alternative -- bootstrapping each system separately and subtracting the two
intervals -- throws away the pairing, and the pairing is the only source of
power at n≈75. Systems in this evaluation share a retriever and a corpus, so
their per-query scores are strongly correlated; an independent bootstrap treats
them as unrelated and inflates the difference's interval by roughly
``sqrt(2(1 - rho))``, where ``rho`` is the correlation between the two systems'
per-query scores. At rho=0.8 that is a 60% wider interval than the data supports.

The sign and Wilcoxon tests are reported as secondary and descriptive. Wilcoxon
assumes the distribution of differences is symmetric, which is precisely what
fails when a system moves a query from rank 1 to rank 50 -- the commonest and
largest effect in retrieval. The sign test assumes only that the two systems
are exchangeable, which is why it is here at all. Neither is a significance
claim: the two primary comparisons were registered before the run, and
everything else is reported without one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

#: Default resamples. 10,000 is enough that Monte-Carlo error on a percentile
#: endpoint is small relative to the interval's own width, which matters more
#: than the raw count: 1,000 would be visibly unstable.
DEFAULT_RESAMPLES = 10_000

#: Seeds run for the main point estimates. More than one because a single seed
#: hides the Monte-Carlo error in the endpoints, and an interval whose endpoints
#: move with the seed is being reported as more precise than it is.
DEFAULT_SEEDS: tuple[int, ...] = (20250930, 11, 4242, 8675309, 20250101)


@dataclass(slots=True, frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval around it."""

    name: str
    point: float
    low: float
    high: float
    n: int
    resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "name": self.name,
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "n": self.n,
            "resamples": self.resamples,
            "contains_zero": self.contains_zero(),
        }

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass(slots=True, frozen=True)
class PairedComparison:
    """The result of comparing one system against another, query by query.

    ``median_difference`` and ``wins/losses/ties`` are the load-bearing fields.
    A mean difference near zero with 3 wins and 3 losses is a different fact
    from a mean difference near zero with 30 wins and 0 losses, and the mean
    alone cannot tell them apart. Both are reported, and neither is a
    significance claim.
    """

    name: str
    system_a: str
    system_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    difference: Interval
    wins: int
    losses: int
    ties: int
    median_difference: float
    sign_test_p: float
    wilcoxon_p: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "system_a": self.system_a,
            "system_b": self.system_b,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "mean_difference": round(self.mean_difference, 6),
            "difference_ci": self.difference.to_dict(),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "median_difference": round(self.median_difference, 6),
            "sign_test_p": round(self.sign_test_p, 6),
            "wilcoxon_p": None if self.wilcoxon_p is None else round(self.wilcoxon_p, 6),
        }


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower]) * (1 - weight) + float(ordered[upper]) * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    """Median of ``values``, 0.0 for an empty sequence.

    ``statistics.median`` raises on empty input, and an empty comparison is a
    reporting condition here (a stratum nobody wrote queries for), not a fault.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    middle = n // 2
    if n % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    name: str = "metric",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEEDS[0],
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``.

    The resampling unit is the **query**. Bootstrapping documents instead would
    claim a precision about individual documents that says nothing about whether
    the mean of 75 queries is stable, which is the only question here.

    Returns a zero-width interval at the mean for a single observation, and an
    interval at 0.0 for no observations -- an empty input is a reporting
    condition, not an error.
    """
    n = len(values)
    if n == 0:
        return Interval(name=name, point=0.0, low=0.0, high=0.0, n=0, resamples=resamples)
    if n == 1:
        only = float(values[0])
        return Interval(name=name, point=only, low=only, high=only, n=1, resamples=resamples)

    rng = random.Random(seed)
    n_resamples = max(1, resamples)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = _percentile(means, alpha / 2)
    high = _percentile(means, 1 - alpha / 2)
    return Interval(
        name=name,
        point=_mean([float(v) for v in values]),
        low=low,
        high=high,
        n=n,
        resamples=n_resamples,
    )


def bootstrap_ci_multi_seed(
    values: Sequence[float],
    *,
    name: str = "metric",
    resamples: int = DEFAULT_RESAMPLES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alpha: float = 0.05,
) -> tuple[Interval, float]:
    """Bootstrap across several seeds, returning the widest interval and its spread.

    The returned ``Interval`` is the **most conservative** one -- the seed whose
    low endpoint is lowest and whose high endpoint is highest. Reporting a
    single seed's interval would report one draw from the Monte-Carlo
    distribution; reporting the envelope reports the range of draws, which is
    the honest summary at these sample sizes.

    The second return value is the spread of the point estimates across seeds,
    which is the part of the uncertainty that comes from the resampling itself.
    """
    intervals = [
        bootstrap_ci(values, name=name, resamples=resamples, seed=seed, alpha=alpha)
        for seed in seeds
    ]
    widest_low = min(i.low for i in intervals)
    widest_high = max(i.high for i in intervals)
    envelope = Interval(
        name=name,
        point=intervals[0].point,
        low=widest_low,
        high=widest_high,
        n=intervals[0].n,
        resamples=resamples * len(seeds),
    )
    points = [i.point for i in intervals]
    return envelope, max(points) - min(points)


def paired_difference_ci(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    name: str = "difference",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEEDS[0],
    alpha: float = 0.05,
) -> Interval:
    """Bootstrap interval for the mean of per-query ``a - b``.

    Each replicate resamples **query indices** and applies the same indices to
    both series. This is the whole point of the function: the two systems were
    run on the same queries, and the correlation between their scores is real
    information about how the comparison should be made, not noise to be
    averaged away.

    The point estimate is ``mean(a) - mean(b)``, which equals the mean of the
    paired differences.

    Length mismatch is an error rather than a truncation: silently pairing the
    first ``min(n, m)`` queries would compare different questions against each
    other and produce a number that looks like a comparison and is not.
    """
    if len(a_values) != len(b_values):
        raise ValueError(
            f"paired_difference_ci needs equal-length series, got {len(a_values)} and "
            f"{len(b_values)}. Truncating would compare different queries."
        )
    n = len(a_values)
    if n == 0:
        return Interval(name=name, point=0.0, low=0.0, high=0.0, n=0, resamples=resamples)

    diffs = [float(a) - float(b) for a, b in zip(a_values, b_values, strict=True)]
    if n == 1:
        return Interval(
            name=name, point=diffs[0], low=diffs[0], high=diffs[0], n=1, resamples=resamples
        )

    rng = random.Random(seed)
    n_resamples = max(1, resamples)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return Interval(
        name=name,
        point=_mean(diffs),
        low=_percentile(means, alpha / 2),
        high=_percentile(means, 1 - alpha / 2),
        n=n,
        resamples=n_resamples,
    )


def sign_test(diffs: Sequence[float]) -> float:
    """Two-sided exact binomial p-value on the signs of the non-zero differences.

    Makes no distributional assumption beyond exchangeability, which is why it
    is the primary non-parametric test here. The null is that each non-zero
    difference is equally likely to be positive or negative -- i.e. that the two
    systems are interchangeable.
    """
    positives = sum(1 for d in diffs if d > 0)
    negatives = sum(1 for d in diffs if d < 0)
    trials = positives + negatives
    if trials == 0:
        return 1.0
    return _exact_two_sided_binomial(positives, trials)


def _exact_two_sided_binomial(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value, summing probabilities no more likely than observed.

    Doubled the smaller tail. This is the standard "probability of a result at
    least as extreme" definition, and it can exceed 1.0 when the observed count
    is near the middle; the cap is applied so a p-value is never reported above
    1.
    """
    if trials <= 0:
        return 1.0
    p = 0.5
    observed = _binomial_pmf(successes, trials, p)
    total = 0.0
    for k in range(trials + 1):
        probability = _binomial_pmf(k, trials, p)
        if probability <= observed + 1e-12:
            total += probability
    return min(1.0, total)


def _binomial_pmf(successes: int, trials: int, p: float) -> float:
    return math.comb(trials, successes) * (p**successes) * ((1 - p) ** (trials - successes))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(diffs: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value via the normal approximation.

    Returns ``None`` when the test is undefined -- all differences zero, or too
    few non-zero differences for the approximation to mean anything. ``None``
    rather than 1.0, because "not applicable" and "no evidence of a difference"
    are different claims and collapsing them would report a test that never ran
    as a test that found nothing.

    The normal approximation is used rather than the exact distribution because
    the exact version needs the null distribution of rank sums for arbitrary
    ``n`` with ties, which is a table this project should not carry. With
    ``n >= 10`` non-zero differences the approximation is standard; below that
    the sign test is the one to read, and the report says which to read.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 10:
        return None

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    position = 0
    while position < n:
        end = position
        while end + 1 < n and abs(nonzero[order[end + 1]]) == abs(nonzero[order[position]]):
            end += 1
        average_rank = (position + end + 2) / 2.0  # ranks are 1-based
        for index in range(position, end + 1):
            ranks[order[index]] = average_rank
        position = end + 1

    positive_rank_sum = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    total = n * (n + 1) / 2.0
    expected = total / 2.0
    # Tie-corrected variance. Without the correction, tied absolute differences
    # -- common here, where many queries score identically under both systems --
    # make the p-value too small.
    correction = 0.0
    seen: dict[float, int] = {}
    for d in nonzero:
        key = abs(d)
        seen[key] = seen.get(key, 0) + 1
    for count in seen.values():
        if count > 1:
            correction += (count**3 - count) / 12.0
    variance = (n * (n + 1) * (2 * n + 1) / 24.0) - correction
    if variance <= 0.0:
        return None

    z = (positive_rank_sum - expected) / math.sqrt(variance)
    return min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z))))


def compare(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    system_a: str,
    system_b: str,
    name: str = "metric",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEEDS[0],
) -> PairedComparison:
    """Full paired comparison of one system against another on one metric.

    Every field is descriptive. The interval says what the query-sampling noise
    is; the wins/losses/ties say what actually happened per query; the two
    p-values are secondary and assumed to be the weaker of the evidence.
    """
    if len(a_values) != len(b_values):
        raise ValueError(
            f"compare needs equal-length series for {system_a} and {system_b}: got "
            f"{len(a_values)} and {len(b_values)}."
        )
    diffs = [float(a) - float(b) for a, b in zip(a_values, b_values, strict=True)]
    median = _median(diffs)
    interval = paired_difference_ci(a_values, b_values, name=name, resamples=resamples, seed=seed)
    return PairedComparison(
        name=name,
        system_a=system_a,
        system_b=system_b,
        mean_a=_mean([float(v) for v in a_values]),
        mean_b=_mean([float(v) for v in b_values]),
        mean_difference=interval.point,
        difference=interval,
        wins=sum(1 for d in diffs if d > 0),
        losses=sum(1 for d in diffs if d < 0),
        ties=sum(1 for d in diffs if d == 0),
        median_difference=median,
        sign_test_p=sign_test(diffs),
        wilcoxon_p=wilcoxon_signed_rank(diffs),
    )
