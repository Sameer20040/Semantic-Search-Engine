"""Aggregation, comparison, and error analysis for an evaluation run.

What this module refuses to produce
-----------------------------------
Three things are absent by decision, and each absence is load-bearing:

**No winner.** Systems are listed, not ranked, and no arm is described as best.
A table with a "best" column invites the reader to ship whichever row is
topmost, and at n=80 with intervals that mostly overlap the top row is often
inside the noise of the rows beneath it. The order of the rows is the order the
arms were registered in.

**No combined score.** There is no weighted sum of P@10, R@10, and MRR. Any
such number is a choice of weights pretending to be a measurement, and the
choice would silently encode which system the author preferred. The metrics
disagree in real runs, and that disagreement is the finding.

**No pooled cross-stratum mean.** Four strata measuring different tasks are
never averaged into one number. A mean over 25 title-matching queries and 30
hand-judged topical ones describes neither population.

What the intervals do and do not cover
--------------------------------------
Every interval here is a bootstrap over **per-query values**, resampling
queries. It quantifies query-sampling noise at n=80 and nothing else. Corpus
choice, query-author bias, single-judge bias, and pooling incompleteness are
all real and all larger. They are not in the interval, and no wording in this
module implies otherwise.

A difference whose interval contains zero is reported as *inconclusive*, never
as "no difference". The two are different claims: the first is what was
measured, the second asserts that no effect exists, which nothing here can
establish.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from semantic_search.data.loader import DocumentStore
from semantic_search.data.tokenizer import Tokenizer
from semantic_search.evaluation.metrics import hits_to_ranked_ids
from semantic_search.evaluation.models import EvalDataset, PerQueryResult
from semantic_search.evaluation.stats import (
    DEFAULT_SEEDS,
    Interval,
    PairedComparison,
    bootstrap_ci_multi_seed,
    compare,
)
from semantic_search.index.bm25 import BM25Index, BM25Result, BM25Searcher
from semantic_search.search.semantic import SemanticSearchService

#: The metrics reported for every system. Order is the order they are printed
#: in, and it is the order they were registered in -- not sorted by value.
METRICS: tuple[str, ...] = (
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
    "reciprocal_rank",
    "ndcg_at_5",
    "ndcg_at_10",
    "average_precision",
)

#: Metrics that are only meaningful when more than one document is judged
#: relevant. With a single gold document, AP reduces to 1/rank and NDCG@k to
#: 1/log2(rank+1) -- both strictly monotone in the same quantity MRR already
#: measures, so reporting them on a single-gold stratum would look like extra
#: evidence while adding none.
MULTI_JUDGMENT_METRICS: frozenset[str] = frozenset({"ndcg_at_5", "ndcg_at_10", "average_precision"})

#: The two comparisons declared as primary before the run, so that the report
#: cannot retroactively promote whichever arm happened to win. Everything else
#: is reported descriptively with no significance claim attached.
PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("hybrid_rrf", "bm25"),
    ("hybrid_rrf", "semantic"),
)


def _value(result: PerQueryResult, metric: str) -> float:
    return float(getattr(result, metric))


def series(results: Sequence[PerQueryResult], system: str, metric: str) -> list[float]:
    """Per-query values of one metric for one system, in dataset order.

    The order is the dataset's, and every arm is indexed by ``query_id`` before
    the values are pulled, so two series are always aligned query-for-query. A
    paired comparison on misaligned series is meaningless and would still
    produce a confident-looking number.
    """
    return [
        _value(r, metric) for r in sorted(results, key=lambda r: r.query_id) if r.system == system
    ]


def by_query_id(results: Sequence[PerQueryResult], system: str) -> dict[str, PerQueryResult]:
    return {r.query_id: r for r in results if r.system == system}


def aggregate_table(
    results: Sequence[PerQueryResult],
    *,
    strata: Sequence[str] | None = None,
    resamples: int = 10_000,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{stratum: {system: {metric: interval}}}`` with intervals on every cell.

    Intervals are the multi-seed **envelope** -- the most conservative of the
    five, not one seed's draw. A single seed's interval looks more precise than
    it is, and the point-estimate spread across seeds is reported alongside so
    the Monte-Carlo component is visible rather than hidden in a rounding.
    """
    present = list(strata) if strata else sorted({r.stratum for r in results})
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for stratum in present:
        subset = [r for r in results if r.stratum == stratum]
        if not subset:
            continue
        systems = sorted({r.system for r in subset})
        table[stratum] = {}
        for system in systems:
            rows = [r for r in subset if r.system == system]
            cell: dict[str, Any] = {}
            for metric in METRICS:
                values = [_value(r, metric) for r in rows]
                envelope, spread = bootstrap_ci_multi_seed(
                    values, name=metric, resamples=resamples, seeds=seeds
                )
                cell[metric] = {**envelope.to_dict(), "point_estimate_spread": round(spread, 9)}
            cell["n_queries"] = len(rows)
            cell["unjudged_at_10_mean"] = round(sum(r.unjudged_at_10 for r in rows) / len(rows), 6)
            # Printed alongside the unjudged fraction, because a fraction with no
            # denominator reads as a coverage problem when it is often just a
            # single-gold stratum's arithmetic.
            cell["n_judged_per_query"] = round(
                sum(len(r.relevant_doc_ids) for r in rows) / len(rows), 6
            )
            table[stratum][system] = cell
    return table


def comparisons(
    results: Sequence[PerQueryResult],
    *,
    stratum: str,
    systems: Sequence[str],
    resamples: int = 10_000,
    seed: int = DEFAULT_SEEDS[0],
) -> list[PairedComparison]:
    """Paired comparisons within one stratum.

    Runs every ordered pair among the arms' *complements* to the primary
    baseline, not every pair in the matrix. Fifteen pairs at n=80 would be
    fifteen chances to find something that looks significant, and only two of
    them were registered in advance.
    """
    subset = [r for r in results if r.stratum == stratum]
    available = [s for s in systems if any(r.system == s for r in subset)]
    out: list[PairedComparison] = []
    for a_name, b_name in PRIMARY_COMPARISONS:
        if a_name not in available or b_name not in available:
            continue
        a_by_id = by_query_id(subset, a_name)
        b_by_id = by_query_id(subset, b_name)
        shared = sorted(set(a_by_id) & set(b_by_id))
        if not shared:
            continue
        for metric in METRICS:
            a_values = [_value(a_by_id[q], metric) for q in shared]
            b_values = [_value(b_by_id[q], metric) for q in shared]
            out.append(
                compare(
                    a_values,
                    b_values,
                    system_a=a_name,
                    system_b=b_name,
                    name=f"{stratum}/{metric}",
                    resamples=resamples,
                    seed=seed,
                )
            )
    return out


def equivalence_audit(
    results: Sequence[PerQueryResult], *, control: str = "hybrid_qp_on"
) -> dict[str, Any]:
    """Did query processing change the string, and did it change the ranking?

    Two questions with two denominators, reported side by side. Collapsing
    them is how a no-op gets reported as a small improvement: a processor that
    rewrites nothing obviously cannot change a ranking, and a processor that
    rewrites something *usually* does not change a ranking either, and only the
    first number is interesting.

    ``identical_top10`` is the honest headline for the ON/OFF pair. The counts
    carry their denominators because "9 of 80 queries changed" reads very
    differently from "9 changed" with no scale attached.
    """
    arms = sorted({r.system for r in results if r.system != control})
    audit: dict[str, Any] = {"control": control, "arms": {}}
    for arm in arms:
        arm_rows = {r.query_id: r for r in results if r.system == arm}
        control_rows = {r.query_id: r for r in results if r.system == control}
        shared = sorted(set(arm_rows) & set(control_rows))
        same_string = sum(
            1
            for q in shared
            if arm_rows[q].query_for_retrieval == control_rows[q].query_for_retrieval
        )
        same_ranking = sum(
            1 for q in shared if arm_rows[q].returned_doc_ids == control_rows[q].returned_doc_ids
        )
        modified = sum(1 for q in shared if arm_rows[q].was_modified)
        audit["arms"][arm] = {
            "n_queries": len(shared),
            "string_identical": same_string,
            "string_identical_fraction": round(same_string / len(shared), 6) if shared else 0.0,
            "ranking_identical": same_ranking,
            "ranking_identical_fraction": round(same_ranking / len(shared), 6) if shared else 0.0,
            "queries_modified": modified,
            "rewrites": sorted({rewrite for q in shared for rewrite in arm_rows[q].corrections}),
            "changed_ranking_queries": [
                {
                    "query_id": q,
                    "query": arm_rows[q].query,
                    "retrieved_with": arm_rows[q].query_for_retrieval,
                    "rewrites": list(arm_rows[q].corrections),
                    "gold_rank_with_processing": arm_rows[q].gold_rank,
                    "gold_rank_control": control_rows[q].gold_rank,
                }
                for q in shared
                if arm_rows[q].returned_doc_ids != control_rows[q].returned_doc_ids
            ],
        }
    return audit


def net_effect(results: Sequence[PerQueryResult], *, arm: str, control: str) -> dict[str, Any]:
    """Miss-to-hit and hit-to-miss counts for one arm against a control.

    A mean-MRR delta of 0.004 is equally consistent with one query fixed and
    another broken, and with nothing happening. The counts say which, and the
    rewrites say why.
    """
    arm_rows = by_query_id(results, arm)
    control_rows = by_query_id(results, control)
    shared = sorted(set(arm_rows) & set(control_rows))
    gained: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    for query_id in shared:
        before, after = control_rows[query_id].gold_rank, arm_rows[query_id].gold_rank
        entry = {
            "query_id": query_id,
            "query": arm_rows[query_id].query,
            "gold_rank_control": before,
            "gold_rank_arm": after,
            "rewrites": list(arm_rows[query_id].corrections),
        }
        if before == 0 and after > 0:
            gained.append(entry)
        elif before > 0 and after == 0:
            lost.append(entry)
        elif before > 0 and after > 0 and after != before:
            (gained if after < before else lost).append(entry)
    return {
        "arm": arm,
        "control": control,
        "n_queries": len(shared),
        "gained": gained,
        "lost": lost,
        "net": len(gained) - len(lost),
    }


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
#: Each category is a predicate over the per-system gold ranks. A category with
#: no members is omitted from the report entirely -- printing an empty section
#: titled "Harmful rewrites" implies the analysis looked for them and implies
#: something about the reader's expectations, and an empty table would say
#: "0 found" which is a different and more confident claim than "not evaluated".
ERROR_CATEGORIES: tuple[tuple[str, str, Any], ...] = (
    (
        "all_systems_miss",
        "No arm found the relevant document anywhere in its ranking",
        lambda ranks: all(rank == 0 for rank in ranks.values()),
    ),
    (
        "both_single_retrievers_miss",
        "BM25 and Semantic both missed; nothing for fusion to combine",
        lambda ranks: ranks.get("bm25", 0) == 0 and ranks.get("semantic", 0) == 0,
    ),
    (
        "semantic_only_miss",
        "BM25 found it, Semantic did not",
        lambda ranks: ranks.get("bm25", 0) > 0 and ranks.get("semantic", 0) == 0,
    ),
    (
        "bm25_only_miss",
        "Semantic found it, BM25 did not",
        lambda ranks: ranks.get("semantic", 0) > 0 and ranks.get("bm25", 0) == 0,
    ),
    (
        "hybrid_recovers",
        "One retriever missed it and Hybrid still found it",
        lambda ranks: ranks.get("hybrid_rrf", 0) > 0
        and 0 in (ranks.get("bm25", 0), ranks.get("semantic", 0)),
    ),
    (
        "hybrid_loses",
        "A retriever found it and Hybrid did not",
        lambda ranks: ranks.get("hybrid_rrf", 0) == 0
        and any(ranks.get(name, 0) > 0 for name in ("bm25", "semantic")),
    ),
    (
        "found_but_deep",
        "Every arm found it, none of them in the top 10",
        # The `any` is load-bearing: without it an all-missed query satisfies
        # `all(rank > 10 ...)` vacuously, because the comprehension filters the
        # misses out before `all` ever sees them. Two categories claiming the
        # same query for opposite reasons is a reporting bug that reads as a
        # finding.
        lambda ranks: bool(ranks)
        and any(rank > 0 for rank in ranks.values())
        and all(rank > 10 for rank in ranks.values() if rank > 0),
    ),
)

#: The two non-error groups, reported separately because folding them into the
#: error analysis would mislabel them. A query every arm puts first is not an
#: error, it is a query carrying no discriminative information -- and if it is
#: a large share of the dataset, it is the single most important number in the
#: report, because it says the mean is high for uninteresting reasons.
EASY_QUERIES: tuple[tuple[str, str, Any], ...] = (
    (
        "all_systems_agree_at_rank_1",
        "Every arm examined put the relevant document first; the query "
        "discriminates between nothing",
        lambda ranks: bool(ranks) and all(rank == 1 for rank in ranks.values()),
    ),
    (
        "all_systems_agree_in_top_3",
        "Every arm examined put the relevant document in the top 3",
        lambda ranks: bool(ranks) and all(1 <= rank <= 3 for rank in ranks.values()),
    ),
)


def _buckets() -> dict[str, list[dict[str, Any]]]:
    names = [name for name, _, _ in (*ERROR_CATEGORIES, *EASY_QUERIES)]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        buckets[name] = []
    return buckets


def error_analysis(
    results: Sequence[PerQueryResult],
    *,
    arms: Sequence[str] = ("bm25", "semantic", "hybrid_rrf"),
    min_members: int = 1,
) -> dict[str, Any]:
    """Group queries by what actually went wrong, and report only real groups.

    Every member carries the query, the gold document's title, and each arm's
    rank for it, so a reader can check the claim against
    ``per_query_results.jsonl`` rather than taking the grouping on trust. No
    example is written by hand: an invented error case is a fabricated finding
    and would be indistinguishable from a real one in the output.

    Categories with fewer than ``min_members`` members are listed under
    ``categories_suppressed`` and not printed. An empty table headed "harmful
    rewrites" says "0 found", which asserts the analysis ran and found nothing;
    suppressing it says only that there was too little to discuss. The two are
    not the same claim.
    """
    per_query: dict[str, dict[str, PerQueryResult]] = {}
    for record in results:
        per_query.setdefault(record.query_id, {})[record.system] = record

    error_buckets = _buckets()
    easy_buckets = _buckets()
    for query_id, records in sorted(per_query.items()):
        ranks = {arm: records[arm].gold_rank for arm in arms if arm in records}
        if not ranks:
            continue
        sample = next(iter(records.values()))
        member = {
            "query_id": query_id,
            "query": sample.query,
            "stratum": sample.stratum,
            "gold_titles": list(sample.gold_titles),
            "gold_rank": dict(ranks),
        }
        for name, _, predicate in ERROR_CATEGORIES:
            if predicate(ranks):
                error_buckets[name].append(member)
        for name, _, predicate in EASY_QUERIES:
            if predicate(ranks):
                easy_buckets[name].append(member)

    easy_ids = {m["query_id"] for name in easy_buckets for m in easy_buckets[name]}

    def _section(buckets: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        descriptions = {
            name: description for name, description, _ in (*ERROR_CATEGORIES, *EASY_QUERIES)
        }
        return {
            name: {
                "description": descriptions[name],
                "n": len(members),
                "query_ids": [m["query_id"] for m in members],
                "examples": members[:5],
            }
            for name, members in buckets.items()
            if len(members) >= min_members
        }

    errors = _section(error_buckets)
    easy = _section(easy_buckets)
    unclassified = [
        query_id
        for query_id in sorted(per_query)
        if not any(error_buckets[name] for name in error_buckets) and query_id not in easy_ids
    ]
    return {
        "arms_examined": list(arms),
        "n_queries": len(per_query),
        "n_easy_queries": len(easy_ids),
        "easy_fraction": round(len(easy_ids) / len(per_query), 6) if per_query else 0.0,
        "n_unclassified": len(unclassified),
        "unclassified_query_ids": unclassified,
        "categories_reported": sorted(errors),
        "categories_suppressed": sorted(name for name in error_buckets if name not in errors),
        "categories": errors,
        "non_discriminating": easy,
    }


def format_table(
    aggregate: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    systems: Sequence[str],
    metrics: Sequence[str] = METRICS,
    label: Mapping[str, str] | None = None,
) -> str:
    """A plain-text table with the interval printed next to every point estimate.

    CIs belong beside the numbers, not in an appendix. A table of point
    estimates invites a reader to treat a 0.03 gap between two rows as a fact,
    and at n=80 it very often is not.
    """
    labels = label or {}
    lines: list[str] = []
    for stratum, cells in aggregate.items():
        present = [s for s in systems if s in cells]
        if not present:
            continue
        lines.append("")
        lines.append(f"=== stratum: {stratum}  (n={cells[present[0]]['n_queries']} queries) ===")
        header = f"{'system':<34}" + "".join(
            f"{m.replace('precision_at_', 'P@').replace('_at_', '@'):>22}" for m in metrics
        )
        lines.append(header)
        lines.append("-" * len(header))
        for system in present:
            row = f"{labels.get(system, system):<34}"
            for metric in metrics:
                cell = cells[system].get(metric)
                if cell is None:
                    row += f"{'n/a':>22}"
                    continue
                row += f"{cell['point']:>9.3f} [{cell['low']:>5.3f},{cell['high']:>5.3f}]"
            lines.append(row)
        unjudged = cells[present[0]].get("unjudged_at_10_mean", 0.0)
        n_judged = cells[present[0]].get("n_judged_per_query", 0)
        # The fraction is of top-10 documents carrying no judgment RECORD, not of
        # documents that are non-relevant. For the known-item strata that number
        # is necessarily 1 - 1/10: exactly one document is judged, so nine of
        # every ten are unjudged by construction, and a reader who saw "90%
        # never judged" without that context would read it as pooling failure.
        # The pooling-incompleteness number is reported per stratum by
        # `format_pool`, which is where it belongs.
        lines.append(
            f"  mean judgments per query: {n_judged:.1f}; fraction of the top-10 with "
            f"no judgment record: {unjudged:.3f}"
        )
        if n_judged <= 1.01:
            lines.append(
                "    (single-gold stratum: exactly one document is judged, so this "
                "is 1-1/10 by construction and is NOT a pooling-coverage figure)"
            )
    return "\n".join(lines)


def format_comparisons(pairs: Iterable[PairedComparison]) -> str:
    """Print paired comparisons, with inconclusive ones labelled as such."""
    lines = ["", "=== paired comparisons (bootstrap over per-query differences) ==="]
    lines.append(
        "Intervals cover query-sampling noise only. A difference whose interval "
        "contains zero is inconclusive at this n, not a demonstrated null."
    )
    for pair in pairs:
        verdict = "INCONCLUSIVE" if pair.difference.contains_zero() else "interval excludes zero"
        lines.append("")
        lines.append(
            f"{pair.name}  {pair.system_a} ({pair.mean_a:.4f}) vs "
            f"{pair.system_b} ({pair.mean_b:.4f})"
        )
        lines.append(
            f"  difference {pair.mean_difference:+.4f} "
            f"[{pair.difference.low:+.4f}, {pair.difference.high:+.4f}]  n={pair.difference.n}  {verdict}"
        )
        lines.append(
            f"  per-query: {pair.wins} wins, {pair.losses} losses, {pair.ties} ties; "
            f"median difference {pair.median_difference:+.4f}"
        )
        lines.append(
            f"  sign test p={pair.sign_test_p:.4f}; "
            f"Wilcoxon p={'not applicable' if pair.wilcoxon_p is None else f'{pair.wilcoxon_p:.4f}'}"
        )
    return "\n".join(lines)


def format_errors(analysis: Mapping[str, Any]) -> str:
    """Render the error analysis, including what was looked for and not found."""
    lines = ["", "=== non-discriminating queries (reported separately: these are not errors) ==="]
    lines.append(
        f"{analysis['n_easy_queries']} of {analysis['n_queries']} queries "
        f"({analysis['easy_fraction']:.1%}) were put in the top 3 by every arm examined "
        f"({', '.join(analysis['arms_examined'])}). A high mean built largely out of "
        "these queries measures the dataset, not the retrievers: a system that "
        "returned results in a fixed order would score well on them."
    )
    for name, data in analysis["non_discriminating"].items():
        lines.append(f"  {name}: {data['n']} queries -- {data['description']}")
        for example in data["examples"]:
            lines.append(f"    {example['query_id']} [{example['stratum']}] {example['query']!r}")
    if analysis["n_unclassified"]:
        lines.append(
            f"  {analysis['n_unclassified']} queries match no error and no easy category: "
            f"{analysis['unclassified_query_ids']}"
        )

    lines.append("")
    lines.append("=== error analysis ===")
    lines.append(
        "A query can appear in several rows: the categories are independent statements "
        "about different retrievers, and their overlap is what makes them informative. "
        f"Categories looked for but with no members are omitted: "
        f"{analysis['categories_suppressed'] or 'none'}."
    )
    for name, data in analysis["categories"].items():
        lines.append("")
        lines.append(
            f"  {name} ({data['n']} of {analysis['n_queries']} queries) - {data['description']}"
        )
        for example in data["examples"]:
            ranks = ", ".join(f"{arm}={rank}" for arm, rank in example["gold_rank"].items())
            gold = "; ".join(example["gold_titles"])
            lines.append(
                f"    {example['query_id']} [{example['stratum']}] "
                f"ranks({ranks}) gold={gold!r} query={example['query']!r}"
            )
    return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class ReportInputs:
    """Everything the report needs, gathered once."""

    aggregate: dict[str, dict[str, dict[str, Any]]]
    pairs: list[PairedComparison]
    audit: dict[str, Any]
    net: dict[str, Any]
    errors: dict[str, Any]
    dataset_summary: dict[str, Any]
    runtime: dict[str, Any]
    ceiling: dict[str, Any]
    pool: list[dict[str, Any]]


def build_report(inputs: ReportInputs, *, systems: Sequence[str], labels: Mapping[str, str]) -> str:
    """The human-readable report. Every section is a measured number or a refusal."""
    parts: list[str] = [
        "RETRIEVAL QUALITY EVALUATION",
        "",
        "Every interval below is a bootstrap over per-query values. It quantifies",
        "query-sampling noise at this n and nothing else. Corpus choice,",
        "query-author bias, single-judge bias, and pooling incompleteness are all",
        "larger and are not quantified anywhere in this report.",
        "",
        f"queries: {inputs.dataset_summary['n_queries']}   "
        f"judgments: {inputs.dataset_summary['n_judgments']}   "
        f"single-judge: {inputs.dataset_summary['judges']}",
        f"dataset freeze verified: {inputs.dataset_summary['integrity']['verified']}",
    ]
    if not inputs.dataset_summary["integrity"]["verified"]:
        parts.append("  WARNING: " + inputs.dataset_summary["integrity"]["reason"])
    parts.append(format_table(inputs.aggregate, systems=systems, label=labels))
    parts.append(format_ceiling(inputs.ceiling))
    parts.append(format_comparisons(inputs.pairs))
    parts.append(format_errors(inputs.errors))
    parts.append(format_pool(inputs.pool))

    parts.append("")
    parts.append("=== query processing: equivalence audit ===")
    for arm, data in inputs.audit["arms"].items():
        parts.append(
            f"  {arm} vs {inputs.audit['control']}: "
            f"string identical on {data['string_identical']}/{data['n_queries']} queries; "
            f"top-10 ranking identical on {data['ranking_identical']}/{data['n_queries']}; "
            f"queries modified: {data['queries_modified']}"
        )
        for rewrite in data["rewrites"]:
            parts.append(f"    rewrite: {rewrite}")
    parts.append("")
    parts.append("=== spell arm net effect ===")
    net = inputs.net
    parts.append(
        f"  gained {len(net['gained'])}, lost {len(net['lost'])}, net {net['net']} "
        f"over {net['n_queries']} queries"
    )
    for entry in net["gained"] + net["lost"]:
        parts.append(
            f"    {entry['query_id']} {entry['query']!r} "
            f"gold rank {entry['gold_rank_control']} -> {entry['gold_rank_arm']} "
            f"rewrites={entry['rewrites']}"
        )

    parts.append("")
    parts.append("=== evaluation runtime (NOT a latency measurement) ===")
    for name, seconds in inputs.runtime.get("per_system", {}).items():
        parts.append(f"  {name}: {seconds:.2f}s")
    parts.append(f"  total: {inputs.runtime.get('total_seconds', 0.0):.2f}s")
    parts.append(
        "  Arms ran sequentially and single-threaded with no warm/cold control. These "
        "numbers say how long the evaluation took; they say nothing about search "
        "latency and are not comparable with the Phase 6-9 benchmark."
    )
    parts.append(format_not_claimed())
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# The stratum-A ceiling, measured rather than argued
# --------------------------------------------------------------------------- #
#: Subset sizes for the corpus-shrinkage check, as a fraction of the corpus.
#: ``None`` is the full corpus and is reported as the reference point.
SHRINKAGE_FRACTIONS: tuple[float, ...] = (0.5, 0.25)


def title_token_overlap(query: str, title: str, tokenizer: Tokenizer) -> str:
    """How much of the title's vocabulary the query reproduces: exact/partial/none.

    Computed from the corpus and the query text only. No retriever output is
    involved, which is the whole point: a label derived from a ranking would be
    the retriever grading its own homework, and this one is fixed before any
    system runs.
    """
    query_terms = set(tokenizer.unique_tokens(query))
    title_terms = set(tokenizer.unique_tokens(title))
    if not title_terms:
        return "none"
    if not query_terms:
        return "none"
    if query_terms == title_terms:
        return "exact"
    if query_terms & title_terms:
        return "partial"
    return "none"


def _by_overlap(
    results: Sequence[PerQueryResult], labels: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """P@1 and MRR grouped by the blind title-overlap label.

    Grouping *after* the run, by a label fixed before it, is what turns "stratum
    A is really just title matching" from an objection into a number. The groups
    are disjoint and the denominators are printed, so a group of two cannot be
    read as a result.
    """
    buckets: dict[str, list[PerQueryResult]] = {"exact": [], "partial": [], "none": []}
    for record in results:
        label = labels.get(record.query_id)
        if label in buckets:
            buckets[label].append(record)
    out: dict[str, dict[str, Any]] = {}
    for name, rows in buckets.items():
        if not rows:
            continue
        out[name] = {
            "n": len(rows),
            "precision_at_1": round(sum(r.precision_at_1 for r in rows) / len(rows), 6),
            "reciprocal_rank": round(sum(r.reciprocal_rank for r in rows) / len(rows), 6),
        }
    return out


def _gold_found(
    bm25: BM25Searcher,
    no_boost_index: BM25Index,
    semantic: SemanticSearchService,
    dataset: EvalDataset,
    depth: int,
) -> dict[str, dict[str, int]]:
    """``{query_id: {arm: 1-based gold rank, 0 if missed}}`` for the ablation arms.

    Written against the ablation arms only, not :func:`error_analysis`, because
    ``title_boost=0`` makes gold documents routinely unfindable. Reusing the
    top-10 records would silently drop exactly the queries the ablation is
    about.
    """
    found: dict[str, dict[str, int]] = {}
    for query in dataset.queries:
        gold = set(dataset.positive_doc_ids(query.query_id))
        row: dict[str, int] = {}
        for name, hits in (
            ("bm25_shipped", bm25.search(query.query, top_k=depth)),
            ("bm25_no_title_boost", _search_rows(no_boost_index, query.query, depth)),
            ("semantic", semantic.search(query.query, top_k=depth)),
        ):
            ranked = tuple(hits_to_ranked_ids(hits))
            row[name] = next(
                (position for position, doc_id in enumerate(ranked, start=1) if doc_id in gold),
                0,
            )
        found[query.query_id] = row
    return found


def _search_rows(index: BM25Index, query: str, top_k: int) -> list[BM25Result]:
    """``index.search`` returning only the ids, for the hand-rolled ablations."""
    return [BM25Result(id_, 0, 0.0, 0, (), {}) for id_ in _bm25_ids(index, query, top_k)]


def _bm25_ids(index: BM25Index, query: str, top_k: int) -> list[str]:
    return _bm25_score(index, query, top_k, saturate=True)


def _bm25_ids_binary_tf(index: BM25Index, query: str, top_k: int) -> list[str]:
    """Same scorer, but every term counts once regardless of how often it occurs."""
    return _bm25_score(index, query, top_k, saturate=False)


def _bm25_score(index: BM25Index, query: str, top_k: int, *, saturate: bool) -> list[str]:
    """Re-run BM25 scoring directly, for arms that need a variant of the index.

    This calls the *index's own* tokenizer, idf, and length normalisation, so
    the only thing that varies between the shipped arm and an ablation is the
    one knob under test. Re-implementing the scoring here would be a second
    BM25, and a difference in its output would be read as a finding about
    ``title_boost`` when it is a bug in the copy.

    ``saturate=False`` is the most-frequent-term baseline: with ``k1 = 0`` the
    standard BM25 term contribution collapses to a constant, so the same code
    path with the saturation switched off is the baseline, rather than a
    separately written loop that could disagree with this one for its own
    reasons.

    Ties break on row, exactly as ``BM25Index.search`` does, so the shipped
    arm's ordering is reproducible here too.
    """
    if not query or not query.strip():
        return []
    query_terms = index.tokenizer.tokenize(query)
    if not query_terms:
        return []
    k1 = index.config.k1 if saturate else 0.0
    b = index.config.b
    scored: list[tuple[float, int]] = []
    for row in range(len(index.doc_ids)):
        tf = index.term_frequencies[row]
        length = index.doc_lengths[row]
        norm = 1.0 - b + b * (length / index.average_length if index.average_length else 1.0)
        total = 0.0
        for term in query_terms:
            frequency = tf.get(term, 0)
            if not frequency:
                continue
            total += index.idf(term) * (frequency * (k1 + 1)) / (frequency + k1 * norm)
        if total > 0.0:
            scored.append((total, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index.doc_ids[row] for _, row in scored[:top_k]]


def _subset_store(
    store: DocumentStore, doc_ids: Sequence[str]
) -> tuple[DocumentStore, dict[str, int]]:
    """A store over a subset, plus the dropped gold documents per query id.

    The drop map is what makes the shrinkage check honest. If a subset loses a
    gold document the query is unanswerable there, its score becomes a
    measurement of the sampling rather than of retrieval, and averaging it in
    silently would make the corpus look like it degrades when it may only have
    dropped the answer. Subsets are **not** re-rolled until they keep the gold.
    """
    keep = set(doc_ids)
    kept = [document for document in store if document.id in keep]
    dropped: dict[str, int] = {}
    for document in store:
        if document.id not in keep:
            dropped[document.id] = 0
    return DocumentStore(kept), dropped


def ceiling_measurements(
    dataset: EvalDataset,
    results: Sequence[PerQueryResult],
    store: DocumentStore,
    *,
    bm25: BM25Searcher,
    semantic: SemanticSearchService,
    tokenizer: Tokenizer,
    depth: int = 50,
    seed: int = 20250930,
) -> dict[str, Any]:
    """Convert "stratum A is a title-matching ceiling" from a caveat to a number.

    Six measurements, all cheap, all reported whether or not they help:

    1. **Title overlap, blind.** Each stratum-A query is labelled ``exact`` /
       ``partial`` / ``none`` against its source article's title using the
       tokenizer only. P@1 and MRR are then reported per label. If the
       ``none`` group scores like the ``exact`` group, the contamination
       argument is wrong and the report says so.
    2. **``title_boost=0`` ablation.** The shipped BM25 index appends one extra
       copy of every title token to every document, and
       ``scripts/build_index.py`` never overrides it. Rebuilding with the boost
       off and re-running the same queries is a direct measurement of how much
       of BM25's stratum-A performance is title matching.
    3. **Corpus shrinkage.** Re-run BM25 over 50% and 25% seeded subsets. If
       halving the corpus does not degrade P@1, the metric is not measuring
       discrimination at all -- which is a finding about the metric.
    4. **Random baseline.** Rank by a seeded shuffle. The floor every real
       system must beat, and the reason a high P@1 is not impressive on 300
       documents without one.
    5. **Most-frequent-term baseline.** Rank by summed IDF over the query's
       terms, ignoring the term's position in any document. A real retriever's
       floor that a random shuffle does not reach.
    6. **All-arms-agree-at-rank-1.** Counted, not inferred; it is also the
       non-discriminating set in :func:`error_analysis`.

    Ablation arms are computed here rather than taken from ``results``, because
    the ablations are not among the six registered arms and putting them in the
    main table would present them as comparable systems.
    """
    overlap_labels = {
        query.query_id: title_token_overlap(
            query.query, _source_title(store, query.source_doc_id), tokenizer
        )
        for query in dataset.queries
        if query.source_doc_id
    }

    no_boost_index = BM25Index.build(list(store), title_boost=0)
    found = _gold_found(bm25, no_boost_index, semantic, dataset, depth)

    # --- shrinkage ------------------------------------------------------- #
    rng = random.Random(seed)
    ordered_ids = sorted(store.document_ids)
    shrinkage: dict[str, Any] = {}
    for fraction in SHRINKAGE_FRACTIONS:
        size = max(1, int(len(ordered_ids) * fraction))
        subset_ids = sorted(rng.sample(ordered_ids, size))
        subset_store, _ = _subset_store(store, subset_ids)
        # Built once per subset, not once per query: BM25Index.build tokenizes
        # every document in the subset, and doing that 80 times would make this
        # check take minutes for no additional information.
        subset_index = BM25Index.build(list(subset_store))
        kept_gold: dict[str, int] = {}
        scores: list[float] = []
        ranks: list[int] = []
        for query in dataset.queries:
            gold = set(dataset.positive_doc_ids(query.query_id))
            if not gold or not gold.issubset(set(subset_ids)):
                # Recorded, not skipped-and-forgotten: the count is the result.
                kept_gold[query.query_id] = 0
                continue
            kept_gold[query.query_id] = 1
            ranked = _bm25_ids(subset_index, query.query, depth)
            position = next((i for i, doc_id in enumerate(ranked, start=1) if doc_id in gold), 0)
            ranks.append(position)
            scores.append(1.0 if position == 1 else 0.0)
        shrinkage[f"{fraction:.0%}"] = {
            "n_documents": size,
            "n_queries_scored": len(scores),
            "n_queries_dropped_gold": sum(1 for v in kept_gold.values() if v == 0),
            "dropped_query_ids": sorted(q for q, v in kept_gold.items() if v == 0),
            "precision_at_1": round(sum(scores) / len(scores), 6) if scores else 0.0,
            "mean_gold_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
        }

    # --- baselines ------------------------------------------------------- #
    corpus_size = len(store)
    random_p1 = _random_baseline_p1(dataset, store, seed)
    mf_p1, mf_mrr = _most_frequent_baseline(dataset, bm25.index, depth)

    agree_at_1 = sum(
        1 for row in found.values() if all(rank == 1 for rank in row.values() if rank >= 0)
    )

    return {
        "stratum_a_title_overlap": {
            "method": "tokenizer-only overlap between the query and its source "
            "article's title; computed from the corpus, never from a ranking",
            "labels_per_stratum": {
                stratum: _label_counts(
                    [
                        overlap_labels[q.query_id]
                        for q in dataset.by_stratum(stratum)
                        if q.query_id in overlap_labels
                    ]
                )
                for stratum in sorted({q.stratum for q in dataset.queries})
            },
            "by_system": {
                system: {
                    stratum: _by_overlap(
                        [r for r in results if r.stratum == stratum and r.system == system],
                        overlap_labels,
                    )
                    for stratum in sorted({r.stratum for r in results})
                    if any(r.stratum == stratum and r.system == system for r in results)
                }
                for system in sorted({r.system for r in results})
            },
        },
        "title_boost_ablation": {
            "shipped_title_boost": 1,
            "ablated_title_boost": 0,
            "note": "the shipped index double-counts title terms: BM25Index.build "
            "appends tokenizer.tokenize(title) once and scripts/build_index.py "
            "never overrides the default",
            "precision_at_1": {
                "bm25_shipped": _mean_rank_stat(found, "bm25_shipped", "p1"),
                "bm25_no_title_boost": _mean_rank_stat(found, "bm25_no_title_boost", "p1"),
            },
            "reciprocal_rank": {
                "bm25_shipped": _mean_rank_stat(found, "bm25_shipped", "mrr"),
                "bm25_no_title_boost": _mean_rank_stat(found, "bm25_no_title_boost", "mrr"),
            },
            "found_at_all": {
                "bm25_shipped": sum(1 for q in found if found[q]["bm25_shipped"] > 0),
                "bm25_no_title_boost": sum(1 for q in found if found[q]["bm25_no_title_boost"] > 0),
            },
        },
        "corpus_shrinkage": {
            "note": "seeded subsets; a subset that drops a gold document makes "
            "that query unanswerable, and the drop is reported rather than "
            "re-rolled",
            "seed": seed,
            "full_corpus": {"n_documents": corpus_size},
            "subsets": shrinkage,
        },
        "baselines": {
            "random_rank_1": {
                "precision_at_1": round(random_p1, 6),
                "note": f"expected P@1 for a fixed order over {corpus_size} documents "
                "is about 1/N; the measured value is a seeded single run, not that formula",
            },
            "most_frequent_term_bm25": {
                "precision_at_1": round(mf_p1, 6),
                "reciprocal_rank": round(mf_mrr, 6),
                "note": "documents ranked by summed idf over the query's terms, "
                "ignoring every term's position and frequency in the document",
            },
        },
        "all_arms_agree_at_rank_1": {
            "n": agree_at_1,
            "n_queries": len(found),
            "fraction": round(agree_at_1 / len(found), 6) if found else 0.0,
            "arms": ["bm25_shipped", "bm25_no_title_boost", "semantic"],
        },
    }


def _mean_rank_stat(
    found: Mapping[str, Mapping[str, int]], arm: str, which: Literal["p1", "mrr"]
) -> float:
    """Mean P@1 or MRR from the ablation gold ranks.

    Computed over **every** query, including the ones the arm missed, which is
    what a metric over this dataset must do. Restricting the denominator to the
    queries an arm found would make the ablation look like a ranking change
    rather than a retrieval failure.
    """
    if not found:
        return 0.0
    total = 0.0
    for row in found.values():
        rank = row.get(arm, 0)
        if rank == 0:
            continue
        if which == "p1":
            total += 1.0 if rank == 1 else 0.0
        else:
            total += 1.0 / rank
    return round(total / len(found), 6)


def _label_counts(labels: Sequence[str]) -> dict[str, int]:
    return {
        name: sum(1 for label in labels if label == name) for name in ("exact", "partial", "none")
    }


def _source_title(store: DocumentStore, doc_id: str) -> str:
    document = store.get(doc_id) if doc_id else None
    return document.title if document is not None else ""


def _random_baseline_p1(dataset: EvalDataset, store: DocumentStore, seed: int) -> float:
    """P@1 of one seeded random ordering, as a single measured run.

    The expected value of P@1 for a uniformly random order is the number of
    relevant documents over the corpus size. Reporting that formula would be
    quoting arithmetic rather than running a baseline, and a reader is right to
    be suspicious of a "baseline" that is a closed-form expression. This is one
    seeded draw, so it is itself noisy at n=80; the closed form is given
    alongside it in the report so the two can be compared.
    """
    rng = random.Random(seed)
    ordered = list(store.document_ids)
    hits = 0
    scored = 0
    for query in dataset.queries:
        gold = set(dataset.positive_doc_ids(query.query_id))
        if not gold:
            continue
        scored += 1
        rng.shuffle(ordered)
        if ordered[0] in gold:
            hits += 1
    return hits / scored if scored else 0.0


def _most_frequent_baseline(
    dataset: EvalDataset, index: BM25Index, depth: int
) -> tuple[float, float]:
    """BM25 with every term's in-document frequency forced to 1.

    A genuine retriever beats this by using the evidence that a term appears
    five times rather than once. This is the same scorer the system uses, so the
    gap between the two is attributable to term frequency and to nothing else
    -- which is the only reason it is a useful baseline rather than a second
    implementation with a second set of bugs.
    """
    scored_queries = 0
    p1 = 0.0
    mrr = 0.0
    for query in dataset.queries:
        gold = set(dataset.positive_doc_ids(query.query_id))
        if not gold:
            continue
        scored_queries += 1
        ranked = _bm25_ids_binary_tf(index, query.query, depth)
        position = next((i for i, doc_id in enumerate(ranked, start=1) if doc_id in gold), 0)
        if position == 1:
            p1 += 1.0
        if position > 0:
            mrr += 1.0 / position
    if scored_queries == 0:
        return 0.0, 0.0
    return p1 / scored_queries, mrr / scored_queries


def format_ceiling(ceiling: Mapping[str, Any]) -> str:
    """Render the ceiling measurements, including the ones that came out flat."""
    lines = ["", "=== how much of this is title matching? (the stratum A ceiling) ==="]
    lines.append(
        "Strata A_restated and A_verbatim were written from article titles and the "
        "shipped indexes weight titles twice (BM25 appends a title copy; the encoder "
        "puts the title first in the text it embeds). The measurements below are the "
        "size of that effect, not a defence of it."
    )

    overlap = ceiling["stratum_a_title_overlap"]
    lines.append("")
    lines.append("  title-token overlap between query and source title, per stratum:")
    for stratum, counts in overlap["labels_per_stratum"].items():
        lines.append(
            f"    {stratum:<14} exact={counts['exact']:>3} "
            f"partial={counts['partial']:>3} none={counts['none']:>3}"
        )
    lines.append("  P@1 / MRR by that blind label (BM25):")
    for stratum, groups in overlap["by_system"].get("bm25", {}).items():
        rendered = ", ".join(
            f"{name} n={g['n']} P@1={g['precision_at_1']:.3f} MRR={g['reciprocal_rank']:.3f}"
            for name, g in groups.items()
        )
        lines.append(f"    {stratum:<14} {rendered}")

    ablation = ceiling["title_boost_ablation"]
    lines.append("")
    lines.append("  BM25 with and without the title boost (the same queries, the same labels):")
    for metric, values in (
        ("P@1", ablation["precision_at_1"]),
        ("MRR", ablation["reciprocal_rank"]),
    ):
        shipped = values["bm25_shipped"]
        ablated = values["bm25_no_title_boost"]
        lines.append(
            f"    {metric:<4} shipped={shipped:.3f}  title_boost=0 {ablated:.3f}  "
            f"delta={shipped - ablated:+.3f}"
        )
    found_shipped = ablation["found_at_all"]["bm25_shipped"]
    found_ablated = ablation["found_at_all"]["bm25_no_title_boost"]
    lines.append(
        f"    gold document retrieved at all: shipped {found_shipped}, "
        f"ablated {found_ablated} (of {len(ceiling['all_arms_agree_at_rank_1']) and ceiling['all_arms_agree_at_rank_1']['n_queries']} queries)"
    )

    lines.append("")
    lines.append("  BM25 against a corpus of half the size, and a quarter:")
    shrinkage = ceiling["corpus_shrinkage"]
    lines.append(f"    full corpus: {shrinkage['full_corpus']['n_documents']} documents")
    for name, data in shrinkage["subsets"].items():
        lines.append(
            f"    {name:<5} n_docs={data['n_documents']:>3} "
            f"n_queries_scored={data['n_queries_scored']:>3} "
            f"dropped_gold={data['n_queries_dropped_gold']:>3} "
            f"P@1={data['precision_at_1']:.3f} mean_gold_rank={data['mean_gold_rank']:.2f}"
        )
    if any(data["n_queries_dropped_gold"] for data in shrinkage["subsets"].values()):
        lines.append(
            "    A non-zero dropped_gold means those queries were unanswerable in that "
            "subset and are excluded from its P@1. That is reported rather than "
            "re-rolled, so this check is partly a test of the sampling too."
        )

    baselines = ceiling["baselines"]
    lines.append("")
    lines.append("  what a system has to beat:")
    lines.append(f"    random fixed order   P@1={baselines['random_rank_1']['precision_at_1']:.3f}")
    lines.append(
        f"    most-frequent-term    P@1={baselines['most_frequent_term_bm25']['precision_at_1']:.3f} "
        f"MRR={baselines['most_frequent_term_bm25']['reciprocal_rank']:.3f}"
    )

    agree = ceiling["all_arms_agree_at_rank_1"]
    lines.append("")
    lines.append(
        f"  queries where all of {', '.join(agree['arms'])} put the gold document "
        f"first: {agree['n']} of {agree['n_queries']} ({agree['fraction']:.1%})"
    )
    return "\n".join(lines)


def format_pool(reports: Sequence[Mapping[str, Any]]) -> str:
    """State the size of the hole under every number computed from pooling.

    Reported per stratum, because a pooled-judgment stratum and a
    known-item stratum do not have the same exposure: stratum A's relevance is
    a property of how the query was built, not a judgement made about a
    candidate list, so its "unseen" figure means something different from
    stratum B's. Both are printed, with the fraction, rather than a reassuring
    round zero.
    """
    if not reports:
        return "\n".join(["", "=== pooling coverage ===", "  no pool report was produced."])
    lines = ["", "=== pooling coverage ==="]
    lines.append(
        "Documents no judge saw are scored as not relevant, so every recall figure "
        "sits on top of unseen material. This is its size. On a 300-document corpus "
        "with six arms pooling to depth 50, the pool is most of the corpus -- which "
        "is a reason the numbers are hard to move, not a reason they are good."
    )
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        by_stratum.setdefault(str(report.get("stratum", "?")), []).append(report)
    for stratum, rows in sorted(by_stratum.items()):
        mean_pool = sum(r["pool_size"] for r in rows) / len(rows)
        mean_unseen = sum(r["unseen_fraction"] for r in rows) / len(rows)
        lines.append(
            f"  {stratum:<14} n={len(rows):>3}  mean pool={mean_pool:6.1f} docs  "
            f"mean fraction of the corpus never pooled={mean_unseen:.3f}"
        )
    return "\n".join(lines)


def not_claimed() -> list[str]:
    """The 'not claimed' tier, as data rather than prose.

    Kept as a list so it is emitted by code on every run, rather than by
    whoever writes the summary remembering to include it. A limitation that only
    appears when someone thinks to write it is a limitation that eventually does
    not appear.
    """
    return [
        "No system is described as best, and no combined score is reported.",
        "The four strata are not pooled into a single mean.",
        "Stratum A is a known-item ceiling on a 300-article corpus, not a measure "
        "of task-independent retrieval quality.",
        "Every judgment is one person's. There is no inter-annotator agreement.",
        "Grading is binary. NDCG and MAP are the binary case of general formulas.",
        "Pooling at depth 50 is incomplete; unjudged documents score as not relevant.",
        "The bootstrap covers query-sampling noise only.",
        "Nothing here generalises to another corpus, encoder, corpus size, or "
        "production traffic.",
        "No latency claim is made; evaluation runtime is not search latency.",
        "Synonym expansion was off in every arm and is unevaluated.",
        "A difference whose interval contains zero is inconclusive at this n, not "
        "evidence that the two systems are equivalent.",
        "The title-boost ablation and the corpus-shrinkage check are diagnostic "
        "measurements, not candidate configurations; neither is a recommendation.",
        "The application search service is not evaluated as a system, because it "
        "re-ranks nothing and would be identical to Hybrid by construction.",
        "No arm was tuned on this dataset, and none could be: no parameter is "
        "fitted to the queries, and no query was revised after seeing a ranking.",
    ]


def format_not_claimed() -> str:
    """The closing section, emitted by code on every run."""
    lines = ["", "=== NOT CLAIMED (read this before quoting any number above) ==="]
    lines.extend(f"  - {item}" for item in not_claimed())
    return "\n".join(lines)


def intervals_to_dicts(intervals: Iterable[Interval]) -> list[dict[str, Any]]:
    return [interval.to_dict() for interval in intervals]
