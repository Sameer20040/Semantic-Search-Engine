"""Retrieval-quality evaluation: a frozen dataset, real retrievers, honest numbers.

This package sits **above** every search component and depends on all of them.
Nothing here is imported by a search path, and nothing in ``search``, ``query``,
``index``, ``data``, ``embeddings``, or ``core`` imports from here. The direction
is one-way and ``tests/integration/test_app_integration.py::TestDependencyDirection``
enforces it -- evaluation is allowed to depend on the system, never the reverse,
because a retriever that could see its own scores would be a different retriever.

Entry point: ``python scripts/evaluate.py``. Everything it produces lands in
``data/evaluation/derived/``.

What this package will not do, deliberately and in code rather than in prose:

* rank the systems, or name a best one;
* combine the metrics into a single score;
* pool the four strata into one mean;
* report a difference whose interval contains zero as a null result;
* treat evaluation runtime as search latency;
* write an error-analysis category that the data does not support.
"""

from semantic_search.evaluation.dataset import (
    DatasetIntegrity,
    DatasetPaths,
    PoolReport,
    file_digest,
    load_dataset,
    pool_report,
    summarise,
    verify_digest,
)
from semantic_search.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    hits_to_ranked_ids,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semantic_search.evaluation.models import (
    STRATA,
    ArmResult,
    EvalDataset,
    EvalQuery,
    PerQueryResult,
    RelevanceJudgment,
)
from semantic_search.evaluation.report import (
    PRIMARY_COMPARISONS,
    ReportInputs,
    aggregate_table,
    build_report,
    ceiling_measurements,
    comparisons,
    equivalence_audit,
    error_analysis,
    net_effect,
    not_claimed,
    title_token_overlap,
)
from semantic_search.evaluation.runner import (
    DEFAULT_DEPTH,
    SYSTEM_LABELS,
    SYSTEMS,
    EvaluationRun,
    EvaluationRunner,
    assert_spell_arm_live,
    build_pool,
    default_report_dir,
    score_run,
    write_json,
    write_per_query,
)
from semantic_search.evaluation.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEEDS,
    Interval,
    PairedComparison,
    bootstrap_ci,
    bootstrap_ci_multi_seed,
    compare,
    paired_difference_ci,
    sign_test,
    wilcoxon_signed_rank,
)

__all__ = [
    "DEFAULT_DEPTH",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEEDS",
    "PRIMARY_COMPARISONS",
    "STRATA",
    "SYSTEMS",
    "SYSTEM_LABELS",
    "ArmResult",
    "DatasetIntegrity",
    "DatasetPaths",
    "EvalDataset",
    "EvalQuery",
    "EvaluationRun",
    "EvaluationRunner",
    "Interval",
    "PairedComparison",
    "PerQueryResult",
    "PoolReport",
    "RelevanceJudgment",
    "ReportInputs",
    "aggregate_table",
    "assert_spell_arm_live",
    "average_precision",
    "bootstrap_ci",
    "bootstrap_ci_multi_seed",
    "build_pool",
    "build_report",
    "ceiling_measurements",
    "compare",
    "comparisons",
    "dcg_at_k",
    "default_report_dir",
    "equivalence_audit",
    "error_analysis",
    "file_digest",
    "hits_to_ranked_ids",
    "load_dataset",
    "ndcg_at_k",
    "net_effect",
    "not_claimed",
    "paired_difference_ci",
    "pool_report",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_run",
    "sign_test",
    "summarise",
    "title_token_overlap",
    "verify_digest",
    "wilcoxon_signed_rank",
    "write_json",
    "write_per_query",
]
