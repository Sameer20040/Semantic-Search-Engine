"""Run the retrieval-quality evaluation over the frozen dataset.

This is the reproducible entry point for Phase 10. Given the same dataset, the
same indexes, and the same seed, it produces the same artifacts and the same
numbers -- the arms are deterministic (BM25 ties break on row, the FAISS index
is a flat inner-product index, the fusion is a stable sort), so a re-run is a
check rather than a coin flip.

What it does NOT do:

* change any default. ``query.spell_check`` and ``query.synonym_expansion``
  stay as they are loaded; the spell arm runs on a *copy* of the settings with
  the flag forced on, and prints which arms ran. Enabling it here does not
  enable it for the next process.
* report a winner, or a combined score.
* treat evaluation runtime as search latency. The per-arm seconds are printed
  as runtime and labelled as such.
* hide the ceiling. The title-boost ablation, the corpus-shrinkage check, the
  random baseline and the all-arms-agree count are all in the output, whether
  or not they flatter the system.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --systems bm25 semantic --depth 50
    python scripts/evaluate.py --run-spell-experiment
    python scripts/evaluate.py --resamples 2000 --seed 7

Artifacts, all written to ``data/evaluation/derived/`` unless ``--report-dir``
says otherwise:

``per_query_results.jsonl``
    One row per (system, query) with the ranking, the metrics, the gold
    document's rank, and the string the arm actually retrieved with. Every
    number in the report is recomputable from this file.
``aggregate.json``
    Per-stratum, per-system metric means with bootstrap intervals.
``bootstrap.json``
    The paired comparisons, the ceiling measurements, the error analysis, the
    equivalence audit, and the pooling coverage.
``errors.md``
    The human-readable report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.config import load_settings
from semantic_search.core.exceptions import SemanticSearchError
from semantic_search.core.logging import configure_logging, get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.data.tokenizer import get_tokenizer
from semantic_search.evaluation.dataset import (
    DatasetPaths,
    load_dataset,
    pool_report,
    summarise,
)
from semantic_search.evaluation.models import STRATA
from semantic_search.evaluation.report import (
    METRICS,
    ReportInputs,
    aggregate_table,
    build_report,
    ceiling_measurements,
    comparisons,
    equivalence_audit,
    error_analysis,
    net_effect,
    not_claimed,
)
from semantic_search.evaluation.runner import (
    SYSTEM_LABELS,
    SYSTEMS,
    EvaluationRunner,
    default_report_dir,
    write_json,
    write_per_query,
)
from semantic_search.evaluation.stats import DEFAULT_SEEDS

logger = get_logger("evaluate")

#: Arms that are part of the shipped configuration. Run in this order unless
#: ``--systems`` says otherwise, and printed in it, so a reader comparing two
#: reports is looking at the same rows in the same places.
DEFAULT_SYSTEMS: tuple[str, ...] = SYSTEMS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the frozen labeled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset root containing queries/ and qrels/ (default: data/evaluation)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(SYSTEMS),
        default=list(DEFAULT_SYSTEMS),
        help="arms to run, in the order they are printed",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="retrieval depth per (arm, query); every cutoff derives from one run",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=None,
        help="bootstrap resamples per interval (10,000 is the default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the comparison bootstrap and the shrinkage subsets",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        choices=list(STRATA),
        default=None,
        help=(
            "restrict the run to these query strata. The default is every stratum "
            "with a positive judgment. Restricting is how a partially-judged "
            "dataset stays usable: the stratum-A queries have complete, "
            "genuinely-produced known-item judgments, so they can be evaluated "
            "on their own and reported as a stratum-A result. The stratum-B "
            "queries are excluded until their judgments exist -- see "
            "data/evaluation/qrels/MANIFEST.md."
        ),
    )
    parser.add_argument(
        "--pool-only",
        action="store_true",
        help=(
            "build the candidate pool WITHOUT scoring it, and write it to "
            "<report-dir>/pool.jsonl. This is the disclosed pool-building run "
            "the stratum-B judging protocol requires: it produces the candidate "
            "list the judgments are made from, and it deliberately produces no "
            "metrics, because a number printed next to a pool someone is about "
            "to judge is an invitation to judge toward it."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="where to write artifacts (default: <dataset>/derived)",
    )
    parser.add_argument(
        "--run-spell-experiment",
        action="store_true",
        default=None,
        help=(
            "include the spell_check=True arm. Runs on a COPY of the settings; "
            "it does not change query.spell_check for any other process."
        ),
    )
    parser.add_argument(
        "--no-ceiling",
        action="store_true",
        help="skip the title-boost ablation and shrinkage checks (faster, and the "
        "report then says nothing about how much of the result is title matching)",
    )
    return parser.parse_args(argv)


def _resolve(value, fallback, cast):
    """Command line beats environment beats the config default, in that order."""
    if value is not None:
        return value
    return cast(fallback)


def _write_pool(dataset, run, store, report_dir: Path, depth: int, systems) -> int:
    """Write the judging candidate list, and nothing else.

    Three properties this file needs and this function is responsible for:

    * **No scores.** There is not one metric in the output. A judge reading a
      candidate list that sits next to "BM25 ranked this 7th" is judging toward
      the ranking, and the blind-judging protocol in ``qrels/MANIFEST.md``
      exists to prevent exactly that.
    * **No system attribution.** The rows are shuffled per query with a fixed
      seed and carry no field saying which arm retrieved what. The judge sees
      documents, not systems.
    * **Coverage is visible.** A query whose source article is in no arm's
      ranking is flagged, and the flag is a *finding* -- per the protocol it is
      recorded, not fixed by inserting the article by hand.
    """
    rng = random.Random(20250930)
    pool_path = report_dir / "pool.jsonl"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    uncovered: list[str] = []
    with pool_path.open("w", encoding="utf-8") as handle:
        for query in dataset.queries:
            candidates = list(run.pool.get(query.query_id, ()))
            titles = []
            for doc_id in candidates:
                document = store.get(doc_id)
                titles.append(
                    {
                        "doc_id": doc_id,
                        "title": document.title if document is not None else doc_id,
                    }
                )
            source_in_pool = query.source_doc_id in candidates
            if query.source_doc_id and not source_in_pool:
                uncovered.append(query.query_id)
            shuffled = list(titles)
            rng.shuffle(shuffled)
            handle.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "stratum": query.stratum,
                        "query": query.query,
                        # Recorded so the judge can check the pool contains the
                        # answer. NOT a relevance judgment, and never treated as
                        # one: the answer text is a paraphrase of the lead.
                        "recorded_answer": query.notes,
                        "source_doc_id": query.source_doc_id,
                        "source_in_pool": source_in_pool,
                        "n_candidates": len(candidates),
                        "candidates": shuffled,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print()
    print("=" * 78)
    print("POOL-BUILDING RUN -- no metrics were computed and none are in the output")
    print("=" * 78)
    print(f"pool      {pool_path}")
    print(f"depth     {depth}, arms pooled: {', '.join(systems)}")
    print(
        f"rows      {len(dataset.queries)} queries; a row's candidates are shuffled "
        "with a fixed seed and carry no system attribution"
    )
    if uncovered:
        print(
            f"\nCOVERAGE FAILURE: {len(uncovered)} queries' source article appears in no "
            f"arm's top-{depth}: {uncovered}"
        )
        print(
            "  Per the protocol this is recorded, not repaired. Hand-adding the "
            "article would make the query unanswerable for every system by "
            "construction and would hide the finding."
        )
    else:
        print(
            "\ncoverage: every query's source article appears in at least one arm's "
            "top-50. That does not make the judgments redundant -- relevance is "
            "still judged by whether the document contains the answer."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    settings = load_settings()
    evaluation = settings.evaluation
    depth = _resolve(args.depth, evaluation.depth, int)
    resamples = _resolve(args.resamples, evaluation.resamples, int)
    seed = _resolve(args.seed, evaluation.seed, int)
    run_spell = (
        evaluation.run_spell_experiment
        if args.run_spell_experiment is None
        else args.run_spell_experiment
    )
    # The flag can only ever add the experiment. It can never turn a default
    # off, and running this script must not change what the next process loads.
    if not run_spell:
        args.systems = [s for s in args.systems if s != "hybrid_qp_spell_experiment"]

    paths = DatasetPaths(root=args.dataset or settings.data.evaluation_dir)
    report_dir = args.report_dir or default_report_dir(paths)
    print("=" * 78)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 78)
    print(f"dataset   {paths.root}")
    print(f"corpus    {settings.data.corpus_file}")
    print(f"depth     {depth}    resamples {resamples}    seed {seed}")
    print(f"arms      {', '.join(args.systems)}")
    print(
        f"settings  query.spell_check={settings.query.spell_check} "
        f"query.synonym_expansion={settings.query.synonym_expansion} "
        "(unchanged by this run)"
    )
    if run_spell:
        print(
            "           the spell arm is an EXPERIMENT on a copy of these settings; "
            "it is not a default and nothing is persisted"
        )

    if not paths.queries.exists():
        print(
            f"\nerror: no queries at {paths.queries}\n"
            "The dataset is committed to the repository. Build it first:\n"
            "    python data/evaluation/_build_queries_a.py\n"
            "    python data/evaluation/_build_queries_b.py\n"
            "    python data/evaluation/_build_qrels_a.py",
            file=sys.stderr,
        )
        return 1

    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    print(f"corpus    loaded {len(store)} documents\n")

    started = time.perf_counter()
    try:
        dataset, integrity = load_dataset(
            paths,
            corpus_doc_ids=store.document_ids,
            # The pool-building run is the one legitimate case for loading a
            # dataset whose judgments are not written yet -- the pool IS the
            # input to the judging. Every other run demands them, so a
            # half-judged dataset can never produce a report.
            require_judgments=not args.pool_only,
            strata=args.strata,
        )
    except SemanticSearchError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    summary = summarise(dataset, integrity)
    if not integrity.verified:
        # Loud, and before any number is printed. A digest mismatch means the
        # queries changed after the freeze, and every figure below would then
        # describe a different experiment than the one the manifest records.
        print("=" * 78)
        print("DATASET INTEGRITY WARNING")
        print("=" * 78)
        print(f"  {integrity.reason}")
        print(f"  expected sha256 {integrity.expected}")
        print(f"  actual   sha256 {integrity.actual}")
        print("  The numbers below describe the CURRENT file, whatever it contains.\n")

    print(
        f"dataset   {summary['n_queries']} queries, {summary['n_judgments']} judgments, "
        f"binary_only={summary['binary_only']}, judge(s)={summary['judges'] or ['?']}"
    )
    print(f"strata    {summary['queries_per_stratum']}")
    print(f"freeze    verified={integrity.verified}\n")

    try:
        runner = EvaluationRunner.from_settings(settings, depth=depth, spell_check=run_spell)
        run = runner.run(dataset, systems=args.systems)
    except SemanticSearchError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    load_seconds = time.perf_counter() - started

    if args.pool_only:
        return _write_pool(dataset, run, store, report_dir, depth, args.systems)

    aggregate = aggregate_table(run.per_query, resamples=resamples, seeds=DEFAULT_SEEDS[:1])
    pairs = [
        pair
        for stratum in sorted({r.stratum for r in run.per_query})
        for pair in comparisons(
            run.per_query,
            stratum=stratum,
            systems=args.systems,
            resamples=resamples,
            seed=seed,
        )
    ]
    audit = equivalence_audit(run.per_query)
    spell_arm = "hybrid_qp_spell_experiment"
    if spell_arm in args.systems:
        net = net_effect(run.per_query, arm=spell_arm, control="hybrid_qp_on")
    else:
        net = {
            "arm": spell_arm,
            "control": "hybrid_qp_on",
            "n_queries": 0,
            "gained": [],
            "lost": [],
            "net": 0,
        }
    errors = error_analysis(run.per_query)

    pool = [report.to_dict() for report in pool_report(dataset, run.pool, depth)]

    ceiling: dict[str, object] = {}
    if not args.no_ceiling:
        print("measuring the stratum-A ceiling (title boost, shrinkage, baselines)...")
        ceiling = ceiling_measurements(
            dataset,
            run.per_query,
            store,
            bm25=runner.bm25,
            semantic=runner.semantic,
            tokenizer=get_tokenizer(
                stopwords=settings.bm25.stopwords,
                use_stemming=settings.bm25.use_stemming,
            ),
            depth=depth,
            seed=seed,
        )
    else:
        print("ceiling measurements SKIPPED (--no-ceiling); the report says so.\n")

    runtime = {
        "per_system": {run_.arm.system: run_.seconds for run_ in run.arms},
        "total_seconds": run.seconds,
        "load_and_score_seconds": load_seconds,
        "not_a_latency_measurement": True,
    }

    report = build_report(
        ReportInputs(
            aggregate=aggregate,
            pairs=pairs,
            audit=audit,
            net=net,
            errors=errors,
            dataset_summary=summary,
            runtime=runtime,
            ceiling=ceiling,
            pool=pool,
        ),
        systems=args.systems,
        labels=SYSTEM_LABELS,
    )
    print()
    print(report)

    # ----------------------------------------------------------------- #
    # Artifacts. Written before the summary so a crash in the pretty
    # printing still leaves the machine-readable record on disk.
    # ----------------------------------------------------------------- #
    per_query_path = write_per_query(report_dir / "per_query_results.jsonl", run.per_query)
    aggregate_path = write_json(
        report_dir / "aggregate.json",
        {
            "metrics": list(METRICS),
            "systems": list(args.systems),
            "depth": depth,
            "resamples": resamples,
            "seed": seed,
            "by_stratum": aggregate,
        },
    )
    bootstrap_path = write_json(
        report_dir / "bootstrap.json",
        {
            "pre_registered_comparisons": [
                {"system_a": a, "system_b": b}
                for a, b in [("hybrid_rrf", "bm25"), ("hybrid_rrf", "semantic")]
            ],
            "paired_comparisons": [p.to_dict() for p in pairs],
            "ceiling": ceiling,
            "error_analysis": errors,
            "equivalence_audit": audit,
            "spell_arm_net_effect": net,
            "pooling": pool,
            "not_claimed": not_claimed(),
        },
    )
    report_path = report_dir / "errors.md"
    report_path.write_text(report + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    print("artifacts")
    print("=" * 78)
    for path in (per_query_path, aggregate_path, bootstrap_path, report_path):
        print(f"  {path}")
    print()
    print(
        "No system is ranked and no combined score is reported. Read the NOT CLAIMED "
        "section before quoting any figure above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
