"""Wiring the frozen dataset to the real retrievers and back to artifacts.

What runs here
--------------
Six arms, all against the **real** pipeline -- the real BM25 index, the real
FAISS index, the real encoder, the real fused service. There is no mock
retriever anywhere in this module, and there is a test that asserts the final
evaluation used none: an evaluation run against a fake encoder measures the fake
encoder.

==========================================  ===============================
Arm                                        What it is
==========================================  ===============================
``bm25``                                    lexical only, no processing
``semantic``                                dense only, no processing
``hybrid_rrf``                              the shipped default
``hybrid_qp_off``                           processing bypassed entirely
``hybrid_qp_on``                            processing on, both stages off
``hybrid_qp_spell_experiment``              EXPERIMENT, not a default
==========================================  ===============================

The last three are the same retriever with different query strings handed to
it, so the question they answer is narrow and worth stating: *does processing
change the string, and does the changed string change the ranking?* Those are
two different questions and the arm records both answers per query, because
collapsing them is how a no-op gets reported as a small improvement.

One retrieval per (arm, query)
------------------------------
Each query is retrieved **once** per arm at the run's full depth, and every
cutoff -- P@1, P@5, R@10, NDCG@10 -- is computed from that single ranking.
Retrieving separately for each cutoff would mean the P@5 and the R@10 came
from different retrievals of a stochastic system, and the report would be
comparing numbers that were never produced by the same run.

The spell arm's silent-no-op tripwire
-------------------------------------
``QueryProcessor`` guards its spelling stage on ``spell_check and corrector is
not None``, and ``SearchService.from_settings`` only builds a corrector when
``spell_check`` is already on. A configuration override that fails to apply
therefore produces a perfectly well-formed run in which the spell arm changes
nothing, and reports it as a null result forever. Phase 8 shipped one instance
of this bug class -- a benchmark timing the wrong processor and reporting
440x the real cost.

So the arm is built explicitly rather than by flipping a flag and hoping, and
:func:`assert_spell_arm_live` refuses to run the experiment if the corrector
is missing or its vocabulary is empty. A spell experiment that cannot possibly
change anything should fail loudly, not report a clean null.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_search.core.exceptions import EvaluationError
from semantic_search.data.loader import DocumentStore
from semantic_search.evaluation.dataset import DatasetPaths
from semantic_search.evaluation.metrics import (
    average_precision,
    hits_to_ranked_ids,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semantic_search.evaluation.models import ArmResult, EvalDataset, PerQueryResult
from semantic_search.index.bm25 import BM25Config, BM25Index, BM25Searcher
from semantic_search.query.service import SearchService
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: Depth of the single retrieval run. 50 is below the hybrid candidate pool
#: (100), so hybrid is never compared at a shallower pool than its own design
#: intends, and deep enough that R@10 and the pooling step both have something
#: to work with.
DEFAULT_DEPTH = 50

SYSTEMS: tuple[str, ...] = (
    "bm25",
    "semantic",
    "hybrid_rrf",
    "hybrid_qp_off",
    "hybrid_qp_on",
    "hybrid_qp_spell_experiment",
)

SYSTEM_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "semantic": "Semantic",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_qp_off": "Hybrid + QP OFF",
    "hybrid_qp_on": "Hybrid + QP ON (default)",
    "hybrid_qp_spell_experiment": "Hybrid + QP ON + spell_check=True (EXPERIMENT)",
}

#: Arms whose query string is produced by the query processor rather than
#: passed through untouched.
PROCESSED_ARMS = frozenset({"hybrid_qp_on", "hybrid_qp_spell_experiment"})


@dataclass(slots=True, frozen=True)
class ArmRun:
    """One arm's output for the whole dataset, plus its wall clock.

    ``seconds`` is the total for this arm across every query and is reported as
    evaluation runtime. It is **not** a latency measurement: the arms are run
    sequentially, single-threaded, with no concurrency and no warm/cold control,
    so a number derived from it says how long the evaluation took and nothing
    about how fast the system is. Phase 6-9 measured latency properly and those
    numbers are not comparable with these.
    """

    arm: ArmResult
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.arm.system, "seconds": round(self.seconds, 3)}


@dataclass(slots=True, frozen=True)
class EvaluationRun:
    """Everything one evaluation produced."""

    per_query: tuple[PerQueryResult, ...]
    arms: tuple[ArmRun, ...]
    pool: dict[str, tuple[str, ...]]
    seconds: float
    depth: int

    def rankings_for(self, system: str) -> dict[str, tuple[str, ...]]:
        for arm in self.arms:
            if arm.arm.system == system:
                return arm.arm.rankings
        raise EvaluationError(
            f"no arm named {system!r} in this run; ran {sorted(a.arm.system for a in self.arms)}"
        )


def _doc_ids(results: Sequence[object]) -> tuple[str, ...]:
    return tuple(hits_to_ranked_ids(results))


def _titles_for(store: DocumentStore, doc_ids: Sequence[str]) -> tuple[str, ...]:
    titles: list[str] = []
    for doc_id in doc_ids:
        document = store.get(doc_id)
        titles.append(document.title if document is not None else doc_id)
    return tuple(titles)


def assert_spell_arm_live(service: SearchService) -> int:
    """Fail loudly if the spell-correction arm cannot possibly do anything.

    Returns the vocabulary size, so the caller can report it.

    Raises:
        EvaluationError: The arm was configured but has no corrector, or a
            corrector with an empty vocabulary. Either way the experiment would
            report a clean "no difference" that means nothing was measured.
    """
    corrector = service.processor.corrector
    if corrector is None:
        raise EvaluationError(
            "the spell-correction arm has no corrector attached, so it would "
            "return exactly the unprocessed query and report a null result. "
            "This is the silent-no-op failure: SearchService.from_settings only "
            "builds a corrector when query.spell_check is already True, so an "
            "override that failed to apply looks exactly like a genuine no-op."
        )
    size = len(corrector.vocabulary)
    if size == 0:
        raise EvaluationError(
            "the spell-correction arm's vocabulary is empty, so no word could be "
            "corrected even in principle. Check that the corpus was loaded before "
            "the corrector was built."
        )
    if not service.processor.config.spell_check:
        raise EvaluationError(
            "the spell-correction arm has a corrector but query.spell_check is False, "
            "so the processor skips the stage anyway."
        )
    return size


def _score_query(ranked: Sequence[str], relevant: dict[str, int]) -> dict[str, float]:
    """All nine metrics for one query, from one ranking.

    The cutoffs are fixed here rather than passed in, because they are the
    registered metric set. A caller that could choose its own cutoffs would
    choose the ones that flattered the system, and the protocol requires the
    cutoffs to be fixed before the run.
    """
    return {
        "precision_at_1": precision_at_k(ranked, relevant, 1),
        "precision_at_5": precision_at_k(ranked, relevant, 5),
        "precision_at_10": precision_at_k(ranked, relevant, 10),
        "recall_at_5": recall_at_k(ranked, relevant, 5),
        "recall_at_10": recall_at_k(ranked, relevant, 10),
        "reciprocal_rank": reciprocal_rank(ranked, relevant),
        "ndcg_at_5": ndcg_at_k(ranked, relevant, 5),
        "ndcg_at_10": ndcg_at_k(ranked, relevant, 10),
        "average_precision": average_precision(ranked, relevant),
    }


def _run_bm25_arm(searcher: BM25Searcher, dataset: EvalDataset, depth: int) -> ArmResult:
    arm = ArmResult(system="bm25", label=SYSTEM_LABELS["bm25"])
    for query in dataset.queries:
        hits = searcher.search(query.query, top_k=depth)
        ids = _doc_ids(hits)
        arm.rankings[query.query_id] = ids
        arm.processed[query.query_id] = query.query
        arm.modified[query.query_id] = False
        arm.corrections[query.query_id] = ()
        arm.titles[query.query_id] = _titles_for(searcher.store, ids)
    return arm


def _run_semantic_arm(
    service: SemanticSearchService, dataset: EvalDataset, depth: int
) -> ArmResult:
    arm = ArmResult(system="semantic", label=SYSTEM_LABELS["semantic"])
    for query in dataset.queries:
        results = service.search(query.query, top_k=depth)
        ids = _doc_ids(results)
        arm.rankings[query.query_id] = ids
        arm.processed[query.query_id] = query.query
        arm.modified[query.query_id] = False
        arm.corrections[query.query_id] = ()
        arm.titles[query.query_id] = _titles_for(service.store, ids)
    return arm


def _run_search_service_arm(
    service: SearchService, dataset: EvalDataset, depth: int, *, system: str, use_processing: bool
) -> ArmResult:
    """Run one arm through the Phase 8 service, so the ON/OFF switch is the real one.

    ``use_processing=False`` calls ``search_raw``, the same method the API calls
    when processing is bypassed, rather than a runner-local reimplementation of
    "off". A local version could drift from the deployed one and the experiment
    would quietly stop measuring the thing it names.
    """
    arm = ArmResult(system=system, label=SYSTEM_LABELS[system])
    for query in dataset.queries:
        response = (
            service.search(query.query, top_k=depth)
            if use_processing
            else service.search_raw(query.query, top_k=depth)
        )
        ids = _doc_ids(response.results)
        processed = response.query
        arm.rankings[query.query_id] = ids
        arm.processed[query.query_id] = processed.query_for_retrieval
        arm.modified[query.query_id] = processed.was_modified
        arm.corrections[query.query_id] = tuple(
            f"{c.from_term}->{c.to_term}" for c in processed.corrections
        )
        arm.titles[query.query_id] = _titles_for(service.hybrid.store, ids)
    return arm


def build_pool(arms: Sequence[ArmResult], dataset: EvalDataset) -> dict[str, tuple[str, ...]]:
    """Union of every arm's ranking for each query, order-stable, de-duplicated.

    This is the candidate list the stratum-B judgments are made from. It is
    built from the systems being evaluated, which is the standard pooling
    assumption and its cost -- a document all six systems miss is invisible to
    the judge -- is reported per query by
    :func:`~semantic_search.evaluation.dataset.pool_report` rather than left
    implicit.
    """
    pool: dict[str, tuple[str, ...]] = {}
    for query in dataset.queries:
        seen: set[str] = set()
        ordered: list[str] = []
        for arm in arms:
            for doc_id in arm.rankings.get(query.query_id, ()):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                ordered.append(doc_id)
        pool[query.query_id] = tuple(ordered)
    return pool


def score_run(
    run_arms: Sequence[ArmRun], dataset: EvalDataset, store: DocumentStore
) -> tuple[PerQueryResult, ...]:
    """Turn each arm's rankings into a scored :class:`PerQueryResult` per query.

    Queries with no positive judgment are **skipped, not scored as zero**, and
    the count is returned so the report can state it. A query whose gold
    document is absent from the judgments measures the dataset, not the
    retriever; averaging its 0.0 into the result would read as every system
    failing a question they were never given a key for.
    """
    results: list[PerQueryResult] = []
    for run in run_arms:
        arm = run.arm
        for query in dataset.queries:
            relevant = dataset.judgments_for(query.query_id)
            if not relevant:
                continue
            ranked = arm.rankings.get(query.query_id, ())
            scores = _score_query(ranked, relevant)
            top10 = ranked[:10]
            unjudged = (
                sum(1 for doc_id in top10 if doc_id not in relevant) / len(top10) if top10 else 0.0
            )
            gold_rank = next(
                (position for position, doc_id in enumerate(ranked, start=1) if doc_id in relevant),
                0,
            )
            results.append(
                PerQueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    system=arm.system,
                    stratum=query.stratum,
                    returned_doc_ids=tuple(ranked[:10]),
                    relevant_doc_ids=tuple(dataset.positive_doc_ids(query.query_id)),
                    gold_rank=gold_rank,
                    gold_titles=_titles_for(store, dataset.positive_doc_ids(query.query_id)),
                    returned_titles=tuple(arm.titles.get(query.query_id, ())[:10]),
                    unjudged_at_10=unjudged,
                    query_for_retrieval=arm.processed.get(query.query_id, ""),
                    was_modified=arm.modified.get(query.query_id, False),
                    corrections=arm.corrections.get(query.query_id, ()),
                    **scores,
                )
            )
    return tuple(results)


class EvaluationRunner:
    """Runs the six arms over a dataset and scores them.

    The services are injected rather than built here so the integration tests
    can supply a real BM25 index and a real FAISS index with a deterministic
    stand-in encoder. The production path is
    :meth:`from_settings`, which loads the real artifacts.
    """

    def __init__(
        self,
        *,
        bm25: BM25Searcher,
        semantic: SemanticSearchService,
        hybrid: HybridSearchService,
        spell_service: SearchService | None,
        store: DocumentStore,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.bm25 = bm25
        self.semantic = semantic
        self.hybrid = hybrid
        self.spell_service = spell_service
        self.store = store
        self.depth = depth

    @classmethod
    def from_settings(
        cls, settings: Any, *, depth: int = DEFAULT_DEPTH, spell_check: bool = False
    ) -> EvaluationRunner:
        """Build from the real settings, loading the real corpus and indexes."""
        store = DocumentStore.load_jsonl(settings.data.corpus_file)
        semantic = SemanticSearchService.from_settings(settings, store=store)
        bm25_config = BM25Config(**settings.bm25.model_dump())
        index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, bm25_config)
        bm25 = BM25Searcher(index, store)
        hybrid = HybridSearchService.from_settings(settings, bm25=bm25, semantic=semantic)
        spell_service = None
        if spell_check:
            spell_settings = _settings_with_spell_check(settings)
            spell_service = SearchService.from_settings(spell_settings, store, hybrid)
            assert_spell_arm_live(spell_service)
        return cls(
            bm25=bm25,
            semantic=semantic,
            hybrid=hybrid,
            spell_service=spell_service,
            store=store,
            depth=depth,
        )

    def run(self, dataset: EvalDataset, systems: Sequence[str] = SYSTEMS) -> EvaluationRun:
        """Execute every requested arm, then score.

        Raises:
            EvaluationError: An unknown system name. A silently ignored name
                would produce a report missing a row that a reader expects to
                be there.
        """
        unknown = [name for name in systems if name not in SYSTEMS]
        if unknown:
            raise EvaluationError(
                f"unknown evaluation system(s): {unknown}. Known: {list(SYSTEMS)}"
            )

        started = time.perf_counter()
        # The default arm and the OFF arm are the SAME callable with the same
        # use_processing=False, so any difference between them is attributable
        # to the switch and to nothing else. That is the whole point of having
        # two of them, so it is enforced by construction here rather than by
        # two near-identical functions that can drift apart.
        unprocessed = ("hybrid_rrf", "hybrid_qp_off")

        def _make_unprocessed_runner(arm_name: str) -> Callable[[EvalDataset], ArmResult]:
            return lambda ds: self._run_hybrid_arm(ds, arm_name)

        runners: dict[str, Callable[[EvalDataset], ArmResult]] = {
            "bm25": lambda ds: _run_bm25_arm(self.bm25, ds, self.depth),
            "semantic": lambda ds: _run_semantic_arm(self.semantic, ds, self.depth),
        }
        for arm in unprocessed:
            runners[arm] = _make_unprocessed_runner(arm)
        for name in systems:
            if name in runners:
                continue
            if name in PROCESSED_ARMS:
                runners[name] = self._processed_runner(name)
            else:
                raise EvaluationError(
                    f"system {name!r} is known but has no runnable; this is a bug in the "
                    f"runner, not a configuration problem. Known: {list(SYSTEMS)}"
                )

        runs: list[ArmRun] = []
        for name in systems:
            arm_started = time.perf_counter()
            result = runners[name](dataset)
            runs.append(ArmRun(arm=result, seconds=time.perf_counter() - arm_started))

        arms = [run.arm for run in runs]
        pool = build_pool(arms, dataset)
        per_query = score_run(runs, dataset, self.store)
        return EvaluationRun(
            per_query=per_query,
            arms=tuple(runs),
            pool=pool,
            seconds=time.perf_counter() - started,
            depth=self.depth,
        )

    def _run_hybrid_arm(self, dataset: EvalDataset, name: str) -> ArmResult:
        """The plain fused service, with the query passed straight through.

        Goes through ``SearchService.search_raw`` rather than calling
        ``HybridSearchService.search`` directly, so the default arm and the
        OFF arm differ only in the processing switch and nothing else -- not in
        which method was called, not in validation, not in top-k handling.
        """
        service = self.spell_service
        if service is None:
            raise EvaluationError(f"the {name} arm needs a SearchService, and none was provided.")
        return _run_search_service_arm(
            service, dataset, self.depth, system=name, use_processing=False
        )

    def _processed_runner(self, name: str) -> Callable[[EvalDataset], ArmResult]:
        service = self.spell_service
        if service is None:
            raise EvaluationError(
                f"the {name} arm needs a SearchService, and none was provided. The "
                "production path builds one with spell_check enabled; the integration "
                "tests pass one in directly."
            )
        return lambda ds: _run_search_service_arm(
            service, ds, self.depth, system=name, use_processing=True
        )


def _settings_with_spell_check(settings: Any) -> Any:
    """A copy of the settings with ``query.spell_check`` forced on.

    Uses ``model_copy(update=...)`` rather than mutating the loaded settings, so
    the experiment cannot leave the process configured for spell correction
    after it finishes. A default changed by a side effect of running an
    experiment is exactly the failure this whole arm exists to investigate.
    """
    query = settings.query.model_copy(update={"spell_check": True})
    return settings.model_copy(update={"query": query})


def write_per_query(path: Path, per_query: Sequence[PerQueryResult]) -> Path:
    """Write the per-query artifact, one JSON object per line.

    This file is the auditable record: every number in the report can be
    recomputed from it without re-running anything, which is the difference
    between an evaluation and a table someone printed.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in per_query:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def default_report_dir(paths: DatasetPaths) -> Path:
    return paths.derived
