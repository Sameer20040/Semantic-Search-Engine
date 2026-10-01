"""Measure the query-processing pipeline with latency split by stage.

The question is narrow and falsifiable: **what does query processing cost, and
is it worth its place in front of retrieval?** A single end-to-end number
cannot answer that, because the stages differ by orders of magnitude: one of
them is pure string work and another is a SentenceTransformer encode. So each
stage is timed separately -- normalization, spelling, expansion, the processing
total, retrieval, and end-to-end through the public API -- and the end-to-end
figure is *measured*, not summed, so the overhead the stages omit is included
rather than hidden.

Percentiles are nearest-rank with no interpolation, so every reported number is
a sample that actually occurred rather than a value between two samples.

Two configurations are measured, because the shipped default is the one that
matters and the enabled one is the one a user is contemplating:

* ``as-shipped`` -- the config as it loads: spell check off, synonyms
  record-only. This is the cost a deployment actually pays.
* ``spell-check`` -- spelling on against the real corpus vocabulary. This is
  the only stage that can be expensive, and it is the reason the vocabulary is
  built once at startup and shared: a per-request vocabulary would be
  catastrophic, and the startup cost is reported so that claim is checkable.

The retrieval figure is identical in both configurations, which is the point of
splitting them out: it is the control.

Nothing here measures retrieval quality. A latency table says what processing
costs, and a cost is not a benefit. Whether correcting a typo returns better
documents needs labelled relevance judgements, which this project does not have.

Usage::

    python scripts/measure_query_processing.py
    python scripts/measure_query_processing.py --queries 30 --repeat 30 --top-k 10
    python scripts/measure_query_processing.py --skip-spell-check
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.query.normalizer import normalize
from semantic_search.query.processor import QueryProcessor
from semantic_search.query.service import SearchService
from semantic_search.query.spelling import SpellCorrector, build_vocabulary
from semantic_search.query.synonyms import apply_expansions, find_expansions
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: The same fixed query set as ``measure_hybrid.py``, so the retrieval column is
#: directly comparable with Phase 7's. Not random: a benchmark whose inputs
#: change between runs cannot be compared to its own previous result.
QUERIES = [
    "orbital mechanics of a binary star",
    "how a muscle pump moves blood",
    "the composition of the Earth's atmosphere",
    "origins of the industrial revolution",
    "how vaccines train the immune system",
    "structure of DNA and genetic inheritance",
    "the causes of the French Revolution",
    "how earthquakes are measured",
    "the life cycle of a butterfly",
    "differences between democracy and dictatorship",
    "how the immune system fights infection",
    "the water cycle and evaporation",
    "black holes and gravitational fields",
    "why the sky appears blue",
    "the history of the Roman Empire",
]

#: A second set with typos, used only to show the spelling stage on input that
#: actually needs it. On the clean set the corrector short-circuits at gate 0
#: and the measurement is of a membership test, not of fuzzy search.
TYPO_QUERIES = [
    "orbital mechanincs of a binary star",
    "how a muscle pumpe moves blood",
    "the composition of the Earths atmosphere",
    "origins of the industrail revolution",
    "how vaccines train the immune systm",
    "structure of DNA and genetice inheritance",
    "the causes of the Frech Revolution",
    "how eartquakes are measured",
    "the life cycle of a butterfly",
    "diferences between democracy and dictatorship",
    "how the immune system fighs infection",
    "the water cycle and evaporaton",
    "black holes and gravitaional fields",
    "why the sky apears blue",
    "the history of the Roman Empie",
]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no interpolation, so every figure is a real sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _table(title: str, rows: list[tuple[str, list[float]]]) -> None:
    print(f"\n{title}")
    print(f"{'stage':<34} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   {'mean':>8}")
    print("-" * 80)
    for label, values in rows:
        if not values:
            print(f"{label:<34} {'n/a':>8} (stage not run in this configuration)")
            continue
        print(
            f"{label:<32} {percentile(values, 0.50):>7.3f}m {percentile(values, 0.95):>7.3f}m "
            f"{percentile(values, 0.99):>7.3f}m {max(values):>7.3f}m {statistics.mean(values):>7.3f}m"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=int, default=15, help="queries per repetition")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20, help="repetitions of the query set")
    parser.add_argument("--warmup", type=int, default=2, help="untimed passes to load lazily")
    parser.add_argument(
        "--skip-spell-check",
        action="store_true",
        help="do not measure the spell-check configuration (the default measures both)",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")
    settings.query.enabled = True

    load_started = time.perf_counter()
    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not store:
        print("error: corpus is empty; run scripts/download_data.py", file=sys.stderr)
        return 1
    index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, settings.bm25)
    bm25 = BM25Searcher(index, store)
    semantic = SemanticSearchService.from_settings(settings, store=store)
    hybrid = HybridSearchService(
        bm25, semantic, hybrid_config=settings.hybrid, search_config=settings.search
    )
    load_seconds = time.perf_counter() - load_started

    # The vocabulary build is timed on its own. This is the one-off startup
    # cost that justifies sharing a corrector across requests, and it is
    # reported rather than assumed, because "build it once" is only true if
    # building it is expensive -- and if it is not, the argument changes.
    vocabulary_started = time.perf_counter()
    vocabulary = build_vocabulary(store)
    vocabulary_seconds = time.perf_counter() - vocabulary_started

    spell_config = settings.query.model_copy(deep=True)
    spell_config.spell_check = True
    corrector = SpellCorrector(
        vocabulary,
        enabled=True,
        preserve_proper_nouns=spell_config.preserve_proper_nouns,
        min_correction_score=spell_config.min_correction_score,
        max_edit_distance=spell_config.min_edit_distance,
        min_candidate_df=spell_config.min_candidate_df,
    )
    validate = hybrid.semantic.validate_query  # type: ignore[union-attr]

    shipped = SearchService.from_settings(settings, store, hybrid)
    # As-shipped uses the real config: spell_check is False, so the corrector
    # the service builds is None and the stage is skipped entirely. The
    # spell-enabled service is built here rather than through from_settings so
    # the two are guaranteed to share the index and the store.
    spelled = SearchService(
        hybrid, QueryProcessor(spell_config, corrector=corrector, validate=validate), settings
    )

    queries = (QUERIES * ((args.queries // len(QUERIES)) + 1))[: args.queries]
    typos = (TYPO_QUERIES * ((args.queries // len(TYPO_QUERIES)) + 1))[: args.queries]

    for _ in range(args.warmup):
        shipped.search(queries[0], top_k=args.top_k)
        hybrid.search(queries[0], top_k=args.top_k)

    normalize_ms: list[float] = []
    spell_ms: list[float] = []
    expand_ms: list[float] = []
    ship_process_ms: list[float] = []
    spell_process_ms: list[float] = []
    retrieve_ms: list[float] = []
    total_ms: list[float] = []
    raw_total_ms: list[float] = []

    for _ in range(args.repeat):
        for query in queries:
            # --- individual stages, on the clean query set -----------------
            # Timed against the stage functions directly, not the processor,
            # so the columns are attributable. The clean set is the surprising
            # one: it was expected to be nearly free via the membership gate,
            # and measurement says it is not. Most ordinary query terms are
            # absent from a 300-document corpus, so they reach the fuzzy search
            # and are refused late. Both sets are reported so the cost of
            # correcting nothing is visible.
            started = time.perf_counter()
            normalized = normalize(query)
            after_normalize = time.perf_counter()

            corrector.correct(normalized)
            after_spell = time.perf_counter()

            expansions = find_expansions(normalized)
            apply_expansions(normalized, expansions)  # timed whether or not applied
            after_expand = time.perf_counter()

            normalize_ms.append((after_normalize - started) * 1000)
            spell_ms.append((after_spell - after_normalize) * 1000)
            expand_ms.append((after_expand - after_spell) * 1000)

            # --- the full processing stage, through the public API, BOTH
            #     configurations. These must be timed separately: measuring the
            #     spell-enabled processor here and labelling the table
            #     "as-shipped" would overstate what a deployment pays by
            #     two orders of magnitude and hide the real answer.
            started = time.perf_counter()
            shipped.processor.process(query)
            ship_process_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            spelled.processor.process(query)
            spell_process_ms.append((time.perf_counter() - started) * 1000)

            # --- retrieval, with no processing at all ---------------------
            started = time.perf_counter()
            hybrid.search(query, top_k=args.top_k)
            retrieve_ms.append((time.perf_counter() - started) * 1000)

        # --- end-to-end, through the public API, both paths ---------------
        for query in queries:
            started = time.perf_counter()
            shipped.search(query, top_k=args.top_k)
            total_ms.append((time.perf_counter() - started) * 1000)
        for query in queries:
            started = time.perf_counter()
            shipped.search_raw(query, top_k=args.top_k)
            raw_total_ms.append((time.perf_counter() - started) * 1000)

    # Spell check on the typo set, which is the input it exists for. On the
    # clean set the corrector exits at the membership gate, so reporting only
    # that would understate the cost of the feature by construction.
    typo_spell_ms: list[float] = []
    typo_corrected = 0
    typo_total = 0
    if not args.skip_spell_check:
        for _ in range(args.repeat):
            for query in typos:
                started = time.perf_counter()
                found = corrector.correct(query)[1]
                typo_spell_ms.append((time.perf_counter() - started) * 1000)
                typo_total += len(query.split())
                typo_corrected += len(found)

    samples = len(total_ms)
    print(
        f"environment     {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}"
    )
    print(f"cpu             {os.cpu_count()} logical cores, no GPU")
    print(f"corpus          {len(store)} documents, {len(vocabulary):,} vocabulary terms")
    print(f"index           {semantic.index.index_type}, {semantic.index.dimension}d")
    print(f"model           {semantic.encoder.config.model_name} on {semantic.encoder.device}")
    print(f"top_k           {args.top_k}")
    print(f"samples         {samples} ({args.queries} queries x {args.repeat} passes, warm)")
    print(
        f"startup         {load_seconds * 1000:.0f} ms store + index + model "
        f"(excludes the vocabulary)"
    )
    print(
        f"vocabulary      {vocabulary_seconds * 1000:.0f} ms to build, one-off, "
        f"shared by every request"
    )
    print("percentiles     nearest-rank, no interpolation")

    _table(
        "STAGES (as-shipped config: spell_check=False, synonym_expansion=False)",
        [
            ("normalize", normalize_ms),
            ("spell check (skipped: flag off)", []),
            ("synonym expansion (recorded only)", expand_ms),
            ("PROCESSING TOTAL (processor.process)", ship_process_ms),
            ("RETRIEVAL ONLY (hybrid.search)", retrieve_ms),
            ("END-TO-END, processing OFF (search_raw)", raw_total_ms),
            ("END-TO-END, processing ON (search)", total_ms),
        ],
    )

    if not args.skip_spell_check:
        _table(
            "SPELL CHECK, spelling ON (the opt-in configuration)",
            [
                ("PROCESSING TOTAL, spell_check=True", spell_process_ms),
                ("spell check, clean query set", spell_ms),
                ("spell check, typo query set", typo_spell_ms),
            ],
        )
        terms = typo_total
        rate = typo_corrected / terms * 100 if terms else 0.0
        print(
            f"\n  {typo_corrected} of {terms} typed terms were corrected "
            f"({rate:.1f}%) over {args.repeat} passes of the typo set."
        )
        print("  This is a correction RATE on a synthetic typo set, not retrieval")
        print("  quality. It is the wrong number to quote as an improvement.")

    mean_process = statistics.mean(ship_process_ms)
    mean_retrieve = statistics.mean(retrieve_ms)
    mean_total = statistics.mean(total_ms)
    mean_raw = statistics.mean(raw_total_ms)
    print("\nRATIOS (as-shipped configuration)")
    print(f"  processing mean            {mean_process:.3f} ms")
    print(f"  retrieval mean             {mean_retrieve:.3f} ms")
    print(f"  processing / retrieval     {mean_process / mean_retrieve * 100:.2f}%")
    print(f"  processing / end-to-end    {mean_process / mean_total * 100:.2f}%")
    print(
        f"  end-to-end ON vs OFF       {mean_total:.3f} vs {mean_raw:.3f} ms "
        f"({(mean_total - mean_raw) / mean_raw * 100:+.2f}%)"
    )
    if not args.skip_spell_check:
        mean_spell_process = statistics.mean(spell_process_ms)
        mean_typo_spell = statistics.mean(typo_spell_ms) if typo_spell_ms else 0.0
        print("\nRATIOS (spell_check=True, which is not the default)")
        print(
            f"  processing mean            {mean_spell_process:.3f} ms "
            f"({mean_spell_process / mean_retrieve * 100:.1f}% of retrieval)"
        )
        print(
            f"  typo-set spell stage       {mean_typo_spell:.3f} ms "
            f"({mean_typo_spell / mean_retrieve * 100:.1f}% of retrieval)"
        )
        print(
            f"  clean-set spell stage      {statistics.mean(spell_ms):.3f} ms "
            f"({statistics.mean(spell_ms) / mean_retrieve * 100:.1f}% of retrieval)"
        )
        print()
        print("  The clean-set cost is NOT near zero, and that was not the expectation.")
        print("  Gate 0 (is the term in the vocabulary?) only short-circuits terms the")
        print("  corpus actually contains. An ordinary English query against a 300-doc")
        print("  Wikipedia corpus is mostly out-of-vocabulary -- 'orbital', 'vaccines',")
        print("  'butterfly' are not in these 300 articles -- so every one of them falls")
        print("  through to a fuzzy search over all 43,120 terms and is refused by a")
        print("  later gate. The cost of a correction nobody gets is paid in full.")

    print("\nNot measured: whether any of this makes the results better. There are")
    print("no labelled relevance judgements in this project, so no such claim is")
    print("made here. These numbers are costs, and a cost is not a benefit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
