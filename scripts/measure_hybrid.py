"""Measure the hybrid pipeline with the latency split by stage.

The question this answers is narrow and falsifiable: **how expensive is the
fusion layer itself?** RRF is assumed to be negligible next to a SentenceTransformer
encode, and that assumption is either right or wrong in a way that a single
end-to-end number cannot show. If fusion were quietly expensive, an end-to-end
figure would show it and nothing would say why.

So each stage is timed separately -- BM25, query encoding, FAISS, RRF fusion,
and result assembly -- and the end-to-end figure is measured through the public
API rather than summed from the parts, so the overhead the stages omit is
included rather than hidden. Percentiles are nearest-rank, no interpolation,
so every reported number is a sample that actually occurred.

The candidate depth is reported because it changes what is being timed: a
deeper pool means more candidates to fuse. ``--candidate-pool 1`` and
``--candidate-pool 100`` bound that effect directly.

Usage::

    python scripts/measure_hybrid.py
    python scripts/measure_hybrid.py --queries 30 --repeat 30 --top-k 10
    python scripts/measure_hybrid.py --candidate-pool 1     # candidate_k == top_k
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
from semantic_search.search.hybrid import BM25, SEMANTIC, HybridSearchService
from semantic_search.search.rrf import RetrievalRun, fuse
from semantic_search.search.semantic import SemanticSearchService

#: The same fixed query set as ``measure_semantic.py``, so the two benchmarks
#: are comparable. Not random: a benchmark whose inputs change between runs
#: cannot be compared to its own previous result.
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


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no interpolation, so every figure is a real sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=int, default=15, help="queries per repetition")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20, help="repetitions of the query set")
    parser.add_argument("--warmup", type=int, default=2, help="untimed passes to load lazily")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=None,
        help="override hybrid.candidate_pool (default: whatever settings.py has)",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")
    if args.candidate_pool is not None:
        settings.hybrid.candidate_pool = args.candidate_pool

    load_started = time.perf_counter()
    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, settings.bm25)
    bm25 = BM25Searcher(index, store)
    semantic = SemanticSearchService.from_settings(settings, store=store)
    service = HybridSearchService(
        bm25, semantic, hybrid_config=settings.hybrid, search_config=settings.search
    )
    startup_seconds = time.perf_counter() - load_started

    queries = (QUERIES * ((args.queries // len(QUERIES)) + 1))[: args.queries]
    candidate_k = max(args.top_k, service.config.candidate_pool)

    # Warm up: the first query triggers a lazy model load and FAISS's lazy
    # internal state, and must not be counted as query latency.
    for _ in range(args.warmup):
        service.search(queries[0], top_k=args.top_k)

    bm25_ms: list[float] = []
    encode_ms: list[float] = []
    faiss_ms: list[float] = []
    fusion_ms: list[float] = []
    resolve_ms: list[float] = []
    total_ms: list[float] = []

    for _ in range(args.repeat):
        for query in queries:
            started = time.perf_counter()
            lexical_hits = bm25.search(query, top_k=candidate_k)
            after_bm25 = time.perf_counter()

            vector = semantic.encoder.encode_query(query)
            after_encode = time.perf_counter()

            dense_hits = semantic.index.search(vector, top_k=candidate_k)
            after_faiss = time.perf_counter()

            # The fusion stage, timed as exactly as the service runs it: both
            # retrievers' output is already in hand, and what is timed is the
            # arithmetic plus the sort -- nothing a retriever would have done.
            # FaissHit carries a store row, not a doc_id, exactly as the
            # service sees it. Resolving through the same store the service
            # uses keeps this benchmark measuring the same shape of work the
            # service does, rather than a cheaper stand-in.
            dense_ids = [store[hit.row].id for hit in dense_hits]
            runs = [
                RetrievalRun(
                    name=BM25,
                    ranked_doc_ids=[hit.doc_id for hit in lexical_hits],
                    scores={hit.doc_id: hit.score for hit in lexical_hits},
                ),
                RetrievalRun(
                    name=SEMANTIC,
                    ranked_doc_ids=dense_ids,
                    scores={
                        doc_id: hit.score for doc_id, hit in zip(dense_ids, dense_hits, strict=True)
                    },
                ),
            ]
            candidates = fuse(runs, k=service.config.rrf_k)
            after_fusion = time.perf_counter()

            raw_scores = {
                BM25: {hit.doc_id: hit.score for hit in lexical_hits},
                SEMANTIC: {
                    doc_id: hit.score for doc_id, hit in zip(dense_ids, dense_hits, strict=True)
                },
            }
            service._to_results(candidates[: args.top_k], raw_scores=raw_scores, query=query)
            after_resolve = time.perf_counter()

            bm25_ms.append((after_bm25 - started) * 1000)
            encode_ms.append((after_encode - after_bm25) * 1000)
            faiss_ms.append((after_faiss - after_encode) * 1000)
            fusion_ms.append((after_fusion - after_faiss) * 1000)
            resolve_ms.append((after_resolve - after_fusion) * 1000)

        # The end-to-end figure comes from the real public API, not from
        # summing the stages, so the overhead the stages omit is included.
        for query in queries:
            started = time.perf_counter()
            service.search(query, top_k=args.top_k)
            total_ms.append((time.perf_counter() - started) * 1000)

    samples = len(total_ms)
    mean_fusion = statistics.mean(fusion_ms)
    print(
        f"environment     {platform.system()} {platform.release()}, Python {platform.python_version()}"
    )
    print(f"cpu             {os.cpu_count()} logical cores, no GPU")
    print(f"corpus          {len(store)} documents")
    print(f"index           {semantic.index.index_type}, {semantic.index.dimension}d")
    print(f"model           {semantic.encoder.config.model_name} on {semantic.encoder.device}")
    print(f"top_k           {args.top_k}")
    print(
        f"candidate depth {candidate_k} per retriever (hybrid.candidate_pool={service.config.candidate_pool})"
    )
    print(f"rrf_k           {service.config.rrf_k}")
    print(f"samples         {samples} ({args.queries} queries x {args.repeat} passes, warm)")
    print(f"startup         {startup_seconds * 1000:.0f} ms to load store + index + model")
    print("percentiles     nearest-rank, no interpolation\n")

    rows = [
        ("bm25 search", bm25_ms),
        ("query encoding", encode_ms),
        ("faiss search", faiss_ms),
        ("RRF FUSION", fusion_ms),
        ("result assembly", resolve_ms),
        ("END-TO-END (hybrid.search)", total_ms),
    ]
    print(f"{'stage':<32} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   {'mean':>8}")
    print("-" * 78)
    for label, values in rows:
        print(
            f"{label:<32} {percentile(values, 0.50):>7.2f}m {percentile(values, 0.95):>7.2f}m "
            f"{percentile(values, 0.99):>7.2f}m {max(values):>7.2f}m {statistics.mean(values):>7.2f}m"
        )

    mean_encode = statistics.mean(encode_ms)
    mean_total = statistics.mean(total_ms)
    print(f"\nfusion as a share of end-to-end mean:  {mean_fusion / mean_total * 100:.2f}%")
    print(f"fusion relative to one query encode:   {mean_fusion / mean_encode:.4f}x")
    print(f"fusion adds {mean_fusion:.3f} ms to a {mean_total:.2f} ms hybrid query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
