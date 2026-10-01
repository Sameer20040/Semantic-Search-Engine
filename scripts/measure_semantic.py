"""Measure the semantic service end to end, with the latency split by stage.

Raw FAISS latency is measured in Phase 5 as p50 0.027 ms at top_k=10. That
number is the small part of a real query: the query must be embedded first,
and the returned rows resolved to documents. Reporting only the FAISS figure
would understate what a user actually waits, so each stage is timed separately
and the sum is shown against the measured end-to-end wall clock.

Usage::

    python scripts/measure_semantic.py
    python scripts/measure_semantic.py --queries 50 --top-k 10
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.config import load_settings
from semantic_search.search.semantic import SemanticSearchService

#: Fixed so repeated runs are comparable. Not random: a benchmark whose inputs
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
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")

    load_started = time.perf_counter()
    service = SemanticSearchService.from_settings(settings)
    startup_seconds = time.perf_counter() - load_started

    queries = (QUERIES * ((args.queries // len(QUERIES)) + 1))[: args.queries]

    # Warm up: the first query triggers a lazy model load and FAISS's lazy
    # internal state, and must not be counted as query latency.
    for _ in range(args.warmup):
        service.search(queries[0], top_k=args.top_k)

    encode_ms: list[float] = []
    faiss_ms: list[float] = []
    retrieve_ms: list[float] = []
    total_ms: list[float] = []

    for _ in range(args.repeat):
        for query in queries:
            started = time.perf_counter()
            vector = service.encoder.encode_query(query)
            after_encode = time.perf_counter()

            hits = service.index.search(vector, top_k=args.top_k)
            after_faiss = time.perf_counter()

            service._to_results(hits, query=query)
            after_retrieve = time.perf_counter()

            encode_ms.append((after_encode - started) * 1000)
            faiss_ms.append((after_faiss - after_encode) * 1000)
            retrieve_ms.append((after_retrieve - after_faiss) * 1000)

        # The end-to-end figure comes from the real public API, not from
        # summing the stages, so the overhead the stages omit is included.
        for query in queries:
            started = time.perf_counter()
            service.search(query, top_k=args.top_k)
            total_ms.append((time.perf_counter() - started) * 1000)

    samples = len(total_ms)
    print(f"corpus          {service.corpus_size} documents")
    print(f"index           {service.index.index_type}, {service.index.dimension}d")
    print(f"model           {service.encoder.config.model_name} on {service.encoder.device}")
    print(f"top_k           {args.top_k}")
    print(f"samples         {samples} ({args.queries} queries x {args.repeat} passes, warm)")
    print(f"startup         {startup_seconds * 1000:.0f} ms to load store + index + model\n")

    rows = [
        ("query encoding", encode_ms),
        ("faiss search", faiss_ms),
        ("document retrieval", retrieve_ms),
        ("END-TO-END (service.search)", total_ms),
    ]
    print(f"{'stage':<32} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   {'mean':>8}")
    print("-" * 78)
    for label, values in rows:
        print(
            f"{label:<32} {percentile(values, 0.50):>7.2f}m {percentile(values, 0.95):>7.2f}m "
            f"{percentile(values, 0.99):>7.2f}m {max(values):>7.2f}m {statistics.mean(values):>7.2f}m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
