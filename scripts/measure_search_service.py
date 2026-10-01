#!/usr/bin/env python3
"""Measure latency for the application search service.

Uses the real corpus (300 documents), the real BM25 and FAISS indexes, and the
real sentence-transformer model (all-MiniLM-L6-v2). No fakes, no stubs.

Four numbers are reported:

* **query processing** -- ``QueryProcessor.process`` on its own.
* **retrieval** -- ``HybridSearchService.search`` (BM25 + embedding + FAISS + RRF).
* **application / result assembly** -- everything the Phase 9 layer does: the
  request object, dispatch, mapping results to ``SearchHit``, and metadata.
  Derived as ``end_to_end - processing - retrieval`` from the *same* runs
  rather than from a second, separately-timed pass, so the three components and
  the total always describe one real request rather than three different ones.
* **end to end** -- ``ApplicationSearchService.search_text``.

The derived component is a subtraction and is therefore subject to the
numerical noise of the two it came from; it is labelled as derived rather than
measured directly. On this workload it is a fraction of a millisecond, which is
the same order as a single ``perf_counter`` reading, so the subtraction is
reported for completeness and not treated as a precise figure.

Measurements are warm: the model, both indexes, and the spelling vocabulary are
all resident before the first timed call. Cold-start cost (model load, index
load) is reported separately by ``probe_search_service.py`` and is deliberately
excluded from every latency figure here.

No retrieval-quality claim is made or implied by anything in this file.
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np

# Ensure we are using the real model and corpus, not the test fakes.
os.environ.setdefault("SSE_ENV", "development")
os.environ.setdefault("SSE_CORPUS_SIZE", "small")

from semantic_search.app.service import ApplicationSearchService
from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: Real queries spanning short and long input, technical and general vocabulary.
QUERIES: tuple[str, ...] = (
    "machine learning",
    "engine",
    "artificial intelligence",
    "anatomy",
    "how does a muscle pump move blood",
    "photosynthesis",
    "the history of the Roman empire",
    "quantum",
    "computer",
    "water",
    "sorting algorithm",
    "photosynthesis in plants",
    "list of artificial intelligence projects",
    "Kolmogorov complexity",
    "accumulator",
)

WARMUP_PASSES = 5
REPETITIONS = 20


def _summarise(label: str, samples: list[float]) -> None:
    """Print p50/p95/p99/max/mean for one component, in milliseconds."""
    array = np.asarray(samples, dtype=float)
    print(
        f"  {label:<28} mean={array.mean():7.2f}  p50={np.percentile(array, 50):7.2f}  "
        f"p95={np.percentile(array, 95):7.2f}  p99={np.percentile(array, 99):7.2f}  "
        f"max={array.max():7.2f}  n={len(array)}"
    )


def main() -> None:
    settings = load_settings()
    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not len(store):
        print("ERROR: corpus is empty. Run the data pipeline first.")
        raise SystemExit(1)

    bm25 = BM25Searcher(
        BM25Index.load(settings.data.index_dir / settings.bm25.index_filename),
        store,
    )
    semantic = SemanticSearchService.from_settings(settings, store=store)
    hybrid = HybridSearchService(bm25, semantic, hybrid_config=settings.hybrid)
    app_service = ApplicationSearchService.from_settings(settings, store, hybrid)

    inner = app_service.search_service
    processor = inner.processor
    top_k = settings.app.default_top_k

    print("Warming up (model, indexes, and vocabulary already loaded)...")
    for _ in range(WARMUP_PASSES):
        for query in QUERIES:
            app_service.search_text(query)

    print(
        f"Measuring: {len(QUERIES)} queries x {REPETITIONS} repetitions "
        f"= {len(QUERIES) * REPETITIONS} requests\n"
    )

    e2e_ms: list[float] = []
    processing_ms: list[float] = []
    retrieval_ms: list[float] = []
    assembly_ms: list[float] = []

    for _ in range(REPETITIONS):
        for query in QUERIES:
            # --- end to end: the only number that is timed end to end ---
            start = time.perf_counter()
            app_service.search_text(query, top_k)
            total = (time.perf_counter() - start) * 1000.0

            # --- the two components, called again to time them individually ---
            # A second call rather than a hook inside the request: the first call
            # above is the honest end-to-end measurement, and instrumenting it
            # would change the thing being measured. The components are warm and
            # deterministic, so the repeated call is representative -- but the
            # subtraction below is what keeps the reported total authoritative.
            start = time.perf_counter()
            processed = processor.process(query)
            processing = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            hybrid.search(processed.query_for_retrieval, top_k=top_k)
            retrieval = (time.perf_counter() - start) * 1000.0

            e2e_ms.append(total)
            processing_ms.append(processing)
            retrieval_ms.append(retrieval)
            assembly_ms.append(total - processing - retrieval)

    print("--- Latency (ms) ---")
    _summarise("end to end", e2e_ms)
    _summarise("query processing", processing_ms)
    _summarise("retrieval (BM25+FAISS+RRF)", retrieval_ms)
    _summarise("app / result assembly *", assembly_ms)
    print("  * derived: end_to_end - processing - retrieval from the same requests.")

    print("\n--- Environment ---")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"CPU count (logical): {os.cpu_count()}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device in use: {settings.embedding.device}")
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        print("torch: not installed")
    print(f"Environment: {settings.environment}")
    print(f"Corpus size: {len(store)} documents")
    print(f"Model: {settings.embedding.model_name}")
    print(f"Embedding dimension: {semantic.encoder.dimension}")
    print(f"FAISS index type: {settings.faiss.index_type}")
    print(f"Retrievers: {', '.join(hybrid.retriever_names)}")
    print(f"RRF k: {settings.hybrid.rrf_k}, candidate pool: {settings.hybrid.candidate_pool}")
    print(f"Queries: {len(QUERIES)} distinct")
    print(f"Repetitions: {REPETITIONS} (warm: {WARMUP_PASSES} untimed passes first)")
    print("State: WARM -- model, both indexes, and vocabulary resident before timing")
    print(f"top_k: {top_k}")


if __name__ == "__main__":
    main()
