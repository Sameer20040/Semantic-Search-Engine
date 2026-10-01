"""Search the built indexes and print the results, for manual inspection.

This is a development tool, not part of the API surface. It exists so that
ranking quality can be *looked at* rather than only measured -- an IR metric
will happily report a mediocre average while a specific failure is obvious by
eye.

Usage::

    python scripts/probe_index.py "reflectivity of the moon and snow"
    python scripts/probe_index.py "how do magnets work" --mode semantic --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.config import load_settings
from semantic_search.core.logging import configure_logging
from semantic_search.data.loader import DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Searcher
from semantic_search.index.faiss_index import FaissIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="+", help="the query text")
    parser.add_argument("--mode", default="both", choices=("bm25", "semantic", "both"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1, help="time N runs and report the mean")
    args = parser.parse_args()

    query = " ".join(args.query)
    settings = load_settings(environment="development")
    configure_logging("WARNING")

    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not store:
        print("error: corpus is empty; run scripts/download_data.py", file=sys.stderr)
        return 1

    print(f'query: "{query}"\ncorpus: {len(store)} documents\n')

    if args.mode in ("bm25", "both"):
        started = time.perf_counter()
        searcher = BM25Searcher.from_documents(store, config=settings.bm25)
        results = searcher.search(query, top_k=args.top_k)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"BM25  ({elapsed:.0f} ms, including {elapsed/2:.0f} ms build)")
        for hit in results:
            print(f"  {hit.rank + 1}. [{hit.score:6.2f}] {store[hit.row].title}")
            if hit.matched_terms:
                print(f"        matched: {', '.join(hit.matched_terms)}")
        if not results:
            print("  (no lexical matches)")
        print()

    if args.mode in ("semantic", "both"):
        encoder = EmbeddingEncoder(settings.embedding)
        # Start the timer after the model is loaded: measuring a cold model load
        # as "search latency" would be meaningless.
        vector = encoder.encode_query(query)
        index = FaissIndex.load(
            settings.data.index_dir / settings.faiss.index_filename,
            expected_model=settings.embedding.model_name,
            expected_documents=len(store),
        )

        timings = []
        hits = []
        for _ in range(max(1, args.repeat)):
            started = time.perf_counter()
            hits = index.search(vector, top_k=args.top_k)
            timings.append((time.perf_counter() - started) * 1000)

        mean = sum(timings) / len(timings)
        suffix = f" (mean of {len(timings)})" if len(timings) > 1 else ""
        print(f"SEMANTIC  {mean:.2f} ms{suffix}")
        for hit in hits:
            print(f"  {hit.rank + 1}. [{hit.score:6.3f}] {store[hit.row].title}")
        if hits:
            print(
                f"\n  top score {hits[0].score:.3f}, "
                f"cutoff {hits[-1].score:.3f}, gap {hits[0].score - hits[-1].score:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
