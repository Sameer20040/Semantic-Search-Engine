#!/usr/bin/env python3
"""Demonstrate the application search service with eight required cases.

This script uses the real 300-document corpus, the real BM25 index, the real
FAISS index, and the real sentence-transformer model (all-MiniLM-L6-v2).

It does NOT claim any retrieval quality or performance numbers; it only shows
that the service works and that the contract is respected.

The eight cases:
1. Normal query.
2. Query containing extra whitespace.
3. Query processing disabled.
4. Query processing enabled.
5. `top_k` below available results.
6. `top_k` above available documents.
7. Invalid/empty query rejection.
8. A query where BM25 and semantic retrieval differ.

For each case, we print a concise description and the actual output.

We do not use the string "reflectivity of the moon and snow".
We may use "how a muscle pump moves blood" only to show that the retrievers
can behave differently, not to claim one is better.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure we are using the real model and corpus, not the test fakes.
os.environ.setdefault("SSE_ENV", "development")
os.environ.setdefault("SSE_CORPUS_SIZE", "small")

from semantic_search.app.service import ApplicationSearchService
from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService


def main() -> None:
    print("Loading real corpus and indexes...")
    t0 = time.time()
    settings = load_settings()

    if not settings.data.corpus_file.exists():
        print(f"ERROR: Corpus file not found at {settings.data.corpus_file}")
        print("Have you run the data pipeline?")
        sys.exit(1)

    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if len(store) == 0:
        print("ERROR: Corpus is empty.")
        sys.exit(1)
    print(f"Corpus loaded: {len(store)} documents")

    # Load BM25
    bm25_index_path = settings.data.index_dir / settings.bm25.index_filename
    if not bm25_index_path.exists():
        print(f"ERROR: BM25 index not found at {bm25_index_path}")
        sys.exit(1)
    bm25_index = BM25Index.load(bm25_index_path)
    bm25 = BM25Searcher(bm25_index, store)
    print("BM25 index loaded")

    # Load Semantic via its from_settings factory (handles model and FAISS)
    semantic = SemanticSearchService.from_settings(settings, store=store)
    print("FAISS index and model loaded")

    hybrid = HybridSearchService(bm25, semantic, hybrid_config=settings.hybrid)
    app_service = ApplicationSearchService.from_settings(settings, store, hybrid)

    t1 = time.time()
    print(f"Setup complete in {t1 - t0:.2f}s\n")

    # Case 1: Normal query
    print("=== Case 1: Normal query ===")
    resp = app_service.search_text("machine learning", top_k=3)
    print(f"Query: {resp.original_query!r}")
    print(f"Processed: {resp.processed_query!r}")
    print(f"Results: {len(resp.results)}")
    for hit in resp.results:
        print(f"  Rank {hit.rank}: {hit.title} (score={hit.score:.4f})")
    print()

    # Case 2: Query containing extra whitespace
    print("=== Case 2: Query with extra whitespace ===")
    resp = app_service.search_text("  machine   learning  ", top_k=3)
    print(f"Original: {resp.original_query!r}")
    print(f"Processed: {resp.processed_query!r}")
    print(f"Was modified: {resp.metadata.was_modified}")
    print(f"Was corrected: {resp.metadata.was_corrected}")
    print(f"Results: {len(resp.results)}")
    for hit in resp.results[:2]:  # show first two
        print(f"  Rank {hit.rank}: {hit.title}")
    print()

    # Case 3: Query processing disabled
    print("=== Case 3: Query processing disabled ===")
    resp = app_service.search_text("machine learning", top_k=3, query_processing=False)
    print(f"Original: {resp.original_query!r}")
    print(f"Processed: {resp.processed_query!r}")
    print(f"Query processing enabled: {resp.metadata.query_processing_enabled}")
    print(f"Results: {len(resp.results)}")
    for hit in resp.results:
        print(f"  Rank {hit.rank}: {hit.title}")
    print()

    # Case 4: Query processing enabled
    print("=== Case 4: Query processing enabled ===")
    resp = app_service.search_text("machine learning", top_k=3, query_processing=True)
    print(f"Original: {resp.original_query!r}")
    print(f"Processed: {resp.processed_query!r}")
    print(f"Query processing enabled: {resp.metadata.query_processing_enabled}")
    print(f"Results: {len(resp.results)}")
    for hit in resp.results:
        print(f"  Rank {hit.rank}: {hit.title}")
    print()

    # Case 5: top_k below available results
    print("=== Case 5: top_k below available results ===")
    resp = app_service.search_text("engine", top_k=2)
    print("Requested top_k: 2")
    print(f"Returned: {len(resp.results)}")
    print(f"Requested (pre-clamp): {resp.metadata.requested_top_k}")
    print(f"Returned count: {resp.metadata.returned_count}")
    print(f"Truncated: {resp.metadata.truncated}")
    for hit in resp.results:
        print(f"  Rank {hit.rank}: {hit.title}")
    print()

    # Case 6: top_k above available documents
    print("=== Case 6: top_k above available documents ===")
    resp = app_service.search_text("engine", top_k=1000)
    print("Requested top_k: 1000")
    print(f"Returned: {len(resp.results)}")
    print(f"Requested (pre-clamp): {resp.metadata.requested_top_k}")
    print(f"Returned count: {resp.metadata.returned_count}")
    print(f"Corpus exhausted: {resp.metadata.corpus_exhausted}")
    print(f"Truncated: {resp.metadata.truncated}")
    # Show all results if corpus is small, else first few
    if len(resp.results) <= 10:
        for hit in resp.results:
            print(f"  Rank {hit.rank}: {hit.title}")
    else:
        for hit in resp.results[:5]:
            print(f"  Rank {hit.rank}: {hit.title}")
        print(f"  ... and {len(resp.results) - 5} more")
    print()

    # Case 7: Invalid/empty query rejection
    print("=== Case 7: Invalid/empty query rejection ===")
    for bad_query in ["", "   ", "\t\n"]:
        try:
            app_service.search_text(bad_query, top_k=5)
            print(f"ERROR: Query {bad_query!r} was not rejected")
        except Exception as e:
            print(f"Query {bad_query!r} -> {type(e).__name__}: {e}")
    print()

    # Case 8: A query where BM25 and semantic retrieval differ
    print("=== Case 8: Query where BM25 and semantic may differ ===")
    query = "how a muscle pump moves blood"
    resp = app_service.search_text(query, top_k=5)
    print(f"Query: {resp.original_query!r}")
    print(f"Processed: {resp.processed_query!r}")
    print(f"Results: {len(resp.results)}")
    for hit in resp.results:
        print(f"  Rank {hit.rank}: {hit.title}")
        for entry in hit.provenance:
            print(f"      {entry.retriever:<9} rank={entry.rank:<3} raw={entry.raw_score:.4f}")
    print()

    # Agreement is not a quality measure. It is a description of what the two
    # retrievers did, and it is here to show the RRF fusion is actually fusing
    # two differing lists rather than echoing one of them.
    bm25_only = sum(1 for h in resp.results if h.sources == ("bm25",))
    semantic_only = sum(1 for h in resp.results if h.sources == ("semantic",))
    agreed = sum(1 for h in resp.results if len(h.sources) == 2)
    print(f"Retriever agreement over the {len(resp.results)} returned documents:")
    print(f"  found by both:  {agreed}")
    print(f"  BM25 only:      {bm25_only}")
    print(f"  semantic only:  {semantic_only}")

    # Where the two disagree most is the honest demonstration: the per-retriever
    # rank each retriever gave a document, which provenance preserves.
    reordered = [
        (h, h.provenance)
        for h in resp.results
        if {p.rank for p in h.provenance} and len({p.rank for p in h.provenance}) > 1
    ]
    if reordered:
        print("\nDocuments the two retrievers ranked differently (RRF placed them):")
        for hit, provenance in reordered:
            ranks = {p.retriever: p.rank for p in provenance}
            print(f"  Rank {hit.rank}: {hit.title}")
            print(f"      bm25 said {ranks.get('bm25')}, semantic said {ranks.get('semantic')}")

    print()
    print("Note: This shows only that the two retrievers can produce different")
    print("orderings, and that RRF fuses them. It is NOT evidence that either one")
    print("is better. Ranking quality has not been evaluated against labelled")
    print("relevance judgments, and nothing here should be read as a quality claim.")


if __name__ == "__main__":
    main()
