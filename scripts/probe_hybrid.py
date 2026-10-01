"""Run a query through BM25, semantic, and hybrid RRF side by side.

This is a development tool, not part of the API surface. ``probe_index.py``
shows one retriever at a time, which is enough to see a ranking but not enough
to see *where two rankings disagree* -- and disagreement is the entire input to
fusion. This prints all three in one pass, with the per-retriever provenance
that makes a fused position explicable: which retrievers found the document,
at what rank, with what raw score, and what each contributed to the total.

It shows no relevance judgement. Whether a document is a *good* answer is a
labelled-evaluation question (Phase 8) and is not answerable by looking. What
is answerable by looking is the mechanical one: does the fused order follow the
fused scores, is provenance attached to every row, and do the retrievers
actually disagree.

Usage::

    python scripts/probe_hybrid.py "how a muscle pump moves blood"
    python scripts/probe_hybrid.py "reflectivity of the moon" --top-k 5
    python scripts/probe_hybrid.py "blood flow" --rrf-k 10 --candidate-pool 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The corpus is Wikipedia, so titles contain characters a cp1252 Windows console
# cannot encode -- 'Ælfheah of Canterbury' and Park Güell both crash the default
# stdout. Reconfigure rather than strip: the title is the point of the line, and
# silently replacing characters with '?' would make a probe disagree with the
# documents it is probing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from semantic_search.core.config import load_settings
from semantic_search.core.logging import configure_logging
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid import BM25, SEMANTIC, HybridSearchService
from semantic_search.search.semantic import SemanticSearchService


def _block(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _bm25_block(service: HybridSearchService, query: str, candidate_k: int, show: int) -> None:
    _block(f"BM25  (lexical, asked for {candidate_k}, showing {show})")
    assert service.bm25 is not None
    hits = service.bm25.search(query, top_k=candidate_k)
    if not hits:
        print("  (no document contained any query term -- see the HYBRID note below)")
        return
    for hit in hits[:show]:
        print(f"  {hit.rank + 1:>2}. [{hit.score:7.3f}] {service.store[hit.row].title}")
        if hit.matched_terms:
            print(f"        matched: {', '.join(hit.matched_terms)}")
    if len(hits) > show:
        print(f"  ... {len(hits) - show} more, of {len(hits)} BM25 candidates")


def _semantic_block(service: HybridSearchService, query: str, candidate_k: int, show: int) -> None:
    _block(f"SEMANTIC  (dense, asked for {candidate_k}, showing {show})")
    assert service.semantic is not None
    hits = service.semantic.search(query, top_k=candidate_k)
    for hit in hits[:show]:
        print(f"  {hit.rank + 1:>2}. [{hit.score:7.3f}] {service.store[hit.row].title}")
    if len(hits) > show:
        print(f"  ... {len(hits) - show} more, of {len(hits)} semantic candidates")


def _hybrid_block(service: HybridSearchService, query: str, top_k: int) -> None:
    _block(f"HYBRID-RRF  (fused, k={service.config.rrf_k}, top {top_k})")
    results = service.search(query, top_k=top_k)
    for result in results:
        marks = "+".join(result.retrievers)
        print(f"  {result.rank + 1:>2}. [rrf {result.rrf_score:.6f}] {result.title}   [{marks}]")
        # One line per retriever that found it. A retriever that did *not* find
        # the document is not listed: it contributed nothing, and printing a
        # zero would read like a real score.
        for hit in result.hits:
            print(
                f"        {hit.retriever:<9} rank {hit.rank + 1:>3}"
                f"  raw {hit.raw_score:7.3f}  contribution {hit.rrf_contribution:.6f}"
            )
    if not results:
        print("  (no results)")
        return
    both = sum(1 for r in results if r.agreement == 2)
    print(f"\n  {both}/{len(results)} found by both retrievers.")
    print("  Read the ranks, not the titles: where the two retrievers disagree,")
    print("  that is the signal fusion is built on. Whether any of these are")
    print("  *good* answers is not visible here -- that needs labelled judgements.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="+", help="the query text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=None, help="override hybrid.rrf_k")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=None,
        help="override hybrid.candidate_pool (how deep each retriever is asked to go)",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    settings = load_settings(environment="development")
    if args.rrf_k is not None:
        settings.hybrid.rrf_k = args.rrf_k
    if args.candidate_pool is not None:
        settings.hybrid.candidate_pool = args.candidate_pool
    configure_logging("WARNING")

    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not store:
        print("error: corpus is empty; run scripts/download_data.py", file=sys.stderr)
        return 1

    # The two retrievers must share one DocumentStore, or they are numbering
    # different documents. from_settings builds exactly that arrangement.
    index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, settings.bm25)
    service = HybridSearchService(
        BM25Searcher(index, store),
        SemanticSearchService.from_settings(settings, store=store),
        hybrid_config=settings.hybrid,
        search_config=settings.search,
    )

    candidate_k = max(args.top_k, service.config.candidate_pool)
    print(f'query: "{query}"')
    print(f"corpus: {len(store)} documents")
    print(f"retrievers: {', '.join(service.retriever_names)}")
    print(f"candidate depth: {candidate_k} per retriever -> {args.top_k} shown")

    _bm25_block(service, query, candidate_k, args.top_k)
    _semantic_block(service, query, candidate_k, args.top_k)
    _hybrid_block(service, query, args.top_k)

    assert service.bm25 is not None and service.semantic is not None
    print(f"\nretriever names used in provenance: {BM25!r}, {SEMANTIC!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
