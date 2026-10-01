"""Run a query through processing OFF and ON, and show what changed.

This answers one mechanical question, which is the only one that can be
answered by looking: **what did processing do to this query, and did the
retrieval query change as a result?** It prints the record the processor built --
original, normalized, corrected, the string that reached retrieval, the
corrections with their scores, the expansions and whether they were applied --
and then the rankings from both paths side by side.

It shows no relevance judgement. A change in the top 5 is a *change*, and this
script reports it as a change. Whether either ranking is better needs labelled
data, which this project does not have, so nothing here claims improvement.

The one-string guarantee is checked, not asserted in prose: the script wraps
each retriever, records the exact string it was handed, and prints PASS/FAIL.
The specification requires that BM25 and semantic never receive differently
rewritten queries, and that is a structural property of
``HybridSearchService``'s fan-out -- but a structural property is worth
watching, and this is where it would show up if it broke.

Note that spell correction ships **off**. A probe with ``--spell-check`` turns
it on for that run; the measurement behind the default is in
``BUILD_STATUS.md``, Phase 8, and it is not a good trade for a general
vocabulary. Run it on the same query to see the rewrites it would have made.

Usage::

    python scripts/probe_query_processing.py "machne learning"
    python scripts/probe_query_processing.py "llm architecture" --spell-check
    python scripts/probe_query_processing.py "machine learning" --top-k 5
    python scripts/probe_query_processing.py "machne learning" --no-semantic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Wikipedia titles contain characters a cp1252 console cannot encode. Reconfigure
# rather than strip: the title is the point of the line, and replacing characters
# with '?' would make a probe disagree with the documents it is probing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.query.service import SearchService
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService


def _block(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _query_block(response, show_expansions: bool = True) -> None:
    query = response.query
    _block("QUERY RECORD")
    print(f"  original            {query.original!r}")
    print(f"  normalized          {query.normalized!r}")
    print(f"  after spelling      {query.corrected!r}")
    print(f"  sent to retrieval   {query.query_for_retrieval!r}")
    print(f"  processing enabled  {query.enabled}")
    print(f"  modified / corrected {query.was_modified} / {query.was_corrected}")

    if query.corrections:
        print("\n  corrections applied:")
        for correction in query.corrections:
            print(
                f"    {correction.from_term!r} -> {correction.to_term!r}"
                f"   similarity {correction.confidence:.1f}"
            )
    else:
        print("\n  corrections: none (nothing was rewritten -- not a claim that")
        print("               nothing *could* have been, just that nothing was)")

    if not show_expansions:
        return
    if query.expansions:
        print("\n  expansions:")
        for expansion in query.expansions:
            status = "APPLIED to the retrieval query" if expansion.applied else "recorded only"
            print(f"    {expansion.from_term!r} -> {', '.join(expansion.to_terms)}  ({status})")
    else:
        print("  expansions: none")


def _ranking_block(title: str, response, top_k: int) -> list[tuple[str, str]]:
    """Print a ranking; return ``(doc_id, title)`` so the diff can be read."""
    _block(title)
    results = response.results
    if not results:
        print("  (no results)")
        return []
    for result in results[:top_k]:
        marks = "+".join(result.retrievers)
        print(f"  {result.rank + 1:>2}. [rrf {result.rrf_score:.6f}] {result.title}   [{marks}]")
    return [(result.doc_id, result.title) for result in results[:top_k]]


def _capture_retriever_inputs(hybrid: HybridSearchService, query: str) -> dict[str, str]:
    """Run one search and record the exact string each retriever was handed.

    Returns a mapping of retriever name -> query string. The retrievers are
    wrapped, not mocked: the real BM25 and the real dense index still do the
    work, so this observes the production path rather than a stand-in.
    """
    seen: dict[str, str] = {}
    bm25 = hybrid.bm25
    semantic = hybrid.semantic
    if bm25 is None or semantic is None:
        return seen

    original_bm25 = bm25.search
    original_semantic = semantic.search

    def bm25_spy(q: str, *args, **kwargs):
        seen["bm25"] = q
        return original_bm25(q, *args, **kwargs)

    def semantic_spy(q: str, *args, **kwargs):
        seen["semantic"] = q
        return original_semantic(q, *args, **kwargs)

    bm25.search = bm25_spy  # type: ignore[method-assign]
    semantic.search = semantic_spy  # type: ignore[method-assign]
    try:
        hybrid.search(query, top_k=10)
    finally:
        bm25.search = original_bm25  # type: ignore[method-assign]
        semantic.search = original_semantic  # type: ignore[method-assign]
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="+", help="the query text, unprocessed")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--spell-check",
        action="store_true",
        help="turn spell correction on for this run (it ships off)",
    )
    parser.add_argument(
        "--synonyms",
        action="store_true",
        help="apply synonym expansion to the retrieval query (it ships record-only)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="BM25 only; skips the one-string check, which needs both retrievers",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")
    settings.query.enabled = True
    settings.query.spell_check = args.spell_check
    settings.query.synonym_expansion = args.synonyms

    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not store:
        print("error: corpus is empty; run scripts/download_data.py", file=sys.stderr)
        return 1

    index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, settings.bm25)
    bm25 = BM25Searcher(index, store)
    semantic = (
        None if args.no_semantic else SemanticSearchService.from_settings(settings, store=store)
    )
    hybrid = HybridSearchService(
        bm25, semantic, hybrid_config=settings.hybrid, search_config=settings.search
    )
    service = SearchService.from_settings(settings, store, hybrid)

    print(f'query:   "{query}"')
    print(f"corpus:  {len(store)} documents")
    print(
        f"modes:   spell_check={settings.query.spell_check}  "
        f"synonym_expansion={settings.query.synonym_expansion}  "
        f"enabled={settings.query.enabled}"
    )
    if args.spell_check:
        print("         note: spell correction is ON for this run. It ships off; see")
        print("         BUILD_STATUS.md Phase 8 for the measurement behind that.")

    # ---- OFF: the always-available baseline ------------------------------- #
    raw = service.search_raw(query, top_k=args.top_k)
    _query_block(raw)
    raw_ids = _ranking_block(
        "PROCESSING OFF  (search_raw, query exactly as typed)", raw, args.top_k
    )

    # ---- ON -------------------------------------------------------------- #
    processed = service.search(query, top_k=args.top_k)
    _query_block(processed)
    on_ids = _ranking_block("PROCESSING ON   (search, query as processed)", processed, args.top_k)

    # ---- Did processing change anything? --------------------------------- #
    _block("DIFFERENCE")
    if processed.query.query_for_retrieval == raw.query.query_for_retrieval:
        print("  the retrieval query was unchanged, so the two rankings are the same")
        print("  query answered twice. Processing ran; it had nothing to do.")
    else:
        print(f"  retrieval query: {raw.query.query_for_retrieval!r}")
        print(f"              ->  {processed.query.query_for_retrieval!r}")
    if raw_ids == on_ids:
        print(f"  top {len(raw_ids)}: identical, in the same order")
    else:
        raw_positions = {doc_id: i for i, (doc_id, _) in enumerate(raw_ids, start=1)}
        on_positions = {doc_id: i for i, (doc_id, _) in enumerate(on_ids, start=1)}
        print(
            f"  top {len(on_ids)}: the ranking changed. "
            f"({len(set(raw_positions) & set(on_positions))} documents in both)"
        )
        for doc_id, title in on_ids:
            was = raw_positions.get(doc_id)
            print(
                f"    {on_positions[doc_id]:>2}. {title}"
                f"   ({'new' if was is None else f'was {was}'})"
            )
        left = [
            f"{title} (was {raw_positions[doc_id]})"
            for doc_id, title in raw_ids
            if doc_id not in on_positions
        ]
        if left:
            print(f"  left the top {args.top_k}: {', '.join(left)}")
    print("\n  This is a change, reported as a change. Whether it is a change for the")
    print("  better is NOT measured -- there are no labelled relevance judgements.")

    # ---- One string, two retrievers -------------------------------------- #
    if not args.no_semantic:
        _block("ONE STRING, BOTH RETRIEVERS")
        seen = _capture_retriever_inputs(hybrid, processed.query.query_for_retrieval)
        for name in ("bm25", "semantic"):
            if name in seen:
                print(f"  {name:<9} received {seen[name]!r}")
        values = set(seen.values())
        if len(values) == 1 and seen:
            print("  PASS: both retrievers received the identical string.")
        else:
            print("  FAIL: the retrievers were handed different strings. This is the one")
            print("        guarantee the architecture exists to provide, and it is broken.")
            return 1

    # ---- What the /stats-shaped report would say ------------------------- #
    _block("CONFIGURATION")
    for key, value in service.info().items():
        print(f"  {key:<28} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
