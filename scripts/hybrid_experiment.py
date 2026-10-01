"""Run a fixed query set through BM25, semantic, and hybrid on the real corpus.

Diagnostic, not an evaluation. This prints the three rankings side by side so
that *where the retrievers disagree* is visible; it cannot say whether the
fused result is better, because there are no relevance labels here. A query
whose BM25 and semantic top-5 sets are identical tells you nothing about
fusion, and one whose sets are disjoint tells you fusion had disagreement to
work with. Both are worth seeing; neither is a quality result.

The number worth watching is the overlap column. If the two retrievers already
agree on nearly everything, fusing them is close to a no-op and any apparent
improvement would be noise. If they disagree everywhere, fusion is doing real
work. Where the truth lies is a property of the corpus and the model, and this
script measures it rather than assuming it.

Usage::

    python scripts/hybrid_experiment.py
    python scripts/hybrid_experiment.py --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: Fixed so the run is reproducible. Deliberately mixed: some queries are
#: keyword-shaped (lexical-friendly), some are natural-language questions
#: (dense-friendly), and one -- "how a muscle pump moves blood" -- was recorded
#: in Phase 6 as a case where the two retrievers have visibly different failure
#: modes. That is the only claim this query supports.
QUERIES = [
    "how a muscle pump moves blood",
    "the water cycle and evaporation",
    "structure of DNA and genetic inheritance",
    "how vaccines train the immune system",
    "black holes and gravitational fields",
    "why the sky appears blue",
    "the history of the Roman Empire",
    "how earthquakes are measured",
    "origins of the industrial revolution",
    "the composition of the Earth's atmosphere",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-fused", type=int, default=3, help="fused rows printed per query")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")
    store = DocumentStore.load_jsonl(settings.data.corpus_file)
    if not store:
        print("error: corpus is empty; run scripts/download_data.py", file=sys.stderr)
        return 1

    index = BM25Index.load(settings.data.index_dir / settings.bm25.index_filename, settings.bm25)
    service = HybridSearchService(
        BM25Searcher(index, store),
        SemanticSearchService.from_settings(settings, store=store),
        hybrid_config=settings.hybrid,
        search_config=settings.search,
    )
    candidate_k = max(args.top_k, service.config.candidate_pool)
    top_k = args.top_k

    print(f"corpus: {len(store)} documents | model: {settings.embedding.model_name}")
    print(f"top_k: {top_k} | candidate depth: {candidate_k} | rrf_k: {service.config.rrf_k}\n")

    overlaps: list[float] = []
    for query in QUERIES:
        assert service.bm25 is not None and service.semantic is not None
        started = time.perf_counter()
        lexical = service.bm25.search(query, top_k=candidate_k)
        dense = service.semantic.search(query, top_k=candidate_k)
        fused = service.search(query, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000

        lex_ids = [hit.doc_id for hit in lexical[:top_k]]
        dense_ids = [hit.doc_id for hit in dense[:top_k]]
        # Overlap of the two *top-k* lists, which is the set fusion can reorder
        # within. Deeper candidates also overlap, but those documents compete
        # for the top k, so the top-k overlap is the honest predictor of how
        # much fusion can change the answer.
        overlap = len(set(lex_ids) & set(dense_ids))
        overlaps.append(overlap / top_k)

        print(f'QUERY: "{query}"')
        print(f"  BM25 top-{top_k}: " + ", ".join(store[store.row_of(i)].title for i in lex_ids))
        print(f"  SEM  top-{top_k}: " + ", ".join(store[store.row_of(i)].title for i in dense_ids))
        print(f"  HYBRID (RRF k={service.config.rrf_k}):")
        for result in fused[: args.show_fused]:
            marks = "+".join(result.retrievers)
            print(f"    {result.rank + 1}. [{result.rrf_score:.6f}] {result.title}  [{marks}]")
        print(
            f"  top-{top_k} overlap: {overlap}/{top_k}   "
            f"agreement in fused top-{min(3, len(fused))}: "
            f"{sum(1 for r in fused[:3] if r.agreement == 2)}/3   "
            f"({elapsed:.0f} ms for all three)"
        )
        print()

    mean_overlap = sum(overlaps) / len(overlaps)
    print(f"mean top-{top_k} overlap between the two retrievers: {mean_overlap * 100:.1f}%")
    print("Diagnostic only. This says how much the retrievers disagree; it says")
    print("nothing about whether the fused ranking is more relevant, which needs")
    print("labelled judgements that this phase does not have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
