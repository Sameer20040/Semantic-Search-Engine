"""Run the real 300-document corpus with query processing OFF and ON, and diff them.

This is a **diagnostic, not an evaluation**. Every number it prints is a
statement about what changed, never about what got better. There are no
labelled relevance judgements in this project, so the question "are the ON
results more relevant than the OFF results" cannot be answered here and is not
answered here.

What *can* be answered by running the thing, and is:

* **How often does processing change the query at all?** Most of the time it
  should not. A pipeline that rewrites most queries is a pipeline that is
  guessing most of the time.
* **When it does change the query, what did it do?** Every rewrite is printed
  with its before, its after, and its confidence, so the decision is auditable
  rather than inferred from a ranking difference.
* **How often does a changed query change the ranking?** Two queries that
  differ can still return the same documents, and that is worth knowing.
* **Which queries did processing silently refuse?** The ones with a typo in
  them and no correction recorded. These are the misses, and they are listed
  because a list of successes hides them.

The query set is fixed, not random, so a run is comparable with the next one.
Roughly a third of the queries carry deliberate typos.

Usage::

    python scripts/query_processing_experiment.py
    python scripts/query_processing_experiment.py --top-k 10
    python scripts/query_processing_experiment.py --no-spell-check
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from semantic_search.core.config import load_settings
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.query.service import SearchService
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: Fixed so runs are comparable with each other. The split is deliberate and is
#: the point of the experiment: CLEAN queries should be left alone, and TYPO
#: queries should be corrected. If the clean group is being rewritten, the
#: corrector is doing damage -- and that is the failure this experiment exists
#: to make visible, so the clean queries are ordinary English, not corpus
#: vocabulary, which is the harder case.
CLEAN_QUERIES: Sequence[str] = (
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
)

TYPO_QUERIES: Sequence[str] = (
    "how a muscle pumpe moves blood",
    "the water cycle and evaporaton",
    "structure of DNA and genetice inheritance",
    "how vaccines train the immune systm",
    "black holes and gravitaional fields",
    "why the sky apears blue",
    "the history of the Roman Empie",
    "how eartquakes are measured",
    "origins of the industrail revolution",
    "the composition of the Earths atmosphere",
)

#: Correctly spelled, and in the 300-document corpus vocabulary. These are the
#: words a *typical* search engine user does not type and this corpus cannot
#: rescue. They are here to measure a specific risk: an out-of-corpus real word
#: being rewritten into a rarer corpus word. `tectonics` and `turbine` were
#: chosen after that was observed happening, not before.
OOV_QUERIES: Sequence[str] = (
    "tectonic plate boundaries",
    "gas turbine power",
    "the virus replication cycle",
    "resistor and capacitor circuits",
    "a transistor radio receiver",
    "enzyme catalysis in the gut",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-spell-check",
        action="store_true",
        help="run with the shipped configuration (processing on, spell check off)",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    settings = load_settings(environment="development")
    settings.query.enabled = True
    settings.query.spell_check = not args.no_spell_check

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
    service = SearchService.from_settings(settings, store, hybrid)

    groups = [
        ("CLEAN  (correctly spelled, ordinary English)", CLEAN_QUERIES),
        ("TYPO   (deliberate misspellings)", TYPO_QUERIES),
        ("OOV    (real words absent from the corpus)", OOV_QUERIES),
    ]

    print(f"corpus            {len(store)} documents")
    print(
        f"configuration     spell_check={settings.query.spell_check}  "
        f"synonym_expansion={settings.query.synonym_expansion}  "
        f"min_score={settings.query.min_correction_score}  "
        f"max_edit={settings.query.min_edit_distance}  "
        f"min_df={settings.query.min_candidate_df}"
    )
    print(f"top_k             {args.top_k}")
    print(f"vocabulary        {service.info()['vocabulary_size']:,} terms")
    print("\nEvery row below is a CHANGE, not an improvement. There are no relevance")
    print("labels in this project, so 'better' is not a thing these numbers can say.")

    totals = {"queries": 0, "changed_query": 0, "changed_rank": 0, "corrections": 0}

    for label, queries in groups:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        group = {"changed_query": 0, "changed_rank": 0, "corrections": 0}
        for query in queries:
            off = service.search_raw(query, top_k=args.top_k)
            on = service.search(query, top_k=args.top_k)

            off_ids = [r.doc_id for r in off.results]
            on_ids = [r.doc_id for r in on.results]
            query_changed = off.query.query_for_retrieval != on.query.query_for_retrieval
            rank_changed = off_ids != on_ids

            print(f"\n  query   {query!r}")
            if on.query.corrections:
                for correction in on.query.corrections:
                    print(
                        f"  rewrite {correction.from_term!r} -> {correction.to_term!r}"
                        f"  (similarity {correction.confidence:.1f})"
                    )
                    group["corrections"] += 1
            elif query_changed:
                print(
                    f"  rewrite {off.query.query_for_retrieval!r} -> "
                    f"{on.query.query_for_retrieval!r}  (no correction recorded)"
                )
            else:
                print("  rewrite none")

            if rank_changed:
                shared = len(set(off_ids) & set(on_ids))
                print(f"  ranking  CHANGED  ({shared}/{len(on_ids)} documents in both)")
                on_positions = {doc_id: i for i, doc_id in enumerate(on_ids, start=1)}
                for result in on.results:
                    was = off_ids.index(result.doc_id) + 1 if result.doc_id in off_ids else None
                    print(
                        f"    {result.rank + 1:>2}. {result.title}"
                        f"   ({'new' if was is None else f'was {was}'})"
                    )
                for result in off.results:
                    if result.doc_id not in on_positions:
                        print(f"    -- {result.title}   (dropped out of the top {args.top_k})")
            else:
                print(f"  ranking  unchanged  (same {len(on_ids)} documents, same order)")

            group["changed_query"] += int(query_changed)
            group["changed_rank"] += int(rank_changed)
            totals["queries"] += 1
            totals["changed_query"] += int(query_changed)
            totals["changed_rank"] += int(rank_changed)

        print(
            f"\n  -- {label}: "
            f"{group['changed_query']}/{len(queries)} queries rewritten, "
            f"{group['corrections']} corrections, "
            f"{group['changed_rank']}/{len(queries)} rankings changed"
        )
        totals["corrections"] += group["corrections"]

    print(f"\n{'=' * 78}\nTOTALS\n{'=' * 78}")
    print(f"  queries                       {totals['queries']}")
    print(f"  queries whose text changed    {totals['changed_query']}")
    print(f"  corrections applied           {totals['corrections']}")
    print(f"  rankings changed              {totals['changed_rank']}")
    print("\nRead the CLEAN and OOV rows before the TYPO ones. The TYPO row is the")
    print("feature working; the other two are the feature costing something. A")
    print("corrector that rewrites clean queries is doing damage, and this table")
    print("is the only thing here that would reveal it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
