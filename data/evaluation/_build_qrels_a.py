# -*- coding: utf-8 -*-
"""Write the relevance judgments for the frozen Tier A queries.

The judgment is *known-item*: the source article is relevant because the query
was written about it, before any retriever ran. Nothing here is derived from a
ranking. The one thing that would contaminate this file is adding a judgment
after seeing which system ranked the document highly, and nothing in this
workflow does that.

Grades are binary (1 = relevant). No graded relevance is invented: there is no
documented protocol for distinguishing "relevant" from "highly relevant" for a
single-judge author working from 300 Wikipedia leads, and inventing one would
produce numbers with no interpretation behind them.

Judged-by is recorded as a single identifier. There is exactly one judge, so
there is no inter-annotator agreement to report, and the report says so rather
than implying a second pass happened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.data.loader import DocumentStore  # noqa: E402

JUDGE = "author-single-judge"


def main() -> int:
    queries_path = Path("data/evaluation/queries/queries.jsonl")
    rows = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
    tier_a = [r for r in rows if r["stratum"].startswith("A_")]
    if not tier_a:
        print("ERROR: no stratum A queries found")
        return 1

    store = DocumentStore.load_jsonl(Path("data/processed/corpus.jsonl"))
    known = set(store.document_ids)

    out = Path("data/evaluation/qrels/qrels.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sorted(tier_a, key=lambda r: r["query_id"]):
            doc_id = row["source_doc_id"]
            if doc_id not in known:
                print(f"ERROR: {row['query_id']} source doc {doc_id} is not in the corpus")
                return 1
            handle.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "doc_id": doc_id,
                        "relevance_grade": 1,
                        "judged_by": JUDGE,
                        "protocol": "known_item",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(tier_a)} judgments -> {out}")

    # Record the hash of the query file. queries.jsonl is frozen from this point;
    # the manifest carries this digest so a later edit is detectable rather than
    # silent.
    digest = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    Path("data/evaluation/queries/queries.sha256").write_text(f"{digest}\n", encoding="utf-8")
    print(f"queries.jsonl sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
