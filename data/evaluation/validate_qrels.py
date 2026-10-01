"""Validate a human-submitted qrels file for the topical stratum.

Run this against every qrels file before it is merged into ``qrels.jsonl``. It
exists because the failure mode it catches is otherwise invisible: a qrels file
with a typo in a ``doc_id`` or a duplicated pair loads without complaint and
silently lowers a system's measured recall.

Usage::

    python data/evaluation/validate_qrels.py data/evaluation/qrels/qrels_b_filled.jsonl
    python data/evaluation/validate_qrels.py --strict-missing   # also require every query judged

Exit code is ``0`` when the file is usable and ``1`` when it is not, so it can
gate a commit or a CI job.

The checks, and why each one earns its place:

* ``unknown_query_id``   -- the query no longer exists, or the id has a typo.
* ``unknown_doc_id``     -- the judgment points outside the 300-document corpus,
  which would make the query unanswerable by construction.
* ``duplicate_pair``     -- two grades for one (query, document) pair. The file
  no longer says what that document is worth, and which of the two wins is
  arbitrary.
* ``invalid_grade``      -- outside the documented 0-3 scale.
* ``missing_judgment``   -- a query in the worksheet with no row in the file.
* ``unjudged_in_pool``   -- a document that was in the pool and in the worksheet
  but has no grade. Silence here is the dangerous case: an ungraded document is
  scored 0 by the metrics, exactly like a judged non-relevant one.
* ``missing_judge_field``-- no ``judged_by``, so the single-judge provenance
  the report states is not actually recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.data.loader import DocumentStore

VALID_GRADES = {0, 1, 2, 3}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Parse JSONL, reporting the line number of the first bad row."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"  MALFORMED  line {number}: {exc.msg}")
                raise SystemExit(1) from exc
            if not isinstance(payload, dict):
                print(f"  MALFORMED  line {number}: expected an object, got a list")
                raise SystemExit(1)
            rows.append(payload)
    return rows


def validate(
    rows: list[dict[str, object]],
    pool_path: Path,
    known_docs: set[str],
    *,
    strict_missing: bool,
) -> dict[str, object]:
    """Run every check and return a machine-readable summary."""
    pool_queries: set[str] = set()
    pool_candidates: dict[str, set[str]] = {}
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Only the topical stratum is judged by this protocol. The known-item
            # queries have their judgments already, derived from the article the
            # question was written about, and demanding pool judgments for them
            # here would be asking for a second and incompatible ground truth.
            if row.get("stratum") != "B_topical":
                continue
            qid = str(row.get("query_id", ""))
            pool_queries.add(qid)
            pool_candidates[qid] = {str(c.get("doc_id")) for c in row.get("candidates", [])}

    judged: dict[str, dict[str, int]] = {}
    errors: dict[str, list[str]] = {}
    seen_pairs: dict[tuple[str, str], int] = {}

    for row in rows:
        qid = str(row.get("query_id", ""))
        did = str(row.get("doc_id", ""))
        grade = row.get("relevance_grade")

        if qid not in pool_queries:
            errors.setdefault("unknown_query_id", []).append(qid)
            continue
        if did not in known_docs:
            errors.setdefault("unknown_doc_id", []).append(did)
            continue
        if not isinstance(grade, int) or grade not in VALID_GRADES:
            errors.setdefault("invalid_grade", []).append(f"{qid}/{did}={grade!r}")
            continue
        if not row.get("judged_by"):
            errors.setdefault("missing_judge_field", []).append(f"{qid}/{did}")

        pair = (qid, did)
        if pair in seen_pairs:
            errors.setdefault("duplicate_pair", []).append(f"{qid}/{did}")
            continue
        seen_pairs[pair] = grade
        judged.setdefault(qid, {})[did] = grade

    # A judged document that is not in the pool for that query. Usually a
    # transcription slip, and always suspicious.
    for qid, did in seen_pairs:
        pool_for_q = pool_candidates.get(qid, set())
        if pool_for_q and did not in pool_for_q:
            errors.setdefault("doc_not_in_pool", []).append(f"{qid}/{did}")

    missing = sorted(pool_queries - set(judged))
    if missing and strict_missing:
        errors.setdefault("missing_judgment", []).extend(missing)

    # Anything that was in the worksheet for a query that WAS otherwise judged.
    unjudged: list[str] = []
    for qid, docs in judged.items():
        pool_for_q = pool_candidates.get(qid, set())
        unjudged.extend(f"{qid}/{did}" for did in sorted(pool_for_q - set(docs)))

    return {
        "n_rows": len(rows),
        "n_queries_judged": len(judged),
        "n_queries_in_pool": len(pool_queries),
        "queries_with_no_rows": missing,
        "unjudged_pooled": unjudged,
        "errors": errors,
        "ok": not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a human-judged topical qrels file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("qrels", type=Path, help="the filled qrels JSONL to validate")
    parser.add_argument(
        "--pool", type=Path, default=Path("data/evaluation/derived/pool.jsonl"), help="pool.jsonl"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/corpus.jsonl"),
        help="corpus.jsonl, for the document-id check",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="treat a query with no judgments at all as an error",
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.qrels)
    store = DocumentStore.load_jsonl(args.corpus)
    result = validate(rows, args.pool, set(store.document_ids), strict_missing=args.strict_missing)

    print(f"qrels        {args.qrels}")
    print(f"rows         {result['n_rows']}")
    print(
        f"queries      {result['n_queries_judged']} judged of {result['n_queries_in_pool']} pooled"
    )
    print(f"unjudged pooled docs (scored 0): {len(result['unjudged_pooled'])}")

    if result["ok"]:
        print("\nVALID: no structural errors found.")
        if result["queries_with_no_rows"]:
            print(
                "  note: some pooled queries have no rows at all. They will be "
                "unanswerable and will not be reported as system failures."
            )
        print("  note: unjudged pooled documents are counted as non-relevant by the")
        print("        metrics. This is a known depression of measured recall.")
        return 0

    print("\nINVALID:")
    for name, problems in result["errors"].items():
        preview = ", ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        print(f"  {name}: {len(problems)}  e.g. {preview}{more}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
