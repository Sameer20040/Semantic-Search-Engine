"""Build the blind human-judging worksheet from a frozen pool.

The worksheet is the mechanism that keeps stratum B honest. Without it, the
only ways to fill ``qrels_b.jsonl`` are to have a person read the documents (the
protocol) or to derive the grades from a ranking (the thing the protocol
exists to prevent). This script makes the first option cheap without making the
second option look official.

What this script does NOT do, by design:

* it never assigns a relevance grade;
* it never names a retrieval system or a rank;
* it never reads a query's ``source_doc_id`` into anything a judge would treat
  as a hint of the answer.

What it does:

* reads ``pool.jsonl`` (one row per query, candidates already shuffled with a
  fixed seed, no system attribution);
* attaches each candidate's title and lead excerpt so the judge can decide
  whether the document *answers* the query without opening 80 files per query;
* writes a JSON template row and a printable Markdown worksheet.

The excerpt is the first ``--excerpt-chars`` characters of the document's lead,
whitespace-collapsed. It is shown to the judge verbatim, never scored or
matched against the query, so it cannot smuggle a relevance decision back in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.logging import configure_logging, get_logger
from semantic_search.data.loader import DocumentStore

logger = get_logger("judging_worksheet")

_WS = re.compile(r"\s+")


def collapse(text: str) -> str:
    """Collapse every run of whitespace to a single space.

    Wikipedia leads contain newlines, tabs, and non-breaking spaces. Rendered
    raw into a worksheet they produce ragged rows and make two identical-looking
    passages look different to a judge skimming for the answer.
    """
    return _WS.sub(" ", text).strip()


def excerpt_for(document, chars: int) -> str:
    """The lead of a document, collapsed and truncated on a word boundary.

    Falls back to the title when the document has no text, because a blank cell
    reads as "the judge skipped this" rather than "this document has no body",
    and those are different answers to the question the worksheet asks.
    """
    if document is None:
        return "(document not found)"
    body = collapse(document.text)
    if not body:
        body = collapse(document.title)
    if len(body) <= chars:
        return body
    cut = body[:chars]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut + "..."


def build_rows(
    pool_path: Path,
    corpus: Path,
    excerpt_chars: int,
    strata: set[str] | None,
    judge_depth: int,
) -> list[dict[str, object]]:
    """One worksheet row per query, with an excerpt on every candidate.

    ``judge_depth`` caps how many of a query's pooled candidates are shown.
    The pool is the union of six systems at depth 50, so it is 50-105 documents
    per query and judging all of it is not something a person does. Truncating
    is a real tradeoff and is not hidden: anything past the depth is unjudged,
    and the metrics will treat it as non-relevant, which depresses measured
    recall by an amount nobody can bound. The alternative -- a judge who quits
    at row four, or grades by skim -- is worse and is not measurable.
    """
    store = DocumentStore.load_jsonl(corpus)
    rows: list[dict[str, object]] = []
    truncated = 0
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pool_row = json.loads(line)
            if strata and pool_row.get("stratum") not in strata:
                continue
            candidates = []
            pooled = pool_row.get("candidates", [])
            if len(pooled) > judge_depth:
                truncated += len(pooled) - judge_depth
            for candidate in pooled[:judge_depth]:
                doc_id = candidate.get("doc_id", "")
                document = store.get(doc_id)
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "title": collapse(candidate.get("title", "") or doc_id),
                        "excerpt": excerpt_for(document, excerpt_chars),
                        "grade": "",  # BLANK. The judge fills this in.
                    }
                )
            rows.append(
                {
                    "query_id": pool_row["query_id"],
                    "stratum": pool_row.get("stratum", ""),
                    "query": collapse(pool_row.get("query", "")),
                    "recorded_answer": collapse(pool_row.get("recorded_answer", "")),
                    "source_in_pool": pool_row.get("source_in_pool"),
                    "candidates": candidates,
                }
            )
    if truncated:
        logger.info(
            "pool truncated for judging: %d candidates sit past the judging depth and "
            "will be unjudged",
            truncated,
        )
    return rows


def render_markdown(rows: list[dict[str, object]], judge_depth: int) -> str:
    """A worksheet a judge can actually work through without a JSON editor."""
    lines = [
        "# Stratum B judging worksheet",
        "",
        "Read the question and the recorded answer, then read each excerpt.",
        "",
        "Mark a candidate `1` if the excerpt **answers the question**, and `0`",
        "otherwise. A document can be completely on-topic and still be `0`.",
        "Judgment is by answer content, not by subject matter.",
        "",
        "No retrieval system, rank, or score is shown anywhere in this file.",
        "",
        f"Candidates shown per query: the first {judge_depth} of the pooled list",
        "in its fixed shuffled order. A relevant document that no system ranked",
        "in that window is unjudged here, and the metrics will count it as",
        "non-relevant. That is a known depression of measured recall, stated in",
        "`docs/evaluation_protocol.md`, not a claim that it does not exist.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['query_id']}")
        lines.append("")
        lines.append(f"**Query.** {row['query']}")
        lines.append("")
        lines.append(f"**Recorded answer.** {row['recorded_answer'] or '(none recorded)'}")
        lines.append("")
        lines.append("| doc_id | title | grade | excerpt |")
        lines.append("|---|---|---|---|")
        for candidate in row["candidates"]:  # type: ignore[index]
            excerpt = str(candidate["excerpt"]).replace("|", "\\|")
            lines.append(f"| `{candidate['doc_id']}` | {candidate['title']} |  | {excerpt} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the blind stratum-B judging worksheet from a frozen pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pool", type=Path, default=None, help="pool.jsonl path")
    parser.add_argument("--corpus", type=Path, default=None, help="corpus.jsonl path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the template and the Markdown worksheet",
    )
    parser.add_argument(
        "--excerpt-chars", type=int, default=600, help="lead characters shown per candidate"
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        default=["B_topical"],
        help="which strata to build a worksheet for (default: the topical stratum, "
        "which is the only one with an undecided judgment protocol)",
    )
    parser.add_argument(
        "--judge-depth",
        type=int,
        default=20,
        help="candidates shown per query, from the shuffled pool. The pool is the "
        "union of six arms at depth 50 and runs 50-105 documents per query; "
        "judging all of it is not a thing a person does.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    pool_path = args.pool or Path("data/evaluation/derived/pool.jsonl")
    corpus_path = args.corpus or Path("data/processed/corpus.jsonl")
    out_dir = args.out_dir or Path("data/evaluation/qrels")
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = set(args.strata) or None

    rows = build_rows(pool_path, corpus_path, args.excerpt_chars, strata, args.judge_depth)
    total_candidates = sum(len(r["candidates"]) for r in rows)  # type: ignore[arg-type]
    total_pooled = 0
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if strata and row.get("stratum") not in strata:
                continue
            total_pooled += len(row.get("candidates", []))

    template_path = out_dir / "qrels_b_template.jsonl"
    with template_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_path = out_dir / "judging_worksheet.md"
    md_path.write_text(render_markdown(rows, args.judge_depth), encoding="utf-8")

    print(f"worksheet {md_path}")
    print(f"template  {template_path}")
    print(f"queries   {len(rows)}")
    print(f"candidates {total_candidates} shown of {total_pooled} pooled")
    print("grades left blank on purpose; a judge fills them in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
