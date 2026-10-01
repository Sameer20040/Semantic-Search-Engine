"""Loading and validating the frozen evaluation dataset.

What this module guarantees, and why each guarantee is here
-----------------------------------------------------------

The evaluation's credibility rests entirely on the queries and judgments being
fixed before anyone runs a retriever. That is a property of the *process*, not
of any file, and no amount of schema validation can enforce it — but three
things can be checked afterwards, and they are:

1. **The queries file still hashes to the recorded digest.** If it does not, a
   query was edited after the freeze, and every number in the report is now
   describing a different experiment. This is the one check that would catch
   the failure mode that actually matters.
2. **The corpus still contains every document a judgment names.** A qrels file
   pointing at a document that no longer exists measures a corpus that is not
   the corpus under test.
3. **The internal shape is consistent** — unique query ids, every judged query
   present in the query file, at least one positive judgment per query.

Check 1 is a warning, not a hard failure, and the reason matters. A person who
legitimately extends the dataset should be able to re-freeze it and carry on,
rather than being told their file is corrupt. A person who quietly edits a query
after seeing a ranking gets a loud line in the report saying exactly that. Hard
failing would make the second person delete the check instead.

Unjudged documents
------------------
A document with no judgment is graded 0 by the metrics, and that is the only
policy. It means a retriever is never rewarded for a document nobody looked at,
and it means every number in the report sits on a floor of unexamined material.
:func:`pool_report` computes that floor per query so the report can state its
size instead of implying it is zero.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_search.core.exceptions import (
    EvaluationDatasetError,
    EvaluationDatasetNotFoundError,
)
from semantic_search.evaluation.models import (
    STRATA,
    EvalDataset,
    EvalQuery,
    RelevanceJudgment,
)

#: Written by the builders next to the queries they write.
DIGEST_FILENAME = "queries.sha256"


@dataclass(slots=True, frozen=True)
class DatasetPaths:
    """Where a dataset's three files live.

    ``root`` is normally ``data/evaluation``; overriding it is what lets the
    tests load a dataset built in a temporary directory instead of the real one.
    """

    root: Path

    @property
    def queries(self) -> Path:
        return self.root / "queries" / "queries.jsonl"

    @property
    def qrels(self) -> Path:
        return self.root / "qrels" / "qrels.jsonl"

    @property
    def digest_file(self) -> Path:
        return self.root / "queries" / DIGEST_FILENAME

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "queries": str(self.queries),
            "qrels": str(self.qrels),
            "digest_file": str(self.digest_file),
        }


@dataclass(slots=True, frozen=True)
class DatasetIntegrity:
    """The result of checking the frozen files against their recorded hash."""

    verified: bool
    expected: str
    actual: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "expected_sha256": self.expected,
            "actual_sha256": self.actual,
            "reason": self.reason,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dicts, reporting the offending line number.

    Line numbers matter here more than usual: a dataset is small, hand-edited,
    and a single malformed row in a frozen file needs to be findable without
    re-reading the whole file with a different tool.
    """
    if not path.exists():
        raise EvaluationDatasetNotFoundError(
            f"evaluation file not found: {path}. It is committed to the repository; "
            "if it is missing, the dataset was deleted rather than never created."
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationDatasetError(
                    f"{path}:{number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise EvaluationDatasetError(
                    f"{path}:{number} is a {type(payload).__name__}, expected a JSON object."
                )
            records.append(payload)
    return records


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes, as recorded by the builders."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_digest(paths: DatasetPaths) -> DatasetIntegrity:
    """Compare the queries file against the digest recorded when it was frozen.

    A missing digest file is reported as unverified with the reason, not raised.
    The file's absence means nobody recorded a freeze, which is a weaker claim
    than "the queries were edited" and must not be reported as the same thing.
    """
    if not paths.queries.exists():
        raise EvaluationDatasetNotFoundError(f"no queries file at {paths.queries}")
    actual = file_digest(paths.queries)
    if not paths.digest_file.exists():
        return DatasetIntegrity(
            verified=False,
            expected="",
            actual=actual,
            reason=(
                f"no {DIGEST_FILENAME} beside the queries, so the freeze cannot be "
                "verified. The queries may still be genuine; the claim is unproven."
            ),
        )
    recorded = paths.digest_file.read_text(encoding="utf-8").strip().split()
    if not recorded:
        return DatasetIntegrity(
            verified=False,
            expected="",
            actual=actual,
            reason=f"{paths.digest_file} is empty.",
        )
    expected = recorded[0]
    if expected == actual:
        return DatasetIntegrity(verified=True, expected=expected, actual=actual)
    return DatasetIntegrity(
        verified=False,
        expected=expected,
        actual=actual,
        reason=(
            "the queries file does not match the digest recorded when it was "
            "frozen. Every number computed from it describes a different query "
            "set than the one that was evaluated."
        ),
    )


def _validate_queries(records: Sequence[Mapping[str, Any]], source: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    seen: dict[str, int] = {}
    for position, record in enumerate(records, start=1):
        for required in ("query_id", "query", "stratum"):
            if required not in record:
                raise EvaluationDatasetError(
                    f"{source}: record {position} is missing required field {required!r}."
                )
        try:
            query = EvalQuery.from_dict(record)
        except (TypeError, ValueError) as exc:
            raise EvaluationDatasetError(
                f"{source}: record {position} could not be read as a query: {exc}"
            ) from exc
        if query.stratum not in STRATA:
            raise EvaluationDatasetError(
                f"{source}: query {query.query_id} has unknown stratum "
                f"{query.stratum!r}. Known strata: {list(STRATA)}."
            )
        if not query.query.strip():
            raise EvaluationDatasetError(
                f"{source}: query {query.query_id} has an empty query string. A blank "
                "query would be scored as a system failure and drag every average down."
            )
        if query.query_id in seen:
            raise EvaluationDatasetError(
                f"{source}: duplicate query_id {query.query_id!r} at records "
                f"{seen[query.query_id]} and {position}. Query ids are the join key "
                "between the queries and the judgments; a duplicate silently splits "
                "the judgments across two copies of the same question."
            )
        seen[query.query_id] = position
        queries.append(query)
    if not queries:
        raise EvaluationDatasetError(f"{source} contains no queries.")
    return queries


def _validate_judgments(
    records: Sequence[Mapping[str, Any]], source: Path, known_query_ids: set[str]
) -> list[RelevanceJudgment]:
    judgments: list[RelevanceJudgment] = []
    seen: set[tuple[str, str]] = set()
    for position, record in enumerate(records, start=1):
        for required in ("query_id", "doc_id"):
            if required not in record:
                raise EvaluationDatasetError(
                    f"{source}: judgment {position} is missing required field {required!r}."
                )
        judgment = RelevanceJudgment.from_dict(record)
        if judgment.query_id not in known_query_ids:
            raise EvaluationDatasetError(
                f"{source}: judgment names query {judgment.query_id!r}, which is not in "
                "the queries file. Either a query was deleted or a judgment was written "
                "for a question that no longer exists."
            )
        if judgment.relevance_grade < 0:
            raise EvaluationDatasetError(
                f"{source}: query {judgment.query_id} has a negative grade for "
                f"{judgment.doc_id}. Grades are 0 or positive; 0 and absent mean the "
                "same thing and neither belongs in the qrels file."
            )
        key = (judgment.query_id, judgment.doc_id)
        if key in seen:
            raise EvaluationDatasetError(
                f"{source}: duplicate judgment for query {judgment.query_id} and document "
                f"{judgment.doc_id}. Two grades for one pair means the file no longer "
                "says what that document is worth."
            )
        seen.add(key)
        judgments.append(judgment)
    return judgments


def load_dataset(
    paths: DatasetPaths,
    *,
    corpus_doc_ids: Iterable[str] | None = None,
    require_judgments: bool = True,
    strata: Collection[str] | None = None,
) -> tuple[EvalDataset, DatasetIntegrity]:
    """Load, validate, and cross-check the frozen dataset.

    ``corpus_doc_ids`` is the live corpus's id set. Passing it turns a
    "judgment points at a document that does not exist" case into a hard error
    rather than a silently unattainable gold label — which would score every
    system zero on that query and look like a retrieval failure.

    ``strata`` restricts the dataset to the named strata, and restricts the
    "every query needs a positive judgment" check to those strata as well. This
    is the one supported way to evaluate a partially-judged dataset: a subset
    with complete, genuinely-produced judgments is a valid experiment, whereas
    dropping the unjudged queries inside a subset by hand would be exactly the
    silent query removal the protocol forbids. Strata naming an unknown value
    raise rather than matching nothing, because a typo that selected zero
    queries would otherwise produce an empty, meaningless report.

    Returns the dataset and the integrity check, so the caller can put the
    verification result in the report next to the numbers it qualifies.
    """
    integrity = verify_digest(paths)
    query_records = _read_jsonl(paths.queries)
    all_queries = _validate_queries(query_records, paths.queries)

    if strata is not None:
        unknown = sorted(set(strata) - set(STRATA))
        if unknown:
            raise EvaluationDatasetError(
                f"unknown stratum filter {unknown}. Known strata: {list(STRATA)}. "
                "A filter that matched nothing would produce an empty report that "
                "reads like a result."
            )
        queries = [q for q in all_queries if q.stratum in set(strata)]
        if not queries:
            raise EvaluationDatasetError(
                f"the stratum filter {sorted(strata)} selected no queries from " f"{paths.queries}."
            )
    else:
        queries = all_queries

    # The judgments are validated against the ids of the WHOLE file, not the
    # filtered subset. A judgment naming a query in another stratum is not a
    # dangling reference -- it is a perfectly good judgment that this run
    # chose not to score -- and checking it against the filtered set made
    # `--strata` unusable on any dataset that was not judged in full at once:
    # restricting to the fully-judged strata failed with "judgment names query
    # 'X', which is not in the queries file", which is precisely the case the
    # flag exists to support. Filtering happens after, below.
    all_query_ids = {q.query_id for q in all_queries}
    query_ids = {q.query_id for q in queries}

    if require_judgments:
        judgment_records = _read_jsonl(paths.qrels)
        judgments = _validate_judgments(judgment_records, paths.qrels, all_query_ids)
        judgments = [j for j in judgments if j.query_id in query_ids]
        missing = [
            qid
            for qid in sorted(query_ids)
            if not any(j.query_id == qid and j.relevance_grade > 0 for j in judgments)
        ]
        if missing:
            raise EvaluationDatasetError(
                f"these queries have no positive judgment: {missing}. A query with no "
                "relevant document scores 0.0 on every metric and would be averaged in "
                "as if the systems had all failed it."
            )
    else:
        judgments = []

    if corpus_doc_ids is not None:
        known_docs = set(corpus_doc_ids)
        unknown = sorted({j.doc_id for j in judgments if j.doc_id not in known_docs})
        if unknown:
            raise EvaluationDatasetError(
                f"judgments name {len(unknown)} document(s) that are not in the corpus, "
                f"starting with {unknown[0]!r}. The dataset and the corpus are out of sync."
            )

    return EvalDataset(queries=tuple(queries), judgments=tuple(judgments)), integrity


@dataclass(slots=True, frozen=True)
class PoolReport:
    """How much of a system's top-k was ever looked at, for one query.

    This is the size of the hole under every metric computed from a pooled
    judgment. ``unjudged`` counts documents the judge saw in the pool and marked
    not relevant; ``never_pooled`` counts documents the judge never saw at all,
    which is the larger and less visible half.
    """

    query_id: str
    stratum: str
    pool_size: int
    judged: int
    unjudged: int
    never_pooled: int

    @property
    def unseen_fraction(self) -> float:
        total = self.judged + self.never_pooled
        return self.never_pooled / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "stratum": self.stratum,
            "pool_size": self.pool_size,
            "judged": self.judged,
            "unjudged": self.unjudged,
            "never_pooled": self.never_pooled,
            "unseen_fraction": round(self.unseen_fraction, 6),
        }


def pool_report(
    dataset: EvalDataset,
    pool: Mapping[str, Sequence[str]],
    top_k: int,
) -> list[PoolReport]:
    """Per-query accounting of what was judged and what was never in the pool.

    ``pool`` maps query id to the candidate list the judge worked through.
    ``never_pooled`` is the number of documents the judge did not see at all,
    counted against the highest rank any system gave them — the only honest
    denominator available, since anything deeper was never retrieved by anything.
    """
    reports: list[PoolReport] = []
    for query in dataset.queries:
        candidates = list(pool.get(query.query_id, ()))
        candidates = list(dict.fromkeys(candidates))
        judged_docs = set(candidates)
        positives = set(dataset.positive_doc_ids(query.query_id))
        reports.append(
            PoolReport(
                query_id=query.query_id,
                stratum=query.stratum,
                pool_size=len(candidates),
                judged=len(judged_docs),
                unjudged=len(judged_docs - positives),
                never_pooled=max(0, top_k - len(judged_docs)),
            )
        )
    return reports


def summarise(dataset: EvalDataset, integrity: DatasetIntegrity) -> dict[str, Any]:
    """A machine-readable description of the loaded dataset.

    Written into the run artifacts so the numbers in the report can be traced
    to the exact dataset that produced them, without re-running anything.
    """
    strata: dict[str, int] = {}
    judgments_per_query: list[int] = []
    for query in dataset.queries:
        strata[query.stratum] = strata.get(query.stratum, 0) + 1
        judgments_per_query.append(len(dataset.positive_doc_ids(query.query_id)))
    return {
        "n_queries": len(dataset.queries),
        "n_judgments": len(dataset.judgments),
        "queries_per_stratum": strata,
        "judgments_per_query_min": min(judgments_per_query) if judgments_per_query else 0,
        "judgments_per_query_max": max(judgments_per_query) if judgments_per_query else 0,
        "binary_only": all(j.relevance_grade == 1 for j in dataset.judgments),
        "judges": sorted({j.judged_by for j in dataset.judgments if j.judged_by}),
        "protocols": sorted({j.protocol for j in dataset.judgments if j.protocol}),
        "integrity": integrity.to_dict(),
    }
