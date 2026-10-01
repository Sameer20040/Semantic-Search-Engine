"""Typed records for the evaluation: dataset, judgments, and per-query results.

These are the shapes that get written to ``data/evaluation/derived/``. Every one
has ``to_dict`` and ``from_dict`` and nothing else that touches disk, so the
serialisation format is stated once per record and the runner never hand-builds
a dict.

The naming is deliberately boring. ``query_id`` and ``doc_id`` match the corpus
and the qrels file; ``stratum`` is the grouping the report is allowed to pool
over; ``relevant_doc_ids`` is the judgment as the evaluator saw it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The four strata. They are never pooled into one number.
STRATA: tuple[str, ...] = ("A_restated", "A_verbatim", "A_typo", "B_topical")

#: Judgments are binary in this dataset. The field is a grade so the metrics can
#: be exercised against graded input in tests, and the constant below is what
#: stops anyone believing the project has a graded protocol.
BINARY_GRADE = 1


@dataclass(slots=True, frozen=True)
class EvalQuery:
    """One frozen query.

    Attributes:
        query_id: Stable id, unique within the dataset.
        query: The query text, exactly as a user would type it.
        stratum: One of :data:`STRATA`.
        source_doc_id: The article the query was written about. For strata A
            this is the relevance judgment in disguise; for stratum B it is the
            article that motivated the question, and is **not** itself a
            judgment.
        derivation: How the query was built (``lead``, ``title``, ``typo:<id>``,
            ``pooled``). Recorded so a reader can tell a known-item query from a
            hand-judged one without consulting the manifest.
        notes: Free text. Empty is fine and is not a warning sign.
    """

    query_id: str
    query: str
    stratum: str
    source_doc_id: str = ""
    derivation: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "stratum": self.stratum,
            "source_doc_id": self.source_doc_id,
            "derivation": self.derivation,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvalQuery:
        return cls(
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            stratum=str(payload["stratum"]),
            source_doc_id=str(payload.get("source_doc_id", "")),
            derivation=str(payload.get("derivation", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(slots=True, frozen=True)
class RelevanceJudgment:
    """One ``(query, document)`` relevance decision."""

    query_id: str
    doc_id: str
    relevance_grade: int = BINARY_GRADE
    judged_by: str = ""
    protocol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "relevance_grade": self.relevance_grade,
            "judged_by": self.judged_by,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RelevanceJudgment:
        return cls(
            query_id=str(payload["query_id"]),
            doc_id=str(payload["doc_id"]),
            relevance_grade=int(payload.get("relevance_grade", BINARY_GRADE)),
            judged_by=str(payload.get("judged_by", "")),
            protocol=str(payload.get("protocol", "")),
        )


@dataclass(slots=True, frozen=True)
class EvalDataset:
    """The frozen queries and their judgments, loaded and cross-checked.

    ``judgments_for`` returns only the positive judgments: a metric that takes
    ``{doc_id: 0}`` for every unjudged document and a mapping that omits them
    entirely are the same thing to the metrics (both read as grade 0), and
    keeping the negative decisions out means a zero grade can never be
    confused with "not looked at".
    """

    queries: tuple[EvalQuery, ...] = ()
    judgments: tuple[RelevanceJudgment, ...] = ()

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def query_ids(self) -> list[str]:
        return [q.query_id for q in self.queries]

    def query(self, query_id: str) -> EvalQuery | None:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        return None

    def by_stratum(self, stratum: str) -> list[EvalQuery]:
        return [q for q in self.queries if q.stratum == stratum]

    def judgments_for(self, query_id: str) -> dict[str, int]:
        return {
            j.doc_id: j.relevance_grade
            for j in self.judgments
            if j.query_id == query_id and j.relevance_grade > 0
        }

    def positive_doc_ids(self, query_id: str) -> list[str]:
        return sorted(self.judgments_for(query_id))

    def to_dicts(self) -> list[dict[str, Any]]:
        return [q.to_dict() for q in self.queries]

    def judgment_dicts(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self.judgments]


@dataclass(slots=True, frozen=True)
class PerQueryResult:
    """One query evaluated against one system. The unit of the whole evaluation.

    This is the record that makes the evaluation auditable after the fact: it
    carries the query, the system's ranking, the judgments, every metric value,
    and what query processing did to the query -- so a reader can recompute any
    number in the report from this line alone, without re-running anything.

    ``returned_doc_ids`` is capped at the run's retrieval depth. A reader who
    wants to know why P@10 fell should be able to see the top 10 without also
    being handed 50 ids they did not ask for.
    """

    query_id: str
    query: str
    system: str
    stratum: str
    returned_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_5: float
    ndcg_at_10: float
    average_precision: float
    #: The string actually handed to the retrievers. Persisted per arm so a
    #: query-processing arm that silently failed to apply its override is
    #: visible in the artifact instead of showing up as a mysterious "no
    #: difference" in the report.
    query_for_retrieval: str = ""
    was_modified: bool = False
    corrections: tuple[str, ...] = ()
    #: Titles for the gold documents, so the artifact is readable without a
    #: corpus join.
    gold_titles: tuple[str, ...] = ()
    returned_titles: tuple[str, ...] = ()
    #: Fraction of the top-10 that no judge ever saw. Reported, never hidden:
    #: it is the size of the hole under every number computed from this row.
    unjudged_at_10: float = 0.0
    #: 1-based position of the first judged-relevant result, 0 if none. Kept
    #: because "where did the gold document end up" is the question the report
    #: actually answers, and recomputing it from ids is a step too many.
    gold_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "system": self.system,
            "stratum": self.stratum,
            "returned_doc_ids": list(self.returned_doc_ids),
            "relevant_doc_ids": list(self.relevant_doc_ids),
            "precision_at_1": round(self.precision_at_1, 6),
            "precision_at_5": round(self.precision_at_5, 6),
            "precision_at_10": round(self.precision_at_10, 6),
            "recall_at_5": round(self.recall_at_5, 6),
            "recall_at_10": round(self.recall_at_10, 6),
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "ndcg_at_5": round(self.ndcg_at_5, 6),
            "ndcg_at_10": round(self.ndcg_at_10, 6),
            "average_precision": round(self.average_precision, 6),
            "query_for_retrieval": self.query_for_retrieval,
            "was_modified": self.was_modified,
            "corrections": list(self.corrections),
            "gold_titles": list(self.gold_titles),
            "returned_titles": list(self.returned_titles),
            "unjudged_at_10": round(self.unjudged_at_10, 6),
            "gold_rank": self.gold_rank,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PerQueryResult:
        return cls(
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            system=str(payload["system"]),
            stratum=str(payload["stratum"]),
            returned_doc_ids=tuple(payload.get("returned_doc_ids", ())),
            relevant_doc_ids=tuple(payload.get("relevant_doc_ids", ())),
            precision_at_1=float(payload.get("precision_at_1", 0.0)),
            precision_at_5=float(payload.get("precision_at_5", 0.0)),
            precision_at_10=float(payload.get("precision_at_10", 0.0)),
            recall_at_5=float(payload.get("recall_at_5", 0.0)),
            recall_at_10=float(payload.get("recall_at_10", 0.0)),
            reciprocal_rank=float(payload.get("reciprocal_rank", 0.0)),
            ndcg_at_5=float(payload.get("ndcg_at_5", 0.0)),
            ndcg_at_10=float(payload.get("ndcg_at_10", 0.0)),
            average_precision=float(payload.get("average_precision", 0.0)),
            query_for_retrieval=str(payload.get("query_for_retrieval", "")),
            was_modified=bool(payload.get("was_modified", False)),
            corrections=tuple(payload.get("corrections", ())),
            gold_titles=tuple(payload.get("gold_titles", ())),
            returned_titles=tuple(payload.get("returned_titles", ())),
            unjudged_at_10=float(payload.get("unjudged_at_10", 0.0)),
            gold_rank=int(payload.get("gold_rank", 0)),
        )


@dataclass(slots=True, frozen=True)
class ArmResult:
    """One system's rankings for every query in the dataset.

    ``rankings`` maps query_id to that system's ordered doc ids, and
    ``processed`` maps query_id to the string the arm actually retrieved with.
    Both are kept so the equivalence audit can ask "did processing change the
    string" and "did it change the ranking" as two separate questions -- they
    have different answers, and conflating them is how a no-op gets reported as
    a small improvement.
    """

    system: str
    label: str
    rankings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    processed: dict[str, str] = field(default_factory=dict)
    modified: dict[str, bool] = field(default_factory=dict)
    corrections: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Document titles parallel to ``rankings``, for readable artifacts.
    titles: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return list(self.rankings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "label": self.label,
            "rankings": {q: list(ids) for q, ids in self.rankings.items()},
            "processed": dict(self.processed),
            "modified": dict(self.modified),
            "corrections": {q: list(c) for q, c in self.corrections.items()},
            "titles": {q: list(t) for q, t in self.titles.items()},
        }


def top_k_ids(ranked: Sequence[str], k: int) -> list[str]:
    """First ``k`` ids of a ranking, as a list."""
    return list(ranked)[:k]


def stratum_of(queries: Iterable[EvalQuery]) -> dict[str, str]:
    """``query_id -> stratum`` lookup."""
    return {q.query_id: q.stratum for q in queries}
