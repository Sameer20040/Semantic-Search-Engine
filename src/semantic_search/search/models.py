"""Structured search result types.

Why a typed model
-----------------
Search results used to be assembled as dicts at the call site, which is how
``{"title": doc["title"], "score": score}`` ends up with a typo'd key that only
shows up in the API response. A frozen dataclass makes the shape checkable,
carries its own serialisation, and keeps the retrievers interchangeable: a
``SearchResult`` from BM25, from the vector index, or from fusion is the same
type, so the API and the frontend do not branch on which retriever ran.

Scoring semantics
-----------------
``score`` is always **higher is better**, and for the vector retriever it is a
cosine similarity in ``[-1, 1]``. That normalisation happens in the service
rather than in FAISS, because ``flat_l2`` reports a distance where smaller is
better, and downstream consumers should not have to know which index type
happened to be built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semantic_search.data.loader import Document


@dataclass(slots=True, frozen=True)
class SearchResult:
    """One ranked search hit.

    Attributes:
        doc_id: Stable document id, as stored in the document store.
        title: Display title, copied at construction so a result stays valid
            even if the store is mutated underneath it.
        snippet: Query-aware excerpt, built by :meth:`Document.snippet`.
        score: Relevance score, higher is better. Cosine similarity for the
            vector retriever; not comparable across queries for BM25.
        rank: 0-based position. This is what Reciprocal Rank Fusion consumes,
            so it must be contiguous and it is never reused as a display index.
        row: Position in the document store, i.e. the vector index id. Kept for
            debugging and for the explainability panel.
        scores: Per-retriever scores that contributed to this result, keyed by
            retriever name (``"bm25"``, ``"semantic"``). Empty for a
            single-retriever search.
    """

    doc_id: str
    title: str
    snippet: str
    score: float
    rank: int
    row: int
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        score: float,
        rank: int,
        row: int,
        snippet_length: int = 320,
        query_terms: list[str] | None = None,
        scores: dict[str, float] | None = None,
    ) -> SearchResult:
        """Build a result from a store document.

        ``query_terms`` centres the snippet on the matching region; without it
        the excerpt falls back to the document head, which for Wikipedia is
        often disambiguation boilerplate.
        """
        return cls(
            doc_id=document.id,
            title=document.title,
            snippet=document.snippet(query_terms, snippet_length),
            score=score,
            rank=rank,
            row=row,
            scores=dict(scores or {}),
        )

    @property
    def doc_index(self) -> int:
        """Alias matching :attr:`BM25Result.doc_index`."""
        return self.row

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.score, 6),
            "rank": self.rank,
        }
        if self.scores:
            payload["scores"] = {name: round(value, 6) for name, value in self.scores.items()}
        return payload
