"""The canonical in-memory document representation and its (de)serialisation.

Every other subsystem -- BM25, FAISS, the API, the evaluator -- speaks
:class:`Document`. Using one typed model (rather than raw dicts) is what makes the
document-id -> row -> document round-trip checkable, which is the single most
common source of silent corruption in hybrid search systems.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_search.core.exceptions import DataError


@dataclass(slots=True)
class Document:
    """A single indexable document.

    Attributes:
        id: Stable, corpus-unique identifier. For Wikipedia this is the article
            id (``"12"`` for Anarchism), which is stable across dataset versions.
        title: Display title, cleaned (disambiguator removed).
        text: Cleaned body text.
        metadata: Arbitrary JSON-serialisable provenance (url, categories, ...).
        title_raw: Original title including any disambiguator, kept for search.
    """

    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    title_raw: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id)
        if not self.id:
            raise DataError("Document id must be a non-empty string.")
        if not self.title:
            raise DataError(f"Document {self.id!r} is missing a title.")
        if self.text is None:
            raise DataError(f"Document {self.id!r} is missing text.")
        if not self.title_raw:
            self.title_raw = self.title

    @property
    def searchable_text(self) -> str:
        """Text used for keyword matching: title (weighted) plus body."""
        return f"{self.title}\n{self.text}"

    @property
    def length(self) -> int:
        return len(self.text)

    def snippet(self, query_terms: Iterable[str] | None = None, length: int = 320) -> str:
        """Return a query-aware excerpt of the body.

        When ``query_terms`` are supplied the excerpt is centred on the densest
        matching region, which is markedly more useful than always returning the
        document head (Wikipedia leads with disambiguation-style preamble).
        """
        text = self.text
        if len(text) <= length:
            return text
        if query_terms:
            lowered = text.lower()
            positions = [lowered.find(term.lower()) for term in query_terms if len(term) > 2]
            positions = [pos for pos in positions if pos >= 0]
            if positions:
                start = max(0, min(positions) - length // 4)
                return (
                    ("..." if start else "")
                    + text[start : start + length].strip()
                    + ("..." if start + length < len(text) else "")
                )
        return text[:length].strip() + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
            "title_raw": self.title_raw,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        try:
            return cls(
                id=str(payload["id"]),
                title=str(payload["title"]),
                text=str(payload.get("text", "")),
                metadata=payload.get("metadata") or {},
                title_raw=str(payload.get("title_raw", payload["title"])),
            )
        except KeyError as exc:
            raise DataError(f"Document payload is missing required key {exc}.") from exc


class DocumentStore:
    """An id-addressable document collection with positional (row) access.

    Both access patterns are needed and they must stay consistent:

    * ``documents[row]`` -- matches the FAISS row id and the BM25 corpus position.
    * ``get(doc_id)``    -- matches what the API returns to a client.

    :meth:`validate_alignment` asserts the two agree, and is called at startup.
    """

    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self._documents: list[Document] = list(documents or [])
        self._index_by_id: dict[str, int] = {}
        for position, document in enumerate(self._documents):
            if document.id in self._index_by_id:
                raise DataError(f"Duplicate document id {document.id!r} in document store.")
            self._index_by_id[document.id] = position

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    def __getitem__(self, position: int) -> Document:
        return self._documents[position]

    def __contains__(self, doc_id: object) -> bool:
        return str(doc_id) in self._index_by_id

    @property
    def document_ids(self) -> list[str]:
        """Ids in row order; this is the vector-index id mapping."""
        return [document.id for document in self._documents]

    def get(self, doc_id: str) -> Document | None:
        position = self._index_by_id.get(str(doc_id))
        return None if position is None else self._documents[position]

    def row_of(self, doc_id: str) -> int | None:
        """The row position for a document id, or ``None`` if absent.

        The reverse of ``store[row].id``. Exposed because fusion arrives holding
        ``doc_id`` strings -- that is what RRF is defined over -- and has to
        attach the store ``row`` to each result. Scanning ``document_ids`` for
        the same information would be O(n) per result, turning a few hundred
        candidates into a few hundred corpus scans per query.
        """
        return self._index_by_id.get(str(doc_id))

    def get_many(self, doc_ids: Iterable[str]) -> list[Document]:
        return [doc for doc in (self.get(d) for d in doc_ids) if doc is not None]

    def at(self, positions: Iterable[int]) -> list[Document]:
        return [
            self._documents[position]
            for position in positions
            if 0 <= position < len(self._documents)
        ]

    def titles(self) -> list[str]:
        return [document.title for document in self._documents]

    def add(self, document: Document) -> int:
        """Append a document and return its row position."""
        if document.id in self._index_by_id:
            raise DataError(f"Document {document.id!r} already exists in the store.")
        self._documents.append(document)
        self._index_by_id[document.id] = len(self._documents) - 1
        return len(self._documents) - 1

    def validate_alignment(self, expected_count: int) -> None:
        """Assert the store size matches an index row count.

        A mismatch means the vector index and document store were built from
        different corpora, which would silently mis-map every search result.
        """
        if len(self._documents) != expected_count:
            raise DataError(
                f"Document store has {len(self._documents)} documents but the index "
                f"expects {expected_count}. Rebuild the index from the same corpus."
            )
        if len(self._index_by_id) != len(self._documents):
            raise DataError("Document id mapping is inconsistent with the document list.")

    # ----------------------------------------------------------------- #
    # Serialisation
    # ----------------------------------------------------------------- #
    def save_jsonl(self, path: str | Path) -> Path:
        """Write one JSON object per line (streamed; never builds a giant list)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for document in self._documents:
                handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
        return target

    @classmethod
    def load_jsonl(cls, path: str | Path, *, skip_malformed: bool = True) -> DocumentStore:
        """Read a JSONL corpus, streaming line by line to bound memory usage.

        Malformed lines are counted and skipped rather than aborting a long build;
        ``skip_malformed=False`` makes them fatal.
        """
        source = Path(path)
        if not source.is_file():
            raise DataError(f"Document store file not found: {source}")
        documents: list[Document] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    documents.append(Document.from_dict(json.loads(line)))
                except (json.JSONDecodeError, DataError, KeyError):
                    if not skip_malformed:
                        raise DataError(f"Malformed document at {source}:{line_number}.") from None
                    continue
        return cls(documents)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [document.to_dict() for document in self._documents]
