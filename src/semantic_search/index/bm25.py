"""Okapi BM25 lexical retrieval.

Why BM25 and not TF-IDF
-----------------------
TF-IDF rewards rare terms but ignores document length, so a long article that
mentions a term once outranks a short, focused one that mentions it three
times. BM25 adds saturating term frequency (repeated mentions stop helping
past a point) and length normalisation. Both matter for Wikipedia, where
article lengths span four orders of magnitude.

Implementation
--------------
Scoring is implemented directly rather than via ``rank_bm25`` so that
per-term contributions are available for explainability -- the "why did this
document match?" panel needs the individual term scores, not just their sum.
The formula is the standard one:

    IDF(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(D) = Σ_t IDF(t) · (f(t,D)·(k1+1)) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))

The IDF form is the non-negative Lucene variant: terms appearing in more than
half the corpus still contribute slightly, which is what users expect when they
type a common word. The classic Robertson IDF would make them negative and
actively harmful.

The index is a plain dict of postings, so it pickles and unpickles quickly and
can be rebuilt in seconds for a development corpus.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_search.core.config import BM25Config
from semantic_search.core.exceptions import DataError, SearchIndexError
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.data.tokenizer import Tokenizer, get_tokenizer

logger = get_logger(__name__)

#: Bumped when the on-disk layout changes in a way older code cannot read.
BM25_INDEX_VERSION = 1


@dataclass(slots=True, frozen=True)
class BM25Result:
    """One scored lexical hit.

    ``rank`` is 0-based and is what Reciprocal Rank Fusion consumes; BM25 scores
    themselves are unbounded and are not comparable across queries, which is
    exactly why fusion uses ranks rather than raw scores.
    """

    doc_id: str
    row: int
    score: float
    rank: int
    matched_terms: tuple[str, ...] = ()
    term_scores: dict[str, float] | None = None

    @property
    def doc_index(self) -> int:
        return self.row


class BM25Index:
    """A built, immutable BM25 index over a fixed document store."""

    def __init__(
        self,
        *,
        doc_ids: Sequence[str],
        doc_lengths: Sequence[int],
        term_frequencies: Sequence[dict[str, int]],
        average_length: float,
        config: BM25Config,
        tokenizer: Tokenizer,
    ) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_lengths = list(doc_lengths)
        self.term_frequencies = [dict(tf) for tf in term_frequencies]
        self.average_length = average_length
        self.config = config
        self.tokenizer = tokenizer
        self._document_frequency: Counter[str] = Counter()
        for tf in self.term_frequencies:
            self._document_frequency.update(tf.keys())

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        documents: Sequence[Document] | DocumentStore,
        config: BM25Config | None = None,
        tokenizer: Tokenizer | None = None,
        *,
        title_boost: int = 1,
    ) -> BM25Index:
        """Build an index by tokenizing every document once.

        Args:
            documents: The corpus to index.
            config: BM25 parameters. Defaults to k1=1.5, b=0.75.
            tokenizer: Analyzer. Defaults to stemming + English stopwords, shared
                with the query processor so index and query terms always agree.
            title_boost: Extra copies of the title's tokens in a document's term
                frequency. A title match is a much stronger signal than a body
                match, and this is the cheapest way to express that inside the
                standard BM25 formula.

        Returns:
            A ready-to-query :class:`BM25Index`.
        """
        config = config or BM25Config()
        tokenizer = tokenizer or get_tokenizer(
            stopwords=config.stopwords, use_stemming=config.use_stemming
        )
        docs = list(documents)
        if not docs:
            raise DataError("Cannot build a BM25 index over an empty corpus.")

        doc_ids: list[str] = []
        doc_lengths: list[int] = []
        term_frequencies: list[dict[str, int]] = []

        for document in docs:
            tokens = tokenizer.tokenize(document.text)
            if title_boost > 0:
                tokens = tokens + tokenizer.tokenize(document.title) * title_boost
            doc_ids.append(document.id)
            doc_lengths.append(len(tokens))
            term_frequencies.append(dict(Counter(tokens)))

        average = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        logger.info(
            "bm25.built",
            extra={
                "event": "bm25.built",
                "documents": len(docs),
                "vocabulary": len({t for tf in term_frequencies for t in tf}),
                "average_length": round(average, 2),
            },
        )
        return cls(
            doc_ids=doc_ids,
            doc_lengths=doc_lengths,
            term_frequencies=term_frequencies,
            average_length=average,
            config=config,
            tokenizer=tokenizer,
        )

    # ----------------------------------------------------------------- #
    # Statistics
    # ----------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._document_frequency)

    def idf(self, term: str) -> float:
        """Lucene-style non-negative IDF for a single term."""
        n = len(self.doc_ids)
        df = self._document_frequency.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ----------------------------------------------------------------- #
    # Scoring
    # ----------------------------------------------------------------- #
    def score(self, row: int, query_terms: Sequence[str]) -> tuple[float, dict[str, float]]:
        """Score one document. Returns ``(total, per_term)``."""
        tf = self.term_frequencies[row]
        length = self.doc_lengths[row]
        k1 = self.config.k1
        b = self.config.b
        # Guard against a zero average (an all-empty corpus would divide by 0).
        norm = 1.0 - b + b * (length / self.average_length if self.average_length else 1.0)

        total = 0.0
        per_term: dict[str, float] = {}
        for term in query_terms:
            frequency = tf.get(term, 0)
            if not frequency:
                continue
            contribution = self.idf(term) * (frequency * (k1 + 1)) / (frequency + k1 * norm)
            per_term[term] = contribution
            total += contribution
        return total, per_term

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Rank documents for a query.

        The implementation scores every document rather than walking an inverted
        index. For a 200 000-document corpus that is ~200 000 dict lookups per
        query, which is roughly 50-100 ms in pure Python -- acceptable, and it
        buys exact per-term scores for explainability. If that ever stops being
        acceptable, the fix is a real inverted index, not a bigger machine.
        """
        if not query or not query.strip():
            return []
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, int, dict[str, float]]] = []
        for row in range(len(self.doc_ids)):
            total, per_term = self.score(row, query_terms)
            if total > 0.0:
                scored.append((total, row, per_term))

        # Ties broken by row so results are fully deterministic; without this,
        # two identical queries could return different orderings.
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[BM25Result] = []
        for rank, (total, row, per_term) in enumerate(scored[:top_k]):
            results.append(
                BM25Result(
                    doc_id=self.doc_ids[row],
                    row=row,
                    score=round(total, 6),
                    rank=rank,
                    matched_terms=tuple(per_term.keys()),
                    term_scores={term: round(value, 6) for term, value in per_term.items()},
                )
            )
        return results

    def explain(self, query: str, doc_id: str) -> dict[str, Any] | None:
        """Per-term score breakdown for one document, for the explain endpoint."""
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms or doc_id not in self.doc_ids:
            return None
        row = self.doc_ids.index(doc_id)
        total, per_term = self.score(row, query_terms)
        return {
            "doc_id": doc_id,
            "row": row,
            "total_score": round(total, 6),
            "matched_terms": list(per_term.keys()),
            "missing_terms": [t for t in query_terms if t not in per_term],
            "term_scores": {t: round(v, 6) for t, v in per_term.items()},
            "term_frequencies": {t: self.term_frequencies[row].get(t, 0) for t in query_terms},
            "document_length": self.doc_lengths[row],
            "average_length": round(self.average_length, 2),
        }

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        """Persist to disk. A sidecar ``.meta.json`` records the parameters.

        The sidecar exists so a mismatch is detectable at load time: rebuilding
        with different k1/b, a different stemmer, or a different corpus would
        otherwise produce an index that loads fine and quietly scores wrongly.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "version": BM25_INDEX_VERSION,
                    "doc_ids": self.doc_ids,
                    "doc_lengths": self.doc_lengths,
                    "term_frequencies": self.term_frequencies,
                    "average_length": self.average_length,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        meta = {
            "version": BM25_INDEX_VERSION,
            "documents": len(self.doc_ids),
            "vocabulary_size": self.vocabulary_size,
            "average_length": round(self.average_length, 4),
            "bm25_k1": self.config.k1,
            "bm25_b": self.config.b,
            "use_stemming": self.config.use_stemming,
            "stopwords": self.config.stopwords,
        }
        with target.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        logger.info("bm25.saved", extra={"event": "bm25.saved", "path": str(target)})
        return target

    @classmethod
    def load(cls, path: str | Path, config: BM25Config | None = None) -> BM25Index:
        """Load a persisted index, verifying that it was built compatibly."""
        source = Path(path)
        if not source.is_file():
            raise SearchIndexError(
                f"BM25 index not found at {source}. Run `make index` to build it."
            )
        try:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise SearchIndexError(f"Could not read BM25 index at {source}: {exc}") from exc

        version = payload.get("version")
        if version != BM25_INDEX_VERSION:
            raise SearchIndexError(
                f"BM25 index at {source} has version {version}, expected {BM25_INDEX_VERSION}. "
                "Rebuild the index."
            )

        config = config or BM25Config()
        tokenizer = get_tokenizer(stopwords=config.stopwords, use_stemming=config.use_stemming)
        index = cls(
            doc_ids=payload["doc_ids"],
            doc_lengths=payload["doc_lengths"],
            term_frequencies=payload["term_frequencies"],
            average_length=payload["average_length"],
            config=config,
            tokenizer=tokenizer,
        )
        index.check_compatibility()
        logger.info(
            "bm25.loaded",
            extra={"event": "bm25.loaded", "path": str(source), "documents": len(index)},
        )
        return index

    def check_compatibility(self) -> None:
        """Raise if the stored parameters disagree with the active config."""
        meta_path = Path(self.config.index_filename).with_suffix(".meta.json")
        if not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        mismatches = []
        for key, active in (
            ("bm25_k1", self.config.k1),
            ("bm25_b", self.config.b),
            ("use_stemming", self.config.use_stemming),
            ("stopwords", self.config.stopwords),
        ):
            stored = meta.get(key)
            if stored is not None and stored != active:
                mismatches.append(f"{key}: index={stored!r} config={active!r}")
        if mismatches:
            raise SearchIndexError(
                "BM25 index was built with different settings than the active config "
                f"({'; '.join(mismatches)}). Rebuild the index or restore the config."
            )


class BM25Searcher:
    """Thin convenience wrapper binding an index to a document store.

    Kept separate from :class:`BM25Index` so the index can be built, saved, and
    unit-tested without any documents in memory.
    """

    def __init__(self, index: BM25Index, store: DocumentStore) -> None:
        if len(index) != len(store):
            raise DataError(
                f"BM25 index covers {len(index)} documents but the store holds {len(store)}. "
                "Rebuild the index from the same corpus."
            )
        self.index = index
        self.store = store

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document] | DocumentStore,
        config: BM25Config | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> BM25Searcher:
        return cls(BM25Index.build(documents, config, tokenizer), DocumentStore(documents))

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        return self.index.search(query, top_k)

    def search_with_documents(
        self, query: str, top_k: int = 10
    ) -> list[tuple[BM25Result, Document]]:
        """Ranked ``(result, document)`` pairs, which is what the search service wants."""
        return [(result, self.store[result.row]) for result in self.index.search(query, top_k)]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.index),
            "vocabulary_size": self.index.vocabulary_size,
            "average_document_length": round(self.index.average_length, 2),
            "bm25_k1": self.index.config.k1,
            "bm25_b": self.index.config.b,
            "use_stemming": self.index.config.use_stemming,
        }

    def iter_ids(self) -> Iterable[str]:
        return iter(self.index.doc_ids)
