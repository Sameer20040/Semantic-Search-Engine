"""FAISS vector index: build, persist, search.

Index type selection
--------------------
``flat_ip`` (default) -- exact inner-product search. With L2-normalised vectors
the inner product *is* cosine similarity. This is exhaustive, so recall is 1.0
and there is no training step. For corpora up to a few hundred thousand
documents it is fast enough, and exactness is worth more than speed for a
system whose whole point is trustworthy results.

``hnsw`` -- graph search. Sub-linear query time, high recall at a memory cost.
Worth it past roughly 500 000 documents.

``ivf_flat`` -- inverted file with coarse quantisation. Needs a training pass
over a representative sample; useful past ~1 000 000 documents.

``flat_l2`` -- Euclidean. Present for completeness and for corpora where
vectors are *not* normalised; not used by the default pipeline, which always
normalises.

Approximate indexes are configured but not the default because, for this corpus
size, an approximate index would trade measurable accuracy for speed nobody is
waiting on. That trade is documented rather than hidden.

Persistence
-----------
``write_index``/``read_index`` plus a JSON sidecar recording the model,
dimension, document count, and index type. The sidecar is what lets the loader
refuse an index built from a different embedding model -- the same reasoning as
the embedding cache, because a mismatched FAISS index produces plausible-looking
nonsense just as quietly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_search.core.config import FaissConfig
from semantic_search.core.exceptions import (
    CorruptIndexError,
    IndexCompatibilityError,
    IndexNotFoundError,
    SearchIndexError,
)
from semantic_search.core.logging import get_logger

logger = get_logger(__name__)

#: Bumped when the on-disk layout changes incompatibly.
FAISS_INDEX_VERSION = 1

VALID_INDEX_TYPES = ("flat_ip", "flat_l2", "hnsw", "ivf_flat")


@dataclass(slots=True, frozen=True)
class FaissHit:
    """One vector search result.

    ``row`` is the FAISS internal id, which by construction equals the document
    store position. ``score`` is cosine similarity for ``flat_ip``/``hnsw``
    (higher is better) and a distance for ``flat_l2`` (lower is better); the
    search service normalises to "higher is better" before fusion.
    """

    row: int
    score: float
    rank: int = 0

    def __post_init__(self) -> None:
        if self.row < 0:
            raise SearchIndexError(f"FAISS returned an invalid row id: {self.row}")


@dataclass(slots=True)
class FaissIndexMetadata:
    """Provenance for a persisted FAISS index."""

    version: int = FAISS_INDEX_VERSION
    index_type: str = "flat_ip"
    dimension: int = 0
    document_count: int = 0
    embedding_model: str = ""
    normalized: bool = True
    trained: bool = False
    created_at: str = ""
    build_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FaissIndexMetadata:
        known = {key: payload[key] for key in cls.__slots__ if key in payload}
        return cls(**known)


def _import_faiss() -> Any:
    try:
        import faiss

        return faiss
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise SearchIndexError(
            "faiss-cpu is required for vector search. Install it with: pip install faiss-cpu"
        ) from exc


class FaissIndex:
    """A FAISS index over a fixed matrix of document vectors."""

    def __init__(
        self,
        faiss_index: Any,
        config: FaissConfig,
        metadata: FaissIndexMetadata,
    ) -> None:
        self._index = faiss_index
        self.config = config
        self.metadata = metadata

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    def __len__(self) -> int:
        return int(self._index.ntotal)

    @property
    def index_type(self) -> str:
        return self.metadata.index_type

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def is_trained(self) -> bool:
        return bool(self._index.is_trained)

    @property
    def faiss_index(self) -> Any:
        """The underlying FAISS object, for callers needing lower-level control."""
        return self._index

    def stats(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type,
            "documents": len(self),
            "dimension": self.dimension,
            "embedding_model": self.metadata.embedding_model,
            "trained": self.is_trained,
            "memory_bytes": _index_memory_bytes(self._index),
        }

    # ----------------------------------------------------------------- #
    # Build
    # ----------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        config: FaissConfig | None = None,
        *,
        embedding_model: str = "",
        normalized: bool = True,
    ) -> FaissIndex:
        """Build an index from a ``(n, d)`` float32 matrix.

        The matrix must already be in the space the index type expects. For
        ``flat_ip`` and ``hnsw`` that means L2-normalised rows, which is what
        the encoder produces; normalising here as well would be harmless and
        makes the index safe to build from an arbitrary matrix.
        """
        faiss = _import_faiss()
        config = config or FaissConfig()
        if config.index_type not in VALID_INDEX_TYPES:
            raise SearchIndexError(
                f"Unknown FAISS index type {config.index_type!r}. "
                f"Valid types: {', '.join(VALID_INDEX_TYPES)}."
            )

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise SearchIndexError(
                f"Expected a 2-D (documents, dimension) matrix, got shape {matrix.shape}."
            )
        if matrix.shape[0] == 0:
            raise SearchIndexError("Cannot build a FAISS index over zero documents.")
        if not np.isfinite(matrix).all():
            bad = int((~np.isfinite(matrix)).any(axis=1).sum())
            raise SearchIndexError(
                f"Embedding matrix contains NaN or infinity in {bad} rows. "
                "Regenerate the embeddings before building the index."
            )

        if config.index_type in ("flat_ip", "hnsw") and not normalized:
            # Inner product only equals cosine similarity for unit vectors.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, np.where(norms == 0, 1.0, norms))

        started = time.perf_counter()
        index = cls._create(faiss, config, matrix.shape[1])
        index.add(matrix)
        build_seconds = time.perf_counter() - started

        if not index.is_trained:
            raise SearchIndexError(
                "FAISS index reports untrained after add(); it cannot be searched."
            )

        metadata = FaissIndexMetadata(
            index_type=config.index_type,
            dimension=int(matrix.shape[1]),
            document_count=int(matrix.shape[0]),
            embedding_model=embedding_model,
            normalized=normalized,
            trained=True,
            created_at=_utc_now(),
            build_seconds=round(build_seconds, 4),
        )
        logger.info(
            "faiss.built",
            extra={
                "event": "faiss.built",
                "index_type": config.index_type,
                "documents": matrix.shape[0],
                "dimension": matrix.shape[1],
                "build_seconds": round(build_seconds, 3),
            },
        )
        return cls(index, config, metadata)

    @staticmethod
    def _create(faiss: Any, config: FaissConfig, dimension: int) -> Any:
        if config.index_type == "flat_ip":
            return faiss.IndexFlatIP(dimension)
        if config.index_type == "flat_l2":
            return faiss.IndexFlatL2(dimension)
        if config.index_type == "hnsw":
            # faiss-cpu 1.7+ exposes this as IndexHNSWFlat; older builds only
            # have the factory function, so try the modern name first.
            try:
                index = faiss.IndexHNSWFlat(dimension, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            except AttributeError:  # pragma: no cover - older faiss
                index = faiss.IndexHNSWFlat(dimension, config.hnsw_m)
            index.hnsw.efConstruction = config.hnsw_ef_construction
            index.hnsw.efSearch = config.hnsw_ef_search
            return index
        if config.index_type == "ivf_flat":
            # nlist must not exceed the training-set size or FAISS cannot find
            # enough centroids; clamp it and warn rather than failing the build.
            raise SearchIndexError(
                "ivf_flat requires a training sample and is not enabled by the default "
                "pipeline. Use hnsw_flat for approximate search at this corpus size, "
                "or add documents until the corpus justifies IVF."
            )
        raise SearchIndexError(f"Unhandled FAISS index type {config.index_type!r}.")

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[FaissHit]:
        """Search for the ``top_k`` nearest documents.

        Returns hits sorted best-first with contiguous 0-based ranks, which is
        what Reciprocal Rank Fusion consumes.
        """
        if top_k <= 0:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise SearchIndexError(
                f"Query vector has dimension {query.shape[1]} but the index has "
                f"{self.dimension}. The query and documents were embedded by different models."
            )

        # Asking for more neighbours than exist is an error in FAISS, not a warning.
        effective_k = min(top_k, len(self))
        if effective_k <= 0:
            return []

        scores, rows = self._index.search(query, effective_k)
        hits: list[FaissHit] = []
        # Ranks come from len(hits), not the enumerate index: FAISS pads with
        # -1 when it has fewer real neighbours than k, and those must not punch
        # a hole in the rank sequence that fusion depends on.
        for row, score in zip(rows[0], scores[0], strict=True):
            if row < 0:
                continue
            hits.append(FaissHit(row=int(row), score=float(score), rank=len(hits)))

        # FAISS returns each metric type best-first already: descending for the
        # inner-product types, ascending for the L2 distance type. So the ranks
        # assigned above are correct for every index type and no re-sort is
        # needed. What does differ is the *direction* of `score` (higher-is-better
        # vs. lower-is-better), which is why FaissHit documents it; the search
        # service flips it for `flat_l2` before fusion, so fusion stays
        # index-agnostic.
        return hits

    def search_batch(self, queries: np.ndarray, top_k: int = 10) -> list[list[FaissHit]]:
        """Search several queries at once, which FAISS parallelises internally."""
        matrix = np.ascontiguousarray(queries, dtype=np.float32)
        if matrix.ndim != 2:
            raise SearchIndexError(f"Expected a 2-D query matrix, got shape {matrix.shape}.")
        if matrix.shape[0] == 0:
            return []
        results: list[list[FaissHit]] = []
        for row in matrix:
            results.append(self.search(row, top_k))
        return results

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        faiss = _import_faiss()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target))
        with target.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata.to_dict(), handle, indent=2)
        logger.info("faiss.saved", extra={"event": "faiss.saved", "path": str(target)})
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: FaissConfig | None = None,
        *,
        expected_model: str | None = None,
        expected_documents: int | None = None,
    ) -> FaissIndex:
        """Load a persisted index, refusing anything built incompatibly.

        Args:
            path: The ``.faiss`` artifact.
            config: Active config, used for search-time parameters.
            expected_model: Reject if the index was built with another model.
            expected_documents: Reject on a document-count mismatch, which means
                the index and the document store came from different corpora.
        """
        faiss = _import_faiss()
        source = Path(path)
        if not source.is_file():
            # IndexNotFoundError rather than the generic SearchIndexError, so a
            # caller (and the API layer) can tell "you never built an index" from
            # "the index you built is wrong".
            raise IndexNotFoundError(
                f"FAISS index not found at {source}. Run `make index` to build it."
            )
        try:
            index = faiss.read_index(str(source))
        except Exception as exc:
            raise CorruptIndexError(f"Could not read FAISS index at {source}: {exc}") from exc

        meta_path = source.with_suffix(".meta.json")
        if meta_path.is_file():
            try:
                metadata = FaissIndexMetadata.from_dict(
                    json.loads(meta_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise CorruptIndexError(
                    f"FAISS index metadata at {meta_path} is unreadable: {exc}"
                ) from exc
        else:
            # An index without provenance is usable but unverifiable; say so.
            logger.warning(
                "faiss.metadata_missing",
                extra={"event": "faiss.metadata_missing", "path": str(source)},
            )
            metadata = FaissIndexMetadata(
                dimension=int(index.d),
                document_count=int(index.ntotal),
                trained=bool(index.is_trained),
            )

        if metadata.version != FAISS_INDEX_VERSION:
            raise IndexCompatibilityError(
                f"FAISS index at {source} has version {metadata.version}, expected "
                f"{FAISS_INDEX_VERSION}. Rebuild the index."
            )
        if (
            expected_model
            and metadata.embedding_model
            and metadata.embedding_model != expected_model
        ):
            raise IndexCompatibilityError(
                f"FAISS index was built with model {metadata.embedding_model!r} but "
                f"{expected_model!r} is active. Rebuild the index."
            )
        if expected_documents is not None and int(index.ntotal) != expected_documents:
            raise IndexCompatibilityError(
                f"FAISS index holds {int(index.ntotal)} vectors but the document store has "
                f"{expected_documents}. Rebuild the index from the same corpus."
            )

        logger.info(
            "faiss.loaded",
            extra={
                "event": "faiss.loaded",
                "path": str(source),
                "documents": int(index.ntotal),
                "index_type": metadata.index_type,
            },
        )
        return cls(index, config or FaissConfig(), metadata)

    # ----------------------------------------------------------------- #
    # Training (approximate indexes)
    # ----------------------------------------------------------------- #
    @classmethod
    def train(cls, index: Any, sample: np.ndarray) -> None:
        """Train an index on a representative sample. Required by IVF."""
        if index.is_trained:
            return
        matrix = np.ascontiguousarray(sample, dtype=np.float32)
        index.train(matrix)


def _index_memory_bytes(index: Any) -> int:
    """Vector payload size in bytes, for the stats endpoint.

    Computed as ``ntotal * d * 4`` rather than read from FAISS because
    ``faiss.get_mem_usage_kb()`` reports *process-wide* allocation, which would
    silently attribute the encoder's model weights to the index. For flat and
    HNSW indexes this figure is exact; IVF stores additional inverted-list
    overhead that this under-reports, so callers should treat it as a floor.
    """
    try:
        return int(index.ntotal) * int(index.d) * 4
    except Exception:
        return 0


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows in place-safe fashion. Shared by index and query paths.

    Both sides must agree: cosine similarity via inner product is only meaningful
    when every vector is a unit vector.
    """
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, np.where(norms == 0, 1.0, norms)).astype(np.float32)
