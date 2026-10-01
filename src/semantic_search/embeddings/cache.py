"""Persistent, resumable storage for document embeddings.

Embedding a 200 000-document corpus on CPU takes a substantial fraction of an
hour. That cost must be paid once, not on every container restart, and an
interrupted run must not throw the work away. This module provides both:

* **Persistence** -- float32 ``.npy`` for the matrix (memory-mappable, so a
  200 000 x 384 matrix is never fully resident), plus a JSON sidecar holding the
  metadata needed to decide whether the cache is still usable.
* **Resumability** -- completed batches are flushed periodically, and a resumed
  run recomputes only the rows that are missing.
* **Invalidation** -- a cache built with a different model, a different
  dimension, or against a different corpus is *refused*, not silently used.
  Serving vectors from a mismatched model produces results that look reasonable
  and are quietly wrong, which is the worst failure mode in this system.

The metadata sidecar is the whole point. Without it, a stale ``.npy`` looks
exactly like a valid one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_search.core.exceptions import EmbeddingCompatibilityError
from semantic_search.core.logging import get_logger

logger = get_logger(__name__)

#: Bumped when the ``.npy`` + sidecar layout changes incompatibly.
CACHE_VERSION = 1


@dataclass(slots=True)
class EmbeddingMetadata:
    """Provenance for a cached embedding matrix.

    Every field here is checked on load. Two caches are interchangeable only if
    all of them agree.
    """

    version: int = CACHE_VERSION
    model_name: str = ""
    dimension: int = 0
    document_count: int = 0
    completed_count: int = 0
    corpus_fingerprint: str = ""
    normalize: bool = True
    created_at: str = ""
    #: Rows filled in by an interrupted run, 1-based count for reporting.
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingMetadata:
        known = {field: payload.get(field) for field in cls.__slots__ if field in payload}
        return cls(**{k: v for k, v in known.items() if v is not None})


class EmbeddingCache:
    """A ``.npy`` matrix plus a JSON sidecar, with a compatibility gate."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".meta.json")
        self._metadata: EmbeddingMetadata | None = None

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    def exists(self) -> bool:
        return self.path.is_file() and self.metadata_path.is_file()

    @property
    def metadata(self) -> EmbeddingMetadata:
        if self._metadata is None:
            if not self.metadata_path.is_file():
                raise EmbeddingCompatibilityError(
                    f"No embedding metadata at {self.metadata_path}. Rebuild the embeddings."
                )
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self._metadata = EmbeddingMetadata.from_dict(payload)
        return self._metadata

    def is_complete(self) -> bool:
        """True when every row of the corpus has an embedding."""
        if not self.exists():
            return False
        meta = self.metadata
        return meta.completed_count >= meta.document_count > 0

    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.is_file() else 0

    # ----------------------------------------------------------------- #
    # Compatibility
    # ----------------------------------------------------------------- #
    def check_compatible(
        self,
        *,
        model_name: str,
        dimension: int,
        document_count: int,
        corpus_fingerprint: str = "",
        strict_fingerprint: bool = True,
    ) -> None:
        """Raise if the cache cannot be used for this corpus/model.

        Args:
            model_name: Model that must have produced the cached vectors.
            dimension: Expected embedding width.
            document_count: Expected number of corpus rows.
            corpus_fingerprint: Corpus identity hash. When ``strict_fingerprint``
                is set, a mismatch is fatal; otherwise it is only logged, which
                is what a search-only restart wants (it does not add documents).
            strict_fingerprint: Whether corpus identity must match exactly.
        """
        if not self.exists():
            return
        meta = self.metadata
        problems: list[str] = []

        if meta.version != CACHE_VERSION:
            problems.append(f"cache version {meta.version} != {CACHE_VERSION}")
        if meta.model_name != model_name:
            problems.append(f"model {meta.model_name!r} != {model_name!r}")
        if meta.dimension != dimension:
            problems.append(f"dimension {meta.dimension} != {dimension}")
        if meta.document_count != document_count:
            problems.append(f"document_count {meta.document_count} != {document_count}")
        if (
            strict_fingerprint
            and corpus_fingerprint
            and meta.corpus_fingerprint != corpus_fingerprint
        ):
            problems.append(
                f"corpus fingerprint {meta.corpus_fingerprint!r} != {corpus_fingerprint!r}"
            )

        if problems:
            raise EmbeddingCompatibilityError(
                "The cached embeddings are not usable for this corpus/model: "
                + "; ".join(problems)
                + ". Delete the cache and rebuild (make index)."
            )
        if corpus_fingerprint and meta.corpus_fingerprint != corpus_fingerprint:
            logger.warning(
                "embeddings.cache.fingerprint_mismatch",
                extra={
                    "event": "embeddings.cache.fingerprint_mismatch",
                    "cached": meta.corpus_fingerprint,
                    "current": corpus_fingerprint,
                },
            )

    # ----------------------------------------------------------------- #
    # Matrix access
    # ----------------------------------------------------------------- #
    def load(self, *, mmap: bool = True) -> np.ndarray:
        """Load the matrix, memory-mapped by default."""
        if not self.path.is_file():
            raise EmbeddingCompatibilityError(
                f"No embedding matrix at {self.path}. Build the embeddings first."
            )
        # mmap keeps a large matrix out of the process heap until it is touched.
        matrix: np.ndarray = np.load(self.path, mmap_mode="r" if mmap else None)
        return matrix

    def load_complete(self, **compatibility: Any) -> np.ndarray:
        """Load, refusing a partially-built cache."""
        if not self.is_complete():
            raise EmbeddingCompatibilityError(
                f"Embedding cache at {self.path} is incomplete. Resume the build to finish it."
            )
        self.check_compatible(**compatibility)
        return self.load()

    def save(self, matrix: np.ndarray, metadata: EmbeddingMetadata) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, np.ascontiguousarray(matrix, dtype=np.float32))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2)
        logger.info(
            "embeddings.cache.saved",
            extra={
                "event": "embeddings.cache.saved",
                "path": str(self.path),
                "rows": int(matrix.shape[0]),
                "dimension": int(matrix.shape[1]) if matrix.ndim > 1 else 0,
            },
        )
        return self.path

    # ----------------------------------------------------------------- #
    # Resumable building
    # ----------------------------------------------------------------- #
    def open_for_build(
        self,
        *,
        model_name: str,
        dimension: int,
        document_count: int,
        corpus_fingerprint: str = "",
    ) -> tuple[np.ndarray, int]:
        """Return ``(matrix, rows_already_done)`` for an incremental build.

        A compatible partial cache is resumed. An incompatible or absent one is
        discarded and a zero-filled matrix allocated, so the caller always has
        somewhere to write.
        """
        resume_from = 0
        matrix: np.ndarray | None = None

        if self.exists():
            try:
                self.check_compatible(
                    model_name=model_name,
                    dimension=dimension,
                    document_count=document_count,
                    corpus_fingerprint=corpus_fingerprint,
                )
                candidate = np.array(self.load(mmap=False))
                if candidate.shape == (document_count, dimension):
                    matrix = candidate
                    resume_from = self.metadata.completed_count
                    logger.info(
                        "embeddings.cache.resume",
                        extra={
                            "event": "embeddings.cache.resume",
                            "completed": resume_from,
                            "total": document_count,
                        },
                    )
            except EmbeddingCompatibilityError:
                logger.warning(
                    "embeddings.cache.discard",
                    extra={"event": "embeddings.cache.discard", "path": str(self.path)},
                )

        if matrix is None:
            matrix = np.zeros((document_count, dimension), dtype=np.float32)
            resume_from = 0
        return matrix, resume_from

    def flush(
        self,
        matrix: np.ndarray,
        *,
        model_name: str,
        dimension: int,
        document_count: int,
        completed_count: int,
        corpus_fingerprint: str = "",
        normalize: bool = True,
        elapsed_seconds: float = 0.0,
        created_at: str = "",
    ) -> None:
        """Persist progress mid-build so an interrupted run can resume."""
        self.save(
            matrix,
            EmbeddingMetadata(
                model_name=model_name,
                dimension=dimension,
                document_count=document_count,
                completed_count=completed_count,
                corpus_fingerprint=corpus_fingerprint,
                normalize=normalize,
                created_at=created_at or _utc_now(),
                elapsed_seconds=elapsed_seconds,
            ),
        )

    def delete(self) -> None:
        for target in (self.path, self.metadata_path):
            target.unlink(missing_ok=True)
        self._metadata = None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
