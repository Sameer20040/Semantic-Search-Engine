"""Sentence-BERT encoding.

Model choice
------------
``all-MiniLM-L6-v2``: 384 dimensions, 6 layers, 33M parameters, ~22M sentences
per second on a modern GPU and a few hundred per second on CPU. It is small
enough to embed a 200 000-document corpus on a CPU-only machine, which matters
because this project must not require a GPU. The trade-off is that it truncates
at 256 word-pieces, so each vector summarises the head of an article rather than
the whole thing -- acceptable here because Wikipedia leads with a definitional
paragraph, which is the most representative part.

Device selection
----------------
CUDA is used when available and not explicitly disabled, otherwise CPU. The
detection is lazy so that merely importing this module does not initialise
torch, which keeps CLI startup fast.

Determinism
-----------
The same text always produces the same vector for a given model, which is what
makes the embedding cache safe to resume. Batching does not change results
because each sequence is encoded independently (mean pooling happens per row).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from semantic_search.core.config import EmbeddingConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging import get_logger
from semantic_search.data.preprocessor import build_embedding_text

logger = get_logger(__name__)


def detect_device(preferred: str | None = None) -> str:
    """Resolve a torch device name without importing torch unless needed.

    Args:
        preferred: ``"auto"``/``None`` to detect, or an explicit device string
            such as ``"cpu"`` or ``"cuda:0"``.

    Returns:
        A device string suitable for ``SentenceTransformer(..., device=...)``.
    """
    if preferred and preferred not in ("auto", "", None):
        return preferred
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


class EmbeddingEncoder:
    """Wraps a Sentence-Transformers model with batching, progress, and timing."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Create an encoder.

        Args:
            config: Model name, batch size, sequence length, normalisation.
            device: Override device selection.
            model: Pre-loaded model, primarily for tests. Supplying this skips
                the Hugging Face download entirely.
        """
        self.config = config or EmbeddingConfig()
        self.device = device or detect_device(self.config.device)
        self._model = model
        self._dimension: int | None = None

    # ----------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------- #
    @property
    def model(self) -> Any:
        """Lazily load the model. Loading is deferred so import stays cheap."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - hard dependency
                raise ConfigurationError(
                    "sentence-transformers is required for embedding generation. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            logger.info(
                "encoder.loading",
                extra={
                    "event": "encoder.loading",
                    "model": self.config.model_name,
                    "device": self.device,
                },
            )
            # No cache_folder is passed: the standard Hugging Face cache is
            # used, so the model is shared with the rest of the ecosystem and a
            # pre-populated cache needs no configuration.
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.device,
            )
            self._model.max_seq_length = self.config.max_seq_length
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding width, read from the model on first use."""
        if self._dimension is None:
            model = self.model
            # sentence-transformers 5.x renamed this to get_embedding_dimension
            # and warns on the old name; support both, prefer the new one.
            get_dimension = getattr(model, "get_embedding_dimension", None) or getattr(
                model, "get_sentence_embedding_dimension", None
            )
            if callable(get_dimension):
                self._dimension = int(get_dimension())
            else:  # pragma: no cover - defensive
                self._dimension = int(model.encode(["dimension probe"]).shape[1])
        return self._dimension

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "batch_size": self.config.batch_size,
            "max_seq_length": self.config.max_seq_length,
            "normalize": self.config.normalize,
        }

    # ----------------------------------------------------------------- #
    # Encoding
    # ----------------------------------------------------------------- #
    def encode(
        self,
        texts: Sequence[str] | Iterable[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
        description: str = "encoding",
    ) -> np.ndarray:
        """Encode texts into a ``(n, dimension)`` float32 matrix.

        An empty input returns a correctly-shaped zero-row array rather than
        raising, so callers do not need a special case.
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)

        started = time.perf_counter()
        vectors = self.model.encode(
            items,
            batch_size=batch_size or self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        elapsed = time.perf_counter() - started
        logger.info(
            "encoder.encoded",
            extra={
                "event": "encoder.encoded",
                "description": description,
                "texts": len(items),
                "elapsed_seconds": round(elapsed, 2),
                "texts_per_second": round(len(items) / elapsed, 1) if elapsed > 0 else None,
            },
        )
        return matrix

    def encode_documents(
        self,
        documents: Iterable[Any],
        *,
        title_max_length: int = 120,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode documents using the title-prefixed representation.

        Prefixing the title supplies the subject that a bare body paragraph
        lacks, which measurably improves recall for short topical queries.
        """
        texts = [
            build_embedding_text(
                getattr(document, "title", ""),
                getattr(document, "text", ""),
                title_max_length,
            )
            for document in documents
        ]
        return self.encode(texts, show_progress=show_progress, description="documents")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query into a ``(1, dimension)`` row.

        A query is prefixed with nothing: it is already a natural sentence-like
        phrase, which is what the model was trained on.
        """
        return self.encode([query], description="query")
