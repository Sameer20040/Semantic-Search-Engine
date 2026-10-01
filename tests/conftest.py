"""Shared pytest fixtures.

Two rules the rest of the suite depends on:

* Tests must never write into the real ``data/`` tree, and must never reach the
  network. ``tmp_settings`` points every data directory at a pytest tmp_path.
* Tests must never load the sentence-transformer model (it is slow, and the
  embedding layer has its own dedicated test with a fake encoder). Anything that
  needs real vectors marks itself ``slow``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from semantic_search.core.config import DataConfig, Settings, load_settings
from semantic_search.data.loader import Document, DocumentStore


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings with all data directories redirected into a temp directory.

    ``environment="testing"`` -- not ``"test"``. ``Environment`` is a
    ``Literal["development", "production", "testing"]`` and ``"test"`` is not a
    member of it, so this fixture raised ``ValidationError`` on every request.
    It went unnoticed because nothing requested it: the suite was green, and a
    fixture that fails is invisible until something depends on it.
    """
    settings = load_settings(
        environment="testing",
        data=DataConfig(data_dir=str(tmp_path / "data")),
        corpus_size="small",
    )
    settings.ensure_dirs()
    return settings


@pytest.fixture
def sample_documents() -> list[Document]:
    """A tiny hand-written corpus with known answers.

    The documents are deliberately topically distinct so BM25 ordering and
    semantic ordering can both be asserted deterministically without a model.
    """
    bodies = [
        (
            "1",
            "Anarchism",
            "Anarchism is a political philosophy and movement. Anarchists reject "
            "hierarchical authority and advocate for self-management by the "
            "community. The term originates from the French word anarchie.",
        ),
        (
            "2",
            "Albedo",
            "In astronomy, albedo is the measure of the reflectivity of a body. "
            "The albedo of the Moon is dark grey, while fresh snow has a high "
            "albedo close to one. Albedo affects the energy balance of a planet.",
        ),
        (
            "3",
            "Machine learning",
            "Machine learning is a field of study in artificial intelligence. "
            "A machine learning model learns patterns from data rather than "
            "having rules programmed explicitly by a human.",
        ),
        (
            "4",
            "Retrieval augmented generation",
            "Retrieval augmented generation combines a search engine with a large "
            "language model. The system first retrieves relevant documents, then "
            "conditions the model on that context to produce a grounded answer.",
        ),
        (
            "5",
            "Anatomy of the human heart",
            "The human heart is a muscular organ that pumps blood through the "
            "circulatory system. It has four chambers: two atria and two ventricles.",
        ),
    ]
    return [
        Document(
            id=doc_id, title=title, text=text, metadata={"url": f"https://example.org/{doc_id}"}
        )
        for doc_id, title, text in bodies
    ]


@pytest.fixture
def document_store(sample_documents: list[Document]) -> DocumentStore:
    return DocumentStore(sample_documents)


@pytest.fixture
def autoclean() -> Iterator[None]:
    """Placeholder hook for future per-test cleanup; intentionally empty."""
    yield
