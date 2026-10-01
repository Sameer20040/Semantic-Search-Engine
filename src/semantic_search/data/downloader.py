"""Corpus acquisition from the Hugging Face ``wikimedia/wikipedia`` dataset.

Design notes
------------
* **Streaming by default.** ``wikimedia/wikipedia`` is ~20 GB compressed. The
  downloader iterates the parquet shards via ``datasets`` streaming mode and
  stops at the configured limit, so a 2 000-document development corpus never
  materialises the full dataset.
* **Local cache first.** A previously written ``data/raw/corpus.jsonl`` is reused
  unless ``force=True``, which makes repeat builds fast and offline.
* **Deterministic subset.** Wikipedia's article order is not alphabetical, so a
  "first N documents" slice is arbitrary but *stable* for a given dataset
  version. ``--sample-strategy`` can instead take a seeded random sample, which
  spreads topics far better -- important for a demo corpus, where a
  head-of-corpus slice is dominated by a few hub articles.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from semantic_search.core.config import Settings
from semantic_search.core.exceptions import DatasetDownloadError, EmptyCorpusError
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import Document
from semantic_search.data.preprocessor import clean_title

logger = get_logger(__name__)

SampleStrategy = Literal["head", "sample", "topics"]


def _raw_path(settings: Settings) -> Path:
    return settings.data.raw_dir / "wikipedia_raw.jsonl"


def _iter_stream(dataset_name: str, dataset_config: str) -> Iterator[dict[str, Any]]:
    """Yield raw records from the dataset, streaming, with actionable errors."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - datasets is a hard dependency
        raise DatasetDownloadError(
            "The `datasets` package is required to download the corpus. "
            "Install it with: pip install datasets"
        ) from exc

    # Newest `datasets` first: `trust_remote_code` was removed, and passing it
    # makes the library log a loud ERROR before the TypeError is even raised.
    # Older builds need it for script-based datasets, so keep it as a fallback.
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    except TypeError:
        try:
            dataset = load_dataset(
                dataset_name, dataset_config, split="train", streaming=True, trust_remote_code=True
            )
        except Exception as exc:
            raise DatasetDownloadError(
                f"Could not load dataset {dataset_name!r}/{dataset_config!r}: {exc}"
            ) from exc
    except Exception as exc:
        raise DatasetDownloadError(
            f"Could not load dataset {dataset_name!r}/{dataset_config!r}: {exc}. "
            "Check network access and the dataset name in your configuration."
        ) from exc

    yield from dataset


def _select(
    records: Iterator[dict[str, Any]],
    limit: int,
    strategy: SampleStrategy,
    oversample: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply the sampling strategy to a (possibly truncated) record stream."""
    if strategy == "head":
        # take() on an iterator: bounded, no need to over-read.
        selected: list[dict[str, Any]] = []
        for record in records:
            selected.append(record)
            if len(selected) >= limit:
                break
        return selected

    # Random/topics sampling needs a larger pool than the final limit to be
    # meaningfully spread out, so over-read by a configurable factor.
    pool: list[dict[str, Any]] = []
    for record in records:
        pool.append(record)
        if len(pool) >= limit * max(1, oversample):
            break

    if not pool:
        return []
    if strategy == "sample":
        random.Random(seed).shuffle(pool)
        return pool[:limit]

    # "topics": bucket round-robin by first title character so a diverse mix of
    # subject areas survives even a very small limit.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in pool:
        title = str(record.get("title", "?"))[:1].upper() or "?"
        buckets.setdefault(title, []).append(record)
    ordered: list[dict[str, Any]] = []
    while len(ordered) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(ordered) < limit:
                ordered.append(buckets[key].pop())
    return ordered


def download_wikipedia(
    settings: Settings,
    *,
    limit: int | None = None,
    strategy: SampleStrategy = "topics",
    force: bool = False,
    seed: int = 42,
    oversample: int = 5,
) -> Path:
    """Fetch raw Wikipedia records and cache them as JSONL.

    Args:
        settings: Active configuration (supplies dataset name and raw dir).
        limit: Document count. Defaults to ``settings.corpus_limit``.
        strategy: ``head`` (dataset order), ``sample`` (seeded random), or
            ``topics`` (round-robin by title initial -- the default, because it
            yields the most *diverse* demo corpus).
        force: Re-download even if the cache exists.
        seed: Seed for ``sample``.
        oversample: Read ``limit * oversample`` records when sampling.

    Returns:
        Path to the raw JSONL file.

    Raises:
        DatasetDownloadError: The dataset could not be reached.
        EmptyCorpusError: The dataset yielded no usable records.
    """
    target = _raw_path(settings)
    document_limit = limit if limit is not None else settings.corpus_limit
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not force:
        existing = sum(1 for _ in target.open("r", encoding="utf-8") if _.strip())
        if existing >= min(document_limit, 1):
            logger.info(
                "corpus.cache_hit",
                extra={"event": "corpus.cache_hit", "path": str(target), "documents": existing},
            )
            return target

    logger.info(
        "corpus.download.start",
        extra={
            "event": "corpus.download.start",
            "dataset": settings.data.dataset_name,
            "config": settings.data.dataset_config,
            "limit": document_limit,
            "strategy": strategy,
        },
    )

    try:
        records = _select(
            _iter_stream(settings.data.dataset_name, settings.data.dataset_config),
            document_limit,
            strategy,
            oversample,
            seed,
        )
    except DatasetDownloadError:
        raise
    except Exception as exc:
        raise DatasetDownloadError(f"Corpus download failed: {exc}") from exc

    if not records:
        raise EmptyCorpusError(
            f"Dataset {settings.data.dataset_name!r} yielded no records. "
            "Check the dataset config name and network access."
        )

    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            row = {
                "id": str(record.get("id", "")),
                "title": str(record.get("title", "")),
                "text": str(record.get("text", "")),
                "url": record.get("url", ""),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "corpus.download.done",
        extra={"event": "corpus.download.done", "path": str(target), "documents": written},
    )
    return target


def load_raw_records(settings: Settings) -> list[Document]:
    """Read the cached raw JSONL, converting rows to :class:`Document`.

    Bodies are returned *uncleaned* -- cleaning is the build script's job, so
    that this stays a pure read of what was downloaded.

    Malformed rows are skipped with a warning rather than aborting a long build.
    """
    path = _raw_path(settings)
    if not path.is_file():
        raise DatasetDownloadError(
            f"No raw corpus found at {path}. Run `make data` (or scripts/download_data.py) first."
        )
    documents: list[Document] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                documents.append(
                    Document(
                        id=str(row["id"]),
                        title=clean_title(str(row.get("title", ""))),
                        text=str(row.get("text", "")),
                        title_raw=str(row.get("title", "")),
                        metadata={
                            "url": row.get("url", ""),
                            "source": settings.data.dataset_name,
                            "dataset_config": settings.data.dataset_config,
                        },
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                skipped += 1
    if skipped:
        logger.warning(
            "corpus.raw.skipped",
            extra={"event": "corpus.raw.skipped", "skipped": skipped, "path": str(path)},
        )
    if not documents:
        raise EmptyCorpusError(f"Raw corpus at {path} contained no valid documents.")
    return documents


def corpus_fingerprint(settings: Settings, document_ids: list[str]) -> str:
    """Stable short hash of the corpus contents, stored in index metadata."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(settings.data.dataset_name.encode())
    digest.update(settings.data.dataset_config.encode())
    for doc_id in document_ids:
        digest.update(doc_id.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]
