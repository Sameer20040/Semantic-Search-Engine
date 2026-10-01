"""Build the BM25 and FAISS indexes from the processed corpus.

The two halves of the hybrid system are built here, in one place, so that a
single command produces an index set that is known to be mutually consistent --
the same document store, the same order, the same embedding model.

Encoding is the slow part and is resumable: the embedding cache is flushed
periodically, so an interrupted run picks up where it stopped instead of
re-encoding the corpus. A cache built by a different model, or from a different
corpus, is discarded rather than reused.

Usage::

    python scripts/build_index.py
    python scripts/build_index.py --force --index-type hnsw
    python scripts/build_index.py --limit 100 --skip-embeddings
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from semantic_search.core.config import load_settings
from semantic_search.core.exceptions import SemanticSearchError
from semantic_search.core.logging import configure_logging, get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.embeddings.cache import EmbeddingCache
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Searcher
from semantic_search.index.faiss_index import FaissIndex

logger = get_logger("build_index")

#: How often to write the embedding cache to disk during a long build.
FLUSH_EVERY = 25


def corpus_fingerprint(store: DocumentStore) -> str:
    """A short digest of document identity and order.

    Order matters as much as identity: the FAISS row ids are positions, so
    reordering the corpus invalidates every vector even though the documents
    themselves are unchanged.
    """
    digest = hashlib.sha256()
    for document in store:
        digest.update(document.id.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(f"|{len(store)}|".encode())
    return digest.hexdigest()[:16]


def build_embeddings(
    store: DocumentStore,
    settings: object,
    *,
    force: bool,
) -> tuple[np.ndarray, str, float, bool]:
    """Return ``(matrix, fingerprint, encode_seconds, from_cache)``.

    ``encode_seconds`` is 0.0 on a cache hit, and ``from_cache`` says so
    explicitly. Reporting the *original* build's duration here would read like
    a fresh measurement when nothing was actually encoded.
    """
    config = settings.embedding  # type: ignore[attr-defined]
    data_config = settings.data  # type: ignore[attr-defined]
    fingerprint = corpus_fingerprint(store)
    cache = EmbeddingCache(data_config.index_dir / config.cache_filename)

    if force and cache.exists():
        logger.info("embeddings.cache.forced_rebuild")
        cache.delete()

    # The fast path: a complete, compatible cache needs no encoder at all,
    # which also means no 262 MB model load.
    if cache.is_complete():
        try:
            matrix = cache.load_complete(
                model_name=config.model_name,
                dimension=cache.metadata.dimension,
                document_count=len(store),
                corpus_fingerprint=fingerprint,
            )
            logger.info(
                "embeddings.cache.hit",
                extra={
                    "event": "embeddings.cache.hit",
                    "documents": matrix.shape[0],
                    "elapsed_seconds": cache.metadata.elapsed_seconds,
                },
            )
            return matrix, fingerprint, 0.0, True
        except SemanticSearchError as exc:
            logger.warning("embeddings.cache.unusable: %s", exc)

    encoder = EmbeddingEncoder(config)
    print(f"  model      {config.model_name} ({encoder.dimension}d, {encoder.device})")
    print(f"  documents  {len(store)}")

    started = time.perf_counter()
    matrix, resume_from = cache.open_for_build(
        model_name=config.model_name,
        dimension=encoder.dimension,
        document_count=len(store),
        corpus_fingerprint=fingerprint,
    )
    if resume_from:
        print(f"  resuming   at row {resume_from} (partial cache found)")

    batch = max(1, config.batch_size)
    # FLUSH_EVERY is a document count; convert to whole batches so a flush never
    # lands mid-batch and rewrites a partially-embedded batch.
    flush_batches = max(1, FLUSH_EVERY // batch)

    for start in range(resume_from, len(store), batch):
        end = min(start + batch, len(store))
        chunk = store.at(range(start, end))
        matrix[start:end] = encoder.encode_documents(chunk)

        done = end
        if done % (batch * flush_batches) == 0 or done == len(store):
            elapsed = time.perf_counter() - started
            rate = (done - resume_from) / elapsed if elapsed > 0 else 0.0
            print(
                f"  encoded    {done}/{len(store)}  ({rate:.1f} docs/s)",
                flush=True,
            )
            cache.flush(
                matrix,
                model_name=config.model_name,
                dimension=encoder.dimension,
                document_count=len(store),
                completed_count=done,
                corpus_fingerprint=fingerprint,
                normalize=config.normalize,
                elapsed_seconds=elapsed,
            )

    total_seconds = time.perf_counter() - started
    cache.flush(
        matrix,
        model_name=config.model_name,
        dimension=encoder.dimension,
        document_count=len(store),
        completed_count=len(store),
        corpus_fingerprint=fingerprint,
        normalize=config.normalize,
        elapsed_seconds=total_seconds,
    )
    return matrix, fingerprint, total_seconds, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="development", help="config environment")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N documents")
    parser.add_argument("--index-type", default="", help="override faiss.index_type")
    parser.add_argument("--force", action="store_true", help="re-encode from scratch")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="build the FAISS index from the existing cache only",
    )
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    settings = load_settings(environment=args.env)
    configure_logging(settings.log_level, settings.log_format)
    data_config = settings.data

    corpus_path = data_config.corpus_file
    if not corpus_path.is_file():
        print(
            f"error: no corpus at {corpus_path}\n" f"       run: python scripts/download_data.py",
            file=sys.stderr,
        )
        return 1

    store = DocumentStore.load_jsonl(corpus_path)
    if args.limit:
        store = DocumentStore(list(store)[: args.limit])
    print(f"corpus      {len(store)} documents from {corpus_path}")

    if not store:
        print("error: corpus is empty", file=sys.stderr)
        return 1

    index_dir = data_config.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- BM25
    if not args.skip_bm25:
        started = time.perf_counter()
        searcher = BM25Searcher.from_documents(store, config=settings.bm25)
        build_seconds = time.perf_counter() - started
        path = searcher.index.save(index_dir / settings.bm25.index_filename)
        stats = searcher.stats()
        print(
            f"bm25        {stats['vocabulary_size']} terms, "
            f"{path.stat().st_size / 1024:.0f} KiB, built in {build_seconds * 1000:.0f} ms"
        )

    # --------------------------------------------------------------- FAISS
    if (
        args.skip_embeddings
        and not (data_config.index_dir / settings.embedding.cache_filename).is_file()
    ):
        print("error: --skip-embeddings given but no embedding cache exists", file=sys.stderr)
        return 1

    matrix, fingerprint, encode_seconds, from_cache = build_embeddings(
        store, settings, force=args.force
    )

    faiss_config = settings.faiss
    if args.index_type:
        faiss_config = faiss_config.model_copy(update={"index_type": args.index_type})

    started = time.perf_counter()
    index = FaissIndex.build(
        matrix,
        faiss_config,
        embedding_model=settings.embedding.model_name,
        normalized=settings.embedding.normalize,
    )
    build_seconds = time.perf_counter() - started
    path = index.save(index_dir / faiss_config.index_filename)

    # ---------------------------------------------------------------------
    # Immediately verify the artifact we just wrote, rather than trusting it.
    # An index that saves but does not reload is worse than no index.
    # ---------------------------------------------------------------------
    reloaded = FaissIndex.load(path, expected_model=settings.embedding.model_name)
    probe = matrix[0]
    if reloaded.search(probe, top_k=1)[0].row != 0:
        print("error: rebuilt index does not reproduce its own self-match", file=sys.stderr)
        return 1

    size_mib = path.stat().st_size / (1024 * 1024)
    print(
        f"faiss       {reloaded.index_type}, {len(reloaded)} vectors x "
        f"{reloaded.dimension}d, {size_mib:.1f} MiB, built in {build_seconds * 1000:.0f} ms"
    )
    if from_cache:
        print(f"encode      cached ({len(store)} documents, no model load)")
    else:
        print(f"encode      {encode_seconds:.1f} s for {len(store)} documents")
    print(f"fingerprint {fingerprint}")
    print(f"artifacts   {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
