"""Download and preprocess the Wikipedia corpus.

Usage::

    python scripts/download_data.py --limit 2000
    python scripts/download_data.py --limit 200 --strategy head --force

Writes ``data/raw/wikipedia_raw.jsonl`` (untouched records) and
``data/processed/corpus.jsonl`` (cleaned documents). The raw file is cached, so
re-running is free unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search.core.config import load_settings
from semantic_search.core.logging import configure_logging, get_logger
from semantic_search.data.downloader import download_wikipedia, load_raw_records
from semantic_search.data.loader import DocumentStore
from semantic_search.data.preprocessor import TextPreprocessor
from semantic_search.data.validator import validate_corpus

logger = get_logger("scripts.download_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Document count.")
    parser.add_argument(
        "--strategy",
        choices=("topics", "head", "sample"),
        default="topics",
        help="Sampling strategy (default: topics, which gives the most diverse corpus).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument("--env", default="development", help="Configuration environment.")
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip corpus validation (not recommended)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(environment=args.env)
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_dirs()

    started = time.perf_counter()
    download_wikipedia(
        settings,
        limit=args.limit,
        strategy=args.strategy,
        force=args.force,
        seed=args.seed,
    )
    raw = load_raw_records(settings)
    logger.info("raw.loaded", extra={"event": "raw.loaded", "documents": len(raw)})

    preprocessor = TextPreprocessor(
        min_length=settings.data.min_text_length, max_length=settings.data.max_text_length
    )
    documents = []
    dropped = 0
    for record in raw:
        try:
            record.text = preprocessor.process(record.text, title=record.title)
            documents.append(record)
        except ValueError:
            dropped += 1
    if dropped:
        logger.warning(
            "preprocess.dropped", extra={"event": "preprocess.dropped", "dropped": dropped}
        )

    store = DocumentStore(documents)
    output = store.save_jsonl(settings.data.corpus_file)
    logger.info(
        "corpus.written",
        extra={
            "event": "corpus.written",
            "path": str(output),
            "documents": len(store),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
    )

    if not args.no_validate:
        report = validate_corpus(
            documents,
            min_length=settings.data.min_text_length,
            max_length=settings.data.max_text_length,
        )
        logger.info("validation", extra={"event": "validation", "summary": report.summary()})
        if not report.valid:
            logger.error(
                "validation.failed",
                extra={
                    "event": "validation.failed",
                    "errors": [str(issue) for issue in report.errors[:10]],
                },
            )
            return 1
        print("\nCorpus statistics")
        for key, value in report.stats.items():
            print(f"  {key:38s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
