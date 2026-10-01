"""Structured logging setup.

Two formatters are available:

* ``console`` -- human-readable, coloured-ish, used in development.
* ``json``    -- one JSON object per line, used in production so log shippers
  (CloudWatch, Loki, Datadog) can parse fields without regex.

Every record carries the standard ``extra`` fields used across the project:
``event``, ``request_id``, ``duration_ms``, ``method``, ``query_len``...
Query text itself is never logged unless ``SSE_LOG_QUERIES`` is explicitly set,
because user queries can carry sensitive content.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CONFIGURED = False

#: Attributes present on every LogRecord; anything else was injected via ``extra``.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line format with a short logger name."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Install the root handler. Idempotent across repeated calls."""
    global _CONFIGURED
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party libraries are noisy at DEBUG; keep them at WARNING unless the
    # operator explicitly asked for DEBUG.
    noisy = ("httpx", "urllib3", "filelock", "datasets", "huggingface_hub", "sentence_transformers")
    for name in noisy:
        logging.getLogger(name).setLevel(
            logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING
        )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log the duration of a block, adding ``duration_ms`` to the record.

    Usage::

        with timed(logger, "search.completed", method="hybrid") as fields:
            ...
        fields["result_count"]
    """
    import time

    start = time.perf_counter()
    context: dict[str, Any] = {}
    try:
        yield context
    except Exception as exc:
        logger.error(
            event,
            extra={
                "event": event,
                "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **context,
                **fields,
            },
        )
        raise
    else:
        logger.info(
            event,
            extra={
                "event": event,
                "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                **context,
                **fields,
            },
        )
