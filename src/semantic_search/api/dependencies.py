"""Shared state and the FastAPI dependencies that read it.

Why the state lives on ``app.state``
------------------------------------
The heavy artifacts -- the encoder, the BM25 index, the FAISS index, the
corpus store -- are loaded **once** at startup and shared by every request.
Loading them per request would make the cheapest endpoint in the system the
most expensive one, and it would mean a process's memory footprint was a
function of its traffic.

Storing them on ``app.state`` rather than in a module global is deliberate: a
module global is shared by every app instance in the process, which makes
"two apps in one test session with different settings" impossible and makes
teardown meaningless. ``app.state`` is per-application, so a test gets its own.

The dependency functions below are the *only* way routes reach that state, so
there is one place to look when asking "where does the service come from".

Startup validation
------------------
:func:`load_service` performs its checks **before** anything is published to
``app.state``. A server that starts successfully and then fails the first
search is worse than one that refuses to start: the failure surfaces as a 500
per request instead of once, at boot, with a message naming the missing file.
So a missing index is a startup failure, not a request failure.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, cast

from fastapi import Depends, Request

from semantic_search.app.service import ApplicationSearchService
from semantic_search.core.config import Settings
from semantic_search.core.exceptions import StartupValidationError
from semantic_search.core.logging import get_logger
from semantic_search.data.loader import DocumentStore
from semantic_search.index.bm25 import BM25Index, BM25Searcher
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

logger = get_logger(__name__)

#: Key under which the assembled service is published on ``app.state``.
SERVICE_KEY = "search_service"
#: Key under which the settings snapshot is published.
SETTINGS_KEY = "settings"
#: Key under which the artifact-loading wall-clock time is published.
STARTUP_SECONDS_KEY = "startup_seconds"


def load_service(settings: Settings) -> tuple[ApplicationSearchService, float]:
    """Load every artifact and assemble the application search service.

    This is the one place the API layer touches the filesystem. Everything
    downstream -- routes, the UI, the evaluation harness -- receives an already
    assembled service, so a route can never accidentally serve a different
    corpus from the one the probes measured.

    Args:
        settings: The resolved configuration.

    Returns:
        The service, and the seconds spent loading.

    Raises:
        StartupValidationError: Any artifact is missing, empty, or inconsistent.
            Every message names the path that was checked, because the fix for
            this failure is always "run the build step or point at the right
            directory" and the operator needs to know which one.
    """
    started = time.perf_counter()

    corpus_file = settings.data.corpus_file
    if not corpus_file.exists():
        raise StartupValidationError(
            f"corpus file not found at {corpus_file}. Build it first: "
            f"python scripts/download_data.py && python -m semantic_search.data.pipeline"
        )
    store = DocumentStore.load_jsonl(corpus_file)
    if len(store) == 0:
        raise StartupValidationError(
            f"{corpus_file} contains no documents. A search over an empty corpus "
            f"returns nothing for every query, which is indistinguishable from a "
            f"working system that found no matches."
        )

    bm25_path = settings.data.index_dir / settings.bm25.index_filename
    if not bm25_path.exists():
        raise StartupValidationError(
            f"BM25 index not found at {bm25_path}. Build it first: "
            f"python scripts/build_index.py"
        )
    bm25_index = BM25Index.load(bm25_path)
    bm25 = BM25Searcher(bm25_index, store)

    # SemanticSearchService.from_settings owns the encoder and the FAISS index,
    # including the checks that the index was built by the model named in the
    # config. It is the same factory the probes use, so the server cannot serve
    # a different index from the one that was measured.
    try:
        semantic = SemanticSearchService.from_settings(settings, store=store)
    except StartupValidationError:
        raise
    except Exception as exc:
        raise StartupValidationError(
            f"failed to load the semantic index and encoder: {type(exc).__name__}: {exc}. "
            f"Rebuild with: python scripts/build_index.py"
        ) from exc

    # A BM25 index and a FAISS index of different lengths cannot be fused: RRF
    # merges by position in two lists, and the shorter one would silently
    # contribute nothing beyond its own length. This is checked at startup
    # because the alternative is a result set that quietly loses a retriever.
    if len(bm25_index) != len(store):
        raise StartupValidationError(
            f"the BM25 index holds {len(bm25_index)} documents but the corpus holds "
            f"{len(store)}. Rebuild the index against the current corpus."
        )

    hybrid = HybridSearchService(bm25, semantic, hybrid_config=settings.hybrid)
    service = ApplicationSearchService.from_settings(settings, store, hybrid)

    elapsed = time.perf_counter() - started
    logger.info(
        "artifacts loaded: %d documents, %s, in %.2fs",
        len(store),
        "/".join(hybrid.retriever_names) or "no retrievers",
        elapsed,
    )
    return service, elapsed


def set_api_state(app: Any, service: ApplicationSearchService, settings: Settings) -> None:
    """Publish the assembled service onto the application state."""
    app.state[SERVICE_KEY] = service
    app.state[SETTINGS_KEY] = settings
    app.state[STARTUP_SECONDS_KEY] = getattr(app.state, STARTUP_SECONDS_KEY, 0.0)


def get_service(request: Request) -> ApplicationSearchService:
    """FastAPI dependency: the assembled search service.

    Raises:
        StartupValidationError: If the state is missing, which means the
            lifespan hook did not run. Reported as a 503 rather than an
            AttributeError, because "the app is not wired" and "the app is
            broken" are both the operator's problem, not the caller's.
    """
    service = getattr(request.app.state, SERVICE_KEY, None)
    if service is None:
        raise StartupValidationError(
            "the search service is not loaded. The application lifespan did not "
            "complete, so the server is not ready to serve requests."
        )
    # getattr off Starlette's State is typed Any; the None check above is the
    # runtime guarantee, and the cast is what tells mypy so.
    return cast(ApplicationSearchService, service)


def get_api_settings(request: Request) -> Settings:
    """FastAPI dependency: the settings this process was started with."""
    settings = getattr(request.app.state, SETTINGS_KEY, None)
    if settings is None:
        raise StartupValidationError("settings are not loaded on the application state.")
    return cast(Settings, settings)


#: Annotated dependency aliases, so route signatures read as the thing they
#: need rather than as a call expression.
ServiceDep = Annotated[ApplicationSearchService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_api_settings)]
