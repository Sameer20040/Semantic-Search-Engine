"""The FastAPI application factory.

One function, :func:`create_app`, that assembles the whole HTTP surface. The
reason it is a factory rather than a module-level ``app = FastAPI(...)`` is
practical: a module-level app is constructed at import time, which means
importing this module loads the corpus, the indexes, and the encoder. That
makes the tests slow and unable to construct two differently-configured apps
in one session, and it makes ``uvicorn --reload`` re-download a model on every
save.

What the factory wires, and in what order
------------------------------------------
1. FastAPI with the configured title, version, and docs URLs. ``debug`` is
   never enabled -- see the note on production below.
2. Exception handlers, so a project exception becomes the standard envelope.
3. ``RequestContextMiddleware``, outermost, so the request id exists for every
   exchange including the ones that fail in middleware.
4. CORS, from config, refusing a wildcard origin in production.
5. The routers.
6. A lifespan hook that loads the artifacts once and publishes them to
   ``app.state``.

On ``debug``
-------------
``FastAPI(debug=True)`` installs a middleware that returns the full traceback
to the client. It is not wired to a flag here. Instead,
``api.expose_stack_traces`` controls only whether *our* 500 handler includes a
traceback, which is a narrower and safer knob, and ``environment`` controls
whether the docs endpoints are served at all. There is therefore no single
setting that turns on both verbose errors and the interactive debugger.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from semantic_search.api.dependencies import (
    SERVICE_KEY,
    SETTINGS_KEY,
    STARTUP_SECONDS_KEY,
    load_service,
    set_api_state,
)
from semantic_search.api.errors import register_exception_handlers
from semantic_search.api.middleware import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    check_cors_configuration,
)
from semantic_search.api.routes import search as search_routes
from semantic_search.api.routes import system as system_routes
from semantic_search.core.config import Settings, load_settings
from semantic_search.core.logging import get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Hybrid retrieval over a 300-article Wikipedia corpus: BM25, a
sentence-transformer vector index, and reciprocal rank fusion over both.

**What the numbers in a response mean.** `score` is a fused RRF score, which is
rank arithmetic, not a relevance probability. `raw_score` in the provenance is
the underlying retriever's own score, and the two retrievers' raw scores are on
different scales and are not comparable with each other. `rank` is 1-based;
`provenance[].rank` is 0-based, as it is everywhere inside the engine.

**What this API does not tell you.** It does not report that a result is
relevant. Ranking and relevance are different things, and this system has
evidence for the first and none for the second.

**What has been measured.** Latency and behaviour on this corpus and these
configurations. Retrieval *quality* has not been fully evaluated: the
relevance judgments for the topical query stratum are still outstanding, so no
accuracy claim is made anywhere in this service.
""".strip()

TAGS_METADATA = [
    {"name": "search", "description": "Run a search and inspect how it was ordered."},
    {"name": "system", "description": "Health, readiness, and effective configuration."},
]


def create_app(settings: Settings | None = None, *, load_artifacts: bool = True) -> FastAPI:
    """Build the application.

    Args:
        settings: Resolved configuration. Read from the environment, the YAML
            file, and ``.env`` when omitted.
        load_artifacts: Whether the lifespan hook loads the corpus, indexes, and
            encoder. ``False`` is for tests that want the HTTP surface without
            a 300 MB model, and for ``--check`` style smoke tests. The service
            is then absent from ``app.state`` and ``/health/ready`` correctly
            reports 503.

    Returns:
        A configured :class:`FastAPI` instance. Nothing is loaded until the
        lifespan runs, so constructing this is cheap.
    """
    resolved = settings if settings is not None else load_settings()
    api_config = resolved.api

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        if load_artifacts:
            try:
                service, elapsed = load_service(resolved)
            except Exception:
                # Startup validation failures are re-raised: a server that
                # starts and then 503s every search is harder to diagnose than
                # one that refuses to start and says which file is missing.
                logger.error(
                    "startup validation failed; refusing to serve. Run "
                    "python scripts/build_index.py and check the data paths in "
                    "your configuration."
                )
                raise
            set_api_state(application, service, resolved)
            setattr(application.state, STARTUP_SECONDS_KEY, elapsed)
        else:
            # Only default the state when nothing has been published already. A
            # caller that assembled the service itself before entering the
            # lifespan -- the integration tests, or a harness that injects a
            # stub -- must not have it cleared here, or the app would come up
            # correctly wired and then serve 503s.
            if getattr(application.state, SERVICE_KEY, None) is None:
                application.state[SETTINGS_KEY] = resolved
                application.state[SERVICE_KEY] = None
        yield
        # Teardown is a no-op on purpose. The model, the FAISS index, and the
        # BM25 index hold native resources with no explicit close; letting the
        # process exit release them is correct, and faking a close that does
        # nothing would be worse than none.

    application = FastAPI(
        title=api_config.title,
        version=resolved.version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        docs_url=api_config.docs_url,
        redoc_url=api_config.redoc_url,
        lifespan=lifespan,
        # Never True. See the module docstring; exposing tracebacks is
        # controlled by api.expose_stack_traces instead.
        debug=False,
    )

    register_exception_handlers(application, expose_stack_traces=api_config.expose_stack_traces)

    # Refused here, before any request can be served, because an unsafe CORS
    # policy is not something an operator should discover from a report.
    check_cors_configuration(api_config, str(resolved.environment))

    # Middleware is added in reverse execution order: Starlette runs the last
    # added first. RequestContext goes on last so it is outermost, which is
    # what makes the request id available to everything inside it.
    if api_config.cors_origins:
        wildcard = "*" in api_config.cors_origins
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_config.cors_origins),
            # Credentials with a wildcard origin is rejected by browsers anyway,
            # so pairing them is a configuration that looks enabled and is not.
            allow_credentials=not wildcard,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", REQUEST_ID_HEADER],
        )

    application.add_middleware(RequestContextMiddleware, api_config=api_config)

    application.include_router(system_routes.router)
    application.include_router(search_routes.router)
    return application


def get_application() -> FastAPI:
    """Entry point for ``uvicorn semantic_search.api.app:get_application --factory``.

    The ``--factory`` flag is what makes this work: uvicorn calls the function
    and uses its return value, so nothing is constructed at import time.
    """
    return create_app()


__all__: list[str] = ["DESCRIPTION", "create_app", "get_application"]
