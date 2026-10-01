"""Health, readiness, and configuration endpoints.

Three endpoints that answer three different questions, kept separate because
conflating them is how a deployment ends up restarting a healthy process.

``GET /health/live``
    "Is this process running?" It does not touch the indexes, does not run a
    query, and returns 200 as long as the event loop can serve a request. This
    is the liveness probe: if it fails, the process should be restarted.

``GET /health/ready``
    "Can this process serve a search?" It reports whether the artifacts loaded
    at startup. A 503 here means "not ready, try again shortly" -- which is a
    wait, not a restart.

``GET /health``
    Both, plus the configuration an operator needs to interpret them.

The split is not ceremony. A liveness probe that also checks the index causes
the orchestrator to kill and restart a process whose index is temporarily
unreadable, which turns a recoverable blip into a crash loop. A readiness probe
that returns 200 while the index is missing means traffic is routed to a process
that answers every search with a 503.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status

from semantic_search.api.dependencies import (
    SERVICE_KEY,
    SETTINGS_KEY,
    STARTUP_SECONDS_KEY,
    ServiceDep,
)
from semantic_search.api.models import ErrorResponse, HealthResponse, StatsResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health/live",
    summary="Liveness probe",
    response_model=dict[str, Any],
)
def liveness() -> dict[str, Any]:
    """Return 200 if the process is able to serve a request at all.

    Deliberately touches nothing. It exists to answer "is this process alive",
    and any work done here is work a probe does thousands of times a day.
    """
    return {"status": "alive"}


def _health_payload(request: Request, response: Response) -> HealthResponse:
    """Build the health body, and set the status code to match it.

    The two ready and not-ready branches appear once, here, rather than in each
    route. The failure mode of duplicating them is not a bug in the code -- it
    is a not-ready branch that grows a field the ready branch does not, and a
    client that learns to read a field it only sometimes receives.
    """
    service = getattr(request.app.state, SERVICE_KEY, None)
    settings = getattr(request.app.state, SETTINGS_KEY, None)
    startup = getattr(request.app.state, STARTUP_SECONDS_KEY, 0.0)
    limit = settings.api.rate_limit_per_minute if settings else 0

    if service is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            ready=False,
            backend="hybrid",
            documents=0,
            retrievers=[],
            startup_seconds=startup,
            rate_limit_per_minute=limit,
            rate_limit_process_scoped=True,
        )

    health = service.health()
    ready = bool(health["ready"])
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if ready else "not_ready",
        ready=ready,
        backend=health["backend"],
        documents=health["documents"],
        retrievers=list(health["retrievers"]),
        startup_seconds=startup,
        rate_limit_per_minute=limit,
        rate_limit_process_scoped=True,
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Artifacts are not loaded."}},
)
def readiness(request: Request, response: Response) -> HealthResponse:
    """Report whether the process can serve searches.

    Returns 200 when the artifacts loaded at startup are present, and 503 when
    they are not. The HTTP status is the point of this endpoint: a body that
    says ``ready: false`` with a 200 is a body a load balancer will ignore.
    """
    return _health_payload(request, response)


@router.get(
    "/health",
    summary="Combined health and configuration summary",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Artifacts are not loaded."}},
)
def health(request: Request, response: Response) -> HealthResponse:
    """Readiness plus the settings needed to interpret it.

    Reports the rate limit alongside ``rate_limit_process_scoped``, because the
    effective per-client limit is the configured value times the number of
    worker processes and an operator reading only ``rate_limit_per_minute``
    would be misled about what a client can actually send.
    """
    return _health_payload(request, response)


@router.get(
    "/stats",
    summary="Effective configuration and corpus summary",
    response_model=StatsResponse,
    responses={503: {"model": ErrorResponse, "description": "Artifacts are not loaded."}},
)
def stats(service: ServiceDep) -> StatsResponse:
    """The configuration this process is actually running with.

    Read from the assembled service rather than re-read from the config files,
    so it reports what is in effect -- including any value overridden by an
    environment variable at startup.
    """
    info = service.info()
    return StatsResponse(
        backend=info["backend"],
        documents=info["documents"],
        retrievers=list(info["retrievers"]),
        default_top_k=info["default_top_k"],
        max_top_k=info["max_top_k"],
        max_query_chars=info["max_query_chars"],
        query_processing_enabled=info["query_processing_enabled"],
        query_processing=info["query_processing"],
    )


__all__ = ["router"]
