"""HTTP middleware: request identity, logging, rate limiting, and the CORS guard.

Why middleware and not dependencies
-----------------------------------
A dependency runs for a route and does not run for a 404, a method-not-allowed,
or a body that failed to parse. A request-ID or rate-limit that silently does
not apply to those is worse than no request-ID at all, because the logs then
contain a *partial* picture and look complete. These run on every exchange.

The four concerns here, and what each one is protecting against:

Request ID
    Every response carries an ``X-Request-ID``, and every log line written while
    handling it carries the same value. Without one, a log line about a slow
    search cannot be tied to the request that caused it once responses are
    interleaved across workers.

Rate limit
    A fixed-window counter keyed by client address, returning the project's
    :class:`RateLimitError` (429) rather than an ad-hoc response, so the payload
    shape is identical to every other error the API returns. Off by default
    when ``rate_limit_enabled`` is false, and the counter is per-process, which
    is stated in the module docstring of the class below -- a limit that is
    multiplied by the worker count is not a limit an operator can reason about,
    so the number of workers is surfaced in ``/health`` alongside it.

Payload cap
    ``max_payload_bytes``. Rejecting an oversized body at the middleware is
    what keeps a 10 MB query from being parsed into a Python object before
    being rejected. The service already rejects a too-long *query*; this rejects
    a too-large *request*.

CORS
    Configured from ``api.cors_origins``. The guard below refuses a wildcard in
    production, because a wildcard with credentials is the difference between a
    demo and an open door.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from semantic_search.core.config import APIConfig
from semantic_search.core.exceptions import (
    ConfigurationError,
    PayloadTooLargeError,
    RateLimitError,
    SemanticSearchError,
)
from semantic_search.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
#: Accepted from the client only if it looks like a correlation id. An
#: arbitrary client string is echoed back into a response header and into the
#: logs, so it is length-capped and character-capped rather than trusted.
MAX_INBOUND_REQUEST_ID = 64

#: Upper bound on distinct client addresses tracked at once. Past this, buckets
#: that have already aged out of the window are dropped. Chosen to be far above
#: the number of clients a single-process deployment sees, and small enough
#: that the map cannot grow without limit on a public one.
_MAX_TRACKED_CLIENTS = 10_000


def _client_address(request: Request) -> str:
    """The address used as the rate-limit key.

    ``request.client.host`` and not ``X-Forwarded-For``: this server may run
    behind a proxy, but it has no way to know whether one is in front of it, and
    a rate limit keyed on a header the client can set is a rate limit the
    client can disable. If a proxy is added, ``proxy_headers`` in the uvicorn
    config is what makes the real address appear here, and it is an explicit
    deployment choice rather than a default.
    """
    if request.client is None:
        # Possible with some ASGI test transports, where no peer is recorded.
        return "unknown"
    return request.client.host


class FixedWindowRateLimiter:
    """Per-address request counter over a fixed window.

    Deliberately not a token bucket and not a sliding window. A token bucket is
    the better algorithm, and it is also the one whose behaviour depends on
    clock resolution and burst parameters that an operator then has to tune. A
    fixed window is predictable from its two numbers, which is the property that
    matters for a limit an operator reads as "60 requests per minute".

    The limitation is stated rather than hidden: the counter is per process, so
    with ``api.workers > 1`` the effective limit is
    ``rate_limit_per_minute * workers``. The window boundary also permits a
    burst of up to 2x the limit across a boundary. Both are acceptable for the
    single-process deployment this ships as, and both would need a shared store
    to fix -- which is a decision for a real deployment, not a default.
    """

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        """Count this request against ``key``.

        Returns:
            ``(allowed, remaining, retry_after_seconds)``. ``retry_after`` is 0
            when the request is allowed.
        """
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                retry_after = max(1, int(self._window - (now - bucket[0])) + 1)
                return False, 0, retry_after
            bucket.append(now)
            # Bound the map: a long-running server must not accumulate an entry
            # per address that has ever connected, forever. Only buckets whose
            # every hit is already outside the window are evictable -- a bucket
            # with a recent hit is the one thing being rate limited, and
            # dropping it would silently reset that client's budget.
            if len(self._hits) > _MAX_TRACKED_CLIENTS:
                stale = [key for key, bucket in self._hits.items() if bucket[-1] < cutoff]
                for key in stale:
                    del self._hits[key]
            return True, self._limit - len(bucket), 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _error_response(exc: SemanticSearchError, request_id: str) -> JSONResponse:
    """Render a project exception as the standard error body.

    The shape is ``{"error": {...}, "request_id": ...}`` in every case --
    raised, validated, or thrown by middleware -- so a client has exactly one
    error format to parse.
    """
    payload = exc.to_dict()
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload, "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


def _inbound_request_id(request: Request) -> str:
    supplied = request.headers.get(REQUEST_ID_HEADER, "")
    if supplied and len(supplied) <= MAX_INBOUND_REQUEST_ID and supplied.isprintable():
        return supplied
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, log the exchange, and enforce the payload cap.

    Order matters and is encoded in how ``app.py`` adds the middleware: this one
    runs outermost so the request id exists for everything inside it, including
    the rate limiter's 429 and the CORS preflight.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], api_config: APIConfig) -> None:
        super().__init__(app)
        self._config = api_config
        self._limiter = FixedWindowRateLimiter(api_config.rate_limit_per_minute, window_seconds=60)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _inbound_request_id(request)
        request.state.request_id = request_id

        # The cap is checked from the Content-Length header rather than by
        # reading the body: reading a 100 MB body in order to reject it is the
        # thing the cap exists to prevent. A request with no Content-Length
        # (chunked) is not capped by this check; the query-length check in the
        # search service still applies, and this asymmetry is noted rather than
        # papered over.
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self._config.max_payload_bytes:
                    return _error_response(
                        PayloadTooLargeError(
                            f"request body of {declared} bytes exceeds the configured "
                            f"limit of {self._config.max_payload_bytes} bytes"
                        ),
                        request_id,
                    )
            except ValueError:
                # A malformed Content-Length is the client's problem and is
                # reported as such rather than crashing the header parse.
                return _error_response(
                    PayloadTooLargeError(f"content-length header {declared!r} is not a number"),
                    request_id,
                )

        # Bound before the branch, because the X-RateLimit-Remaining header is
        # set after the route runs. Leaving it to be assigned only on the
        # allowed path makes the header code depend on the branch above it.
        remaining = 0
        limited = self._config.rate_limit_enabled and request.url.path not in (
            "/health",
            "/health/live",
        )
        if limited:
            allowed, remaining, retry_after = self._limiter.check(_client_address(request))
            if not allowed:
                limited_response = _error_response(
                    RateLimitError(
                        f"rate limit of {self._config.rate_limit_per_minute} requests per "
                        f"minute exceeded"
                    ),
                    request_id,
                )
                limited_response.headers["Retry-After"] = str(retry_after)
                limited_response.headers["X-RateLimit-Limit"] = str(
                    self._config.rate_limit_per_minute
                )
                return limited_response

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Logged with the request id and re-raised. The exception handler
            # registered on the app turns it into a response; swallowing it here
            # would produce a 200 with an empty body and no trace.
            logger.exception(
                "unhandled error during %s %s [request_id=%s]",
                request.method,
                request.url.path,
                request_id,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        if limited:
            response.headers["X-RateLimit-Limit"] = str(self._config.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        # The service already times retrieval and reports it in the response
        # body. This is the whole-exchange number, which is a different thing:
        # it includes serialisation, and it is what a latency budget is
        # actually written against.
        logger.info(
            "%s %s -> %d in %.1fms [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        if self._config.log_queries and request.method == "POST":
            logger.info("search query logged [request_id=%s]", request_id)
        return response


def check_cors_configuration(api_config: APIConfig, environment: str) -> None:
    """Refuse a CORS configuration that is safe locally and not in production.

    A wildcard origin lets any page on the internet read this API's responses
    from a user's browser. That is the right default for a local demo, and it
    is a vulnerability in a deployment, so the combination is refused at
    startup rather than left for someone to notice.

    An **empty** origin list is explicitly allowed in production: it disables
    CORS, which is the correct configuration when the UI is served from the same
    origin as the API. Only the wildcard is refused.

    Raises:
        ConfigurationError: ``cors_origins`` contains ``"*"`` while
            ``environment`` is ``production``.
    """
    if environment == "production" and "*" in api_config.cors_origins:
        raise ConfigurationError(
            "api.cors_origins contains '*' in production, which lets any website read "
            "this API's responses from a user's browser. Set it to the specific origins "
            "that should be allowed, or to an empty list to disable CORS entirely."
        )


__all__ = [
    "REQUEST_ID_HEADER",
    "FixedWindowRateLimiter",
    "RequestContextMiddleware",
    "check_cors_configuration",
]
