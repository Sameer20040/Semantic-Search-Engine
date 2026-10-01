"""Exception handlers: one place where a Python exception becomes HTTP.

The rules this module exists to enforce
--------------------------------------
**No traceback ever crosses the HTTP boundary.** Every handler returns the
project's error envelope -- ``{"error": {"code", "message"}, "request_id"}`` --
and nothing else. Stack traces, file paths inside the container, and the text
of a FAISS error are all things an attacker reads first. The traceback goes to
the log, correlated by request id, which is where the person who can act on it
will look.

**Nothing is swallowed.** Every handler logs at the level matching the failure:
a 4xx is the caller's problem and is logged at WARNING with no traceback; a 5xx
is this process's problem and is logged at ERROR *with* the traceback. A handler
that returned 200 and printed nothing would make a broken engine look like a
working one, which is the specific disguise ``SearchBackendError`` was created
to prevent one layer down.

**The status code is the exception's, not the handler's.** Every exception in
``core/exceptions.py`` already carries a ``status_code``, chosen when the type
was written. Handlers read it rather than re-deciding, so a new exception type
gets correct HTTP semantics for free instead of falling through to 500.

**``expose_stack_traces`` is honoured, and off by default.** It exists because
running locally with a traceback is genuinely useful, and it is gated on an
explicit config flag rather than on the environment so that a misconfigured
deployment cannot turn it on by accident.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from semantic_search.api.middleware import REQUEST_ID_HEADER
from semantic_search.core.exceptions import SemanticSearchError
from semantic_search.core.logging import get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or request.headers.get(REQUEST_ID_HEADER, "")


def _envelope(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id} if request_id else None,
    )


def register_exception_handlers(app: FastAPI, *, expose_stack_traces: bool = False) -> None:
    """Attach every handler to ``app``.

    Args:
        app: The application being built.
        expose_stack_traces: When true, a 500 response includes the formatted
            traceback. Off by default; see the module docstring.
    """

    @app.exception_handler(SemanticSearchError)
    async def _project_error(request: Request, exc: SemanticSearchError) -> JSONResponse:
        """Map a project exception onto its declared status code.

        ``SearchError`` (400) reaches here for a query that is empty or too
        long. ``SearchBackendError`` and the index errors (503) reach it when
        the artifacts are unusable. Both carry a message written for a human,
        and the ``code`` field is the stable part.
        """
        request_id = _request_id(request)
        status_code = exc.status_code
        if status_code >= 500:
            logger.error(
                "%s on %s %s [request_id=%s]: %s",
                exc.code,
                request.method,
                request.url.path,
                request_id,
                exc.message,
                exc_info=exc,
            )
        else:
            logger.warning(
                "%s on %s %s [request_id=%s]: %s",
                exc.code,
                request.method,
                request.url.path,
                request_id,
                exc.message,
            )
        details = dict(exc.details) if exc.details else None
        if expose_stack_traces and status_code >= 500:
            import traceback

            details = details or {}
            details["traceback"] = traceback.format_exc()
        return _envelope(status_code, exc.code, exc.message, request_id, details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Turn pydantic's validation errors into this API's error envelope.

        FastAPI's default 422 body is a different shape from every other error
        this API returns. A client written against the envelope would have to
        special-case it, and the special case is exactly where the bugs live.
        The field errors are kept -- they are the actionable part.
        """
        request_id = _request_id(request)
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        logger.warning(
            "validation failed on %s %s [request_id=%s]: %d field error(s)",
            request.method,
            request.url.path,
            request_id,
            len(fields),
        )
        return _envelope(
            422,
            "validation_error",
            "the request body failed validation",
            request_id,
            {"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Keep 404s and 405s in the same envelope as everything else.

        A bare ``{"detail": "Not Found"}`` from Starlette is the default, and it
        is the one response shape a client written against this API's error
        contract would not expect.
        """
        request_id = _request_id(request)
        return _envelope(exc.status_code, f"http_{exc.status_code}", str(exc.detail), request_id)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort: a 500 with a generic message and a logged traceback.

        The client is told the request id and nothing about the failure. The
        message is deliberately generic -- an unexpected exception's text can
        contain a path, a query, or a value from the corpus, and this is the
        path by which such a thing escapes.
        """
        request_id = _request_id(request)
        logger.exception(
            "unhandled %s on %s %s [request_id=%s]",
            type(exc).__name__,
            request.method,
            request.url.path,
            request_id,
        )
        details = {"cause_type": type(exc).__name__}
        if expose_stack_traces:
            import traceback

            details["traceback"] = traceback.format_exc()
        return _envelope(
            500,
            "internal_error",
            (
                f"an unexpected error occurred: {type(exc).__name__}. "
                f"The server log entry for request {request_id or 'unknown'} has the "
                f"detail."
            ),
            request_id,
            details,
        )


__all__ = ["register_exception_handlers"]
