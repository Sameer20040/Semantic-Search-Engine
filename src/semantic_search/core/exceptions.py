"""Exception hierarchy for the semantic search engine.

Every error carries a stable ``code`` used in API error payloads and a
``status_code`` that the API layer maps directly onto an HTTP response. The
API never leaks a traceback to clients in production.
"""

from __future__ import annotations

from typing import Any


class SemanticSearchError(Exception):
    """Base class for all engine errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


# --------------------------------------------------------------------------- #
# Configuration / startup
# --------------------------------------------------------------------------- #
class ConfigurationError(SemanticSearchError):
    """Invalid or contradictory configuration."""

    code = "configuration_error"
    status_code = 500


class StartupValidationError(SemanticSearchError):
    """The engine cannot start because an artifact is missing or incompatible."""

    code = "startup_validation_error"
    status_code = 503


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #
class DataError(SemanticSearchError):
    """Base class for ingestion and preprocessing failures."""

    code = "data_error"


class DatasetDownloadError(DataError):
    """The corpus could not be downloaded or read from its cache."""

    code = "dataset_download_error"


class EmptyCorpusError(DataError):
    """The corpus contained no usable documents after preprocessing."""

    code = "empty_corpus"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class SearchIndexError(SemanticSearchError):
    """Base class for index build/load failures.

    Named ``SearchIndexError`` rather than ``IndexError`` so it does not shadow
    the builtin of that name.
    """

    code = "index_error"


class IndexNotFoundError(SearchIndexError):
    """No index artifact exists at the expected path."""

    code = "index_not_found"
    status_code = 503


class IndexCompatibilityError(SearchIndexError):
    """An index artifact exists but was built with incompatible settings."""

    code = "index_incompatible"
    status_code = 503


class CorruptIndexError(SearchIndexError):
    """An index artifact exists but cannot be deserialised."""

    code = "index_corrupt"
    status_code = 503


class EmbeddingCompatibilityError(SearchIndexError):
    """Cached embeddings do not match the configured model/corpus."""

    code = "embedding_incompatible"
    status_code = 503


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
class SearchError(SemanticSearchError):
    """Base class for query-time failures."""

    code = "search_error"
    status_code = 400


class EmptyQueryError(SearchError):
    """The query contained no searchable terms."""

    code = "empty_query"


class QueryTooLongError(SearchError):
    """The query exceeded the configured maximum length."""

    code = "query_too_long"


class InvalidFilterError(SearchError):
    """A metadata filter was malformed or unsupported."""

    code = "invalid_filter"


class DocumentNotFoundError(SearchError):
    """A document id was not present in the store."""

    code = "document_not_found"
    status_code = 404


class SearchBackendError(SearchIndexError):
    """The retrieval stack failed while serving an otherwise valid request.

    Why this type exists
    --------------------
    A search system has two very different ways to return nothing, and a caller
    cannot tell them apart from ``results == []``:

    * the query was valid and nothing matched it, versus
    * the index could not be read, the store and the index disagree, or the
      encoder and the index were built by different models.

    The second case used to reach the application layer as a bare ``DataError``
    (a 500 with ``code="data_error"``) or an index exception, and the natural
    defensive response to a 500 is to render an empty list. That is exactly
    the disguise this type exists to prevent: it is a 503, it carries the cause,
    and the application layer re-raises rather than converting it into results.

    It subclasses :class:`SearchIndexError` rather than :class:`SearchError`
    because every cause it wraps is an infrastructure failure, not a bad
    request -- a client that sent a valid query should retry, not fix its input.
    """

    code = "search_backend_error"
    #: Stated explicitly rather than inherited. ``SearchIndexError`` itself
    #: declares no ``status_code``, so it inherits 500 from
    #: :class:`SemanticSearchError` -- and an infrastructure failure reported as
    #: a 500 is precisely what teaches a caller to swallow it and render an
    #: empty page. Every other concrete subclass below sets its own; this one
    #: had not, which is how it shipped as a 500.
    status_code = 503

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, details=details)
        #: The original exception, kept for logging. Not serialised into the
        #: client payload -- a stack trace must never cross this boundary.
        self.cause = cause


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class EvaluationError(SemanticSearchError):
    """Base class for retrieval-quality evaluation failures."""

    code = "evaluation_error"


class EvaluationDatasetError(EvaluationError):
    """The evaluation dataset is malformed, inconsistent, or fails validation.

    Distinct from :class:`EvaluationDatasetNotFoundError`: this fires when the
    files are there but cannot be trusted, which is a bug in the dataset rather
    than a missing artifact, and the two deserve different messages.
    """

    code = "evaluation_dataset_error"


class EvaluationDatasetNotFoundError(EvaluationError):
    """The frozen evaluation dataset is missing.

    ``status_code`` is 503 rather than 500 because a missing labelled dataset is
    a startup-artifact problem -- the same class as a missing index -- and the
    API layer already maps that class onto "not ready, try again". An evaluation
    dataset is not a user-facing resource, so this will not surface as a 404 to
    a caller who asked for a document; it surfaces as a failed run.
    """

    code = "evaluation_dataset_not_found"
    status_code = 503


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class RateLimitError(SemanticSearchError):
    """The caller exceeded the configured request budget."""

    code = "rate_limited"
    status_code = 429


class PayloadTooLargeError(SemanticSearchError):
    """The request body exceeded the configured maximum size."""

    code = "payload_too_large"
    status_code = 413
