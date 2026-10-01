"""Production-grade hybrid semantic search engine.

Public entry points are re-exported here for convenience::

    from semantic_search import get_settings, SearchService

See ``docs/architecture.md`` for the system design.
"""

from semantic_search.core.config import Settings, get_settings, load_settings
from semantic_search.core.exceptions import SemanticSearchError
from semantic_search.core.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "SemanticSearchError",
    "Settings",
    "__version__",
    "configure_logging",
    "get_logger",
    "get_settings",
    "load_settings",
]


def __getattr__(name: str) -> object:
    """Lazily expose heavy symbols without importing torch/faiss at package import.

    ``import semantic_search`` is cheap; ``semantic_search.SearchService``
    triggers the real import chain on first access.
    """
    lazy: dict[str, str] = {
        "SearchService": "semantic_search.query.service",
        "Document": "semantic_search.data.loader",
        "SearchRequest": "semantic_search.app.models",
        "ApplicationSearchResponse": "semantic_search.app.models",
        "SearchRequestBody": "semantic_search.api.models",
        "ExplanationResponse": "semantic_search.api.models",
    }
    if name in lazy:
        import importlib

        module = importlib.import_module(lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
