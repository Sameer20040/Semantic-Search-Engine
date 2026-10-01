"""Route modules, one per concern.

``search``  -- the retrieval endpoints.
``system``  -- health, readiness, and effective configuration.

Split by what a caller is trying to do rather than by HTTP verb, because the
verb is not what a reader of the docs is looking for.
"""

from semantic_search.api.routes import search, system

__all__ = ["search", "system"]
