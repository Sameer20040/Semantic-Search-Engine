"""The HTTP API layer.

Sits above ``app`` in the dependency direction, like ``evaluation`` and
``explainability``. It adds transport concerns -- HTTP status codes, request
validation, rate limiting, CORS, an error envelope -- and **no retrieval
behaviour at all**. Every result this API returns was produced by
``ApplicationSearchService``; nothing here re-sorts, filters, or re-scores one.
``POST /search/explain`` returns the identical results to ``POST /search`` and
that identity is asserted in the tests, because an explanation that quietly
changed the answer would be worse than no explanation.
"""

from semantic_search.api.app import create_app, get_application

__all__ = ["create_app", "get_application"]
