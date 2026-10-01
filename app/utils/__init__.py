"""Frontend helpers: the API client and the pure view-model functions."""

from app.utils.api import (
    ApiClient,
    ApiError,
    build_health_view,
    build_search_view,
    default_base_url,
    describe_error,
)

__all__ = [
    "ApiClient",
    "ApiError",
    "build_health_view",
    "build_search_view",
    "default_base_url",
    "describe_error",
]
