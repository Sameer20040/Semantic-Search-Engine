"""Streamlit entry point for the search UI.

Run it with::

    streamlit run app/Home.py

The backend must be running first::

    uvicorn semantic_search.api.app:get_application --factory --port 8000

This module holds **no retrieval logic**. It reads configuration, decides when a
request should be made, and hands the response to a renderer. Every number the
page shows came from the API; nothing here ranks, scores, tokenises, or
corrects. See ``app/utils/api.py`` for why that split exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes the target script as a module named after the file, so the
# repository root is not importable by default and `import app.utils.api` fails
# on a clean checkout. Adding the root to sys.path is the same fix scripts/
# applies, and pyproject's per-file-ignores already waive E402 for app/*.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.components.render import (
    render_connection,
    render_controls,
    render_explanation_panel,
    render_query_panel,
    render_query_processing,
    render_results,
    render_scope_note,
)
from app.utils.api import (
    ApiClient,
    ApiError,
    build_search_view,
    default_base_url,
    describe_error,
)

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔎",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Configuration — every knob is read from the environment, none is a literal
# --------------------------------------------------------------------------- #


@st.cache_resource
def _client() -> ApiClient:
    """One client for the session.

    Cached because building a ``requests.Session`` per rerun is wasteful and,
    more importantly, because a session pools its connections — which is what
    keeps the round trip to a local API from being dominated by TCP setup.
    """
    return ApiClient(base_url=default_base_url())


def _server_defaults() -> dict[str, int]:
    """Read ``/stats`` for the default and maximum result counts.

    Falls back to conservative values if the backend is unreachable. The
    fallback is used only to build the sidebar, and the sidebar is not shown at
    all in that case, so a wrong fallback cannot produce a request that the
    server would reject.
    """
    try:
        stats = _client().stats()
    except ApiError:
        return {"default_top_k": 10, "max_top_k": 100}
    return {
        "default_top_k": int(stats.get("default_top_k", 10)),
        "max_top_k": int(stats.get("max_top_k", 100)),
    }


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Semantic Search Engine")
st.markdown(
    "Hybrid retrieval over a corpus of Wikipedia articles: BM25 and a "
    "sentence-transformer vector index, fused with reciprocal rank fusion. "
    "Each result shows how it was ordered and which retrievers found it."
)
render_scope_note()

client = _client()

# --------------------------------------------------------------------------- #
# Backend check
# --------------------------------------------------------------------------- #

health = render_connection(client)

if health is None:
    # Nothing below this point can work, so the search form is not drawn at
    # all. A disabled search box next to a "backend unreachable" message is a
    # worse version of the same information.
    st.stop()

defaults = _server_defaults()

with st.sidebar:
    st.markdown("---")
    st.caption(f"API: `{client.base_url}`")
    st.caption("Change it with the `STREAMLIT_API_BASE_URL` environment variable.")

controls = render_controls(defaults["default_top_k"], defaults["max_top_k"])

# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

query, submitted = render_query_panel()

if submitted:
    query = query.strip()

    if not query:
        st.info("Type a query and press Search.")
    else:
        try:
            with st.spinner("Searching…"):
                payload = client.search(
                    query,
                    top_k=controls["top_k"],
                    query_processing=controls["query_processing"],
                    explain=True,
                )
        except ApiError as exc:
            # describe_error is the single place the wording lives, and it is
            # deliberately unable to render this as "no results found".
            described = describe_error(exc)
            st.error(f"**{described['heading']}** — {described['detail']}")
            if described["hint"]:
                st.caption(described["hint"])
        else:
            view = build_search_view(payload, explain=True)
            render_query_processing(view)
            render_results(view)
            render_explanation_panel(view)
