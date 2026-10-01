"""Rendering functions for the search page.

Every function here is **presentation only**: it takes a view object and draws
it. None of them fetch anything, decide whether a search should run, or
interpret a score. That split exists so the decisions can be tested without a
browser and so a rendering change cannot quietly change behaviour.

A note on the wording, because it is deliberate
------------------------------------------------
Nothing in this module says a result is *relevant*, *correct*, or *the best
match*. The backend has measured how documents were ordered; it has not
measured whether any of them answers the question. The retrieval-quality
evaluation that would support such a claim is blocked pending independent human
relevance judgments. So the page says "rank 1 of 5" and shows the retrievers'
ranks, which are facts, and stops there.
"""

from __future__ import annotations

import streamlit as st

from app.utils.api import ResultView, SearchView

#: Shown in the sidebar under the connection indicator. Stating the scope of
#: the evidence is part of the product, not a disclaimer bolted onto it.
SCOPE_NOTE = (
    "This page shows how documents were ordered by BM25, a sentence-transformer "
    "vector index, and reciprocal rank fusion. It does not show whether any "
    "document answers your question -- no independent relevance judgments "
    "exist for this corpus yet, so retrieval quality is unmeasured."
)


def render_scope_note() -> None:
    """The standing honesty note. Rendered once, above the results."""
    st.info(SCOPE_NOTE, icon="ℹ️")


def render_connection(client: object) -> dict[str, object] | None:
    """Probe the backend and report what state it is in.

    Returns the flattened health view when the backend is ready, and ``None`` in
    every other case. The caller uses that to decide whether to show the search
    form at all: offering a search box that cannot work is worse than saying the
    backend is not up.

    A transport failure and a 503 are rendered differently, because they have
    different fixes -- "start the server" versus "the server's artifacts did not
    load".
    """
    from app.utils.api import (  # imported here to keep the module import graph flat
        ApiError,
        build_health_view,
    )

    try:
        payload = client.health()
    except ApiError as exc:
        if exc.transport_failure:
            st.error(
                f"Cannot reach the search API at `{exc.error_code}`.\n\n"
                "Start it with `uvicorn semantic_search.api.app:get_application "
                "--factory --port 8000`, then reload this page."
            )
        elif exc.status_code == 503:
            st.error(
                "The API is running but reports **not ready** — its search "
                "artifacts are not loaded. Build them with `python scripts/"
                "build_index.py` and restart the server."
            )
        else:
            st.error(f"The API returned an error: {exc.message}")
        return None

    view = build_health_view(payload)
    corpus = view["documents"] or 0
    st.sidebar.success(f"Backend ready — {corpus} documents indexed")
    with st.sidebar.expander("Backend details"):
        st.json(view)
    return view


def render_controls(default_top_k: int, max_top_k: int) -> dict[str, object]:
    """The sidebar controls. Returns a plain dict of the chosen values.

    ``top_k`` is bounded by what the *server* reports, not by a number typed
    into this file. The server's cap is the real one; hardcoding a matching
    value here would drift the moment the configuration changed.
    """
    top_k = st.sidebar.slider(
        "Results to show",
        min_value=1,
        max_value=max(1, max_top_k),
        # The server's own default, not a number this file invented.
        value=int(min(max(1, default_top_k), max(1, max_top_k))),
        help=(
            "How many ranked results to request. The server may return fewer "
            "if that is all it found."
        ),
    )
    query_processing = st.sidebar.checkbox(
        "Run query processing",
        value=True,
        help=(
            "Normalization, spelling correction, and synonym expansion, as "
            "configured on the server. Off sends the query exactly as typed. "
            "Spell correction ships disabled by default — see docs/"
            "evaluation_protocol.md."
        ),
    )
    return {"top_k": top_k, "query_processing": query_processing}


def render_query_panel() -> tuple[str, bool]:
    """The search form. Returns ``(query, submitted)``."""
    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Search the corpus",
            placeholder="e.g. how does a muscle pump move blood",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search", type="primary")
    return query, submitted


def render_query_processing(view: SearchView) -> None:
    """What the query pipeline did, or that it did nothing.

    Both states are rendered. A panel that only appears when something happened
    makes its own absence ambiguous: the reader cannot tell "nothing was
    changed" from "the feature is not reporting".
    """
    if not view.query_processing_enabled:
        st.caption("Query processing was **off** for this request — the query was sent as typed.")
        return

    with st.expander("Query processing", expanded=view.was_modified):
        if view.original_query != view.processed_query:
            st.write(f"**Typed:** `{view.original_query}`")
            st.write(f"**Searched:** `{view.processed_query}`")
        else:
            st.write(f"**Searched:** `{view.processed_query}` (unchanged)")

        if view.corrections:
            st.write("Corrections applied:")
            st.dataframe(view.corrections, use_container_width=True, hide_index=True)
        if view.expansions:
            st.write("Synonym expansions **recorded** (not applied to the search):")
            st.dataframe(view.expansions, use_container_width=True, hide_index=True)
        if view.query_processing_lines:
            for line in view.query_processing_lines:
                st.text(line)


def render_empty_state(view: SearchView) -> None:
    """The no-results state.

    Distinct from the error state, deliberately. A healthy engine that matched
    nothing and a broken engine that returned nothing look identical in a naive
    UI, and conflating them teaches a user to ignore the failure.
    """
    st.warning(
        f"No documents matched `{view.original_query}`. "
        "The engine is healthy — this query simply matched nothing in the corpus.",
        icon="🔍",
    )
    st.caption(
        "The corpus is 300 Wikipedia articles. Most topics are not in it, and a "
        "query outside the corpus will not find them."
    )


def render_summary(view: SearchView) -> None:
    """One line of context above the results."""
    requested = view.requested_top_k
    suffix = f" (asked for {requested})" if requested and requested != len(view.results) else ""
    st.markdown(
        f"**{len(view.results)} result{'s' if len(view.results) != 1 else ''}** "
        f"from **{view.document_count}** documents in **{view.took_ms:.0f} ms**{suffix}"
    )


def render_provenance(result: ResultView) -> None:
    """The per-retriever breakdown for one result.

    Shows the retriever's own rank, its raw score, and its RRF contribution.
    The two score columns are on different scales and are not comparable with
    each other, which is stated here rather than left for the reader to work
    out from a column header.
    """
    if not result.provenance:
        st.caption("No per-retriever provenance was returned for this result.")
        return

    rows = [
        {
            "retriever": p.retriever,
            "rank (0-based)": p.rank,
            "display rank": p.rank + 1,
            "raw score": p.raw_score,
            "RRF contribution": p.rrf_contribution,
        }
        for p in result.provenance
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Raw scores come from different retrievers and are **not comparable** "
        "with each other. The RRF contribution is the only part that sums, and "
        "it is derived from rank, not from the raw score."
    )


def render_result(result: ResultView) -> None:
    """One result card.

    ``result.rank`` is used as-is because it is **already 1-based** -- the
    application layer converts once at ``_to_hit`` and the API carries that
    value unchanged. The provenance table shows *both* forms for the per
    retriever ranks, because those genuinely are 0-based, and a reader
    comparing "rank 3" in a heading against "rank (0-based) 2" in a table
    should be able to reconcile the two conventions without guessing.
    """
    sources = ", ".join(result.retrievers) if result.retrievers else "unknown"
    agreement = "both retrievers" if result.found_by_both else "one retriever"

    with st.container(border=True):
        st.markdown(
            f"**{result.rank}. {result.title}**  \n"
            f"doc_id `{result.doc_id}` · fused score `{result.score:.5f}` · "
            f"found by {agreement} ({sources})"
        )
        st.write(result.snippet)

        if result.explanation_lines:
            with st.expander("Why this rank", expanded=False):
                for line in result.explanation_lines:
                    st.text(line)
                render_provenance(result)
        else:
            with st.expander("Retrieval provenance", expanded=False):
                render_provenance(result)


def render_results(view: SearchView) -> None:
    """The results list, or the empty state."""
    if view.is_empty:
        render_empty_state(view)
        return
    render_summary(view)
    for result in view.results:
        render_result(result)


def render_explanation_panel(view: SearchView) -> None:
    """The caveats the backend attaches to every explained response.

    These are the server's own words, passed through unmodified. The backend
    enforces and tests the guarantee that an explanation never claims relevance;
    rewriting them in the UI would discard that.
    """
    if not view.caveats:
        return
    with st.expander("What this ranking does and does not mean", expanded=True):
        for caveat in view.caveats:
            st.text(f"• {caveat}")


__all__ = [
    "SCOPE_NOTE",
    "render_connection",
    "render_controls",
    "render_empty_state",
    "render_explanation_panel",
    "render_provenance",
    "render_query_panel",
    "render_query_processing",
    "render_result",
    "render_results",
    "render_summary",
]
