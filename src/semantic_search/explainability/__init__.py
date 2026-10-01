"""Explainability: render a result's provenance as statements a reader can check.

This package is a **renderer**, not a component of retrieval. It reads the
provenance the hybrid service already produces and turns it into sentences. It
imports the application models for their types and nothing else: no index, no
retriever, no query, no corpus. That restriction is what makes "the explanation
describes the ranking that actually happened" a structural property rather than
a promise -- there is no code path here that could re-rank a result.

Sits above ``app`` in the dependency direction, like ``evaluation``.

The sibling package ``metrics`` in ``evaluation`` is the precedent for a module
whose value is in being testable in isolation: pure functions, no I/O, nothing
to construct. This one is the same idea applied to prose.
"""

from semantic_search.explainability.explanations import (
    CAVEAT,
    RETRIEVER_LABELS,
    Contribution,
    Explanation,
    ResponseExplanation,
    explain_hit,
    explain_query_processing,
    explain_response,
    retriever_label,
)

__all__ = [
    "CAVEAT",
    "RETRIEVER_LABELS",
    "Contribution",
    "Explanation",
    "ResponseExplanation",
    "explain_hit",
    "explain_query_processing",
    "explain_response",
    "retriever_label",
]
