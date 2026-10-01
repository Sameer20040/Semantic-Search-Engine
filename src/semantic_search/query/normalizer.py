"""Query normalisation.

This module exists to make *one* decision explicit -- that query text gets the
same Unicode and whitespace treatment as document text -- and then to delegate
the actual work to the functions that already do it for documents. There is
deliberately no normalisation logic here.

Why reuse rather than reimplement
---------------------------------
``normalize_unicode`` and ``normalize_whitespace`` were written for the
indexing pipeline and tuned against real Wikipedia text: NFKC rather than NFC
because articles are full of compatibility forms, zero-width characters stripped
because pastes carry them, control characters turned into spaces rather than
dropped because removing them would fuse two words together. A second
implementation in the query path would be free to drift from the first, and the
failure would be quiet -- queries tokenized slightly differently from documents,
with no error, just worse results nobody can attribute.

A query is also a *shorter* string than a document, so the same function is
both correct and cheap here.

What normalization does not do
------------------------------
It does not lowercase, stem, or remove stopwords. Those are the tokenizer's job
at retrieval time, and doing them here would mean the explainability panel
showed the user a stem they never typed.

The one thing it does beyond the two document functions is **flatten newlines**.
``normalize_whitespace`` deliberately preserves paragraph breaks -- a document
that loses its paragraph structure has been altered -- but a query is a single
line, and a user who pastes a multi-line question means ``"machine\\nlearning"``
as one query. ``SpellCorrector.correct`` splits on ``" "``, so a surviving
newline would split the query into two unspellable words and the typo pass would
silently skip both. A document-preserving rule and a query-preserving rule
disagree here, and the query is the one that matters on this side.
"""

from __future__ import annotations

from semantic_search.data.preprocessor import normalize_unicode, normalize_whitespace


def normalize(query: str) -> str:
    """Return ``query`` with Unicode forms and whitespace runs cleaned up.

    The result may be empty: a whitespace-only or empty input normalizes to the
    empty string. Deciding whether that is an *error* is the caller's job --
    the query layer needs to be able to normalize before it validates, so that
    ``"   "`` is seen for what it is rather than being rejected on its way in.

    Idempotent: ``normalize(normalize(q)) == normalize(q)``. Both underlying
    functions are, and so is the newline flattening, and the test suite asserts
    it on the composed function so a future change to any of them cannot quietly
    break that property.
    """
    if not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}.")
    return normalize_whitespace(normalize_unicode(query)).replace("\n", " ").strip()
