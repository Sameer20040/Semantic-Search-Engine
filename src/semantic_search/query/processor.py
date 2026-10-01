"""The query-understanding pipeline.

    raw query
      -> validate          (empty? too long?)
      -> normalize         (NFKC, whitespace, zero-width)
      -> spell-correct     (three gates, usually declines)
      -> expand            (recorded; applied only if explicitly enabled)
      -> one string for retrieval

Two properties are enforced here rather than left to the callers below.

**The processor does not retrieve.** There is no BM25, no FAISS, no embedding,
no ranking, and no import of anything from ``semantic_search.search`` or
``.index``. That is what makes the layer *above* retrieval a layer above it
rather than a peer of it -- if this module could reach an index, "the same
processed query reaches both retrievers" would stop being a structural property
and start depending on nobody reordering the calls.

**One string leaves, and it is the same string for both retrievers.** The
processor produces a single ``query_for_retrieval``. There is no
per-retriever variant and no way to express one through this API, so BM25 and
semantic cannot be handed differently-rewritten queries by accident.

Validation is injected rather than imported for the same reason: the processor
takes a ``validate`` callable and ``SearchService`` passes in the semantic
service's own validator. The rule -- what counts as an empty or over-long query
-- then has exactly one definition in the codebase, and this module does not
have to import the search package to obtain it.
"""

from __future__ import annotations

from collections.abc import Callable

from semantic_search.core.config import QueryConfig
from semantic_search.core.exceptions import EmptyQueryError, QueryTooLongError
from semantic_search.query import synonyms
from semantic_search.query.models import Correction, Expansion, ProcessedQuery
from semantic_search.query.normalizer import normalize
from semantic_search.query.spelling import SpellCorrector


def _default_validate(query: str) -> str:
    """Fallback used when the caller injects nothing.

    Kept deliberately close to the real validator's two rules -- non-empty after
    collapsing whitespace, and within the length limit -- so that a standalone
    processor behaves the same as one wired into the service. It is not a
    replacement: ``SearchService`` always injects the real one.
    """
    if not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}.")
    cleaned = " ".join(query.split())
    if not cleaned:
        raise EmptyQueryError("Query must contain at least one non-whitespace character.")
    return cleaned


class QueryProcessor:
    """Runs a query through normalize -> correct -> expand.

    The corrector is optional: passing ``None`` skips the spelling stage
    entirely, which is what a caller with no corpus vocabulary (tests, an empty
    index) should do rather than constructing an empty corrector that would
    match nothing.
    """

    def __init__(
        self,
        config: QueryConfig,
        *,
        corrector: SpellCorrector | None = None,
        validate: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        self.corrector = corrector
        self._validate = validate or _default_validate

    def process(self, query: str) -> ProcessedQuery:
        """Return the full record of what processing did to ``query``.

        With ``query.enabled = False`` this returns immediately: every field
        equals ``original``, both correction and expansion collections are
        empty, and ``enabled`` is False. A bypassed processor is therefore
        indistinguishable from a no-op one, which is the correct behaviour --
        the alternative would be reporting a processing mode that did nothing.
        """
        if not self.config.enabled:
            return ProcessedQuery(
                original=query,
                normalized=query,
                corrected=query,
                query_for_retrieval=query,
                enabled=False,
            )

        # Validate first, on the raw text. A query that is empty is empty
        # whether or not it is normalized, and rejecting it here gives the
        # caller the original string in the error rather than a normalized one
        # that looks like a different query.
        self._validate(query)

        normalized = normalize(query)
        # Normalization can empty a string that was not empty -- a query of
        # only zero-width characters collapses to nothing. Validate again
        # rather than letting an empty string reach an encoder.
        self._validate(normalized)

        corrections: tuple[Correction, ...] = ()
        corrected = normalized
        if self.config.spell_check and self.corrector is not None:
            corrected, corrections = self.corrector.correct(normalized)

        expansions: tuple[Expansion, ...] = synonyms.find_expansions(corrected)
        query_for_retrieval = corrected
        if self.config.synonym_expansion and expansions:
            query_for_retrieval = synonyms.apply_expansions(corrected, expansions)
            if query_for_retrieval != corrected:
                # Re-flag only the expansions that actually reached the string.
                expansions = tuple(
                    Expansion(e.from_term, e.to_terms, applied=True) for e in expansions
                )

        # The expansion is the only stage that can make the query longer than
        # it started, and it is the only stage that can therefore push a query
        # past the limit *after* the entry check. Catching it here means the
        # error names the stage that caused it, instead of surfacing as a
        # QueryTooLongError from inside the semantic service with no indication
        # that the user's own query was fine.
        self._enforce_max_length(query_for_retrieval)

        return ProcessedQuery(
            original=query,
            normalized=normalized,
            corrected=corrected,
            query_for_retrieval=query_for_retrieval,
            corrections=corrections,
            expansions=expansions,
        )

    def _enforce_max_length(self, processed: str) -> None:
        """Refuse a processed query longer than the configured limit.

        This is the same limit the semantic service applies to the query it
        receives. Checking it at the boundary means a future expansion is
        caught here, at the layer that introduced it.
        """
        if len(processed) > self.config.max_length:
            raise QueryTooLongError(
                f"Processed query is {len(processed)} characters, over the "
                f"{self.config.max_length} limit. The query itself was within the "
                f"limit; query processing lengthened it."
            )
