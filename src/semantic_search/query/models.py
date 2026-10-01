"""The record of what query processing did, and what it must never do to it.

Every field the processor produces is derived; ``original`` is the one field
that is not. It is the exact string the user submitted, kept verbatim -- not
trimmed, not case-folded, not round-tripped through the tokenizer. If
processing fails, raises, or is switched off, that string is still the answer
to "what did the user ask?", and the raw baseline path is still available.

The intermediate stages are stored rather than collapsed, because a single
``"machine learning"`` field cannot distinguish "we did nothing" from "we
normalised whitespace and stopped" -- and those are different things to show a
user who is trying to work out why a typo was not fixed. ``was_modified`` and
``to_dict`` both report what actually happened, and both refuse to dress up an
unchanged query as a processed one.

Correctness notes
-----------------
``corrections`` is empty whenever no term was rewritten. An already-correct
query must not carry a "correction" that is identical to its input: that would
show the user a spelling change that did not happen, which is the one error
this layer cannot afford.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Correction:
    """One term rewritten by the spell corrector.

    Attributes:
        from_term: The term exactly as the user typed it, including its case.
        to_term: The corpus term it was rewritten to. Always different from
            ``from_term`` -- an identical rewrite is not a correction and is
            never recorded as one.
        confidence: rapidfuzz similarity (0-100) between the two. This is *not*
            a probability; it is the score that cleared the configured
            threshold, reported so the decision is inspectable.
    """

    from_term: str
    to_term: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.from_term:
            raise ValueError("A correction must have a non-empty from_term.")
        if not self.to_term:
            raise ValueError("A correction must have a non-empty to_term.")
        if self.from_term == self.to_term:
            # Recording an identity rewrite would make `corrections` non-empty
            # for an unchanged query, and `was_modified` would then lie.
            raise ValueError(
                f"Correction {self.from_term!r} -> {self.to_term!r} does not change anything. "
                "An unchanged term is not a correction."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_term,
            "to": self.to_term,
            "confidence": round(self.confidence, 2),
        }

    def __str__(self) -> str:
        return f"{self.from_term} -> {self.to_term} ({self.confidence:.1f})"


@dataclass(slots=True, frozen=True)
class Expansion:
    """One term that the synonym map recognised.

    Recorded whether or not the expansion was applied to the retrieval query.
    In the default record-only mode this describes what *would* have happened,
    which is what makes the feature inspectable without being able to silently
    change results.
    """

    from_term: str
    to_terms: tuple[str, ...]
    #: True when this expansion actually reached the retrievers. False in the
    #: default record-only mode, and the distinction is the whole point of
    #: recording expansions at all.
    applied: bool = False

    def __post_init__(self) -> None:
        if not self.from_term:
            raise ValueError("An expansion must have a non-empty from_term.")
        if not self.to_terms:
            raise ValueError(f"Expansion {self.from_term!r} has no target terms.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_term,
            "to": list(self.to_terms),
            "applied": self.applied,
        }

    def __str__(self) -> str:
        verb = "expands to" if self.applied else "would expand to (not applied)"
        return f"{self.from_term} {verb} {', '.join(self.to_terms)}"


@dataclass(slots=True, frozen=True)
class ProcessedQuery:
    """The full, auditable result of running a query through the processor.

    Attributes:
        original: The user's query, verbatim. Never derived, never modified.
        normalized: After Unicode NFKC and whitespace collapsing. May equal
            ``original``.
        corrected: After spelling correction. May equal ``normalized``.
        corrected_query: The string the spelling stage would send to
            retrieval -- ``corrected``.
        query_for_retrieval: The one string that actually reaches the
            retrievers. Identical to ``corrected`` unless synonym expansion is
            enabled *and* found something.
        corrections: Terms actually rewritten. Empty when nothing changed.
        expansions: Synonyms recognised. Populated in record-only mode too.
        enabled: Whether processing ran at all. When False every intermediate
            field equals ``original`` and both collections are empty, so a
            bypassed processor is indistinguishable from a no-op one -- which
            is the correct behaviour, and is asserted by the tests.
    """

    original: str
    normalized: str
    corrected: str
    query_for_retrieval: str
    corrections: tuple[Correction, ...] = ()
    expansions: tuple[Expansion, ...] = ()
    enabled: bool = True

    @property
    def was_modified(self) -> bool:
        """Did processing change the query at all?

        This compares against ``original`` rather than counting corrections,
        so normalisation-only changes are reported honestly too. A caller that
        shows "we improved your query" is asserting more than this property
        supports if they read it as "we fixed a typo"; the wording has to match
        the property.
        """
        return self.query_for_retrieval != self.original

    @property
    def was_corrected(self) -> bool:
        """Were any terms actually rewritten? Excludes whitespace-only changes."""
        return bool(self.corrections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "corrected": self.corrected,
            "query_for_retrieval": self.query_for_retrieval,
            "was_modified": self.was_modified,
            "was_corrected": self.was_corrected,
            "enabled": self.enabled,
            "corrections": [c.to_dict() for c in self.corrections],
            "expansions": [e.to_dict() for e in self.expansions],
        }


@dataclass(slots=True, frozen=True)
class Vocabulary:
    """The corpus vocabulary a spell corrector is allowed to correct towards.

    Built once from the document store and then immutable. Doc frequency -- not
    raw term count -- is what it stores, because a word repeated fifty times in
    one article is not fifty pieces of evidence that it is a common word.

    Attributes:
        terms: Casefolded surface forms, e.g. ``"machine"`` not ``"machin"``.
            These come from ``Tokenizer.surface_terms``, which does not stem:
            spelling correction needs the real word, and comparing a stemmed
            vocabulary against unstemmed input would reject every correct term.
        doc_frequency: term -> number of documents containing it.
    """

    terms: frozenset[str] = field(default_factory=frozenset)
    doc_frequency: dict[str, int] = field(default_factory=dict)
    #: Terms the tokenizer removes as stopwords, present in almost every English
    #: query and therefore in almost no document. They are held separately
    #: because they are the one category that the *corpus* cannot protect: a
    #: word like ``"and"`` is absent from ``terms`` not because the corpus lacks
    #: it but because ``surface_terms`` strips it, so gate 0 cannot vouch for it
    #: and the fuzzy search is free to replace it with a rare article title.
    #: See the "Stopwords" section of ``spelling.py``.
    stopwords: frozenset[str] = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, term: object) -> bool:
        return term in self.terms

    def frequency(self, term: str) -> int:
        """Documents containing ``term``; 0 when it is not in the vocabulary."""
        return self.doc_frequency.get(term, 0)
