"""Turn a result's provenance into words a reader can check.

The design constraint that shapes this module: **an explanation must be derived
from the ranking that actually happened, and must not be able to change it.**
So this file takes an already-computed :class:`SearchHit` and renders it. It
imports the application model for its types and nothing else from the retrieval
stack; it has no access to a query, an index, or a retriever, and therefore no
mechanism to re-rank, boost, or filter. If a future change added a "relevance"
term to a rendered explanation, the number would stop being the rank the user
saw.

What it will not say
--------------------
* Not "this document is relevant because ..." — the system has no evidence for
  a relevance claim. It knows a BM25 score and a cosine, and those are not
  evidence about the question.
* Not "matched your query closely" — an RRF contribution is close to
  rank-conversion arithmetic, not a similarity judgement.
* Not a numeric comparison between a BM25 score and a cosine. The two live on
  different, non-interchangeable scales; ``raw_score`` of 40 from BM25 and 0.4
  from the vector retriever are not 100x apart in any meaningful sense, and a
  panel that let a reader conclude that would be actively misleading.

What it will say, and what each claim rests on:

``"BM25 rank 2, RRF contribution 0.0161"``
    Directly readable from the provenance. Verifiable by re-running BM25.
``"Retrieved by both retrievers"``
    The set of retrievers that returned the document. Verifiable from the
    provenance alone.
``"Only the vector retriever found it"``
    The complement of the above -- which is a real and useful fact, because it
    is the whole argument for hybrid retrieval in one line.
``"Fused score 0.0161 = BM25 0.0161 + vector 0.0000"``
    An identity, printed as an identity so a reader can add it up.

Every explanation also carries the ``caveat`` field naming what it does not
establish, because an honest explanation without its own limits reads as a
stronger claim than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semantic_search.app.models import SearchHit, SearchMetadata

#: Display names. Internal retriever keys are lowercase and short; a result
#: panel is read by people, and "vector" alone does not say what it is.
RETRIEVER_LABELS: dict[str, str] = {
    "bm25": "BM25 (lexical)",
    "semantic": "Sentence-BERT (semantic)",
}


def retriever_label(name: str) -> str:
    """Human-readable name for a retriever key, passing unknown keys through.

    Unknown keys pass through rather than raising: a new retriever added to the
    system should be explainable before someone updates this dictionary, and a
    missing label must never turn a successful search into an error.
    """
    return RETRIEVER_LABELS.get(name, name)


@dataclass(slots=True, frozen=True)
class Contribution:
    """One retriever's arithmetic contribution to a fused score.

    Typed rather than a ``dict`` so the sort order and the rendered strings are
    checked against real field types, and so a caller cannot mutate a rendered
    explanation by writing into the dict it was handed.
    """

    retriever: str
    label: str
    rank: int
    raw_score: float
    rrf_contribution: float

    @property
    def display_rank(self) -> int:
        """1-based rank for display.

        ``HitProvenance.rank`` is 0-based, deliberately and consistently, so the
        conversion happens once and here -- the same rule
        ``evaluation.metrics.hits_to_ranked_ids`` follows.
        """
        return self.rank + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": self.retriever,
            "label": self.label,
            "rank": self.rank,
            "display_rank": self.display_rank,
            "raw_score": round(self.raw_score, 6),
            "rrf_contribution": round(self.rrf_contribution, 8),
        }


@dataclass(slots=True, frozen=True)
class Explanation:
    """One result, explained.

    Attributes:
        doc_id / title / rank: Copied from the hit, so an explanation always
            identifies which result it is about.
        sources: Which retrievers returned it. Empty is impossible for a fused
            result, but is rendered as "unknown" rather than raising.
        both_retrievers: Whether both retrievers found it. This single boolean is
            the most informative fact in the whole panel, and it is the one that
            is exactly true rather than approximately informative.
        rrf_score: The fused score, which is what the ranking used.
        contributions: One entry per retriever that returned it, with that
            retriever's rank and its arithmetic contribution.
        lines: The rendered sentences, in the order they should be read.
        caveat: What this explanation does not establish.
    """

    doc_id: str
    title: str
    rank: int
    sources: tuple[str, ...]
    both_retrievers: bool
    rrf_score: float
    contributions: tuple[Contribution, ...]
    lines: tuple[str, ...]
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "rank": self.rank,
            "sources": list(self.sources),
            "both_retrievers": self.both_retrievers,
            "rrf_score": round(self.rrf_score, 8),
            # Each Contribution is serialised through its own to_dict, not
            # handed to json as a dataclass. The API contract is a plain JSON
            # object, and asdict() here would drop display_rank.
            "contributions": [c.to_dict() for c in self.contributions],
            "lines": list(self.lines),
            "caveat": self.caveat,
        }


#: The same caveat for every result. Stated once, on every one, rather than
#: once in a footer that most readers never reach.
CAVEAT = (
    "This explains how the result was ordered. It is not a statement that the "
    "document answers your question -- that is a human judgment, and the "
    "evaluation dataset records the two separately for exactly this reason."
)


def explain_hit(hit: SearchHit) -> Explanation:
    """Render one result's provenance into checkable statements.

    Pure: reads the hit, returns an :class:`Explanation`, changes nothing. Safe
    to call on any result, in any order, any number of times.
    """
    sources = tuple(hit.sources) or ("unknown",)
    both = len(sources) > 1

    contributions = sorted(
        (
            Contribution(
                retriever=entry.retriever,
                label=retriever_label(entry.retriever),
                rank=entry.rank,
                raw_score=entry.raw_score,
                rrf_contribution=entry.rrf_contribution,
            )
            for entry in hit.provenance
        ),
        key=lambda c: c.rrf_contribution,
        reverse=True,
    )

    lines: list[str] = []
    if both:
        lines.append(
            "Retrieved by both " + " and ".join(retriever_label(name) for name in sources) + "."
        )
        lines.append(
            "Agreement between a lexical and a semantic retriever is the strongest "
            "ordering evidence this system produces."
        )
    elif len(sources) == 1:
        lines.append(
            "Found by "
            + retriever_label(sources[0])
            + " only. The other retriever did not return it in its pool, so this "
            "rank rests on one signal rather than two."
        )
    else:
        lines.append("Retrieval source unknown; no provenance was recorded.")

    for contribution in contributions:
        lines.append(
            f"{contribution.label}: rank {contribution.display_rank}, "
            f"RRF contribution {contribution.rrf_contribution:.6f}."
        )

    if contributions:
        total = sum(c.rrf_contribution for c in contributions)
        parts = " + ".join(f"{c.rrf_contribution:.6f}" for c in contributions)
        lines.append(
            f"Fused score {total:.6f} = {parts} " f"(rank {hit.rank}, by reciprocal rank fusion)."
        )

    # The raw scores are deliberately printed WITHOUT the word "similarity" and
    # without any comparison between them. They are shown because a reader
    # debugging the system wants them, and labelled as not-comparable.
    if len(contributions) > 1:
        lines.append(
            "Raw scores ("
            + ", ".join(f"{c.label} {c.raw_score}" for c in contributions)
            + ") are on different scales and are not comparable with each other."
        )

    return Explanation(
        doc_id=hit.doc_id,
        title=hit.title,
        rank=hit.rank,
        sources=sources,
        both_retrievers=both,
        rrf_score=hit.score,
        contributions=tuple(contributions),
        lines=tuple(lines),
        caveat=CAVEAT,
    )


def explain_query_processing(
    metadata: SearchMetadata | None,
    original: str = "",
    processed: str = "",
) -> dict[str, Any]:
    """Explain what the query pipeline did, or did not do.

    The absence is reported as carefully as the presence. A panel that only
    describes rewrites when they occur is indistinguishable, to a reader, from a
    system where the pipeline never runs -- which is the exact ambiguity Phase 8
    was written to remove.

    ``original`` and ``processed`` are passed in rather than read from
    ``metadata`` because they are fields of the *response*, not of the metadata
    block. Reading them off the metadata with a ``hasattr`` guard would quietly
    print an empty string instead of the user's query, which is the kind of
    failure that looks like a design choice in the output.
    """
    if metadata is None:
        return {
            "enabled": None,
            "original": original,
            "processed": processed,
            "was_modified": None,
            "was_corrected": None,
            "corrections": [],
            "expansions": [],
            "lines": ["Query processing state was not recorded for this response."],
        }

    lines: list[str] = []
    if not metadata.query_processing_enabled:
        lines.append(
            "Query processing did not run for this request: the query was passed "
            "to the retrievers unchanged."
        )
    elif not metadata.was_modified:
        lines.append(
            "Query processing ran and left the query unchanged. Normalisation "
            "found nothing to fix."
        )
    else:
        lines.append(f"Query rewritten before retrieval: {processed!r}")

    if metadata.was_corrected and metadata.corrections:
        for correction in metadata.corrections:
            original_term = correction.get("original", "?")
            corrected = correction.get("corrected", "?")
            lines.append(f"Spelling correction applied: {original_term!r} -> {corrected!r}.")
    elif metadata.was_corrected:
        lines.append("The query was marked as corrected but no correction was recorded.")

    if metadata.expansions:
        for expansion in metadata.expansions:
            term = expansion.get("term", expansion.get("original", "?"))
            added = expansion.get("expansions", expansion.get("added", []))
            lines.append(f"Synonym expansion: {term!r} -> {list(added)}.")

    return {
        "enabled": metadata.query_processing_enabled,
        "original": original,
        "processed": processed,
        "was_modified": metadata.was_modified,
        "was_corrected": metadata.was_corrected,
        "corrections": list(metadata.corrections),
        "expansions": list(metadata.expansions),
        "lines": lines,
    }


@dataclass(slots=True, frozen=True)
class ResponseExplanation:
    """Everything a reader needs to audit one search response."""

    query_explanation: dict[str, Any]
    results: tuple[Explanation, ...] = field(default=())
    caveats: tuple[str, ...] = (CAVEAT,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_processing": self.query_explanation,
            "results": [e.to_dict() for e in self.results],
            "caveats": list(self.caveats),
        }


def explain_response(response: Any) -> ResponseExplanation:
    """Explain a whole :class:`ApplicationSearchResponse`.

    Accepts the response object rather than importing its type, so this module
    depends on nothing that could route a request. The only fields read are the
    results and the metadata; nothing here can trigger retrieval.
    """
    results = tuple(explain_hit(hit) for hit in getattr(response, "results", ()))
    return ResponseExplanation(
        query_explanation=explain_query_processing(
            getattr(response, "metadata", None),
            original=getattr(response, "original_query", ""),
            processed=getattr(response, "processed_query", ""),
        ),
        results=results,
    )
