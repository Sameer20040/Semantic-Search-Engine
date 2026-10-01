"""Synonym expansion, recorded by default and applied only on request.

The map is small, static, and in-repo. That is a deliberate constraint, not a
starting point: an external thesaurus (WordNet, an embedding-based nearest-
neighbour lookup, an LLM) would each be a larger dependency, a source of
version-dependent behaviour, and -- more importantly -- a way for a query to be
rewritten by something nobody can read. Thirty pairs that a reviewer can check
in a minute beat a resource that silently disagrees with itself across versions.

Why record-only by default
--------------------------
Applying an expansion changes the query the retrievers see, and with no labelled
relevance judgements in this project there is no evidence that any particular
expansion helps. So the default is to compute the expansions, report them, and
send the retrievers the *uncorrected-by-expansion* string. Turning the feature
on is a one-line config change for whoever has a use case and an evaluation to
justify it; turning it on by default would be a claim this project cannot make.

Why these pairs are safe
------------------------
Every entry below is a pair where one term is a near-unambiguous alternative
naming of the other in *technical or encyclopedic* prose -- the register this
corpus is written in. The map deliberately excludes the two categories that make
thesaurus expansion dangerous:

**Polysemy.** ``bass`` is a fish and an instrument, ``bat`` an animal and a club
of wood, ``jaguar`` a car and an animal, ``mercury`` a metal and a planet,
``bark`` a dog's sound and a tree's surface. Expanding any of these would add
the wrong sense to roughly half the queries that contain them. They are absent
from the map and the absence is the point.

**Wrong granularity or register.** ``die`` (singular of dice) and ``cat``
(fishing boat vs animal) and ``kid`` (child vs young goat) are not synonyms in
general usage. A technical corpus is not a general one, and a map tuned for
one does not transfer to the other.

There is no automated check that the map stays free of ambiguous pairs -- that
would require the judgement the map exists to avoid depending on. What there is
instead is this reasoning, written down where the next person adding an entry
will read it.
"""

from __future__ import annotations

from semantic_search.query.models import Expansion

#: term -> the terms it may be expanded to. Keys and values are lowercase
#: surface forms; matching is done on casefolded terms.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # Computing. Each pair is one term or the other in computing prose; the
    # senses do not compete outside a technical register.
    "ml": ("machine learning",),
    "ai": ("artificial intelligence",),
    "nlp": ("natural language processing",),
    "llm": ("large language model",),
    "db": ("database",),
    "api": ("application programming interface",),
    # Science, where a common noun and its formal name are the same thing.
    "solar system": ("sun planetary system",),
    "heart attack": ("myocardial infarction",),
    "high blood pressure": ("hypertension",),
    "sugar": ("glucose",),
    "fever": ("pyrexia",),
    "tumour": ("tumor",),
    "tumour cell": ("tumor cell",),
    "organism": ("living thing",),
    "photosynthesis": ("photo synthesis",),
    "evolution": ("darwinian evolution",),
    "big bang": ("big bang theory",),
    "cell division": ("mitosis",),
    "blood circulation": ("haem circulation",),
    "red blood cell": ("erythrocyte",),
    "white blood cell": ("leukocyte",),
    # Everyday words that a general reader would search with and an
    # encyclopedia is not written with. None of these has a second sense that
    # would make the expansion misleading.
    "favourite": ("favorite",),
    "colour": ("color",),
    "centre": ("center",),
    "theatre": ("theater",),
    "metre": ("meter",),
    "litre": ("liter",),
    "grey": ("gray",),
    "analyse": ("analyze",),
    "organise": ("organize",),
}


def find_expansions(query: str) -> tuple[Expansion, ...]:
    """Return the expansions the map recognises, without applying any of them.

    Matching is whole-token and case-insensitive, so ``"AI"`` expands but
    ``"air"`` does not -- a substring rule would fire on the second and add
    "artificial intelligence" to a query about the atmosphere, which is the
    single most common way synonym expansion goes wrong.

    Returns them in query order, and each term at most once, so the result is
    deterministic and a repeated term does not produce a duplicated report.
    """
    found: list[Expansion] = []
    seen: set[str] = set()
    for token in query.casefold().split():
        cleaned = token.strip(".,;:!?\"'()[]")
        if cleaned in seen or cleaned not in SYNONYMS:
            continue
        seen.add(cleaned)
        found.append(Expansion(from_term=cleaned, to_terms=SYNONYMS[cleaned], applied=False))
    return tuple(found)


def apply_expansions(query: str, expansions: tuple[Expansion, ...]) -> str:
    """Append the target terms to ``query``, preserving the original text.

    Expansion is *additive*. The original term is kept and the synonym appended
    after it, rather than replacing it, because replacement is how a query loses
    precision: a document about the specific concept the user named is the one
    they most likely want, and discarding its term to search for a neighbour is
    a bet that need not be taken.

    Terms already present in the query are not added again.
    """
    if not expansions:
        return query
    present = {token.strip(".,;:!?\"'()[]") for token in query.casefold().split()}
    additions = [
        target for expansion in expansions for target in expansion.to_terms if target not in present
    ]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
