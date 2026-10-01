"""Spell correction: the conservative cases, and the ones that must be refused.

The corrector is constructed from a hand-written term -> document-frequency map
rather than from a real corpus. That is not a shortcut around the logic: it
*is* the logic, with the corpus replaced by a fixture small enough to reason
about completely. The gates under test -- similarity, edit distance, frequency
-- all read from that map, and ``tests/unit/test_query_processing.py`` covers
the corpus-building path separately.

The cases are split into the two groups that matter, because a corrector is
judged more on the second than the first:

* **must correct** -- a user who typed a typo wants help.
* **must refuse** -- everything else. A confident wrong rewrite is worse than a
  missed correction, because the user gets wrong results and no signal that
  anything was changed. The refusal cases are the point of the module.
"""

from __future__ import annotations

import pytest

from semantic_search.query.models import Vocabulary
from semantic_search.query.spelling import SpellCorrector, build_vocabulary

#: term -> document frequency. Deliberately hand-written so each case can be
#: traced: `machine` is common (so it is a legitimate correction target) and
#: `serkan` is rare (so it is not). The real corpus gives 16 and 1.
FREQUENCIES: dict[str, int] = {
    "machine": 16,
    "learning": 17,
    "engine": 15,
    "receive": 12,
    "evolution": 14,
    "physics": 8,
    "temperature": 20,
    "water": 25,
    "blood": 18,
    "protein": 11,
    "membrane": 6,
    "democracy": 5,
    "hamilton": 6,
    "obama": 4,
    "einstein": 3,
    "turin": 3,
    "virtues": 3,
    "tectonic": 5,
    "serkan": 1,
    "hemoglobinn": 1,
    "arakan": 1,
}


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(terms=frozenset(FREQUENCIES), doc_frequency=dict(FREQUENCIES))


@pytest.fixture
def corrector(vocabulary: Vocabulary) -> SpellCorrector:
    return SpellCorrector(vocabulary)


# --------------------------------------------------------------------------- #
# Must correct
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("machne", "machine"),
        ("learnign", "learning"),
        ("ngine", "engine"),
        ("recieve", "receive"),
        ("evolutoin", "evolution"),
        ("phsyics", "physics"),
        ("temperture", "temperature"),
    ],
)
def test_typos_are_corrected(corrector: SpellCorrector, typo: str, expected: str) -> None:
    corrected, corrections = corrector.correct(typo)
    assert corrected == expected
    assert len(corrections) == 1
    assert corrections[0].from_term == typo
    assert corrections[0].to_term == expected


def test_multiple_typos_in_one_query_are_all_corrected(corrector: SpellCorrector) -> None:
    corrected, corrections = corrector.correct("machne learnign water")
    assert corrected == "machine learning water"
    assert [c.to_term for c in corrections] == ["machine", "learning"]


def test_correction_preserves_surrounding_punctuation(corrector: SpellCorrector) -> None:
    # "learning," must not become "learning" -- the comma changes how the next
    # query is tokenized, so dropping it is a retrieval effect hiding in what
    # looks like a cosmetic bug.
    corrected, corrections = corrector.correct("machne, learnign.")
    assert corrected == "machine, learning."
    assert len(corrections) == 2


# --------------------------------------------------------------------------- #
# Must refuse -- the important half
# --------------------------------------------------------------------------- #


def test_an_already_correct_query_records_no_corrections(corrector: SpellCorrector) -> None:
    """The spec's rule: never claim a correction that did not happen."""
    corrected, corrections = corrector.correct("machine learning water")
    assert corrected == "machine learning water"
    assert corrections == ()


@pytest.mark.parametrize(
    "query",
    [
        "Serkan",  # proper noun
        "Bonn",
        "Obma",  # a real typo, but capitalized, and therefore refused
        "hemoglobinn",  # too far from any vocabulary word
        "quantum chromodynamics",  # out-of-vocabulary technical terms
    ],
)
def test_refusals_record_no_corrections(corrector: SpellCorrector, query: str) -> None:
    corrected, corrections = corrector.correct(query)
    assert corrected == query
    assert corrections == ()


def test_proper_nouns_are_refused_by_capitalisation(corrector: SpellCorrector) -> None:
    """`preserve_proper_nouns` is a capitalisation test, not a name database.

    "Serkan" is protected because it is capitalized, not because we know it is
    a name. The cost is a capitalized typo like "Machne" being missed, which
    the next test pins down explicitly rather than leaving as folklore.
    """
    assert corrector.correct("Serkan")[0] == "Serkan"
    assert corrector.correct("Machne")[0] == "Machne"


def test_turning_off_proper_noun_preservation_allows_the_missed_correction() -> None:
    """The capitalized-typo cost, demonstrated rather than asserted in a comment.

    "Machne" is the case the proper-noun rule refuses by default. With the flag
    off it is corrected -- which is what makes the trade-off real and the flag
    load-bearing, rather than a setting that appears to do nothing.

    The corrected form is lowercase because the candidate comes from the
    vocabulary, which is casefolded. That is a separate, deliberate choice:
    casing is normalized once, at query time, by the same code that normalizes
    documents, rather than reconstructed here.
    """
    vocabulary = Vocabulary(
        terms=frozenset({"machin", "machine"}),
        doc_frequency={"machin": 1, "machine": 16},
    )
    relaxed = SpellCorrector(vocabulary, preserve_proper_nouns=False)
    corrected, corrections = relaxed.correct("Machne")
    assert corrected == "machine"
    assert corrections[0].from_term == "Machne"
    assert corrections[0].to_term == "machine"

    default = SpellCorrector(vocabulary, preserve_proper_nouns=True)
    assert default.correct("Machne")[0] == "Machne"


def test_a_rare_name_is_not_corrected_to_a_common_word(corrector: SpellCorrector) -> None:
    """The frequency gate: `arakan` exists, but only in one document.

    A rare word's nearest neighbour is usually a coincidence. This is the case
    the third gate exists for, and it is asserted directly rather than through
    a similarity number, because the similarity is low anyway -- what makes it
    a gate test is that the word is *in the vocabulary* and still refused.
    """
    corrected, corrections = corrector.correct("Serkan")
    assert corrected == "Serkan"
    assert all(c.to_term != "arakan" for c in corrections)


def test_a_short_typo_is_still_corrected(corrector: SpellCorrector) -> None:
    # There is deliberately no length floor. A floor looked defensible -- short
    # words have ambiguous candidates -- but it earned nothing: 0 of the 1,587
    # three-letter terms in the real corpus would be rewritten without one,
    # because they are all already in the vocabulary and short-circuit at gate
    # 0. A rule that prevents no measured harm is a rule that only makes the
    # corrector miss things, so it was removed.
    assert corrector.correct("watr")[0] == "water"


def test_empty_and_whitespace_queries_are_safe(corrector: SpellCorrector) -> None:
    assert corrector.correct("") == ("", ())
    assert corrector.correct("   ") == ("   ", ())


# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #


def test_a_stopword_is_never_corrected_away() -> None:
    """Gate 0b. Found by running the 300-document experiment, not by reasoning.

    A stopword is absent from the vocabulary because ``surface_terms`` strips
    it, not because the corpus lacks it -- so gate 0 could not vouch for it and
    the fuzzy search was free to replace it. The two real cases:

        "black holes and gravitaional fields"  ->  "and"   rewritten to "andy"
        "how eartquakes are measured"          ->  "are"   rewritten to "acre"

    Both at similarity 85.7, both confidently wrong, both moving documents in and
    out of the top 5. The fixture reproduces the shape: "andy" and "acre" are
    plausible corpus words here, exactly as "Andy" and "Acre" are in the real
    corpus. The stopword set is what makes the difference.
    """
    vocabulary = Vocabulary(
        terms=frozenset(FREQUENCIES) | {"andy", "acre"},
        doc_frequency={**FREQUENCIES, "andy": 4, "acre": 4},
        stopwords=frozenset({"and", "are", "the", "of", "to", "in"}),
    )
    # The typo targets have to be in the vocabulary themselves, or the corrector
    # is only being asked to refuse and the "still corrects" half proves nothing.
    # In the real corpus "earthquakes" and "gravitational" both clear the
    # frequency gate on their own; here they stand in for that.
    with_targets = Vocabulary(
        terms=vocabulary.terms | {"earthquakes", "gravitational"},
        doc_frequency={**vocabulary.doc_frequency, "earthquakes": 6, "gravitational": 6},
        stopwords=vocabulary.stopwords,
    )
    guarded = SpellCorrector(with_targets)

    # The stopword is refused and the genuine typo in the same query is still
    # fixed, so the gate refuses one term without disabling the stage.
    corrected, corrections = guarded.correct("eartquakes are measured")
    assert corrected == "earthquakes are measured"
    assert [c.from_term for c in corrections] == ["eartquakes"]

    # With the gate off, the same two queries reproduce the real failure
    # exactly, with the same 85.7 score. This is what makes the test a gate
    # test: it fails without the gate rather than merely passing with it.
    unguarded = SpellCorrector(with_targets, protect_stopwords=False)
    assert unguarded.correct("eartquakes are measured")[0] == "earthquakes acre measured"
    assert unguarded.correct("black holes and gravitaional fields")[0] == (
        "black holes andy gravitational fields"
    )
    assert [
        round(c.confidence, 1)
        for c in unguarded.correct("black holes and gravitaional fields")[1]
        if c.from_term == "and"
    ] == [85.7]


def test_a_stopword_is_not_a_correction_target() -> None:
    """A stopword could be the *fix* for a typo, and that is refused too.

    `"teh"` would otherwise map to `"tech"` (85.7, distance 1) -- a real word
    that happens to appear in exactly one document, which is the entire wrong
    answer. Making stopwords un-correctable is what prevents it. The cost is
    that a user who types `"teh"` gets no help, and that trade is deliberate.
    """
    vocabulary = Vocabulary(
        terms=frozenset({"tech"}),
        doc_frequency={"tech": 1},
        stopwords=frozenset({"teh", "the"}),
    )
    guarded = SpellCorrector(vocabulary)
    assert guarded.correct("teh") == ("teh", ())
    assert guarded.correct("te") == ("te", ())


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_disabled_corrector_changes_nothing(corrector: SpellCorrector) -> None:
    disabled = SpellCorrector(corrector.vocabulary, enabled=False)
    assert disabled.correct("machne learning") == ("machne learning", ())


def test_corrector_is_deterministic(corrector: SpellCorrector) -> None:
    """Same input, same output, every time.

    The vocabulary is sorted at construction so that rapidfuzz's tie-break
    between equally-scored candidates does not depend on frozenset iteration
    order, which varies per process. A correction that differed between two
    runs of the same query would be a bug that reproduces intermittently.
    """
    first = corrector.correct("machne learnign recieve")
    for _ in range(5):
        assert corrector.correct("machne learnign recieve") == first


def test_a_new_corrector_over_the_same_vocabulary_agrees(corrector: SpellCorrector) -> None:
    fresh = SpellCorrector(corrector.vocabulary)
    assert fresh.correct("machne learnign") == corrector.correct("machne learnign")


def test_raising_the_similarity_threshold_suppresses_corrections(
    corrector: SpellCorrector,
) -> None:
    # "phsyics" -> "physics" scores 85.7, so a threshold of 90 refuses it.
    # This is what the shipped 92.0 default would have done to the whole
    # feature, and is why the default is 80.
    strict = SpellCorrector(corrector.vocabulary, min_correction_score=90.0)
    assert strict.correct("phsyics")[1] == ()
    assert corrector.correct("phsyics")[1] != ()


def test_raising_the_frequency_threshold_suppresses_corrections(
    corrector: SpellCorrector,
) -> None:
    # "physics" has df 8 here; require 50 and no correction is possible.
    strict = SpellCorrector(corrector.vocabulary, min_candidate_df=50)
    assert strict.correct("phsyics")[1] == ()


def test_tightening_the_edit_distance_budget_suppresses_corrections(
    corrector: SpellCorrector,
) -> None:
    # "recieve" -> "receive" is distance 2, so a budget of 1 refuses it.
    tight = SpellCorrector(corrector.vocabulary, max_edit_distance=1)
    assert tight.correct("recieve")[1] == ()
    assert tight.correct("machne")[1] != ()  # distance 1, still allowed


def test_an_empty_vocabulary_corrects_nothing(vocabulary: Vocabulary) -> None:
    # An index with no documents must not invent corrections from an empty
    # vocabulary; the fuzzy search over zero terms returns nothing.
    empty = SpellCorrector(Vocabulary())
    assert empty.correct("machne learning") == ("machne learning", ())


# --------------------------------------------------------------------------- #
# Vocabulary construction
# --------------------------------------------------------------------------- #


def test_vocabulary_uses_unstemmed_surface_forms() -> None:
    """The vocabulary holds real words, not stems.

    `Tokenizer.surface_terms` is unstemmed; `tokenize`/`unique_tokens` are not.
    A stemmed vocabulary would contain "learn" and "machin" and would then
    reject every correctly-spelled query term. This asserts the property the
    builder depends on, against the real tokenizer.
    """
    from semantic_search.data.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(use_stemming=False)
    surface = tokenizer.surface_terms("learning machine")
    assert "learning" in surface
    assert "learn" not in surface


def test_document_frequency_counts_documents_not_occurrences() -> None:
    """One article repeating a word 50 times is one document's worth of evidence."""
    from semantic_search.data.loader import Document, DocumentStore

    store = DocumentStore(
        [
            Document(id="1", title="A", text="protein protein protein protein"),
            Document(id="2", title="B", text="protein and water"),
        ]
    )
    vocabulary = build_vocabulary(store)
    assert vocabulary.frequency("protein") == 2  # not 5
    assert vocabulary.frequency("water") == 1
    assert vocabulary.frequency("nonexistent") == 0


def test_vocabulary_includes_titles_as_well_as_bodies() -> None:
    """BM25 indexes the title too, so a title-only term must be correctable.

    A vocabulary built from `document.text` alone would not contain words that
    appear only in a heading -- which is exactly the kind of term a user types.
    """
    from semantic_search.data.loader import Document, DocumentStore

    store = DocumentStore([Document(id="1", title="Photosynthesis", text="A process.")])
    assert "photosynthesis" in build_vocabulary(store)


def test_vocabulary_reports_its_size_and_membership() -> None:
    from semantic_search.data.loader import Document, DocumentStore

    store = DocumentStore([Document(id="1", title="Water", text="Water flows.")])
    vocab = build_vocabulary(store)
    assert len(vocab) == len(vocab.terms)
    assert "water" in vocab
    assert "nonexistent" not in vocab
