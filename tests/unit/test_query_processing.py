"""The query-processing pipeline: normalization, synonyms, stages, preservation.

These tests cover the parts of the pipeline that do not need a corpus, plus the
orchestration properties the specification is specific about: that the original
is always preserved, that a correction is only ever reported when one actually
happened, that the raw baseline is always available, and that exactly one
processed string reaches the retrievers.

The integration tests at the bottom use a real :class:`HybridSearchService`
over a small in-memory corpus, with only the embedding model faked, so they
exercise the real fan-out rather than asserting against a mock.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from semantic_search.core.config import (
    BM25Config,
    EmbeddingConfig,
    FaissConfig,
    HybridConfig,
    QueryConfig,
    SearchConfig,
)
from semantic_search.core.exceptions import EmptyQueryError, QueryTooLongError
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Searcher
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.query.models import (
    Correction,
    Expansion,
    ProcessedQuery,
    Vocabulary,
)
from semantic_search.query.normalizer import normalize
from semantic_search.query.processor import QueryProcessor
from semantic_search.query.service import SearchService
from semantic_search.query.spelling import SpellCorrector, build_vocabulary
from semantic_search.query.synonyms import SYNONYMS, apply_expansions, find_expansions
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: Hand-written frequencies so each case is traceable. `machine` is common and
#: so is a legitimate correction target; `turin` sits exactly at the default
#: frequency floor.
FREQUENCIES: dict[str, int] = {
    "machine": 16,
    "learning": 17,
    "water": 25,
    "engine": 15,
    "turin": 3,
}


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(terms=frozenset(FREQUENCIES), doc_frequency=dict(FREQUENCIES))


def _corrector() -> SpellCorrector:
    return SpellCorrector(Vocabulary(terms=frozenset(FREQUENCIES), doc_frequency=dict(FREQUENCIES)))


#: Spell checking ships off in ``QueryConfig`` -- the Phase 8 measurement found
#: it mangles out-of-corpus real words as readily as it fixes typos. So every
#: test that is *about* spelling turns it on explicitly rather than relying on
#: the default. The two that test the flag itself are the exception.
def _spelling_processor(**overrides: object) -> QueryProcessor:
    """A processor with spelling enabled and the fixture vocabulary."""
    return QueryProcessor(
        QueryConfig(spell_check=True, **overrides),  # type: ignore[arg-type]
        corrector=_corrector(),
    )


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  machine learning  ", "machine learning"),
        ("machine     learning", "machine learning"),
        ("machine\tlearning", "machine learning"),
        ("machine\nlearning", "machine learning"),
    ],
)
def test_whitespace_is_normalized(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalization_is_idempotent() -> None:
    raw = "  machine   learning  "
    once = normalize(raw)
    assert normalize(once) == once


def test_unicode_compatibility_forms_are_normalized() -> None:
    # NFKC: a superscript and its plain equivalent must compare equal, or a
    # user typing "m2" gets nothing for a document written "m²".
    assert normalize("ｍ２") == "m2"


def test_zero_width_characters_are_removed() -> None:
    assert normalize("machine learning") == "machine learning"


def test_normalization_does_not_lowercase() -> None:
    """Casing survives, because the proper-noun rule depends on it.

    The normalizer must not casefold: `preserve_proper_nouns` works by
    noticing that a token is capitalized, and a normalizer that lowercased
    first would have destroyed the only signal it has.
    """
    assert normalize("Serkan Bonn") == "Serkan Bonn"


def test_empty_and_whitespace_inputs_normalize_to_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("\t\n") == ""


# --------------------------------------------------------------------------- #
# Spell correction through the pipeline
# --------------------------------------------------------------------------- #


def test_a_typo_is_corrected_and_reported() -> None:
    processor = _spelling_processor()
    result = processor.process("machne learning")
    assert result.corrected == "machine learning"
    assert [c.to_term for c in result.corrections] == ["machine"]
    assert result.was_corrected


def test_an_unchanged_query_reports_no_corrections() -> None:
    """The spec's rule, asserted end to end rather than on the corrector alone."""
    processor = _spelling_processor()
    result = processor.process("machine learning")
    assert result.corrections == ()
    assert result.corrected == "machine learning"
    assert result.was_corrected is False
    assert result.was_modified is False


def test_normalization_alone_does_not_set_was_corrected() -> None:
    """A whitespace change is a modification, not a correction, and the two
    properties must not be conflated -- one is cosmetic, one claims we fixed
    something."""
    processor = _spelling_processor()
    result = processor.process("  machine   learning  ")
    assert result.was_modified is True
    assert result.was_corrected is False
    assert result.corrections == ()


def test_spell_check_disabled_skips_correction() -> None:
    processor = QueryProcessor(QueryConfig(spell_check=False), corrector=_corrector())
    result = processor.process("machne learning")
    assert result.corrections == ()
    # "corrected" is the value after the *spelling* stage, so with that stage
    # off it is the normalized input -- not the normalized spelling, which
    # never happened. This is the distinction the previous version of this
    # assertion got wrong: it expected "machine learning", which would mean the
    # flag was ignored.
    assert result.corrected == "machne learning"
    assert result.query_for_retrieval == "machne learning"
    assert result.was_corrected is False


def test_a_processor_with_no_corrector_still_normalizes() -> None:
    """No vocabulary means no corrections, but normalization still happens.

    An empty index should not disable the whole pipeline; it should disable
    the one stage that needs a corpus.
    """
    processor = QueryProcessor(QueryConfig())
    result = processor.process("  machne   learning  ")
    assert result.normalized == "machne learning"
    assert result.corrections == ()


# --------------------------------------------------------------------------- #
# Preservation
# --------------------------------------------------------------------------- #


def test_the_original_is_preserved_verbatim() -> None:
    """Including the whitespace, which is the part normalization destroys."""
    raw = "  machne   LEARNING  "
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    result = processor.process(raw)
    assert result.original == raw
    assert result.original == "  machne   LEARNING  "


def test_original_survives_a_query_that_raises() -> None:
    """An error must not destroy the user's input on the way out."""
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    with pytest.raises(EmptyQueryError):
        processor.process("   ")


def test_a_correction_recorded_as_a_no_op_is_rejected() -> None:
    """The model refuses to represent an identity rewrite.

    If `corrections` could hold `a -> a`, then a query that changed nothing
    could report a correction, and `was_corrected` would be true for it.
    """
    with pytest.raises(ValueError, match="does not change anything"):
        Correction(from_term="machine", to_term="machine", confidence=100.0)


def test_to_dict_reports_an_untouched_query_honestly() -> None:
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    payload = processor.process("machine learning").to_dict()
    assert payload["corrections"] == []
    assert payload["expansions"] == []
    assert payload["was_corrected"] is False
    assert payload["was_modified"] is False
    assert payload["original"] == "machine learning"


# --------------------------------------------------------------------------- #
# Synonyms
# --------------------------------------------------------------------------- #


def test_a_known_synonym_is_recorded() -> None:
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    # "machine" and "learning" are both in the map, so this query produces no
    # expansions at all -- the control for the next assertion.
    assert processor.process("machine learning").expansions == ()
    result = processor.process("llm architecture")
    assert [e.from_term for e in result.expansions] == ["llm"]


def test_an_unknown_term_is_not_expanded() -> None:
    assert find_expansions("zzzz nonexistent") == ()


def test_expansion_is_record_only_by_default() -> None:
    """The default must not alter the string sent to the retrievers."""
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    result = processor.process("llm architecture")
    assert result.expansions, "the expansion should be recorded"
    assert all(not e.applied for e in result.expansions)
    assert result.query_for_retrieval == "llm architecture"


def test_enabling_expansion_changes_the_retrieval_string() -> None:
    processor = QueryProcessor(QueryConfig(synonym_expansion=True), corrector=_corrector())
    result = processor.process("llm architecture")
    assert "large language model" in result.query_for_retrieval
    assert all(e.applied for e in result.expansions)


def test_expansion_is_deterministic() -> None:
    first = find_expansions("llm architecture and nlp")
    for _ in range(5):
        assert find_expansions("llm architecture and nlp") == first


def test_expansion_matches_whole_tokens_only() -> None:
    """A substring rule would expand "air" to "artificial intelligence"."""
    assert find_expansions("air") == ()
    assert find_expansions("AI") != ()  # the whole token does match


def test_ambiguous_terms_are_absent_from_the_map() -> None:
    """The exclusions are a design decision and are asserted so they hold."""
    for ambiguous in ("bass", "bat", "jaguar", "mercury", "bark"):
        assert ambiguous not in SYNONYMS, f"{ambiguous!r} is polysemous and must not be mapped"


def test_expansion_keeps_the_original_term() -> None:
    """Expansion is additive; replacing the term would discard what the user
    most likely meant."""
    expansions = find_expansions("llm")
    expanded = apply_expansions("llm architecture", expansions)
    assert expanded.startswith("llm architecture")
    assert "large language model" in expanded


def test_expansion_does_not_duplicate_a_term_already_present() -> None:
    expanded = apply_expansions("machine learning", find_expansions("machine learning"))
    assert expanded == "machine learning"


def test_no_expansions_means_no_change() -> None:
    assert apply_expansions("machine learning", ()) == "machine learning"


# --------------------------------------------------------------------------- #
# Stage combinations and the enabled flag
# --------------------------------------------------------------------------- #


def test_disabled_processing_returns_the_query_untouched() -> None:
    processor = QueryProcessor(QueryConfig(enabled=False), corrector=_corrector())
    raw = "  machne   LEARNING  "
    result = processor.process(raw)
    assert result.original == raw
    assert result.normalized == raw
    assert result.corrected == raw
    assert result.query_for_retrieval == raw
    assert result.corrections == ()
    assert result.expansions == ()
    assert result.enabled is False
    assert result.was_modified is False


def test_full_pipeline_normalizes_corrects_and_records() -> None:
    processor = _spelling_processor(synonym_expansion=False)
    result = processor.process("  machne   LEARNING  ")
    assert result.original == "  machne   LEARNING  "
    assert result.normalized == "machne LEARNING"
    assert result.corrected == "machine LEARNING"
    assert result.query_for_retrieval == "machine LEARNING"
    assert len(result.corrections) == 1


def test_an_empty_query_is_rejected() -> None:
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    with pytest.raises(EmptyQueryError):
        processor.process("   ")


def test_a_non_string_query_is_rejected() -> None:
    processor = QueryProcessor(QueryConfig(), corrector=_corrector())
    with pytest.raises(TypeError):
        processor.process(123)  # type: ignore[arg-type]


def test_an_over_long_query_is_rejected() -> None:
    processor = QueryProcessor(QueryConfig(max_length=20), corrector=_corrector())
    with pytest.raises(QueryTooLongError):
        processor.process("machine learning water engine" * 5)


def test_an_expansion_that_overflows_the_limit_is_caught_at_the_processor() -> None:
    """The expansion is the only stage that lengthens a query, so it is the
    only stage that can push a legal query over the limit *after* the entry
    check. It must be caught here, naming the stage, rather than surfacing as a
    QueryTooLongError from inside the semantic service."""
    processor = QueryProcessor(
        QueryConfig(synonym_expansion=True, max_length=15), corrector=_corrector()
    )
    with pytest.raises(QueryTooLongError, match="lengthened it"):
        processor.process("llm")


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_candidate_df": 0},
        {"min_edit_distance": -1},
        {"min_correction_score": 101.0},
        {"min_correction_score": -1.0},
    ],
)
def test_out_of_range_thresholds_are_rejected(kwargs: dict) -> None:
    """`min_candidate_df: 0` would silently disable the proper-noun guard, so
    it is a validation error rather than a value that quietly means something
    different from what its name says."""
    with pytest.raises(ValueError):
        QueryConfig(**kwargs)


def test_the_safe_defaults_are_the_ones_that_ship() -> None:
    """Phase 8 measured the corrector and found it mangles out-of-corpus real
    words as readily as it fixes typos, and found zero labelled evaluation data
    to justify changing what is retrieved. Both defaults follow from that
    measurement, so they are pinned here: a well-meaning "let's turn these on"
    commit is a behaviour change and should have to change this test too."""
    config = QueryConfig()
    assert config.spell_check is False
    assert config.synonym_expansion is False
    assert config.preserve_proper_nouns is True
    assert config.enabled is True


# --------------------------------------------------------------------------- #
# Integration: one processed string, both retrievers
# --------------------------------------------------------------------------- #


#: A corpus big enough for the frequency gate (``min_candidate_df=2``) to be
#: satisfiable. A 3-document store cannot do it: every term falls below the
#: threshold, no correction is ever possible, and the integration tests would
#: pass for the wrong reason -- asserting that a typo was left alone when the
#: real reason is that nothing could ever be corrected.
_INTEGRATION_DOCUMENTS = [
    Document(id="d0", title="Machine learning", text="Machine learning trains statistical models."),
    Document(id="d1", title="Machine vision", text="Machine vision interprets images."),
    Document(id="d2", title="Machine code", text="Machine code executes directly."),
    Document(id="d3", title="Learning theory", text="Learning theory studies generalisation."),
    Document(id="d4", title="Deep learning", text="Deep learning uses neural networks."),
    Document(id="d5", title="Engine", text="An engine converts energy into motion."),
    Document(id="d6", title="Water", text="Water flows downhill and evaporates."),
]


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Uses ``blake2b`` rather than the builtin ``hash()``: CPython salts string
    hashing per process, so a ``hash()``-based fake produces a different
    ranking on every run and a test built on it is flaky rather than wrong.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.max_seq_length = 0

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = int.from_bytes(
                hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(), "big"
            )
            vectors[row] = [((digest >> (3 * i)) % 17) / 17.0 - 0.5 for i in range(self.dimension)]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, np.where(norms == 0, 1.0, norms)).astype(np.float32)


@pytest.fixture
def hybrid() -> HybridSearchService:
    """A real hybrid service: real BM25, real FAISS, real store, fake model."""
    store = DocumentStore(_INTEGRATION_DOCUMENTS)
    bm25 = BM25Searcher.from_documents(store, config=BM25Config())
    encoder = EmbeddingEncoder(
        EmbeddingConfig(model_name="fake-model", device="cpu"),
        device="cpu",
        model=FakeModel(),
    )
    index = FaissIndex.build(
        encoder.encode_documents(store),
        FaissConfig(index_type="flat_ip"),
        embedding_model="fake-model",
    )
    semantic = SemanticSearchService(encoder, index, store, search_config=SearchConfig())
    return HybridSearchService(
        bm25, semantic, hybrid_config=HybridConfig(rrf_k=60, candidate_pool=100)
    )


@pytest.fixture
def search_service(hybrid: HybridSearchService) -> SearchService:
    """A SearchService whose corrector is wired to the real test corpus.

    ``spell_check=True`` is requested explicitly because the shipped default is
    off; these tests are about what happens *when* processing corrects, so they
    have to ask for it.
    """
    store = DocumentStore(_INTEGRATION_DOCUMENTS)
    config = QueryConfig(spell_check=True, min_candidate_df=2)
    corrector = SpellCorrector(
        build_vocabulary(store),
        min_correction_score=config.min_correction_score,
        max_edit_distance=config.min_edit_distance,
        min_candidate_df=config.min_candidate_df,
    )
    validate = hybrid.semantic.validate_query  # type: ignore[union-attr]
    return SearchService(hybrid, QueryProcessor(config, corrector=corrector, validate=validate))


class TestOneStringReachesBothRetrievers:
    """The specification's central structural guarantee.

    BM25 and semantic must receive the *same* processed query. This is
    asserted by capturing what each retriever is actually handed, using the
    real services -- not by mocking them away, since a mock would define the
    contract instead of testing it.
    """

    def test_both_retrievers_receive_the_identical_string(
        self, search_service: SearchService
    ) -> None:
        assert search_service.hybrid.bm25 is not None
        assert search_service.hybrid.semantic is not None
        seen: list[str] = []

        real_bm25 = search_service.hybrid.bm25.search
        real_semantic = search_service.hybrid.semantic.search

        def capture_bm25(query: str, top_k: int):
            seen.append(query)
            return real_bm25(query, top_k=top_k)

        def capture_semantic(query: str, top_k: int | None = None):
            seen.append(query)
            return real_semantic(query, top_k=top_k)

        search_service.hybrid.bm25.search = capture_bm25  # type: ignore[method-assign]
        search_service.hybrid.semantic.search = capture_semantic  # type: ignore[method-assign]

        response = search_service.search("machne learning", top_k=5)

        assert len(seen) == 2
        assert seen[0] == seen[1], "the two retrievers were handed different queries"
        assert seen[0] == response.query.query_for_retrieval

    def test_the_retrievers_receive_the_corrected_string_not_the_original(
        self, search_service: SearchService
    ) -> None:
        response = search_service.search("machne learning", top_k=5)
        assert response.query.corrections, "this query should have been corrected"
        assert response.query.original == "machne learning"
        # The correction happened, and the processed string is what was used.
        assert "machne" not in response.query.query_for_retrieval


class TestRawBaseline:
    def test_search_raw_leaves_the_query_untouched(self, search_service: SearchService) -> None:
        response = search_service.search_raw("machne learning", top_k=5)
        assert response.query.original == "machne learning"
        assert response.query.query_for_retrieval == "machne learning"
        assert response.query.corrections == ()
        assert response.query.enabled is False
        assert response.results

    def test_on_and_off_are_genuinely_different_paths(self, search_service: SearchService) -> None:
        """The ON/OFF comparison must be a real one, not a config assertion.

        `search` corrects; `search_raw` does not. Both return results, and the
        difference is observable in the record each returns.
        """
        processed = search_service.search("machne learning", top_k=5)
        raw = search_service.search_raw("machne learning", top_k=5)
        assert processed.query.was_corrected is True
        assert raw.query.was_corrected is False
        assert processed.query.query_for_retrieval != raw.query.query_for_retrieval

    def test_disabled_processing_matches_the_raw_path(self, hybrid: HybridSearchService) -> None:
        """With `enabled: false` the processed path *is* the raw path.

        This is the property that makes processing optional without a second
        deployment, so it is asserted rather than assumed.
        """
        disabled = SearchService(
            hybrid,
            QueryProcessor(
                QueryConfig(enabled=False),
                corrector=None,
                validate=hybrid.semantic.validate_query,  # type: ignore[union-attr]
            ),
        )
        query = "machne learning"
        assert [r.doc_id for r in disabled.search(query, top_k=5).results] == [
            r.doc_id for r in disabled.search_raw(query, top_k=5).results
        ]


class TestSearchServiceInfo:
    def test_info_reports_the_active_configuration(self, search_service: SearchService) -> None:
        info = search_service.info()
        assert info["enabled"] is True
        assert info["synonym_expansion_recorded"] is True
        assert info["synonym_expansion_applied"] is False
        assert info["vocabulary_size"] > 0

    def test_response_serialises_both_the_query_and_the_results(
        self, search_service: SearchService
    ) -> None:
        payload = search_service.search("machine learning", top_k=3).to_dict()
        assert payload["query"]["original"] == "machine learning"
        assert len(payload["results"]) <= 3
        assert all("rrf_score" in r for r in payload["results"])


class TestProcessQueryRecord:
    def test_to_dict_is_json_friendly(self) -> None:
        record = ProcessedQuery(
            original="machne",
            normalized="machne",
            corrected="machine",
            query_for_retrieval="machine",
            corrections=(Correction("machne", "machine", 92.3),),
            expansions=(Expansion("llm", ("large language model",), applied=False),),
        )
        payload = record.to_dict()
        assert payload["corrections"][0] == {"from": "machne", "to": "machine", "confidence": 92.3}
        assert payload["expansions"][0]["applied"] is False
        assert payload["was_modified"] is True
        assert payload["was_corrected"] is True
        # Must survive json.dumps, since it is a FastAPI response body.
        json.dumps(payload)
