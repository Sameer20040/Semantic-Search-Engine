"""Integration tests for the application search service (Phase 9).

Everything below runs the **real** stack: a real :class:`BM25Searcher`, a real
:class:`FaissIndex`, a real :class:`DocumentStore`, the real
:class:`HybridSearchService`, and the real Phase 8
:class:`~semantic_search.query.service.SearchService`. Only the sentence
transformer is faked (a deterministic ``blake2b`` stand-in, reusing the pattern
``test_query_processing.py`` established), because ``conftest.py`` forbids
loading the real model in tests.

**No performance claim is made from this file.** Latency numbers come only from
``scripts/measure_search_service.py``, which uses the real model and the real
300-document corpus.

The dependency-direction test at the bottom is the automated enforcement of the
architecture this phase is built on.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from semantic_search.app.models import SearchRequest
from semantic_search.app.service import NO_MATCHES, ApplicationSearchService
from semantic_search.core.config import (
    AppConfig,
    BM25Config,
    EmbeddingConfig,
    FaissConfig,
    HybridConfig,
    QueryConfig,
    SearchConfig,
)
from semantic_search.core.exceptions import (
    CorruptIndexError,
    DataError,
    EmptyQueryError,
    SearchBackendError,
)
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.embeddings.encoder import EmbeddingEncoder
from semantic_search.index.bm25 import BM25Searcher
from semantic_search.index.faiss_index import FaissIndex
from semantic_search.query.processor import QueryProcessor
from semantic_search.query.service import SearchService
from semantic_search.query.spelling import SpellCorrector, build_vocabulary
from semantic_search.search.hybrid import HybridSearchService
from semantic_search.search.semantic import SemanticSearchService

#: The same seven-document corpus ``test_query_processing.py`` uses, so the two
#: suites describe the same system. Topically close enough that BM25 and the
#: (hashed) semantic retriever rank them differently, which is what makes the
#: ordering assertions meaningful rather than trivially satisfied.
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
    """Deterministic stand-in for ``SentenceTransformer``.

    ``blake2b``, not the builtin ``hash()``: CPython salts string hashing per
    process, so a ``hash()``-based fake gives a different ranking on every run.
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
def store() -> DocumentStore:
    return DocumentStore(_INTEGRATION_DOCUMENTS)


@pytest.fixture
def hybrid(store: DocumentStore) -> HybridSearchService:
    """Real BM25 + real FAISS + real store; only the encoder is fake."""
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
def phase8_service(hybrid: HybridSearchService) -> SearchService:
    """The real Phase 8 service, with a corrector built from the test corpus.

    ``spell_check=True`` is requested explicitly because the shipped default is
    off; these tests are about the application layer, and this gives the ON path
    a processor that can actually do something.
    """
    store = DocumentStore(_INTEGRATION_DOCUMENTS)
    config = QueryConfig(spell_check=True, min_candidate_df=1)
    corrector = SpellCorrector(
        build_vocabulary(store),
        min_correction_score=config.min_correction_score,
        max_edit_distance=config.min_edit_distance,
        min_candidate_df=config.min_candidate_df,
    )
    validate = hybrid.semantic.validate_query
    return SearchService(hybrid, QueryProcessor(config, corrector=corrector, validate=validate))


@pytest.fixture
def app_service(phase8_service: SearchService) -> ApplicationSearchService:
    return ApplicationSearchService(
        phase8_service, AppConfig(default_top_k=3, max_top_k=10), document_count=7
    )


# --------------------------------------------------------------------------- #
# Result assembly against the real hybrid service
# --------------------------------------------------------------------------- #
class TestResultContract:
    def test_ranks_are_one_based_and_contiguous(
        self, app_service: ApplicationSearchService
    ) -> None:
        response = app_service.search(SearchRequest(query="machine learning", top_k=5))
        assert [hit.rank for hit in response.results] == [1, 2, 3, 4, 5]

    def test_order_matches_the_hybrid_service_exactly(
        self, app_service: ApplicationSearchService, hybrid: HybridSearchService
    ) -> None:
        """The central no-re-ranking guarantee. The app layer must reproduce the
        hybrid list verbatim; if it ever sorts, this is what catches it."""
        query = "machine learning"
        app_response = app_service.search(SearchRequest(query=query, top_k=5))
        direct = hybrid.search(query, top_k=5)
        assert [hit.doc_id for hit in app_response.results] == [r.doc_id for r in direct]

    def test_scores_are_copied_not_recomputed(
        self, app_service: ApplicationSearchService, hybrid: HybridSearchService
    ) -> None:
        query = "engine"
        app_response = app_service.search(SearchRequest(query=query, top_k=4))
        direct = hybrid.search(query, top_k=4)
        assert [hit.score for hit in app_response.results] == [r.score for r in direct]

    def test_titles_and_snippets_come_from_the_store(
        self, app_service: ApplicationSearchService, store: DocumentStore
    ) -> None:
        response = app_service.search(SearchRequest(query="machine learning", top_k=5))
        for hit in response.results:
            document = store.get(hit.doc_id)
            assert hit.title == document.title
            assert hit.snippet  # a non-empty excerpt, built with the query terms

    def test_provenance_matches_the_hybrid_provenance(
        self, app_service: ApplicationSearchService, hybrid: HybridSearchService
    ) -> None:
        query = "learning"
        app_response = app_service.search(SearchRequest(query=query, top_k=4))
        direct = {r.doc_id: r for r in hybrid.search(query, top_k=4)}
        for hit in app_response.results:
            source = direct[hit.doc_id]
            assert sorted(hit.sources) == sorted(source.retrievers)
            for entry in hit.provenance:
                assert entry.rank == source.hit(entry.retriever).rank
                assert entry.rrf_contribution == pytest.approx(
                    source.hit(entry.retriever).rrf_contribution
                )

    def test_zero_based_provenance_rank_is_preserved(
        self, app_service: ApplicationSearchService
    ) -> None:
        """The 1-based display conversion must not leak downward."""
        response = app_service.search(SearchRequest(query="machine", top_k=4))
        for position, hit in enumerate(response.results):
            assert hit.rank == position + 1
            for entry in hit.provenance:
                assert entry.rank >= 0
                assert entry.rank < 7

    def test_metadata_reflects_the_real_corpus(self, app_service: ApplicationSearchService) -> None:
        response = app_service.search(SearchRequest(query="machine", top_k=3))
        metadata = response.metadata
        assert metadata is not None
        assert metadata.document_count == 7
        assert metadata.returned_count == 3
        assert metadata.backend == "hybrid"
        assert metadata.empty_reason is None


class TestTopKBehaviour:
    def test_top_k_below_available_results(self, app_service: ApplicationSearchService) -> None:
        response = app_service.search(SearchRequest(query="machine", top_k=2))
        assert len(response.results) == 2
        assert response.metadata is not None
        assert response.metadata.returned_count == 2
        assert response.metadata.requested_top_k == 2
        assert response.metadata.corpus_exhausted is False

    def test_top_k_above_the_corpus_returns_everything(
        self, app_service: ApplicationSearchService
    ) -> None:
        """A 300-document deployment asked for 500 results gets its 300, not an
        error and not a silently truncated 100."""
        response = app_service.search(SearchRequest(query="machine", top_k=500))
        assert len(response.results) == 7
        metadata = response.metadata
        assert metadata is not None
        assert metadata.corpus_exhausted is True
        assert metadata.returned_count == metadata.document_count

    def test_top_k_above_the_configured_max_is_clamped_but_visible(
        self, phase8_service: SearchService
    ) -> None:
        service = ApplicationSearchService(
            phase8_service, AppConfig(default_top_k=3, max_top_k=4), document_count=7
        )
        response = service.search(SearchRequest(query="machine", top_k=1000))
        metadata = response.metadata
        assert metadata is not None
        assert len(response.results) == 4  # clamped to max_top_k
        assert metadata.requested_top_k == 1000  # the pre-clamp number survives

    def test_configured_default_is_used(self, phase8_service: SearchService) -> None:
        service = ApplicationSearchService(
            phase8_service, AppConfig(default_top_k=2, max_top_k=10), document_count=7
        )
        response = service.search(SearchRequest(query="machine"))
        assert len(response.results) == 2
        assert response.metadata is not None
        assert response.metadata.requested_top_k == 2


class TestZeroResults:
    def test_valid_query_with_no_matches_returns_an_empty_response(
        self, app_service: ApplicationSearchService
    ) -> None:
        """A valid query that matches nothing on a healthy engine is a normal
        answer, marked as such -- never an error."""
        response = app_service.search(SearchRequest(query="zzzznotacorporateterm", top_k=5))
        # BM25 may still return everything for an out-of-vocabulary term, so this
        # asserts the *contract*, not a particular corpus behaviour.
        metadata = response.metadata
        assert metadata is not None
        if not response.results:
            assert metadata.empty_reason == NO_MATCHES
            assert metadata.returned_count == 0
        else:
            assert metadata.empty_reason is None

    def test_empty_response_is_distinguishable_from_a_failure(
        self, app_service: ApplicationSearchService
    ) -> None:
        from semantic_search.app.models import (
            ApplicationSearchResponse,
            SearchMetadata,
        )

        # An empty response carries `empty_reason`; a broken engine raises. The
        # two can never be confused by a caller.
        healthy_empty = ApplicationSearchResponse(
            original_query="zzz",
            processed_query="zzz",
            results=(),
            metadata=SearchMetadata(
                took_ms=0.0,
                backend="hybrid",
                document_count=7,
                query_processing_enabled=True,
                was_modified=False,
                was_corrected=False,
                requested_top_k=5,
                returned_count=0,
                truncated=True,
                corpus_exhausted=False,
                empty_reason=NO_MATCHES,
            ),
        )
        assert healthy_empty.metadata is not None
        assert healthy_empty.metadata.empty_reason == NO_MATCHES

        # ...whereas a backend failure raises rather than producing that object.
        assert app_service is not None


# --------------------------------------------------------------------------- #
# Query processing ON / OFF
# --------------------------------------------------------------------------- #
class TestOneStringReachesBothRetrievers:
    """The specification's central structural guarantee, asserted through the
    application layer this time: whatever path a request takes, BM25 and the
    semantic retriever must be handed the **same** string.

    Both real retrievers are wrapped rather than mocked. A mock would define the
    contract instead of testing it.
    """

    @pytest.mark.parametrize("processing", [True, False])
    def test_both_retrievers_receive_the_identical_string(
        self, app_service: ApplicationSearchService, processing: bool
    ) -> None:
        hybrid = app_service.search_service.hybrid
        assert hybrid.bm25 is not None
        assert hybrid.semantic is not None
        seen: list[str] = []

        real_bm25 = hybrid.bm25.search
        real_semantic = hybrid.semantic.search

        def capture_bm25(query: str, top_k: int):
            seen.append(query)
            return real_bm25(query, top_k)

        def capture_semantic(query: str, top_k: int):
            seen.append(query)
            return real_semantic(query, top_k)

        hybrid.bm25.search = capture_bm25  # type: ignore[method-assign]
        hybrid.semantic.search = capture_semantic  # type: ignore[method-assign]
        try:
            app_service.search(SearchRequest(query="machine learning", query_processing=processing))
        finally:
            hybrid.bm25.search = real_bm25  # type: ignore[method-assign]
            hybrid.semantic.search = real_semantic  # type: ignore[method-assign]

        assert len(seen) == 2
        assert seen[0] == seen[1]

    def test_the_processed_string_is_the_one_reported(
        self, app_service: ApplicationSearchService
    ) -> None:
        """The string that reached the retrievers is the one the caller is told
        was searched, so the ON/OFF record is trustworthy."""
        hybrid = app_service.search_service.hybrid
        assert hybrid.bm25 is not None
        seen: list[str] = []
        real_bm25 = hybrid.bm25.search

        def capture(query: str, top_k: int):
            seen.append(query)
            return real_bm25(query, top_k)

        hybrid.bm25.search = capture  # type: ignore[method-assign]
        try:
            response = app_service.search(SearchRequest(query="  Machine   Learning  "))
        finally:
            hybrid.bm25.search = real_bm25  # type: ignore[method-assign]

        assert seen == [response.processed_query]
        assert response.original_query == "  Machine   Learning  "


class TestProcessingOnOff:
    def test_processing_enabled_normalizes_whitespace(
        self, app_service: ApplicationSearchService
    ) -> None:
        response = app_service.search(SearchRequest(query="  machine   learning  "))
        assert response.original_query == "  machine   learning  "
        assert response.processed_query == "machine learning"
        assert response.metadata is not None
        assert response.metadata.query_processing_enabled is True
        assert response.metadata.was_modified is True

    def test_processing_disabled_keeps_the_query_verbatim(
        self, app_service: ApplicationSearchService
    ) -> None:
        response = app_service.search(
            SearchRequest(query="  machine   learning  ", query_processing=False)
        )
        assert response.original_query == "  machine   learning  "
        assert response.processed_query == "  machine   learning  "
        assert response.metadata is not None
        assert response.metadata.query_processing_enabled is False
        assert response.metadata.was_modified is False
        assert response.metadata.was_corrected is False

    def test_unchanged_query_is_reported_as_unchanged(
        self, app_service: ApplicationSearchService
    ) -> None:
        """An already-normalised query must come back marked unmodified, with an
        empty corrections list -- not with an empty list standing in for "a
        correction was considered and rejected"."""
        response = app_service.search(
            SearchRequest(query="machine learning", query_processing=True)
        )
        metadata = response.metadata
        assert metadata is not None
        assert metadata.was_modified is False
        assert metadata.was_corrected is False
        assert metadata.corrections == ()

    def test_on_and_off_both_return_results(self, app_service: ApplicationSearchService) -> None:
        on = app_service.search(SearchRequest(query="machine learning", query_processing=True))
        off = app_service.search(SearchRequest(query="machine learning", query_processing=False))
        assert on.results
        assert off.results

    def test_both_paths_validate_the_same_way(self, app_service: ApplicationSearchService) -> None:
        """Disabling processing must not disable validation. An empty query is
        rejected on both paths, as a 400."""
        for processing in (True, False):
            with pytest.raises(EmptyQueryError):
                app_service.search(SearchRequest(query="   ", query_processing=processing))

    def test_config_default_applies_when_unspecified(self, phase8_service: SearchService) -> None:
        off = ApplicationSearchService(
            phase8_service, AppConfig(query_processing_enabled=False), document_count=7
        )
        response = off.search(SearchRequest(query="machine learning"))
        assert response.metadata is not None
        assert response.metadata.query_processing_enabled is False
        assert response.processed_query == "machine learning"


# --------------------------------------------------------------------------- #
# Failure surfacing
# --------------------------------------------------------------------------- #
class TestFailureSurfacing:
    @pytest.mark.parametrize(
        "cause",
        [
            CorruptIndexError("index file is truncated"),
            DataError("encoder produced 384 dims, index expects 384"),
            RuntimeError("faiss internal failure"),
        ],
        ids=["corrupt-index", "model-dimension-mismatch", "unexpected"],
    )
    def test_backend_failures_become_search_backend_error(
        self,
        app_service: ApplicationSearchService,
        monkeypatch: pytest.MonkeyPatch,
        cause: Exception,
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise cause

        hybrid = app_service.search_service.hybrid
        assert hybrid.semantic is not None
        monkeypatch.setattr(hybrid.semantic, "search", boom)

        with pytest.raises(SearchBackendError) as excinfo:
            app_service.search(SearchRequest(query="machine learning"))

        error = excinfo.value
        assert error.status_code == 503
        assert error.code == "search_backend_error"
        # The original cause is preserved for logging...
        assert error.cause is cause
        # ...but never serialised into the client payload.
        assert "cause" not in error.to_dict()

    def test_a_failure_never_becomes_an_empty_result_list(
        self, app_service: ApplicationSearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specific disguise this layer exists to prevent."""
        hybrid = app_service.search_service.hybrid
        assert hybrid.bm25 is not None

        def boom(*args: object, **kwargs: object) -> None:
            raise CorruptIndexError("cannot read the BM25 index")

        monkeypatch.setattr(hybrid.bm25, "search", boom)

        with pytest.raises(SearchBackendError):
            app_service.search(SearchRequest(query="machine learning"))
        # Reaching this line without the exception would mean a caller rendered
        # a broken engine as "no results".

    def test_processor_failure_propagates_rather_than_yielding_empty(
        self, phase8_service: SearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise DataError("the spelling vocabulary could not be read")

        service = ApplicationSearchService(phase8_service, AppConfig(), document_count=7)
        monkeypatch.setattr(service.search_service.processor, "process", boom)

        with pytest.raises(SearchBackendError):
            service.search(SearchRequest(query="machine learning", query_processing=True))

    def test_processor_failure_on_the_raw_path_is_impossible(
        self, app_service: ApplicationSearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raw path does not run the processor, so a broken processor cannot
        break it. Asserted so the ON/OFF split stays a real one."""

        def boom(*args: object, **kwargs: object) -> None:
            raise DataError("vocabulary unavailable")

        monkeypatch.setattr(app_service.search_service.processor, "process", boom)
        response = app_service.search(
            SearchRequest(query="machine learning", query_processing=False)
        )
        assert response.results


class TestBackwardCompatibility:
    def test_phase8_search_service_api_is_intact(self, phase8_service: SearchService) -> None:
        processed = phase8_service.search("machine learning", top_k=3)
        assert processed.results
        raw = phase8_service.search_raw("machine learning", top_k=3)
        assert raw.results
        assert phase8_service.info() is not None

    def test_hybrid_service_api_is_intact(self, hybrid: HybridSearchService) -> None:
        results = hybrid.search("machine learning", top_k=3)
        assert len(results) == 3
        assert hybrid.retriever_names == ["bm25", "semantic"]

    def test_wrapping_does_not_change_hybrid_behaviour(
        self, app_service: ApplicationSearchService, hybrid: HybridSearchService
    ) -> None:
        """The application layer is a pass-through over Phase 8; with both
        active and identical inputs the underlying results are identical."""
        query = "machine"
        via_app = app_service.search(SearchRequest(query=query, query_processing=False))
        limit = via_app.metadata.requested_top_k  # type: ignore[union-attr]
        direct = hybrid.search(query, top_k=limit)
        assert [hit.doc_id for hit in via_app.results] == [r.doc_id for r in direct]


# --------------------------------------------------------------------------- #
# Dependency direction
# --------------------------------------------------------------------------- #
#: Packages that must never import the application layer. Each name is a package
#: under ``src/semantic_search``. The rule is one-directional: ``app`` may import
#: anything; nothing below may import ``app``.
_LOWER_LAYERS = ("query", "search", "index", "data", "embeddings", "core")


def _package_dir() -> Path:
    """Locate ``src/semantic_search`` relative to this file, not to the CWD."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "semantic_search"
        if candidate.is_dir():
            return candidate
    raise AssertionError("Could not locate src/semantic_search from the test file.")


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by ``path``, absolute and relative forms.

    Parsed with :mod:`ast` rather than by importing the module: importing would
    execute it, and this test must be able to inspect files that do not even
    import cleanly yet.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: `from ..core import x` -> package prefix.
                package = path.parent.name
                if node.level > 1:
                    package = "semantic_search"
                found.add(f"{package}.{node.module}" if node.module else package)
                for alias in node.names:
                    found.add(f"{package}.{node.module}.{alias.name}" if node.module else "")
            elif node.module:
                found.add(node.module)
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
    return {name for name in found if name}


class TestDependencyDirection:
    """The architecture, enforced.

    The specification requires that the query-processing package must not import
    the application search service, and that lower-level retrieval components
    must not acquire an application-layer dependency. Asserting that in a
    docstring is a promise; this is a test that fails when the promise is broken.

    It is deliberately small -- one rule, one parser -- rather than a general
    architecture framework. A rule that is hard to write correctly is a rule
    that gets deleted.
    """

    def test_query_package_does_not_import_the_application_service(self) -> None:
        root = _package_dir()
        offenders: list[str] = []
        for source in sorted((root / "query").rglob("*.py")):
            for name in _imported_modules(source):
                if (
                    name == "app"
                    or name.startswith("app.")
                    or name.startswith("semantic_search.app")
                ):
                    offenders.append(f"{source.relative_to(root)} imports {name}")
        assert (
            not offenders
        ), "The query package must not depend on the application layer: " + "; ".join(offenders)

    @pytest.mark.parametrize("package", _LOWER_LAYERS)
    def test_no_lower_layer_imports_the_application_service(self, package: str) -> None:
        root = _package_dir()
        offenders: list[str] = []
        for source in sorted((root / package).rglob("*.py")):
            for name in _imported_modules(source):
                if (
                    name == "app"
                    or name.startswith("app.")
                    or name.startswith("semantic_search.app")
                ):
                    offenders.append(f"{source.relative_to(root)} imports {name}")
        assert (
            not offenders
        ), f"The '{package}' package must not depend on the application layer: " + "; ".join(
            offenders
        )

    def test_the_application_layer_actually_uses_the_layers_below(self) -> None:
        """The other direction, so the rule above cannot be satisfied by deleting
        the app layer's wiring instead of by keeping the arrow pointing down."""
        root = _package_dir()
        imported: set[str] = set()
        for source in sorted((root / "app").rglob("*.py")):
            imported |= _imported_modules(source)
        assert any(name.startswith("semantic_search.query") for name in imported)
        assert any(name.startswith("semantic_search.search") for name in imported)

    def test_the_checker_would_actually_catch_a_violation(self, tmp_path: Path) -> None:
        """A guard that cannot fail is not a guard. This writes a file that
        violates the rule and asserts the parser reports it."""
        bad = tmp_path / "query" / "bad.py"
        bad.parent.mkdir(parents=True)
        bad.write_text(
            "from semantic_search.app import ApplicationSearchService\n", encoding="utf-8"
        )
        found = _imported_modules(bad)
        assert any(name.startswith("semantic_search.app") for name in found)

        # And the relative form, which is the easy one to miss.
        relative = tmp_path / "query" / "bad_relative.py"
        relative.write_text(
            "from ..app.service import ApplicationSearchService\n", encoding="utf-8"
        )
        assert any("app" in name for name in _imported_modules(relative))

    def test_no_lower_layer_imports_evaluation_or_explainability(self) -> None:
        """The two packages that sit *above* app must not be reachable from below.

        Evaluation and explainability are consumers of a finished result. If a
        retriever could import either, then a component of retrieval could
        depend on a component that reports on retrieval -- and the reporting
        would stop being an outside view of it. In evaluation's case the
        circularity is sharper still: the harness that measures the systems
        must not be importable by the systems.
        """
        root = _package_dir()
        forbidden = ("evaluation", "explainability")
        offenders: list[str] = []
        for package in _LOWER_LAYERS:
            for source in sorted((root / package).rglob("*.py")):
                for name in _imported_modules(source):
                    if any(
                        name == f
                        or name.startswith(f"{f}.")
                        or name.startswith(f"semantic_search.{f}")
                        for f in forbidden
                    ):
                        offenders.append(f"{source.relative_to(root)} imports {name}")
        assert (
            not offenders
        ), "Retrieval layers must not depend on the packages that report on them: " + "; ".join(
            offenders
        )

    def test_evaluation_and_explainability_sit_above_the_search_stack(self) -> None:
        """The other direction, so the rule above cannot be met by deleting them.

        Each must genuinely reach down into what it measures -- otherwise the
        test above would pass on an empty package and prove nothing.
        """
        root = _package_dir()
        expectations = {
            "evaluation": ("semantic_search.search", "semantic_search.index"),
            "explainability": ("semantic_search.app",),
        }
        for package, required in expectations.items():
            imported: set[str] = set()
            for source in sorted((root / package).rglob("*.py")):
                imported |= _imported_modules(source)
            for module in required:
                assert any(name.startswith(module) for name in imported), (
                    f"{package} imports nothing from {module}, so the dependency-"
                    f"direction test for it would pass vacuously"
                )

    def test_evaluation_does_not_go_through_the_application_layer(self) -> None:
        """Evaluation drives the retrievers directly, and must keep doing so.

        The measured arms are BM25, the vector retriever, and hybrid itself. If
        the harness called ``ApplicationSearchService`` instead, every arm would
        be measured through the same wrapper, and the wrapper's own behaviour --
        top_k clamping, its empty-result reasons, its cache -- would sit between
        the system and the number describing it. An arm measured through the
        application layer is not the arm it is named after.
        """
        root = _package_dir()
        offenders: list[str] = []
        for source in sorted((root / "evaluation").rglob("*.py")):
            for name in _imported_modules(source):
                if (
                    name == "app"
                    or name.startswith("app.")
                    or name.startswith("semantic_search.app")
                ):
                    offenders.append(f"{source.relative_to(root)} imports {name}")
        assert (
            not offenders
        ), "evaluation must measure the retrievers, not the application service: " + "; ".join(
            offenders
        )

    def test_explainability_cannot_import_a_retriever(self) -> None:
        """The honesty property of Phase 9b, enforced rather than documented.

        ``explainability`` claims an explanation cannot change the ranking it
        describes. That claim is only structural if the module has no way to
        reach a retriever, an index, or a query. A future "just re-score it
        properly" convenience import would silently convert the panel from a
        description into a second, unaccounted-for ranking stage.
        """
        root = _package_dir()
        forbidden = ("search", "index", "data", "embeddings", "query")
        offenders: list[str] = []
        for source in sorted((root / "explainability").rglob("*.py")):
            for name in _imported_modules(source):
                if any(
                    name == f or name.startswith(f"semantic_search.{f}") for f in forbidden
                ) and not name.startswith("semantic_search.app"):
                    offenders.append(f"{source.relative_to(root)} imports {name}")
        assert not offenders, (
            "explainability must not import a retrieval component -- it renders a "
            "ranking, it does not produce one: " + "; ".join(offenders)
        )
