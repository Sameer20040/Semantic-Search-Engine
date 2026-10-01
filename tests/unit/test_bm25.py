"""Unit tests for BM25 lexical retrieval.

Two kinds of check here:

* **Behavioural** -- does the right document rank first for a known query?
* **Numerical** -- does one score match the textbook formula computed by hand?
  Without the second, a subtly wrong BM25 (a missing b, a bad IDF variant) still
  returns plausible-looking rankings, and nothing would catch it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from semantic_search.core.config import BM25Config
from semantic_search.core.exceptions import DataError, SearchIndexError
from semantic_search.data.loader import Document, DocumentStore
from semantic_search.data.tokenizer import Tokenizer
from semantic_search.index.bm25 import BM25Index, BM25Result, BM25Searcher


@pytest.fixture
def searcher(sample_documents: list[Document]) -> BM25Searcher:
    return BM25Searcher.from_documents(sample_documents)


class TestBM25Ranking:
    def test_exact_term_ranks_its_own_document_first(self, searcher: BM25Searcher) -> None:
        results = searcher.search("albedo")
        assert results[0].doc_id == "2"

    def test_multi_term_query(self, searcher: BM25Searcher) -> None:
        results = searcher.search("albedo reflectivity")
        assert results[0].doc_id == "2"

    def test_stemmed_query_matches_inflected_document(self, searcher: BM25Searcher) -> None:
        # The document says "pumps"/"muscular"; the query says "pump"/"muscle".
        # Both must reduce to the same stem for the hit to register.
        results = searcher.search("pumping blood")
        assert "5" in [result.doc_id for result in results]

    def test_title_match_outweighs_body_match(self, searcher: BM25Searcher) -> None:
        # "retrieval" appears in document 4's title; if it appeared only in
        # another document's body, the title match should still win.
        results = searcher.search("retrieval")
        assert results[0].doc_id == "4"

    def test_stopword_only_query_returns_nothing(self, searcher: BM25Searcher) -> None:
        assert searcher.search("the and of") == []

    def test_unknown_term_returns_nothing(self, searcher: BM25Searcher) -> None:
        assert searcher.search("xylophonequux") == []

    def test_empty_query_returns_nothing(self, searcher: BM25Searcher) -> None:
        assert searcher.search("") == []
        assert searcher.search("   ") == []

    def test_respects_top_k(self, searcher: BM25Searcher) -> None:
        assert len(searcher.search("the machine model", top_k=1)) == 1

    def test_ranks_are_contiguous_from_zero(self, searcher: BM25Searcher) -> None:
        results = searcher.search("document text model")
        assert [result.rank for result in results] == list(range(len(results)))

    def test_results_are_deterministic(self, searcher: BM25Searcher) -> None:
        assert [r.doc_id for r in searcher.search("model")] == [
            r.doc_id for r in searcher.search("model")
        ]

    def test_matches_documents_to_correct_rows(self, searcher: BM25Searcher) -> None:
        # The row/doc_id mismatch is the classic silent-corruption bug.
        for result in searcher.search("albedo heart anarchism model retrieval"):
            assert searcher.store[result.row].id == result.doc_id


class TestBM25Numerics:
    def test_matches_hand_computed_score(self) -> None:
        """Verify the formula itself on a two-document corpus.

        Documents: "alpha" and "alpha beta gamma" (no stemming or stopwords, so
        the tokens are exactly as written). Query "alpha".

        N=2, df(alpha)=2, so
            IDF = ln(1 + (2 - 2 + 0.5)/(2 + 0.5)) = ln(1.2)
        doc 0: len=1, f=1; doc 1: len=3, f=1. avgdl=2.
        k1=1.5, b=0.75.
            norm(d) = 1 - 0.75 + 0.75·len/2
            score   = IDF · (1·(1.5+1)) / (1 + 1.5·norm)
        The shorter document must score higher -- that is b doing its job.
        """
        config = BM25Config(k1=1.5, b=0.75, use_stemming=False, stopwords="none")
        tokenizer = Tokenizer(stopwords=None, use_stemming=False, remove_stopwords=False)
        docs = [
            Document(id="0", title="d0", text="alpha"),
            Document(id="1", title="d1", text="alpha beta gamma"),
        ]
        index = BM25Index.build(docs, config, tokenizer, title_boost=0)

        idf = math.log(1.0 + (2 - 2 + 0.5) / (2 + 0.5))
        expected_norm = 1.0 - 0.75 + 0.75 * (1 / 2)
        expected = idf * (1 * (1.5 + 1)) / (1 + 1.5 * expected_norm)

        total, per_term = index.score(0, ["alpha"])
        assert per_term["alpha"] == pytest.approx(expected, rel=1e-9)
        assert total == pytest.approx(expected, rel=1e-9)

        # ...and the longer document scores strictly lower, so the single-token
        # document outranks it.
        ranked = index.search("alpha")
        assert ranked[0].doc_id == "0"
        assert ranked[0].score == pytest.approx(expected, rel=1e-4)
        assert ranked[0].score > ranked[1].score

    def test_b_zero_disables_length_normalisation(self) -> None:
        docs = [
            Document(id="0", title="d0", text="alpha"),
            Document(id="1", title="d1", text="alpha beta gamma"),
        ]
        tokenizer = Tokenizer(stopwords=None, use_stemming=False, remove_stopwords=False)
        index = BM25Index.build(
            docs, BM25Config(b=0.0, use_stemming=False, stopwords="none"), tokenizer, title_boost=0
        )
        assert index.score(0, ["alpha"])[0] == pytest.approx(index.score(1, ["alpha"])[0])

    def test_idf_is_non_negative(self, searcher: BM25Searcher) -> None:
        # A term in every document is the classic case where the Robertson IDF
        # would go negative; the Lucene variant must not.
        for term in ("a", "document", "the", "is"):
            assert searcher.index.idf(term) >= 0.0

    def test_rare_term_scores_higher_than_common_term(self) -> None:
        docs = [
            Document(id="0", title="d0", text="common word here"),
            Document(id="1", title="d1", text="common word here xylophone"),
        ]
        tokenizer = Tokenizer(stopwords=None, use_stemming=False, remove_stopwords=False)
        index = BM25Index.build(docs, BM25Config(stopwords="none", use_stemming=False), tokenizer)
        assert index.idf("xylophone") > index.idf("common")

    def test_term_frequency_saturates(self) -> None:
        # Mentioning a term 100x should not score 100x a single mention.
        docs = [
            Document(id="0", title="d0", text="target"),
            Document(id="1", title="d1", text="target " * 100),
        ]
        tokenizer = Tokenizer(stopwords=None, use_stemming=False, remove_stopwords=False)
        index = BM25Index.build(docs, BM25Config(stopwords="none", use_stemming=False), tokenizer)
        single = index.score(0, ["target"])[0]
        hundred = index.score(1, ["target"])[0]
        assert 0 < hundred < single * 2


class TestExplain:
    def test_explain_reports_per_term_scores(self, searcher: BM25Searcher) -> None:
        explanation = searcher.index.explain("albedo reflectivity", "2")
        assert explanation is not None
        assert "albedo" in explanation["term_scores"]
        assert explanation["total_score"] == pytest.approx(
            sum(explanation["term_scores"].values()), rel=1e-4
        )

    def test_explain_lists_missing_terms(self, searcher: BM25Searcher) -> None:
        explanation = searcher.index.explain("albedo heart", "2")
        assert explanation is not None
        assert "heart" in explanation["missing_terms"]

    def test_explain_unknown_doc_returns_none(self, searcher: BM25Searcher) -> None:
        assert searcher.index.explain("albedo", "does-not-exist") is None

    def test_result_carries_matched_terms(self, searcher: BM25Searcher) -> None:
        result = searcher.search("albedo")[0]
        assert "albedo" in result.matched_terms
        assert result.term_scores is not None


class TestBuildAndPersistence:
    def test_build_over_empty_corpus_raises(self) -> None:
        with pytest.raises(DataError, match="empty corpus"):
            BM25Index.build([])

    def test_roundtrip_preserves_results(self, searcher: BM25Searcher, tmp_path: Path) -> None:
        path = searcher.index.save(tmp_path / "bm25.pkl")
        reloaded = BM25Index.load(path, config=searcher.index.config)
        assert reloaded.search("albedo", top_k=5)[0].doc_id == "2"
        assert [r.doc_id for r in reloaded.search("model")] == [
            r.doc_id for r in searcher.search("model")
        ]

    def test_save_writes_metadata_sidecar(self, searcher: BM25Searcher, tmp_path: Path) -> None:
        import json

        path = searcher.index.save(tmp_path / "bm25.pkl")
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["documents"] == 5
        assert meta["bm25_k1"] == searcher.index.config.k1

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SearchIndexError, match="not found"):
            BM25Index.load(tmp_path / "absent.pkl")

    def test_load_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bm25.pkl"
        path.write_bytes(b"not a pickle at all")
        with pytest.raises(SearchIndexError, match="Could not read"):
            BM25Index.load(path)

    def test_load_rejects_future_version(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "bm25.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "version": 999,
                    "doc_ids": [],
                    "doc_lengths": [],
                    "term_frequencies": [],
                    "average_length": 0.0,
                },
                handle,
            )
        with pytest.raises(SearchIndexError, match="version 999"):
            BM25Index.load(path)

    def test_title_boost_increases_title_match_score(
        self, sample_documents: list[Document]
    ) -> None:
        plain = BM25Index.build(sample_documents, title_boost=0)
        boosted = BM25Index.build(sample_documents, title_boost=3)
        # Document 4 has "Retrieval" in its title, so boosting titles must not
        # reduce its score.
        assert boosted.score(3, ["retrieval"])[0] >= plain.score(3, ["retrieval"])[0]


class TestSearcher:
    def test_rejects_mismatched_store(self, sample_documents: list[Document]) -> None:
        index = BM25Index.build(sample_documents)
        with pytest.raises(DataError, match="Rebuild the index"):
            BM25Searcher(index, DocumentStore(sample_documents[:2]))

    def test_search_with_documents_returns_pairs(self, searcher: BM25Searcher) -> None:
        pairs = searcher.search_with_documents("albedo", top_k=1)
        assert pairs[0][1].title == "Albedo"

    def test_stats(self, searcher: BM25Searcher) -> None:
        stats = searcher.stats()
        assert stats["documents"] == 5
        assert stats["vocabulary_size"] > 0

    def test_result_is_frozen(self, searcher: BM25Searcher) -> None:
        result = searcher.search("albedo")[0]
        assert isinstance(result, BM25Result)
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.0  # type: ignore[misc]
