"""Tests for the explainability renderer.

These test the *wording* as much as the data, because the failure mode of an
explainability layer is not a wrong number -- the numbers come straight from
provenance and are hard to get wrong -- it is a sentence that overstates what
the system knows. So there are tests here that assert the absence of specific
claims ("relevant because", "similarity") as well as the presence of the
specific, checkable ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.app.models import (
    ApplicationSearchResponse,
    HitProvenance,
    SearchHit,
    SearchMetadata,
)
from semantic_search.explainability import (
    CAVEAT,
    Contribution,
    explain_hit,
    explain_query_processing,
    explain_response,
    retriever_label,
)

BM25, SEMANTIC = "bm25", "semantic"


def _hit(sources=("bm25", "semantic"), rank=1, provenance=None) -> SearchHit:
    if provenance is None:
        provenance = (
            HitProvenance(retriever=BM25, rank=0, raw_score=12.5, rrf_contribution=1 / 61),
            HitProvenance(retriever=SEMANTIC, rank=2, raw_score=0.83, rrf_contribution=1 / 63),
        )
    return SearchHit(
        doc_id="3437",
        title="Piano Trio No. 1",
        snippet="Beethoven published three piano trios...",
        rank=rank,
        score=sum(p.rrf_contribution for p in provenance),
        sources=tuple(sources),
        provenance=provenance,
    )


def _metadata(**overrides) -> SearchMetadata:
    defaults: dict[str, object] = {
        "took_ms": 12.5,
        "backend": "hybrid",
        "document_count": 300,
        "query_processing_enabled": True,
        "was_modified": False,
        "was_corrected": False,
        "requested_top_k": 10,
        "returned_count": 1,
        "truncated": False,
        "corpus_exhausted": False,
        "empty_reason": None,
    }
    base = {**defaults, **overrides}
    return SearchMetadata(**base)


class TestContribution:
    def test_display_rank_is_one_based(self):
        assert Contribution(BM25, "BM25", 0, 1.0, 0.5).display_rank == 1
        assert Contribution(BM25, "BM25", 9, 1.0, 0.5).display_rank == 10

    def test_to_dict_carries_both_rank_conventions(self):
        # Both conventions are in the dict on purpose: a consumer of the JSON
        # should never have to guess which one the provenance field uses.
        payload = Contribution(BM25, "BM25", 4, 1.25, 0.0164).to_dict()
        assert payload["rank"] == 4
        assert payload["display_rank"] == 5


class TestExplainHit:
    def test_contributions_sort_by_rrf_contribution_descending(self):
        e = explain_hit(_hit())
        assert [c.retriever for c in e.contributions] == [BM25, SEMANTIC]

    def test_rrf_contribution_sum_equals_the_fused_score(self):
        hit = _hit()
        e = explain_hit(hit)
        assert abs(sum(c.rrf_contribution for c in e.contributions) - hit.score) < 1e-12

    def test_both_retrievers_flag_and_wording(self):
        e = explain_hit(_hit(sources=(BM25, SEMANTIC)))
        assert e.both_retrievers is True
        assert any("Retrieved by both" in line for line in e.lines)

    def test_single_retriever_says_so_and_does_not_claim_agreement(self):
        e = explain_hit(
            _hit(sources=(SEMANTIC,), provenance=(HitProvenance(SEMANTIC, 4, 0.7, 1 / 65),))
        )
        assert e.both_retrievers is False
        joined = " ".join(e.lines)
        assert "only" in joined
        assert "Retrieved by both" not in joined
        assert "Agreement" not in joined

    def test_lines_report_each_retriever_rank_and_contribution(self):
        e = explain_hit(_hit())
        joined = " ".join(e.lines)
        # 0-based provenance rank 2 must be rendered as 1-based rank 3.
        assert "rank 3" in joined
        assert f"{1 / 63:.6f}" in joined

    def test_raw_scores_are_labelled_not_comparable(self):
        e = explain_hit(_hit())
        joined = " ".join(e.lines)
        assert "not comparable" in joined
        # A bare comparison of the two raw scores would be the misleading form.
        assert "similarity" not in joined.lower()

    def test_never_claims_relevance(self):
        joined = " ".join(explain_hit(_hit()).lines).lower()
        for forbidden in (
            "relevant because",
            "is relevant",
            "answers your",
            "semantically similar",
        ):
            assert forbidden not in joined

    def test_caveat_is_attached_to_every_explanation(self):
        e = explain_hit(_hit())
        assert e.caveat == CAVEAT
        assert "not a statement that the document answers" in e.caveat

    def test_missing_sources_render_as_unknown_rather_than_raising(self):
        hit = SearchHit(doc_id="1", title="t", snippet="s", rank=1, score=0.5, sources=())
        e = explain_hit(hit)
        assert e.sources == ("unknown",)
        assert any("unknown" in line.lower() for line in e.lines)

    def test_unknown_retriever_key_passes_through_the_label(self):
        # A new retriever must be explainable before someone edits the dict.
        assert retriever_label("bm25") == "BM25 (lexical)"
        assert retriever_label("sparse-neural") == "sparse-neural"

    def test_explanation_preserves_identity_and_rank(self):
        hit = _hit(rank=7)
        e = explain_hit(hit)
        assert e.doc_id == hit.doc_id
        assert e.rank == 7
        assert e.title == hit.title

    def test_explanation_is_pure(self):
        # Rendering twice must be identical, and must not mutate the hit.
        hit = _hit()
        before = hit.to_dict()
        assert explain_hit(hit).to_dict() == explain_hit(hit).to_dict()
        assert hit.to_dict() == before


class TestExplainQueryProcessing:
    def test_disabled_is_stated_explicitly(self):
        # A panel that only speaks when a rewrite happens is indistinguishable
        # from a pipeline that never runs.
        out = explain_query_processing(_metadata(query_processing_enabled=False))
        assert "did not run" in " ".join(out["lines"])

    def test_enabled_but_no_change_is_stated(self):
        out = explain_query_processing(_metadata(was_modified=False))
        assert "left the query unchanged" in " ".join(out["lines"])

    def test_rewrite_is_quoted(self):
        out = explain_query_processing(
            _metadata(was_modified=True), original="a  b", processed="a b"
        )
        assert "rewritten" in " ".join(out["lines"])
        assert out["processed"] == "a b"

    def test_correction_is_reported_with_both_terms(self):
        md = _metadata(
            was_corrected=True,
            corrections=({"original": "turbine", "corrected": "turin"},),
        )
        joined = " ".join(explain_query_processing(md)["lines"])
        assert "turbine" in joined and "turin" in joined

    def test_corrected_flag_with_no_record_is_called_out(self):
        md = _metadata(was_corrected=True, corrections=())
        joined = " ".join(explain_query_processing(md)["lines"])
        assert "no correction was recorded" in joined

    def test_synonym_expansion_is_reported(self):
        md = _metadata(expansions=({"term": "car", "expansions": ["auto"]},))
        joined = " ".join(explain_query_processing(md)["lines"])
        assert "Synonym expansion" in joined and "car" in joined

    def test_absent_metadata_says_so_instead_of_guessing(self):
        out = explain_query_processing(None)
        assert out["enabled"] is None
        assert "not recorded" in " ".join(out["lines"])


class TestExplainResponse:
    def _response(self, **meta):
        return ApplicationSearchResponse(
            original_query="beethoven piano trios",
            processed_query="beethoven piano trios",
            results=(_hit(rank=1),),
            metadata=_metadata(**meta),
        )

    def test_explains_every_result(self):
        out = explain_response(self._response())
        assert len(out.results) == 1
        assert out.results[0].doc_id == "3437"

    def test_passes_original_and_processed_queries_through(self):
        out = explain_response(self._response())
        assert out.query_explanation["original"] == "beethoven piano trios"

    def test_empty_response_yields_no_results_and_still_explains(self):
        response = ApplicationSearchResponse(
            original_query="q", processed_query="q", results=(), metadata=_metadata()
        )
        out = explain_response(response)
        assert out.results == ()
        assert out.query_explanation["enabled"] is True

    def test_to_dict_is_json_serialisable(self):
        import json

        out = explain_response(self._response())
        assert json.loads(json.dumps(out.to_dict()))["results"][0]["both_retrievers"] is True


class TestNoRankingSideEffects:
    def test_explanation_cannot_reorder_results(self):
        # The renderer is handed a finished response. Whatever order it emits
        # explanations in, the order it *receives* is the served ranking.
        response = ApplicationSearchResponse(
            original_query="q",
            processed_query="q",
            results=(_hit(rank=1), _hit(rank=2), _hit(rank=3)),
            metadata=_metadata(),
        )
        before = [h.doc_id for h in response.results]
        explain_response(response)
        assert [h.doc_id for h in response.results] == before

    def test_module_imports_no_retrieval_component(self):
        # Structural guarantee, not a style preference: if this module ever
        # imports a retriever it could, in principle, re-rank.
        import semantic_search.explainability.explanations as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "semantic_search.search",
            "semantic_search.index",
            "semantic_search.query",
            "semantic_search.embeddings",
        ):
            assert forbidden not in source
