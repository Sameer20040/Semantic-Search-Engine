"""Unit tests for corpus validation.

Validation is the build gate: an invalid corpus must fail the build rather than
surface later as mysteriously poor recall. These tests pin down both the
error/warning split and the stats reported to the operator.
"""

from __future__ import annotations

from semantic_search.data.loader import Document
from semantic_search.data.validator import (
    ValidationIssue,
    compute_corpus_stats,
    validate_corpus,
)


def make_doc(
    doc_id: str = "1",
    title: str = "Title",
    text: str | None = None,
) -> Document:
    return Document(id=doc_id, title=title, text=text if text is not None else "x" * 200)


def codes(issues: list[ValidationIssue]) -> set[str]:
    return {issue.code for issue in issues}


class TestValidateCorpus:
    def test_clean_corpus_is_valid(self) -> None:
        report = validate_corpus([make_doc()], min_length=50, max_length=1000)
        assert report.valid is True
        assert report.errors == []

    def test_empty_corpus_is_an_error(self) -> None:
        report = validate_corpus([], min_length=50)
        assert report.valid is False
        assert "empty_corpus" in codes(report.issues)

    def test_detects_empty_body(self) -> None:
        report = validate_corpus([make_doc(text="   ")], min_length=10)
        assert "empty_text" in codes(report.issues)
        assert report.valid is False

    def test_detects_too_short(self) -> None:
        report = validate_corpus([make_doc(text="tiny")], min_length=50)
        assert "too_short" in codes(report.issues)
        assert report.valid is False

    def test_overlong_is_a_warning_not_an_error(self) -> None:
        # Truncation already ran by this point, so a long body is a reportable
        # anomaly but not a build stopper.
        report = validate_corpus([make_doc(text="x" * 500)], min_length=10, max_length=100)
        assert "too_long" in codes(report.warnings)
        assert report.valid is True

    def test_strict_mode_promotes_warnings(self) -> None:
        report = validate_corpus(
            [make_doc(text="x" * 500)], min_length=10, max_length=100, strict=True
        )
        assert report.valid is False

    def test_detects_duplicate_ids(self) -> None:
        docs = [make_doc("1", "A"), make_doc("1", "B")]
        report = validate_corpus(docs, min_length=10)
        assert "duplicate_ids" in codes(report.issues)
        assert report.valid is False

    def test_detects_replacement_characters(self) -> None:
        report = validate_corpus([make_doc(text="good text � bad " + "x" * 200)], min_length=10)
        assert "replacement_char" in codes(report.warnings)

    def test_detects_control_characters(self) -> None:
        report = validate_corpus(
            [make_doc(text="text\x01with control " + "x" * 200)], min_length=10
        )
        assert "control_chars" in codes(report.warnings)

    def test_duplicate_titles_are_only_a_warning(self) -> None:
        docs = [make_doc("1", "Same"), make_doc("2", "same")]
        report = validate_corpus(docs, min_length=10)
        assert "duplicate_titles" in codes(report.warnings)
        assert report.valid is True

    def test_issue_recording_is_capped(self) -> None:
        # The cap bounds the per-document findings; the "issue_truncated" notice
        # and the aggregate warnings are still recorded after it.
        docs = [make_doc(str(i), f"T{i}", "tiny") for i in range(50)]
        report = validate_corpus(docs, min_length=50, max_reported_issues=10)
        assert len(report.issues) <= 12
        assert "issue_truncated" in codes(report.issues)
        assert "high_short_ratio" in codes(report.issues)

    def test_report_to_dict_is_serialisable_shape(self) -> None:
        payload = validate_corpus([make_doc(text="tiny")], min_length=50).to_dict()
        assert set(payload) >= {"valid", "document_count", "error_count", "issues", "stats"}
        assert payload["error_count"] >= 1

    def test_summary_is_human_readable(self) -> None:
        assert "documents" in validate_corpus([make_doc()], min_length=10).summary()

    def test_issue_str_includes_document_id(self) -> None:
        issue = ValidationIssue("error", "too_short", "Body is 2 chars.", "42")
        assert "doc 42" in str(issue)

    def test_stats_are_populated(self) -> None:
        report = validate_corpus([make_doc(text="x" * 200)], min_length=10)
        assert report.stats["document_count"] == 1
        assert report.stats["total_characters"] == 200


class TestComputeCorpusStats:
    def test_empty_input(self) -> None:
        assert compute_corpus_stats([]) == {"document_count": 0}

    def test_reports_lengths_and_vocabulary(self) -> None:
        docs = [
            Document(id="1", title="Dog", text="dog dog cat"),
            Document(id="2", title="Cat", text="cat"),
        ]
        stats = compute_corpus_stats(docs)
        assert stats["document_count"] == 2
        assert stats["unique_ids"] == 2
        assert stats["max_length"] == 11
        assert stats["min_length"] == 3
        # mean_tokens is measured on searchable_text (title + body), so the
        # tokenizer-equivalent word count includes the title: "Dog dog dog cat"
        # is 4 tokens and "Cat cat" is 2, averaging 3.0.
        assert stats["mean_tokens"] == 3.0
        assert stats["vocabulary_size"] == 2

    def test_percentiles_are_ordered(self) -> None:
        docs = [Document(id=str(i), title="T", text="x" * (i + 1) * 10) for i in range(100)]
        stats = compute_corpus_stats(docs)
        assert (
            stats["min_length"]
            <= stats["median_length"]
            <= stats["p95_length"]
            <= stats["max_length"]
        )

    def test_title_case_duplicates_counted_case_insensitively(self) -> None:
        docs = [Document(id="1", title="Dog", text="a"), Document(id="2", title="dog", text="b")]
        assert compute_corpus_stats(docs)["unique_titles"] == 1
