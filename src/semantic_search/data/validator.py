"""Corpus validation and quality reporting.

The build pipeline runs this after preprocessing so that a bad corpus fails
loudly at build time rather than surfacing as mysteriously poor recall at query
time. Checks are split into *errors* (which stop the build) and *warnings* (which
are reported and, optionally, tolerated).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from semantic_search.data.loader import Document

# Characters that indicate a badly decoded or binary payload.
_REPLACEMENT_CHAR = "�"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(slots=True)
class ValidationIssue:
    """A single finding, tagged by severity."""

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    document_id: str | None = None

    def __str__(self) -> str:
        location = f" [doc {self.document_id}]" if self.document_id else ""
        return f"{self.severity.upper()}: {self.code}{location}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    """Aggregate result of validating a corpus."""

    document_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str, document_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, document_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "document_count": self.document_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "document_id": issue.document_id,
                }
                for issue in self.issues[:100]
            ],
            "stats": self.stats,
        }

    def summary(self) -> str:
        return (
            f"{self.document_count} documents | "
            f"{len(self.errors)} errors | {len(self.warnings)} warnings | "
            f"valid={self.valid}"
        )


def compute_corpus_stats(documents: Sequence[Document]) -> dict[str, Any]:
    """Descriptive statistics used for reporting and index metadata."""
    if not documents:
        return {"document_count": 0}
    lengths = [document.length for document in documents]
    total_chars = sum(lengths)
    unique_ids = len({document.id for document in documents})
    unique_titles = len({document.title.lower() for document in documents})
    token_estimate = [len(document.searchable_text.split()) for document in documents]
    sorted_lengths = sorted(lengths)

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, int(fraction * (len(values) - 1)))
        return values[index]

    return {
        "document_count": len(documents),
        "unique_ids": unique_ids,
        "unique_titles": unique_titles,
        "total_characters": total_chars,
        "min_length": sorted_lengths[0],
        "max_length": sorted_lengths[-1],
        "mean_length": round(total_chars / len(documents), 1),
        "median_length": percentile(sorted_lengths, 0.5),
        "p95_length": percentile(sorted_lengths, 0.95),
        "mean_tokens": round(sum(token_estimate) / len(token_estimate), 1),
        "vocabulary_size": len(
            {token for document in documents for token in document.searchable_text.lower().split()}
        ),
    }


def validate_corpus(
    documents: Sequence[Document],
    *,
    min_length: int = 50,
    max_length: int = 20_000,
    strict: bool = False,
    max_reported_issues: int = 200,
) -> ValidationReport:
    """Validate a preprocessed corpus.

    Args:
        documents: The corpus to check.
        min_length: Bodies shorter than this are errors.
        max_length: Bodies longer than this are warnings (they were expected to
            be truncated during preprocessing).
        strict: Promote warnings to errors.
        max_reported_issues: Cap on recorded issues so a badly broken corpus
            cannot produce a multi-megabyte report.

    Returns:
        A :class:`ValidationReport`. ``report.valid`` is the build gate.
    """
    report = ValidationReport(document_count=len(documents))
    if not documents:
        report.add("error", "empty_corpus", "Corpus contains no documents.")
        return report

    seen_ids: Counter[str] = Counter()
    seen_titles: Counter[str] = Counter()
    overlong = 0
    short = 0
    replacement_chars = 0
    control_chars = 0

    for document in documents:
        if len(report.issues) >= max_reported_issues:
            report.add(
                "info",
                "issue_truncated",
                f"Stopped recording issues after {max_reported_issues}; run again to see more.",
            )
            break

        seen_ids[document.id] += 1
        seen_titles[document.title.lower().strip()] += 1

        if not document.id.strip():
            report.add("error", "missing_id", "Document has an empty id.", document.id)
        if not document.title.strip():
            report.add("error", "missing_title", "Document has an empty title.", document.id)
        if not document.text.strip():
            report.add("error", "empty_text", "Document body is empty.", document.id)
            continue

        if len(document.text) < min_length:
            short += 1
            report.add("error", "too_short", f"Body is {len(document.text)} chars.", document.id)
        if len(document.text) > max_length:
            overlong += 1
            report.add(
                "warning",
                "too_long",
                f"Body is {len(document.text)} chars (>{max_length}).",
                document.id,
            )
        if _REPLACEMENT_CHAR in document.text:
            replacement_chars += 1
            report.add(
                "warning",
                "replacement_char",
                "Body contains U+FFFD replacement characters.",
                document.id,
            )
        if _CONTROL_RE.search(document.text):
            control_chars += 1
            report.add("warning", "control_chars", "Body contains control characters.", document.id)

    duplicates = [doc_id for doc_id, count in seen_ids.items() if count > 1]
    if duplicates:
        report.add(
            "error",
            "duplicate_ids",
            f"{len(duplicates)} duplicate document ids (first: {duplicates[:5]}).",
        )
    duplicate_titles = [title for title, count in seen_titles.items() if count > 1 and title]
    if duplicate_titles:
        report.add(
            "warning",
            "duplicate_titles",
            f"{len(duplicate_titles)} titles appear more than once (first: {duplicate_titles[:5]}).",
        )
    if short > len(documents) * 0.01:
        report.add("warning", "high_short_ratio", f"{short} documents are below min_length.")
    if overlong > len(documents) * 0.01:
        report.add("warning", "high_long_ratio", f"{overlong} documents exceed max_length.")

    report.stats = {
        **compute_corpus_stats(documents),
        "short_documents": short,
        "overlong_documents": overlong,
        "documents_with_replacement_chars": replacement_chars,
        "documents_with_control_chars": control_chars,
        "duplicate_id_count": len(duplicates),
        "duplicate_title_count": len(duplicate_titles),
    }
    if strict and report.warnings:
        for warning in report.warnings:
            warning.severity = "error"
    return report
