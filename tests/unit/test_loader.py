"""Unit tests for the Document model and DocumentStore.

The store's id/row duality is the thing under test: if positional access and
id access ever disagree, every search result is silently attached to the wrong
document. ``validate_alignment`` is the guard, so it gets its own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_search.core.exceptions import DataError
from semantic_search.data.loader import Document, DocumentStore


class TestDocument:
    def test_requires_id(self) -> None:
        with pytest.raises(DataError, match="non-empty string"):
            Document(id="", title="T", text="body")

    def test_requires_title(self) -> None:
        with pytest.raises(DataError, match="missing a title"):
            Document(id="1", title="", text="body")

    def test_coerces_numeric_id(self) -> None:
        # Wikipedia ids arrive as ints from the dataset; row keys must be str.
        assert Document(id=12, title="T", text="b").id == "12"  # type: ignore[arg-type]

    def test_title_raw_defaults_to_title(self) -> None:
        assert Document(id="1", title="Mercury (planet)", text="b").title_raw == "Mercury (planet)"

    def test_searchable_text_includes_title(self) -> None:
        assert "Albedo" in Document(id="1", title="Albedo", text="body").searchable_text

    def test_length_counts_body_only(self) -> None:
        assert Document(id="1", title="Title", text="12345").length == 5

    def test_roundtrip_through_dict(self) -> None:
        original = Document(id="1", title="T", text="body", metadata={"url": "u"}, title_raw="raw")
        assert Document.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_from_dict_reports_missing_key(self) -> None:
        with pytest.raises(DataError, match="missing required key"):
            Document.from_dict({"title": "T", "text": "b"})

    def test_snippet_returns_short_text_whole(self) -> None:
        doc = Document(id="1", title="T", text="short body")
        assert doc.snippet(["body"]) == "short body"

    def test_snippet_centres_on_query_terms(self) -> None:
        text = ("filler " * 80) + "NEEDLE here " + ("filler " * 80)
        doc = Document(id="1", title="T", text=text)
        snippet = doc.snippet(["needle"], length=120)
        assert "NEEDLE" in snippet

    def test_snippet_falls_back_to_head_when_no_match(self) -> None:
        doc = Document(id="1", title="T", text="alpha " * 200)
        assert doc.snippet(["zzz"], length=50).startswith("alpha")

    def test_snippet_marks_ellipsis(self) -> None:
        doc = Document(id="1", title="T", text="word " * 500)
        assert doc.snippet(["word"], length=60).endswith("...")


class TestDocumentStore:
    def test_construction_and_length(self, sample_documents: list[Document]) -> None:
        assert len(DocumentStore(sample_documents)) == 5

    def test_rejects_duplicate_ids(self) -> None:
        docs = [Document(id="1", title="A", text="x"), Document(id="1", title="B", text="y")]
        with pytest.raises(DataError, match="Duplicate document id"):
            DocumentStore(docs)

    def test_positional_and_id_access_agree(self, document_store: DocumentStore) -> None:
        for position, doc_id in enumerate(document_store.document_ids):
            assert document_store[position].id == doc_id
            assert document_store.get(doc_id) is document_store[position]

    def test_contains(self, document_store: DocumentStore) -> None:
        assert "1" in document_store
        assert "999" not in document_store

    def test_get_missing_returns_none(self, document_store: DocumentStore) -> None:
        assert document_store.get("nope") is None

    def test_get_many_skips_unknown(self, document_store: DocumentStore) -> None:
        assert [d.id for d in document_store.get_many(["1", "nope", "3"])] == ["1", "3"]

    def test_at_filters_out_of_range(self, document_store: DocumentStore) -> None:
        # FAISS can return -1 rows; those must not raise IndexError.
        assert [d.id for d in document_store.at([-1, 0, 99])] == [document_store[0].id]

    def test_titles(self, document_store: DocumentStore) -> None:
        assert "Albedo" in document_store.titles()

    def test_add_appends_and_returns_position(self, document_store: DocumentStore) -> None:
        position = document_store.add(Document(id="new", title="New", text="body"))
        assert position == len(document_store) - 1
        assert document_store.get("new") is not None

    def test_add_rejects_duplicate(self, document_store: DocumentStore) -> None:
        with pytest.raises(DataError, match="already exists"):
            document_store.add(Document(id="1", title="Dup", text="body"))


class TestAlignment:
    def test_passes_when_counts_match(self, document_store: DocumentStore) -> None:
        document_store.validate_alignment(5)

    def test_fails_on_count_mismatch(self, document_store: DocumentStore) -> None:
        with pytest.raises(DataError, match="Rebuild the index"):
            document_store.validate_alignment(4)


class TestSerialisation:
    def test_jsonl_roundtrip(self, document_store: DocumentStore, tmp_path: Path) -> None:
        path = document_store.save_jsonl(tmp_path / "docs.jsonl")
        assert DocumentStore.load_jsonl(path).document_ids == document_store.document_ids

    def test_creates_parent_directory(self, document_store: DocumentStore, tmp_path: Path) -> None:
        path = document_store.save_jsonl(tmp_path / "nested" / "deep" / "docs.jsonl")
        assert path.is_file()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            DocumentStore.load_jsonl(tmp_path / "absent.jsonl")

    def test_skips_malformed_lines_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "docs.jsonl"
        path.write_text(
            json.dumps({"id": "1", "title": "A", "text": "ok"})
            + "\n{not json}\n"
            + json.dumps({"title": "no id", "text": "x"})
            + "\n",
            encoding="utf-8",
        )
        assert len(DocumentStore.load_jsonl(path)) == 1

    def test_malformed_lines_can_be_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "docs.jsonl"
        path.write_text(json.dumps({"id": "1", "title": "A", "text": "ok"}) + "\n{bad}\n", "utf-8")
        with pytest.raises(DataError, match="Malformed document"):
            DocumentStore.load_jsonl(path, skip_malformed=False)

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "docs.jsonl"
        path.write_text(
            "\n\n" + json.dumps({"id": "1", "title": "A", "text": "ok"}) + "\n\n", "utf-8"
        )
        assert len(DocumentStore.load_jsonl(path)) == 1

    def test_unicode_survives_roundtrip(self, tmp_path: Path) -> None:
        store = DocumentStore([Document(id="1", title="Beyoncé", text="café life")])
        path = store.save_jsonl(tmp_path / "docs.jsonl")
        assert DocumentStore.load_jsonl(path)[0].title == "Beyoncé"
