"""Unit tests for the text cleaning pipeline.

Cleaning is where silent recall loss happens: a stage that is too aggressive
strips exactly the rare entities keyword search depends on. The tests below
assert both directions -- that noise is removed, and that signal survives.
"""

from __future__ import annotations

import pytest

from semantic_search.data.preprocessor import (
    TextPreprocessor,
    build_embedding_text,
    clean_title,
    clean_wiki_markup,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
    truncate_text,
)


class TestNormalizeUnicode:
    def test_folds_fullwidth_characters(self) -> None:
        assert normalize_unicode("ＡＢＣ") == "ABC"

    def test_removes_zero_width_characters(self) -> None:
        assert normalize_unicode("Anar​chism") == "Anarchism"

    def test_removes_bidi_control_characters(self) -> None:
        # U+202E REVERSED OVERRIDE can reverse displayed text -- a real spoofing
        # vector in a search result, so it must not survive into the index.
        assert normalize_unicode("evil‮txt") == "eviltxt"

    def test_preserves_newlines_and_tabs(self) -> None:
        assert normalize_unicode("a\nb\tc") == "a\nb\tc"

    def test_removes_null_and_bell_characters(self) -> None:
        assert normalize_unicode("a\x00b\x07c") == "abc"

    def test_empty_string(self) -> None:
        assert normalize_unicode("") == ""


class TestStripHtml:
    def test_removes_tags_but_keeps_content(self) -> None:
        assert "hello" in strip_html("<b>hello</b>")
        assert "<b>" not in strip_html("<b>hello</b>")

    def test_unescapes_entities(self) -> None:
        assert strip_html("caf&eacute;") == "café"

    def test_drops_script_contents_entirely(self) -> None:
        result = strip_html("<script>alert('x')</script>safe")
        assert "alert" not in result
        assert "safe" in result

    def test_drops_style_contents_entirely(self) -> None:
        assert "color" not in strip_html("<style>color: red</style>text")

    def test_handles_unclosed_tag(self) -> None:
        # Wikipedia dumps contain unbalanced tags; must not raise.
        assert strip_html("a <b b").startswith("a")


class TestCleanWikiMarkup:
    def test_extracts_piped_link_label(self) -> None:
        assert clean_wiki_markup("[https://example.org Wikipedia]").strip() == "Wikipedia"

    def test_removes_bare_urls(self) -> None:
        assert "http" not in clean_wiki_markup("see https://example.org/x now")

    def test_removes_reference_brackets(self) -> None:
        assert "ref" not in clean_wiki_markup("text[12]more").lower()

    def test_removes_citation_tails(self) -> None:
        result = clean_wiki_markup("a claim (citation needed) here")
        assert "citation needed" not in result

    def test_keeps_section_heading_word(self) -> None:
        # "== History ==" carries navigational noise but "History" is a
        # meaningful term a user might search for, so the word is kept.
        assert "History" in clean_wiki_markup("== History ==\nBody text.")

    def test_removes_list_bullets(self) -> None:
        result = clean_wiki_markup("* one\n* two")
        assert not result.lstrip().startswith("*")


class TestNormalizeWhitespace:
    def test_collapses_spaces(self) -> None:
        assert normalize_whitespace("a    b") == "a b"

    def test_collapses_blank_lines(self) -> None:
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_trailing_whitespace(self) -> None:
        assert normalize_whitespace("  padded  ") == "padded"

    def test_is_idempotent(self) -> None:
        once = normalize_whitespace("a  \n\n\n b   c ")
        assert normalize_whitespace(once) == once


class TestCleanTitle:
    def test_strips_disambiguation_suffix(self) -> None:
        assert clean_title("Mercury (disambiguation)") == "Mercury"

    def test_keeps_other_parenthetical_qualifiers(self) -> None:
        # "Mercury (planet)" is a real title; only the disambiguator is removed.
        assert clean_title("Mercury (planet)") == "Mercury (planet)"

    def test_unescapes_entities(self) -> None:
        assert clean_title("Beyoncé") == "Beyoncé"

    def test_handles_none(self) -> None:
        assert clean_title("") == ""


class TestBuildEmbeddingText:
    def test_prefixes_title(self) -> None:
        assert build_embedding_text("Albedo", "Reflectivity measure.") == (
            "Albedo. Reflectivity measure."
        )

    def test_returns_body_when_title_empty(self) -> None:
        assert build_embedding_text("", "Body only.") == "Body only."

    def test_truncates_runaway_title(self) -> None:
        result = build_embedding_text("T" * 500, "Body", title_max_length=10)
        assert result.startswith("T" * 10 + ". ")

    def test_never_duplicates_an_inline_title(self) -> None:
        # The preprocessor strips the title from the body, so the embedding text
        # must not contain it twice.
        assert build_embedding_text("Foo", "Bar baz.").count("Foo") == 1


class TestTruncateText:
    def test_no_truncation_when_short(self) -> None:
        assert truncate_text("short", 100) == ("short", False)

    def test_truncates_on_sentence_boundary(self) -> None:
        # The sentence-boundary branch only fires when the last terminator is
        # past 60% of the limit; cutting at 5% would throw away most of the text.
        text = "First sentence here. Second sentence here. Third one."
        result, was_truncated = truncate_text(text, 50)
        assert was_truncated is True
        assert result == "First sentence here. Second sentence here."
        assert len(result) <= 50

    def test_falls_back_to_word_boundary_when_cut_is_too_early(self) -> None:
        text = "First sentence here. Second sentence here. Third one."
        result, was_truncated = truncate_text(text, 40)
        assert was_truncated is True
        assert result == "First sentence here. Second sentence"

    def test_truncates_on_word_boundary_without_punctuation(self) -> None:
        result, was_truncated = truncate_text("alpha beta gamma delta epsilon", 12)
        assert was_truncated is True
        assert " " not in result.split()[-1] or result.endswith("a")

    def test_zero_max_length_is_a_noop(self) -> None:
        # max_length <= 0 means "unbounded" rather than "empty".
        assert truncate_text("abc", 0) == ("abc", False)

    def test_result_never_exceeds_limit(self) -> None:
        text = "word " * 1000
        result, _ = truncate_text(text, 50)
        assert len(result) <= 50


class TestTextPreprocessor:
    def test_full_pipeline(self) -> None:
        pre = TextPreprocessor(min_length=10, max_length=1000)
        cleaned = pre.process("<p>  Anarchism   is  a  political  philosophy.  </p>")
        assert cleaned.startswith("Anarchism is a")

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="empty after cleaning"):
            TextPreprocessor(min_length=5).process("   \n\n   ")

    def test_rejects_short_text(self) -> None:
        with pytest.raises(ValueError, match="shorter than min_length"):
            TextPreprocessor(min_length=100).process("too short")

    def test_truncates_long_text(self) -> None:
        pre = TextPreprocessor(min_length=10, max_length=100)
        assert len(pre.process("word " * 500)) <= 100

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            TextPreprocessor().process(None)  # type: ignore[arg-type]

    def test_stages_can_be_disabled_individually(self) -> None:
        pre = TextPreprocessor(strip_html_enabled=False, collapse_whitespace_enabled=False)
        assert "<b>" in pre.clean_body("<b>text</b>")

    def test_is_valid_threshold(self) -> None:
        pre = TextPreprocessor(min_length=20)
        assert pre.is_valid("x" * 20) is True
        assert pre.is_valid("x" * 19) is False

    def test_process_does_not_inline_title(self) -> None:
        pre = TextPreprocessor(min_length=5)
        assert "MyTitle" not in pre.process("body text here", title="MyTitle")

    def test_is_deterministic(self) -> None:
        pre = TextPreprocessor(min_length=5)
        text = "  <b>Some</b>   messy   text  with   markup.  "
        assert pre.process(text) == pre.process(text)
