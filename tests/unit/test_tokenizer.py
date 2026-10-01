"""Unit tests for the shared tokenizer and stemmer.

The stemmer is the component most likely to silently degrade retrieval quality,
so it is tested against two tables:

* words that MUST reduce to a specific stem (recall requirement);
* words that MUST be left alone (precision requirement -- over-stemming such as
  ``general -> generate`` is the classic IR failure mode).
"""

from __future__ import annotations

import pytest

from semantic_search.data.tokenizer import (
    ENGLISH_STOPWORDS,
    Tokenizer,
    get_tokenizer,
    stem,
)

# word -> expected stem. Only unambiguous regular inflections belong here.
MUST_REDUCE = {
    "running": "run",
    "flies": "fly",
    "cities": "city",
    "studies": "study",
    "dogs": "dog",
    "cats": "cat",
    "happiness": "happy",
    "organizations": "organize",
    "nationalization": "nationalize",
    "abilities": "ability",
    "buses": "bus",
    "boxes": "box",
    "churches": "church",
    "watches": "watch",
    "dishes": "dish",
    "quickly": "quick",
    "played": "play",
    "moves": "mov",
}

# Words that must survive unchanged. Over-stemming these is a precision bug.
MUST_NOT_CHANGE = [
    "photosynthesis",
    "analysis",
    "basis",
    "series",
    "species",
    "business",
    "university",
    "general",
    "generate",
    "national",
    "always",
    "gas",
    "axis",
    "politics",
    "mathematics",
    "thus",
    "plus",
    "bias",
    "chaos",
    "process",
    "class",
    "dress",
    "is",
    "as",
    "was",
    "the",
]


@pytest.mark.parametrize(("word", "expected"), sorted(MUST_REDUCE.items()))
def test_stem_reduces_inflections(word: str, expected: str) -> None:
    assert stem(word) == expected


@pytest.mark.parametrize("word", MUST_NOT_CHANGE)
def test_stem_leaves_irregular_words_alone(word: str) -> None:
    assert stem(word) == word


def test_stem_is_deterministic() -> None:
    assert stem("organizations") == stem("organizations") == "organize"


def test_stem_handles_non_alphabetic_input() -> None:
    # Digits, hyphens and short tokens must not crash or be mangled.
    assert stem("1234") == "1234"
    assert stem("a") == "a"
    assert stem("") == ""


class TestTokenizer:
    def test_lowercases_and_splits(self) -> None:
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("The Quick Brown Fox") == ["quick", "brown", "fox"]

    def test_removes_english_stopwords(self) -> None:
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("the and of a an")
        assert tokens == []

    def test_stemming_can_be_disabled(self) -> None:
        tokenizer = Tokenizer(use_stemming=False)
        assert "running" in tokenizer.tokenize("running dogs")

    def test_stopword_removal_can_be_disabled(self) -> None:
        tokenizer = Tokenizer(remove_stopwords=False, use_stemming=False)
        assert "the" in tokenizer.tokenize("the fox")

    def test_keep_hyphenated_and_apostrophe_tokens(self) -> None:
        tokenizer = Tokenizer(use_stemming=False)
        tokens = tokenizer.tokenize("state-of-the-art don't")
        assert "state-of-the-art" in tokens
        assert "don't" in tokens

    def test_filters_short_and_long_tokens(self) -> None:
        tokenizer = Tokenizer(min_length=3, max_length=10, use_stemming=False)
        tokens = tokenizer.tokenize("a ab abc abcdefghijklmnop")
        assert tokens == ["abc"]

    def test_empty_input(self) -> None:
        assert Tokenizer().tokenize("") == []
        assert Tokenizer().tokenize("   ") == []

    def test_unicode_normalization(self) -> None:
        # Full-width Latin should fold to ASCII via NFKC.
        assert Tokenizer(use_stemming=False).tokenize("ＡＢＣ") == ["abc"]

    def test_unique_tokens_preserves_order(self) -> None:
        assert Tokenizer().unique_tokens("dog cat dog bird") == ["dog", "cat", "bird"]

    def test_surface_terms_skips_stemming(self) -> None:
        # Used for explanations/highlighting: show the user's actual words.
        assert Tokenizer().surface_terms("The Running Dogs") == ["running", "dogs"]

    def test_rejects_unknown_stopword_name(self) -> None:
        with pytest.raises(ValueError, match="Unsupported stopword set"):
            Tokenizer(stopwords="klingon")

    def test_repr_is_informative(self) -> None:
        assert "stemming=True" in repr(Tokenizer())

    def test_callable_alias(self) -> None:
        assert Tokenizer()("dogs") == Tokenizer().tokenize("dogs")

    def test_deterministic(self) -> None:
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("The Running Dogs") == tokenizer.tokenize("The Running Dogs")


def test_get_tokenizer_is_cached() -> None:
    assert get_tokenizer() is get_tokenizer()


def test_stopword_list_is_english() -> None:
    assert {"the", "and", "of", "is"} <= ENGLISH_STOPWORDS
