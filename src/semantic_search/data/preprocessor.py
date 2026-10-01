"""Text cleaning and normalization for the Wikipedia corpus.

The pipeline is deliberately conservative: over-aggressive cleaning destroys the
rare entities that make keyword search work, so each stage is a separate,
independently testable function and the whole chain is opt-out per stage.

Stages (in order):

1. Unicode normalization (NFKC) -- folds full-width forms, ligatures, etc.
2. HTML/markup removal -- strips tags and unescapes entities.
3. Control/whitespace normalization -- collapses runs of whitespace, removes
   zero-width and bidi control characters.
4. Truncation -- hard-capped so one pathological article cannot dominate the
   BM25 length normalisation.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

# Characters that carry no linguistic content but break naive tokenization.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[-‏‪-‮⁠-⁤﻿]")
_WHITESPACE = re.compile(r"[ \t  - ]+")
_NEWLINES = re.compile(r"\n{3,}")
# Wikipedia section headers such as "== History ==" or "=== Early life ===".
_SECTION_HEADING = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$", re.MULTILINE)
_REFERENCE_BRACKET = re.compile(r"\[\s*\d+\s*\]")
_LIST_BULLET = re.compile(r"^\s*[*#:;•]\s+", re.MULTILINE)
_PIPE_LINK = re.compile(r"\[(?:https?://\S+)\s+([^\]]*?)\]")
_BARE_URL = re.compile(r"https?://\S+")
_CITATION_TAIL = re.compile(r"\s*\((?:citation needed|who|whom)\)\s*", re.IGNORECASE)
_MULTI_SPACE = re.compile(r" {2,}")


@dataclass(slots=True)
class PreprocessStats:
    """Counters describing what a preprocessing run changed.

    Surfaced by ``scripts/preprocess_data.py`` so data quality is observable
    rather than assumed.
    """

    total: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_malformed: int = 0
    truncated: int = 0
    duplicates: int = 0
    unicode_replacements: int = 0
    characters_in: int = 0
    characters_out: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return (
            self.total
            - self.dropped_empty
            - self.dropped_too_short
            - self.dropped_malformed
            - self.duplicates
        )

    def summary(self) -> str:
        return (
            f"kept={self.kept}/{self.total} dropped_empty={self.dropped_empty} "
            f"dropped_short={self.dropped_too_short} dropped_malformed={self.dropped_malformed} "
            f"duplicates={self.duplicates} truncated={self.truncated} "
            f"chars {self.characters_in}->{self.characters_out}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_empty": self.dropped_empty,
            "dropped_too_short": self.dropped_too_short,
            "dropped_malformed": self.dropped_malformed,
            "truncated": self.truncated,
            "duplicates": self.duplicates,
            "characters_in": self.characters_in,
            "characters_out": self.characters_out,
            "errors": self.errors[:20],
        }


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization and strip zero-width/bidi control characters.

    NFKC is preferred over NFC here because Wikipedia text is full of
    compatibility forms (superscripts, ligatures, unit symbols) that make exact
    keyword matching fail when a user types the plain ASCII equivalent.
    """
    text = _ZERO_WIDTH.sub("", text)
    normalized = unicodedata.normalize("NFKC", text)
    # ``unicodedata.normalize`` leaves some invisible format characters intact.
    return "".join(
        ch
        for ch in normalized
        if ch == "\n" or ch == "\t" or unicodedata.category(ch) not in ("Cf", "Cc")
    )


def strip_html(text: str) -> str:
    """Remove HTML/markup and unescape entities.

    Wikipedia dumps from ``wikimedia/wikipedia`` are mostly plaintext, but mixed
    sources contain inline ``<ref>``/``<sup>`` markup, which would otherwise be
    tokenized as noise. Tag *contents* are preserved (so footnote text survives),
    only the tags themselves are dropped.
    """
    text = re.sub(
        r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]{0,400}>", " ", text)
    return html.unescape(text)


def clean_wiki_markup(text: str) -> str:
    """Remove Wikipedia-specific markup that adds no retrieval value."""
    text = _CITATION_TAIL.sub(" ", text)
    text = _REFERENCE_BRACKET.sub(" ", text)
    # Convert [https://example.org Label] -> Label, then drop bare URLs.
    text = _PIPE_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)
    text = _LIST_BULLET.sub("", text)
    # Section headings carry navigation, not content; keep the heading word itself.
    text = _SECTION_HEADING.sub(r"\1. ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace runs and excess blank lines."""
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_title(title: str) -> str:
    """Normalize a Wikipedia article title.

    Disambiguators such as ``Mercury (planet)`` are stripped from the display
    title but preserved in :attr:`Document.title_raw` so they remain searchable.
    """
    title = normalize_unicode(html.unescape(title or "")).strip()
    title = re.sub(r"\s*\(disambiguation\)\s*$", "", title, flags=re.IGNORECASE)
    return _WHITESPACE.sub(" ", title).strip()


def build_embedding_text(title: str, text: str, title_max_length: int = 120) -> str:
    """Compose the string actually fed to the sentence encoder.

    Sentence-BERT was trained on natural sentences; a bare paragraph body is a
    weaker signal than one that carries its subject. Prefixing the title supplies
    that subject cheaply -- the standard "title-as-context" trick that improves
    recall for short queries. The title is truncated so an absurdly long title
    (mostly disambiguators) cannot dominate the representation.
    """
    title = (title or "").strip()
    if not title:
        return text
    if title_max_length > 0 and len(title) > title_max_length:
        title = title[:title_max_length].rstrip()
    return f"{title}. {text}"


def truncate_text(text: str, max_length: int) -> tuple[str, bool]:
    """Truncate on a sentence/word boundary. Returns ``(text, was_truncated)``."""
    if max_length <= 0 or len(text) <= max_length:
        return text, False
    window = text[:max_length]
    # Prefer to end on the last sentence terminator, else the last space.
    for terminator in (". ", "! ", "? "):
        cut = window.rfind(terminator)
        if cut > max_length * 0.6:
            return window[: cut + 1], True
    space = window.rfind(" ")
    return (window[:space], True) if space > 0 else (window, True)


class TextPreprocessor:
    """Configurable cleaning pipeline.

    Example::

        pre = TextPreprocessor(min_length=50, max_length=20_000)
        cleaned = pre.process("  Some   <b>text</b>  ", title="Foo")
    """

    def __init__(
        self,
        *,
        min_length: int = 50,
        max_length: int = 20_000,
        normalize_unicode_enabled: bool = True,
        strip_html_enabled: bool = True,
        strip_wiki_markup_enabled: bool = True,
        collapse_whitespace_enabled: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_unicode_enabled = normalize_unicode_enabled
        self.strip_html_enabled = strip_html_enabled
        self.strip_wiki_markup_enabled = strip_wiki_markup_enabled
        self.collapse_whitespace_enabled = collapse_whitespace_enabled

    def clean_body(self, text: str) -> str:
        """Run the configured stages over a document body."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if self.normalize_unicode_enabled:
            text = normalize_unicode(text)
        if self.strip_html_enabled:
            text = strip_html(text)
        if self.strip_wiki_markup_enabled:
            text = clean_wiki_markup(text)
        if self.collapse_whitespace_enabled:
            text = normalize_whitespace(text)
        return text

    def is_valid(self, text: str) -> bool:
        """True when ``text`` carries enough content to be worth indexing."""
        return isinstance(text, str) and len(text.strip()) >= self.min_length

    def process(self, text: str, title: str = "") -> str:
        """Clean, validate, and truncate a body. Raises ``ValueError`` if unusable.

        ``title`` is accepted (and cleaned) for API symmetry with the rest of the
        pipeline; the body returned here never has the title inlined -- that is
        the job of :func:`build_embedding_text`.
        """
        if title:
            clean_title(title)
        cleaned = self.clean_body(text)
        if not cleaned:
            raise ValueError("Document text is empty after cleaning.")
        if len(cleaned) < self.min_length:
            raise ValueError(f"Document text is shorter than min_length={self.min_length}.")
        cleaned, _ = truncate_text(cleaned, self.max_length)
        return cleaned
