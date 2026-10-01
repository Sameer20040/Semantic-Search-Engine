"""Tokenization shared by BM25, query processing, suggestions, and explanations.

Keeping a single tokenizer matters for a hybrid system: if the index tokenized
``"running"`` to ``"run"`` but the query processor did not, BM25 would silently
lose recall on inflected forms. Both sides import :class:`Tokenizer`.

Stemming uses a compact, dependency-free Porter-style suffix stripper
(:func:`stem`) rather than NLTK or Snowball. It is deterministic, needs no
corpus download at runtime, and covers the cases that matter for English
Wikipedia. This is a deliberate trade-off, documented in ``docs/architecture.md``.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Words are maximal runs of alphanumeric characters, keeping intra-word hyphens
# and apostrophes ("state-of-the-art", "don't") which carry real meaning.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

# A small, explicit English stopword list. A curated list beats NLTK's here: it
# has no download step and no risk of a runtime network dependency.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

# Extremely common Wikipedia boilerplate that carries no topical signal.
_BOILERPLATE_STOPWORDS: frozenset[str] = frozenset(
    {
        "may",
        "also",
        "however",
        "although",
        "thus",
        "therefore",
        "within",
        "upon",
        "which",
        "including",
        "according",
        "following",
        "one",
        "two",
        "three",
        "first",
        "second",
        "many",
        "much",
        "well",
        "way",
        "using",
        "used",
        "use",
        "known",
        "began",
        "born",
        "died",
    }
)

_VOWELS = frozenset("aeiou")

#: Words that end in ``-s`` without being plural. A rule-based stemmer cannot
#: distinguish these morphologically, so they are listed explicitly. This is the
#: same trade-off Porter's step 1a makes with its own stop list.
_INVARIANT_S = frozenset(
    {
        "always",
        "thus",
        "plus",
        "its",
        "this",
        "his",
        "as",
        "us",
        "was",
        "has",
        "is",
        "gas",
        "lens",
        "bias",
        "less",
        "across",
        "atlas",
        "canvas",
        "chaos",
        # Latin/Greek singulars that end in a consonant + "s".
        "series",
        "species",
        "basis",
        "analysis",
        "crisis",
        "thesis",
        "diagnosis",
        "oasis",
        "axis",
        "genesis",
        "hypothesis",
        "synthesis",
        "physics",
        "mathematics",
        "economics",
        "politics",
        "statistics",
        "news",
        # The stem of "business" is "busine": "-ness" here is part of the stem, not
        # the abstract-noun suffix, so the -ness rule must not fire.
        "business",
        "businesses",
        "busy",
        "witness",
        "witnesses",
        "darkness",
        "illness",
        "readiness",
        "willingness",
        "fitness",
        "blindness",
        "politeness",
        "sharpness",
    }
)

#: Endings that mark a word as already a stem; never strip past them.
_STEM_ENDINGS = ("sis", "ics", "ism", "us", "um", "ous", "ive", "ial", "ual", "ing", "ate")

#: Ordered (suffix, replacement) rules. Order is significant: the first match
#: wins, so longer/more specific suffixes must come first.
_DERIVATIONAL_RULES: tuple[tuple[str, str], ...] = (
    ("izations", "ize"),
    ("ization", "ize"),
    ("ational", "ate"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("alities", "al"),
    ("ality", "al"),
    ("ements", "e"),
    ("ement", "e"),
    ("ments", "ment"),
    ("ingly", ""),
    ("edly", ""),
    ("sses", "ss"),
    ("ies", "y"),
    ("ied", "y"),
    ("ing", ""),
    ("ness", ""),
    ("ment", ""),
    ("able", ""),
    ("ible", ""),
    ("ance", ""),
    ("ence", ""),
    ("ly", ""),
)

#: Suffixes whose removal legitimately leaves a trailing ``i``; for these the
#: orthographic ``-y`` restore below must not fire (the stem of "happiness" is
#: "happy", the stem of "information" is "informat").
_STEM_ENDING_I = ("at", "ic", "ess", "ific", "tion", "sion", "ate", "ise", "ize")

#: Sibilant suffixes ("-es", "-ance", ...). A stem already ending in s/x/z before
#: one of these is a consonant cluster, not a suffix boundary, so the rule is
#: skipped -- otherwise "wishes" would be stemmed to "wish" and "boxes" to "boxe".
_SIBILANT_SUFFIXES = ("ance", "ence", "es", "ed", "er", "est", "ly")


def _collapse_doubled_consonant(stem: str) -> str:
    """``runn`` -> ``run``; ``stopp`` -> ``stop``. Leaves ``-ss``/``-ll`` intact."""
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] not in _VOWELS
        and stem[-1] != "s"
        and stem[-1] != "l"
    ):
        return stem[:-1]
    return stem


def _restore_y(stem: str, suffix: str) -> str:
    """``happi`` -> ``happy``: many -ness stems are orthographic -y words.

    Only applied when the removed suffix could leave a spurious ``i``. Notably
    it must not fire for ``-ness`` removal from ``busine`` (whose real stem is
    ``busine``, not ``busy``) nor for ``-tion``/``-sion`` (whose -i is real).
    """
    if suffix not in _STEM_ENDING_I and stem.endswith("i") and len(stem) >= 2:
        return stem[:-1] + "y"
    return stem


@lru_cache(maxsize=1 << 16)
def stem(word: str) -> str:
    """Reduce an English word to a crude stem.

    A compact Porter-style stripper, not a full Porter/Snowball implementation.
    It covers the regular inflections that dominate an encyclopedia vocabulary
    (plurals, ``-ing``/``-ed``, ``-tion``/``-sion``, ``-ness``, ``-ly``).

    Over-stemming is the main hazard in IR -- collapsing ``general`` into
    ``generate`` loses precision -- so three guards apply to every rule:

    * words shorter than 4 characters, and words in :data:`_INVARIANT_S`, are
      returned unchanged;
    * a rule only fires if at least 3 characters of stem remain;
    * a word already ending in a stem-like ending (:data:`_STEM_ENDINGS`) is not
      stripped past that point.

    Args:
        word: A lowercase, alphabetic token.

    Returns:
        The stem, or ``word`` unchanged if no rule applied.
    """
    if len(word) < 4 or not word.isalpha() or word.upper() == word or word in _INVARIANT_S:
        return word
    if not any(ch in _VOWELS for ch in word):
        return word  # no vowel -> no English suffix to strip

    # --- Step 1: derivational suffixes --------------------------------------
    # Derivational rules run BEFORE the plural strip. "happiness" must reach the
    # -ness rule as a whole word; stripping the plural -s first would leave
    # "happines", which no rule matches.
    if len(word) >= 4:
        for suffix, replacement in _DERIVATIONAL_RULES:
            if not word.endswith(suffix) or word == suffix:
                continue
            remainder = word[: -len(suffix)]
            if len(remainder) < 3:
                continue
            if suffix in _STEM_ENDINGS and remainder.endswith(_STEM_ENDINGS):
                # e.g. do not turn "analysis" into "analysi".
                continue
            if suffix in _SIBILANT_SUFFIXES and remainder.endswith(("s", "x", "z")):
                # "happiness" -> remainder "happi" ends in i, fine, but
                # "business" -> remainder "busi" is blocked by _INVARIANT_S.
                # This guard covers sibilant stems such as "wishes" -> "wish".
                continue
            candidate = remainder + replacement
            if candidate.endswith("i") and replacement == "":
                candidate = _restore_y(candidate, suffix)
            return _collapse_doubled_consonant(candidate)

    # --- Step 2: past tense and plural endings (Porter step 1b/1a) -----------
    # "-ied" restores the "y" ("studied"->"study") and must be tried before the
    # generic "-ed" rule, which would leave "studyd".
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    # "-ed" after a voiceless consonant drops the "e" ("played"->"play");
    # after a t/d it only the "d" is inflectional ("wanted"->"want").
    if word.endswith("ed") and len(word) > 3 and word[-3] not in "aeioulrs":
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # Sibilant plurals: the "e" is a sound marker and is not part of the stem.
    # "boxes"->"box", "churches"->"church", "dishes"->"dish", "watches"->"watch".
    if word.endswith(("shes", "ches", "xes", "zes", "sses")):
        return word[:-2]
    # A voiced consonant before "-es" implies a silent "e" that belongs to the
    # stem: "crises"->"crisis", "buses"->"bus", "makes"->"make". The consonant
    # tested is the one *inside* the "es" suffix, i.e. word[-2].
    if word.endswith("es") and len(word) > 3 and word[-3] not in "aeiou":
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "as", "os")):
        return word[:-1]
    return word


class Tokenizer:
    """Configurable analyzer producing a normalized token stream.

    Pipeline: unicode normalize -> regex tokenize -> casefold -> length filter ->
    stopword removal -> optional stemming. The order matters: stopword removal
    happens on the *surface* form, so a stem can never accidentally re-introduce
    a stopword.

    Example::

        tok = Tokenizer(stopwords="en", use_stemming=True)
        tok.tokenize("The Running Dogs are running!")
        # -> ['run', 'dog']
    """

    def __init__(
        self,
        *,
        stopwords: str | frozenset[str] | None = "en",
        use_stemming: bool = True,
        min_length: int = 2,
        max_length: int = 40,
        remove_stopwords: bool = True,
        remove_boilerplate: bool = False,
    ) -> None:
        self.use_stemming = use_stemming
        self.min_length = min_length
        self.max_length = max_length
        self.remove_stopwords = remove_stopwords
        self.remove_boilerplate = remove_boilerplate

        if isinstance(stopwords, frozenset | set):
            self.stopwords: frozenset[str] = frozenset(stopwords)
        elif stopwords == "en":
            self.stopwords = ENGLISH_STOPWORDS
        elif stopwords in (None, "", "none"):
            self.stopwords = frozenset()
        else:
            raise ValueError(
                f"Unsupported stopword set: {stopwords!r} (use 'en', 'none', or a set)."
            )

        if remove_boilerplate:
            self.stopwords = self.stopwords | _BOILERPLATE_STOPWORDS

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        """Return the analyzed token stream for ``text``."""
        if not text:
            return []
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens: list[str] = []
        for match in _TOKEN_RE.finditer(normalized):
            token = match.group(0)
            if not self.min_length <= len(token) <= self.max_length:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            if any(ch.isdigit() for ch in token) and not any(ch.isalpha() for ch in token):
                continue  # pure numbers carry little signal in an encyclopedia corpus
            tokens.append(stem(token) if self.use_stemming else token)
        return tokens

    def analyze(self, text: str) -> list[str]:
        """Alias of :meth:`tokenize`, for symmetry with analyzer-style call sites."""
        return self.tokenize(text)

    def unique_tokens(self, text: str) -> list[str]:
        """Order-preserving unique tokens (used for phrase/keyword extraction)."""
        seen: set[str] = set()
        result: list[str] = []
        for token in self.tokenize(text):
            if token not in seen:
                seen.add(token)
                result.append(token)
        return result

    def surface_terms(self, text: str) -> list[str]:
        """Casefolded surface forms with stopwords removed but *no* stemming.

        Used for explanations and highlighting, where showing the user the actual
        word they typed is more useful than showing a stem.
        """
        analyzer = Tokenizer(
            stopwords=self.stopwords if self.remove_stopwords else None,
            use_stemming=False,
            min_length=self.min_length,
            max_length=self.max_length,
            remove_boilerplate=self.remove_boilerplate,
        )
        return analyzer.tokenize(text)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(stemming={self.use_stemming}, stopwords={len(self.stopwords)}, "
            f"min_length={self.min_length})"
        )


@lru_cache(maxsize=4)
def get_tokenizer(
    stopwords: str | None = "en", use_stemming: bool = True, min_length: int = 2
) -> Tokenizer:
    """Cached tokenizer factory -- tokenizers are immutable and cheap to share."""
    return Tokenizer(stopwords=stopwords, use_stemming=use_stemming, min_length=min_length)
