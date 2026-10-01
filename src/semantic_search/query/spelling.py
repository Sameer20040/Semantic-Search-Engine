"""Conservative spell correction against the corpus vocabulary.

The hard part of spell correction in a search engine is not fixing typos. It is
**not breaking things that were not broken**, because a confident wrong rewrite
is worse than a missed correction: the user gets confidently wrong results and
no indication that anything was changed. Everything below follows from that.

Why one similarity threshold is not enough
------------------------------------------
I calibrated this against the real 300-document corpus vocabulary (39,796
terms) before writing the rule. The result was that no single threshold
separates typos from names:

* ``"machne" -> "machine"`` scores 92.3, but ``"teh"`` -- a typo every reader
  would agree is a typo -- scores only 85.7, and ``"recieve"`` 85.7.
* Lowering the threshold to catch those also "corrects" the proper noun
  ``"Serkan" -> "arakan"`` (66.7) and ``"hemoglobinn" ->
  "methemoglobinemia"`` (71.4).

A threshold high enough to protect names misses most typos. One low enough to
catch every typo rewrites names. The similarity gate alone is therefore not a
rule, it is the first of three.

The three gates
---------------
1. **Similarity** -- ``fuzz.ratio >= min_correction_score`` (default 80).
   Catches every real typo in the calibration set; rejects both the name cases
   outright.
2. **Edit distance** -- ``Levenshtein <= min_edit_distance`` (default 2).
   This is the gate that rejects ``"hemoglobinn"`` -> ``"methemoglobinemia"``
   (distance 7) despite its 71.4 similarity, and it is what makes an
   *already-correct* word safe: ``"machine"`` resolves to itself at distance 0
   and is returned unchanged, because "the term is in the vocabulary" is
   checked first and short-circuits everything.
3. **Corpus frequency** -- the candidate must appear in at least
   ``min_candidate_df`` documents. This is the proper-noun guard, and it is
   the gate that does the work similarity cannot. A rare word's nearest
   neighbour is usually a coincidence (``Serkan`` -> ``arakan``); a frequent
   word's is evidence of a typo. Measured: ``machine`` df 16, ``engine`` df 15;
   the names to protect sit at df 1-6.

Gate 0 runs before all three and is load-bearing rather than an optimization:
a term that is **already in the vocabulary** is returned unchanged. Without
it the gates are not self-refusing -- ``"the"`` maps to ``"te"`` at ratio 80.0,
distance 2, df 3, satisfying all three. The membership test is what makes
correctly-spelled words a no-op, and it is also why no length floor is needed:
of the 1,587 three-letter terms in the corpus, **zero** would be rewritten
without one, because every one of them is already in the vocabulary.

Two options that were tried and rejected
---------------------------------------
* ``fuzz.WRatio`` is unusable. Its partial-ratio component matches substrings,
  and on short words that produces garbage: ``machien -> ach``,
  ``learnign -> ear``, ``hemoglobinn -> bin``. Plain ``fuzz.ratio`` is the
  correct scorer here.
* Margin-over-runner-up does not separate the cases either. ``recieve`` has a
  margin of 0.0 and must be corrected; ``Serkan`` has a margin of 0.0 and must
  not be. The rule is not in the code because it does not work.

Proper nouns
------------
Capitalized tokens are never corrected. ``Serkan``, ``Bonn``, ``Obma`` and
scientific terms are all left alone. The cost is a real one -- a capitalized
typo like ``Machne`` at the start of a sentence is missed -- and it is paid
deliberately: guessing at a name is worse than missing a correction, because
the name is the part the user was most likely typing from memory. This also
implements the declared ``preserve_proper_nouns`` flag.

What this is not
----------------
These gates were calibrated against one 300-document corpus, and the
calibration set and its verdict are recorded in the Phase 8 section of
``BUILD_STATUS.md``. That is evidence the configuration is sensible *for this
corpus*, not evidence it generalizes. Three known boundaries, all found by
running the calibration rather than by reasoning about it:

**Stopwords cannot be corrected, in either direction.** The vocabulary is built
from ``surface_terms``, which removes stopwords, so ``"teh"`` has no
correction available. The similarity-only rule would have mapped it to
``"tech"`` (85.7, distance 1) -- a real word, one document, entirely the wrong
answer, and a demonstration that the frequency gate is doing independent work.
The converse held too, and the 300-document experiment found it: a stopword is
absent from ``terms`` not because the corpus lacks it but because the tokenizer
strips it, so gate 0 could not vouch for it, and a correct query containing one
had it replaced. ``"black holes and gravitaional fields"`` lost ``"and"`` to
``"andy"`` (85.7) and ``"how eartquakes are measured"`` lost ``"are"`` to
``"acre"`` (85.7). Both rewrites were confidently wrong, both moved documents in
and out of the top 5, and neither was a typo.

Gate 0b fixes exactly this and nothing else: a typed term that is in the
tokenizer's stopword list is refused. A stopword can never be a useful
correction *target* -- the retriever discards it -- so refusing is strictly
safe, and it closes the only hole in gate 0. It does not make stopword typos
correctable, which is the separate trade-off below.

**A typo whose correction is a stopword is therefore missed.** This is a real
missed opportunity, not a protection: the user who types ``"teh"`` gets no
help. Recovering it would mean keeping a second, stopword-inclusive frequency
table and accepting that the gate's evidence is weaker -- the gate relies on a
candidate being *frequent*, and stopwords are frequent in the wrong way. That
is a possible extension, not one taken here, and the tests assert the current
behaviour so the limitation is visible rather than accidental.

**Rare technical terms remain exposed.** A typo over a term that occurs in one
or two documents is still likely to be corrected away, because the frequency
gate pushes in the wrong direction for exactly the words where a guess is most
risky. Nothing here detects that case; it is inherent to correcting against a
frequency signal.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from semantic_search.data.loader import DocumentStore
from semantic_search.data.tokenizer import Tokenizer, get_tokenizer
from semantic_search.query.models import Correction, Vocabulary

#: rapidfuzz scores below this are not worth even considering, whatever the
#: configured threshold. Checking 0.5 of a 40k vocabulary to find that a token
#: matches nothing is wasted work, and this constant is comfortably below any
#: sane ``min_correction_score`` -- if a user configures a threshold below 40
#: they have asked for something this corrector should not do anyway.
_MIN_USEFUL_SCORE = 40.0


def build_vocabulary(store: DocumentStore, tokenizer: Tokenizer | None = None) -> Vocabulary:
    """Build the correction vocabulary from a document store.

    Uses ``Tokenizer.surface_terms`` -- casefolded, stopwords removed, **not
    stemmed**. The no-stemming part is load-bearing: ``tokenize`` maps
    ``"learning"`` to ``"learn"`` and ``"machine"`` to ``"machin"``, so a
    stemmed vocabulary compared against unstemmed input would find that no
    correctly-spelled query term is in the vocabulary at all.

    Document frequency is counted per document, not per occurrence, so one
    article repeating a word fifty times does not make it look common.
    """
    analyzer = tokenizer or get_tokenizer(use_stemming=False)
    frequency: dict[str, int] = {}
    for document in store:
        # set() because a term repeated within one document is one document.
        for term in set(analyzer.surface_terms(document.searchable_text)):
            frequency[term] = frequency.get(term, 0) + 1
    return Vocabulary(
        terms=frozenset(frequency),
        doc_frequency=frequency,
        stopwords=frozenset(analyzer.stopwords),
    )


class SpellCorrector:
    """Rewrites query terms towards the corpus vocabulary, or declines to.

    The corrector is stateless with respect to queries: it holds only the
    vocabulary and the thresholds, so the same query always produces the same
    corrections. Nothing is cached across calls and no decision depends on
    call order.

    Construction cost is building the vocabulary, which is O(corpus). Build it
    once at application startup and share the corrector -- that is what
    ``SearchService`` does.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        enabled: bool = True,
        preserve_proper_nouns: bool = True,
        protect_stopwords: bool = True,
        min_correction_score: float = 80.0,
        max_edit_distance: int = 2,
        min_candidate_df: int = 3,
        max_candidates: int = 5,
    ) -> None:
        self.vocabulary = vocabulary
        self.enabled = enabled
        self.preserve_proper_nouns = preserve_proper_nouns
        self.protect_stopwords = protect_stopwords
        self.min_correction_score = min_correction_score
        self.max_edit_distance = max_edit_distance
        self.min_candidate_df = min_candidate_df
        self.max_candidates = max_candidates
        # Sorted once so that the rapidfuzz search order -- and therefore the
        # tie-break when two candidates score identically -- is identical on
        # every process. Iterating a frozenset would not be, and a correction
        # that differed between two runs of the same query would be a bug that
        # only reproduces sometimes.
        self._terms: tuple[str, ...] = tuple(sorted(vocabulary.terms))

    def correct(self, query: str) -> tuple[str, tuple[Correction, ...]]:
        """Return ``(corrected_query, corrections)``.

        Terms are corrected independently and positionally, so a query with
        two typos produces two corrections and both are reported. The original
        spacing is reconstructed from the corrected terms, which means the
        returned string is normalized even when nothing was corrected -- that
        is the normalizer's doing, not a correction, and the corrections tuple
        stays empty.
        """
        if not self.enabled or not self._terms:
            return query, ()

        words = query.split(" ")
        corrected_words: list[str] = []
        corrections: list[Correction] = []

        for word in words:
            replacement = self._correct_word(word)
            if replacement is None:
                corrected_words.append(word)
                continue
            candidate, score = replacement
            # Belt and braces. `_correct_word` already refuses identity
            # rewrites, but `corrections` is the record a user is shown, and a
            # "correction" that changed nothing is a false statement about their
            # query. Asserted here as well as in the corrector so the
            # guarantee does not depend on a caller's internals.
            if candidate == word or candidate.casefold() == word.casefold():
                corrected_words.append(word)
                continue
            corrected_words.append(candidate)
            corrections.append(Correction(from_term=word, to_term=candidate, confidence=score))

        return " ".join(corrected_words), tuple(corrections)

    def _correct_word(self, word: str) -> tuple[str, float] | None:
        """Decide whether one token should be rewritten. ``None`` means leave it."""
        # Punctuation is stripped for *matching* only, and reattached on the way
        # out, so "learning," still corrects and still returns "learning,".
        core = word.strip(".,;:!?\"'()[]")
        if not core:
            return None

        # Gate 0, and the most important one: a term that is already in the
        # vocabulary is correct. This short-circuits before any fuzzy search,
        # which is what makes an already-correct query cheap and guarantees a
        # known word is never rewritten.
        folded = core.casefold()
        if folded in self.vocabulary:
            return None

        # Gate 0b, for the one category the corpus cannot vouch for. See the
        # "Stopwords" section of this module's docstring: without this,
        # `"black holes and gravitaional fields"` had its "and" rewritten to
        # "andy" at similarity 85.7, and `"how eartquakes are measured"` had
        # its "are" rewritten to "acre". Both are correctly spelled English,
        # both were mangled, and both changes moved documents in and out of the
        # top 5. A stopword can never be a useful correction target -- the
        # retriever discards it anyway -- so refusing is strictly safe.
        if self.protect_stopwords and folded in self.vocabulary.stopwords:
            return None

        # The proper-noun rule. Note it is a *capitalization* test, not a
        # dictionary lookup: "Serkan" is protected because it is capitalized,
        # not because we know it is a name. That is deliberate -- a name
        # database is exactly the kind of growing external resource this layer
        # is meant not to need.
        if self.preserve_proper_nouns and core[0].isupper():
            return None

        # Gate 1, 2 and 3 applied to the ranked candidates. rapidfuzz returns
        # the best match per query by default, so the candidate list is
        # re-ranked below rather than taken in order.
        #
        # `core.casefold()` rather than `core`: fuzz.ratio is case-SENSITIVE,
        # and a capital letter costs roughly 15 similarity points. Scoring
        # "Machne" against "machine" gives 76.9, below the 80 floor, so a
        # capitalized typo would never be corrected no matter how permissive
        # the thresholds were. The vocabulary is already casefolded, so
        # casefolding the query first compares like with like.
        ranked = process.extract(
            folded,
            self._terms,
            scorer=fuzz.ratio,
            limit=self.max_candidates,
            score_cutoff=_MIN_USEFUL_SCORE,
        )
        if not ranked:
            return None

        best_term, best_score, *_ = ranked[0]
        if best_score < self.min_correction_score:
            return None
        if Levenshtein.distance(folded, best_term) > self.max_edit_distance:
            return None
        if self.vocabulary.frequency(best_term) < self.min_candidate_df:
            return None

        # Reattach the punctuation the strip removed, so the output is the same
        # shape of token the user typed.
        return best_term + _detached_affix(word, core), best_score


def _detached_affix(word: str, core: str) -> str:
    """The punctuation around ``core`` in ``word``, preserved for reassembly.

    Correcting ``"learning,"`` should give ``"learning,"`` and not
    ``"learning"``. Losing a comma silently changes how the next query is
    tokenized, which is a real retrieval effect for a cosmetic-looking bug.
    """
    start = word.index(core)
    return word[:start] + word[start + len(core) :]
