# Evaluation queries — construction record

`sha256(queries.jsonl) = 0480ffe5830a5a1e1bd8d251086d5c60310301c8bcdab6269372b5a243d4f25a`

Verify with:

```
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('data/evaluation/queries/queries.jsonl').read_bytes()).hexdigest())"
```

**80 queries. Both files in this directory were written and hashed before any
retrieval run.** Nothing below was revised after a system's output was seen.

| Stratum | n | Ids | Construction |
|---|---|---|---|
| `A_restated` | 25 | `A01`–`A25` | known-item, natural question, gold = source article |
| `A_verbatim` | 15 | `V01`–`V15` | known-item, query = article title verbatim |
| `A_typo` | 10 | `T01`–`T10` | frozen misspelling of an `A_restated` query |
| `B_topical` | 30 | `B01`–`B30` | hand-written question, gold judged from a pooled candidate list |

## The four strata are never pooled into one number

A single mean over 25 easy title-matching queries and 30 hand-judged topical
ones describes neither population. The report presents them separately and
computes no cross-stratum average. This is a decision, not an omission.

## Stratum A: known-item

### A_restated (25)

The source articles were drawn mechanically:

```python
random.Random(20250930).sample(list(corpus), 50)
```

then 25 got a hand-written question and 15 of those got the title-verbatim
treatment, so the two A strata share source articles and the difference between
them is the query form and nothing else.

**Why the draw is random and not chosen.** Picking articles by hand means
picking the ones I could write an interesting question about, which correlates
with the ones the retrievers find easy. The seed is in the code so the draw is
reproducible; the questions themselves are recorded in
`../_build_tier_a.py`, so re-running it must produce this exact file.

A query is written from the article's lead, in a natural question form, without
the title. `source_doc_id` is the gold document and is recorded in the dataset
— the ground truth for this stratum *is* the article the question was written
about. That is a known-item protocol (`known_item` in the qrels file) and it is
the standard protocol for this kind of dataset. Its limitation is stated below
under "What this dataset cannot tell you".

### A_verbatim (15)

Query = `Document.title`, character for character, including the
disambiguator where one survives preprocessing (`V07` is
`Andersonville, Georgia`, not `Andersonville`, because both are in the corpus
and the disambiguator is the only thing that distinguishes them).

**Queries are never built from `title_raw`.** 18 of the 300 articles have
`title_raw != title` after preprocessing, and only `title` is indexed — a
`title_raw` query would be unanswerable by construction. This is enforced in
`../_build_tier_a.py`.

### A_typo (10)

Each row is a frozen misspelling of an `A_restated` query, `derivation: "typo:Axx"`
records the parent, and `notes` records the mechanism (transposition, or a
dropped letter).

**Lowercase content words only.** `SpellCorrector._correct_word` refuses to
rewrite a capitalised term when `preserve_proper_nouns` is on, so a typo in
`Alexandria` could never be repaired and the row would measure nothing.

**`T02` is deliberately unmisspelled.** Not every word is typosable, and
asserting a uniform 10/10 typo rate would be a false claim about the stratum.
`T02` is a superset of `A02` — the same query, spelled correctly — so the
stratum contains one query on which the spell arm is known a priori to be a
no-op. That is a feature of the design: the stratum that can move the spell arm
is measured against one that provably cannot.

## Stratum B: hand-written topical (30)

Ten articles, drawn mechanically, three questions each:

```python
random.Random(20251001).sample(
    [d for d in corpus if d.id not in stratum_A_source_ids], 10
)
```

The questions are in `../_build_queries_b.py`, one `SPEC` entry per sampled
article, three `(question, answer)` pairs each. The script re-derives the sample
from the seed and **refuses to run** if the drawn articles and the written
questions have drifted apart, so the questions can never be quietly reassigned
to different articles.

Each question was written so that the source article is a good answer, and
each row's `notes` records the answer as found in that article. Those notes are
used by the judging pass to check the pooled document actually contains the
answer, and are **not** used to build the judgments.

**Deliberately avoided in every stratum-B question:** the article's title words,
its disambiguator, and the phrasing of its first sentence. Where a country name
was the only natural way to ask, the question was rephrased around the
institution instead — `B01` asks "how many seats does the upper chamber have
and who appoints them", not "what can Barbados do with a money bill". This is
what makes stratum B harder than stratum A by construction rather than by
accident.

**Stratum B's known limitation.** This corpus is 300 loosely-related Wikipedia
articles, so a "topical" query is often answerable by exactly one article in it.
That is why the *pooled* judging protocol exists rather than assuming the
source article: the pool is built from the systems' output and the source
article can be found not to appear in it, which is a real finding about
pooling rather than a bug to be patched by relabeling.

## Leakage rules, and what enforces each

| Rule | Enforced by |
|---|---|
| Queries fixed before any retrieval | both builders hash the file; the hash above is checked by `evaluation.dataset` |
| No query revised after seeing a ranking | builders have no ranking input; nothing in the workflow runs a retriever |
| No relevance label derived from a retriever's ranking | stratum-A labels are `source_doc_id`, fixed at write time; stratum-B labels come from a pooled list the judge has to argue for |
| No `title_raw` queries | `_build_tier_a.py` asserts every verbatim query equals `Document.title` |
| Source articles disjoint between strata | `_build_queries_b.py` excludes stratum-A `source_doc_id`s before sampling |
| No prior-art queries | the worked examples named in `BUILD_STATUS.md` and in the Phase 8/9 reports are excluded — see below |

### Excluded prior-art queries

The retrieval examples used in earlier phases were checked against this set and
**are not present**:

`how a muscle pump moves blood` · `reflectivity of the moon and snow` ·
`what is the capital of France` · `Python list comprehension performance` ·
`tell me about the Renaissance` · `photosynthesis`

Also absent: any query consisting of a common English word that merely happens
to be an article title (`Basis`, `Alternative`, `Bar`, `Bit`, `Bead`, `Aga`,
`Arno`, `Beavis and ButtHead`). These exist in the corpus and are deliberately
*not* used as verbatim queries, because a title query for `Basis` measures
title matching and nothing about retrieval.

## No train/test split — a deliberate refusal

There is no held-out split, and this is a decision rather than an oversight.

Nothing in this phase fits a parameter. No threshold, weight, or fusion constant
is tuned against these queries, so there is no training set to hold out. At
n=80 a split would leave 40 queries per side, roughly halving the resolution of
every interval in the report while reading as methodological rigor it does not
deliver. Splitting would be the more impressive-looking choice and the less
honest one.

The corresponding risk is real and is named rather than solved: if these
queries are later used to choose a configuration, they stop being a held-out
set at that moment, and the report produced afterwards is a development-set
number. `data/evaluation/derived/` is regenerated on every run and is not
committed; the queries and judgments are.

## What this dataset cannot tell you

Stated here rather than only in the final report, because the dataset is the
thing being read:

1. **The author wrote the queries and is the author of the retrievers.** Any
   question whose answer I failed to anticipate is missing. There is no
   second author and no inter-annotator agreement.
2. **The corpus is 300 Wikipedia articles.** Not user traffic, not a domain
   corpus, not multilingual. It is a size setting.
3. **Stratum A is not task-independent** — see below. It is a ceiling.
4. **80 queries is small.** The bootstrap intervals in the report quantify
   query-sampling noise at n=80 and nothing else. Corpus choice, query-author
   bias, and single-judge bias are all larger and none of them is quantified.
5. **Every judgment is one person's.** "Relevant" means "I judged it relevant".

## Stratum A is a ceiling, and the contamination is measurable

Three facts, all verified against the code and the corpus rather than assumed:

- `BM25Index.build` appends `tokenize(document.title) * title_boost`, with
  `title_boost=1` by default, and `scripts/build_index.py` does not override
  it — **the shipped index counts every title term twice**.
- `build_embedding_text` returns `f"{title}. {text}"`, so the title is
  literally the first chunk of every embedded document.
- 22 of 300 leads restate their title.

So a title-bearing query is close to the easiest possible input for both
retrievers, and 15 of the 80 queries are exactly a title. Stratum A is reported
as a ceiling on this corpus, with a `title_boost=0` ablation attached, and
never as "retrieval quality".
