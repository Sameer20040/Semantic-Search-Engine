# Evaluation judgments (qrels) — construction record

Format: one JSON object per line, TREC qrels fields plus provenance.

```json
{"query_id": "A01", "doc_id": "3437", "relevance_grade": 1, "judged_by": "author-single-judge", "protocol": "known_item"}
```

**Grading is binary.** Every judgment in this file has
`relevance_grade: 1`; there are no grade-0 rows and no rows above 1. The metrics
implement graded NDCG because the formula is standard and the tests exercise it,
but no judgment here uses a scale, and the report will not describe NDCG as
measuring graded relevance — it is the binary case of a general formula.

**`judged_by` is one person, on every row.** There is no second judge and no
inter-annotator agreement figure anywhere in this repository. Every use of
"relevant" in the report means "judged relevant by the author of the code being
evaluated". A retrieval evaluation with one judge and no agreement statistic
carries that judge's bias directly into every number, and the only honest
mitigation is to say so.

**`protocol` is which rule produced the judgment**, and it is not the same rule
for the two kinds of query. `known_item` means the gold document is the article
the question was written from. `pooled_judged` means the document was found in a
pooled candidate list and judged against the question. Reading the two as
interchangeable would overstate what the pooled set measures.

## Stratum A — 50 judgments, written before any retrieval

```
A01-A25  A_restated   25 judgments, one per query: the source article
V01-V15  A_verbatim   15 judgments, one per query: the source article
T01-T10  A_typo       10 judgments, one per query: the same article as its parent
```

Written by `data/evaluation/_build_qrels_a.py`, which reads the query rows'
`source_doc_id` and emits a judgment for it. There is no judgment step to
speak of: the ground truth is a property of how the query was constructed, and
the file records that fact rather than a decision.

**What this means for the metrics.** With one relevant document per query, AP
is `1/rank` and NDCG@k is `1/log₂(rank+1)`. Both are strictly monotone
transformations of the gold document's rank, so on these 50 queries MAP and
NDCG carry no information that MRR does not already carry. They are reported
for stratum B only, and stating the reason here prevents the report from
implying that it measured something extra by printing a second column.

## Stratum B — judgments still to be written

The 30 `B_topical` queries are frozen (see
[`../queries/MANIFEST.md`](../queries/MANIFEST.md)) but have no judgments yet.
They are judged **after** the first retrieval run, by this protocol:

1. **Pool.** Union of all six systems' top-50 for that query, de-duplicated and
   kept in rank order. The systems that will be scored are the systems that
   build the pool — this is the standard pooling assumption, and its cost is
   recorded below rather than hidden.
2. **Judged blind to system.** The pool is presented in a fixed shuffled order
   with no system name attached. A judge who knows which system retrieved a
   document will judge it more generously; blinding is cheap and removes the
   channel entirely.
3. **Judged by answer, not topic.** A document is relevant if it *contains or
   states the answer* recorded in the query's `notes` field. Topical
   relatedness is not relevance. A country article that discusses elections but
   not this question's fact is not relevant, and marking it relevant is the
   single easiest way to make a corpus look better than it is.
4. **Frozen before the reported run.** The qrels file is written and hashed
   before the run whose numbers are reported. The pool-building run is a
   separate, disclosed run, and the report says which one it was.
5. **Source article not assumed to be in the pool.** If the article a question
   was written about does not appear in any system's top-50, that is recorded
   as a coverage failure, **not** fixed by adding it to the pool by hand. A
   hand-added gold document would make the query unanswerable for every system
   by construction, and it would hide exactly the finding that matters.

### Known limitations of this protocol, stated before the numbers exist

- **Pooling at depth 50 is incomplete.** A relevant document that no system
  retrieved at rank ≤ 50 is invisible to the judge, and the metrics will treat
  it as non-relevant. This depresses every system's recall by an unknown
  amount. `evaluation.dataset.pool_report` counts the unjudged and never-pooled
  fractions per query and the report prints them, because a recall number with
  an unstated pooling depth is not interpretable.
- **One judge.** Same limitation as stratum A, and here it is worse: pooled
  judging is more subjective than "the article the question came from", because
  "does this document answer the question" is a judgement rather than a fact.
- **Pooling bias.** The pool comes from the systems being evaluated, so a
  document that all six systems systematically miss is systematically absent
  from the judgments. The only defence is the honest report of the unjudged
  fraction, which is why it is per-query and not a footnote.
- **Answers recorded in `notes` are the author's paraphrase** of what the lead
  says, not a quote. Where the paraphrased phrase does not literally appear in
  the document, the judge reads the document rather than string-matching it.

## Cross-checks enforced by `evaluation.dataset`

Loading this file raises rather than warns when:

- a judgment names a `query_id` that is not in `queries.jsonl` (a query was
  deleted, or a judgment outlived its question);
- a judgment names a `doc_id` that is not in the corpus (the qrels file and
  `corpus.jsonl` are out of sync);
- the same `(query_id, doc_id)` pair appears twice with different grades;
- a query in the file has no positive judgment, which would score it 0.0 on
  every metric and read as a system failure rather than a dataset gap.
