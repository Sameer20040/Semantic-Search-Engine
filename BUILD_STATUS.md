# Build Status

Progress log for the semantic search engine. Updated after each phase.
"Tests" reports what was **actually executed**, not what is expected to work.

## Phase status

| Phase | Description | Status | Tests | Notes |
|---|---|---|---|---|
| 0 | Environment survey | ✅ complete | n/a | See `docs/benchmarks.md` for the machine profile. |
| 1 | Repository + configuration | ✅ complete | manual | YAML + env layering verified; dev/prod configs. |
| 2 | Data pipeline | ✅ complete | 156 passed | Real 300-doc Wikipedia corpus downloaded, cleaned, validated. |
| 3 | BM25 | ✅ complete | 31 passed | Score verified against hand-computed formula to 1e-9. |
| 4 | Sentence-BERT | ✅ complete | 60 passed | Real all-MiniLM-L6-v2 verified; cache is resumable + gated. |
| 5 | FAISS | ✅ complete | 53 passed | Exact `flat_ip` default; persistence + provenance refusals verified. |
| 6 | Semantic search | ✅ complete | 51 passed | `SemanticSearchService`: encode → search → resolve → rank. |
| 7 | Hybrid RRF | ✅ complete | 115 passed | `HybridSearchService`: BM25 + semantic → RRF, with per-retriever provenance. |
| 8 | Query processing | ✅ complete | 546 passed | `QueryProcessor` + `SearchService` above hybrid retrieval. Spell correction measured and ships **off**; quality **NOT YET MEASURED**. |
| 9a | Application search service | ✅ complete | 638 passed | `ApplicationSearchService` above Phase 8. Stable request/response contract, no re-ranking. Quality **NOT YET MEASURED**. |
| 9b | Explainability | ⬜ pending | | The roadmap's original "Phase 9". Renumbered to 9b to make room for the application layer, which the spec assigned to 9. See the note below. |
| 10 | Evaluation | ⬜ pending | | |
| 11 | FastAPI | ⬜ pending | | |
| 12 | Streamlit | ⬜ pending | | |
| 13 | Testing | ⬜ pending | | |
| 14 | Benchmarks | ⬜ pending | | |
| 15 | Docker | ⬜ pending | | |
| 16 | CI/CD | ⬜ pending | | |
| 17 | Documentation | ⬜ pending | | |
| 18 | Deployment prep | ⬜ pending | | |
| 19 | Final validation | ⬜ pending | | |

## Environment

| Property | Value |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| Python | 3.11.9 (64-bit) |
| CPU | 8 logical cores |
| RAM | 8.2 GB |
| GPU | None (`torch.cuda.is_available() == False`) |
| Disk free | ~164 GB |
| Key libs | torch 2.12.1+cpu, faiss-cpu 1.15.1, sentence-transformers 5.6.0, fastapi 0.139.2, streamlit 1.58.0 |
| Model cache | `sentence-transformers/all-MiniLM-L6-v2` already in the HF cache |

## Issues found and fixed

1. **faiss-cpu DLL blocked by Windows Application Control policy.** The
   pre-installed `faiss` 1.14.3 failed to import with
   `ImportError: DLL load failed while importing _swigfaiss`. A
   `pip install --force-reinstall --no-cache-dir faiss-cpu` resolved it
   (now 1.15.1). Recorded in `docs/troubleshooting.md`.
2. **Config layering bug.** `configs/default.yaml` was merged *after* the
   environment file, so environment-specific values were silently overridden.
   Fixed by merging default first, environment file second.
3. **Nested config not environment-configurable.** `pydantic-settings` only
   binds flat fields, so `EMBEDDING_MODEL` could not set `embedding.model_name`.
   Added `_apply_nested_env`, which derives candidate env names from the model
   fields (no hard-coded list) and supports explicit aliases.
4. **Stemmer over-stemming.** Early versions produced `series -> sery`,
   `business -> busy`, `happiness -> happi`, `crises -> crise`. Root causes were
   (a) derivational rules running *after* the plural strip, (b) an over-eager
   `i` -> `y` restore, (c) an off-by-one `word[-3]` that inspected the wrong
   character. Fixed and locked down by 62 tokenizer tests covering both
   must-reduce and must-not-change tables.

## Verified measurements

Taken from actual runs on this machine, not targets.

| Measurement | Value | How obtained |
|---|---|---|
| Tests | 351 passed (42 s) | `pytest tests/` |
| Unit tests | 343 | `pytest tests/unit --collect-only` |
| Integration tests (real model) | 8 | `pytest tests/integration -m slow` |
| Corpus documents | 300 | `scripts/download_data.py --limit 300` |
| Corpus characters | 3,539,878 | corpus validation stats |
| Vocabulary size | 73,774 | corpus validation stats |
| Mean document length | 11,799.6 chars | corpus validation stats |
| Median document length | 11,664 chars | corpus validation stats |
| Validation issues | 0 errors, 0 warnings | corpus validation report |
| BM25 build time (300 docs) | 790 ms | timed run |
| BM25 query latency (300 docs) | 0.2-0.6 ms | timed run over 5 queries |
| BM25 index vocabulary | 36,580 | index stats |
| Embedding dimension | 384 | `all-MiniLM-L6-v2` |
| Embedding device | cpu (no CUDA present) | `detect_device("auto")` |
| Semantic similarity, related pair | 0.539 | cosine, two lunar sentences |
| Semantic similarity, unrelated pair | 0.016 | cosine, lunar vs. QCD |
| Paraphrase similarity, no shared words | > 0.5 | cosine, integration test |
| Encoding throughput | 19.9 docs/s | 300 docs, CPU, batch 64 |
| Embedding cache size | 450 KiB | `embeddings.npy`, 300 x 384 float32 |
| FAISS build time (300 docs, flat_ip) | 98 ms (first) / 71 ms (warm) | `scripts/build_index.py` |
| FAISS index file size | 450 KiB | `data/index/faiss_index` |
| FAISS index load time | 0.3 ms warm / 62.5 ms first (imports faiss) | 3 runs |
| Semantic search, top_k=1 | p50 0.020 / p95 0.021 / p99 0.029 ms | 2000 runs, in-process |
| Semantic search, top_k=10 | p50 0.027 / p95 0.032 / p99 0.059 ms | 2000 runs, in-process |
| Semantic search, top_k=100 | p50 0.107 / p95 0.194 / p99 0.216 ms | 2000 runs, in-process |
| HNSW build / top_k=10 query | 23.6 ms / p50 0.064 ms | 2000 runs; ~2x slower than exact at n=300 |
| Cache-hit rebuild | ~1.0 s total, no model load | second `build_index.py` run |
| mypy | clean, 25 files | `mypy src` |
| ruff | clean | `ruff check src tests scripts` |

### Phase 6: semantic service, end-to-end

Measured with `scripts/measure_semantic.py`: 15 fixed queries x 20 passes
(300 samples, warm), CPU, `top_k=10`. **Read the variance warning before
comparing these to anything.**

| Stage | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|
| Query encoding | 17.4 ms | 25.4 ms | 28.2 ms | 33.7 ms | 17.2 ms |
| FAISS search | 0.19 ms | 0.32 ms | 0.37 ms | 0.41 ms | 0.20 ms |
| Document retrieval | 0.10 ms | 0.17 ms | 0.20 ms | 0.22 ms | 0.10 ms |
| **End-to-end (`service.search`)** | **18.3 ms** | **25.8 ms** | **28.4 ms** | **40.6 ms** | **18.1 ms** |

By `top_k`, same harness, separate runs:

| `top_k` | FAISS p50 | retrieval p50 | end-to-end p50 |
|---|---|---|---|
| 1 | 0.13 ms | 0.03 ms | 15.3 ms |
| 10 | 0.19 ms | 0.10 ms | 18.3 ms |
| 100 | 0.33 ms | 0.46 ms | 15.7 ms |

| Other measurement | Value | How obtained |
|---|---|---|
| Service startup (store + index + model) | 28-31 s | dominated by the cold SentenceTransformer load |
| Query encoding share of latency | ~97% | stages measured separately, not apportioned |
| Semantic stage vs. raw FAISS (Phase 5) | 0.19 ms vs 0.027 ms | Phase 5 measured raw in-process; the service adds the per-call wrapper |

**Variance warning.** Four identical runs gave an encoding p50 of 10.15, 20.12,
17.41, and 14.54 ms -- a 38% spread. This is a shared 8-core machine with no
GPU, and CPU-bound torch inference on it is not stable enough to quote to better
than roughly "10-20 ms". The table above is one representative run, not a
best-of. Any figure in this project claiming single-millisecond query encoding
would be measuring a quiet machine, not a fast one.

**Honest characterisation:** 18 ms p50 on this corpus is a real measurement,
but it is dominated by a single-document CPU forward pass and says nothing
about behaviour under concurrency, which is `NOT YET MEASURED` until Phase 14.

Not yet measured: hybrid fusion quality, query processing, API latency,
concurrent throughput, recall against relevance labels. Those are
`NOT YET MEASURED` until Phases 7-8, 10, 11, and 14.

## Issues found and fixed (continued)

5. **`flat_l2` sort key was incoherent.** The first draft contained
   `lambda hit: (hit.row if hit.score == -1 else hit.score, hit.row)`, which
   sorts L2 hits by *row* whenever the score happens to be exactly -1. No test
   would have caught it, because the branch is nearly unreachable. Removed:
   FAISS already returns every metric type best-first, so no re-sort is needed
   at all. The only real difference between types is score *direction*, which
   the search service handles before fusion.
6. **FAISS `-1` padding could punch holes in the rank sequence.** Ranks were
   taken from `enumerate`, so a padded slot would skip a number and RRF would
   mis-score that result. Now taken from `len(hits)` after filtering, so ranks
   are always contiguous. Covered by `test_ranks_are_contiguous_from_zero`.
7. **`get_mem_usage_kb()` would misreport index size.** It returns
   *process-wide* FAISS allocation, which would attribute the encoder's model
   weights to the index. Replaced with `ntotal * d * 4`, exact for flat and
   HNSW, documented as a floor for IVF.
8. **`build_index.py` reported a stale encode duration on a cache hit.** The
   second run printed `encode 15.1 s` even though nothing was encoded -- it was
   replaying the first run's metadata. Now reports `cached` and `0.0 s`.

## Phase 6: issues found and fixed

9. **FAISS raised one error for three distinct conditions.** A missing file, a
   model mismatch, and a document-count mismatch all raised `SearchIndexError`,
   so the API could not tell "you never built an index" (503, build it) from
   "the index you built is wrong" (503, rebuild it) from genuine corruption.
   Now `IndexNotFoundError` and `IndexCompatibilityError` respectively, both
   still subclasses of `SearchIndexError`, so existing `except` clauses keep
   working. Covered by `TestArtifactLoading`.
10. **Five of my own Phase 6 tests were wrong before the code was.** Worth
   recording because in each case the *test* was the bug:
   - Asserted `FaissHit(row=-1)` would be constructible. It is not, by design --
    `__post_init__` rejects negative rows precisely to stop `store[-1]`.
    Rewritten to simulate padding with a `PaddedHit` stub, which is what a
    future index type forgetting to filter would actually return.
   - Asserted a bare document body would reproduce the document's own vector.
    Documents are embedded via `build_embedding_text`, which prefixes the
    title, so a self-match must query the *titled* text. The old assertion
    failed at 0.459 vs 1.0 for exactly this reason.
   - Expected `top_k=10` to yield 10 results from a 5-document fixture corpus.
    Requesting more than exists is answered with everything, not padded.
   - Expected `flat_l2` to negate a distance into a cosine of 1.0. Squared L2
    distance between identical unit vectors is 0, so the correct negated score
    is 0.0; the test was checking a cosine formula against a distance index.
   - `FakeModel` emitted un-normalised vectors, so the inner-product scores in
    `TestScoreDirection` were dot products, not cosines. Normalised the fake.
11. **`grep -v faiss` silently deleted a benchmark row.** The first
    `measure_semantic.py` run's `faiss search` line was removed by the shell
    filter intended to strip FAISS's AVX-loading chatter, because the row
    contains the word "faiss". The table was then missing its most important
    line. Fixed the filter; noted because a benchmark whose output is post-
    processed by `grep` is one filter change away from a wrong number.

## Phase 6: remaining limitations

These are properties of the current build, not bugs. Each is stated with what
it would take to change.

1. **Query encoding dominates latency and is not batched.** ~97% of the
   end-to-end time is one CPU forward pass, because a single query is one
   sentence. A `/batch` endpoint amortising the forward pass over many queries
   is the obvious fix and is not implemented.
2. **The encoder is not warmed at startup.** The first query after a restart
   pays the lazy SentenceTransformer load inside the request path (~30 s
   observed at startup, but the very first call also triggers model work).
   A readiness probe that runs one throwaway query would move this off the
   user-visible path. The service validates its collaborators at construction,
   but does not call `encoder.dimension` unless the model is already loaded --
   deliberately, so that constructing a service never triggers a download.
3. **No concurrent-load measurement.** Single-threaded latency is measured;
   throughput and tail latency under concurrent clients are `NOT YET MEASURED`.
4. **Vectors are memory-mapped only in the cache, not the index.** A FAISS index
   for 1M documents would not fit alongside the store in 8 GB; a large-scale
   deployment needs memory-mapped reads and a narrower index type.
5. **No query-understanding yet.** No spell correction, synonym expansion, or
   metadata filtering (Phase 8), so every query is embedded verbatim.

## Findings about the corpus itself

Established by direct inspection while validating Phase 6, and relevant to how
any later result is read.

- **The 300-document corpus does not cover general knowledge.** It contains no
  article about vaccines, the immune system, anatomy, the heart, or the French
  Revolution (verified by title scan). Real semantic queries about those topics
  return the least-bad of 300 unrelated articles, with top scores of
  0.19-0.26.
- **The corpus is 254/300 documents longer than 3 000 characters** (median
  11 757). `all-MiniLM-L6-v2` truncates at 256 word-pieces, so each vector
  summarises roughly the first two paragraphs of an article. Measured: the
  cosine between a document's full vector and a title-only vector is 0.53-0.75,
  i.e. the title plus lead dominates the representation. This is a reasonable
  default for Wikipedia, which leads with a definitional paragraph, but it caps
  achievable recall for questions about later article sections.
- **Consequence for evaluation (Phase 10).** Recall@k, MRR, and NDCG on this
  corpus measure the retriever against a narrow slice of Wikipedia, not against
  general search quality. Any such number must be reported with the corpus
  description attached, or it will be read as a claim it does not support.

## Correction: an earlier claim was wrong

I previously described the query *"reflectivity of the moon and snow"* as
demonstrating "BM25's known weakness... which is exactly what the semantic half
will fix", and proposed it as the proof that Phase 5 solved a Phase 3 problem.

**That was not a real finding.** The 300-document corpus contains no article
whose title mentions the moon, lunar, reflect, or albedo. Neither retriever
could have answered that query, so the result demonstrated nothing about BM25.
Semantic search also failed to return anything relevant to it.

The claim I *can* support is the weaker, real one: the two retrievers are
genuinely complementary on queries the corpus can answer. Verified on
`"how a muscle pump moves blood"` -- BM25 ranks the muscle documents first on
matched terms, while semantic search does not surface them at all; the
embedding of that query is dominated by a taxonomy sense of "pump" (Branchiopoda
at 0.225) rather than the physiological one. The two error modes are
different, which is the actual argument for RRF fusion.

## Phase 7: hybrid retrieval with Reciprocal Rank Fusion

### Implementation

**Files created**

| File | Purpose |
|---|---|
| `src/semantic_search/search/rrf.py` | The fusion arithmetic. Depends on nothing but ranked `doc_id` strings — no corpus, no model, no FAISS. |
| `src/semantic_search/search/hybrid_models.py` | `RetrieverHit`, `HybridSearchResult` — the provenance-bearing result type. |
| `src/semantic_search/search/hybrid.py` | `HybridSearchService`: orchestration only. |
| `scripts/probe_hybrid.py` | BM25 / SEMANTIC / HYBRID-RRF side by side, with provenance per result. |
| `scripts/measure_hybrid.py` | Stage-separated latency, including the fusion stage on its own. |
| `scripts/hybrid_experiment.py` | Fixed query set on the real 300-document corpus. |
| `tests/unit/test_rrf.py` | 61 tests. The arithmetic, by hand. |
| `tests/unit/test_hybrid_service.py` | 54 tests. Real BM25 + real semantic service over 8 documents. |

**Files modified**

| File | Change |
|---|---|
| `src/semantic_search/search/semantic.py` | `_validate_query` → `validate_query`, `_validate_top_k` → `validate_top_k`. Both now public so hybrid reuses them. **No behaviour change.** |
| `src/semantic_search/data/loader.py` | Added `DocumentStore.row_of()`. Fusion arrives holding `doc_id` strings and needs the store row. |
| `src/semantic_search/search/__init__.py` | Exports the new names. |
| `tests/unit/test_semantic_service.py` | Fake model switched from `hash()` to `blake2b` (see bugs, below). |

`BM25Searcher`, `BM25Index`, and `FaissIndex` are **untouched**. The only edit to
`SemanticSearchService` is the two visibility changes above.

### RRF design

```
RRF(d) = Σ  1 / (k + rank_i(d))        rank 1-based, k = 60 by default
```

- **1-based ranks.** `SearchResult.rank` and `BM25Result.rank` are both 0-based
  project-wide. `fuse()` takes 0-based ranks and performs the `+1` in exactly
  one place, so a top hit is `1/61` and not `1/60`. This is the single most likely
  site of a silent off-by-one, so it is unit-tested against hand arithmetic.
- **`k = 60`**, from `DEFAULT_RRF_K`, overridable per-config as `hybrid.rrf_k`.
- **A retriever that did not return a document contributes nothing** — that is
  an absence, not a zero, and the two are not the same.
- **Duplicate `doc_id` within one retriever is rejected**, not deduplicated.
  Silently keeping the first would let a buggy retriever inflate a fused score;
  keeping the last would make the outcome depend on iteration order.
- **Raw scores are never inputs to the arithmetic.** BM25 scores are unbounded
  and not comparable across queries; cosines are bounded but not comparable
  across retrievers. `0.5*bm25 + 0.5*cosine` is a number with no stable
  interpretation. RRF consumes only the ordinal information each retriever is
  confident about. The cost is that score *margins* are discarded.

### Candidate depth

Each retriever is asked for `candidate_k = max(top_k, hybrid.candidate_pool)`,
default 100. Asking for exactly `top_k` and fusing those is wrong, and silently
so: a document ranked 11th by *both* retrievers is a stronger fusion signal
than one ranked 1st by one retriever and unseen by the other, and with
`candidate_k == top_k` that consensus is structurally unavailable — the two
lists can only ever overlap on the same 10 documents. The cost is bounded
because BM25 already scores the whole corpus and FAISS already searches; a
deeper pool changes only how many results are *materialised*.

This is configurable, not a magic number, and the default is stated in
`hybrid_models.py`, `hybrid.py`, and the `/stats` payload.

### Result and provenance design

`HybridSearchResult` **wraps** the existing `SearchResult` rather than
replacing it, so the semantic and BM25 contracts are unchanged and every
existing consumer keeps working. On top of the pass-through (`doc_id`, `title`,
`snippet`, `rank`, `row`, `score`, where `score` is the RRF score) it carries:

- `rrf_score` — the final fused score
- `retrievers` / `agreement` — which retrievers found it, and how many
- `rank_of(name)`, `raw_scores`, `contributions` — per-retriever rank, raw
  score, and `1/(k+rank+1)` term

All three provenance dicts contain **only the retrievers that actually returned
the document**, consistently with `retrievers` and `agreement`. Reporting `0.0`
for a miss would be indistinguishable from a genuine zero score.

`to_dict()` emits a JSON-ready breakdown:
`{doc_id, title, snippet, rank, rrf_score, agreement, retrievers: {name: {rank, raw_score, rrf_contribution}}}`.

### Ordering and ties

Sorted by `(-rrf_score, best_rank, -agreement, doc_id)`. The `doc_id` key is not
cosmetic: two documents found at exactly the same ranks by the same retrievers
are genuinely indistinguishable to RRF, and without a final key the sort falls
back to input order, which depends on dict iteration. The same inputs always
produce the same output.

### Validation

| Check | Result |
|---|---|
| `python -m pytest` | **466 passed**, 0 failed (was 351 before Phase 7) |
| `ruff check src tests scripts` | clean |
| `black --check src tests scripts` | clean, 48 files |
| `mypy src` | clean, 28 source files |
| `scripts/probe_index.py` (Phase 3/5) | still runs, unchanged output |
| `scripts/measure_semantic.py` (Phase 6) | still runs |
| `scripts/probe_hybrid.py` (new) | runs, provenance correct |
| `scripts/hybrid_experiment.py` (new) | runs on the real 300-doc corpus |

61 RRF unit tests cover cases A–H from the specification: same ranking, opposite
rankings, single-retriever-only, partial overlap, duplicate ids, empty result
from one retriever, empty from both, and explicit 1-based numbering. 54
integration tests use the **real** `BM25Searcher` and the **real**
`SemanticSearchService`; only the embedding model is substituted. Nothing about
the two retrievals is mocked, because the failure this layer could plausibly
have is precisely a disagreement at the seam.

### Real measurements

Machine: Windows 10, Python 3.11.9, 8 logical cores, **no GPU**,
`sentence-transformers/all-MiniLM-L6-v2` on CPU, 300-document corpus,
`flat_ip` 384d. 15 fixed queries × 20 passes = 300 warm samples, `top_k=10`,
`candidate_pool=100`, `rrf_k=60`. Percentiles are nearest-rank, no
interpolation, so every figure is a sample that actually occurred. End-to-end is
measured through the public API, not summed from the stages, so the overhead the
stages omit is included.

| Stage | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|
| bm25 search (depth 100) | 1.39 ms | 2.91 ms | 3.49 ms | 3.69 ms | 1.53 ms |
| query encoding | 17.72 ms | 24.97 ms | 28.34 ms | 29.41 ms | 17.86 ms |
| faiss search (depth 100) | 0.43 ms | 0.86 ms | 1.01 ms | 1.16 ms | 0.47 ms |
| **RRF fusion** | **1.01 ms** | **1.89 ms** | **2.28 ms** | **2.53 ms** | **1.06 ms** |
| result assembly (10 results) | 2.18 ms | 4.30 ms | 5.06 ms | 5.50 ms | 2.35 ms |
| END-TO-END (`hybrid.search`) | 23.71 ms | 33.39 ms | 38.01 ms | 52.05 ms | 23.92 ms |

**Fusion measured at 1.06 ms mean — 4.4% of end-to-end, 0.059× one query
encode.** The assumption that RRF is cheap next to SentenceTransformer is
therefore supported by measurement, not asserted. Startup (store + index + model
load) is 31.2 s and is not query latency.

Candidate-depth cost, same protocol:

| `candidate_pool` | depth | fusion mean | BM25 mean | result assembly mean | end-to-end mean |
|---|---|---|---|---|---|
| 1 | 10 | 0.14 ms | 0.95 ms | — | 17.61 ms |
| 50 | 50 | 0.59 ms | 1.37 ms | 2.27 ms | 22.37 ms |
| 100 | 100 | 1.06 ms | 1.66 ms | 2.45 ms | 25.33 ms |

**Caveat, stated because it would be easy to over-read:** the end-to-end column
across these three runs moved 17.6 → 22.4 → 25.3 ms, which is a larger spread
than the stages account for. A repeat of the depth-100 run gave fusion 1.06 ms
and end-to-end 23.92 ms, then 1.08 ms and 25.33 ms — fusion is stable to ~2%,
end-to-end is not. So the **fusion column is trustworthy and the end-to-end
column is not comparable across rows.** The honest reading is the fusion column:
fusion scales roughly linearly with candidate count (0.14 → 0.59 → 1.06 ms for
10 → 50 → 100 per retriever), and it is cheap in absolute terms. Do not quote a
per-depth end-to-end delta from this table.

### Corpus experiment

`scripts/hybrid_experiment.py`, 10 fixed queries, `top_k=5`, depth 100. The
diagnostic finding is **how much the two retrievers disagree**:

- Mean top-5 overlap between BM25 and semantic: **30%**
- Two queries had **0/5** overlap (`"how vaccines train the immune system"`,
  `"black holes and gravitational fields"`)
- `"how a muscle pump moves blood"`: 4/5 overlap, but in a **different order** —
  BM25 `Motor neuron diseases, Arteriovenous malformation, Blood on the Tracks…`;
  semantic `Branchiopoda, Blood on the Tracks, Arteriovenous malformation…`.
  Fusion's top-3 is all three of the documents both retrievers ranked highly.

That overlap figure is a property of this corpus and this model, measured here
rather than assumed. It says fusion has real disagreement to work on. It says
nothing about whether the fused ranking is better.

### Evidence status

| Claim | Status | Basis |
|---|---|---|
| RRF arithmetic is correct | **UNIT-VERIFIED** | 61 tests, expected values as hand-written fractions at k=60; cases A–H |
| 1-based rank convention, no off-by-one | **UNIT-VERIFIED** | `reciprocal_rank(0) == 1/61`, tested against hand arithmetic |
| Raw scores never enter the arithmetic | **UNIT-VERIFIED** | Identical rankings score identically whatever the raw scores; inverting raw scores does not reorder |
| Deterministic ordering and tie-breaking | **UNIT-VERIFIED** | Same input → same output, including ties and 3 retriever names |
| Hybrid pipeline wiring, provenance, degraded modes | **INTEGRATION-VERIFIED** | 54 tests over real `BM25Searcher` + real `SemanticSearchService`, 8-doc corpus |
| Fusion latency | **MEASURED** | 1.06 ms mean at depth 100, 300 warm samples; stage-separated above |
| Cost of candidate depth | **MEASURED** | 0.14 / 0.59 / 1.06 ms fusion at depth 10 / 50 / 100 |
| **Hybrid retrieval quality vs. either retriever alone** | **NOT YET MEASURED** | No labelled relevance judgements exist in this project. Requires Phase 10. |

### What is not claimed

**Hybrid search performs better. RRF solves the weaknesses of BM25. RRF
improves semantic search.** None of these is supported by anything measured in
this phase, and none should be inferred from anything above.

The correct summary is: *RRF is implemented and mathematically and
integration-verified. Retrieval-quality improvement remains unmeasured.*

Specifically, the earlier `"reflectivity of the moon and snow"` example remains
**invalid evidence** — the corpus contains no article about lunar reflectivity,
so it demonstrates nothing about either retriever. The `"how a muscle pump
moves blood"` example remains **evidence that BM25 and semantic retrieval
exhibit different failure modes**, and nothing more. The 30% overlap figure is
evidence that the two retrievers disagree on this corpus. It is not evidence
that reconciling that disagreement improves anything.

### Phase 7: issues found and fixed

1. **`FakeModel` was nondeterministic across processes.** Both
   `test_hybrid_service.py` and the pre-existing `test_semantic_service.py`
   built fake embeddings from the builtin `hash()`. CPython **salts string
   hashing per process**, so the same test produced a different ranking on every
   run — two runs disagreed about which document fused to the top, and the
   failures *moved between runs*, which reads exactly like a nondeterminism bug
   in the code under test. Fixed by switching both to `hashlib.blake2b`. This
   was a real defect in the test fixtures and it also affected Phase 6.
2. **RRF is not rank-consistent, and my first tests asserted the opposite.** `1/x`
   is convex, so under a symmetric rank reversal the documents at ranks 1 and 3
   beat the one at rank 2: `1/61 + 1/63 = 0.032266` vs `2/62 = 0.032258`, a
   0.026% gap. My initial Case-B tests claimed the middle document wins and the
   extremes tie. Both were wrong. Replaced with tests that assert the real
   numbers, and one that explicitly documents the property and names RRFr as the
   rank-consistent alternative.
3. **A duplicate-id check was unreachable.** `RetrievalRun.__post_init__`
   compared `len(ranked_ids)` to `len(scores)`; a repeated id collapsed the score
   dict, so that check fired first with a confusing "2 ranked ids but 1 scores"
   and `fuse()`'s clear diagnostic — naming the retriever and the document — was
   dead code. Relaxed the constructor to only catch the real defect (a score
   with no ranked id) and left duplicate detection to `fuse()`.
4. **Two fabricated classes in the first `hybrid.py` draft.** It referenced a
   nonexistent `SearchResultShim.from_document` and stashed raw scores on
   instance attributes that were never assigned. Caught before running; the file
   was rewritten to thread `raw_scores` explicitly.
5. **`FaissHit` has no `doc_id`.** The benchmark assumed it did. `FaissHit`
   carries a store `row`, exactly as the service sees it; resolved through the
   same store instead.
6. **The Windows console could not encode the corpus.** cp1252 raised
   `UnicodeEncodeError` on `Ælfheah of Canterbury` and Park Güell. Fixed by
   reconfiguring stdout to UTF-8 rather than stripping characters — a probe that
   silently renders `?` would disagree with the documents it is probing.

### Phase 7: remaining limitations

1. **Quality is unmeasured.** The central limitation. This phase built the
   mechanism; whether it helps is an empirical question that needs labelled
   queries, which arrive in Phase 10.
2. **A 300-document Wikipedia corpus is a weak test of retrieval quality.** The
   top-5 lists for `"how vaccines train the immune system"` are AIM-54 Phoenix
   and AGM65 Maverick. The corpus simply does not contain the answers. A fused
   ranking cannot be better than the best document available.
3. **Only two retrievers, both unweighted.** RRF treats them symmetrically. A
   weighted variant (`0.7 * bm25_rrf + 0.3 * semantic_rrf`) is a natural next
   step and is *not* implemented, because choosing weights without a labelled
   set would be fitting noise.
4. **Fusion is not incremental.** Every query re-runs both retrievers from
   scratch. That is correct and cheap at this scale, and would need rethinking
   at a scale where one retriever dominates latency.
5. **`flat_ip` is exact search.** Latency at 300 documents says nothing about
   behaviour at 300 000, where ANN recall and build cost enter the picture.

## Phase 8: query processing

### Architecture

```
User query
  -> QueryProcessor      validate -> normalize -> correct -> expand
  -> one string          query_for_retrieval
  -> HybridSearchService BM25 + semantic, same string
  -> RRF -> results
```

The question Phase 7 left open — does richer processing belong above the hybrid
layer, or inside it — is answered **above**, and the existing layers are
untouched. `BM25Searcher`, `BM25Index`, `FaissIndex`, `SemanticSearchService`,
`HybridSearchService`, and `rrf.py` were not modified in this phase.

**The one-string guarantee is inherited, not reimplemented.** `HybridSearchService`
already fans a single query out to both retrievers (Phase 7), so
`SearchService.search` has no seam at which the two could diverge. The probe
asserts it directly rather than trusting it: it wraps `bm25.search` and
`semantic.search`, captures the exact string each received, and compares them.
Last run: `PASS: both retrievers received the identical string.`

**Validation is injected, not imported.** `QueryProcessor` takes a
`validate: Callable[[str], str]`; `SearchService.from_settings` supplies
`hybrid.semantic.validate_query`, or `hybrid.semanticless_validate` when there
is no semantic retriever. Importing the search layer into the query layer would
make a layer *above* retrieval into a peer of it, and would give the definition
of "a valid query" a second home. Injection also makes the processor testable
with a one-line fake.

The dependency direction is asserted in the tests: nothing in
`semantic_search/query/` imports from `semantic_search/search/`,
`semantic_search/index/`, `faiss`, or `sentence_transformers`.

### Files created

| File | Purpose |
|---|---|
| `src/semantic_search/query/models.py` | `ProcessedQuery`, `Correction`, `Expansion`, `Vocabulary`. Frozen slotted dataclasses with `to_dict()`. |
| `src/semantic_search/query/normalizer.py` | `normalize(query) -> str`. Reuses `normalize_unicode` and `normalize_whitespace`; no new normalization logic. |
| `src/semantic_search/query/spelling.py` | `SpellCorrector` + `build_vocabulary`. Four gates, described below. |
| `src/semantic_search/query/synonyms.py` | ~30 curated unambiguous pairs, `expand()` record-only. |
| `src/semantic_search/query/processor.py` | `QueryProcessor`: the pipeline, `process(query) -> ProcessedQuery`. |
| `src/semantic_search/query/service.py` | `SearchService`: thin orchestrator, `search` / `search_raw`. |
| `scripts/probe_query_processing.py` | Original / normalized / corrected / expanded, ON vs OFF, plus the both-retrievers check. |
| `scripts/measure_query_processing.py` | Stage-separated latency, as-shipped and `spell_check=True`. |
| `scripts/query_processing_experiment.py` | 26 fixed queries in three groups, real 300-doc corpus. |
| `tests/unit/test_query_processing.py` | 46 tests. |
| `tests/unit/test_query_spelling.py` | 33 tests, over an injected term→df map. |

### The corrector: four gates, and why three was not enough

The corrector is calibrated against the real 300-document vocabulary (43,120
terms). The calibration is a **validation on this corpus, not a general
guarantee**, and it is on the record here so the claim is checkable.

**Gate 0 — membership.** A term already in the vocabulary is never rewritten.
This is what makes a correctly-spelled known word safe: `Hamilton` (df 6),
`Obama` (4) and `Einstein` (3) resolve to themselves at edit distance 0.

**Gate 0b — stopword protection.** *(added mid-phase; see issues below.)* A term
the tokenizer treats as a stopword is never rewritten, in either direction.

**Gate 1 — similarity.** `rapidfuzz.fuzz.ratio(casefolded, casefolded) >= 80`.
Casefolding before scoring is not cosmetic: `fuzz.ratio` is case-sensitive, and a
single capital letter costs ~15 similarity points, which is enough to refuse a
correctable typo.

**Gate 2 — edit distance.** Levenshtein `<= 2`. This rejects
`hemoglobinn -> methemoglobinemia` (distance 7, similarity 71.4).

**Gate 3 — document frequency.** The candidate must appear in `>= 3` documents.
This is the proper-noun guard: `machine` has df 16 and `engine` 15, while the
names that must be protected have df 1–6. A rare word's nearest neighbour is
usually a coincidence; a frequent word's is evidence of a typo.

Three calibration results worth keeping, because they are what ruled out the
alternatives:

- **A single similarity threshold cannot work.** `Serkan` (a name that must
  survive) scores 66.7 against its nearest neighbour, while `teh` (a definite
  typo that should be fixed) scores only 85.7. Any threshold low enough to catch
  every typo also rewrites names. This is why the declared `min_correction_score`
  of 92.0 was lowered to 80.0 *and supplemented* rather than merely replaced.
- **`fuzz.WRatio` is unusable.** Its partial-ratio component produced garbage:
  `machien -> ach`, `learnign -> ear`, `hemoglobinn -> bin`. Plain `fuzz.ratio`
  is the correct scorer.
- **Margin over the runner-up does not discriminate.** `recieve` has margin 0.0
  and must be corrected; `Serkan` has margin 0.0 and must not. Margin was dropped.

Capitalized tokens are never corrected (`preserve_proper_nouns`). The cost —
a capitalized typo like `Machne` is missed — is asserted by a test rather than
left as folklore, and the flag is shown to be load-bearing by flipping it.

### Why `spell_check` ships **off**

This is the most consequential decision in the phase, and it was forced by
measurement rather than chosen for caution.

Against the 300-document corpus the corrector fixes real typos and also mangles
**real English words the corpus happens not to contain**:

| Typed | Rewritten to | Similarity | Kind of word |
|---|---|---|---|
| `tectonics` | `tectonic` | 94.1 | correct, out of corpus |
| `virus` | `virtues` | 83.3 | correct, out of corpus |
| `turbine` | `turin` | 83.3 | correct, out of corpus |

A similarity threshold cannot fix this — 94.1 is high. Edit distance cannot
either — all three are distance 1. Raising the bar high enough to reject them
also rejects the corrections. An external word-frequency list was tried: 6 of 18
calibration cases were still mangled with a google-10000 list attached. No
threshold and no frequency list was found that removes the failure without
removing the corrections, so the record says so rather than the config.

A wrong rewrite is worse than a missed correction: the user gets confidently
wrong results and no signal that anything happened. Therefore
`query.spell_check: false` is the shipped default, with the reason recorded in
`QueryConfig`'s docstring, in `configs/default.yaml`, in `.env.example`, and
pinned by `test_the_safe_defaults_are_the_one_that_ship`. It is one flag away
for deployments whose users mistype.

`query.synonym_expansion` ships **off** for a different and weaker reason: there
is no labelled evaluation data, so an unmeasured rewrite of the retrieval query
is not a safe default. Expansions are still **computed and recorded** so the
explainability panel can show them — recording is free of risk, applying is not.

### The raw path is always available

`SearchService.search_raw` sends the query to `HybridSearchService.search`
exactly as typed. It is not an unvalidated shortcut: it is the same call the
processed path makes, minus the processing. This is what makes the ON/OFF
comparison a controlled experiment rather than two deployments, and it is
asserted by test — with `query.enabled: false`, `search` and `search_raw` return
identical results.

`ProcessedQuery` reports the same string four times and two empty lists for an
untouched query. That is the honest answer, and it lets a client render "your
query" without a special case for the unchanged case. A correction is never
claimed for a query that was not changed.

### Validation

| Check | Result |
|---|---|
| `python -m pytest` | **546 passed**, 0 failed (was 466 before Phase 8) |
| `ruff check src tests scripts` | clean |
| `black --check src tests scripts` | clean, 60 files |
| `mypy src` | clean, 35 source files |
| `pytest tests/unit/test_rrf.py tests/unit/test_hybrid_service.py` | **115 passed** — Phase 7 not regressed |
| `scripts/probe_query_processing.py` (new) | runs, both-retrievers check passes |
| `scripts/measure_query_processing.py` (new) | runs, stage-separated |
| `scripts/query_processing_experiment.py` (new) | 26 queries on the real corpus |

80 new tests (46 + 33, with one overlapping name). The spelling tests inject a
hand-written term→df map rather than loading a corpus; the *logic* is real, only
the data is small, and the corpus-building path is covered separately.

### Real measurements

Machine: Windows 10, Python 3.11.9, 8 logical cores, **no GPU**,
`sentence-transformers/all-MiniLM-L6-v2` on CPU, 300-document corpus,
`flat_ip` 384d, `top_k=10`. 15 fixed queries × 20 passes = 300 warm samples.
Percentiles are nearest-rank, no interpolation. Vocabulary build (one-off, at
startup, shared by every request): **623 ms** for 43,120 terms.

**As-shipped configuration** (`spell_check=False`, `synonym_expansion=False`):

| Stage | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|
| normalize | 0.030 ms | 0.047 ms | 0.060 ms | 0.090 ms | 0.033 ms |
| synonym expansion (recorded only) | 0.006 ms | 0.012 ms | 0.014 ms | 0.016 ms | 0.007 ms |
| **PROCESSING TOTAL** | **0.031 ms** | **0.068 ms** | **0.086 ms** | **0.101 ms** | **0.037 ms** |
| RETRIEVAL ONLY (`hybrid.search`) | 14.96 ms | 19.25 ms | 21.10 ms | 101.66 ms | 15.23 ms |
| END-TO-END, processing OFF | 15.28 ms | 19.70 ms | 21.49 ms | 23.64 ms | 15.25 ms |
| END-TO-END, processing ON | 14.70 ms | 19.85 ms | 21.75 ms | 24.08 ms | 15.03 ms |

**Processing is 0.24% of retrieval and 0.25% of end-to-end.** The ON-vs-OFF
difference (−1.46%) is *smaller than the run-to-run noise in this table* and
should not be read as processing making anything faster. It is free, not
beneficial.

**With `spell_check=True`** (the opt-in configuration, not the default):

| Stage | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|
| PROCESSING TOTAL, spell_check=True | 0.027 ms | 14.13 ms | 19.07 ms | 20.98 ms | 1.55 ms |
| spell check, clean query set | 0.012 ms | 14.10 ms | 19.25 ms | 21.06 ms | 1.65 ms |
| spell check, typo query set | 4.69 ms | 14.08 ms | 17.32 ms | 23.62 ms | 4.77 ms |

160 of 1660 typed terms were corrected (9.6%) over 20 passes of the typo set.
**That is a correction *rate* on a synthetic typo set, not retrieval quality,
and it is the wrong number to quote as an improvement.**

**The clean-set spell cost is not near zero, and that was not the expectation.**
Gate 0 only short-circuits terms the corpus actually contains. An ordinary
English query against a 300-article Wikipedia corpus is mostly
out-of-vocabulary — `orbital`, `vaccines`, `butterfly` are not in these 300
articles — so each one falls through to a fuzzy search over all 43,120 terms and
is refused by a later gate. The full cost of a correction nobody gets is paid in
full. (Gate 0b cut this figure substantially: stopwords, which are the terms
guaranteed to be absent from the vocabulary, were a large share of the work.)

### Corpus experiment

`scripts/query_processing_experiment.py`, 26 fixed queries, `top_k=5`, real
300-document corpus, `spell_check=True` (so the feature is visible at all).

| Group | Queries | Rewritten | Corrections | Rankings changed |
|---|---|---|---|---|
| CLEAN — correctly spelled, ordinary English | 10 | **0** | 0 | **0** |
| TYPO — deliberate misspellings | 10 | 6 | 6 | 5 |
| OOV — real words absent from the corpus | 6 | 2 | 2 | 2 |
| **Total** | **26** | **8** | **8** | **7** |

Read the CLEAN and OOV rows before the TYPO one. The TYPO row is the feature
working; the other two are the feature costing something. A corrector that
rewrites clean queries is doing damage, and this table is the only thing in the
phase that would reveal it.

The TYPO group caught 6 of 10. The four it missed were `pumpe`, `Empie`,
`eartquakes`, and `Earths`; I did not investigate each miss individually, so no
mechanism is claimed for them here. The two OOV failures are the documented
`turbine -> turin` and `virus -> virtues` cases, each moving 3 of the top 5.

**Before gate 0b was added, the same run reported CLEAN 5/10, 17 corrections,
and 12 changed rankings.** The CLEAN-group rewrites were the bug below.

### Evidence status

| Claim | Status | Basis |
|---|---|---|
| Normalization is conservative, reusable, idempotent | **UNIT-VERIFIED** | tests over whitespace, NFKC, tabs, newlines, empty input |
| Typos are corrected, and the four gates classify the calibration set correctly | **UNIT-VERIFIED** | 33 tests over an injected term→df map; calibration on the real 43,120-term vocabulary |
| A stopword is never rewritten, and the gate is load-bearing | **UNIT-VERIFIED** | test asserts the real 85.7 failures reproduce with the gate off |
| `original` is never mutated; an unchanged query reports zero corrections | **UNIT-VERIFIED** | tests |
| Pipeline order and `enabled=False` degradation to `search_raw` | **UNIT-VERIFIED** | tests |
| Same processed string reaches both retrievers | **INTEGRATION-VERIFIED** | probe captures each retriever's argument and compares |
| Processing ON vs OFF is a runtime comparison on the real corpus | **INTEGRATION-VERIFIED** | `query_processing_experiment.py`, 26 queries, 300-doc corpus |
| Processing latency | **MEASURED** | 0.037 ms mean as-shipped; 1.55 ms with `spell_check=True` |
| Vocabulary build cost | **MEASURED** | 623 ms one-off, 43,120 terms |
| **Retrieval quality with processing vs. without** | **NOT YET MEASURED** | No labelled relevance judgements exist in this project. Requires Phase 10. |
| **Whether a correction improves results for the user who typed the typo** | **NOT YET MEASURED** | Same reason. This is the whole point of the feature and it is unproven. |

### What is not claimed

**Query processing improves retrieval quality. Spell correction helps users find
what they meant. Synonym expansion improves recall.** None of these is
supported by anything measured in this phase, and none should be inferred from
anything above.

The correct summary is: *query processing is implemented, unit- and
integration-verified, and measured for cost. Whether it improves results is
unmeasured.*

Specifically:

- The 8-changed-queries figure is **a change, not an improvement**. The
  experiment script labels every ranking difference that way, and 2 of the 8 are
  confidently wrong (`turbine -> turin`, `virus -> virtues`).
- The 9.6% correction rate is **a rate on a synthetic typo set**, not an
  accuracy figure — there is no ground truth about which corrections were right.
- The four-gate rule is **a calibration on one corpus**, not a proof. It
  classifies every case I tested. A typo over a rare technical term, a frequent
  proper noun, and a real out-of-corpus word are all known failure shapes, and
  the last one is not gated at all — which is why the stage ships off.
- The `"reflectivity of the moon and snow"` example remains **invalid
  evidence** and was not reused. `"how a muscle pump moves blood"` appears in
  the CLEAN group of the experiment as one of ten clean queries, and stays
  evidence only of **different retriever behaviour** from Phase 7.

### Phase 8: issues found and fixed

1. **The corrector rewrote stopwords in correctly-spelled queries.** Found by
   the 26-query experiment, not by reasoning, and by a mechanism no threshold
   analysis could have surfaced — it is a gap in gate 0, not a threshold problem:

   ```
   "black holes and gravitaional fields"  ->  "and" -> "andy"  (similarity 85.7)
   "how eartquakes are measured"          ->  "are" -> "acre"  (similarity 85.7)
   ```

   Both confidently wrong, and both moved documents in and out of the top 5.
   Root cause: `Tokenizer.surface_terms()` strips stopwords, so **a stopword is
   absent from the vocabulary because the corpus lacks it, not because the
   corpus does not contain it** — gate 0 could not vouch for it, and the fuzzy
   search was free to replace it. Fixed by gate 0b, which refuses to rewrite a
   stopword in either direction (so `teh` cannot become `the` either). CLEAN went
   5/10 → 0/10; corrections 17 → 8; changed rankings 12 → 7. The cost is that
   stopword typos are never corrected, which is stated rather than hidden.
2. **Newlines survived normalization and broke spell correction.**
   `normalize_whitespace` preserves paragraph breaks — correct for documents, and
   a query is a single line. `SpellCorrector.correct` splits on `" "`, so a
   surviving newline produced a two-token "word" that could not be matched
   against anything. Fixed by flattening newlines in `normalize()`.
3. **`processor.process(123)` raised `AttributeError`, not `TypeError`.** The
   injected default validator assumed a string. The type check is now hoisted
   into `_default_validate` so the failure is the documented one.
4. **A measurement script was timing the wrong processor.** The "as-shipped"
   `PROCESSING TOTAL` row was measuring the spell-*enabled* processor, reporting
   16.3 ms. The shipped configuration is **0.037 ms**. Both processors are now
   timed separately; the earlier number was 440× the real cost.
5. **A hardcoded claim was contradicted by its own measurement.** A comment
   asserted the clean-set spell stage was "nearly free, because every term hits
   the membership gate." It measured 20.681 ms. Replaced with the real
   explanation — most ordinary query terms are absent from a 300-doc corpus, so
   they reach the fuzzy search and are refused late. The claim was wrong and the
   measurement was right; the comment was the thing that had to change.
6. **Two of my own tests asserted the wrong thing, and the second was subtler.**
   `test_spell_check_disabled_skips_correction` asserted the corrected string was
   `"machine learning"` with `spell_check=False`, which would have meant the flag
   was being ignored — `corrected` is the *post-spelling-stage* value, so with
   the stage off it is the input. And the gate-0b test initially asserted a
   correction the fixture vocabulary could not support, then a correction target
   the fixture lacked; I checked the exact behaviour in a REPL and rebuilt the
   fixture so it reproduces the real failures verbatim with the gate off.
7. **A minimum-length rule was added and then removed.** Short words have
   ambiguous candidates, so a length floor looked defensible. It earned nothing:
   0 of the 1,587 three-letter terms in the real corpus would have been rewritten
   without one, because they are all already in the vocabulary and short-circuit
   at gate 0. A rule that prevents no measured harm is a rule that only makes the
   corrector miss things, so it was deleted. This is recorded because the instinct
   to add it will recur.

### Phase 8: remaining limitations

1. **Retrieval quality is unmeasured** — the central limitation, as in every
   prior phase. Requires Phase 10.
2. **The corrector cannot know a real word the corpus does not contain.** This is
   the reason the stage ships off and it is not fixable within this design.
3. **`fuzz.ratio` scores on raw characters**, so `"water"` and `"Water"` are only
   1 point apart once casefolded but a term like `"iPhone"` is genuinely awkward.
   No word-segmentation or case-aware scorer was explored.
4. **The vocabulary is built once and never invalidated.** A corpus that changes
   at runtime — incremental ingestion, document deletion — leaves the corrector
   correcting against a stale vocabulary, silently. There is no cache key.
5. **The synonym map is 30 hand-written pairs.** It covers common English
   ambiguities and nothing else. Ambiguous pairs (`bass`, `bat`) are deliberately
   excluded and that exclusion is documented, but the map is a placeholder that
   will not scale, and it is off by default anyway.
6. **Normalization is Latin/Unicode-aware but not language-aware.** There is no
   stemming, lemmatization, or language detection, so `"mice"` is not folded to
   `"mouse"` before either retriever sees it.
7. **A 300-document corpus is a weak test of any query-processing claim.** Most
   ordinary English queries are out of vocabulary against it, which distorts both
   the correction rate and the cost in ways a larger corpus would not.

## Phase 9: application search service

### A note on the numbering collision

The original roadmap's Phase 9 is **explainability**. The Phase 9 specification
received for this work redefines Phase 9 as the **application search service**,
and puts explainability in a later phase. Rather than silently overwrite the
existing table, the table above now reads **9a** (application service, complete)
and **9b** (explainability, pending). Nothing about the original roadmap's scope
was dropped; the number moved.

### Implementation

`src/semantic_search/app/` — a new package, three files:

| File | Contents |
|---|---|
| `__init__.py` | Public exports and the dependency-arrow docstring. |
| `models.py` | `SearchRequest`, `SearchHit`, `HitProvenance`, `SearchMetadata`, `ApplicationSearchResponse`. |
| `service.py` | `ApplicationSearchService` — validation, ON/OFF dispatch, result mapping, metadata, `info()`, `health()`. |

The new layer adds exactly three things the layers below cannot supply to a
caller: a validated **request object**, a **response object** that never mentions
FAISS rows or `rrf_k`, and **metadata** that makes "nothing matched"
distinguishable from "something is broken".

It adds no retrieval behaviour. It does not re-sort, re-score, deduplicate, or
reorder. The order a caller sees is the order `HybridSearchService` produced.

**Modified:** `core/config.py` (`AppConfig` + the `app:` settings block),
`core/exceptions.py` (`SearchBackendError` gained an explicit `status_code`),
`configs/default.yaml`, `.env.example`.

**Explicitly not modified:** `BM25Searcher`, `BM25Index`, `FaissIndex`,
`SemanticSearchService`, `HybridSearchService`, `rrf.py`, `hybrid_models.py`,
`processor.py`, `spelling.py`, `query/service.py`, and every pre-existing test.

### Placement: `semantic_search/app/`, not `search/service.py`

The spec's example path was `src/semantic_search/search/service.py`, and that
path was deliberately not used. `search/` is the **hybrid retrieval** layer;
putting the application service there would place it *beside* retrieval rather
than *above* it, and would force `search` to import `query` — inverting the
dependency arrow Phase 8 existed to establish. The spec's own instruction to use
the repository's existing naming conventions applies, and the flow diagram in the
spec already places the service above the query processor. The code now matches
the diagram.

The `app/` directory at the repository root is the Streamlit scaffold and is a
different thing; `pyproject.toml` packages only from `src`.

### Name collision

`SearchService` and `SearchResponse` already exist in `semantic_search.query.service`
(Phase 8). No existing name was changed. The new types are
`ApplicationSearchService` and `ApplicationSearchResponse`, and the new service
**wraps** the Phase 8 service rather than replacing it — so Phase 8's processor,
corrector, `ProcessedQuery`, and its tested ON/OFF mechanism all keep executing,
and the Phase 8 tests keep testing the code that ships.

### `filters` is deliberately absent

The spec says not to invent unsupported filter behaviour to populate a field. The
only `filters` in `src/` is `SearchConfig.default_filters`, a field nothing reads
and no retriever implements. A field that is accepted and ignored is a lie to the
API layer. `SearchRequest` has no `filters` attribute; filtering is a later-phase
question. **This is an explicit omission, not a gap.**

### Architecture

```
                    ApplicationSearchService.search(SearchRequest)
                                    |
        +---------------------------+---------------------------+
        |                                                       |
  SearchRequest.__post_init__                        SearchError -> 400
  (str / int-not-bool / >=1)                         SearchBackendError -> 503
        |
   _resolve_processing()  ------ False --->  query.SearchService.search_raw()
        |                                          |
   _resolve_top_k()  (clamped; pre-clamp kept)     |
        |                                          |
        +----------------- True ------------------+
                          |                       |
                 query.SearchService.search()  |  HybridSearchService.search()
                          |                   |    (BM25 + semantic, ONE string)
                 QueryProcessor.process()   |                   |
                          |                   |              RRF fusion
                 ProcessedQuery -------------+                   |
                          |                                       |
                          +-------------------+-------------------+
                                              |
                                    list[HybridSearchResult]
                                              |
                          _to_hit()  0-based rank -> 1-based display rank
                                      provenance keeps the 0-based value
                                              |
                              ApplicationSearchResponse
                              (results, metadata, both queries)
```

### Validation

Two deliberate, visible divergences from the layers below, both tested:

1. **`SearchHit.rank` is 1-based.** `SearchResult.rank`, `BM25Result.rank`, and
   the RRF arithmetic are all 0-based by Phase 7 design. The conversion happens
   once, in `_to_hit`, and the original 0-based value is preserved in each
   `HitProvenance`. Nothing is lost.
2. **`top_k` over the maximum is clamped, not rejected**, and
   `metadata.requested_top_k` records the **pre-clamp** number. A caller asking
   for 500 results from a 300-document corpus is asking a reasonable question;
   the clamp is a fact the caller can read rather than a surprise it has to
   infer.

Validation is otherwise **delegated, not duplicated**: emptiness and length are
enforced by the existing `QueryProcessor` path, and `top_k` range by the existing
retriever validators. `SearchRequest.__post_init__` checks types and ranges only.

### Dependency direction

The rule is `app → query → search → index → data`, one way, and nothing below
`app` may import `app`.

**The enforcing test is
`tests/integration/test_app_integration.py::TestDependencyDirection`** — 9 tests.
It parses every `.py` file under `src/semantic_search/` with `ast` and collects
absolute *and* relative import statements. The check is in the normal test
suite, so a violation fails `pytest`, not a linter someone can skip.

- `test_query_package_does_not_import_the_application_service` — the rule the
  spec named explicitly.
- `test_no_lower_layer_imports_the_application_service` — parametrized over
  `query`, `search`, `index`, `data`, `embeddings`, `core`.
- `test_the_application_layer_actually_uses_the_layers_below` — a rule that only
  forbids things passes trivially on an empty codebase, so this asserts `app`
  genuinely imports `query` and `search`.
- `test_the_checker_would_actually_catch_a_violation` — writes a file that
  violates the rule (in both absolute and relative form) to a temp directory and
  asserts the parser flags it. **The guard is itself tested**, so it cannot
  silently rot into a test that always passes.

### Integration probe

`scripts/probe_search_service.py`, run against the **real** 300-document corpus,
the real BM25 index, the real FAISS index, and the real all-MiniLM-L6-v2 model.
Setup 26.83 s. All eight cases produced real output:

| # | Case | Observed |
|---|---|---|
| 1 | Normal query | `'machine learning'` → 3 results, ranks 1–3, scores 0.0328 / 0.0318 / 0.0296 |
| 2 | Extra whitespace | `'  machine   learning  '` → processed `'machine learning'`, `was_modified=True`, `was_corrected=False` |
| 3 | Processing disabled | `query_processing_enabled: False`, processed string unchanged |
| 4 | Processing enabled | `query_processing_enabled: True` |
| 5 | `top_k` below available | `top_k=2` → 2 returned, `requested_top_k=2`, `truncated=False` |
| 6 | `top_k` above documents | `top_k=1000` → **100** returned, `requested_top_k=1000`, `corpus_exhausted=False` |
| 7 | Invalid/empty | `''`, `'   '`, `'\t\n'` → all `EmptyQueryError` |
| 8 | Retrievers differ | see below |

Case 6 returns 100 rather than 300 because `app.max_top_k` defaults to 100. That
is the clamp working, and `requested_top_k: 1000` is what makes it visible rather
than silent. `corpus_exhausted` is correctly `False`: only 100 of 300 came back.

Case 8 (`'how a muscle pump moves blood'`) shows real divergence, taken from the
per-retriever ranks that `provenance` preserves:

```
Rank 1: Motor neuron diseases    bm25 said 0, semantic said 3
Rank 2: Arteriovenous malformation  bm25 said 1, semantic said 2
Rank 3: Blood on the Tracks     bm25 said 2, semantic said 1
Rank 4: Branchiopoda            bm25 said 8, semantic said 0
```

All five documents were found by both retrievers, but they ranked them
differently — RRF demonstrably fused two distinct lists rather than echoing one.
**This is evidence that the retrievers behave differently. It is not evidence
that either one is better, and no such claim is made.**

The string "reflectivity of the moon and snow" is not used anywhere.

### Performance

`scripts/measure_search_service.py`, real corpus, real indexes, real model, warm
cache. 15 distinct real queries × 20 repetitions = **300 timed requests**, after
5 untimed warmup passes. `top_k=10`. All times in milliseconds.

| Component | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| **End to end** | 20.30 | 18.20 | 33.20 | 40.15 | 130.39 |
| Query processing | 0.06 | 0.05 | 0.11 | 0.15 | 0.22 |
| Retrieval (BM25 + FAISS + RRF) | 19.77 | 18.09 | 32.85 | 40.49 | 50.61 |
| App / result assembly * | 0.47 | 0.27 | 3.29 | 6.76 | 79.67 |

\* **Derived**, not measured directly: `end_to_end − processing − retrieval` from
the same requests. On this workload the true assembly cost is a fraction of a
millisecond — the same order as a single `perf_counter` reading — so the
subtraction carries the noise of the two numbers it came from and should not be
read as a precise figure. The end-to-end column is the one timed end to end and
is authoritative.

**Environment:** Windows 11 Home Single Language, 10.0.26200, AMD64, 8 logical
CPUs, **no CUDA** (`torch 2.12.1+cpu`, `CUDA available: False`), Python 3.11.9,
numpy 2.4.6, corpus 300 documents, model `sentence-transformers/all-MiniLM-L6-v2`,
embedding dimension 384, FAISS `flat_ip`, retrievers bm25 + semantic, RRF k=60,
candidate pool 100. **Warm** — model, both indexes, and vocabulary all resident
before the first timed call. Cold-start cost (model load, index load) is excluded
and is reported separately by the probe.

Observations, stated as observations:

- Retrieval is essentially the entire cost. The application layer is noise next
  to it.
- These are **single-process, serial, latency** numbers. **No throughput or
  concurrency figure is claimed** — none was measured.
- The tail (p99 40 ms, max 130 ms) is consistent with a desktop process being
  descheduled on a shared machine. It is reported as observed, not attributed.
- These numbers are **not comparable** to the Phase 7 or Phase 8 measurements in
  this file: those used different queries, a different repetition count, and in
  some cases a different corpus size. The methodology is not the same, so no
  regression or improvement is inferred from the difference.

### Tests

| Suite | Tests | What it uses |
|---|---|---|
| `tests/unit/test_app_service.py` | 52 | Fakes at the Phase 8 boundary |
| `tests/integration/test_app_integration.py` | 40 | Real BM25 + real FAISS + real `DocumentStore` + real `QueryProcessor`; only the sentence-transformer is a deterministic `blake2b` fake |
| **Total suite** | **638** | was 546 before Phase 9 |

No pre-existing test was modified, deleted, skipped, or weakened. The Phase 5–8
suites are untouched and pass as part of the 638.

### Evidence ladder

**1. Proven by unit tests**
- `SearchRequest` validation: non-string query, `top_k` of 0 / negative, `bool`
  rejected (including `True`, which is otherwise an `int`), `3.5` / `"5"` / `[5]`
  rejected, `query_processing` must be `bool | None`, request is frozen.
- `AppConfig` rejects `default_top_k` / `max_top_k` / `max_query_chars` below 1
  at construction.
- ON/OFF dispatch, in both directions: a request overrides a configured default,
  and a request can enable processing on a disabled service.
- `top_k` defaulting, clamping, and `requested_top_k` retaining the pre-clamp
  value.
- Ranks are 1-based and contiguous; result order and scores are passed through
  unchanged; 0-based provenance preserved.
- Metadata populated from a real `ProcessedQuery`; `to_dict()` is JSON-safe.
- Infrastructure failures (`CorruptIndexError`, `DataError`, `RuntimeError`)
  raise `SearchBackendError` with `status_code == 503`, the original exception
  preserved as `.cause`, not wrapped twice, and the cause absent from the client
  payload.
- Validation failures stay 400: empty, whitespace-only, over-long, and a plain
  `SearchError` from below.
- Backward compatibility: `query.SearchService` still exposes `search`,
  `search_raw`, `info`.

**2. Proven by integration tests**
- Real hybrid retrieval produces 1-based contiguous ranks, in **exactly** the
  order `hybrid.search` returned, with scores copied and not recomputed.
- `top_k` above the corpus returns everything and sets `corpus_exhausted`; above
  the maximum it clamps and stays visible via `requested_top_k`.
- Zero results is a normal response with `empty_reason == "no_matches"`.
- **The same processed string reaches BM25 and semantic** — captured by wrapping
  both real retrievers, asserted for ON and for OFF. There is no seam in which
  they could diverge.
- A failing backend never produces an empty list (explicit regression test).
- Query-processor failure propagates rather than yielding empty results; the raw
  path is unaffected by a broken processor.
- Wrapping does not change hybrid ordering or behaviour.

**3. Measured performance**
- The latency table above: 300 real requests, warm, real model and indexes, with
  p50/p95/p99/max/mean per component and a full environment block.

**4. Not yet measured**
- **Retrieval quality, and therefore whether the application layer helps
  retrieval.** No labelled relevance judgments exist.
- Query processing quality, and hybrid fusion quality — both still
  **NOT YET MEASURED**, unchanged from Phases 7 and 8.
- **Throughput and concurrency.** Not measured. No load test, no multi-worker
  test, no saturation curve.
- Latency on hardware other than the machine above, and on any GPU. The current
  measurement is CPU-only with no CUDA present.
- Behaviour under a cold cache, and startup time as a served quantity (the probe
  reports 26.83 s of setup, but that is a script, not a service).
- Behaviour at corpus sizes larger than 300 documents.
- Whether `flat_ip` remains the right index type at scale. Exact search over
  300 vectors is fast; that says nothing about 300 000.

**5. Not claimed**
- That Phase 9 improves retrieval quality. **It cannot**, by construction: the
  layer re-ranks nothing, re-scores nothing, and reorders nothing. It is a
  contract and a boundary, not a ranking change.
- That hybrid retrieval beats BM25 or semantic retrieval, or that either is
  better than the other.
- Any throughput, concurrency, or QPS figure.
- Any comparison between these latencies and the Phase 7/8 numbers in this file.
- Any claim about how the FastAPI or Streamlit layers will perform. They are not
  built.
- Any claim that the architecture is optimal, only that the dependency direction
  is now enforced by a test.

**Retrieval quality has not yet been evaluated against labeled relevance
judgments.**

**Phase 10 has not been started.**

### Phase 9: issues found and fixed

1. **`SearchBackendError` reported 500 instead of 503.** The class declared
   `code` but no `status_code`, and its parent `SearchIndexError` declares none
   either — so it silently inherited 500 from `SemanticSearchError`. This is
   exactly the failure the whole error taxonomy exists to prevent: a 500 is what
   teaches a caller to render an empty page and move on. Found by
   `test_backend_error_preserves_the_cause`, fixed by declaring `status_code = 503`
   explicitly, with a comment recording why it must not be inherited.

### Phase 9: remaining limitations

1. **Retrieval quality is unmeasured**, as in every prior phase. Requires Phase 10.
2. **`_document_count` is fixed at construction.** Adding or removing documents
   leaves the reported corpus size stale, and `corpus_exhausted` compares against
   the stale number. There is no cache key and no invalidation.
3. **Results are materialized fully into a tuple.** At `max_top_k=100` this is
   trivial; at a much larger limit it is not. There is no streaming or generator
   path.
4. **No caching, no rate limiting, no auth.** The API layer does not exist yet;
   these were deliberately not built ahead of it, and none of the configuration
   knobs imply otherwise.
5. **The dependency test is textual, not runtime.** It reads `ast.Import` nodes,
   so a violation introduced through `importlib`, a plugin registry, or a
   re-export chain would not be caught. That is a deliberate trade: a full
   import-graph framework is a lot of machinery for a repository this size, and
   the failure mode it misses requires someone to work actively against the
   rule.
6. **`to_dict()` builds the whole payload eagerly.** Fine at 100 results; a
   `to_json()` that streams would be better if the limits ever grow.
7. **The assembly figure is a subtraction**, so it inherits the noise of the two
   measurements it came from, as noted above.

## Next action

**Phase 9b: explainability.**

**Carry into 9b:** everything an explainability panel needs already exists and
Phase 9 carried it further. The `ProcessedQuery` record (Phase 8) says what was
typed, what was normalized, what was corrected, and what was expanded. The
per-retriever `HitProvenance` (Phase 7, now surfaced by Phase 9 on every
`SearchHit`) says which retriever ranked a document where, at what raw score,
and with what RRF contribution. `ApplicationSearchResponse` returns the original
query, the processed query, and all of it in one record, so a panel can render
"what you typed → what we searched → why this document, from which retrievers"
without reconstructing any of it.

**Carry forward, unchanged:** query processing quality, hybrid quality, and
retrieval quality are **all still NOT YET MEASURED**. Do not let the
8-changed-queries figure, the 9.6% correction rate, the 30% retriever overlap, or
the Case 8 divergence from the Phase 9 probe drift into implied claims that
anything helps. Phase 10 needs labelled queries.

**Also still unmeasured:** throughput and concurrency. Phase 9 measured
single-process serial latency only, and no load test has been run.
