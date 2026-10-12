# Architecture Overview

Layered one-directional dependency:
`API → Application → Search → Index → Data → Core`

Enforced by `tests/integration/test_app_integration.py` via AST import scanning.

Components:
- `src/semantic_search/core/` — config, exceptions, logging
- `src/semantic_search/data/` — corpus downloader / preprocessor
- `src/semantic_search/embeddings/` — Sentence-BERT encoder + cache
- `src/semantic_search/index/` — BM25 (pickle) + FAISS (flat_ip) builders
- `src/semantic_search/search/` — BM25 / Semantic / Hybrid + RRF fusion
- `src/semantic_search/query/` — normalize / spell / synonyms / SearchService
- `src/semantic_search/app/` — request/response contracts + ApplicationSearchService
- `src/semantic_search/api/` — FastAPI factory, routes, middleware (CORS, rate limit, request-ID)
- `cli.py` — `sse` console entry point

Evaluation harness (frozen, hashed before runs):
- Stratum A_restated: 25 queries, 25 judgments — evaluated
- A_verbatim: 15 / 15 — evaluated
- A_typo: 10 / 10 — evaluated
- B_topical: 30 / 0 — blocked (pending independent human qrels)

Deployment notes:
- No Docker required; standard Python 3.10+ env
- Two processes: `uvicorn` (API :8000) + `streamlit` (UI :8501)
- Index builds via `python scripts/build_index.py --force`
- Corpus download via `python scripts/download_data.py --limit 300`
