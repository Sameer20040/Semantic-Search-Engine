# Portfolio Notes — Semantic Search Engine

Why this project for a job:
- Real architecture with enforced dependency direction and integration tests
- Production concerns (rate limiting, error taxonomy, request IDs, CORS)
- Honest evaluation: frozen qrels, unmeasured stratum B clearly noted (integrity)
- Tech stack: Python 3.10+, FastAPI, Streamlit, FAISS, Sentence-BERT, BM25, RRF

Live demo: add URL here after deploying (Render / Railway / Streamlit Cloud)
GitHub: https://github.com/Sameer20040/Semantic-Search-Engine

Quick one-line pitch for recruiters:
"Hybrid BM25 + dense retrieval over a Wikipedia corpus, fused with RRF. FastAPI + Streamlit. 893+ tests, dependency-direction enforced by AST scanner, honest evaluation protocol — no fabricated metrics."
