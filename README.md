# Semantic Search Engine

Hybrid semantic + lexical search over a Wikipedia corpus: **BM25** and
**Sentence-BERT dense vectors** are retrieved independently and fused with
**Reciprocal Rank Fusion (RRF)**. Served as a **FastAPI** REST API with a
**Streamlit** front end, and driveable from the command line via `sse`.

No Docker. No containers. Standard Python environment.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Running the app](#running-the-app)
- [Command line](#command-line)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Development](#development)
- [API reference](#api-reference)
- [License](#license)

---

## What it does

Three retrieval arms, one fused answer:

| Arm | How it matches | Best at |
| --- | --- | --- |
| `bm25` | Lexical, term-frequency scoring over a tokenized index | Exact names, rare terms, precise wording |
| `semantic` | Dense cosine similarity via `all-MiniLM-L6-v2` + FAISS `flat_ip` | Paraphrases, concepts, related wording |
| `hybrid` | RRF over both | Queries where either arm alone would miss |

**RRF is rank-based.** For a document `d`:

```
score(d) = Σ  1 / (k + rank_i(d))        k = 60 (default)
```

Each retriever contributes its reciprocal rank, so the fusion is insensitive to
the two retrievers' incomparable score scales. It does **not** arithmetically
combine raw BM25 and cosine scores, and the API's explainability output says so.

Every search result carries **provenance** — which retriever produced it, its
rank in that retriever's own list, and its contribution to the fused score.

---

## Architecture

```
Streamlit (app/)                FastAPI (semantic_search/api/)
        │                                │
        └──────── HTTP /search ──────────┘
                         │
              ApplicationSearchService          ← the single search entry point
                         │
              ┌──────────┴──────────┐
              │                     │
     HybridSearchService      QueryProcessor (normalize / spell / synonyms)
              │                     │
     ┌────────┴────────┐
     │                 │
 BM25Index        FAISSIndex        ← both built by scripts/build_index.py
 (pickle)          (flat_ip)            validated by sse validate
```

Layering is one-directional: `api → app → query/search → index → data →
core`. An integration test enforces it, so the API layer cannot reach past the
application layer into retrieval internals.

---

## Quick start

Requires **Python 3.10+**.

### Windows (PowerShell)

```powershell
git clone https://github.com/Sameer20040/Semantic-Search-Engine.git
cd Semantic-Search-Engine

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e .
pip install -r requirements.txt

python scripts/download_data.py --limit 300
python scripts/build_index.py --force

streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501
```

### Linux / macOS

```bash
git clone https://github.com/Sameer20040/Semantic-Search-Engine.git
cd Semantic-Search-Engine

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements.txt

python scripts/download_data.py --limit 300
python scripts/build_index.py --force

streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501
```

> PowerShell may refuse to activate the venv. If so:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## Running the app

Two processes. Run them in **two terminals**.

### Terminal 1 — the API

```bash
# Windows (PowerShell)
uvicorn semantic_search.api.app:get_application --factory --host 0.0.0.0 --port 8000

# Linux / macOS
uvicorn semantic_search.api.app:get_application --factory --host 0.0.0.0 --port 8000
```

### Terminal 2 — the Streamlit UI

```bash
# Point the UI at the API (loopback default if unset)
export STREAMLIT_API_BASE_URL=http://127.0.0.1:8000        # Linux / macOS
$env:STREAMLIT_API_BASE_URL = "http://127.0.0.1:8000"     # Windows PowerShell

streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

- **UI** → <http://localhost:8501>
- **API docs** → <http://localhost:8000/docs>
- **Health** → <http://localhost:8000/health>

`app/Home.py` probes `/health` on load. If the API is down it shows a clear
error and a read-only page rather than a stack trace.

---

## Command line

The package installs an `sse` console script.

```bash
pip install -e .
sse --help
```

| Command | Purpose |
| --- | --- |
| `sse version` | Version from installed package metadata |
| `sse config [key]` | Effective configuration; secret values redacted |
| `sse validate` | Check config, corpus, and both indexes. Never rebuilds |
| `sse build-index` | Build BM25 + FAISS indexes |
| `sse search "query"` | One search, printed to the terminal |
| `sse evaluate` | Metrics over the frozen evaluation dataset |
| `sse benchmark` | Latency/throughput measurement |
| `sse smoke-test` | End-to-end check of the whole pipeline |

```console
$ sse search "how does a muscle pump move blood" --top-k 5
$ sse validate
$ sse evaluate --strata A_restated A_verbatim A_typo
```

Exit codes: `0` success, `1` error, `2` usage, `3` artifacts not ready.

---

## Configuration

Settings resolve in this order (later wins):

1. `configs/default.yaml`
2. `configs/{development,production}.yaml`
3. Environment variables / `.env`
4. Explicit `--env` selection

Copy `.env.example` to `.env` and edit. **`.env` is gitignored — never commit
one.** Every variable is documented inline in that file.

Common environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SSE_ENV` | `development` | Config environment |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | `console` | `console` or `json` |
| `EMBEDDING_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `RRF_K` | `60` | RRF smoothing constant |
| `DEFAULT_TOP_K` | `10` | Default results returned |
| `STREAMLIT_API_BASE_URL` | `http://127.0.0.1:8000` | API the UI talks to |

Spell correction and synonym expansion are **off by default** — measured against
this corpus, they mangled out-of-corpus words about as often as they fixed real
typos. See `docs/evaluation_protocol.md`.

---

## Evaluation

The evaluation set is **frozen and hashed before any retrieval runs**, so no
query was tuned after seeing a ranking.

```bash
sse evaluate --strata A_restated A_verbatim A_typo
```

**Current honest status — the topical evaluation is incomplete.**

| Stratum | Queries | Human judgments | Status |
| --- | --- | --- | --- |
| `A_restated` | 25 | 25 | Evaluated |
| `A_verbatim` | 15 | 15 | Evaluated |
| `A_typo` | 10 | 10 | Evaluated |
| `B_topical` | 30 | **0** | **Blocked — pending independent human qrels** |

Stratum B has no human relevance judgments. Those queries are **excluded**
rather than scored against labels inferred from the retrieval system's own
output, which would be circular. Nothing in this repo fabricates qrels.

What stratum A measures: known-item retrieval on a 300-article corpus, which is
the *easiest* case for both arms — it is a **ceiling**, not a general
retrieval-quality figure. It does not generalise to another corpus, encoder,
corpus size, or to production traffic. See `docs/evaluation_protocol.md` for
the full protocol and its stated limits.

---

## Project layout

```
app/                     Streamlit UI (Home.py, components/, utils/)
src/semantic_search/
  api/                   FastAPI app, routes, middleware
  app/                   ApplicationSearchService — the search entry point
  query/                 Query processing (normalize, spell, synonyms)
  search/                BM25, semantic, hybrid searchers + RRF fusion
  index/                 BM25 and FAISS index builders
  data/                  Corpus loader and preprocessing
  embeddings/            Sentence-BERT encoder + cache
  explainability/        Per-retriever provenance
  evaluation/            Dataset, metrics, statistics, runner, report
  core/                  Config, exceptions, logging, rate limiting
  cli.py                 The `sse` entry point
configs/                 default / development / production YAML
scripts/                 download_data, build_index, evaluate, measure_*
data/                    Corpus, indexes, frozen evaluation set (gitignored)
docs/                    Architecture, API, evaluation, deployment
tests/                   979 unit + integration tests
```

---

## Development

```bash
pip install -e ".[dev]"

pytest                              # full suite
pytest -m "not slow"                # skip model-loading tests
ruff check src tests scripts app
black --check src tests scripts app
mypy src
```

Current state: **893 test functions (~979 tests by pytest collection), 6 skipped (current suite).**

Index files, the corpus, and the raw Wikipedia dump are gitignored — a clone
starts empty and rebuilds them with `scripts/build_index.py`.

---

## API reference

Full schema at `/docs` when the server is running.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Combined liveness + readiness |
| `GET` | `/health/live` | Process is up |
| `GET` | `/health/ready` | Indexes loaded and searchable |
| `POST` | `/search` | Search; returns hits with provenance |
| `POST` | `/search/explain` | Search with per-retriever explanation |
| `GET` | `/stats` | Corpus size, retriever names, index metadata |

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does a muscle pump move blood", "top_k": 5}'
```

---

## License

MIT. See `pyproject.toml`.
