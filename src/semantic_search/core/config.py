"""Central configuration for the semantic search engine.

Precedence (highest to lowest):

1. Explicit constructor arguments.
2. Environment variables (optionally loaded from a ``.env`` file).
3. A YAML configuration file (``configs/<environment>.yaml`` or an explicit path).
4. Field defaults declared on :class:`Settings`.

Every tunable in the roadmap -- paths, model names, batch size, index type,
top-K, BM25 parameters, RRF k, port, rate limit -- lives here. Nothing in the
codebase hard-codes these values.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from semantic_search.core.exceptions import ConfigurationError

Environment = Literal["development", "production", "testing"]
CorpusSize = Literal["small", "medium", "full"]
SearchMethod = Literal["bm25", "semantic", "hybrid"]
FaissIndexType = Literal["flat_ip", "flat_l2", "ivf_flat", "hnsw"]

#: Number of documents pulled from Wikipedia per corpus size.
CORPUS_SIZE_LIMITS: dict[str, int] = {
    "small": 2_000,
    "medium": 50_000,
    "full": 200_000,
}

#: Friendlier environment-variable names for nested fields. ``pydantic-settings``
#: binds environment variables to flat fields, so a nested field such as
#: ``embedding.model_name`` would otherwise only be settable as
#: ``EMBEDDING_MODEL_NAME``. These aliases give the shorter, more idiomatic
#: spelling documented in ``.env.example``. Both spellings work.
ENV_ALIASES: dict[str, str] = {
    "EMBEDDING_MODEL": "embedding.model_name",
    "EMBEDDING_CACHE_FILENAME": "embedding.cache_filename",
    "RRF_K": "hybrid.rrf_k",
    "DEFAULT_SEARCH_METHOD": "hybrid.default_method",
    "CANDIDATE_POOL": "hybrid.candidate_pool",
    "DEFAULT_TOP_K": "search.default_top_k",
    "MAX_TOP_K": "search.max_top_k",
    "SNIPPET_LENGTH": "search.snippet_length",
    "DATA_DIR": "data.data_dir",
    # The evaluation section would otherwise be spelled EVALUATION_DEPTH. The
    # short form matches the SSE_-prefix convention the API section already
    # uses (API_HOST, not APPLICATION_HOST), and the alias table is where that
    # decision has to be made explicit -- the names derived from the section
    # name are not negotiable.
    "EVAL_DEPTH": "evaluation.depth",
    "EVAL_RESAMPLES": "evaluation.resamples",
    "EVAL_SEED": "evaluation.seed",
    "EVAL_RUN_SPELL_EXPERIMENT": "evaluation.run_spell_experiment",
}


def project_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``).

    Falls back to the current working directory when the package is installed
    standalone, so the engine also runs outside a checkout.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


class DataConfig(BaseModel):
    """Where raw/processed corpora and built index artifacts live."""

    data_dir: Path = Path("data")
    raw_dirname: str = "raw"
    processed_dirname: str = "processed"
    index_dirname: str = "index"
    evaluation_dirname: str = "evaluation"
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    #: Text shorter than this (in characters) is treated as malformed and dropped.
    min_text_length: int = 50
    #: Text longer than this is truncated so one article cannot dominate BM25 length norm.
    max_text_length: int = 20_000
    #: Title text is prefixed to the embedded body; this caps its contribution.
    title_prefix_max_length: int = 120

    def _sub(self, name: str) -> Path:
        base = self.data_dir if self.data_dir.is_absolute() else project_root() / self.data_dir
        return base / name

    @property
    def raw_dir(self) -> Path:
        return self._sub(self.raw_dirname)

    @property
    def processed_dir(self) -> Path:
        return self._sub(self.processed_dirname)

    @property
    def index_dir(self) -> Path:
        return self._sub(self.index_dirname)

    @property
    def evaluation_dir(self) -> Path:
        return self._sub(self.evaluation_dirname)

    @property
    def corpus_file(self) -> Path:
        return self.processed_dir / "corpus.jsonl"

    @property
    def doc_store_file(self) -> Path:
        return self.index_dir / "documents.jsonl"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.index_dir, self.evaluation_dir):
            path.mkdir(parents=True, exist_ok=True)


class EmbeddingConfig(BaseModel):
    """Sentence-BERT encoder settings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 64
    max_seq_length: int = 256
    normalize: bool = True
    #: Cache directory for the persisted embedding matrix.
    cache_filename: str = "embeddings.npy"
    metadata_filename: str = "embeddings_meta.json"


class FaissConfig(BaseModel):
    """Vector index settings."""

    index_type: FaissIndexType = "flat_ip"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    ivf_nlist: int = 1024
    ivf_nprobe: int = 16
    index_filename: str = "faiss_index"


class BM25Config(BaseModel):
    """Okapi BM25 parameters and tokenizer behaviour."""

    k1: float = 1.5
    b: float = 0.75
    min_token_length: int = 2
    max_token_length: int = 40
    use_stemming: bool = True
    stopwords: str = "en"
    index_filename: str = "bm25_index.pkl"


class HybridConfig(BaseModel):
    """Rank-fusion settings."""

    rrf_k: int = 60
    default_method: SearchMethod = "hybrid"
    #: How many candidates each retriever contributes before fusion.
    candidate_pool: int = 100


class EvaluationConfig(BaseModel):
    """Settings for the retrieval-quality evaluation harness (Phase 10).

    These configure the *evaluation*, not the system under test. Nothing here
    changes a retrieval default: ``EVAL_DEPTH`` says how deep a single
    evaluation run retrieves, and no production code path reads this section.
    The distinction matters enough to be worth a separate config object -- a
    knob that lives beside ``bm25.k1`` and ``hybrid.rrf_k`` invites the
    assumption that turning it changes search behaviour, and it does not.

    The defaults are the registered protocol's values. They are configuration
    rather than constants only so a run can be reproduced at a different
    depth or with more resamples without editing source; changing them changes
    the experiment, and the report records the values it ran with.
    """

    #: Depth of the single retrieval run per (system, query). Every cutoff
    #: metric is computed from that one ranking, so P@5 and R@10 are always
    #: derived from the same run. Kept at or below ``hybrid.candidate_pool`` so
    #: the fused arm is not evaluated at a shallower pool than its own design
    #: intends.
    depth: int = Field(default=50, ge=1)
    #: Bootstrap resamples. 1,000 is visibly unstable at the endpoints; 10,000
    #: makes Monte-Carlo error small relative to the interval's own width.
    resamples: int = Field(default=10_000, ge=100)
    #: Seed for the bootstrap. Fixed by default so two runs of the same
    #: evaluation produce identical intervals, and reported in the artifacts so
    #: a differing number can be traced to a differing seed rather than to
    #: noise.
    seed: int = 20250930
    #: Whether the spell-correction arm runs. Off by default, and the flag
    #: exists so the experiment is opt-in rather than something a run does
    #: implicitly. Enabling it does **not** change ``query.spell_check`` for
    #: production: the arm builds its own service from a copied settings object.
    run_spell_experiment: bool = False


class QueryConfig(BaseModel):
    """Query-understanding pipeline toggles.

    A correction must clear **three** gates, not one. ``min_correction_score``
    alone cannot separate a typo from a proper noun -- on the real corpus,
    ``"Serkan"`` scores 66.7 against its nearest neighbour while ``"teh"`` (a
    definite typo) scores only 85.7, so any single threshold that catches every
    typo also rewrites names. The other two gates close that gap:

    ``min_edit_distance``
        Levenshtein distance from the typed term to the candidate. This is what
        rejects ``"hemoglobinn" -> "methemoglobinemia"`` (distance 7) despite a
        71.4 similarity, and what keeps a *correctly spelled* known word from
        ever being rewritten -- it resolves to itself at distance 0.

    ``min_candidate_df``
        How many documents must contain the candidate term. This is the
        proper-noun guard: a rare word's nearest neighbour is usually a
        coincidence, whereas a frequent word's is evidence of a typo. Measured
        on the 300-document corpus, ``machine`` has df 16 and ``engine`` 15,
        while the names that must be protected have df 1-6. Setting this to 0
        disables the guard, which is why it is constrained to at least 1.

    The three gates were calibrated against the real corpus vocabulary, which
    makes them a validated-on-this-corpus configuration and not a general
    guarantee; see the Phase 8 section of ``BUILD_STATUS.md``.

    The measured consequence is that ``spell_check`` ships **off**. Against the
    300-document corpus the corrector fixes real typos and also rewrites
    out-of-corpus real words it is unfamiliar with -- ``"tectonics" ->
    "tectonic"``, ``"turbine" -> "turin"``, ``"virus" -> "virtues"``. A wrong
    rewrite is worse than a missed correction, because the user gets confidently
    wrong results and no signal that anything changed, and no threshold and no
    frequency list was found that removes the failure without also removing the
    corrections. It is one flag away for deployments whose users mistype.
    """

    #: Master switch. When False the processor is bypassed entirely and every
    #: query reaches the retrievers exactly as typed.
    enabled: bool = True
    #: Off by default -- see the class docstring for the measurement.
    spell_check: bool = False
    #: rapidfuzz similarity (0-100) below which a correction is rejected.
    min_correction_score: float = Field(default=80.0, ge=0.0, le=100.0)
    #: Maximum Levenshtein distance a candidate may be from the typed term.
    min_edit_distance: int = Field(default=2, ge=0)
    #: Minimum number of documents that must contain the candidate term.
    min_candidate_df: int = Field(default=3, ge=1)
    #: Record-only by default. Expansions are computed and reported, but the
    #: retrieval query is left alone unless this is explicitly turned on -- see
    #: the Phase 8 section for why an unmeasured rewrite is not the safe default.
    synonym_expansion: bool = False
    max_length: int = 512
    preserve_proper_nouns: bool = True
    #: Refuse to correct a term the tokenizer treats as a stopword, in either
    #: direction. A stopword is missing from the corpus vocabulary because the
    #: tokenizer strips it, not because the corpus lacks it, so without this gate
    #: a correct query can have ``"and"`` rewritten to a rare article title --
    #: measured on the 300-document corpus, and it moved documents in and out of
    #: the top 5. Refusing is strictly safe, since the retriever discards
    #: stopwords anyway. The cost is that stopword typos are never corrected.
    protect_stopwords: bool = True


class SearchConfig(BaseModel):
    """Result shaping defaults."""

    default_top_k: int = 10
    max_top_k: int = 100
    snippet_length: int = 320
    default_filters: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Settings for the application-level search service (Phase 9).

    These are the knobs the *caller* can vary per request or per deployment,
    as distinct from the retriever-level knobs above. They deliberately carry
    the same defaults as the sections they mirror -- ``search.default_top_k``,
    ``search.max_top_k``, ``query.max_length``, ``query.enabled`` -- so that
    adding this section changes no existing behaviour.

    The duplication is deliberate and bounded to four integers and one flag.
    The alternative, reading ``search.max_top_k`` from here, would couple the
    application layer to the retrieval layer's config object and make it
    impossible to expose an application-level limit that differs from the
    internal one. Which is the entire point of an application boundary: a UI
    asking for 1000 results per page should hit *its own* cap, not silently
    inherit whichever cap the retriever happened to have.
    """

    #: Results returned when a request does not ask for a specific count.
    default_top_k: int = Field(default=10, ge=1)
    #: Hard ceiling on ``top_k``. Exceeding it clamps rather than errors: asking
    #: for more than the corpus holds is a reasonable user action, and the
    #: response reports the clamp in ``metadata.requested_top_k`` so it is
    #: visible rather than silent.
    max_top_k: int = Field(default=100, ge=1)
    #: Whether query processing runs by default. A per-request
    #: ``SearchRequest`` may override it in either direction, which is what
    #: makes the Phase 8 ON/OFF comparison a runtime experiment.
    query_processing_enabled: bool = True
    #: Longest accepted query, in characters. Mirrors ``query.max_length``.
    max_query_chars: int = Field(default=512, ge=1)


class APIConfig(BaseModel):
    """HTTP service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    rate_limit_enabled: bool = True
    cache_size: int = 512
    cache_ttl_seconds: int = 300
    max_payload_bytes: int = 256 * 1024
    expose_stack_traces: bool = False
    log_queries: bool = False
    title: str = "Semantic Search Engine API"
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class Settings(BaseSettings):
    """Top-level settings object assembled from YAML + environment.

    Top-level fields use the ``SSE_`` prefix (``SSE_ENV``). Nested sections are
    populated by :func:`_apply_nested_env`, which reads unprefixed names such as
    ``EMBEDDING_MODEL`` or ``BM25_K1`` -- see ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Semantic Search Engine"
    version: str = "0.1.0"
    environment: Environment = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    log_queries: bool = False
    cache_backend: Literal["memory", "disk"] = "memory"
    corpus_size: CorpusSize = "small"
    streamlit_api_base_url: str = "http://localhost:8000"

    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def corpus_limit(self) -> int:
        return CORPUS_SIZE_LIMITS[self.corpus_size]

    def ensure_dirs(self) -> None:
        self.data.ensure_dirs()

    def to_dict(self) -> dict[str, Any]:
        """Serialise settings, resolving Path objects so the result is JSON-safe."""
        raw = self.model_dump(mode="json")
        round_tripped: dict[str, Any] = json.loads(json.dumps(raw))
        return round_tripped


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping at the top level.")
    return loaded


def _config_candidates(environment: str) -> list[Path]:
    """Config files for an environment, in increasing order of precedence."""
    root = project_root()
    return [root / "configs" / "default.yaml", root / "configs" / f"{environment}.yaml"]


def _apply_nested_env(settings: Settings) -> Settings:
    """Overlay environment variables onto the nested config sections.

    ``pydantic-settings`` binds environment variables to flat top-level fields
    only. This pass populates the nested sections, accepting three spellings per
    field so the names in ``.env.example`` all work:

    * ``BM25_K1``            -> ``bm25.k1``           (section + field, implicit)
    * ``BM25_K1`` via alias  -> ``hybrid.rrf_k``      (explicit alias)
    * ``SSE_BM25_K1``        -> ``bm25.k1``           (explicit prefix)

    The candidate names are derived from the model fields rather than hard-coded,
    so adding a field to a section automatically makes it environment-configurable.
    """
    sections: dict[str, dict[str, Any]] = {
        "data": settings.data.model_dump(),
        "embedding": settings.embedding.model_dump(),
        "faiss": settings.faiss.model_dump(),
        "bm25": settings.bm25.model_dump(),
        "hybrid": settings.hybrid.model_dump(),
        "query": settings.query.model_dump(),
        "search": settings.search.model_dump(),
        "app": settings.app.model_dump(),
        "api": settings.api.model_dump(),
        "evaluation": settings.evaluation.model_dump(),
    }

    # env variable name -> "section.field" target
    resolved: dict[str, str] = {}
    for section, fields in sections.items():
        for field_name in fields:
            resolved[f"{section}_{field_name}".lower()] = f"{section}.{field_name}"
            resolved[f"sse_{section}_{field_name}".lower()] = f"{section}.{field_name}"
    for env_name, target in ENV_ALIASES.items():
        resolved[env_name.lower()] = target

    # Collect the raw strings, then let pydantic coerce them.
    pending: dict[str, dict[str, str]] = {name: {} for name in sections}
    for env_key, raw_value in os.environ.items():
        match = resolved.get(env_key.lower())
        if match is None:
            continue
        section, _, field_name = match.partition(".")
        if field_name in sections[section]:
            pending[section][field_name] = raw_value

    for section, values in pending.items():
        if not values:
            continue
        current = getattr(settings, section)
        try:
            setattr(
                settings, section, type(current).model_validate({**current.model_dump(), **values})
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid environment override for section '{section}': {exc}"
            ) from exc
    return settings


def load_settings(
    environment: str | None = None,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build a :class:`Settings` instance.

    Args:
        environment: ``development`` / ``production`` / ``testing``. Defaults to
            the ``SSE_ENV`` environment variable, then ``development``.
        config_path: Explicit YAML file that replaces the layered lookup.
        **overrides: Overrides applied to the assembled model, e.g.
            ``api=APIConfig(port=9000)``.
    """
    env = environment or os.getenv("SSE_ENV", "development")

    if config_path:
        payload: dict[str, Any] = dict(_read_yaml(Path(config_path)))
    else:
        # Layered: configs/default.yaml provides the base, and the
        # environment-specific file is merged on top so it wins.
        payload = _read_yaml(project_root() / "configs" / "default.yaml")
        payload = _deep_merge(payload, _read_yaml(project_root() / "configs" / f"{env}.yaml"))

    # The resolved environment name always wins over whatever the YAML declared,
    # so a single `configs/*.yaml` can be reused across environments.
    payload["environment"] = env
    settings = Settings(**payload)
    settings = _apply_nested_env(settings)
    if overrides:
        settings = settings.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    return settings


@lru_cache(maxsize=8)
def get_settings(environment: str | None = None) -> Settings:
    """Cached accessor used across the app (import once, reuse everywhere)."""
    return load_settings(environment=environment)


def reset_settings_cache() -> None:
    """Clear the cached settings; used by tests that mutate the environment."""
    get_settings.cache_clear()
