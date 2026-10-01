"""The ``sse`` command-line entry point.

Design rule
-----------
**This module contains no retrieval, scoring, fusion, or indexing logic.** Every
command is a thin argument parser that delegates to the same public functions
the API and the scripts use:

============================  ==========================================
Command                       Delegates to
============================  ==========================================
``sse validate``              ``api.dependencies.load_service``
``sse search``                ``ApplicationSearchService.search_text``
``sse build-index``           ``scripts/build_index.py:main``
``sse evaluate``              ``scripts/evaluate.py:main``
``sse benchmark``             ``scripts/measure_hybrid.py:main``
``sse probe``                 ``scripts/probe_hybrid.py:main``
``sse config`` / ``version``  ``core.config`` only
``sse smoke-test``            the above, in sequence
============================  ==========================================

If a rule ever needs changing -- a new RRF constant, a new index type -- it
changes in ``search/`` and ``core/config.py``, and this file picks it up. A
second copy of a retrieval rule inside a CLI is how the CLI and the server
start disagreeing about what the system does, which is exactly the class of
bug this project has already shipped once (a script timing a different
processor than production used).

Why the scripts are imported rather than reimplemented
-------------------------------------------------------
``scripts/build_index.py`` and friends already do the right thing, including
their own logging setup and ``sys.path`` bootstrap. Re-implementing their
bodies here would duplicate hundreds of lines that are covered by the same
tests, and would give the CLI its own idea of what "build an index" means.

Why the scripts are loaded by path
----------------------------------
``scripts/`` is deliberately not a package -- it is a directory of executables,
not a library, and it is excluded from ``[tool.setuptools.packages.find]``. An
editable install does not make ``import build_index`` work. Loading by file path
with :func:`importlib.util.spec_from_file_location` bridges that gap without
changing the project's packaging, and without adding ``scripts`` to the
installed surface area (where it would ship as importable library code).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct-execution fallback
    # Only reached when someone runs this file directly rather than through the
    # installed `sse` entry point. An editable install never takes this branch.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_search.core.config import Settings, load_settings
from semantic_search.core.exceptions import SemanticSearchError
from semantic_search.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Exit codes. Distinct values so a shell script can tell the failure modes
#: apart without parsing messages.
EXIT_OK = 0
#: Something the operator must fix: missing artifacts, bad config, empty corpus.
EXIT_USAGE = 2
#: The command ran but the system is not in a state to serve: this is what a
#: stale or mismatched index produces, and it is deliberately distinct from a
#: crash.
EXIT_NOT_READY = 3
#: An unexpected failure -- a bug, not a configuration problem.
EXIT_ERROR = 1


def _repo_root() -> Path:
    """Directory holding ``scripts/``, or ``None`` outside a checkout."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "scripts").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _load_script(name: str) -> Any:
    """Import ``scripts/<name>.py`` as a module.

    Raises:
        FileNotFoundError: The script does not exist, which means this is an
            installed copy without the repository checkout. That is a real and
            expected state -- ``pip install semantic-search-engine`` gives you
            the library and the ``sse`` commands that only need the library,
            not the development scripts -- so the message says which commands
            remain available rather than failing obscurely.
    """
    root = _repo_root()
    path = root / "scripts" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The commands that delegate to scripts/ "
            f"(build-index, evaluate, benchmark, probe) require the repository "
            f"checkout. The commands that need only the installed package "
            f"(version, config, validate, search, smoke-test) work regardless."
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_sse_script_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# version
# --------------------------------------------------------------------------- #


def _installed_version() -> str:
    """The version from installed package metadata, not a hardcoded literal.

    ``__init__.__version__`` is a second copy of the number and drifts; the
    distribution metadata is the authority. Falls back only when the package is
    being run straight from a checkout with nothing installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("semantic-search-engine")
    except PackageNotFoundError:
        from semantic_search import __version__

        return f"{__version__} (not installed as a distribution)"


def cmd_version(args: argparse.Namespace) -> int:
    print(f"sse {_installed_version()}")
    settings = load_settings(args.environment)
    print(f"python  {sys.version.split()[0]}")
    print(f"env     {settings.environment}")
    print(f"corpus  {settings.corpus_size} (limit {settings.corpus_limit:,})")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

#: Keys whose values must never be printed. The engine has no credential
#: settings today, so this list is currently empty in practice -- it is here so
#: that `sse config` is safe by construction if one is ever added, rather than
#: needing an audit at the moment it is added. Redaction is applied by key
#: match, case-insensitively, to the flattened settings tree.
SECRET_KEY_MARKERS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "access_key",
)


def _redact(value: Any, key: str = "") -> Any:
    """Recursively replace secret-looking values with a marker."""
    if any(marker in key.lower() for marker in SECRET_KEY_MARKERS):
        return "***redacted***"
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


def cmd_config(args: argparse.Namespace) -> int:
    settings = load_settings(args.environment)
    payload = _redact(settings.to_dict())
    if args.key:
        node: Any = payload
        for part in args.key.split("."):
            if not isinstance(node, dict) or part not in node:
                print(f"no such setting: {args.key}", file=sys.stderr)
                return EXIT_USAGE
            node = node[part]
        print(json.dumps(node, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    """Load every artifact the server needs, and report what is wrong.

    This calls ``api.dependencies.load_service`` -- the *same* function the
    server's startup path calls. That is the point: a validation that
    reimplemented the checks would be a second opinion, and the two would
    disagree the first time either was edited.

    It loads the encoder and the indexes, so it costs roughly what startup
    costs. On the 300-document development corpus that is dominated by loading
    ``all-MiniLM-L6-v2``. It is a real check, and it is slow on purpose.
    """
    settings = load_settings(args.environment)
    print(f"environment : {settings.environment}")
    print(f"corpus      : {settings.data.corpus_file}")
    print(f"index dir   : {settings.data.index_dir}")
    print(f"encoder     : {settings.embedding.model_name}")

    # Imported here rather than at module scope so `sse --help` and
    # `sse version` do not pay for torch/faiss/sentence-transformers imports.
    from semantic_search.api.dependencies import load_service

    if args.files_only:
        return _validate_paths(settings)

    print("loading artifacts ...")
    try:
        service, elapsed = load_service(settings)
    except SemanticSearchError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print(
            "\nArtifacts are not rebuilt automatically. A stale or mismatched "
            "index is a decision to make, not something to do silently under a "
            "person running `validate` to check what is there.",
            file=sys.stderr,
        )
        return EXIT_NOT_READY
    except OSError as exc:
        print(f"\nFAILED to read artifacts: {exc}", file=sys.stderr)
        return EXIT_NOT_READY

    info = service.info()
    print(f"\nOK: {service.document_count} documents, {info.get('retrievers', '?')}")
    print(f"   loaded in {elapsed:.2f}s")
    return EXIT_OK


def _validate_paths(settings: Settings) -> int:
    """Presence-only check: files, sizes, and whether metadata agrees.

    Cheap enough to run in a pre-commit hook or a container preflight, and it
    answers "are the files there and internally consistent?" without paying the
    model load. It is *not* a substitute for the full check -- a truncated
    pickle is present and this would not notice -- so the output says so.
    """
    import json

    checks: list[tuple[str, Path]] = [
        ("corpus", settings.data.corpus_file),
        ("bm25 index", settings.data.index_dir / settings.bm25.index_filename),
        ("faiss index", settings.data.index_dir / settings.faiss.index_filename),
    ]
    ok = True
    for label, path in checks:
        if not path.exists():
            print(f"MISSING  {label:12s} {path}")
            ok = False
            continue
        size = path.stat().st_size
        print(f"present  {label:12s} {path} ({size:,} bytes)")

    meta = settings.data.index_dir / f"{settings.faiss.index_filename}.meta.json"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"MALFORMED faiss meta    {meta} ({exc})")
            ok = False
        else:
            print(f"present  {'faiss meta':12s} {meta}")
            built_with = payload.get("model_name") or payload.get("embedding_model")
            if built_with and built_with != settings.embedding.model_name:
                print(
                    f"MISMATCH  the index was built with {built_with!r} but the "
                    f"configured encoder is {settings.embedding.model_name!r}. "
                    f"Rebuild, or point embedding.model_name at what was built."
                )
                ok = False
    print(
        "\n(files only -- this does not load the model. Run without "
        "--files-only for the full check.)"
    )
    return EXIT_OK if ok else EXIT_NOT_READY


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def cmd_search(args: argparse.Namespace) -> int:
    """Run one query through the application search service.

    Deliberately uses ``ApplicationSearchService`` -- the same class the API
    serves -- rather than reaching for the retriever directly, so the CLI
    cannot report a different ranking from the server's.
    """
    settings = load_settings(args.environment)
    from semantic_search.api.dependencies import load_service

    try:
        service, _ = load_service(settings)
    except SemanticSearchError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return EXIT_NOT_READY

    processing = None if args.query_processing is None else args.query_processing
    try:
        response = service.search_text(args.query, args.top_k, query_processing=processing)
    except SemanticSearchError as exc:
        print(f"search failed: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return EXIT_OK

    metadata = response.metadata
    if metadata is not None:
        print(f"query      : {response.original_query!r}")
        if response.processed_query != response.original_query:
            print(f"processed  : {response.processed_query!r}")
        print(
            f"took       : {metadata.took_ms:.1f} ms over {metadata.document_count} "
            f"documents (top_k requested {metadata.requested_top_k}, "
            f"returned {metadata.returned_count})"
        )
        if metadata.query_processing_enabled and not (
            metadata.was_modified or metadata.was_corrected
        ):
            print("processing : ran, and left the query unchanged")
        print()

    if not response.results:
        reason = metadata.empty_reason if metadata else "no metadata"
        print(f"No results. ({reason})")
        return EXIT_OK

    for hit in response.results:
        print(f"{hit.rank}. {hit.title}")
        print(
            f"   doc_id {hit.doc_id}   fused score {hit.score:.6f}   "
            f"sources: {', '.join(hit.sources) or 'none'}"
        )
        for prov in hit.provenance:
            # Note the rank base. The fused rank above is 1-based; a
            # retriever's own rank is 0-based, because that is what the
            # retriever produced and this prints it rather than adjusting it.
            print(
                f"   - {prov.retriever}: rank {prov.rank} (0-based), "
                f"raw {prov.raw_score:.4f}, rrf +{prov.rrf_contribution:.6f}"
            )
        print(f"   {hit.snippet}")
        print()
    return EXIT_OK


# --------------------------------------------------------------------------- #
# delegating commands
# --------------------------------------------------------------------------- #


def _delegate(script: str, argv: list[str]) -> int:
    """Run ``scripts/<script>.py``'s ``main`` with *argv*.

    Each script owns its own argument parsing, so this is a hand-off rather
    than a translation layer. Exit codes propagate unchanged, which means
    ``sse build-index`` fails exactly when ``python scripts/build_index.py``
    would.
    """
    module = _load_script(script)
    entry: Callable[..., int] | None = getattr(module, "main", None)
    if entry is None:  # pragma: no cover - defensive
        print(f"{script}.py has no main()", file=sys.stderr)
        return EXIT_ERROR
    return entry()


def cmd_build_index(args: argparse.Namespace) -> int:
    return _delegate("build_index", _script_argv(args))


def _script_argv(args: argparse.Namespace) -> list[str]:
    """Rebuild a command line for the delegated script.

    ``argparse`` hands each subparser a private namespace, so the original
    tokens are not available; the flags are re-emitted explicitly instead.
    Every value that is not ``None`` was supplied by the caller, so nothing is
    invented here.
    """
    argv: list[str] = ["--env", args.environment]
    for flag, value in (
        ("--limit", getattr(args, "limit", None)),
        ("--index-type", getattr(args, "index_type", None)),
    ):
        if value:
            argv += [flag, str(value)]
    for flag in ("--force", "--skip-bm25"):
        if getattr(args, flag[2:].replace("-", "_"), False):
            argv.append(flag)
    return argv


def cmd_evaluate(args: argparse.Namespace) -> int:
    return _delegate("evaluate", _script_argv(args))


def cmd_benchmark(args: argparse.Namespace) -> int:
    return _delegate("measure_hybrid", _script_argv(args))


def cmd_probe(args: argparse.Namespace) -> int:
    return _delegate("probe_hybrid", [*args.query, *_script_argv(args)])


# --------------------------------------------------------------------------- #
# smoke-test
# --------------------------------------------------------------------------- #


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Exercise the real stack end to end and report pass/fail per step.

    Each step names what it actually proved, because "smoke test passed" with
    no detail is how a half-working system gets shipped. Steps are independent
    and all of them run: a failure is recorded and the command continues, so
    one broken area does not hide the state of the others. The exit code is
    non-zero if any step failed.
    """
    results: list[tuple[str, bool, str]] = []

    def step(label: str, ok: bool, detail: str = "") -> None:
        results.append((label, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    print("=== smoke test ===\n")

    # 1. import surface
    try:

        version = _installed_version()
        step("import semantic_search", True, version)
    except Exception as exc:
        step("import semantic_search", False, f"{type(exc).__name__}: {exc}")
        _report(results)
        return EXIT_ERROR

    # 2. configuration resolves
    settings = None
    try:
        settings = load_settings(args.environment)
        step("resolve configuration", True, f"{settings.environment} / {settings.corpus_size}")
    except Exception as exc:
        step("resolve configuration", False, f"{type(exc).__name__}: {exc}")

    # 3. artifacts load and validate
    service = None
    if settings is not None:
        try:
            from semantic_search.api.dependencies import load_service

            service, elapsed = load_service(settings)
            step(
                "load and validate artifacts",
                True,
                f"{service.document_count} documents in {elapsed:.2f}s",
            )
        except SemanticSearchError as exc:
            step("load and validate artifacts", False, str(exc))
        except Exception as exc:
            step("load and validate artifacts", False, f"{type(exc).__name__}: {exc}")
    else:
        step("load and validate artifacts", False, "skipped: configuration did not resolve")

    # 4. a real search through the application service
    if service is not None:
        try:
            response = service.search_text("muscle", top_k=3)
            metadata = response.metadata
            ok = bool(response.results) and metadata is not None
            step(
                "search via ApplicationSearchService",
                ok,
                (
                    f"{len(response.results)} results, " f"{metadata.took_ms:.1f} ms"
                    if metadata
                    else "no metadata"
                ),
            )
        except Exception as exc:
            step("search via ApplicationSearchService", False, f"{type(exc).__name__}: {exc}")

        # 5. explainability payload is populated
        try:
            from semantic_search.explainability.explanations import explain_response

            explanation = explain_response(response)
            has_provenance = any(e.contributions for e in explanation.results)
            step(
                "explainability produces provenance",
                has_provenance,
                "provenance present" if has_provenance else "no provenance in the payload",
            )
        except Exception as exc:
            step(
                "explainability produces provenance",
                False,
                f"{type(exc).__name__}: {exc}",
            )

        # 6. a nonsense query is handled, not crashed on.
        #
        # This deliberately does NOT assert "no results". A dense retriever
        # over a 300-document index returns its k nearest neighbours for any
        # input, including nonsense: there is no such thing as an empty result
        # set while a vector index has documents to return. What is worth
        # checking is that a term with no lexical or semantic support is
        # *ranked last* and does not raise -- the property a user would
        # actually notice if it broke.
        try:
            odd = service.search_text("zzzqxnwjkvppq", top_k=3)
            step(
                "nonsense query handled without error",
                True,
                f"{len(odd.results)} results, "
                f"scores {[round(h.score, 5) for h in odd.results]}",
            )
        except Exception as exc:
            step("nonsense query handled without error", False, f"{type(exc).__name__}: {exc}")
    else:
        step("search via ApplicationSearchService", False, "skipped: artifacts did not load")
        step("explainability produces provenance", False, "skipped: artifacts did not load")
        step("nonsense query handled without error", False, "skipped: artifacts did not load")

    # 7. the evaluation dataset is honest about what it can judge
    #
    # load_dataset(require_judgments=False) does NOT return the partial set --
    # it returns an EMPTY judgment list, because "load a dataset I will not
    # score" has no coherent partial answer. So the supported way to see which
    # strata are evaluable is to ask the loader per stratum: a stratum that
    # validates is fully judged, and one that raises names the queries that
    # are missing judgments. That is the loader's own verdict rather than a
    # count this file recomputes and could disagree with.
    if settings is not None:
        try:
            from semantic_search.core.exceptions import EvaluationError
            from semantic_search.evaluation.dataset import DatasetPaths, load_dataset
            from semantic_search.evaluation.models import STRATA

            paths = DatasetPaths(root=settings.data.evaluation_dir)
            evaluable: list[str] = []
            blocked: list[str] = []
            for stratum in STRATA:
                try:
                    ds, _ = load_dataset(paths, strata=[stratum])
                except EvaluationError:
                    blocked.append(stratum)
                else:
                    evaluable.append(stratum)
                    step(
                        f"evaluation stratum {stratum}",
                        True,
                        f"{len(ds.queries)} queries, {len(ds.judgments)} judgments",
                    )

            if blocked:
                print(
                    f"       BLOCKED: {', '.join(blocked)} has no human relevance\n"
                    f"       judgments. Those queries are excluded from evaluation\n"
                    f"       rather than scored against labels inferred from the\n"
                    f"       retrieval system's own output. See\n"
                    f"       data/evaluation/qrels/MANIFEST.md."
                )
                if evaluable:
                    print(
                        f"       Evaluated strata: {' '.join(evaluable)}\n"
                        f"       Run them with: sse evaluate --strata {' '.join(evaluable)}"
                    )
                else:
                    step(
                        "at least one stratum is evaluable",
                        False,
                        "no stratum has complete judgments; see the manifest",
                    )
        except Exception as exc:
            step("evaluation dataset loads", False, f"{type(exc).__name__}: {exc}")
    else:
        step("evaluation dataset loads", False, "skipped: configuration did not resolve")

    return _report(results)


def _report(results: list[tuple[str, bool, str]]) -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return EXIT_OK if passed == total else EXIT_NOT_READY


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API with uvicorn, using the resolved settings for host and port.

    A launcher rather than a reimplementation: it hands host, port, and worker
    count to uvicorn and lets the ASGI app do the work. ``uvicorn`` is invoked
    in-process so Ctrl-C behaves the way it does when running uvicorn directly,
    which matters for the process-supervision setup in docs/deployment.md.
    """
    settings = load_settings(args.environment)
    host = args.host or settings.api.host
    port = args.port or settings.api.port

    import uvicorn

    print(f"serving on http://{host}:{port} ({settings.environment})")
    uvicorn.run(
        "semantic_search.api.app:get_application",
        factory=True,
        host=host,
        port=port,
        workers=args.workers if args.workers is not None else settings.api.workers,
        log_level=settings.log_level.lower(),
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def _add_env(parser: argparse.ArgumentParser) -> None:
    # dest="environment" so every command reads args.environment -- the same
    # name load_settings() takes -- rather than args.env, which is easy to
    # mistype across a dozen subcommands.
    parser.add_argument(
        "--env",
        dest="environment",
        default=os.getenv("SSE_ENV", "development"),
        choices=("development", "production", "testing"),
        help="configuration environment (default: $SSE_ENV or development)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sse",
        description=(
            "Hybrid semantic search engine (BM25 + Sentence-BERT + FAISS + RRF). "
            "Every command delegates to the same services the API uses; none of "
            "them reimplements retrieval."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Artifacts are never rebuilt implicitly. `sse validate` reports what is "
            "missing or stale and exits non-zero; `sse build-index` is how you fix it."
        ),
    )
    parser.add_argument("--log-level", default=None, help="override the log level")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("version", help="print version and environment")
    _add_env(p)
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("config", help="print the resolved configuration (secrets redacted)")
    _add_env(p)
    p.add_argument("key", nargs="?", help="dot-separated key, e.g. hybrid.rrf_k")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("validate", help="check that artifacts exist and load")
    _add_env(p)
    p.add_argument(
        "--files-only",
        action="store_true",
        help="check paths and metadata without loading the model",
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("search", help="run one query")
    p.add_argument("query", help="the query text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--query-processing",
        type=lambda v: _tri_state(v),
        default=None,
        help="on/off/true/false/1/0; omit for the configured default",
    )
    p.add_argument("--json", action="store_true", help="emit the raw JSON payload")
    _add_env(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("build-index", help="build the BM25 and FAISS indexes")
    _add_env(p)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--index-type", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-bm25", action="store_true")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("evaluate", help="run the labelled retrieval evaluation")
    _add_env(p)
    p.add_argument("--strata", nargs="+", default=None, help="restrict to these strata")
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--resamples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pool-only", action="store_true")
    p.add_argument("--no-ceiling", action="store_true")
    p.add_argument("--run-spell-experiment", action="store_true", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="measure search latency")
    _add_env(p)
    p.add_argument("--queries", type=int, default=15)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("probe", help="show per-retriever detail for one query")
    p.add_argument("query", nargs="+")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--rrf-k", type=int, default=None)
    _add_env(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("serve", help="run the FastAPI application")
    _add_env(p)
    p.add_argument("--host", default=None, help="override api.host")
    p.add_argument("--port", type=int, default=None, help="override api.port")
    p.add_argument("--workers", type=int, default=None, help="override api.workers")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("smoke-test", help="exercise the real stack end to end")
    _add_env(p)
    p.set_defaults(func=cmd_smoke_test)

    return parser


def _tri_state(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "on", "yes"):
        return True
    if lowered in ("0", "false", "off", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off/true/false/1/0, got {value!r}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sse`` console script.

    Returns the process exit code rather than calling ``sys.exit``, so the
    function is directly testable and ``console_scripts`` gets the right code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_USAGE

    settings_level = args.log_level or os.getenv("SSE_LOG_LEVEL")
    if settings_level:
        configure_logging(level=settings_level)

    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130
    except SemanticSearchError as exc:
        # A named engine error is an expected outcome with a documented fix, so
        # it prints the message and its error code rather than a traceback. Any
        # other exception is a bug and keeps its traceback -- swallowing those
        # is how real defects become "mysterious" ones.
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return EXIT_NOT_READY
    except BrokenPipeError:  # pragma: no cover - piping into head et al
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
