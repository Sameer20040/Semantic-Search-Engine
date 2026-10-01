"""Startup and import validation for the Streamlit app.

Why this exists
---------------
``app/Home.py`` and ``app/components/render.py`` are presentation code: nearly
every statement calls ``st.something``, which means they cannot be unit tested
without a browser. What *can* be checked without one is whether they are
importable at all -- and that is worth a great deal, because the two ways a
Streamlit app breaks silently are a bad ``sys.path`` setup and a renamed
import, and both surface only as a blank page in the browser.

``streamlit run`` executes the target file as ``__main__``, not as a module, so
importing it here would execute the whole page including ``st.stop()``. This
file therefore checks that the modules **import** without executing their
bodies' side effects:

* the modules are located and read by the AST, not executed;
* every name each module imports is actually defined somewhere reachable;
* the page's ``streamlit`` invocation names a path that exists.

What this does **not** verify: rendering, widget behaviour, a live search, or
that Streamlit can bind its port. Those need a running backend and a browser
and are described as a manual procedure in ``docs/deployment.md``. Nothing here
should be read as "the UI was seen working".
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = _ROOT / "app"
HOME = APP_DIR / "Home.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _module_paths() -> list[Path]:
    return sorted(p for p in APP_DIR.rglob("*.py"))


def _local_imports(path: Path) -> set[str]:
    """Names this module pulls from the ``app`` package.

    Only ``from app.* import X`` is considered. Third-party and stdlib names are
    excluded on purpose: the question this answers is "did someone rename
    something inside this project", and padding the candidate set with every
    builtin would make the check unable to fail for most mistakes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "app":
            names |= {alias.asname or alias.name for alias in node.names}
    return names


def _defined_names(path: Path) -> set[str]:
    """Top-level names a module defines."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class TestLayout:
    def test_the_entry_point_exists(self) -> None:
        # `streamlit run app/Home.py` is the documented command. If this file
        # is renamed, that command in docs/deployment.md and README.md stops
        # working, and nothing else in the repository would notice.
        assert HOME.exists(), f"the documented entry point {HOME} is missing"

    def test_the_components_and_utils_packages_exist(self) -> None:
        assert (APP_DIR / "components" / "__init__.py").exists()
        assert (APP_DIR / "utils" / "__init__.py").exists()

    @pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
    def test_every_module_parses(self, path: Path) -> None:
        # A syntax error in a Streamlit file is invisible until the page is
        # opened in a browser.
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestImportsResolve:
    @pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
    def test_local_imports_exist(self, path: Path) -> None:
        """Every ``from app.* import X`` names something this repo defines."""
        available: set[str] = set()
        for other in _module_paths():
            available |= _defined_names(other)

        missing = sorted(_local_imports(path) - available)
        assert missing == [], f"{path.name} imports names nothing defines: {missing}"

    def test_every_rendered_helper_is_imported_by_the_page(self) -> None:
        """A helper the page calls but did not import is a NameError in the browser.

        Only ``render_*``/``build_*``/``describe_*`` are checked -- the names
        whose absence would happen at render time, deep into a page the author
        has not reloaded since the rename.
        """
        tree = ast.parse(HOME.read_text(encoding="utf-8"), filename=str(HOME))
        imported = _local_imports(HOME)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith(("render_", "build_", "describe_"))
        }
        assert called, "the page should be calling at least one renderer"
        assert called - imported == set()


class TestDocumentedCommandIsValid:
    def test_the_readme_style_command_names_a_real_file(self) -> None:
        # docs/deployment.md, README.md and .env.example all say
        # `streamlit run app/Home.py`. This asserts the path half of it.
        assert HOME.is_file()

    def test_the_home_page_registers_a_page_config(self) -> None:
        # Without set_page_config the default title is the file path and the
        # layout is fixed-width; it is also the first thing the page runs, so
        # its absence would be a visible regression.
        assert "st.set_page_config(" in HOME.read_text(encoding="utf-8")


class TestNoBackendUrlIsHardcoded:
    """The backend address must come from configuration.

    Rule F: no hardcoded deployment assumptions. A UI with a URL baked in
    cannot be pointed at a staging host without editing source.
    """

    def test_no_literal_http_url_other_than_the_loopback_default(self) -> None:
        offenders: list[str] = []
        for path in _module_paths():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "http://" not in line and "https://" not in line:
                    continue
                # The documented loopback fallback and the prose in comments
                # that tell an operator what to run are allowed; anything
                # pointing at a named host is not.
                if "localhost" in line or "127.0.0.1" in line or line.strip().startswith("#"):
                    continue
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert offenders == [], f"hardcoded backend URLs in the frontend: {offenders}"

    def test_the_base_url_is_read_from_the_environment(self) -> None:
        api_source = (APP_DIR / "utils" / "api.py").read_text(encoding="utf-8")
        assert "os.environ.get(" in api_source
        assert "API_BASE_URL_ENV" in api_source
