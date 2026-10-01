"""Execute the real Streamlit page against the real API, with no browser.

Why this is a test and not a browser test
-----------------------------------------
``app/Home.py`` runs top-to-bottom on every Streamlit rerun, so almost any
defect in it -- a renamed import, a typo'd widget kwarg, a view model read from
a field the API does not send -- is a ``NameError`` or ``TypeError`` on the
first render. In a browser that is a blank page and a stack trace in the
terminal nobody is watching. Here it is an assertion.

The approach: replace the ``streamlit`` module with a recorder that returns
context managers and typed values the way the real one does, then ``runpy`` the
page. Every ``st.*`` call is recorded, so the test can also assert on *what was
drawn* -- that ten results were rendered, that the first one is numbered 1,
that a live backend produces the ready state rather than an error.

What this does **not** verify: pixels, CSS, widget interactivity, or that
Streamlit binds its port. Those need a browser. See ``docs/deployment.md`` for
the manual procedure and for what remains unverified.

The fixture defaults to a stubbed service so the test is deterministic and does
not need the real 300 MB model. The ``live`` case -- which really does search
the running backend -- is opt-in and skips unless ``SSE_TEST_LIVE_API`` names a
base URL, because a CI runner has no server and a test that fails for want of
one is worse than no test.
"""

from __future__ import annotations

import os
import runpy
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
HOME = _ROOT / "app" / "Home.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Query the page searches. A real one, so a live run exercises retrieval.
QUERY = "how does a muscle pump move blood"


class _Recorder:
    """Records every ``st.*`` call and returns plausible values.

    Models the two things that are easy to get wrong in a stub and that would
    otherwise fail the page for reasons unrelated to the page: context-manager
    methods (``expander``, ``container``, ``form``) and the value-returning
    widget methods (``slider``, ``checkbox``, ``text_input``).
    """

    def __init__(self, query: str = QUERY, submit: bool = True) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.query = query
        self.submit = submit
        self.errors: list[str] = []

    # -- helpers ---------------------------------------------------------- #

    def names(self) -> set[str]:
        return {name for name, _, _ in self.calls}

    def count(self, name: str) -> int:
        return sum(1 for n, _, _ in self.calls if n == name)

    def messages(self, name: str) -> list[str]:
        return [str(args) for n, args, _ in self.calls if n == name]

    # -- the fake module --------------------------------------------------- #

    def build(self) -> types.ModuleType:
        module = types.ModuleType("streamlit")
        calls = self.calls

        def record(name: str, result: Any) -> Any:
            def call(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return result

            return call

        class _Ctx:
            """Stands in for the widgets that return a context manager."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                calls.append((name, args, kwargs))

            def __enter__(self) -> _Ctx:
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

        class _Sidebar:
            """``st.sidebar`` is itself a context manager *and* a namespace.

            Modelling only one of the two makes the page fail for a reason that
            has nothing to do with the page.
            """

            def __enter__(self) -> _Sidebar:
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def __getattr__(self, name: str) -> Any:
                def call(*args: Any, **kwargs: Any) -> Any:
                    calls.append((f"sidebar.{name}", args, kwargs))
                    if name == "slider":
                        return kwargs.get("value", 10)
                    if name == "checkbox":
                        return kwargs.get("value", True)
                    return _Ctx(name)

                return call

        module.sidebar = _Sidebar()  # type: ignore[attr-defined]
        for name in (
            "title",
            "markdown",
            "caption",
            "info",
            "error",
            "warning",
            "success",
            "json",
            "write",
            "text",
            "code",
            "set_page_config",
            "dataframe",
        ):
            setattr(module, name, record(name, None))
        for name in ("expander", "container", "form", "spinner", "tabs"):
            setattr(module, name, record(name, _Ctx(name)))

        module.text_input = record("text_input", self.query)  # type: ignore[attr-defined]
        module.form_submit_button = record("form_submit_button", self.submit)  # type: ignore[attr-defined]

        def stop() -> None:
            raise _PageStoppedError()

        module.stop = stop  # type: ignore[attr-defined]

        # cache_resource must actually cache. A decorator that re-invokes the
        # function on every access would make the page probe /stats repeatedly,
        # and a test that only passed because of that would be measuring the
        # stub rather than the page.
        cache: dict[Any, Any] = {}

        def cache_resource(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if fn not in cache:
                    cache[fn] = fn(*args, **kwargs)
                return cache[fn]

            return wrapper

        module.cache_resource = cache_resource  # type: ignore[attr-defined]
        return module


class _PageStoppedError(Exception):
    """Raised by the stubbed ``st.stop()``. Expected, not a failure."""


def _app_modules() -> list[str]:
    return [name for name in sys.modules if name == "app" or name.startswith("app.")]


def _execute_page(recorder: _Recorder) -> None:
    """``runpy`` the page against ``recorder``'s stub.

    ``app/components/render.py`` and ``app/utils/api.py`` do ``import streamlit
    as st`` at *module* scope, so a stub installed while those modules are
    already in ``sys.modules`` would never be seen by them -- the second run in
    this file would silently record nothing and every assertion would pass
    vacuously. They are therefore evicted before the run and restored after.
    """
    saved_app = {name: sys.modules.pop(name) for name in _app_modules()}
    real_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = recorder.build()
    try:
        runpy.run_path(str(HOME), run_name="__main__")
    except _PageStoppedError:
        pass
    finally:
        for name in list(_app_modules()):
            sys.modules.pop(name, None)
        sys.modules.update(saved_app)
        if real_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = real_streamlit


class TestPageExecutes:
    """The page runs to completion, whether or not the backend is up."""

    def test_a_missing_backend_is_reported_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The most common first-run mistake, handled without an exception.

        ``Home.py`` calls ``st.stop()`` after a failed health probe, so a user
        who starts Streamlit before the API gets a message and a read-only
        page -- not a traceback.
        """
        recorder = _Recorder()
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:1")  # closed port
        _execute_page(recorder)
        assert recorder.count("error") == 1, "an unreachable backend must say so"
        message = recorder.messages("error")[0]
        assert "Cannot reach the search API" in message

    def test_the_search_form_is_not_offered_when_the_backend_is_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A search box that cannot work is worse than an explanation.

        ``st.stop()`` halts the script, so the controls are never drawn.
        """
        recorder = _Recorder()
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:1")
        _execute_page(recorder)
        assert recorder.count("text_input") == 0
        assert recorder.count("slider") == 0

    def test_the_page_config_is_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:1")
        _execute_page(recorder)
        assert recorder.count("set_page_config") == 1


class TestPageAgainstALiveBackend:
    """Opt-in. Real HTTP, real corpus, real model.

    Skipped unless ``SSE_TEST_LIVE_API`` is set, because CI has no server. This
    is the check that caught the 1-based/0-based rank bug, so it is worth being
    able to run -- but a test that fails for want of infrastructure is worse
    than a test that declares it did not run.
    """

    @pytest.fixture
    def live_url(self) -> Iterator[str]:
        url = os.environ.get("SSE_TEST_LIVE_API")
        if not url:
            pytest.skip("set SSE_TEST_LIVE_API to run against a live backend")
        yield url

    @pytest.fixture
    def live_recorder(self, monkeypatch: pytest.MonkeyPatch, live_url: str) -> Iterator[_Recorder]:
        monkeypatch.setenv("STREAMLIT_API_BASE_URL", live_url)
        recorder = _Recorder()
        yield recorder

    def test_a_real_search_is_rendered(self, live_recorder: _Recorder) -> None:
        _execute_page(live_recorder)

        assert live_recorder.count("error") == 0, live_recorder.messages("error")
        ready = live_recorder.messages("sidebar.success")
        assert ready and "documents indexed" in ready[0]

        headings = [
            args[0]
            for name, args, _ in live_recorder.calls
            if name == "markdown" and isinstance(args[0], str) and "fused score" in args[0]
        ]
        assert headings, "no result cards were rendered"
        assert len(headings) == 10  # the stub's default top_k

    def test_the_first_result_is_numbered_one(self, live_recorder: _Recorder) -> None:
        """Regression test: the fused rank arrives 1-based and must be used as-is.

        ``SearchHit.rank`` is ``position + 1``. A UI that added 1 again
        rendered the top result as "#2" and the list as #2..#11. Only a real
        response catches that -- a hand-written fixture encodes whatever the
        author believed, including the bug.
        """
        _execute_page(live_recorder)
        headings = [
            args[0]
            for name, args, _ in live_recorder.calls
            if name == "markdown" and isinstance(args[0], str) and "fused score" in args[0]
        ]
        assert headings[0].startswith("**1."), headings[0][:60]

    def test_every_retrieved_document_is_rendered_once(self, live_recorder: _Recorder) -> None:
        _execute_page(live_recorder)
        headings = [
            args[0]
            for name, args, _ in live_recorder.calls
            if name == "markdown" and isinstance(args[0], str) and "fused score" in args[0]
        ]
        doc_ids = [h.split("doc_id `")[1].split("`")[0] for h in headings]
        assert len(doc_ids) == len(set(doc_ids)), "a document was rendered twice"

    def test_the_scope_note_is_always_shown(self, live_recorder: _Recorder) -> None:
        _execute_page(live_recorder)
        notes = live_recorder.messages("info")
        assert notes and "retrieval quality is unmeasured" in notes[0]

    def test_the_backend_caveat_is_passed_through(self, live_recorder: _Recorder) -> None:
        """The server's own caveat reaches the page unmodified."""
        _execute_page(live_recorder)
        texts = " ".join(live_recorder.messages("text"))
        assert "not a statement that the document answers your question" in texts

    def test_the_sidebar_shows_the_configured_backend(self, live_recorder: _Recorder) -> None:
        _execute_page(live_recorder)
        captions = " ".join(live_recorder.messages("caption"))
        assert "STREAMLIT_API_BASE_URL" in captions


class TestEveryWidgetThePageCallsExists:
    """Guards against a widget that the real Streamlit does not have.

    The stub records whatever names the page asks for, so a typo'd widget would
    silently "work" here. Comparing against the real module closes that gap
    when Streamlit is installed.
    """

    def test_the_recorder_does_not_mask_a_missing_widget(self) -> None:
        import streamlit

        names = {
            "title",
            "markdown",
            "caption",
            "info",
            "error",
            "warning",
            "success",
            "json",
            "write",
            "text",
            "code",
            "set_page_config",
            "dataframe",
            "text_input",
            "form_submit_button",
            "expander",
            "container",
            "form",
            "spinner",
            "stop",
            "cache_resource",
            "sidebar",
        }
        missing = sorted(n for n in names if not hasattr(streamlit, n))
        assert missing == [], f"the page uses widgets this Streamlit lacks: {missing}"
