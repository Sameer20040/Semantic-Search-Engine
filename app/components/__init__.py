"""Streamlit presentation components.

Each function here takes already-built view objects and returns nothing --
they render and nothing else. No component performs I/O, and none constructs a
search view; that keeps the logic under test in ``app/utils/api.py`` where it
can be exercised without a browser.
"""
