"""Shared fakes. Each test file imports from here rather than redefining them."""

import contextlib
from typing import Any

import pyflows.orch as orch

FINDING = {"file": "a.py", "line": 1, "summary": "s", "failure": "f"}


@contextlib.contextmanager
def patched(obj, name, value):
    """Swap an attribute for the duration of a block.

    Replaces the orig/restore/try/finally dance that every fake needed.
    """
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def message(**kw):
    """A minimal ResultMessage; override any field."""
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "result": "ok",
        "session_id": "s",
        "total_cost_usd": 0.0,
    }
    return orch.ResultMessage(**(fields | kw))
