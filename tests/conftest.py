"""Shared fakes. Each test file imports from here rather than redefining them."""

import atexit
import contextlib
import os
import shutil
import tempfile
from typing import Any

import pytest

# Before anything imports pyflows.orch: CACHE is opened and RUN_LOG resolved at
# import time, so a fixture alone is too late — the import would already have
# opened ~/.cache/pyflows. The bug: every pytest run appended fake agents to the
# real run log and stored fake results in the real cache.
_SCRATCH = tempfile.mkdtemp(prefix="pyflows-tests-")
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
os.environ["PYFLOWS_CACHE"] = os.path.join(_SCRATCH, "cache")
os.environ["PYFLOWS_RUN_LOG"] = os.path.join(_SCRATCH, "runs", "run.jsonl")

import pyflows.orch as orch  # noqa: E402
import pyflows.panel as panel  # noqa: E402

FINDING = {"file": "a.py", "line": 1, "summary": "s", "failure": "f"}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path):
    """An empty cache and a fresh run log per test, so no test reads another's results.

    The cache is the one opened from PYFLOWS_CACHE above, cleared rather than
    reopened: a new SQLite-backed Cache per test made the suite 13x slower.
    Both RUN_LOG bindings: orch writes the log, panel imported its own copy.
    """
    # Clearing is destructive: if the redirect above ever stops applying, this
    # would wipe the user's real cache. It did, once, during a mutation check.
    assert os.path.commonpath([orch.CACHE.directory, _SCRATCH]) == _SCRATCH, (
        f"refusing to clear a cache outside the test scratch dir: {orch.CACHE.directory}"
    )
    orch.CACHE.clear()
    log = tmp_path / "runs" / "run.jsonl"
    with patched(orch, "RUN_LOG", log), patched(panel, "RUN_LOG", log):
        yield


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
