"""panel: renders the run log. Knows no workflow."""

import asyncio

import pyflows.examples.research as research
import pyflows.examples.review as cr
import pyflows.orch as orch
import pyflows.panel as panel
from conftest import patched


def test_agents_command_renders_a_run(capsys, tmp_path):
    log = tmp_path / "run.jsonl"
    # Both bindings: orch writes the log, panel reads it.
    with patched(orch, "RUN_LOG", log), patched(panel, "RUN_LOG", log):
        cr.main(["agents"])
        orch._log_run({"outcome": "run_start", "target": "src", "mode": "quick", "lenses": 1})
        orch._log_run(
            {
                "label": "find/correctness",
                "outcome": "ok",
                "input": 12000,
                "output": 900,
                "seconds": 42.0,
            }
        )
        orch._log_run({"label": "verify/a.py:3", "outcome": "failed"})
        assert cr.main(["agents"]) == 0

    out = capsys.readouterr().out
    assert "src" in out and "quick" in out
    assert "find/correctness" in out
    assert "FAIL" in out and "1 failed" in out
    assert "12.0k" in out, out


def test_started_without_a_terminal_record_shows_running():
    # What a watcher draws mid-flight.
    run = [
        {"outcome": "run_start", "target": "src", "mode": "full", "lenses": 4},
        {"label": "find/security", "outcome": "started"},
        {"label": "find/perf", "outcome": "ok", "input": 1000, "output": 50, "seconds": 12.0},
        {"label": "find/security", "outcome": "ok", "input": 2000, "output": 60, "seconds": 30.0},
        {"label": "find/errors", "outcome": "started"},
    ]
    rows = panel._stage_rows(run)
    by_label = {label: (status, tok) for label, status, tok, _, _ in rows}
    assert by_label["find/errors"] == ("running", "")
    assert by_label["find/perf"] == ("done", "1.0k")
    assert by_label["find/security"] == ("done", "2.0k"), "start/finish must pair by label"


def test_render_truncates_paths_from_the_left():
    # The tail of a path identifies it; the prefix does not.
    out = panel._render(
        [
            {"outcome": "run_start", "target": "src", "mode": "full", "lenses": 4},
            {"label": "verify/" + "x" * 40 + "/deep/file.py:12", "outcome": "failed"},
        ]
    )
    assert "…" in out
    assert "file.py:12" in out, out


def test_watch_exits_cleanly_on_interrupt(monkeypatch):
    calls = []

    def stop_after_one(_interval):
        calls.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(panel.time, "sleep", stop_after_one)
    monkeypatch.setattr(panel, "read_runs", lambda: [[{"outcome": "run_start", "target": "s"}]])
    assert panel.watch() == 0
    assert calls == [1]


def test_duplicate_labels_get_separate_rows():
    # Keying rows by label collapsed two agents into one and left the second
    # stuck at 'running' forever. Pair by occurrence instead.
    run = [
        {"outcome": "run_start", "target": "src", "mode": "full", "lenses": 2},
        {"label": "verify/a.py:1", "outcome": "started"},
        {"label": "verify/a.py:1", "outcome": "started"},
        {"label": "verify/a.py:1", "outcome": "ok", "input": 1000, "output": 50, "seconds": 10.0},
        {"label": "verify/a.py:1", "outcome": "ok", "input": 2000, "output": 60, "seconds": 20.0},
    ]
    rows = panel._stage_rows(run)
    assert len(rows) == 2, rows
    assert [r[1] for r in rows] == ["done", "done"], rows
    # Each finish claims its own start, so the tokens do not overwrite.
    assert [r[2] for r in rows] == ["1.0k", "2.0k"], rows


def test_one_of_two_duplicate_labels_still_running():
    # A start with no finish must stay visible as running, not vanish.
    rows = panel._stage_rows(
        [
            {"outcome": "run_start", "target": "s", "mode": "f", "lenses": 2},
            {"label": "verify/x", "outcome": "started"},
            {"label": "verify/x", "outcome": "started"},
            {"label": "verify/x", "outcome": "ok", "input": 500, "output": 20, "seconds": 5.0},
        ]
    )
    assert [(r[1], r[2]) for r in rows] == [("done", "0.5k"), ("running", "")], rows


def test_panel_labels_a_budget_stop_distinctly_from_a_failure():
    # It is not a FAIL: a budget stop means the run was stopped on purpose,
    # and counting it as a failure sends you to the agent count instead of
    # the bill.
    status, _ = panel.OUTCOMES["over_budget"]
    assert status == "BUDGET"
    assert status != "FAIL"

    run = [
        {"outcome": "run_start", "target": "src", "mode": "full", "lenses": 4},
        {"label": "find/correctness", "outcome": "started"},
        {"label": "find/correctness", "outcome": "over_budget"},
    ]
    rows = panel._stage_rows(run)
    assert [r[1] for r in rows] == ["BUDGET"]


def test_render_uses_the_right_noun_for_each_mode():
    # Research fans out over angles; review over lenses. One lens is "lens",
    # not "lenses" — and calling angles lenses reads as a bug.
    research_out = panel._render(
        [
            {"outcome": "run_start", "target": "what is X?", "mode": "research", "lenses": 2},
            {"label": "sweep/primary", "outcome": "started"},
        ]
    )
    assert "2 angles]" in research_out and "lens" not in research_out, research_out
    assert "running" in research_out

    one = panel._render([{"outcome": "run_start", "target": "src", "mode": "quick", "lenses": 1}])
    assert "1 lens]" in one and "lenses]" not in one, one

    # Questions are long and arbitrary, paths are short — an untruncated
    # question wraps the panel and pushes every stage column off the right.
    long_q = panel._render(
        [{"outcome": "run_start", "target": "q" * 200, "mode": "research", "lenses": 4}]
    )
    head = long_q.splitlines()[0]
    assert len(head) < 90, len(head)
    assert "…" in head and "[research, 4 angles]" in head, head


def test_research_labels_fit_the_panel_column():
    # Every stage label must survive the panel's column without truncation.
    # Reads panel.STAGE_COL rather than hardcoding 32 — a literal here passed
    # even after the column was narrowed to 20, which is the bug this guards.
    labels = []

    async def fake(prompt, schema, **kw):
        labels.append(kw.get("label", ""))
        if "Synthesize" in prompt:
            return {
                "answer": "a",
                "agreements": [],
                "contradictions": [],
                "gaps": [],
                "sources": [{"claim": "c", "url": "https://example.com/a/very/long/path/here"}],
            }
        return {"findings": []}

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", depth=1))

    assert labels, "no stage labels captured"
    for label in labels:
        assert len(label) <= panel.STAGE_COL, f"{label!r} would be truncated by the panel"
