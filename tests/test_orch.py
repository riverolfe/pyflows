"""orch: the framework — ask, gather, budget, cache, accounting."""

import asyncio
import json
import os
from pathlib import Path

import pytest

import pyflows.orch as orch
from conftest import message, patched
from pyflows.orch import (
    BudgetExhausted,
    _Spend,
    cache_key,
    gather,
    read_runs,
    stat_signature,
)


def test_gather_keeps_good_results_when_others_die():
    # A dead stage is None, not a crash that takes the whole round with it.
    async def ok():
        return "fine"

    async def boom():
        raise RuntimeError("agent died")

    assert asyncio.run(gather([ok(), boom(), ok()])) == ["fine", None, "fine"]


def test_gather_does_not_swallow_a_blown_budget():
    # Budget exhaustion used to land in a None slot, so the ceiling read as a
    # short result set instead of a stopped run.
    spend = _Spend(0)

    async def stage():
        spend.take()

    with pytest.raises(BudgetExhausted):
        asyncio.run(gather([stage()]))


def test_ask_confines_cwd_and_drops_write_tools():
    # Read is pre-approved, so confinement must come from cwd and the
    # disallowed list — otherwise an agent reads ~/.aws/credentials.
    captured = {}

    async def fake_run(prompt, options):
        captured["options"] = options
        raise RuntimeError("stop here — we only inspect the options")

    with patched(orch, "_run", fake_run):
        asyncio.run(orch.ask("p", {"type": "object"}, cwd="src"))

    opts = captured["options"]
    assert opts.cwd.endswith("src"), opts.cwd
    assert {"Bash", "Write", "Edit"} <= set(opts.disallowed_tools), opts.disallowed_tools
    assert opts.permission_mode == "dontAsk", opts.permission_mode


def test_cache_key_separates_every_input_it_is_built_from():
    # Every one of these was a real collision or a stale-hit bug.
    base = ("p", {"t": 1}, ("Read",))
    assert cache_key(*base, "", ".", "sig") != cache_key(*base, "", ".", "other")
    assert cache_key(*base, "", ".", "sig") != cache_key(*base, "", ".")
    # Named, not positional: the two calls below differ only in cwd, which
    # reads as the same call written twice.
    assert cache_key(*base, system="", cwd="a") != cache_key(*base, system="", cwd="b")
    # A system prompt merged into tools collided with another tools tuple.
    assert cache_key(*base, "") != cache_key("p", {"t": 1}, ("", "Read"))
    assert cache_key("p", {"t": 1}, ("Read",), "s1") != cache_key("p", {"t": 1}, ("Read",), "s2")
    assert cache_key("p", {"t": 1}, ("Read",)) == cache_key("p", {"t": 1}, ("Read",))


def test_stat_signature_catches_an_edit_the_clock_did_not_see(tmp_path):
    # The bug: ext4 stamps mtime from a clock that ticks every few ms, so two
    # same-size writes in one tick got the same mtime_ns and the same key — a
    # review served the result for the code before the edit. Pin the mtime by
    # hand so this fails on every filesystem, not just a coarse one.
    src = tmp_path / "a.py"
    src.write_text("x = 1\n")
    stamp = src.stat().st_mtime_ns
    before = stat_signature(tmp_path)

    src.write_text("x = 2\n")
    os.utime(src, ns=(stamp, stamp))
    assert stat_signature(tmp_path) != before

    # Outside the window the stat is trusted and the file is not read — the
    # cheap path the fingerprint exists for. Hashing everything would pass the
    # assertion above and fail this one.
    old = stamp - 10 * orch.RACY_WINDOW_NS
    os.utime(src, ns=(old, old))
    before = stat_signature(tmp_path)
    src.write_text("x = 3\n")
    os.utime(src, ns=(old, old))
    assert stat_signature(tmp_path) == before


def test_stat_signature_tracks_source_edits_not_other_files(tmp_path):
    src = tmp_path / "a.py"
    src.write_text("x = 1\n")
    before = stat_signature(tmp_path)

    src.write_text("x = 2\n")  # same byte count — only ns mtime catches this
    assert stat_signature(tmp_path) != before

    (tmp_path / "b.py").write_text("y = 1\n")
    assert stat_signature(tmp_path) != before


def test_stat_signature_sees_non_python_sources(tmp_path):
    # The bug: exts defaulted to .py only, so every non-Python repo
    # fingerprinted to the empty hash — two TS repos collided and edits
    # never invalidated. A review of a TS repo served stale results forever.
    (tmp_path / "a.ts").write_text("x = 1\n")
    before = stat_signature(tmp_path)

    (tmp_path / "a.ts").write_text("x = 22\n")
    assert stat_signature(tmp_path) != before

    other = tmp_path.parent / "other-repo"
    other.mkdir()
    (other / "b.ts").write_text("y = 1\n")
    assert stat_signature(other) != before, "two different repos must not collide"


def test_stat_signature_ignores_dependency_and_build_directories(tmp_path):
    # Otherwise node_modules churn keeps the cache permanently cold.
    (tmp_path / "app.js").write_text("a = 1\n")
    before = stat_signature(tmp_path)

    vendored = tmp_path / "node_modules" / "dep"
    vendored.mkdir(parents=True)
    (vendored / "index.js").write_text("module.exports = 1\n")
    assert stat_signature(tmp_path) == before

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "app.js").write_text("minified output\n")
    assert stat_signature(tmp_path) == before


def test_stat_signature_ignores_a_sibling_worktree(tmp_path):
    # A `git worktree add .worktrees/unit-01` inside the repo puts a second
    # copy of the source under the review root. Fingerprinted, unit-01's code
    # lands in unit-02's cache key and the fan-out caches a result computed
    # against another unit's tree — green, and stale.
    (tmp_path / "unit.py").write_text("x = 1\n")
    before = stat_signature(tmp_path)

    unit = tmp_path / ".worktrees" / "unit-01"
    unit.mkdir(parents=True)
    (unit / "unit.py").write_text("x = 1\n")
    (unit / "only_here.py").write_text("y = 1\n")
    assert stat_signature(tmp_path) == before


def test_stat_signature_prunes_on_the_relative_path_not_the_absolute_one(tmp_path):
    # The repo under test may itself live at .../build/. Skipping any path
    # containing a skipped dir name would fingerprint it as empty.
    nested = tmp_path / "build" / "repo"
    nested.mkdir(parents=True)
    (nested / "real.py").write_text("z = 1\n")
    assert stat_signature(nested) != stat_signature(nested.parent / "empty")


def test_stat_signature_covers_pipeline_configs(tmp_path):
    # A pipeline is code with the same review needs; a review that skips CI
    # config never sees the bug that only CI triggers.
    (tmp_path / "ci.yml").write_text("on: push\n")
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n")
    before = stat_signature(tmp_path)

    (tmp_path / "ci.yml").write_text("on: pull_request\n")
    assert stat_signature(tmp_path) != before

    (tmp_path / "docker-compose.yaml").write_text("services: {a: 1}\n")
    assert stat_signature(tmp_path) != before


def test_stat_signature_does_not_prune_dot_github_as_dot_git(tmp_path):
    # SKIP_DIRS prunes by directory name during the walk. ".github" must not
    # be caught by ".git" — the workflows in it are exactly what you review.
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("on: push\n")
    assert stat_signature(tmp_path) != "e3b0c44298fc1c14"


def test_ask_does_not_serve_a_cached_result_for_changed_sources(tmp_path):
    # The regression that mattered: the prompt never quotes the file, so a key
    # without a source fingerprint served reviews of code that had changed.
    (tmp_path / "a.py").write_text("x = 1\n")
    calls = []

    async def counting(prompt, options):
        calls.append(prompt)
        return message(structured_output={"findings": []})

    schema = {"type": "object", "required": ["findings"]}

    def ask():
        return orch.ask("review", schema, cwd=tmp_path, depends_on=tmp_path)

    with patched(orch, "_run", counting):
        asyncio.run(ask())
        asyncio.run(ask())
        assert len(calls) == 1, "unchanged sources should hit the cache"

        (tmp_path / "a.py").write_text("x = 999\n")
        asyncio.run(ask())
    assert len(calls) == 2, "edited sources must invalidate the cache"


def test_ask_returns_none_rather_than_an_invalid_payload():
    # The SDK hands back structured_output unvalidated; an invalid payload used
    # to be indistinguishable from a dead agent.
    async def bad(prompt, options):
        return message(structured_output={"wrong": "shape"})

    schema = {"type": "object", "required": ["a"]}
    with patched(orch, "_run", bad):
        out = asyncio.run(orch.ask("p", schema, use_cache=False))
    assert out is None


def test_ask_recovers_when_the_retry_is_valid():
    # One correction attempt, then accept the good payload.
    calls = []

    async def bad_then_good(prompt, options):
        calls.append(prompt)
        shape = {"wrong": "shape"} if len(calls) == 1 else {"a": 1}
        return message(structured_output=shape)

    schema = {"type": "object", "required": ["a"]}
    with patched(orch, "_run", bad_then_good):
        out = asyncio.run(orch.ask("p", schema, use_cache=False))
    assert out == {"a": 1}, out
    assert len(calls) == 2, calls


def test_ask_raises_when_a_call_hits_its_usd_ceiling():
    # The CLI stops on max_budget_usd and returns a well-formed ResultMessage
    # with is_error=True and no structured_output — the exact shape of a dead
    # agent. Counting it as dead reported "ran out of money" as "all lenses
    # failed", which sends you looking at the agent count instead of the bill.
    async def over_budget(prompt, options):
        return message(subtype="error_max_budget_usd", is_error=True, total_cost_usd=0.0322)

    schema = {"type": "object", "required": ["a"]}
    orch.STATS.reset()
    with patched(orch, "_run", over_budget), pytest.raises(orch.BudgetExhausted):
        asyncio.run(orch.ask("p", schema, use_cache=False, max_budget_usd=0.01))
    assert orch.STATS.dead == 0, "a budget stop is not a dead stage"


def test_budget_message_reports_actual_spend_not_the_ceiling():
    # The CLI checks between turns and overshoots: $0.01 came back at $0.0322.
    # Saying "spent $0.01" would understate every overrun in the log.
    async def over_budget(prompt, options):
        return message(subtype="error_max_budget_usd", is_error=True, total_cost_usd=0.0322)

    schema = {"type": "object", "required": ["a"]}
    with (
        patched(orch, "_run", over_budget),
        pytest.raises(orch.BudgetExhausted, match=r"\$0\.0322"),
    ):
        asyncio.run(orch.ask("p", schema, use_cache=False, max_budget_usd=0.01))


def test_ask_does_not_raise_on_an_ordinary_error_result():
    # Only the budget subtype escalates. Every other error is still a dead
    # stage: one bad call must not take down a fan-out of seven.
    async def errored(prompt, options):
        return message(subtype="error_during_execution", is_error=True)

    schema = {"type": "object", "required": ["a"]}
    orch.STATS.reset()
    with patched(orch, "_run", errored):
        out = asyncio.run(orch.ask("p", schema, use_cache=False, max_budget_usd=0.01))
    assert out is None
    assert orch.STATS.dead == 1


def test_gather_re_raises_a_budget_stop_from_a_fan_out():
    # A blown USD budget must stop the run, not truncate it into a short
    # result set — same rule as a blown call budget. gather re-raises, so the
    # exception escapes rather than coming back as a None in one slot.
    over = message(subtype="error_max_budget_usd", is_error=True, total_cost_usd=9.0)

    async def one_ok():
        return {"a": 1}

    async def blown():
        orch._raise_if_over_budget(over, "find/x", 1.0)

    with pytest.raises(orch.BudgetExhausted, match="find/x"):
        asyncio.run(orch.gather([one_ok(), blown(), one_ok()]))


def test_stats_reads_flat_usage_and_tolerates_null_duration():
    # usage is flat, not a per-model dict — iterating .values() as buckets
    # silently reported 0 tokens for every run. duration_api_ms is typed int
    # but the CLI emits null.
    orch.STATS.reset()
    orch.STATS.record(
        message(
            duration_api_ms=None,
            total_cost_usd=0.5,
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 7,
            },
        ),
        "t",
    )
    assert orch.STATS.input_tokens == 100, orch.STATS.summary()
    assert orch.STATS.output_tokens == 20
    assert (orch.STATS.cache_read, orch.STATS.cache_write) == (5, 7)
    assert orch.STATS.cost_usd == 0.5
    assert orch.STATS.seconds == 0.0


def test_default_cache_paths_are_not_relative_to_the_cwd(tmp_path, monkeypatch):
    # Path("~/.cache/x") is RELATIVE — the "~" is just a directory name. Both
    # defaults landed in ./~/.cache/ inside the repo being scanned, where
    # `rm -rf ~/.cache/pyflows` and .gitignore never reach them.
    monkeypatch.delenv("PYFLOWS_CACHE", raising=False)
    monkeypatch.delenv("PYFLOWS_RUN_LOG", raising=False)
    monkeypatch.chdir(tmp_path)

    cache = Path(os.environ.get("PYFLOWS_CACHE", "~/.cache/pyflows")).expanduser()
    assert cache.is_absolute(), cache
    assert not cache.is_relative_to(tmp_path), cache


def test_run_log_survives_cache_wipe_and_appends_json(tmp_path):
    # Clearing the cache is how you force a cold run; it must not delete the
    # record of what the run cost.
    assert (
        "cache-runs" in str(orch.RUN_LOG)
        or orch.RUN_LOG.parent
        != Path(orch.os.environ.get("PYFLOWS_CACHE", "~/.cache/pyflows")).expanduser()
    ), orch.RUN_LOG

    log = tmp_path / "run.jsonl"
    with patched(orch, "RUN_LOG", log):
        orch._log_run({"label": "find/security", "outcome": "ok"})
        orch._log_run({"label": "find/perf", "outcome": "failed"})

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["label"] for r in rows] == ["find/security", "find/perf"]
    assert [r["outcome"] for r in rows] == ["ok", "failed"]


def test_read_runs_skips_corrupt_lines(tmp_path):
    # A truncated final line (killed run) must not lose the whole log.
    log = tmp_path / "run.jsonl"
    log.write_text('{"outcome": "run_start"}\n{"label": "x", "outc\n')
    with patched(orch, "RUN_LOG", log):
        runs = read_runs()
    assert len(runs) == 1 and runs[0][0]["outcome"] == "run_start"


def test_ask_logs_a_started_record_before_the_call(tmp_path):
    # Live tracking depends on this: a watcher must see an agent while it is
    # still running, not only after it finishes.
    seen = []

    async def watching(prompt, options):
        seen.append(orch.read_runs()[-1][-1].get("outcome") if orch.read_runs() else None)
        return message(structured_output={"ok": 1})

    log = tmp_path / "run.jsonl"
    with patched(orch, "RUN_LOG", log), patched(orch, "_run", watching):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False, label="find/x"))

    outcomes = [json.loads(line)["outcome"] for line in log.read_text().splitlines()]
    assert seen == ["started"], f"started record must precede the call, saw {seen}"
    assert outcomes == ["started", "ok"], outcomes
