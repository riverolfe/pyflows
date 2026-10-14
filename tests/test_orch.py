"""orch: the framework — ask, gather, budget, cache, accounting."""

import asyncio
import json
import os
from typing import Any, cast

import pytest

import pyflows.orch as orch
import pyflows.panel as panel
from conftest import message, patched
from pyflows.orch import (
    BudgetExhausted,
    SeamError,
    _Spend,
    cache_key,
    gather,
    guard,
    read_runs,
    stat_signature,
)

# The producer's contract, checked where the consumer reads it.
FINDINGS = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "url": {"type": "string"}},
                "required": ["claim", "url"],
            },
        }
    },
    "required": ["findings"],
}


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


def _hook_decision(root, tool, tool_input):
    hook = orch._confine(root)
    out = asyncio.run(hook({"tool_name": tool, "tool_input": tool_input}, None, None))
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")


def test_confinement_denies_every_route_out_of_cwd(tmp_path):
    # The bug: cwd was the only boundary, and a bare "Read" in allowed_tools
    # approves the whole tool — measured live, Read, Grep and Glob all returned
    # a canary file outside cwd. Each of these is a different way out.
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "leak").symlink_to(tmp_path)  # a link in the repo pointing out of it
    (root / "deep" / "sub").mkdir(parents=True)
    (root / "inner").symlink_to(root / "deep" / "sub")  # a link that stays inside
    outside = str(tmp_path / "secret.env")

    for tool, tool_input in [
        ("Read", {"file_path": outside}),
        ("Read", {"file_path": "../secret.env"}),
        ("Read", {"file_path": "~/.ssh/id_rsa"}),
        ("Read", {"file_path": "leak/secret.env"}),
        ("Grep", {"pattern": "KEY", "path": str(tmp_path)}),
        ("Glob", {"pattern": f"{tmp_path}/*.env"}),
        ("Glob", {"pattern": "../*.env"}),
        ("Glob", {"pattern": "*.env", "path": ".."}),
        ("Glob", {"pattern": "../../*", "path": "src"}),
        ("Glob", {"pattern": "src/*/../../../*.env"}),  # inside prefix, climbs out
        ("Grep", {"pattern": "KEY", "path": "inner/../leak"}),  # ".." after an inside link
        ("Glob", {"pattern": "*", "path": "inner/../leak"}),
        ("Read", {"file_path": "inner/../leak/secret.env"}),
    ]:
        assert _hook_decision(root, tool, tool_input) == "deny", (tool, tool_input)


def test_confinement_allows_everything_inside_cwd(tmp_path):
    # Over-blocking is its own failure: a review that cannot read the repo
    # returns "no findings" — the dead-run-reads-as-clean shape again.
    root = tmp_path.resolve()
    (root / "src").mkdir()
    for tool, tool_input in [
        ("Read", {"file_path": "src/app.py"}),
        ("Read", {"file_path": str(root / "src" / "app.py")}),
        ("Grep", {"pattern": "x"}),  # no path: searches cwd
        ("Glob", {"pattern": "**/*.py"}),
        ("Glob", {"pattern": "*.py", "path": "src"}),
        ("Glob", {"pattern": "../*.py", "path": "src"}),  # relative to path: still root
        ("WebFetch", {"url": "https://example.com"}),  # not a path tool
    ]:
        assert _hook_decision(root, tool, tool_input) == "allow", (tool, tool_input)


def test_confinement_allows_granted_read_roots_and_nothing_else(tmp_path):
    root = (tmp_path / "repo").resolve()
    extra = (tmp_path / "evidence").resolve()
    root.mkdir()
    extra.mkdir()
    hook = orch._confine(root, (extra,))

    def decision(tool, tool_input):
        out = asyncio.run(hook({"tool_name": tool, "tool_input": tool_input}, None, None))
        return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")

    assert decision("Read", {"file_path": str(extra / "trace.log")}) == "allow"
    assert decision("Grep", {"pattern": "x", "path": str(extra)}) == "allow"
    assert decision("Read", {"file_path": "../evidence/trace.log"}) == "allow"
    assert decision("Read", {"file_path": str(tmp_path / "secret.env")}) == "deny"
    assert decision("Read", {"file_path": "../evidence/../secret.env"}) == "deny"


def test_ask_passes_reads_to_the_confinement_hook(tmp_path):
    captured = {}

    async def fake_run(prompt, options):
        captured["options"] = options
        raise RuntimeError("stop here — we only inspect the options")

    root = tmp_path / "repo"
    extra = tmp_path / "evidence"
    root.mkdir()
    extra.mkdir()
    with patched(orch, "_run", fake_run):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False, cwd=root, reads=(extra,)))

    (matcher,) = captured["options"].hooks["PreToolUse"]
    (hook,) = matcher.hooks

    def probe(path):
        out = asyncio.run(
            hook({"tool_name": "Read", "tool_input": {"file_path": str(path)}}, None, None)
        )
        return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")

    assert probe(extra / "a.log") == "allow"
    assert probe(tmp_path / "secret.env") == "deny"


def test_ask_installs_the_confinement_hook_for_every_path_tool():
    captured = {}

    async def fake_run(prompt, options):
        captured["options"] = options
        raise RuntimeError("stop here — we only inspect the options")

    with patched(orch, "_run", fake_run):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False))

    (matcher,) = captured["options"].hooks["PreToolUse"]
    assert set(matcher.matcher.split("|")) == {"Read", "Grep", "Glob"}, matcher.matcher


def test_ask_loads_no_mcp_servers_not_even_the_users():
    # The bug: agents inherited every MCP server the user had configured —
    # measured live, 5 servers and 36 mcp__ tools, some with side effects
    # outside the repo. strict_mcp_config is what stops the CLI loading its own.
    captured = {}

    async def fake_run(prompt, options):
        captured["options"] = options
        raise RuntimeError("stop here — we only inspect the options")

    with patched(orch, "_run", fake_run):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False))

    opts = captured["options"]
    assert opts.strict_mcp_config is True
    assert opts.mcp_servers == {}, opts.mcp_servers
    # No settings, skills or plugins either.
    assert opts.setting_sources == [], opts.setting_sources
    assert opts.plugins == [] and opts.skills is None, (opts.plugins, opts.skills)


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
    assert cache_key(*base, reads='["/a"]') != cache_key(*base, reads='["/b"]')
    assert cache_key(*base, reads='["/a"]') != cache_key(*base)


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
    # The clock is pinned inside the window, so a slow run cannot leave it.
    with patched(orch.time, "time_ns", lambda: stamp + 1):
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


def test_ask_strips_terminal_control_characters_from_model_output():
    # Model output is derived from untrusted repos and web pages and printed
    # raw: an ESC sequence in a finding's summary can clear the screen or
    # rewrite earlier lines, a bidi override can reverse what a line says.
    # Cleaned once in ask(), cached result included, so no workflow forgets.
    async def hostile(prompt, options):
        return message(structured_output={"s": ["ok\x1b[2J\r\u202eevil", "line\nnext\tcol"]})

    with patched(orch, "_run", hostile):
        out = asyncio.run(orch.ask("p", {"type": "object"}))
        cached = asyncio.run(orch.ask("p", {"type": "object"}))

    assert out == cached == {"s": ["ok[2Jevil", "line\nnext\tcol"]}, out

    # An entry cached before cleaning existed is cleaned on the way out too.
    key = cache_key("q", {"type": "object"}, ("Read", "Grep", "Glob"), "", ".", "")
    orch.CACHE.set(key, {"s": "old\x1b[2J"})
    assert asyncio.run(orch.ask("q", {"type": "object"})) == {"s": "old[2J"}


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
    # The message reports actual spend, not the ceiling: the CLI checks between
    # turns and overshoots — $0.01 came back at $0.0322.
    async def over_budget(prompt, options):
        return message(subtype="error_max_budget_usd", is_error=True, total_cost_usd=0.0322)

    schema = {"type": "object", "required": ["a"]}
    orch.STATS.reset()
    with (
        patched(orch, "_run", over_budget),
        pytest.raises(orch.BudgetExhausted, match=r"\$0\.0322"),
    ):
        asyncio.run(orch.ask("p", schema, use_cache=False, max_budget_usd=0.01))
    assert orch.STATS.dead == 0, "a budget stop is not a dead stage"


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
    )
    assert orch.STATS.input_tokens == 100, orch.STATS.summary()
    assert orch.STATS.output_tokens == 20
    assert (orch.STATS.cache_read, orch.STATS.cache_write) == (5, 7)
    assert orch.STATS.cost_usd == 0.5
    assert orch.STATS.seconds == 0.0


def _log():
    return [json.loads(line) for line in orch.RUN_LOG.read_text().splitlines()]


def test_each_log_record_carries_its_own_calls_usage():
    # The bug: records logged the run's running totals, and the panel sums the
    # records — two equal 10k calls showed as 10k + 20k = 30k of input.
    usage = {"input_tokens": 10_000, "output_tokens": 500, "cache_read_input_tokens": 3}

    async def run(prompt, options):
        return message(structured_output={"a": 1}, usage=usage)

    orch.STATS.reset()
    with patched(orch, "_run", run):
        for label in ("find/a", "find/b"):
            asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False, label=label))

    done = [r for r in _log() if r["outcome"] == "ok"]
    assert [(r["input"], r["output"], r["cache_read"]) for r in done] == [(10_000, 500, 3)] * 2
    assert orch.STATS.input_tokens == 20_000, "STATS keeps the run total"


def test_every_agent_logs_exactly_one_terminal_record():
    # The bug: a retry or a budget stop logged an "ok" first, which claimed the
    # panel row as done; the real outcome landed as an orphan row, so an agent
    # that never produced valid output displayed as finished.
    schema = {"type": "object", "required": ["a"]}
    usage = {"input_tokens": 100, "output_tokens": 10}
    replies = {
        "fixed-on-retry": [{"wrong": 1}, {"a": 1}],
        "never-valid": [{"wrong": 1}, {"wrong": 2}],
    }

    for label, outcome in [("fixed-on-retry", "done"), ("never-valid", "FAIL")]:
        orch.RUN_LOG.unlink(missing_ok=True)
        queue = list(replies[label])

        async def run(prompt, options, queue=queue):
            return message(structured_output=queue.pop(0), usage=usage)

        with patched(orch, "_run", run):
            asyncio.run(orch.ask("p", schema, use_cache=False, label=label))

        records = _log()
        assert [r["outcome"] for r in records][0] == "started", records
        assert len(records) == 2, f"{label}: one start, one terminal record — got {records}"
        assert records[1]["retry"] is True and records[1]["input"] == 200, records[1]
        # A FAIL row still made two calls: it shows what they cost, not blanks.
        assert [r[1:3] for r in panel._stage_rows(records)] == [(outcome, "0.2k")], records

    orch.RUN_LOG.unlink(missing_ok=True)

    async def broke(prompt, options):
        return message(subtype="error_max_budget_usd", is_error=True, usage=usage)

    with patched(orch, "_run", broke), pytest.raises(orch.BudgetExhausted):
        asyncio.run(orch.ask("p", schema, use_cache=False, label="b", max_budget_usd=0.01))
    assert [r[1] for r in panel._stage_rows(_log())] == ["BUDGET"], _log()

    # Budget runs out at the schema retry: still one terminal record.
    orch.RUN_LOG.unlink(missing_ok=True)

    async def invalid(prompt, options):
        return message(structured_output={"wrong": 1}, usage=usage)

    with patched(orch, "_run", invalid), pytest.raises(orch.BudgetExhausted):
        asyncio.run(orch.ask("p", schema, use_cache=False, label="r", spend=orch._Spend(1)))
    assert [r[1:3] for r in panel._stage_rows(_log())] == [("BUDGET", "0.1k")], _log()


def test_summary_counts_cache_tokens_and_cost():
    # The bug: "tokens: 0.01M total" for a run that processed 0.73M — cache
    # reads and writes were left out of the total, and no cost was printed.
    orch.STATS.reset()
    orch.STATS.record(
        message(
            total_cost_usd=2.5,
            usage={
                "input_tokens": 34,
                "output_tokens": 14_500,
                "cache_read_input_tokens": 440_000,
                "cache_creation_input_tokens": 270_000,
            },
        )
    )
    text = orch.STATS.summary()
    assert "0.72M processed" in text, text
    assert "$2.50" in text, text


def test_default_paths_are_absolute_and_keep_the_log_out_of_the_cache(tmp_path, monkeypatch):
    # Path("~/.cache/x") is RELATIVE — the "~" is just a directory name. Both
    # defaults landed in ./~/.cache/ inside the repo being scanned, where
    # `rm -rf ~/.cache/pyflows` and .gitignore never reach them. And the log
    # lives outside the cache dir: wiping the cache must not erase what runs cost.
    monkeypatch.delenv("PYFLOWS_CACHE", raising=False)
    monkeypatch.delenv("PYFLOWS_RUN_LOG", raising=False)
    monkeypatch.chdir(tmp_path)

    cache = orch._env_path("PYFLOWS_CACHE", orch.CACHE_DEFAULT)
    log = orch._env_path("PYFLOWS_RUN_LOG", orch.RUN_LOG_DEFAULT)
    for path in (cache, log):
        assert path.is_absolute() and not path.is_relative_to(tmp_path), path
    assert not log.is_relative_to(cache), (log, cache)

    monkeypatch.setenv("PYFLOWS_CACHE", "~/elsewhere")
    assert orch._env_path("PYFLOWS_CACHE", orch.CACHE_DEFAULT).is_absolute()


def test_read_runs_skips_corrupt_lines():
    # A truncated final line (killed run) must not lose the whole log.
    orch.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    orch.RUN_LOG.write_text('{"outcome": "run_start"}\n{"label": "x", "outc\n')
    runs = read_runs()
    assert len(runs) == 1 and runs[0][0]["outcome"] == "run_start"


def test_ask_logs_a_started_record_before_the_call():
    # Live tracking depends on this: a watcher must see an agent while it is
    # still running, not only after it finishes.
    seen = []

    async def watching(prompt, options):
        seen.append(orch.read_runs()[-1][-1].get("outcome") if orch.read_runs() else None)
        return message(structured_output={"ok": 1})

    with patched(orch, "_run", watching):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False, label="find/x"))

    outcomes = [json.loads(line)["outcome"] for line in orch.RUN_LOG.read_text().splitlines()]
    assert seen == ["started"], f"started record must precede the call, saw {seen}"
    assert outcomes == ["started", "ok"], outcomes


def _options_for(ext, **kw):
    captured = {}

    async def fake_run(prompt, options):
        captured["options"] = options
        raise RuntimeError("stop here — we only inspect the options")

    with patched(orch, "_run", fake_run):
        asyncio.run(orch.ask("p", {"type": "object"}, use_cache=False, ext=ext, **kw))
    return captured["options"]


def test_a_granted_user_skill_loads_without_loading_user_settings(tmp_path):
    skills = tmp_path / "skills"
    (skills / "qmd" / "references").mkdir(parents=True)
    (skills / "qmd" / "SKILL.md").write_text("---\nname: qmd\n---\n")
    (skills / "other").mkdir()
    (skills / "other" / "SKILL.md").write_text("x")
    repo = tmp_path / "repo"
    repo.mkdir()

    with patched(orch, "USER_SKILLS", skills):
        opts = _options_for(orch.Extensions(skills=("qmd",)), cwd=repo)

    assert opts.setting_sources == [], opts.setting_sources
    assert opts.skills == ["pyflows-user:qmd"], opts.skills
    (plugin,) = opts.plugins
    shim = orch.Path(plugin["path"])
    assert json.loads((shim / ".claude-plugin" / "plugin.json").read_text())["name"] == (
        "pyflows-user"
    )
    assert (shim / "skills" / "qmd").resolve() == (skills / "qmd").resolve()
    assert not (shim / "skills" / "other").exists(), "only the granted skill is served"

    # The granted skill's dir is readable; nothing else outside cwd is.
    (matcher,) = opts.hooks["PreToolUse"]
    hook = matcher.hooks[0]

    def decision(path):
        out = asyncio.run(
            hook({"tool_name": "Read", "tool_input": {"file_path": path}}, None, None)
        )
        return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")

    assert decision(str(skills / "qmd" / "references" / "a.md")) == "allow"
    assert decision(str(shim / "skills" / "qmd" / "SKILL.md")) == "allow"
    assert decision(str(skills / "other" / "SKILL.md")) == "deny"
    assert decision(str(tmp_path / "secret")) == "deny"


def test_a_missing_skill_or_plugin_stops_the_run_not_one_slot(tmp_path):
    with patched(orch, "USER_SKILLS", tmp_path):
        plugin = tmp_path / "plug"
        (plugin / ".claude-plugin").mkdir(parents=True)
        (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "plug"}')
        for ext in (
            orch.Extensions(skills=("nope",)),
            orch.Extensions(plugins=(str(tmp_path / "nope"),)),
            orch.Extensions(skills=("plug:nope",), plugins=(str(plugin),)),
            orch.Extensions(skills=("nope:skill",), plugins=(str(plugin),)),
        ):
            with pytest.raises(orch.ExtensionError, match="nope"):
                asyncio.run(gather([orch.ask("p", {"type": "object"}, ext=ext)]))


def test_granted_plugins_skills_and_mcp_are_passed_and_allowed(tmp_path):
    plugin = tmp_path / "fw-guru"
    (plugin / "skills" / "crg").mkdir(parents=True)
    (plugin / "skills" / "crg" / "SKILL.md").write_text("x")
    server = {"type": "stdio", "command": "crg", "args": ["serve"]}
    opts = _options_for(
        orch.Extensions(skills=("fw-guru:crg",), plugins=(str(plugin),), mcp={"crg": server})
    )

    assert opts.plugins == [{"type": "local", "path": str(plugin.resolve())}], opts.plugins
    assert opts.skills == ["fw-guru:crg"], opts.skills
    assert opts.mcp_servers == {"crg": server} and opts.strict_mcp_config is True
    assert "mcp__crg" in opts.allowed_tools, opts.allowed_tools
    assert opts.setting_sources == [], opts.setting_sources


def test_inherit_loads_every_setting_source_including_local():
    opts = _options_for(orch.Extensions(inherit=True))
    assert opts.setting_sources == ["user", "project", "local"], opts.setting_sources
    assert opts.skills == "all", opts.skills
    assert opts.strict_mcp_config is False


def test_extensions_are_part_of_the_cache_key():
    async def fake(prompt, options):
        return message(structured_output={"skills": options.skills})

    with patched(orch, "_run", fake):
        bare = asyncio.run(orch.ask("p", {"type": "object"}))
        inherited = asyncio.run(
            orch.ask("p", {"type": "object"}, ext=orch.Extensions(inherit=True))
        )
    assert bare == {"skills": None} and inherited == {"skills": "all"}, (bare, inherited)


def test_extension_flags_survive_cli_forwarding(tmp_path):
    import argparse

    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"crg": {"command": "crg"}}}))
    parser = argparse.ArgumentParser()
    orch.add_extension_args(parser)
    argv = ["--skill", "a", "--skill", "p:b", "--plugin", "/x"]
    argv += ["--mcp-config", str(config), "--inherit-settings"]
    first = parser.parse_args(argv)
    again = parser.parse_args(orch.extension_argv(first))

    ext = orch.extensions_from(again)
    assert ext == orch.Extensions(
        skills=("a", "p:b"), plugins=("/x",), mcp={"crg": {"command": "crg"}}, inherit=True
    ), ext
    assert orch.extensions_from(parser.parse_args([])) == orch.NO_EXTENSIONS


def test_agents_sharing_a_label_get_distinct_ones():
    async def ok(prompt, options):
        return message(structured_output={})

    with patched(orch, "_run", ok):
        asyncio.run(
            gather([orch.ask(f"p{i}", {"type": "object"}, label="verify/x") for i in range(3)])
        )
        asyncio.run(orch.ask("other", {"type": "object"}, label="find/y"))

    finished = [r["label"] for r in read_runs()[-1] if r.get("outcome") == "ok"]
    assert sorted(finished) == ["find/y", "verify/x", "verify/x#2", "verify/x#3"], finished

    orch.STATS.reset()
    assert orch.STATS.unique("verify/x") == "verify/x"


def test_extensions_reject_bad_shapes_and_cannot_be_changed():
    # Annotated dict[str, Any] because the point of the list is that the
    # values are the wrong type for every field — the union ty infers from an
    # unannotated tuple is exactly what the constructor is meant to reject.
    bad_shapes: tuple[dict[str, Any], ...] = (
        {"skills": "qmd"},
        {"plugins": [1]},
        {"mcp": {"x": ["not", "a", "config"]}},
        {"mcp": {"x": {"obj": object()}}},
    )
    for kw in bad_shapes:
        with pytest.raises(orch.ExtensionError):
            orch.Extensions(**kw)

    source = {"crg": {"command": "crg", "args": ["serve"]}}
    ext = orch.Extensions(mcp=source)
    source["crg"]["command"] = "changed"
    assert ext.mcp_servers() == {"crg": {"command": "crg", "args": ["serve"]}}
    # cast, not a suppression comment: the test deliberately lies to the type
    # checker — `mcp` is a frozen Mapping, and the point is that pretending it
    # is a dict does not make assignment work at runtime.
    as_dict = cast(dict[str, Any], orch.NO_EXTENSIONS.mcp)
    with pytest.raises(TypeError):
        as_dict["x"] = {}
    with pytest.raises(TypeError):
        # No suppression needed: mcp's values are Any, so the inner assignment
        # type-checks. The old `# type: ignore[index]` was for mypy, which this
        # repo does not run.
        ext.mcp["crg"]["command"] = "x"
    assert hash(ext) == hash(orch.Extensions(mcp={"crg": {"command": "crg", "args": ["serve"]}}))
    assert orch.NO_EXTENSIONS.key() == ""


def test_a_stale_shim_link_is_replaced(tmp_path):
    skills = tmp_path / "skills"
    (skills / "qmd").mkdir(parents=True)
    (skills / "qmd" / "SKILL.md").write_text("x")
    with patched(orch, "USER_SKILLS", skills):
        shim = orch._user_skill_plugin(("qmd",))
        link = shim / "skills" / "qmd"
        link.unlink()
        link.symlink_to(tmp_path)  # planted elsewhere
        assert orch._user_skill_plugin(("qmd",)) == shim
    assert link.resolve() == (skills / "qmd").resolve()


def test_cache_failures_warn_and_keep_the_result(capsys):
    async def ok(prompt, options):
        return message(structured_output={"a": 1})

    class Broken:
        directory = orch.CACHE.directory

        def get(self, key):
            raise orch.sqlite3.OperationalError("database is locked")

        def set(self, key, value):
            raise OSError("disk full")

    with patched(orch, "_run", ok), patched(orch, "CACHE", Broken()):
        assert asyncio.run(orch.ask("p", {"type": "object"}, label="x")) == {"a": 1}
    text = capsys.readouterr().out
    assert "cache read failed [x]" in text and "cache write failed [x]" in text, text


def test_gather_prints_what_it_swallows(capsys):
    async def boom():
        raise KeyError("lost")

    assert asyncio.run(gather([boom()])) == [None]
    assert "stage raised: KeyError" in capsys.readouterr().out


def test_stat_signature_skips_files_that_are_not_regular(tmp_path):
    import threading

    (tmp_path / "real.py").write_text("x")
    without = stat_signature(tmp_path)
    os.mkfifo(tmp_path / "trap.py")  # a fresh fifo: reading it would block
    got = []
    worker = threading.Thread(target=lambda: got.append(stat_signature(tmp_path)), daemon=True)
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        with open(tmp_path / "trap.py", "w"):  # unblock the reader before failing
            pass
        pytest.fail("stat_signature blocked reading a fifo")
    assert got == [without]


def test_run_keeps_the_last_result_and_fails_without_one(monkeypatch):
    first, last = message(result="first"), message(result="last")

    def stream(*items):
        async def query(prompt, options):
            for item in items:
                yield item

        return query

    monkeypatch.setattr(orch, "query", stream("progress", first, "more", last))
    assert asyncio.run(orch._run("p", cast(Any, None))) is last
    monkeypatch.setattr(orch, "query", stream("progress"))
    with pytest.raises(RuntimeError, match="without a result message"):
        asyncio.run(orch._run("p", cast(Any, None)))


def test_a_retry_that_raises_is_a_dead_agent():
    calls = []

    async def bad_then_raise(prompt, options):
        calls.append(prompt)
        if len(calls) == 1:
            return message(structured_output={"wrong": "shape"})
        raise RuntimeError("connection lost")

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    with patched(orch, "_run", bad_then_raise):
        assert asyncio.run(orch.ask("p", schema, use_cache=False, label="flaky")) is None
    assert len(calls) == 2 and orch.STATS.dead == 1
    outcomes = [r.get("outcome") for r in map(json.loads, orch.RUN_LOG.read_text().splitlines())]
    assert outcomes[-1] == "failed_retry"


def test_a_retry_over_the_per_call_ceiling_is_budget_exhausted():
    calls = []

    async def bad_then_broke(prompt, options):
        calls.append(prompt)
        if len(calls) == 1:
            return message(structured_output={"wrong": "shape"})
        return message(subtype="error_max_budget_usd", is_error=True)

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    with patched(orch, "_run", bad_then_broke), pytest.raises(orch.BudgetExhausted):
        asyncio.run(orch.ask("p", schema, use_cache=False, max_budget_usd=0.01))
    assert len(calls) == 2


def test_stats_fall_back_to_per_model_usage():
    stats = orch.Stats()
    per_model = {
        "a": {
            "inputTokens": 10,
            "outputTokens": 2,
            "cacheReadInputTokens": 5,
            "cacheCreationInputTokens": 1,
        },
        "b": {"inputTokens": 7, "outputTokens": 3},
    }
    used = stats.record(message(usage={}, model_usage=per_model))
    assert (stats.input_tokens, stats.output_tokens) == (17, 5)
    assert used["input"] == 17 and used["output"] == 5
    stats.record(message(usage={"input_tokens": 4, "output_tokens": 1}, model_usage=per_model))
    assert (stats.input_tokens, stats.output_tokens) == (21, 6)


def test_read_runs_splits_at_each_run_start():
    for target in ("first", "second", "third"):
        orch._log_run({"outcome": "run_start", "target": target, "mode": "full", "lenses": 1})
        orch._log_run({"label": f"find/{target}", "outcome": "ok"})
    runs = orch.read_runs()
    assert [r[0]["target"] for r in runs] == ["first", "second", "third"]
    assert all(len(r) == 2 for r in runs)


def test_guard_returns_the_payload_when_it_matches_the_producers_schema():
    group = {"findings": [{"claim": "x is true", "url": "https://e/x"}]}
    assert guard(group, FINDINGS, "synthesize") is group


def test_guard_raises_at_the_seam_instead_of_letting_a_missing_key_pass():
    # What research.py saw: a finding with no url, which downstream reads as
    # f.get("url") -> None and drops silently.
    with pytest.raises(SeamError, match="synthesize.*findings"):
        guard({"findings": [{"claim": "x is true"}]}, FINDINGS, "synthesize")
