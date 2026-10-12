"""Agent calls with a subprocess gate, a disk cache, and a token ceiling.

Three functions is the whole "framework". They stay here until a second
workflow needs them somewhere else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import diskcache
import jsonschema
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

# Each query() spawns a CLI subprocess. Without this cap, fan-out is unbounded.
GATE = asyncio.Semaphore(8)

# Subscription ceiling. A dead agent returns None rather than raising, so a
# failed stage costs one slot, not the round.
MAX_CALLS = 40

# expanduser(), not a bare "~": Path("~/.cache/x") is a RELATIVE path, so the
# default landed in ./~/.cache/ — inside the repo, where `rm -rf` and gitignore
# never find it.
CACHE = diskcache.Cache(str(Path(os.environ.get("PYFLOWS_CACHE", "~/.cache/pyflows")).expanduser()))

# One JSON line per ask(), so a finished run can be interrogated per agent:
#   jq -r 'select(.label|startswith("verify/")) | "\(.label) \(.input)"' run.jsonl
# Deliberately NOT inside the cache dir — `rm -rf ~/.cache/pyflows` is how you
# force a cold run, and it must not delete the record of what that run cost.
RUN_LOG = Path(os.environ.get("PYFLOWS_RUN_LOG", "~/.cache/pyflows-runs/run.jsonl")).expanduser()


def _log_run(record: dict) -> None:
    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # logging must never take down a run


def read_runs() -> list[list[dict]]:
    """Split the run log into runs. A run_start record begins each one."""
    try:
        lines = RUN_LOG.read_text().splitlines()
    except OSError:
        return []

    runs: list[list[dict]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("outcome") == "run_start" or not runs:
            runs.append([])
        runs[-1].append(record)
    return runs


class BudgetExhausted(RuntimeError):
    pass


class Stats:
    """Live accounting for a run. Process-global; reset with reset()."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.agents = 0  # agent calls actually issued
        self.cache_hits = 0  # calls served from CACHE, cost nothing
        self.dead = 0  # calls that failed or failed validation
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0
        self.cost_usd = 0.0
        self.seconds = 0.0

    def record(self, result: ResultMessage, label: str = "", retry: bool = False) -> None:
        self.agents += 1
        # duration_api_ms is typed int but the CLI can emit JSON null, and
        # getattr's default does not cover an explicit None.
        secs = (getattr(result, "duration_api_ms", None) or 0) / 1000
        self.seconds += secs
        self.cost_usd += getattr(result, "total_cost_usd", None) or 0.0
        usage = getattr(result, "usage", None) or {}
        # usage is flat (input_tokens at the top level); model_usage is the
        # per-model breakdown. Fall back per-result — testing the cumulative
        # fields would latch the fallback off after the first non-empty one.
        got = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        if got:
            self.input_tokens += usage.get("input_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or 0
            self.cache_read += usage.get("cache_read_input_tokens") or 0
            self.cache_write += usage.get("cache_creation_input_tokens") or 0
        else:
            for m in (getattr(result, "model_usage", None) or {}).values():
                self.input_tokens += m.get("inputTokens", 0) or 0
                self.output_tokens += m.get("outputTokens", 0) or 0
                self.cache_read += m.get("cacheReadInputTokens", 0) or 0
                self.cache_write += m.get("cacheCreationInputTokens", 0) or 0
        _log_run(
            {
                "label": label,
                "retry": retry,
                "outcome": "ok",
                "agents": self.agents,
                "input": self.input_tokens,
                "output": self.output_tokens,
                "cache_read": self.cache_read,
                "seconds": round(secs, 1),
                "cost_usd": round(getattr(result, "total_cost_usd", None) or 0.0, 4),
            }
        )

    def note(self, label: str, outcome: str) -> None:
        _log_run({"label": label, "outcome": outcome, "agents": self.agents})

    def summary(self) -> str:
        total = self.input_tokens + self.output_tokens
        return (
            f"agents={self.agents} cache_hits={self.cache_hits} dead={self.dead}\n"
            f"tokens: {total / 1e6:.2f}M total "
            f"(in {self.input_tokens / 1e6:.2f}M / out {self.output_tokens / 1e3:.1f}k / "
            f"cache read {self.cache_read / 1e6:.2f}M / write {self.cache_write / 1e6:.2f}M)\n"
            f"api time: {self.seconds:.0f}s across {self.agents} agents"
        )


STATS = Stats()


class _Spend:
    """Countdown for the whole run. Not a token meter — a blast radius."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.remaining = max_calls

    def take(self) -> None:
        if self.remaining <= 0:
            raise BudgetExhausted(f"exhausted {self.max_calls}-call budget")
        self.remaining -= 1


# The extensions a review is likely to be pointed at. Defaulting to .py alone
# meant every non-Python repo fingerprinted to the empty hash — two different
# TS repos collided, and edits never invalidated. An extension list, not "every
# file": node_modules and .git churn on every build and would keep the cache
# permanently cold.
# .yml/.yaml because a pipeline is code with the same review needs, and a
# review that skipped CI config never sees the bug that only CI triggers.
DEFAULT_EXTS = (
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".rb", ".java", ".kt", ".swift", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".php", ".scala", ".ex", ".exs",
    ".yml", ".yaml",
)  # fmt: skip

# Files modified this recently get their content hashed, not just stat'd: a
# filesystem whose clock ticks coarser than this can stamp two writes alike.
# 2s covers ext4's few-ms tick and FAT's 2s resolution.
RACY_WINDOW_NS = 2_000_000_000

# Never fingerprinted: dependencies and build output, not the code under review.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        "vendor",
        ".next",
        ".nuxt",
        "coverage",
        # A sibling worktree's source is not the tree under review. Left in,
        # .worktrees/unit-01 lands inside unit-02's fingerprint and a fan-out
        # over worktrees caches a result computed against another unit's code.
        ".worktrees",
    }
)


def stat_signature(root: str | Path, exts: tuple[str, ...] | None = None) -> str:
    """Cheap fingerprint of a source tree: name/size/mtime of matching files.

    A review's answer depends on the files, but the prompt never quotes them —
    so without this the cache serves results for code that has since changed.
    mtime_ns+size, not content: a stat walk is orders of magnitude cheaper
    than hashing every file, and good enough to invalidate on edit. Nanosecond
    mtime, not seconds — two edits in the same second must not collide.

    Nanosecond mtime is not nanosecond *precision*: ext4 stamps from a clock
    that ticks every few ms, so two same-size writes in one tick look
    identical. A file modified within RACY_WINDOW_NS has its content hashed
    too — git's "racy git" rule. Old files stay a stat; only fresh ones are read.

    Pass `exts` to fingerprint something narrower than DEFAULT_EXTS.
    """
    exts = DEFAULT_EXTS if exts is None else exts
    now = time.time_ns()
    parts = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place: rglob would still descend into node_modules and only
        # throw the results away afterwards. Skipped on the *relative* path, so
        # a repo that happens to live under .../build/ is not skipped wholesale.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).suffix not in exts:
                continue
            path = Path(dirpath, name)
            try:
                st = path.stat()
                part = f"{path}:{st.st_size}:{st.st_mtime_ns}"
                if now - st.st_mtime_ns < RACY_WINDOW_NS:
                    part += ":" + hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            parts.append(part)
    parts.sort()  # walk order is filesystem order; the key must not depend on it
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def cache_key(
    prompt: str,
    schema: dict,
    tools: tuple[str, ...],
    system: str = "",
    cwd: str = ".",
    depends_on: str = "",
) -> str:
    # Explicit fields, not a merged tuple — a system prompt concatenated onto
    # tools can collide with a different tools tuple.
    blob = json.dumps([prompt, schema, sorted(tools), system, cwd, depends_on], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


async def ask(
    prompt: str,
    schema: dict,
    *,
    tools: tuple[str, ...] = ("Read", "Grep", "Glob"),
    use_cache: bool = True,
    spend: _Spend | None = None,
    system: str | None = None,
    cwd: str | Path = ".",
    label: str = "",
    depends_on: str | Path | None = None,
    max_budget_usd: float | None = None,
) -> Any | None:
    """One agent call. Returns output that validates against `schema`, else None.

    One retry is spent on a schema mismatch before giving up — the SDK returns
    structured_output unvalidated, and an invalid payload used to be
    indistinguishable from a dead agent.

    `depends_on` points at a source tree to fingerprint into the cache key.
    Use it whenever the answer depends on files the prompt never quotes —
    otherwise the cache happily serves a review of code that has since changed.

    `label` names this stage in failure output — without it a dead stage is
    just an anonymous None, indistinguishable from a stage that found nothing.

    `max_budget_usd` caps what ONE call may spend, checked by the CLI between
    turns. Distinct from `spend`, which caps calls for the whole run: this one
    stops a single runaway agent, that one bounds the fan-out. Exceeding it
    raises BudgetExhausted rather than returning None — a run that ran out of
    money is not a run with a dead stage, and reporting it as one sends you
    looking at the wrong thing.

    `cwd` sets the agent's working directory and disallowed_tools drops the
    write/exec surface. That is not a jail: Read still accepts absolute paths,
    so an agent can reach ~/.aws/credentials or a .env. ponytail: cwd + a
    read-only tool list covers the accidental case; for a real boundary use a
    PreToolUse hook — can_use_tool is NOT consulted for a bare "Read" entry in
    `tools`, because an allowed_tools entry without a specifier auto-approves
    the whole tool before the callback runs.
    """
    key = cache_key(
        prompt,
        schema,
        tools,
        system or "",
        str(cwd),
        stat_signature(depends_on) if depends_on else "",
    )
    if use_cache and (hit := CACHE.get(key)) is not None:
        STATS.cache_hits += 1
        STATS.note(label, "cache_hit")
        return hit

    if spend is not None:
        spend.take()

    # Logged before the call so a watcher can see in-flight agents, not just
    # finished ones. Followed by exactly one terminal record per path below.
    STATS.note(label, "started")

    root = str(Path(cwd).resolve())
    options = ClaudeAgentOptions(
        output_format={"type": "json_schema", "schema": schema},
        allowed_tools=list(tools),
        disallowed_tools=["Bash", "Write", "Edit"],
        system_prompt=system,
        cwd=root,
        permission_mode="dontAsk",
        max_budget_usd=max_budget_usd,
    )
    try:
        async with GATE:
            result = await _run(prompt, options)
    except Exception as exc:  # noqa: BLE001 — a dead stage is None, not a crash
        STATS.dead += 1
        STATS.note(label, "failed")
        print(f"  stage failed [{label}]: {type(exc).__name__}: {exc}")
        return None

    STATS.record(result, label)
    # Checked before structured_output, because a budget stop arrives as a
    # well-formed ResultMessage with is_error=True and no output — exactly the
    # shape of a dead agent. Counting that as dead makes "ran out of money"
    # report as "all lenses failed".
    _raise_if_over_budget(result, label, max_budget_usd)
    out = getattr(result, "structured_output", None)
    if out is None:
        STATS.dead += 1
        STATS.note(label, "no_output")
        return None

    # The SDK hands back structured_output unvalidated. The JS runtime retries
    # on schema mismatch; here a malformed response would silently land in the
    # same None slot as a dead agent, so check it and give it one correction.
    error = _schema_error(out, schema)
    if error is not None:
        if spend is not None:
            spend.take()
        retry_prompt = (
            f"{prompt}\n\nYour previous response did not satisfy the required "
            f"schema: {error}\nReturn corrected JSON matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        try:
            async with GATE:
                result = await _run(retry_prompt, options)
        except Exception as exc:  # noqa: BLE001
            STATS.dead += 1
            STATS.note(label, "failed_retry")
            print(f"  stage failed [{label}] retry: {type(exc).__name__}: {exc}")
            return None
        STATS.record(result, label, retry=True)
        _raise_if_over_budget(result, f"{label} (retry)", max_budget_usd)
        out = getattr(result, "structured_output", None)
        if out is None or (error := _schema_error(out, schema)) is not None:
            STATS.dead += 1
            STATS.note(label, "invalid")
            print(f"  stage invalid [{label}]: {error or 'no structured output'}")
            return None

    if use_cache:
        CACHE.set(key, out)
    return out


def _raise_if_over_budget(result: Any, label: str, max_budget_usd: float | None) -> None:
    """Turn a CLI budget stop into BudgetExhausted instead of a dead stage.

    The CLI checks the ceiling between turns, so it can overshoot by the cost
    of one turn — $0.01 came back at $0.0322 in testing. Raise with the real
    spend, not the ceiling, or the message reads as "we spent $0.01".
    """
    if getattr(result, "subtype", "") != "error_max_budget_usd":
        return
    spent = getattr(result, "total_cost_usd", None) or 0.0
    STATS.note(label, "over_budget")
    raise BudgetExhausted(
        f"stage {label!r} hit its ${max_budget_usd:g} budget after ${spent:.4f}"
        if max_budget_usd is not None
        else f"stage {label!r} exceeded its budget after ${spent:.4f}"
    )


def _schema_error(data: Any, schema: dict) -> str | None:
    """First validation error, or None if the payload satisfies the schema."""
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        return f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
    return None


async def _run(prompt: str, options: ClaudeAgentOptions) -> ResultMessage:
    """query() is an async iterator; the structured payload rides the final message."""
    result = None
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            result = msg
    if result is None:
        raise RuntimeError("stream ended without a result message")
    return result


async def gather(coros: list[Any]) -> list[Any]:
    """asyncio.gather that swallows per-item failures the way JS parallel() does.

    Returns a list with None in the slot of every call that failed. A blown
    budget is not a dead stage — it re-raises, or the ceiling reads as a short
    result set instead of a stopped run.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    for r in results:
        if isinstance(r, BudgetExhausted):
            raise r
    return [None if isinstance(r, BaseException) else r for r in results]
