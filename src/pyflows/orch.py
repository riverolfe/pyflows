"""Agent calls with a subprocess gate, a disk cache, and a token ceiling.

Three functions is the whole "framework". They stay here until a second
workflow needs them somewhere else.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import diskcache
import jsonschema
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

# Each query() spawns a CLI subprocess. Without this cap, fan-out is unbounded.
GATE = asyncio.Semaphore(8)

# Subscription ceiling. A dead agent returns None rather than raising, so a
# failed stage costs one slot, not the round.
MAX_CALLS = 40

CACHE_DEFAULT = "~/.cache/pyflows"
# Deliberately NOT inside the cache dir — `rm -rf ~/.cache/pyflows` is how you
# force a cold run, and it must not delete the record of what that run cost.
RUN_LOG_DEFAULT = "~/.cache/pyflows-runs/run.jsonl"


def _env_path(var: str, default: str) -> Path:
    # expanduser(), not a bare "~": Path("~/.cache/x") is a RELATIVE path, so the
    # default landed in ./~/.cache/ — inside the repo, where `rm -rf` and
    # gitignore never find it.
    return Path(os.environ.get(var, default)).expanduser()


CACHE = diskcache.Cache(str(_env_path("PYFLOWS_CACHE", CACHE_DEFAULT)))

# One JSON line per ask(), so a finished run can be interrogated per agent:
#   jq -r 'select(.label|startswith("verify/")) | "\(.label) \(.input)"' run.jsonl
RUN_LOG = _env_path("PYFLOWS_RUN_LOG", RUN_LOG_DEFAULT)


# Terminal control characters: C0/C1 (ESC, CR, backspace…) and the bidi
# overrides that reorder what a line appears to say. Model output reaches the
# terminal raw, and it is derived from untrusted repos and web pages — an ESC
# sequence in a "summary" can clear the screen or rewrite earlier lines.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean_text(text: str, *, newlines: bool = True) -> str:
    """Strip control characters; keep \\n and \\t unless `newlines` is False."""
    text = _CONTROL.sub("", text)
    return text if newlines else text.replace("\n", " ").replace("\t", " ")


def _clean(value: Any) -> Any:
    """clean_text() every string in a JSON-shaped value."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


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


class ExtensionError(ValueError):
    """A granted skill, plugin or MCP config that does not exist or will not load."""


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
        self.labels: dict[str, int] = {}  # times each label was handed out

    def unique(self, label: str) -> str:
        """`label`, or `label#n` for the n-th use of it this run."""
        if not label:
            return label
        n = self.labels[label] = self.labels.get(label, 0) + 1
        return label if n == 1 else f"{label}#{n}"

    def record(self, result: ResultMessage) -> dict:
        """Add one call's usage to the run totals and return that call's usage.

        Does not log: ask() writes one terminal record per agent, carrying the
        usage of its first call plus any retry.
        """
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
            tok_in = usage.get("input_tokens") or 0
            tok_out = usage.get("output_tokens") or 0
            read = usage.get("cache_read_input_tokens") or 0
            write = usage.get("cache_creation_input_tokens") or 0
        else:
            tok_in = tok_out = read = write = 0
            for m in (getattr(result, "model_usage", None) or {}).values():
                tok_in += m.get("inputTokens", 0) or 0
                tok_out += m.get("outputTokens", 0) or 0
                read += m.get("cacheReadInputTokens", 0) or 0
                write += m.get("cacheCreationInputTokens", 0) or 0
        self.input_tokens += tok_in
        self.output_tokens += tok_out
        self.cache_read += read
        self.cache_write += write
        return {
            "input": tok_in,
            "output": tok_out,
            "cache_read": read,
            "cache_write": write,
            "seconds": secs,
            "cost_usd": getattr(result, "total_cost_usd", None) or 0.0,
        }

    def note(self, label: str, outcome: str) -> None:
        _log_run({"label": label, "outcome": outcome, "agents": self.agents})

    def finish(self, label: str, outcome: str, usage: dict, retry: bool = False) -> None:
        """The one terminal record for an agent that made at least one call.

        Exactly one per "started": a retry or a budget stop used to log an
        extra "ok" first, which claimed the panel row as done and left the real
        outcome as an orphan row — a dead agent displayed as finished. This
        call's usage, not the run's running totals: the panel sums the records.
        """
        _log_run(
            {
                "label": label,
                "retry": retry,
                "outcome": outcome,
                "agents": self.agents,
                "input": usage["input"],
                "output": usage["output"],
                "cache_read": usage["cache_read"],
                "seconds": round(usage["seconds"], 1),
                "cost_usd": round(usage["cost_usd"], 4),
            }
        )

    def summary(self) -> str:
        # Everything the model processed, cache included. Fresh input plus output
        # alone read 0.01M for a run that processed 0.73M — cache reads and
        # writes are most of a run, and most of what it costs.
        total = self.input_tokens + self.output_tokens + self.cache_read + self.cache_write
        return (
            f"agents={self.agents} cache_hits={self.cache_hits} dead={self.dead}\n"
            f"tokens: {total / 1e6:.2f}M processed "
            f"(fresh in {self.input_tokens / 1e6:.2f}M / out {self.output_tokens / 1e3:.1f}k / "
            f"cache read {self.cache_read / 1e6:.2f}M / write {self.cache_write / 1e6:.2f}M)\n"
            f"cost: ${self.cost_usd:.2f} at API prices\n"
            f"api time: {self.seconds:.0f}s across {self.agents} agents"
        )


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


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
    extensions: str = "",
) -> str:
    # Explicit fields, not a merged tuple — a system prompt concatenated onto
    # tools can collide with a different tools tuple.
    fields = [prompt, schema, sorted(tools), system, cwd, depends_on]
    if extensions:
        fields.append(extensions)
    blob = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# The tools that take a filesystem path, and the input fields holding one.
PATH_TOOLS = {"Read": ("file_path",), "Grep": ("path",), "Glob": ("path", "pattern")}


def _escapes(root: Path, tool: str, tool_input: dict, also: tuple[Path, ...] = ()) -> str | None:
    """The first path in `tool_input` outside `root` and every `also` dir, else None."""
    base = root
    for field in PATH_TOOLS.get(tool, ()):
        value = tool_input.get(field)
        if not value:
            continue
        # resolve() follows symlinks: a link in the repo pointing at ~/.ssh
        # resolves outside root and is denied, not read through. A Glob pattern
        # is resolved whole, not cut at its first wildcard: "src/*/../../x" has
        # an inside prefix and still climbs out — wildcard segments resolve as
        # plain names, so the ".." after them is counted.
        target = (base / Path(value).expanduser()).resolve()
        if not any(target.is_relative_to(r) for r in (root, *also)):
            return str(value)
        if field == "path":
            base = target  # Glob's pattern is relative to its path
    return None


def _confine(root: Path, also: tuple[Path, ...] = ()):
    """PreToolUse hook: deny Read/Grep/Glob on any path outside `root` (and `also`).

    The boundary cwd alone is not. A bare "Read" in allowed_tools approves the
    whole tool before can_use_tool is consulted, so absolute paths walked
    straight out — measured live: Read, Grep and Glob all returned a canary
    file outside cwd. A PreToolUse hook runs before that approval.
    """

    async def hook(input_data: Any, _tool_use_id: str | None, _context: Any) -> dict:
        tool = input_data.get("tool_name", "")
        escaped = _escapes(root, tool, input_data.get("tool_input") or {}, also)
        if escaped is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"{tool} on {escaped} is outside the working directory {root}; "
                    f"only files under it may be read."
                ),
            }
        }

    return hook


USER_SKILLS = Path("~/.claude/skills").expanduser()
SHIM_PLUGIN = "pyflows-user"


@dataclass(frozen=True)
class Extensions:
    """What an agent may load beyond its `tools`. Empty by default.

    `skills`  — skill names: "plugin:skill", or a bare user skill name.
    `plugins` — local plugin dirs.
    `mcp`     — MCP server configs by name; their `mcp__<name>` tools are allowed.
    `inherit` — load user, project and local settings, all skills and MCP servers.
    """

    skills: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    mcp: dict[str, dict] = dataclasses.field(default_factory=dict)
    inherit: bool = False

    def key(self) -> str:
        """Cache-key part: empty for the default, so isolated keys do not move."""
        if self == Extensions():
            return ""
        return json.dumps(
            [sorted(self.skills), sorted(self.plugins), self.mcp, self.inherit], sort_keys=True
        )


NO_EXTENSIONS = Extensions()


def _user_skill_plugin(names: tuple[str, ...]) -> Path:
    """A local plugin dir linking user skills `names`, served as `pyflows-user:<name>`."""
    missing = [n for n in names if not (USER_SKILLS / n / "SKILL.md").is_file()]
    if missing:
        raise ExtensionError(f"no such user skill in {USER_SKILLS}: {', '.join(missing)}")
    digest = hashlib.sha256("|".join(sorted(names)).encode()).hexdigest()[:12]
    shim = Path(CACHE.directory) / "skill-shims" / digest
    (shim / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (shim / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": SHIM_PLUGIN, "version": "0.0.0"})
    )
    (shim / "skills").mkdir(exist_ok=True)
    for name in names:
        link = shim / "skills" / name
        if not link.is_symlink():
            link.symlink_to(USER_SKILLS / name)
    return shim


def _extension_options(ext: Extensions) -> tuple[dict, list[str], tuple[Path, ...]]:
    """(ClaudeAgentOptions fields, extra allowed tools, extra read roots) for `ext`.

    Raises ExtensionError on a plugin or skill that does not exist: a typo must
    fail the run, not quietly run an agent without the skill it was given.
    """
    plugins = []
    for p in ext.plugins:
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            raise ExtensionError(f"no such plugin dir: {p}")
        plugins.append(path)
    allowed = [f"mcp__{name}" for name in ext.mcp]

    if ext.inherit:
        skills: list[str] | str = list(ext.skills) or "all"
        reads = (*plugins, USER_SKILLS.resolve())
        fields: dict[str, Any] = {
            # Explicit list: with skills set the SDK defaults to user+project,
            # silently dropping settings.local.json.
            "setting_sources": ["user", "project", "local"],
            "mcp_servers": dict(ext.mcp),
            "strict_mcp_config": False,
        }
    else:
        user = tuple(n for n in ext.skills if ":" not in n)
        skills = [n for n in ext.skills if ":" in n]
        if user:
            plugins.append(_user_skill_plugin(user))
            skills += [f"{SHIM_PLUGIN}:{n}" for n in user]
        reads = (*plugins, *((USER_SKILLS / n).resolve() for n in user))
        fields = {"setting_sources": [], "mcp_servers": dict(ext.mcp), "strict_mcp_config": True}

    fields["plugins"] = [{"type": "local", "path": str(p)} for p in plugins]
    if skills:
        fields["skills"] = skills
    return fields, allowed, reads


def add_extension_args(parser: Any) -> None:
    """--skill / --plugin / --mcp-config / --inherit-settings on a workflow's CLI."""
    g = parser.add_argument_group("extensions (agents load none unless named)")
    g.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="grant a skill: a user skill, or plugin:skill with --plugin (repeatable)",
    )
    g.add_argument(
        "--plugin", action="append", default=[], metavar="DIR", help="load a local plugin dir"
    )
    g.add_argument(
        "--mcp-config",
        metavar="FILE",
        default=None,
        help='JSON file of {"mcpServers": {name: config}} to load',
    )
    g.add_argument(
        "--inherit-settings",
        action="store_true",
        help="load user/project/local settings as Claude Code would — trusted repos only",
    )


def extensions_from(args: Any) -> Extensions:
    mcp = {}
    if args.mcp_config:
        try:
            mcp = json.loads(Path(args.mcp_config).expanduser().read_text())["mcpServers"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExtensionError(f"--mcp-config {args.mcp_config}: {exc!r}") from exc
    return Extensions(
        skills=tuple(args.skill),
        plugins=tuple(args.plugin),
        mcp=mcp,
        inherit=args.inherit_settings,
    )


def extension_argv(args: Any) -> list[str]:
    """The extension flags in `args`, as argv."""
    out = [a for s in args.skill for a in ("--skill", s)]
    out += [a for p in args.plugin for a in ("--plugin", p)]
    if args.mcp_config:
        out += ["--mcp-config", args.mcp_config]
    if args.inherit_settings:
        out.append("--inherit-settings")
    return out


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
    ext: Extensions = NO_EXTENSIONS,
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

    `cwd` sets the agent's working directory, disallowed_tools drops the
    write/exec surface, and a PreToolUse hook (`_confine`) denies Read, Grep
    and Glob on any path that resolves outside `cwd` — absolute, `..`, `~` or
    a symlink. Web tools are not confined: an agent given WebFetch can still
    send what it read under `cwd` to a URL. No MCP servers are loaded, the
    user's own included, and no settings, skills or plugins. `ext` grants them.
    """
    ext_fields, ext_tools, ext_reads = _extension_options(ext)
    label = STATS.unique(label)
    key = cache_key(
        prompt,
        schema,
        tools,
        system or "",
        str(cwd),
        stat_signature(depends_on) if depends_on else "",
        ext.key(),
    )
    if use_cache and (hit := CACHE.get(key)) is not None:
        STATS.cache_hits += 1
        STATS.note(label, "cache_hit")
        return _clean(hit)  # entries cached before cleaning existed

    if spend is not None:
        spend.take()

    # Logged before the call so a watcher can see in-flight agents, not just
    # finished ones. Followed by exactly one terminal record per path below.
    STATS.note(label, "started")

    root = str(Path(cwd).resolve())
    options = ClaudeAgentOptions(
        output_format={"type": "json_schema", "schema": schema},
        allowed_tools=[*tools, *ext_tools],
        disallowed_tools=["Bash", "Write", "Edit"],
        system_prompt=system,
        cwd=root,
        permission_mode="dontAsk",
        max_budget_usd=max_budget_usd,
        # Settings, skills, plugins and MCP servers: none unless `ext` grants them.
        **ext_fields,
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="|".join(PATH_TOOLS), hooks=[_confine(Path(root), ext_reads)])
            ]
        },
    )
    try:
        async with GATE:
            result = await _run(prompt, options)
    except Exception as exc:  # noqa: BLE001 — a dead stage is None, not a crash
        STATS.dead += 1
        STATS.note(label, "failed")
        print(f"  stage failed [{label}]: {type(exc).__name__}: {exc}")
        return None

    usage = STATS.record(result)
    # Checked before structured_output, because a budget stop arrives as a
    # well-formed ResultMessage with is_error=True and no output — exactly the
    # shape of a dead agent. Counting that as dead makes "ran out of money"
    # report as "all lenses failed".
    if _over_budget(result):
        STATS.finish(label, "over_budget", usage)
        _raise_budget(result, label, max_budget_usd)
    out = getattr(result, "structured_output", None)
    if out is None:
        STATS.dead += 1
        STATS.finish(label, "no_output", usage)
        return None

    # The SDK hands back structured_output unvalidated. The JS runtime retries
    # on schema mismatch; here a malformed response would silently land in the
    # same None slot as a dead agent, so check it and give it one correction.
    error = _schema_error(out, schema)
    retried = error is not None
    if retried:
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
            STATS.finish(label, "failed_retry", usage, retry=True)
            print(f"  stage failed [{label}] retry: {type(exc).__name__}: {exc}")
            return None
        usage = _add_usage(usage, STATS.record(result))
        if _over_budget(result):
            STATS.finish(label, "over_budget", usage, retry=True)
            _raise_budget(result, f"{label} (retry)", max_budget_usd)
        out = getattr(result, "structured_output", None)
        if out is None or (error := _schema_error(out, schema)) is not None:
            STATS.dead += 1
            STATS.finish(label, "invalid", usage, retry=True)
            print(f"  stage invalid [{label}]: {error or 'no structured output'}")
            return None

    STATS.finish(label, "ok", usage, retry=retried)
    # Cleaned once here, so no workflow has to remember to before printing.
    out = _clean(out)
    if use_cache:
        CACHE.set(key, out)
    return out


def _over_budget(result: Any) -> bool:
    return getattr(result, "subtype", "") == "error_max_budget_usd"


def _raise_budget(result: Any, label: str, max_budget_usd: float | None) -> None:
    """Turn a CLI budget stop into BudgetExhausted instead of a dead stage.

    The CLI checks the ceiling between turns, so it can overshoot by the cost
    of one turn — $0.01 came back at $0.0322 in testing. Raise with the real
    spend, not the ceiling, or the message reads as "we spent $0.01".
    """
    spent = getattr(result, "total_cost_usd", None) or 0.0
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
        # A bad extension is a config error, identical in every slot: as None
        # it would report "all lenses failed" for a typo in --skill.
        if isinstance(r, (BudgetExhausted, ExtensionError)):
            raise r
    return [None if isinstance(r, BaseException) else r for r in results]
