# pyflows

Agent orchestration in Python. Part of a program is ordinary code; some
expressions are agent calls. The code decides what to call, in what order, and
when to stop. The agent decides the answer.

```python
async def review(target):
    lenses = await gather(ask(f"Review {target} for {lens}…", FINDINGS) for lens in LENSES)
    unique = dedupe(flattens(lenses))
    verdicts = await gather(ask(f"Refute: {f}", VERDICT) for f in chunks(unique, 5))
    return [f for f, v in zip(unique, verdicts) if not v["refuted"]]
```

## Install

Not published to PyPI yet. From a clone:

```bash
git clone https://github.com/riverolfe/pyflows
cd pyflows
uv sync
uv run pyflows --help
```

Or install into your own project:

```bash
uv add "pyflows @ git+https://github.com/riverolfe/pyflows"
```

Requires Python 3.12+.

Uses your existing Claude Code login — Pro, Max, Team or Enterprise. No API key.
In CI, set `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`.

## Commands

```bash
pyflows review src              # 4 lenses, batched verification
pyflows review src --quick      # 1 lens, 1 verifier — ~4× cheaper
pyflows review src --lenses 2   # middle ground

pyflows research "your question"           # multi-modal sweep → synthesize → verify
pyflows research "q" --angles 2 --depth 3  # fewer angles, more citations checked
pyflows research "q" --no-verify           # skip citation checking

pyflows upgrade .                # what would upgrading this repo's deps break?
pyflows upgrade . --limit 5      # audit the 5 highest-blast-radius dependencies
pyflows upgrade . --depth 5      # verify 5 candidate breaks instead of 3

pyflows decide "which queue?" redis rabbitmq kafka   # pick by tournament
pyflows decide "q" "a" "b" --max 4  # enter only the first 4 options

pyflows agents -n 3             # browse past runs
pyflows agents --json           # raw records
pyflows watch                   # live panel while a run is in flight
```

Every command exits non-zero when the run did not complete — a dead run never
reports "clean".

## Four example workflows

All live in `src/pyflows/examples/` — they are examples of the framework, not
part of it. Everything they use lives in `orch.py` and `panel.py`.

**`review`** — find → dedup → refute. Four lenses read the same tree
independently; findings are deduplicated; one agent per batch tries to *refute*
each survivor. Cheap: a cached re-run costs nothing.

```
                    ┌ correctness ────┐
                    ├ security ───────┤
   src ────────────▶├ performance ─────┼──▶ ▓▓ barrier ▓▓ ──▶ dedupe ──▶ 5/agent ──▶ confirmed
                    ├ error-handling ──┤     all 4 lenses in
                    └──────────────────┘
```

**`research`** — sweep → synthesize → verify. The four angles are different
*modalities*, not keyword variants, and each is blind to the others:

| angle | searches |
|---|---|
| `primary` | official docs, specs, papers, standards |
| `practitioner` | forum threads, postmortems, issue trackers |
| `contrarian` | what critics, alternatives, or abandoned approaches say |
| `recent` | changelogs, releases, recent announcements |

A searcher given a reworded query finds what the first one found. A searcher
given a different modality finds what a keyword variant can't.

```
                    ┌ primary ────────┐
   question ───────▶├ practitioner ───┼──▶ ▓▓ barrier ▓▓ ──▶ synthesize ──▶ citations ──▶ brief
                    ├ contrarian ─────┤        4 angles in
                    └ recent ─────────┘
```

The synthesizer is told to **report contradictions rather than resolve them** —
handed thirty claims it will otherwise average them into one confident answer.
Then each citation is checked against its source. A source whose `claim` is just
its own URL is skipped: a verifier handed a bare URL can only confirm the page
exists.

**`upgrade`** — inventory → sweep → verify. One agent reads the dependency
manifest, then one agent per dependency asks what upgrading it would break
*here*. Two differences from the other two make it the interesting one:

| | |
|---|---|
| **The fan-out width is discovered, not declared.** | `review` fans out over a fixed lens tuple, `research` over a fixed angle tuple. Here stage one returns the dependency list and *that* decides how many agents run — the first case where the call count isn't known before the run starts, and the reason `MAX_CALLS` is a real constraint rather than a formality. When the budget can't cover everything found, it says so rather than quietly auditing a prefix. |
| **Every agent is a diff against this repo.** | An agent that only knew "Django 5.0 removed `django.utils.timezone.utc`" would report a break on a repo that never used it. Each agent holds both the repo and the web, and is told that an empty findings array is the common, correct answer. |

```
                    ┌ inventory ──────┐
   target ─────────▶│ N deps, found  │──▶ the N below is not known before the run starts
                    └─────────────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        dep/django ──▶ dep/requests ──▶ dep/attrs ──▶ …        N agents, blind to
             │                │                │            each other, repo + web
             └────────────────┴────────────────┘
                              │
                              ▼
                     ▓▓ barrier ▓▓ ──▶ dedupe ──▶ 5/agent ──▶ confirmed brief
```

Dependencies are ranked direct-first then by how many places use them, so a
truncated audit spends its budget where the blast radius is. Each agent must
also report the changelogs it actually read: an agent whose web tools fail can
still return a schema-valid "no breaks", and without that field an unreachable
dependency is indistinguishable from a safe one.

**`decide`** — advocate → bracket → winner. The one that is not a pipeline.
The others fan out, hit a barrier, then fan out again; this is a knockout
bracket, where each verdict decides who plays next.

```
   question ──▶ case/a ─┐
                case/b ─┴──▶ judge ──▶ W1 ─┐
                case/c ─┐                  ├──▶ final ──▶ winner
                case/d ─┴──▶ judge ──▶ W2 ─┘
```

Four options: two judges in round one, one in the final. An odd field gives the
last option a bye into the next round.

The other three all have a correct answer to check a finding against, so they
verify. "Which should we use?" has none — a single confident answer is one
agent's preference wearing a suit. So every option gets an advocate who cannot
see the others, and options are eliminated head to head until one is left.

| | |
|---|---|
| **Judges are blind to identity.** | A judge sees "Candidate 0" and "Candidate 1", never the option name, so it cannot pick on brand over merit. The two are presented in an order seeded by their own contents, so a first-position bias cannot win every match by default — and seeding it means the cache does not freeze whichever order ran first. |
| **Single elimination, not round robin.** | 4 options cost 7 agents: 4 advocates + 3 judges. Round robin would spend 6 judges on the same field. |
| **An undecided matchup advances both.** | A dead judge must not eliminate a contender — that is a result nobody voted for, and the champion would inherit a field that shrank for no stated reason. When a whole round decides nothing, the remaining options are a tie and there is no winner at all. Either way the command exits non-zero. |

## Measured

Reviewing ~250 lines of `src/pyflows/orch.py`, cold cache:

| mode | agents | tokens | wall clock |
|---|---|---|---|
| `--quick` | 2 | 0.06M | 45s |
| full | 7 | 0.29M | 146s |

Batching alone took the full run from 16 agents / 0.82M / 708s to 7 / 0.29M /
146s with the same 9 findings surviving. Verification is the expensive half.

Research, 3 angles + synthesis + 2 citations: 6 agents, 0.30M, 193s.

Decide, 4 options — 4 advocates + 3 judges: 7 agents, 0.19M, 151s wall clock
across 538s of API time. The cheapest of the four, because no agent reads a
repo: it is the only example that costs nothing to re-run over a tree that
hasn't changed, since nothing in its cache key depends on one.

## How it works

Two framework modules and four examples:

| | |
|---|---|
| `orch.py` | `ask()` — one agent call, output validated against a JSON Schema else `None`. `gather()` — dead stage → `None`, blown budget still stops the run. `stat_signature()` — fingerprints a source tree into the cache key. Plus `CACHE` (diskcache, so a re-run costs nothing), `GATE` (8-deep semaphore — each call is a CLI subprocess), `STATS`, and the run log. |
| `panel.py` | `agents` and `watch` — renders the run log. Knows nothing about any workflow, so any new one is observable for free. |
| `examples/review.py` | find → dedup → refute, plus the CLI dispatcher (it is the workflow that dispatches to) |
| `examples/research.py` | sweep → synthesize → verify |
| `examples/upgrade.py` | inventory → sweep → verify. The fan-out width is discovered at runtime |
| `examples/decide.py` | advocate → bracket → winner. A dataflow, not a pipeline |

The examples share the framework and nothing else. That is deliberate: the shape
of a code review, a research sweep, a dependency audit and a knockout bracket
have little in common beyond the transport, and each new one has been what
proved it. **Add a fifth workflow next to them in `examples/`; add abstractions
when it asks for them, not before.** `orch.py` gained nothing from the third —
the runtime fan-out just meant calling `gather()` on a list whose length came
from stage one — and nothing from the fourth: a bracket is a `while` loop around
`gather()`. The two things the tournament needed and did *not* find in the
framework were a per-matchup decision (which is the pattern) and a way to keep
a judge from picking on position (which is one `zlib.crc32` sort).

Two ceilings, not one: `_Spend` bounds *calls* across a run, `max_budget_usd`
bounds *dollars* within a call. The second catches the runaway single agent the
first cannot see.

## Invariants

These are the rules the tests exist to protect. Each one is a bug that shipped.

- **A checked dependency is not the same as a silent one.** A web tool that
  fails leaves the agent still able to return a clean, schema-valid "no breaks"
  without ever reading a changelog. Every upgrade agent must report the sources
  it actually consulted; one that consulted nothing is `unchecked`, not audited,
  and `pyflows upgrade` exits non-zero rather than print "safe to bump".
  (Observed: a run that spent 0.41M tokens and 917s reporting "0 breaks" while
  every WebFetch silently failed.)
- **A dead run never exits clean.** Every lens dead, or every verifier dead,
  raises. Otherwise "reviewed nothing" reads as "found nothing".
- **A dead stage is `None`, a blown budget is an exception.** Exhausting
  `MAX_CALLS` must stop the run, not truncate it into a short result.
- **A budget stop is not a dead stage.** The CLI enforces `max_budget_usd`
  between turns and returns a well-formed `ResultMessage` with
  `is_error=True` and no `structured_output` — byte-identical to a dead agent.
  Treated as one, "ran out of money" reported as "all lenses failed" and sent
  you to the agent count instead of the bill. It raises `BudgetExhausted`, and
  the panel shows it as `BUDGET`, not `FAIL`.
- **Output is validated.** The SDK returns `structured_output` unvalidated; an
  invalid payload is indistinguishable from a dead agent. One retry is spent
  on a mismatch before giving up.
- **The cache key covers everything the answer depends on** — including a
  fingerprint of the source tree, since the prompt never quotes the files.
  mtime in nanoseconds: two edits in the same second must not collide.
  The fingerprint covers the ~25 languages in `DEFAULT_EXTS` — plus `.yml`/`.yaml`,
  since a pipeline is code with the same review needs — and skips
  `node_modules`, `dist`, `target` and friends. Defaulting to `.py` alone
  meant every non-Python repo fingerprinted to the same empty hash: two TS
  repos collided, and edits never invalidated. `.worktrees` is in that list
  because `git worktree add .worktrees/unit-01` puts a second copy of the
  source under the review root: fingerprinted, one unit's code lands in
  another unit's cache key and a fan-out caches a result computed against
  the wrong tree — green, and stale.
- **Agents are confined.** `cwd` plus `disallowed_tools`. Not a jail — `Read`
  still takes absolute paths. For a real boundary use `can_use_tool`.
- **Batched verdicts are paired by occurrence, not by label.** Labels repeat;
  keying on them merged two agents into one row and hung the second at
  `running` forever.
- **A citation is a claim plus a URL, never a bare URL.** A verifier handed
  only a URL can check nothing — it defaults to "unsupported" having verified
  nothing, which costs an agent per source and learns nothing.
- **Sweep agents stay blind to each other.** A searcher that saw the others'
  claims would echo them back instead of finding what they missed.
- **An advocate that lost its agent never played.** A `decide` option whose
  advocate died is reported as `no_case` and excluded — it did not lose the
  tournament, it never entered it, and a bracket that quietly drops a branch
  hands the crown to a smaller field than the user asked about.
- **A dead judge eliminates nobody — until nobody can be eliminated.** Both
  contenders advance, and the winner is reported provisional. But a round that
  decides *no* matchup does not shrink the field, so the bracket stops there:
  the remaining options are a tie and the run says so. Looping on it is not
  patience, it is 100k judge calls and no answer. (Shipped: a dead judge in a
  2-option field re-ran the same matchup forever.)
- **A judge votes on position, not identity.** `winner: [0, 1]` indexes the
  order the *prompt* presented, which is seeded by the options' own contents so
  position bias cannot become a deterministic win rate across a cached rerun.
  Keying on the argument order instead swaps the two options the judge saw.

## Environment

| variable | default | |
|---|---|---|
| `PYFLOWS_BATCH` | `5` | findings per verifying agent. `1` = one agent per finding |
| `PYFLOWS_CACHE` | `~/.cache/pyflows` | cache dir |
| `PYFLOWS_RUN_LOG` | `~/.cache/pyflows-runs/run.jsonl` | agent log. Outside the cache dir on purpose — clearing the cache must not erase the record of what the run cost |

## Develop

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

Tests are regression checks: each pins a bug this package shipped. A test that
merely executes code gets cut. One file per module, so a failure names its
owner:

| file | covers |
|---|---|
| `test_orch.py` | `ask`, `gather`, both budget ceilings, the cache key and its fingerprint, `STATS` |
| `test_panel.py` | run-log rendering — row pairing, truncation, the outcome table |
| `test_review.py` | the review example — batching, dedup, dead-run guards |
| `test_research.py` | the research example — sweep blindness, citation checking |
| `test_upgrade.py` | the upgrade example — runtime fan-out width, budget capping, refutation |
| `test_decide.py` | the decide example — bracket cost, dead judges, the no-progress guard, `--max`, presented-vs-argument order |
| `conftest.py` | the shared fakes (`patched`, `message`, `FINDING`) |

To decide whether a test earns its place, break the thing it covers and check
that it fails. A test that survives the mutation is decoration:

```bash
# narrow the panel column; test_research_labels_fit_the_panel_column should fail
sed -i '' 's/STAGE_COL = 32/STAGE_COL = 20/' src/pyflows/panel.py && uv run pytest -q
```

That check earned its keep here — a `len(label) <= 32` literal passed after the
column was narrowed to 20, because it asserted a constant against itself.

**A run costs ~0.3M tokens across ~7 agents.** Don't launch one unprompted —
run `pytest`, which uses fakes.

## Known limits

- `cwd` is not a sandbox. An agent can still `Read` an absolute path outside it.
- `can_use_tool` is **not** a fix for that here. An `allowed_tools` entry with
  no specifier — `"Read"`, which is what pyflows sends — auto-approves the whole
  tool *before* the callback is consulted, so the callback never fires and a
  deny-everything gate still leaks. Measured: `allowed_tools=["Read"]` returned
  the contents of a `.env` outside `cwd`. Real boundary means narrowing the
  entries (`"Read(src/**)"`) or a PreToolUse hook.
- `SandboxSettings` sandboxes **bash**, and pyflows disallows `Bash` — so it
  protects nothing here.
- Batching trades independence for cost — a verifier that misreads finding 1
  can carry that into findings 2–5. Fine at 5, not at 20.
- `MAX_CALLS = 40` is a guess at a subscription budget, not a measured limit.
- `max_budget_usd` is checked by the CLI *between turns*, so it overshoots by up
  to one turn's cost — a $0.01 ceiling came back at $0.0322.
- `--quick` gives up independent lenses. Cheap, but a finding then survives on
  one refutation attempt rather than four opinions disagreeing.
- Research verification checks the first `--depth` sources, which is an
  arbitrary prefix, not the most load-bearing ones. Pick by hand for now.
- `decide` advocates run with no tools. They argue from the question, they do
  not go read the repo, so "which library should we use?" is decided on what the
  question says rather than on what this codebase actually does. Put the
  relevant facts in the question, or pass the repo's context in it.
- Stage labels are unique by convention, not by construction. If a workflow
  ever emits two agents sharing a label, the panel merges them again — the run
  log keeps both records, only the display does.