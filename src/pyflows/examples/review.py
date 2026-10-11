"""Example workflow: review a repo through independent lenses, then refute.

Shape is find -> barrier -> dedup -> verify -> confirm. This is the pipeline
you'd write in a JS workflow script, in Python.

Also owns the CLI dispatcher, since it is the workflow that dispatches to. The
run-log panel lives in panel.py — it renders any workflow, not just this one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from ..orch import (
    MAX_CALLS,
    STATS,
    BudgetExhausted,
    _log_run,
    _Spend,
    ask,
    gather,
)
from ..panel import show_agents, watch

FINDINGS = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "summary": {"type": "string"},
                    "failure": {"type": "string"},
                },
                "required": ["file", "line", "summary", "failure"],
            },
        }
    },
    "required": ["findings"],
}

# One agent judges a BATCH of findings. Reading the source dominates the cost,
# so judging 5 in one call is far cheaper than 5 calls that each re-read it.
BATCH = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "refuted": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "refuted", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}

# Findings per verifying agent. BATCH_SIZE=1 is the old one-agent-per-finding
# behaviour, useful for debugging a single suspect finding.
BATCH_SIZE = int(os.environ.get("PYFLOWS_BATCH", 5))

# Quick mode: one lens, one verifier, one batch. A fraction of the cost and
# wall clock, with no independent cross-check — a finding only survives if that
# single refutation attempt fails to kill it.
QUICK_LENSES = ("correctness",)

LENSES = ("correctness", "security", "performance", "error-handling")


def _key(f: dict) -> tuple[str, int]:
    return (f["file"], f["line"])


class ReviewFailed(RuntimeError):
    """No lens came back. A dead run must never read as a clean one."""


async def review(
    target: str,
    lenses: tuple[str, ...] = LENSES,
    batch_size: int | None = None,
) -> list[dict]:
    """Return findings that survived verification.

    `batch_size` None means one verifier per batch of BATCH_SIZE findings;
    pass a value >= the finding count to force a single verifier.
    """
    batch_size = batch_size or BATCH_SIZE
    spend = _Spend(MAX_CALLS)

    found = await gather(
        [
            ask(
                f"Review {target} for {lens} problems. Report only defects you can "
                f"point at in a specific file and line.",
                FINDINGS,
                spend=spend,
                cwd=target,
                depends_on=target,
                label=f"find/{lens}",
                system=f"You review code for {lens} defects. No style opinions.",
            )
            for lens in lenses
        ]
    )

    # A dead lens and a lens that found nothing are both None/{} here — count
    # the dead ones, or "reviewed nothing" reads as "found nothing".
    alive = [g for g in found if g is not None]
    if not alive:
        raise ReviewFailed(f"all {len(lenses)} lenses failed")

    # Dedup needs every finder's result at once — the one place a barrier is right.
    seen, unique = set(), []
    for group in alive:
        for f in group.get("findings", []):
            if _key(f) not in seen:
                seen.add(_key(f))
                unique.append(f)

    print(f"{len(unique)} unique findings from {len(alive)}/{len(lenses)} lenses, verifying…")

    batches = [unique[i : i + batch_size] for i in range(0, len(unique), batch_size)]
    results = await gather(
        [
            ask(
                "Try to REFUTE each claim about "
                f"{target}. Read the actual code once, then judge every claim.\n\n"
                + "\n\n".join(
                    f"[{i}] {f['file']}:{f['line']} — {f['summary']}\n"
                    f"    Failure it would cause: {f['failure']}"
                    for i, f in enumerate(batch)
                )
                + "\n\nDefault to refuted=true if the code does not back a claim up. "
                f"Return one verdict per index, {len(batch)} in total.",
                BATCH,
                tools=("Read", "Grep"),
                spend=spend,
                cwd=target,
                depends_on=target,
                label=f"verify/{batch[0]['file']}:{len(batch)}",
                system="You are an adversarial reviewer. Default to refuted=true.",
            )
            for batch in batches
        ]
    )

    confirmed = []
    unconfirmed = 0
    for batch, result in zip(batches, results, strict=True):
        if result is None:
            unconfirmed += len(batch)
            continue
        by_index = {v.get("index"): v for v in result.get("verdicts", [])}
        for i, f in enumerate(batch):
            v = by_index.get(i)
            if v is None:
                # An agent that skipped a finding has not refuted it.
                unconfirmed += 1
            elif not v.get("refuted"):
                confirmed.append({**f, "reason": v.get("reason", "")})

    if unconfirmed:
        print(f"  {unconfirmed}/{len(unique)} findings unconfirmed (failed or skipped)")

    # Every verifier dead is a dead run, not a clean one — same rule as the
    # find stage above. Guarded here because the check must be where the
    # evidence is: a batch that died cannot be told from one that refuted.
    if not any(results):
        raise ReviewFailed(f"all {len(batches)} verification batches failed")

    return confirmed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyflows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rev = sub.add_parser("review", help="review a source tree")
    rev.add_argument("target", nargs="?", default=".")
    rev.add_argument(
        "--quick",
        action="store_true",
        help="one lens, one verifier — cheap, but no independent cross-check",
    )
    rev.add_argument("--lenses", type=int, default=None, help="limit finder agents")

    ag = sub.add_parser("agents", help="browse agents from past runs")
    ag.add_argument("-n", "--last", type=int, default=1, help="how many runs back")
    ag.add_argument("--json", action="store_true", help="emit the raw records")

    wt = sub.add_parser("watch", help="live view of the running agents")
    wt.add_argument("--interval", type=float, default=0.5, help="refresh seconds")

    rs = sub.add_parser("research", help="research a question")
    rs.add_argument("question")
    rs.add_argument("--angles", type=int, default=None, help="limit search angles")
    rs.add_argument("--depth", type=int, default=3, help="citations to verify")
    rs.add_argument("--no-verify", action="store_true", help="skip citation checking")

    up = sub.add_parser("upgrade", help="audit dependency upgrades against a repo")
    up.add_argument("target", nargs="?", default=".")
    up.add_argument("--limit", type=int, default=None, help="max dependencies to audit")
    up.add_argument("--depth", type=int, default=3, help="candidate breaks to verify")
    up.add_argument("--no-verify", action="store_true", help="skip verification")

    dc = sub.add_parser("decide", help="pick between options by tournament")
    dc.add_argument("question")
    dc.add_argument("options", nargs="+", help="two or more options to choose between")
    dc.add_argument("--max", type=int, default=None, help="cap the field size")

    args = parser.parse_args(argv)

    if args.command == "decide":
        from .decide import main as decide_main

        return decide_main(
            [args.question, *args.options] + (["--max", str(args.max)] if args.max else [])
        )
    if args.command == "agents":
        return show_agents(args.last, as_json=args.json)
    if args.command == "watch":
        return watch(args.interval)
    if args.command == "upgrade":
        from .upgrade import main as upgrade_main

        return upgrade_main(
            [args.target, "--depth", str(args.depth)]
            + (["--no-verify"] if args.no_verify else [])
            + (["--limit", str(args.limit)] if args.limit else [])
        )
    if args.command == "research":
        from .research import main as research_main

        return research_main(
            ["research", args.question, "--depth", str(args.depth)]
            + (["--no-verify"] if args.no_verify else [])
            + (["--angles", str(args.angles)] if args.angles else [])
        )

    target = Path(args.target)
    if not target.exists():
        print(f"no such path: {target}")
        return 1

    lenses = LENSES[: args.lenses] if args.lenses else (QUICK_LENSES if args.quick else LENSES)
    # Quick mode forces one batch regardless of how many findings turn up.
    batch_size = MAX_CALLS if args.quick else None

    STATS.reset()
    started = time.monotonic()
    mode = "quick" if args.quick else "full"
    _log_run({"outcome": "run_start", "target": str(target), "lenses": len(lenses), "mode": mode})
    print(f"{mode}: {len(lenses)} lens{'es' if len(lenses) != 1 else ''} on {target}")

    try:
        confirmed = asyncio.run(review(str(target), lenses, batch_size))
    except (ReviewFailed, BudgetExhausted) as exc:
        print(f"review did not complete: {exc}\n{STATS.summary()}")
        return 1

    if not confirmed:
        print("nothing survived verification")
        print(STATS.summary())
        return 0

    for f in confirmed:
        print(f"\n{f['file']}:{f['line']}  {f['summary']}\n  {f['reason']}")
    print(f"\n{len(confirmed)} confirmed")
    print(STATS.summary())
    print(f"wall clock: {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
