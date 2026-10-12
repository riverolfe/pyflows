"""Example workflow: pick between options by tournament instead of by asking.

Shape is advocate -> bracket -> winner, and it is the first example here that
is not a pipeline: review, research and upgrade all fan out, take a barrier,
then fan out again. This one is a *dataflow* — a knockout bracket, where each
judge's verdict decides who plays next.

That is the whole reason it exists. "Which should we use?" has no correct
answer to check against, so there is nothing to refute and a single confident
answer is just one agent's preference wearing a suit. Instead every option
gets an advocate who cannot see the others, and the options are eliminated
head to head until one is left.

Two rules do the real work:

- **Advocates are blind to each other.** An agent that saw the other options
  would converge on the most defensible-sounding one, which is the same answer
  you already had. Same invariant research.py's sweep angles hold.
- **Judges see plans, not option names.** Each side is shown as "Candidate
  0/1" with its plan and caveat, in an order seeded by the options' text.

Single elimination, not round robin: n advocates + (n-1) judges, so 4 options
cost 7 agents. Round robin would cost 6 judges for the same 4.

An undecided matchup — a dead judge — advances BOTH contenders rather than
silently eliminating one. A bracket that quietly drops a branch because an
agent crashed is a winner chosen from a smaller field, which is the exact lie
this package exists not to tell.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import zlib
from typing import Any

from ..orch import (
    MAX_CALLS,
    NO_EXTENSIONS,
    STATS,
    BudgetExhausted,
    ExtensionError,
    Extensions,
    _log_run,
    _Spend,
    add_extension_args,
    ask,
    extensions_from,
    gather,
)

# Advocates and judges run with no tools.
TOOLS: tuple[str, ...] = ()

CASE = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        # Not decoration. A judge comparing two plans can only weigh a
        # trade-off it has been told about; without this the advocate's most
        # flattering framing is the one that gets compared.
        "caveat": {"type": "string"},
    },
    "required": ["plan", "caveat"],
}

PAIR = {
    "type": "object",
    "properties": {
        # enum, not plain integer: a judge answering "Candidate 2" as index 2
        # or -1 for the first is not a verdict, and jsonschema then spends
        # ask()'s built-in retry on it. Validating here is cheaper than
        # detecting a nonsense winner after the fact.
        "winner": {"type": "integer", "enum": [0, 1]},
        "why": {"type": "string"},
    },
    "required": ["winner", "why"],
}


class DecideFailed(RuntimeError):
    """Nothing to decide between. A dead run must never read as a decision."""


def _order(a: dict, b: dict) -> list[dict]:
    """Presentation order for a matchup: stable, and not the argument order.

    Seeded from the options' own text, not random(). A random() here flips
    between runs, the judge is cached, and the first order it ever saw becomes
    permanent — so half the field would never be shown first again.
    """
    return sorted([a, b], key=lambda o: zlib.crc32(o["option"].encode()))


async def decide(
    question: str,
    options: list[str],
    spend: _Spend | None = None,
    ext: Extensions = NO_EXTENSIONS,
) -> dict:
    """Run a knockout bracket over `options`. Returns a brief.

    `spend` is injectable so a caller can share one call budget across a
    decide() and whatever else it is doing.
    """
    if len(options) < 2:
        raise DecideFailed("a tournament needs at least two options")
    spend = spend or _Spend(MAX_CALLS)
    started = time.monotonic()

    cases = await gather(
        [
            ask(
                f"{question}\n\nMake the strongest honest case for option {i}: "
                f"{opt}\n\nArgue FOR it. Do not hedge and do not compare — you "
                f"cannot see the other options, and naming them would only "
                f"soften your argument. Do not name the option itself in the "
                f"plan or caveat; describe what it does.",
                CASE,
                tools=TOOLS,
                spend=spend,
                ext=ext,
                label=f"case/{opt}"[:32],
                system="You advocate one option, alone. Be its best advocate.",
            )
            for i, opt in enumerate(options)
        ]
    )

    # An option whose advocate died did not lose the tournament, it never
    # played. Different thing, and the difference is the whole point. One pass:
    # two traversals of the same zip can drift apart the moment either filter
    # gains a clause.
    contenders: list[dict[str, Any]] = []
    no_case: list[str] = []
    for option, case in zip(options, cases, strict=True):
        if case:
            contenders.append({"option": option, "case": case})
        else:
            no_case.append(option)
    if no_case:
        print(f"  {len(no_case)} option(s) got no case and cannot compete: {', '.join(no_case)}")
    if len(contenders) < 2:
        raise DecideFailed(
            f"only {len(contenders)}/{len(options)} options were argued; that is not a decision"
        )

    rounds: list[dict] = []
    undecided = 0
    while len(contenders) > 1:
        # (matchup, presentation order) built once: the judge reads `shown`
        # and its index comes back against the same pair. Recomputing the order
        # at verdict time makes the binding depend on _order staying pure.
        games = [(_order(*m) if len(m) == 2 else None, m) for m in _matchups(contenders)]
        shown = [pair for pair, _ in games if pair is not None]

        verdicts = iter(
            await gather(
                [
                    ask(
                        f"{question}\n\nPick the stronger of these two plans.\n\n"
                        + "\n\n".join(
                            f"[{i}] Candidate {i}\n    Plan: {c['case']['plan']}\n"
                            f"    Weakest point: {c['case']['caveat']}"
                            for i, c in enumerate(pair)
                        )
                        + "\n\nJudge on merit, not on which is written first. Pick "
                        'one — "both are fine" is not a verdict you can return. '
                        "Return the index of the better plan.",
                        PAIR,
                        tools=TOOLS,
                        spend=spend,
                        ext=ext,
                        label=f"judge/{pair[0]['option']}"[:32],
                        system="You are an impartial judge. One winner or nothing.",
                    )
                    for pair in shown
                ]
            )
        )

        nxt = []
        for pair, m in games:
            if pair is None:
                nxt.append(m[0])  # bye
                continue
            verdict = next(verdicts)
            if verdict is None:
                # Dead judge. Advancing both keeps the bracket honest: picking
                # one would be an elimination nobody voted for.
                undecided += 1
                rounds.append({"match": [c["option"] for c in pair], "outcome": "undecided"})
                nxt.extend(m)
                continue
            nxt.append(pair[verdict["winner"]])
            rounds.append(
                {"match": [c["option"] for c in pair], "outcome": "winner", "why": verdict["why"]}
            )
        # A round that eliminates nobody does not shrink the field, so the
        # bracket would re-run the same matchups forever. That is not "both
        # advance, carry on" — it is a tournament that cannot be resolved, and
        # the honest result is no result.
        if len(nxt) >= len(contenders):
            print(f"  no matchup could be decided; {len(contenders)} options remain tied")
            break
        contenders = nxt

    tied = [c["option"] for c in contenders]
    return {
        "question": question,
        "options": options,
        # None when the bracket stalled — a provisional winner is still a lie,
        # and this is the case where there is nothing to be provisional about.
        "winner": None
        if len(tied) > 1
        else {"option": tied[0], "plan": contenders[0]["case"]["plan"]},
        "tied": tied if len(tied) > 1 else [],
        "rounds": rounds,
        "no_case": no_case,
        # The winner is provisional whenever a judge died: the bracket above it
        # had a matchup nobody scored.
        "undecided": undecided,
        "elapsed": time.monotonic() - started,
    }


def _matchups(contenders: list[dict]) -> list[list[dict]]:
    return [contenders[i : i + 2] for i in range(0, len(contenders), 2)]


def render(brief: dict) -> None:
    print(f"\n{brief['question']}")
    if brief["winner"] is None:
        print(f"\nno winner: {', '.join(brief['tied'])} could not be separated")
    else:
        print(f"\nbest option: {brief['winner']['option']}\n  {brief['winner']['plan']}")

    print("\nrounds:")
    for r in brief["rounds"]:
        a, b = r["match"]
        if r["outcome"] == "undecided":
            print(f"  {a}  vs  {b}  —  no judge; both advance")
        else:
            print(f"  {a}  beats  {b}\n    {r['why']}")

    if brief["undecided"] and brief["winner"] is not None:
        print(
            f"\n{brief['undecided']} matchup(s) went undecided — the winner is "
            f"provisional, not chosen."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyflows decide", description=__doc__)
    parser.add_argument("question")
    parser.add_argument("options", nargs="+", help="two or more options to choose between")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="cap the field size (default: as many as the call budget allows)",
    )
    add_extension_args(parser)
    args = parser.parse_args(argv)

    options = list(dict.fromkeys(args.options))
    if len(options) < 2:
        print("need at least two distinct options")
        return 1
    # `is not None`, not truthiness: --max 0 is a request for an empty field,
    # and a negative one slices options off the end. Both are user error worth
    # naming rather than a field they did not ask for.
    if args.max is not None and args.max < 2:
        print(f"--max {args.max} leaves no tournament; it needs at least 2")
        return 1

    # n advocates + (n-1) judges must fit the call budget.
    capped = ""
    field = options
    spend = _Spend(MAX_CALLS)
    if args.max is not None:
        field = options[: args.max]
        if len(field) < len(options):
            capped = f" (capped from {len(options)})"
    if len(field) * 2 - 1 > spend.remaining:
        fits = (spend.remaining + 1) // 2
        capped = f" — only {fits}/{len(options)} fit the budget"
        field = field[:fits]

    STATS.reset()
    started = time.monotonic()
    _log_run(
        {
            "outcome": "run_start",
            "target": args.question[:60],
            "mode": "decide",
            "lenses": len(field),
        }
    )
    print(f"{len(field)} option{'s' if len(field) != 1 else ''}{capped}")

    try:
        brief = asyncio.run(decide(args.question, field, spend, extensions_from(args)))
    except (DecideFailed, BudgetExhausted, ExtensionError) as exc:
        print(f"decision did not complete: {exc}\n{STATS.summary()}")
        return 1

    render(brief)
    print(f"\n{STATS.summary()}")
    print(f"wall clock: {time.monotonic() - started:.0f}s")
    # A provisional winner is not a decision, an option that never played is
    # not a loss, and a field that never got entered is not a field. Same rule
    # upgrade() follows over unchecked dependencies.
    return 1 if brief["undecided"] or brief["no_case"] or brief["winner"] is None or capped else 0


if __name__ == "__main__":
    raise SystemExit(main())
