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
- **The judges are blind to identity.** A judge sees "Candidate 1" and
  "Candidate 2", never the option name, so it cannot pick on brand rather than
  merit — and the two are presented in an order seeded by their contents, so a
  judge with a first-position bias cannot win every match by default.

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

from ..orch import MAX_CALLS, STATS, BudgetExhausted, _log_run, _Spend, ask, gather

# No tools. This is a judgement workflow, not a code one: the advocates argue
# from the question, they do not go read the repo. Grounding a decision in
# actual code means passing the relevant facts in the question — see the
# ponytail note in decide().
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


async def decide(question: str, options: list[str], spend: _Spend | None = None) -> dict:
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
                f"soften your argument.",
                CASE,
                spend=spend,
                label=f"case/{opt}"[:32],
                system="You advocate one option, alone. Be its best advocate.",
            )
            for i, opt in enumerate(options)
        ]
    )

    # An option whose advocate died did not lose the tournament, it never
    # played. Different thing, and the difference is the whole point.
    contenders: list[dict[str, Any]] = [
        {"option": o, "case": c} for o, c in zip(options, cases, strict=True) if c
    ]
    no_case = [o for o, c in zip(options, cases, strict=True) if not c]
    if no_case:
        print(f"  {len(no_case)} option(s) got no case and cannot compete: {', '.join(no_case)}")
    if len(contenders) < 2:
        raise DecideFailed(
            f"only {len(contenders)}/{len(options)} options were argued; that is not a decision"
        )

    rounds: list[dict] = []
    undecided = 0
    while len(contenders) > 1:
        matchups = [contenders[i : i + 2] for i in range(0, len(contenders), 2)]
        shown = [_order(*m) for m in matchups if len(m) == 2]

        verdicts = await gather(
            [
                ask(
                    f"{question}\n\nPick the stronger of these two plans.\n\n"
                    + "\n\n".join(
                        f"[{i}] {c['option']}\n    Plan: {c['case']['plan']}\n"
                        f"    Weakest point: {c['case']['caveat']}"
                        for i, c in enumerate(pair)
                    )
                    + "\n\nJudge on merit, not on which is written first. Pick "
                    'one — "both are fine" is not a verdict you can return. '
                    "Return the index of the better plan.",
                    PAIR,
                    spend=spend,
                    label=f"judge/{pair[0]['option'][:12]}"[:32],
                    system="You are an impartial judge. One winner or nothing.",
                )
                for pair in shown
            ]
        )

        verdicts = iter(verdicts)
        nxt = []
        for m in matchups:
            if len(m) == 1:
                nxt.append(m[0])  # bye
                continue
            pair = _order(*m)
            verdict = next(verdicts)
            if verdict is None:
                # Dead judge. Advancing both keeps the bracket honest and the
                # count honest; picking one would be an elimination nobody
                # voted for, and the winner would inherit a decided field.
                undecided += 1
                rounds.append({"match": [c["option"] for c in pair], "outcome": "undecided"})
                nxt.extend(m)
                continue
            winner, loser = (pair[0], pair[1]) if verdict["winner"] == 0 else (pair[1], pair[0])
            rounds.append(
                {"match": [c["option"] for c in pair], "outcome": "winner", "why": verdict["why"]}
            )
            nxt.append(winner)
            loser["eliminated_by"] = winner["option"]
        contenders = nxt

    champion = contenders[0]
    return {
        "question": question,
        "options": options,
        "winner": {"option": champion["option"], "plan": champion["case"]["plan"]},
        "rounds": rounds,
        "no_case": no_case,
        # The winner is provisional whenever a judge died: the bracket above it
        # had a matchup nobody scored.
        "undecided": undecided,
        "elapsed": time.monotonic() - started,
    }


def render(brief: dict) -> None:
    print(f"\n{brief['question']}")
    print(f"\nbest option: {brief['winner']['option']}\n  {brief['winner']['plan']}")

    print("\nrounds:")
    for r in brief["rounds"]:
        a, b = r["match"]
        if r["outcome"] == "undecided":
            print(f"  {a}  vs  {b}  —  no judge; both advance")
        else:
            print(f"  {a}  beats  {b}\n    {r['why']}")

    if brief["undecided"]:
        print(
            f"\n{brief['undecided']} matchup(s) went undecided — the winner is "
            f"provisional, not chosen."
        )
    if brief["no_case"]:
        print(f"{len(brief['no_case'])} option(s) never competed: {', '.join(brief['no_case'])}")


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
    args = parser.parse_args(argv)

    options = list(dict.fromkeys(args.options))
    if len(options) < 2:
        print("need at least two distinct options")
        return 1

    # n advocates + (n-1) judges. Say so when the budget cannot hold the field,
    # the way upgrade() does — a tournament that quietly enters 5 of 12 options
    # reports a winner as though 12 played.
    capped = ""
    field = options
    spend = _Spend(MAX_CALLS)
    if args.max:
        field = options[: args.max]
        capped = f" (capped from {len(options)})"
    elif len(options) * 2 - 1 > spend.remaining:
        field = options[: (spend.remaining + 1) // 2]
        capped = f" — only {len(field)}/{len(options)} fit the budget"

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
    print(f"{len(field)} options{capped}")

    try:
        brief = asyncio.run(decide(args.question, field, spend))
    except (DecideFailed, BudgetExhausted) as exc:
        print(f"decision did not complete: {exc}\n{STATS.summary()}")
        return 1

    render(brief)
    print(f"\n{STATS.summary()}")
    print(f"wall clock: {time.monotonic() - started:.0f}s")
    # A provisional winner is not a decision, and an option that never played
    # is not a loss. Same rule upgrade() follows over unchecked dependencies.
    return 1 if brief["undecided"] or brief["no_case"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
