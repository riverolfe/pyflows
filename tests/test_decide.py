"""decide: the tournament example."""

import asyncio
import re

import jsonschema
import pytest

import pyflows.examples.decide as dc
from conftest import patched


def _tournament(dead_options=(), dead_judges=(), winner=0):
    """A fake ask() that records every prompt it is handed.

    `dead_options` are the options whose advocate dies. `dead_judges` are
    0-based indices into the sequence of matchups. `winner` is what every
    live judge answers, as an index into the order the prompt presented.
    """
    prompts = []

    async def fake(prompt, schema, **kw):
        prompts.append(prompt)
        if "strongest honest case" in prompt:
            match = re.search(r"case for option \d+: (.+)", prompt)
            assert match, "advocate prompt lost the option it was asked to argue"
            option = match.group(1)
            if option in dead_options:
                return None
            return {"plan": f"do {option}", "caveat": f"{option} is fiddly"}
        if "Pick the stronger" in prompt:
            if dead_judges and len([p for p in prompts if "Pick the stronger" in p]) - 1 in (
                dead_judges
            ):
                return None
            return {"winner": winner, "why": "sharper plan"}
        raise AssertionError(f"unexpected stage: {prompt[:40]}")

    # Validate against the schema like ask() does against a real agent, so a
    # fake cannot drift into a shape the schema does not describe.
    original = fake

    async def checked(prompt, schema, **kw):
        out = await original(prompt, schema, **kw)
        if out is not None:
            jsonschema.validate(out, schema)
        return out

    return checked, prompts


# -------------------------------------------------------------- advocates


def test_an_option_whose_advocate_died_is_reported_not_silently_dropped():
    # An option that never played did not lose; dropping it quietly would make
    # a 2-option tournament look like a unanimous result.
    fake, _ = _tournament(dead_options=("sqlite",))
    with patched(dc, "ask", fake):
        brief = asyncio.run(dc.decide("which db?", ["postgres", "mysql", "sqlite"]))

    assert brief["no_case"] == ["sqlite"]
    assert brief["winner"]["option"] != "sqlite"


def test_no_surviving_advocates_is_a_failure_not_a_winner():
    fake, _ = _tournament(dead_options=("a", "b"))
    with patched(dc, "ask", fake), pytest.raises(dc.DecideFailed):
        asyncio.run(dc.decide("which?", ["a", "b"]))


def test_a_tournament_needs_two_options():
    fake, _ = _tournament()
    with patched(dc, "ask", fake), pytest.raises(dc.DecideFailed):
        asyncio.run(dc.decide("which?", ["only"]))


# --------------------------------------------------------------- bracket


def test_a_dead_judge_advances_both_instead_of_eliminating_one():
    # Picking a winner the judge never gave is an elimination nobody voted
    # for, and the champion inherits a field that shrank for no stated reason.
    fake, _ = _tournament(dead_judges={0})
    with patched(dc, "ask", fake):
        brief = asyncio.run(dc.decide("which db?", ["postgres", "mysql", "sqlite"]))

    assert brief["undecided"] == 1
    assert brief["rounds"][0]["outcome"] == "undecided"
    # Both advanced: the undecided match is the two that were on the floor.
    assert set(brief["rounds"][0]["match"]) == {"postgres", "mysql"}


def test_single_elimination_costs_one_judge_per_match_not_a_round_robin():
    # 4 options = 4 advocates + 3 judges. Round robin would be 6.
    fake, prompts = _tournament()
    with patched(dc, "ask", fake):
        asyncio.run(dc.decide("which db?", ["a", "b", "c", "d"]))

    advocates = [p for p in prompts if "strongest honest case" in p]
    judges = [p for p in prompts if "Pick the stronger" in p]
    assert (len(advocates), len(judges)) == (4, 3)


def test_an_odd_field_gives_the_last_option_a_bye():
    fake, prompts = _tournament()
    with patched(dc, "ask", fake):
        asyncio.run(dc.decide("which db?", ["a", "b", "c"]))

    # 3 options: b fights a, c sits the round out, then the winner meets c.
    assert len([p for p in prompts if "Pick the stronger" in p]) == 2


def test_the_winner_index_reads_the_presented_order_not_the_argument_order():
    # Judges see "Candidate 0/1" with no names, so index 0 means whatever was
    # presented first — which is seeded, so it is not necessarily options[0].
    for index in (0, 1):
        fake, prompts = _tournament(winner=index)
        with patched(dc, "ask", fake):
            brief = asyncio.run(dc.decide("which db?", ["alpha", "beta"]))
        judge = next(p for p in prompts if "Pick the stronger" in p)
        shown = re.findall(r"\[(\d)\] (\w+)", judge)
        assert brief["winner"]["option"] == shown[index][1]


def test_a_judge_that_names_an_out_of_range_candidate_fails_validation():
    # The enum is what stops this: winner=2 is a judge pointing at a candidate
    # that was never offered, and index -1 for "the first" is the same mistake.
    # Caught by the schema, so ask() spends its one correction retry on it
    # instead of the winner being nonsense after the fact.
    for bad in (2, -1):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"winner": bad, "why": "the third one"}, dc.PAIR)
