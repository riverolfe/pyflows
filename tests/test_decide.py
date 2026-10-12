"""decide: the tournament example."""

import asyncio
import re

import jsonschema
import pytest

import pyflows.examples.decide as dc
import pyflows.orch as orch
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
    # for. A 4-option field keeps shrinking via the other matchup, so the run
    # still finishes — with a provisional winner.
    fake, _ = _tournament(dead_judges={0})
    with patched(dc, "ask", fake):
        brief = asyncio.run(dc.decide("which db?", ["postgres", "mysql", "redis", "sqlite"]))

    assert brief["undecided"] == 1
    assert brief["rounds"][0]["outcome"] == "undecided"
    # Both advanced: the undecided match is the two that were on the floor.
    assert set(brief["rounds"][0]["match"]) == {"postgres", "mysql"}
    assert brief["winner"] is not None


def test_a_round_that_eliminates_nobody_ends_the_bracket_in_a_draw():
    # Both-advance is right for one matchup, but a round where nothing was
    # decided does not shrink the field — looping on it re-runs the same
    # matchups forever. Measured: 104k judge calls with no termination.
    fake, _ = _tournament(dead_judges={0})
    with patched(dc, "ask", fake):
        brief = asyncio.run(dc.decide("which db?", ["postgres", "mysql"]))

    assert brief["undecided"] == 1
    assert brief["winner"] is None
    assert set(brief["tied"]) == {"postgres", "mysql"}


def test_a_field_where_no_judge_ever_answers_terminates():
    fake, prompts = _tournament(dead_judges={0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    with patched(dc, "ask", fake):
        brief = asyncio.run(dc.decide("which db?", ["a", "b", "c", "d"]))

    assert brief["winner"] is None
    assert len(brief["tied"]) == 4
    # One round, then it stops — not one round per option, not forever.
    assert len([p for p in prompts if "Pick the stronger" in p]) == 2


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
    for index in (0, 1):
        fake, prompts = _tournament(winner=index)
        with patched(dc, "ask", fake):
            brief = asyncio.run(dc.decide("which db?", ["alpha", "beta"]))
        judge = next(p for p in prompts if "Pick the stronger" in p)
        shown = re.findall(r"Plan: do (\w+)", judge)
        assert brief["winner"]["option"] == shown[index]


def test_a_judge_that_names_an_out_of_range_candidate_fails_validation():
    # The enum is what stops this: winner=2 is a judge pointing at a candidate
    # that was never offered, and index -1 for "the first" is the same mistake.
    # Caught by the schema, so ask() spends its one correction retry on it
    # instead of the winner being nonsense after the fact.
    for bad in (2, -1):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"winner": bad, "why": "the third one"}, dc.PAIR)


# ------------------------------------------------------------------- the CLI


def test_a_cap_that_would_leave_no_tournament_is_refused(capsys):
    # --max 0 is falsy, so a truthiness test runs the FULL field instead of the
    # empty one asked for; --max -1 slices the last option off the end. Neither
    # is a tournament the user meant to enter.
    for bad in ("0", "-1", "-2"):
        assert dc.main(["which?", "a", "b", "--max", bad]) == 1
    assert "needs at least 2" in capsys.readouterr().out


def test_a_field_that_never_got_entered_is_not_a_clean_run(capsys):
    fake, _ = _tournament()
    with patched(dc, "ask", fake):
        code = dc.main(["which?", "a", "b", "c", "d", "--max", "2"])
    assert code == 1
    assert "capped from 4" in capsys.readouterr().out


def test_a_cap_above_the_field_size_does_not_claim_a_truncation(capsys):
    # options[:99] drops nothing; saying "capped from 2" asserts an edit that
    # never happened.
    fake, _ = _tournament()
    with patched(dc, "ask", fake):
        assert dc.main(["which?", "a", "b", "--max", "99"]) == 0
    assert "capped" not in capsys.readouterr().out


def test_judges_see_candidates_not_option_names_and_no_agent_gets_tools():
    seen = []
    fake, prompts = _tournament(winner=0)

    async def spy(prompt, schema, **kw):
        seen.append(kw.get("tools"))
        return await fake(prompt, schema, **kw)

    with patched(dc, "ask", spy):
        asyncio.run(dc.decide("which db?", ["alpha", "beta", "gamma"]))

    judges = [p for p in prompts if "Pick the stronger" in p]
    assert judges
    for judge in judges:
        headings = re.findall(r"^\[\d\] (.+)$", judge, flags=re.M)
        assert headings == ["Candidate 0", "Candidate 1"], headings
    assert seen and all(t == () for t in seen), seen


GRANT = orch.Extensions(skills=("s",), inherit=True)


def _exts_seen(fake):
    seen = []

    async def spy(prompt, schema, **kw):
        seen.append(kw.get("ext"))
        return await fake(prompt, schema, **kw)

    return spy, seen


def test_every_decide_agent_gets_the_granted_extensions():
    fake, _ = _tournament()
    spy, seen = _exts_seen(fake)
    with patched(dc, "ask", spy):
        asyncio.run(dc.decide("q", ["a", "b", "c"], ext=GRANT))
    assert len(seen) == 5 and all(e is GRANT for e in seen), seen


@pytest.mark.parametrize("kw", [{"dead_judges": (0,)}, {"dead_options": ("b",)}])
def test_a_provisional_or_short_field_decision_exits_non_zero(kw):
    fake, _ = _tournament(**kw)
    with patched(dc, "ask", fake):
        assert dc.main(["which?", "a", "b", "c"]) == 1


def test_a_clean_decision_exits_zero():
    fake, _ = _tournament()
    with patched(dc, "ask", fake):
        assert dc.main(["which?", "a", "b", "c"]) == 0


@pytest.mark.parametrize("extra", [[], ["--max", "30"]])
def test_the_field_is_capped_to_the_call_budget(extra, capsys):
    options = [f"o{i}" for i in range(30)]
    fake, prompts = _tournament()
    with patched(dc, "ask", fake):
        code = dc.main(["which?", *options, *extra])
    fits = (dc.MAX_CALLS + 1) // 2
    assert f"only {fits}/30 fit the budget" in capsys.readouterr().out
    assert len([p for p in prompts if "strongest honest case" in p]) == fits
    assert code == 1


def test_presentation_order_is_stable_and_not_argument_order():
    opts = [{"option": f"o{i}"} for i in range(8)]
    pairs = [(a, b) for a in opts for b in opts if a is not b]
    assert all(dc._order(a, b) == dc._order(b, a) for a, b in pairs)
    assert any(dc._order(a, b)[0] is b for a, b in pairs)
