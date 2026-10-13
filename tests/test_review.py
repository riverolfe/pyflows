"""review: the example workflow."""

import asyncio

import pytest

import pyflows.examples.review as cr
import pyflows.orch as orch
from conftest import FINDING, patched


def test_batching_cuts_agent_count_and_keeps_findings_distinct(monkeypatch):
    # 12 findings at batch=5 is 3 verifiers, not 12 — and indices are
    # per-batch, so batch k covers globals [5k..5k+4].
    monkeypatch.setattr(cr, "BATCH_SIZE", 5)
    verifiers = []

    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            verifiers.append(prompt)
            n = len([line for line in prompt.splitlines() if line.startswith("[")])
            return {
                "verdicts": [
                    {"index": i, "refuted": i % 2 == 0, "reason": f"r{i}"} for i in range(n)
                ]
            }
        return {
            "findings": [
                {**FINDING, "file": f"f{i}.py", "line": i + 1, "summary": f"s{i}"}
                for i in range(12)
            ]
        }

    with patched(cr, "ask", fake):
        found = asyncio.run(cr.review("."))["confirmed"]

    assert len(verifiers) == 3, f"expected 3 verifiers, got {len(verifiers)}"
    assert {f["summary"] for f in found} == {"s1", "s3", "s6", "s8", "s11"}


def test_skipped_verdict_is_unconfirmed_not_confirmed(monkeypatch):
    # An agent that omits an index has not refuted it.
    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            return {"verdicts": [{"index": 0, "refuted": True, "reason": "no"}]}
        # Distinct lines: three copies of one FINDING dedup to a single claim,
        # and the verdict for index 0 then covers it — nothing is ever skipped.
        return {"findings": [{**FINDING, "line": i} for i in range(3)]}

    with patched(cr, "ask", fake):
        out = asyncio.run(cr.review("."))
    assert out == {"confirmed": [], "unconfirmed": 2, "dead": 0}, out


def test_all_lenses_dead_is_not_a_clean_run():
    async def dead(*a, **kw):
        return None

    with patched(cr, "ask", dead), pytest.raises(cr.ReviewFailed):
        asyncio.run(cr.review("."))


def test_all_verifiers_dead_is_not_a_clean_run():
    # Batching reintroduced this after the find stage was already guarded.
    async def fake(prompt, schema, **kw):
        return None if "REFUTE" in prompt else {"findings": [FINDING]}

    with patched(cr, "ask", fake), pytest.raises(cr.ReviewFailed):
        asyncio.run(cr.review("."))


def test_live_lenses_with_no_findings_is_a_clean_run():
    # The bug: zero findings meant zero batches, and any([]) is False — a clean
    # tree raised "all 0 verification batches failed" and exited non-zero.
    async def fake(prompt, schema, **kw):
        assert "REFUTE" not in prompt, "nothing to verify, no verifier should run"
        return {"findings": []}

    with patched(cr, "ask", fake):
        assert asyncio.run(cr.review(".")) == {"confirmed": [], "unconfirmed": 0, "dead": 0}


def test_review_survives_a_partially_dead_fleet(tmp_path, capsys):
    calls = []

    async def flaky(prompt, schema, **kw):
        calls.append(prompt)
        if "REFUTE" in prompt:
            return {"verdicts": [{"index": 0, "refuted": True, "reason": "nope"}]}
        if len(calls) == 1:
            return None
        return {"findings": [FINDING]}

    with patched(cr, "ask", flaky):
        out = asyncio.run(cr.review("."))
        calls.clear()
        code = cr.main(["review", str(tmp_path), "--lenses", "2"])

    assert out["confirmed"] == [] and out["dead"] == 1, out
    assert code == 1
    assert "1/2 lenses failed" in capsys.readouterr().out


def test_dead_lenses_beside_empty_ones_are_not_a_clean_run(tmp_path):
    calls = []

    async def fake(prompt, schema, **kw):
        calls.append(prompt)
        return None if len(calls) == 1 else {"findings": []}

    with patched(cr, "ask", fake):
        assert cr.main(["review", str(tmp_path), "--lenses", "2"]) == 1


def test_quick_mode_uses_one_lens_and_one_verifier(monkeypatch):
    # The whole point: 2 agents, not 7.
    monkeypatch.setattr(cr, "BATCH_SIZE", 5)
    verifiers, finders = [], []

    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            verifiers.append(prompt)
            n = len([line for line in prompt.splitlines() if line.startswith("[")])
            return {
                "verdicts": [{"index": i, "refuted": False, "reason": "holds"} for i in range(n)]
            }
        finders.append(prompt)
        return {"findings": [{**FINDING, "file": f"f{i}.py", "line": i + 1} for i in range(11)]}

    with patched(cr, "ask", fake):
        found = asyncio.run(cr.review(".", cr.QUICK_LENSES, cr.MAX_CALLS))["confirmed"]

    assert len(finders) == 1, f"quick mode should use one lens, used {len(finders)}"
    assert len(verifiers) == 1, f"quick mode should use one verifier, used {len(verifiers)}"
    assert len(found) == 11, "all findings judged in the single batch"


def test_a_partially_verified_review_exits_non_zero(tmp_path, capsys):
    # The bug: one dead verifier batch out of two only printed "N unconfirmed"
    # and exited 0 — CI read a review that never judged half its findings as clean.
    batches = []

    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            batches.append(prompt)
            if len(batches) == 1:
                return None
            n = len([line for line in prompt.splitlines() if line.startswith("[")])
            return {"verdicts": [{"index": i, "refuted": True, "reason": "no"} for i in range(n)]}
        return {"findings": [{**FINDING, "line": i} for i in range(6)]}

    with patched(cr, "ask", fake):
        code = cr.main(["review", str(tmp_path), "--lenses", "1"])

    assert len(batches) == 2, batches
    assert code == 1
    assert "NOT verified" in capsys.readouterr().out


def test_a_fully_refuted_review_exits_zero(tmp_path):
    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            return {"verdicts": [{"index": 0, "refuted": True, "reason": "no"}]}
        return {"findings": [FINDING]}

    with patched(cr, "ask", fake):
        assert cr.main(["review", str(tmp_path), "--lenses", "1"]) == 0


GRANT = orch.Extensions(skills=("s",), inherit=True)


def _exts_seen(fake):
    seen = []

    async def spy(prompt, schema, **kw):
        seen.append(kw.get("ext"))
        return await fake(prompt, schema, **kw)

    return spy, seen


def test_every_review_agent_gets_the_granted_extensions():
    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            return {"verdicts": [{"index": 0, "refuted": True, "reason": "no"}]}
        return {"findings": [FINDING]}

    spy, seen = _exts_seen(fake)
    with patched(cr, "ask", spy):
        asyncio.run(cr.review(".", ext=GRANT))
    assert len(seen) == len(cr.LENSES) + 1 and all(e is GRANT for e in seen), seen


def test_the_cli_forwards_extension_flags_to_every_workflow(monkeypatch):
    import pyflows.examples.decide as dc
    import pyflows.examples.research as rs
    import pyflows.examples.upgrade as up

    got = {}
    for name, mod in (("research", rs), ("upgrade", up), ("decide", dc)):
        monkeypatch.setattr(mod, "main", lambda argv, name=name: got.setdefault(name, argv) and 0)

    flags = ["--skill", "s", "--plugin", "/p", "--inherit-settings"]
    cr.main(["research", "q", *flags])
    cr.main(["upgrade", ".", *flags])
    cr.main(["decide", "q", "a", "b", *flags])
    for name in ("research", "upgrade", "decide"):
        assert got[name][-len(flags) :] == flags, (name, got[name])


def test_lens_picks_lenses_by_name_and_lenses_caps_them(tmp_path):
    systems = []

    async def fake(prompt, schema, **kw):
        if "REFUTE" not in prompt:
            systems.append(kw["system"])
        return {"findings": []}

    with patched(cr, "ask", fake):
        assert (
            cr.main(["review", str(tmp_path), "--lens", "performance", "--lens", "error-handling"])
            == 0
        )
        picked = list(systems)
        systems.clear()
        assert (
            cr.main(
                [
                    "review",
                    str(tmp_path),
                    "--lens",
                    "security",
                    "--lens",
                    "performance",
                    "--lenses",
                    "1",
                ]
            )
            == 0
        )

    assert [s.split(" for ")[1].split(" defects")[0] for s in picked] == [
        "performance",
        "error-handling",
    ]
    assert len(systems) == 1 and "security" in systems[0]


def test_an_unknown_lens_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        cr.main(["review", str(tmp_path), "--lens", "style"])
