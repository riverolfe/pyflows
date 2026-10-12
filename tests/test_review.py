"""review: the example workflow."""

import asyncio

import pytest

import pyflows.examples.review as cr
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
    assert out == {"confirmed": [], "unconfirmed": 2}, out


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
        assert asyncio.run(cr.review(".")) == {"confirmed": [], "unconfirmed": 0}


def test_review_survives_a_partially_dead_fleet():
    # One dead lens among four must not sink the run, and must not silently
    # become "this lens found nothing".
    calls = []

    async def flaky(prompt, schema, **kw):
        calls.append(prompt)
        if "REFUTE" in prompt:
            return {"verdicts": [{"index": 0, "refuted": True, "reason": "nope"}]}
        if len(calls) == 1:
            return None
        return {"findings": [FINDING]}

    with patched(cr, "ask", flaky):
        assert asyncio.run(cr.review("."))["confirmed"] == []


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
