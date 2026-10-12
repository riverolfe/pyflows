"""research: the example workflow."""

import asyncio

import pytest

import pyflows.examples.research as research
from conftest import patched


def test_synthesizer_is_told_to_report_contradictions_not_resolve_them():
    # The synthesizer's job is to report disagreement, not to pick a winner.
    captured = {}

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            captured["prompt"] = prompt
            return {
                "answer": "a",
                "agreements": [],
                "contradictions": [],
                "gaps": [],
                "sources": [],
            }
        return {"findings": []}

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", verify=False))
    assert "Do not resolve contradictions by picking a side" in captured["prompt"]


def test_research_sweep_agents_are_blind_to_each_other():
    # Blindness is the point — a searcher that saw the others' claims would
    # just echo them back.
    seen = []

    async def fake(prompt, schema, **kw):
        seen.append(prompt)
        return {"findings": []}

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", verify=False))
    sweeps = [p for p in seen if "Research this question" in p]
    assert len(sweeps) == len(research.ANGLES), len(sweeps)
    for prompt in sweeps:
        assert "Claims gathered" not in prompt, "sweep must not see other angles"


def test_research_all_angles_dead_is_not_a_clean_run():
    async def dead(*a, **kw):
        return None

    with patched(research, "ask", dead), pytest.raises(research.ResearchFailed):
        asyncio.run(research.research("q", verify=False))


def test_research_flags_unsupported_citations():
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return {
                "answer": "a",
                "agreements": [],
                "contradictions": [],
                "gaps": [],
                "sources": [
                    {"claim": "uv ships 3.14t builds", "url": "http://a"},
                    {"claim": "uv prefers the GIL build", "url": "http://b"},
                ],
            }
        if "actually support" in prompt:
            supported = "http://b" in prompt  # only one holds up
            return {"supported": supported, "reason": "checked"}
        return {"findings": []}

    with patched(research, "ask", fake):
        out = asyncio.run(research.research("q", depth=2))

    assert out["verified"] == 1, out
    assert [b["url"] for b in out["unsupported"]] == ["http://a"]
    assert out["unsupported"][0]["why"] == "checked"


def test_citations_carry_a_real_claim_not_a_bare_url():
    # The bug this fixes: sources used to be bare URLs, so the verifier was
    # handed (url, url) and could only report "this page exists".
    pairs = research._citations(
        {
            "sources": [
                {"claim": "uv ships 3.14t builds", "url": "http://a"},
                {"claim": "http://a", "url": "http://a"},  # self-referential
                {"claim": "", "url": "http://b"},  # no assertion
                {"claim": "x", "url": ""},  # no source
                {"claim": "other claim, same url", "url": "http://a"},  # deduped
            ]
        }
    )
    assert pairs == [("uv ships 3.14t builds", "http://a")], pairs


def test_citations_ignore_legacy_string_sources():
    # Tolerate the pre-fix shape rather than crashing on an old brief.
    assert research._citations({"sources": ["http://a", "http://b"]}) == []


def test_verify_prompt_states_the_actual_assertion():
    # The verifier must receive the claim, not a bare URL.
    seen = []

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return {
                "answer": "a",
                "agreements": [],
                "contradictions": [],
                "gaps": [],
                "sources": [{"claim": "uv ships 3.14t builds", "url": "http://docs.astral.sh/x"}],
            }
        if "actually support" in prompt:
            seen.append(prompt)
            return {"supported": True, "reason": "ok"}
        return {"findings": []}

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", depth=1))

    assert seen and "Claim: uv ships 3.14t builds" in seen[0], seen
