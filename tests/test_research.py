"""research: the example workflow."""

import asyncio

import pytest

import pyflows.examples.research as research
import pyflows.orch as orch
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


def _brief_with_sources(*urls):
    return {
        "answer": "a",
        "agreements": [],
        "contradictions": [],
        "gaps": [],
        "sources": [{"claim": f"claim from {u}", "url": u} for u in urls],
    }


def test_a_dead_citation_check_is_unchecked_not_a_crash(capsys):
    # The bug: a verifier that died was kept as "unsupported", then
    # v.get("reason") on None raised AttributeError and lost the whole brief.
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources("http://a", "http://b")
        if "actually support" in prompt:
            return None if "http://a" in prompt else {"supported": True, "reason": "ok"}
        return {"findings": []}

    with patched(research, "ask", fake):
        out = asyncio.run(research.research("q", depth=2))
        code = research.main(["research", "q", "--depth", "2"])

    assert [c["url"] for c in out["unchecked"]] == ["http://a"], out
    assert out["unsupported"] == [] and out["verified"] == 1, out
    assert code == 1, "a citation nobody checked is not one that held up"
    text = capsys.readouterr().out
    assert "NOT checked" in text and "held up" not in text, text


def test_every_citation_check_dead_is_not_a_clean_run():
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources("http://a")
        if "actually support" in prompt:
            return None
        return {"findings": []}

    with patched(research, "ask", fake), pytest.raises(research.ResearchFailed):
        asyncio.run(research.research("q", depth=1))


def test_claims_without_a_url_never_reach_the_synthesizer(capsys):
    # The bug: they were counted as "dropped" and then passed on anyway, so
    # the synthesizer cited "(source: , …)" for a claim nothing could check.
    captured = {}

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            captured["prompt"] = prompt
            return _brief_with_sources()
        return {
            "findings": [
                {"claim": "sourced", "url": "http://a", "confidence": "high"},
                {"claim": "unsourced", "url": "", "confidence": "high"},
            ]
        }

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", angles=research.ANGLES[:1], verify=False))

    assert "sourced (source: http://a" in captured["prompt"]
    assert "unsourced" not in captured["prompt"], captured["prompt"]
    assert "1 claims had no URL" in capsys.readouterr().out


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


def test_citation_check_labels_keep_the_end_of_the_source():
    labels = []

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources(
                "file:///home/user/repo/fwlibs/anc-lib/doc/parser.md",
                "file:///home/user/repo/fwlibs/video-lib/doc/vpid.md",
            )
        if "actually support" in prompt:
            labels.append(kw["label"])
            return {"supported": True, "reason": "ok"}
        return {"findings": []}

    with patched(research, "ask", fake):
        asyncio.run(research.research("q", depth=2))

    assert len(set(labels)) == 2, labels
    assert labels[0].endswith("doc/parser.md") and labels[1].endswith("doc/vpid.md"), labels


def test_a_dead_search_angle_is_not_a_clean_run(capsys):
    calls = []

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources("http://a")
        if "actually support" in prompt:
            return {"supported": True, "reason": "ok"}
        calls.append(prompt)
        return None if len(calls) == 1 else {"findings": []}

    with patched(research, "ask", fake):
        code = research.main(["research", "q", "--angles", "2", "--depth", "1"])

    assert code == 1
    assert "1 search angles failed" in capsys.readouterr().out


GRANT = orch.Extensions(skills=("s",), inherit=True)


def _exts_seen(fake):
    seen = []

    async def spy(prompt, schema, **kw):
        seen.append(kw.get("ext"))
        return await fake(prompt, schema, **kw)

    return spy, seen


def test_every_research_agent_gets_the_granted_extensions():
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources("http://a")
        if "actually support" in prompt:
            return {"supported": True, "reason": "ok"}
        return {"findings": []}

    spy, seen = _exts_seen(fake)
    with patched(research, "ask", spy):
        asyncio.run(research.research("q", depth=1, ext=GRANT))
    assert len(seen) == len(research.ANGLES) + 2 and all(e is GRANT for e in seen), seen


def test_a_brief_with_no_checkable_citation_is_not_a_clean_run(capsys):
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return {**_brief_with_sources(), "sources": [{"claim": "x", "url": ""}]}
        return {"findings": []}

    with patched(research, "ask", fake):
        code = research.main(["research", "q"])
    text = capsys.readouterr().out
    assert code == 1
    assert "No citation could be checked" in text and "held up" not in text, text


def test_a_dead_synthesizer_is_a_failed_run(capsys):
    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return None
        return {"findings": [{"claim": "c", "source": "http://a", "confidence": "high"}]}

    with patched(research, "ask", fake):
        with pytest.raises(research.ResearchFailed, match="synthesis failed"):
            asyncio.run(research.research("q", verify=False))
        assert research.main(["research", "q", "--no-verify"]) == 1
    assert "research did not complete: synthesis failed" in capsys.readouterr().out


def test_only_depth_citations_are_checked():
    checked = []

    async def fake(prompt, schema, **kw):
        if "Synthesize" in prompt:
            return _brief_with_sources("http://a", "http://b", "http://c")
        if "actually support" in prompt:
            checked.append(prompt)
            return {"supported": True, "reason": "ok"}
        return {"findings": []}

    with patched(research, "ask", fake):
        out = asyncio.run(research.research("q", depth=1))
    assert len(checked) == 1 and out["verified"] == 1
