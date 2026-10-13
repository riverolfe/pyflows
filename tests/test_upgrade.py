"""upgrade: the example workflow."""

import asyncio

import jsonschema
import pytest

import pyflows.examples.upgrade as up
import pyflows.orch as orch
from conftest import patched

DEP = {"name": "django", "current": "4.2", "direct": True, "uses": 12}
BREAK = {
    "file": "app/views.py",
    "line": 40,
    "dependency": "django",
    "summary": "django.utils.timezone.utc removed in 5.0",
    "failure": "ImportError at boot",
    "url": "https://docs.djangoproject.com/en/5.0/releases/5.0/",
}


def _agents(findings_per_dep, verdicts=None, sources=("https://example.com/changelog",)):
    """A fake ask() that records every prompt it is handed.

    `sources` defaults to one changelog, i.e. an agent that actually checked
    something. Pass `sources=()` to model an agent whose web tools failed.
    """
    prompts = []

    async def fake(prompt, schema, **kw):
        prompts.append((prompt, kw))
        out = _respond(prompt)
        # The real agent's output is validated against `schema` in ask(). So is
        # the fake's: a fake free to drift from the schema is how a module ends
        # up reading a key the schema does not define.
        jsonschema.validate(out, schema)
        return out

    def _respond(prompt):
        if "manifest" in prompt or "dependency manifest" in prompt:
            return {
                "manifest": "pyproject.toml",
                "dependencies": [
                    {"name": "django", "current": "4.2", "direct": True, "uses": 12},
                    {"name": "requests", "current": "2.28", "direct": True, "uses": 3},
                    {"name": "attrs", "current": "21.0", "direct": False, "uses": 0},
                ],
            }
        if "REFUTE" in prompt:
            if verdicts is not None:
                return verdicts
            n = len([ln for ln in prompt.splitlines() if ln.startswith("[")])
            return {
                "verdicts": [
                    {"index": i, "breaks_here": True, "reason": "confirmed"} for i in range(n)
                ]
            }
        return {"findings": findings_per_dep, "sources_checked": list(sources)}

    return fake, prompts


# ------------------------------------------------------------- inventory


def test_inventory_failure_is_not_an_empty_manifest():
    # "No manifest found" and "the inventory agent died" are different
    # outcomes, and the first must never be reported as the second.
    async def dead(*a, **kw):
        return None

    with patched(up, "ask", dead), pytest.raises(up.UpgradeFailed):
        asyncio.run(up.upgrade("."))


def test_manifest_with_no_dependencies_is_not_a_clean_run():
    async def empty(*a, **kw):
        return {"manifest": "pyproject.toml", "dependencies": []}

    with patched(up, "ask", empty), pytest.raises(up.UpgradeFailed):
        asyncio.run(up.upgrade("."))


def test_direct_dependencies_are_audited_before_transitive_ones():
    # An upgrade to a direct dep breaks callers; a transitive bump mostly does
    # not. Ranking by blast radius is what makes a limited budget useful.
    deps = [
        {"name": "attrs", "current": "1", "direct": False, "uses": 0},
        {"name": "requests", "current": "1", "direct": True, "uses": 2},
        {"name": "django", "current": "1", "direct": True, "uses": 50},
    ]
    assert [d["name"] for d in up._pick(deps, None)] == ["django", "requests", "attrs"]


def test_pick_is_deterministic_so_a_cached_rerun_stays_valid():
    # The cache keys on the prompt, so the same repo must select the same
    # dependencies or a second run audits something the first one skipped.
    deps = [{"name": f"p{i}", "current": "1", "direct": True, "uses": i} for i in range(8)]
    assert up._pick(deps, None) == up._pick(list(reversed(deps)), None)


def test_an_agent_without_web_tools_cannot_audit_an_upgrade():
    # The bug this guards: an agent handed only local tools can still return a
    # schema-valid "no breaks" having read no changelog. WebSearch/WebFetch in
    # TOOLS is the only reason the answer means anything — dropping them makes
    # every dependency silently look safe.
    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {
                "manifest": "package.json",
                "dependencies": [{"name": "django", "current": "4.2", "direct": True, "uses": 3}],
            }
        assert "WebSearch" in kw["tools"] and "WebFetch" in kw["tools"], kw["tools"]
        return {"findings": [], "sources_checked": ["https://example.com/5.0"]}

    with patched(up, "ask", fake):
        asyncio.run(up.upgrade(".", verify=False))


# ------------------------------------------------------------ fan-out


def test_fanout_width_comes_from_the_inventory_not_a_fixed_tuple():
    # This is what makes upgrade different from review and research: the
    # number of agents is decided by what stage one found.
    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {
                "manifest": "package.json",
                "dependencies": [
                    {"name": f"p{i}", "current": "1", "direct": True, "uses": i} for i in range(7)
                ],
            }
        return {"findings": []}

    agents = []

    async def counting(prompt, schema, **kw):
        agents.append(kw.get("label"))
        return await fake(prompt, schema, **kw)

    with patched(up, "ask", counting):
        asyncio.run(up.upgrade(".", verify=False))

    assert len([a for a in agents if a.startswith("dep/")]) == 7, agents


# ----------------------------------------------- checked vs merely silent


def test_sources_checked_is_required_by_the_schema():
    # Make it optional and an agent can simply omit it — which is exactly the
    # silent-failure shape the field exists to prevent, since a missing key
    # reads the same as an empty one.
    assert "sources_checked" in up.BREAKING["required"]


def test_a_fully_unchecked_run_is_not_a_clean_run():
    # Nothing was checked, so nothing may be concluded. Exits non-zero rather
    # than printing "the deps look safe to bump".
    fake, _ = _agents([], sources=())
    with patched(up, "ask", fake), pytest.raises(up.UpgradeFailed):
        asyncio.run(up.upgrade(".", limit=2, verify=True))


def test_each_dependency_is_audited_or_unchecked_on_its_own_sources():
    # The failure this guards: WebFetch dies, the agent still returns a clean
    # schema-valid "no breaks", and the run reports the deps as safe to bump.
    # It cost 0.41M tokens and 917s to learn nothing that way. Mixed on
    # purpose: one blank agent must not mark the checked one unchecked either.
    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {
                "manifest": "package.json",
                "dependencies": [
                    {"name": "django", "current": "4.2", "direct": True, "uses": 9},
                    {"name": "requests", "current": "2.28", "direct": True, "uses": 9},
                ],
            }
        # django checked a changelog; requests could not reach one.
        checked = "requests" in prompt
        return {
            "findings": [],
            "sources_checked": ["https://example.com/5.0"] if checked else [],
        }

    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", verify=False))

    assert out["audited"] == 1 and out["unchecked"] == 1, out


def test_a_fully_checked_run_exits_zero(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([])

    with patched(up, "ask", fake):
        assert up.main([str(tmp_path), "--no-verify"]) == 0


def test_the_exit_code_reports_a_partial_audit(tmp_path):
    # The exit code is the only thing a CI job reads. A partial audit that
    # exits 0 gets merged and the unchecked dependencies are never re-run.
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([], sources=())

    with patched(up, "ask", fake):
        assert up.main([str(tmp_path), "--no-verify"]) == 1


def test_render_does_not_claim_safety_over_unchecked_dependencies(capsys):
    # "no confirmed breaks" is only a statement about what was checked.
    fake, _ = _agents([], sources=())
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1, verify=False))
    capsys.readouterr()
    up.render(out)
    text = capsys.readouterr().out

    assert "safe to bump" not in text, text
    assert "NOT checked" in text, text


def test_render_still_claims_safety_when_everything_was_checked(capsys):
    fake, _ = _agents([])
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1, verify=False))
    capsys.readouterr()
    up.render(out)
    assert "safe to bump" in capsys.readouterr().out


def test_all_agents_dead_is_not_a_clean_run():
    async def dead_after_inventory(prompt, schema, **kw):
        if "manifest" in prompt:
            return {"manifest": "pyproject.toml", "dependencies": [DEP]}
        return None

    with patched(up, "ask", dead_after_inventory), pytest.raises(up.UpgradeFailed):
        asyncio.run(up.upgrade(".", verify=False))


def test_budget_caps_the_fanout_and_says_so(capsys):
    # The run cannot cover 40 dependencies inside MAX_CALLS. Silently auditing
    # the first 12 reads as "these 12 are the only ones that matter".
    many = [{"name": f"p{i}", "current": "1", "direct": True, "uses": i} for i in range(40)]

    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {"manifest": "package.json", "dependencies": many}
        return {"findings": []}

    with patched(up, "ask", fake):
        asyncio.run(up.upgrade(".", verify=False))

    out = capsys.readouterr().out
    assert "fit the budget" in out, out


def test_explicit_limit_is_not_reported_as_a_budget_shortfall(capsys):
    # The user asked for 2. That is not a ceiling we failed to reach.
    fake, _ = _agents([])
    with patched(up, "ask", fake):
        asyncio.run(up.upgrade(".", limit=2, verify=False))
    assert "fit the budget" not in capsys.readouterr().out


# ------------------------------------------------------------- verifying


def test_suspected_breaks_are_refuted_before_they_are_reported():
    fake, prompts = _agents([BREAK])
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1))

    assert len(out["confirmed"]) == 1, out
    verify = [p for p, _ in prompts if "REFUTE" in p]
    assert verify, "a candidate break was reported without a refutation attempt"
    assert "not evidence that this repo hits it" in verify[0]


def test_a_refuted_break_is_reported_as_refuted_not_dropped():
    # Silently dropping it makes a clean run and a wrong call look identical.
    fake, _ = _agents(
        [BREAK], verdicts={"verdicts": [{"index": 0, "breaks_here": False, "reason": "unaffected"}]}
    )
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1))

    assert out["confirmed"] == []
    assert len(out["refuted"]) == 1
    assert out["refuted"][0]["why"] == "unaffected"


def test_a_dead_verifier_leaves_its_claims_unconfirmed_not_refuted(tmp_path, capsys):
    # The bug: a verifier that died put its claims in "refuted", rendered as
    # "upstream changes that do not hit this repo" — then "safe to bump", exit 0.
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([BREAK])

    async def dead_verifier(prompt, schema, **kw):
        return None if "REFUTE" in prompt else await fake(prompt, schema, **kw)

    with patched(up, "ask", dead_verifier):
        out = asyncio.run(up.upgrade(".", limit=1))
        code = up.main([str(tmp_path), "--limit", "1"])

    assert out["refuted"] == [] and len(out["unconfirmed"]) == 1, out
    assert code == 1
    assert "safe to bump" not in capsys.readouterr().out


def test_a_skipped_claim_is_unconfirmed_not_refuted():
    fake, _ = _agents([BREAK, {**BREAK, "line": 41}], verdicts={"verdicts": [
        {"index": 0, "breaks_here": False, "reason": "unaffected"}
    ]})  # fmt: skip
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1))

    assert [e["why"] for e in out["refuted"]] == ["unaffected"]
    assert [e["finding"]["line"] for e in out["unconfirmed"]] == [41]


def test_findings_beyond_depth_are_unconfirmed_not_safe(tmp_path, capsys):
    # The bug: --depth 3 of 10 findings verified 3, refuted all 3, and the
    # other 7 vanished from the verdict — "safe to bump" over 7 unread claims.
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents(
        [{**BREAK, "line": n} for n in range(4)],
        verdicts={"verdicts": [{"index": 0, "breaks_here": False, "reason": "no"}]},
    )
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1, depth=1))
        code = up.main([str(tmp_path), "--limit", "1", "--depth", "1"])
    text = capsys.readouterr().out

    assert len(out["refuted"]) == 1 and len(out["unconfirmed"]) == 3, out
    assert code == 1
    assert "safe to bump" not in text and "NOT verified" in text, text


def test_no_verify_does_not_call_candidate_breaks_safe(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([BREAK])
    with patched(up, "ask", fake):
        assert up.main([str(tmp_path), "--no-verify"]) == 1
    assert "safe to bump" not in capsys.readouterr().out


def test_a_finding_with_no_changelog_url_cannot_be_verified(capsys):
    # The verifier can only fetch a URL; without one it would confirm a
    # breaking change it never saw evidence for.
    fake, prompts = _agents([{**BREAK, "url": ""}])
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1))

    assert out["findings"] == []
    assert not [p for p, _ in prompts if "REFUTE" in p]
    assert "cannot be verified" in capsys.readouterr().out


def test_the_same_break_from_two_agents_is_one_finding():
    # django and requests both flag the same line; it is one line item.
    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {
                "manifest": "package.json",
                "dependencies": [
                    {"name": "django", "current": "4.2", "direct": True, "uses": 5},
                    {"name": "requests", "current": "2.28", "direct": True, "uses": 5},
                ],
            }
        return {"findings": [BREAK]}

    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", verify=False))
    assert len(out["findings"]) == 1, out["findings"]


def test_two_dependencies_breaking_the_same_line_stay_separate():
    # Deduplicating on (file, line) alone merges two distinct problems into
    # one and reports only whichever agent happened to answer first — the
    # second dependency's upgrade then looks clean.
    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {
                "manifest": "package.json",
                "dependencies": [
                    {"name": "django", "current": "4.2", "direct": True, "uses": 5},
                    {"name": "requests", "current": "2.28", "direct": True, "uses": 5},
                ],
            }
        # Same file:line, different dependency — two real breaks.
        return {"findings": [BREAK, {**BREAK, "dependency": "requests", "summary": "s"}]}

    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", verify=False))

    assert len(out["findings"]) == 2, out["findings"]
    assert {f["dependency"] for f in out["findings"]} == {"django", "requests"}


def test_no_verify_skips_the_refutation_stage():
    fake, prompts = _agents([BREAK])
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1, verify=False))
    assert not [p for p, _ in prompts if "REFUTE" in p]
    assert out["findings"] == [BREAK]


# ------------------------------------------------------------------ cli


def test_manifest_lookup_prefers_the_first_known_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "package.json").write_text("")
    assert up.find_manifest(tmp_path) == "package.json"

    (tmp_path / "package.json").unlink()
    assert up.find_manifest(tmp_path) == "pyproject.toml"


def test_missing_manifest_exits_before_spending_anything(tmp_path, capsys):
    # A wrong path or a repo with no manifest is a CLI error, not a run that
    # discovers nothing and exits clean.
    assert up.main([str(tmp_path / "nope")]) == 1
    assert "no such path" in capsys.readouterr().out

    assert up.main([str(tmp_path)]) == 1
    assert "no manifest" in capsys.readouterr().out


def test_a_dead_dependency_agent_is_not_a_clean_audit(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([])

    async def one_dead(prompt, schema, **kw):
        if "`requests`" in prompt:
            return None
        return await fake(prompt, schema, **kw)

    with patched(up, "ask", one_dead):
        code = up.main([str(tmp_path), "--no-verify"])

    text = capsys.readouterr().out
    assert code == 1
    assert "1/3 dependency agents failed" in text and "safe to bump" not in text, text


GRANT = orch.Extensions(skills=("s",), inherit=True)


def _exts_seen(fake):
    seen = []

    async def spy(prompt, schema, **kw):
        seen.append(kw.get("ext"))
        return await fake(prompt, schema, **kw)

    return spy, seen


def test_every_upgrade_agent_gets_the_granted_extensions():
    fake, _ = _agents([BREAK])
    spy, seen = _exts_seen(fake)
    with patched(up, "ask", spy):
        asyncio.run(up.upgrade(".", ext=GRANT))
    # inventory + 3 dependency agents + 1 verifier
    assert len(seen) == 5 and all(e is GRANT for e in seen), seen


def test_verdicts_bind_by_index_not_by_position():
    other = {**BREAK, "file": "app/models.py", "line": 7}

    async def fake(prompt, schema, **kw):
        if "REFUTE" in prompt:
            return {
                "verdicts": [
                    {"index": 1, "breaks_here": True, "reason": "views break"},
                    {"index": 0, "breaks_here": False, "reason": "models fine"},
                    {"index": 5, "breaks_here": True, "reason": "out of range"},
                ]
            }
        f, _ = _agents([BREAK, other])
        return await f(prompt, schema, **kw)

    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", limit=1))

    confirmed = [(e["finding"]["file"], e["why"]) for e in out["confirmed"]]
    # Verifier sees findings sorted by file: [0] models.py, [1] views.py.
    assert confirmed == [("app/views.py", "views break")], out
    assert [e["finding"]["file"] for e in out["refuted"]] == ["app/models.py"], out


def test_budget_skipped_dependencies_fail_the_exit_code(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("")
    many = [{"name": f"p{i}", "current": "1", "direct": True, "uses": i} for i in range(40)]

    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {"manifest": "pyproject.toml", "dependencies": many}
        return {"findings": [], "sources_checked": ["https://x"]}

    with patched(up, "ask", fake):
        code = up.main([str(tmp_path), "--no-verify"])
    text = capsys.readouterr().out
    assert code == 1
    assert "skipped by the call budget" in text and "safe to bump" not in text, text


def test_a_break_without_a_url_is_unconfirmed_not_dropped(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    fake, _ = _agents([{**BREAK, "url": ""}])
    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(str(tmp_path), limit=1))
        code = up.main([str(tmp_path), "--limit", "1"])
    assert [e["why"] for e in out["unconfirmed"]] == ["no changelog URL"], out
    assert code == 1


def test_the_budget_cut_audits_only_what_fits():
    many = [{"name": f"p{i:02d}", "current": "1", "direct": True, "uses": i} for i in range(40)]
    audited = []

    async def fake(prompt, schema, **kw):
        if "manifest" in prompt:
            return {"manifest": "package.json", "dependencies": many}
        audited.append(prompt)
        return {"findings": [], "sources_checked": ["https://example.com/changelog"]}

    with patched(up, "ask", fake):
        out = asyncio.run(up.upgrade(".", verify=False))
    room = orch.MAX_CALLS // 2
    assert len(audited) == room
    assert out["skipped"] == 40 - room


def test_pick_breaks_ties_by_name():
    deps = [
        {"name": n, "current": "1", "direct": True, "uses": 5} for n in ("zeta", "alpha", "mid")
    ]
    assert [d["name"] for d in up._pick(deps, None)] == ["alpha", "mid", "zeta"]
