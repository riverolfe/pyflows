"""Example workflow: audit dependency upgrades against the code that uses them.

Shape is inventory -> sweep -> barrier -> verify. This is the third example, and
it differs from the other two in one structural way: **the fan-out width is not
known until the first stage returns.** review fans out over a fixed lens tuple;
research over a fixed angle tuple. Here the inventory decides how many upgrade
agents run, so the call count is discovered at runtime — which is exactly the
case `_Spend` and MAX_CALLS exist for.

A second difference: every agent here is a *diff* against this repo. An agent
that only knew "Django 5.0 removed `django.utils.timezone.utc`" would report a
breaking change on a repo that never used it. Each agent is handed the repo and
asked what breaks *here*, so a generic changelog entry cannot pass for a finding.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

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

# Manifests we know how to read, in the order we prefer them. First hit wins —
# a repo can carry several (a monorepo root with a nested package).
MANIFESTS = (
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "Cargo.toml",
    "Gemfile",
)

# Each upgrade agent gets the repo AND the web. It has to find the local usage
# and check the upstream changelog, or it is guessing at one of the two.
TOOLS = ("Read", "Grep", "Glob", "WebSearch", "WebFetch")

INVENTORY = {
    "type": "object",
    "properties": {
        "manifest": {"type": "string"},
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current": {"type": "string"},
                    # Direct deps break callers; a transitive bump is worth a
                    # look but not a dedicated agent.
                    "direct": {"type": "boolean"},
                    "uses": {"type": "integer"},
                },
                "required": ["name", "current", "direct", "uses"],
            },
        },
    },
    "required": ["manifest", "dependencies"],
}

BREAKING = {
    "type": "object",
    "properties": {
        # Required, and the load-bearing field of this schema. A web tool that
        # silently fails leaves the agent able to return a clean, schema-valid
        # "no breaks here" without ever having read a changelog. Naming the
        # sources it actually consulted is the only way the caller can tell an
        # audited dependency from an unreachable one — and an agent that
        # consulted nothing must say so rather than imply a clean bill.
        "sources_checked": {
            "type": "array",
            "items": {"type": "string"},
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "dependency": {"type": "string"},
                    "summary": {"type": "string"},
                    # What breaks *here*, not upstream's wording of it.
                    "failure": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["file", "line", "dependency", "summary", "failure", "url"],
            },
        },
    },
    "required": ["findings", "sources_checked"],
}

# One agent judges a BATCH of breaks, same shape review.py's BATCH uses: the
# source read dominates cost, so judging 5 in one call beats 5 re-reads. An
# array keyed by index, not a flat object — a single object cannot name which
# of the 5 claims it is answering for.
VERDICTS = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "breaks_here": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "breaks_here", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


class UpgradeFailed(RuntimeError):
    """No inventory came back. A dead run must never read as 'nothing to upgrade'."""


def find_manifest(target: Path) -> str | None:
    for name in MANIFESTS:
        if (target / name).is_file():
            return name
    return None


def _pick(deps: list[dict], limit: int | None) -> list[dict]:
    """Direct deps first, then most-used — the blast radius of an upgrade.

    Sorted so the run is reproducible: two runs over the same repo pick the
    same dependencies, and the cache means the second one is free.
    """
    ranked = sorted(
        (d for d in deps if d.get("name")),
        key=lambda d: (not d.get("direct"), -(d.get("uses") or 0), d["name"]),
    )
    return ranked[:limit] if limit else ranked


async def inventory(target: str, spend: _Spend, ext: Extensions = NO_EXTENSIONS) -> list[dict]:
    """One agent reads the manifest. This is what sets the fan-out width."""
    found = await ask(
        f"In {target}, find the dependency manifest and list its dependencies.\n\n"
        f"Look for: {', '.join(MANIFESTS)}. Count how many places in the "
        f"source each dependency is referenced — set `uses` to 0 if you cannot "
        f"tell. Mark `direct` true only for entries named in the manifest "
        f"itself, not ones you inferred.",
        INVENTORY,
        tools=TOOLS,
        spend=spend,
        ext=ext,
        cwd=target,
        depends_on=target,
        label="inventory",
        system="You read manifests. Report only what the manifest states.",
    )
    if found is None:
        raise UpgradeFailed("could not read a dependency manifest")
    return found.get("dependencies") or []


async def upgrade(
    target: str,
    limit: int | None = None,
    depth: int = 3,
    verify: bool = True,
    ext: Extensions = NO_EXTENSIONS,
) -> dict:
    """Audit what upgrading this repo's dependencies would break. Returns a brief."""
    spend = _Spend(MAX_CALLS)
    started = time.monotonic()

    deps = _pick(await inventory(target, spend, ext), limit)
    if not deps:
        raise UpgradeFailed("manifest listed no dependencies")

    # The ceiling is a call count and the width is now known, so say out loud
    # when the run cannot cover everything it found. Silently auditing the
    # first 12 of 40 dependencies reads as "these 12 are the only ones".
    # Two calls per dependency — one upgrade agent, one verification batch.
    capped = ""
    if not limit and len(deps) * 2 > spend.remaining:
        room = max(1, spend.remaining // 2)
        capped = f"\n  {len(deps)} dependencies found, {room} fit the budget — rest skipped"
        deps = deps[:room]
    print(f"auditing {len(deps)} dependencies in {target}{capped}")

    results = await gather(
        [
            ask(
                f"Check whether upgrading `{d['name']}` from {d['current']} to its "
                f"latest release would break this repository.\n\n"
                f"Find where {d['name']} is actually used in {target}. Then check "
                f"the upstream changelog or migration guide for breaking changes "
                f"between {d['current']} and now. Report a finding ONLY where a "
                f"specific breaking change hits a specific file in this repo.\n\n"
                f"If this repo is unaffected, return an empty findings array — "
                f"that is the common case and is a valid answer.\n\n"
                f"List every changelog, release note or migration guide you "
                f"actually read in `sources_checked`. If your web tools fail or "
                f"return nothing, list nothing — an empty `sources_checked` is "
                f"how the caller knows you could not check, and it will not be "
                f"counted as evidence that this dependency is safe.",
                BREAKING,
                tools=TOOLS,
                spend=spend,
                ext=ext,
                cwd=target,
                depends_on=target,
                label=f"dep/{d['name']}"[:32],
                system="You audit one dependency. No upstream changelog that "
                "this repo does not actually hit is a finding.",
            )
            for d in deps
        ]
    )

    alive = [r for r in results if r is not None]
    if not alive:
        raise UpgradeFailed(f"all {len(deps)} dependency agents failed")

    # An agent that consulted no source was never able to check anything, and
    # its "no breaks" is indistinguishable from a real clean bill. Separate
    # them: blanked dependencies are NOT audited, and if that is every one of
    # them the run says so instead of reporting the deps as safe.
    blanked = [r for r in alive if not r.get("sources_checked")]
    checked = [r for r in alive if r.get("sources_checked")]
    if blanked:
        print(
            f"  {len(blanked)}/{len(alive)} agents consulted no source — "
            f"their 'no breaks' is unchecked, not clean"
        )

    # Barrier: every result at once, so the same breaking change reported by
    # two dependencies collapses to one line item.
    seen, findings = set(), []
    for group in alive:
        for f in group.get("findings", []):
            key = (f["file"], f["line"], f["dependency"])
            if key not in seen:
                seen.add(key)
                findings.append(f)

    quoted = sum(1 for f in findings if not f.get("url"))
    if quoted:
        print(f"  {quoted} findings had no changelog URL and cannot be verified — dropped")
        findings = [f for f in findings if f.get("url")]

    print(f"{len(findings)} candidate breaks from {len(alive)}/{len(deps)} agents")
    result = {
        "target": target,
        "dependencies": deps,
        "audited": len(checked),
        "unchecked": len(blanked),
        "findings": findings,
        "confirmed": [],
        "refuted": [],
        # Never verified: no verdict came back, or the claim was never sent.
        # Not refuted — "safe to bump" over these is a guess, not a finding.
        "unconfirmed": [],
    }

    # Nothing was actually checked. Say so rather than reporting the deps safe.
    # Skipped only when the caller already asked for no verification at all.
    if not checked and verify:
        raise UpgradeFailed(
            f"no agent could reach a changelog ({len(blanked)} consulted no source) — "
            f"this looks like the web tools are unavailable, not that the deps are safe"
        )

    if not verify or not findings:
        result["unconfirmed"] = [{"finding": f, "why": "verification skipped"} for f in findings]
        result["elapsed"] = time.monotonic() - started
        return result

    ranked = sorted(findings, key=lambda f: (f["dependency"], f["file"]))
    targets = ranked[:depth]
    result["unconfirmed"] = [{"finding": f, "why": "beyond --depth"} for f in ranked[depth:]]
    print(f"checking {len(targets)} of {len(findings)} candidate breaks…")

    # One verifier per batch, like review: reading the repo again per finding
    # is what made the unbatched version cost 16 agents instead of 7.
    verdicts = await gather(
        [
            ask(
                "Try to REFUTE each claim that upgrading a dependency breaks "
                "this repository. Read the actual code at each location.\n\n"
                + "\n\n".join(
                    f"[{i}] {f['file']}:{f['line']} — {f['summary']}\n"
                    f"    Dependency: {f['dependency']}\n"
                    f"    Failure: {f['failure']}\n"
                    f"    Claimed source: {f['url']}"
                    for i, f in enumerate(batch)
                )
                + "\n\nDefault to breaks_here=false. The cited changelog "
                "existing is not evidence that this repo hits it. Return one "
                "verdict per claim, keyed by its [n] index.",
                VERDICTS,
                tools=TOOLS,
                spend=spend,
                ext=ext,
                cwd=target,
                depends_on=target,
                label=f"verify/{batch[0]['dependency']}:{len(batch)}"[:32],
                system="You are an adversarial reviewer. Default to breaks_here=false.",
            )
            for batch in _batches(targets, 5)
        ]
    )

    for batch, verdict in zip(_batches(targets, 5), verdicts, strict=True):
        if verdict is None:
            result["unconfirmed"].extend(
                {"finding": f, "why": "verifier did not answer"} for f in batch
            )
            continue
        # Keyed by index, not position: an agent that returns its verdicts out
        # of order must not attribute a reason to the wrong claim. A claim with
        # no verdict was never refuted — it falls into unconfirmed, not refuted.
        by_index = {v.get("index"): v for v in verdict.get("verdicts", [])}
        for i, f in enumerate(batch):
            v = by_index.get(i)
            if v is None:
                result["unconfirmed"].append({"finding": f, "why": "verifier skipped this claim"})
            else:
                entry = {"finding": f, "why": v.get("reason", "")}
                (result["confirmed"] if v.get("breaks_here") else result["refuted"]).append(entry)

    result["elapsed"] = time.monotonic() - started
    return result


def _batches(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def render(brief: dict) -> None:
    print(
        f"\n{brief['target']}: {len(brief['findings'])} candidate breaks, "
        f"{len(brief['confirmed'])} confirmed"
    )

    by_dep: dict[str, list] = {}
    for entry in brief["confirmed"]:
        by_dep.setdefault(entry["finding"]["dependency"], []).append(entry)

    for dep, entries in sorted(by_dep.items()):
        print(f"\n{dep}:")
        for entry in entries:
            f = entry["finding"]
            print(f"  {f['file']}:{f['line']}  {f['summary']}\n    {entry['why']}")

    if brief["refuted"]:
        print(f"\nRefuted ({len(brief['refuted'])}): upstream changes that do not hit this repo")

    # "No breaks" is only a finding about the dependencies that were checked.
    unchecked = brief.get("unchecked") or 0
    if unchecked:
        print(
            f"\n{unchecked}/{brief['audited'] + unchecked} dependencies were NOT checked — "
            f"their agents reached no changelog. Absence of findings there is unchecked, not safe."
        )
    # A candidate nobody verified is neither a break nor a refutation.
    unconfirmed = brief.get("unconfirmed") or []
    if unconfirmed:
        print(
            f"\n{len(unconfirmed)}/{len(brief['findings'])} candidate breaks were NOT verified "
            f"(dead or skipped verifier, beyond --depth, or --no-verify):"
        )
        for entry in unconfirmed:
            f = entry["finding"]
            print(f"  {f['dependency']}  {f['file']}:{f['line']}  {f['summary']}  [{entry['why']}]")
    if not brief["confirmed"] and not unchecked and not unconfirmed:
        print("no confirmed breaks — the deps look safe to bump")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyflows upgrade", description=__doc__)
    parser.add_argument("target", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=None, help="max dependencies to audit")
    parser.add_argument("--depth", type=int, default=3, help="candidate breaks to verify")
    parser.add_argument("--no-verify", action="store_true", help="skip verification")
    add_extension_args(parser)
    args = parser.parse_args(argv)

    target = Path(args.target)
    if not target.exists():
        print(f"no such path: {target}")
        return 1
    if not find_manifest(target):
        print(f"no manifest found (looked for {', '.join(MANIFESTS)})")
        return 1

    STATS.reset()
    started = time.monotonic()
    _log_run(
        {
            "outcome": "run_start",
            "target": str(target),
            "mode": "upgrade",
            # `lenses` is the fan-out field every mode writes; here it is the
            # dependency count, which is not known until the inventory lands.
            "lenses": None,
        }
    )
    try:
        brief = asyncio.run(
            upgrade(str(target), args.limit, args.depth, not args.no_verify, extensions_from(args))
        )
    except (UpgradeFailed, BudgetExhausted, ExtensionError) as exc:
        print(f"upgrade audit did not complete: {exc}\n{STATS.summary()}")
        return 1

    render(brief)
    print(f"\n{STATS.summary()}")
    print(f"wall clock: {time.monotonic() - started:.0f}s")
    # A partial audit is not a clean one — same rule the other two examples
    # follow, and the reason `main` must not exit 0 over unchecked deps or
    # candidate breaks nobody verified.
    return 1 if brief.get("unchecked") or brief.get("unconfirmed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
