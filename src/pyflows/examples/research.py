"""Research workflow: sweep several angles blind, then synthesize, then verify.

Shape is sweep -> barrier -> synthesize -> verify.

The sweep is multi-modal: each agent searches a different way and cannot see
what the others found, so they surface things a single searcher would miss.
The barrier before synthesis is real — the synthesizer needs every result at
once to spot agreement and contradiction. Verification is adversarial on
*citations*, not on conclusions: the weakest link in research is a claim
pointing at a source that does not say that.
"""

from __future__ import annotations

import argparse
import asyncio
import time

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

# Web tools only. No local filesystem: this workflow has no source tree to
# read, and asking one to touch the disk is how a research run wanders off.
WEB = ("WebSearch", "WebFetch")

# Distinct search angles. Deliberately not "top 5 queries" — an agent given a
# different *modality* finds what a keyword variant would not.
# The short name is what shows in `pyflows watch`; keep both under ~30 chars,
# which is panel.STAGE_COL minus the "verify/" prefix.
ANGLES = (
    "primary sources: official docs, specs, papers, standards",
    "practitioner experience: forum threads, postmortems, issue trackers",
    "contrarian view: what critics, alternatives, or abandoned approaches say",
    "recent developments: changelogs, releases, recent announcements",
)
ANGLE_NAMES = ("primary", "practitioner", "contrarian", "recent")

QUESTIONS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["q", "why"],
            },
        }
    },
    "required": ["questions"],
}

FINDINGS = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "url": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["claim", "url", "confidence"],
            },
        }
    },
    "required": ["findings"],
}

BRIEF = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "agreements": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        # claim + url, never a bare url. A verifier handed only a URL can
        # check nothing: it has no assertion to compare the page against, so
        # it can only ever report "this URL exists" — or default to
        # unsupported, having verified nothing.
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["claim", "url"],
            },
        },
    },
    "required": ["answer", "agreements", "contradictions", "gaps", "sources"],
}

VERDICT = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "reason"],
}


class ResearchFailed(RuntimeError):
    """No angle came back. A dead run must never read as 'nothing found'."""


def _citations(brief: dict) -> list[tuple[str, str]]:
    """(claim, url) pairs worth checking — deduped by URL, best first.

    Sources arrive as {claim, url}. A source whose claim restates its own URL
    carries no assertion, so there is nothing to verify — drop it rather than
    spend an agent confirming a page exists.
    """
    out, seen = [], set()
    for source in brief.get("sources", []):
        if isinstance(source, str):  # pre-fix shape, or a lazy synthesizer
            continue
        claim, url = (source.get("claim", "") or "").strip(), (source.get("url", "") or "").strip()
        if not claim or not url or claim == url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((claim, url))
    return out


async def research(
    question: str,
    angles: tuple[str, ...] = ANGLES,
    depth: int = 3,
    verify: bool = True,
    ext: Extensions = NO_EXTENSIONS,
) -> dict:
    """Research a question. Returns the brief plus a verification summary."""
    spend = _Spend(MAX_CALLS)
    started = time.monotonic()

    print(f"sweeping {len(angles)} angles on: {question}")

    # Multi-modal sweep. Each agent is blind to the others — that blindness is
    # the point, so no barrier between them.
    found = await gather(
        [
            ask(
                f"Research this question: {question}\n\n"
                f"Search specifically for {angle}. Report only what a source "
                f"actually states, with its URL. Do not speculate; if you find "
                f"nothing, return an empty findings array.",
                FINDINGS,
                tools=WEB,
                spend=spend,
                ext=ext,
                # Index into the short names so `watch` stays readable.
                label=f"sweep/{ANGLE_NAMES[i % len(ANGLE_NAMES)]}",
                system=f"You research via {angle}. Every claim needs a source URL.",
            )
            for i, angle in enumerate(angles)
        ]
    )

    alive = [f for f in found if f is not None]
    if not alive:
        raise ResearchFailed(f"all {len(angles)} search angles failed")

    claims, dropped = [], 0
    for group in alive:
        for f in group.get("findings", []):
            # The synthesizer would cite "(source: , …)" — a claim it cannot
            # source is one the citation check below can never catch.
            if not f.get("url"):
                dropped += 1
                continue
            claims.append((f["claim"], f["url"], f.get("confidence", "low")))

    if dropped:
        print(f"  {dropped} claims had no URL and cannot be verified — dropped")

    print(f"{len(claims)} claims from {len(alive)}/{len(angles)} angles, synthesizing…")

    brief = await ask(
        f"Synthesize an answer to: {question}\n\n"
        f"Claims gathered by independent searchers:\n"
        + "\n".join(f"- {c} (source: {u}, confidence: {conf})" for c, u, conf in claims)
        + "\n\nName where they agree, where they contradict each other, and what "
        "nobody has covered. Do not resolve contradictions by picking a side — "
        "report them.\n\n"
        "For sources, list each assertion from your answer together with the URL "
        "that backs it: {claim, url}. The claim must be the specific assertion "
        "you are making, NOT the URL restated — a bare URL cannot be checked.",
        BRIEF,
        tools=WEB,
        spend=spend,
        ext=ext,
        label="synthesize",
        system="You synthesize. Report disagreement rather than resolving it.",
    )
    if brief is None:
        raise ResearchFailed("synthesis failed")

    result = {
        **brief,
        "verified": None,
        "unsupported": [],
        # Citations whose verifier died: neither held up nor refuted.
        "unchecked": [],
        "angles_alive": len(alive),
    }

    if not verify:
        result["elapsed"] = time.monotonic() - started
        return result

    # Verify the citations, not the conclusions — a brief can be well-reasoned
    # and still rest on a source that does not say what it is claimed to say.
    targets = _citations(brief)[:depth]
    print(f"checking {len(targets)} source citations…")

    verdicts = await gather(
        [
            ask(
                f"Does this source actually support the claim it is cited for?\n\n"
                f"Source: {url}\nClaim: {claim}\n\n"
                f"Fetch the source. Default to supported=false if it does not "
                f"clearly back the claim.",
                VERDICT,
                tools=WEB,
                spend=spend,
                ext=ext,
                # The source's last 25 chars, to fit panel.STAGE_COL.
                label=f"verify/{url.split('//')[-1][-25:]}",
                system="You check citations. Default to unsupported.",
            )
            for claim, url in targets
        ]
    )

    # The bug: a dead verifier (None) was kept as "unsupported", then v.get()
    # crashed the run. A dead verifier checked nothing — it is unchecked.
    pairs = list(zip(targets, verdicts, strict=True))
    unchecked = [{"claim": c, "url": u} for (c, u), v in pairs if v is None]
    unsupported = [
        {"claim": c, "url": u, "why": v.get("reason", "")}
        for (c, u), v in pairs
        if v is not None and not v.get("supported")
    ]
    if targets and len(unchecked) == len(targets):
        raise ResearchFailed(f"all {len(targets)} citation checks failed")
    result["verified"] = len(targets) - len(unsupported) - len(unchecked)
    result["unsupported"] = unsupported
    result["unchecked"] = unchecked
    result["elapsed"] = time.monotonic() - started
    return result


def render(brief: dict) -> None:
    print(f"\n{brief['answer']}")

    for heading, key in (
        ("Agreements", "agreements"),
        ("Contradictions", "contradictions"),
        ("Gaps", "gaps"),
    ):
        items = brief.get(key) or []
        if items:
            print(f"\n{heading}:")
            for item in items:
                print(f"  • {item}")

    if brief.get("unsupported"):
        print(f"\nUnsupported citations ({len(brief['unsupported'])}):")
        for bad in brief["unsupported"]:
            print(f"  ✗ {bad['url']}\n    {bad['claim']}\n    {bad['why']}")
    unchecked = brief.get("unchecked") or []
    if unchecked:
        print(f"\nCitations NOT checked ({len(unchecked)}) — their verifier did not answer:")
        for c in unchecked:
            print(f"  ? {c['url']}\n    {c['claim']}")
    if brief.get("verified") is not None and not brief["unsupported"] and not unchecked:
        print(f"\nAll {brief['verified']} checked citations held up.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyflows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rs = sub.add_parser("research", help="research a question")
    rs.add_argument("question")
    rs.add_argument("--angles", type=int, default=None, help="limit search angles")
    rs.add_argument("--depth", type=int, default=3, help="citations to verify")
    rs.add_argument("--no-verify", action="store_true", help="skip citation checking")
    add_extension_args(rs)

    args = parser.parse_args(argv)
    angles = ANGLES[: args.angles] if args.angles else ANGLES

    STATS.reset()
    _log_run(
        {"outcome": "run_start", "target": args.question, "mode": "research", "lenses": len(angles)}
    )
    print(f"research: {len(angles)} angles")
    try:
        brief = asyncio.run(
            research(args.question, angles, args.depth, not args.no_verify, extensions_from(args))
        )
    except (ResearchFailed, BudgetExhausted, ExtensionError) as exc:
        print(f"research did not complete: {exc}\n{STATS.summary()}")
        return 1

    render(brief)
    print(f"\n{STATS.summary()}")
    print(f"wall clock: {brief['elapsed']:.0f}s")
    # A citation nobody checked is not one that held up.
    return 1 if brief.get("unchecked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
