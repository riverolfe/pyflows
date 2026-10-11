"""Terminal view of the run log. Framework-level: renders any workflow.

Knows nothing about review or research — it reads RUN_LOG, which orch.py
writes, and nothing else. A third workflow is observable for free.
"""

from __future__ import annotations

import json
import sys
import time

from .orch import RUN_LOG, read_runs

# The 32-char stage column below is the contract every workflow labels against.
STAGE_COL = 32

OUTCOMES = {
    "started": ("running", ""),
    "ok": ("done", ""),
    "cache_hit": ("cached", "served from cache, cost nothing"),
    "failed": ("FAIL", "agent call raised"),
    "failed_retry": ("FAIL", "retry after schema mismatch raised"),
    "invalid": ("FAIL", "never produced schema-valid output"),
    "no_output": ("FAIL", "returned no structured output"),
    "over_budget": ("BUDGET", "hit its per-call USD ceiling"),
}


def _stage_rows(run: list[dict]) -> list[tuple[str, str, str, str, str]]:
    """Collapse start/finish pairs into one row per agent, in start order.

    A 'started' record with no terminal record after it is still running.

    Labels are not unique — two agents can share one — so start/finish are
    paired by occurrence: each finish claims the oldest unclaimed start with a
    matching label. Keying on the label alone let two agents sharing one label
    both write to the first row while the second hung at 'running' forever.
    """
    rows: list[list] = []
    pending: dict[str, list[int]] = {}
    for record in run:
        if record.get("outcome") == "run_start":
            continue
        label = record.get("label") or "-"
        outcome = record.get("outcome", "")
        queue = pending.setdefault(label, [])
        if outcome == "started":
            queue.append(len(rows))
            rows.append([label, "running", "", "", ""])
        elif queue:
            row = rows[queue.pop(0)]  # oldest unclaimed start for this label
            row[1] = OUTCOMES.get(outcome, ("?", ""))[0]
            if outcome == "ok":
                row[2] = f"{(record.get('input') or 0) / 1000:.1f}k"
                row[3] = f"{(record.get('output') or 0) / 1000:.1f}k"
                row[4] = f"{record.get('seconds', 0):.1f}"
        else:  # a terminal record with no start (older logs, or a wiped log)
            row = [label, OUTCOMES.get(outcome, ("?", ""))[0], "-", "-", "-"]
            if outcome == "ok":
                row[2] = f"{(record.get('input') or 0) / 1000:.1f}k"
                row[3] = f"{(record.get('output') or 0) / 1000:.1f}k"
                row[4] = f"{record.get('seconds', 0):.1f}"
            rows.append(row)
    return [tuple(r) for r in rows]


def _render(run: list[dict]) -> str:
    start = run[0] if run else {}
    mode = start.get("mode", "full")
    # Research fans out over search angles; review over code lenses. Same
    # field, different word — calling angles "lenses" reads as a bug.
    count = start.get("lenses", "?")
    noun = {"research": "angle", "quick": "lens"}.get(mode, "lens")
    if count != 1:
        noun += "s"
    target = start.get("target", "?")
    # Questions are long and arbitrary; paths are short. Truncate the question.
    if mode == "research" and len(target) > 58:
        target = target[:57] + "…"
    rows = _stage_rows(run)
    lines = [
        f"{target}  [{mode}, {count} {noun}]",
        f"{'stage':<{STAGE_COL}} {'status':<8} {'in':>8} {'out':>7} {'sec':>6}",
    ]
    for label, status, tok_in, tok_out, secs in rows:
        # Truncate from the left: the tail of a path is the useful half.
        stage = label if len(label) <= STAGE_COL else "…" + label[-(STAGE_COL - 1) :]
        lines.append(f"{stage:<{STAGE_COL}} {status:<8} {tok_in:>8} {tok_out:>7} {secs:>6}")

    failed = sum(1 for r in rows if r[1] == "FAIL")
    cached = sum(1 for r in run if r.get("outcome") == "cache_hit")
    running = sum(1 for r in rows if r[1] == "running")
    spent = sum(r.get("input", 0) or 0 for r in run)
    lines.append("-" * 66)
    lines.append(
        f"{len(rows)} stages · {failed} failed · {cached} cached · "
        f"{spent / 1000:.1f}k input tokens" + (f" · {running} running" if running else "")
    )
    return "\n".join(lines)


def show_agents(last: int, as_json: bool = False) -> int:
    """Browse agents from past runs — the poor man's /workflows panel."""
    runs = read_runs()
    if not runs:
        print(f"no runs logged yet ({RUN_LOG})")
        return 0

    for run in runs[-last:]:
        if as_json:
            print(json.dumps(run, indent=2))
        else:
            print(f"\n{_render(run)}")
    return 0


def watch(interval: float = 0.5) -> int:
    """Live panel: redraw the latest run as its agents report in.

    Always the newest run — watching an older one has no meaning.
    """
    try:
        while True:
            runs = read_runs()
            if runs:
                # Clear the screen and any leftover lines from the last draw.
                sys.stdout.write("\033[2J\033[H" if sys.stdout.isatty() else "")
                sys.stdout.write(f"\n{_render(runs[-1])}\n")
                sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
