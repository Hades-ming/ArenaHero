"""Battle telemetry: parse ``game.log`` into KPIs and detect bottlenecks.

This is the real-time monitoring layer (user requirement #5). ``play.py``
writes one compact line per tick to ``game.log``; this module turns those lines
into actionable signals so the tactic can be iterated on (user requirement #3)
and the user can watch the war.

Run:

    uv run python meta/monitor.py            # analyzes ./game.log
    uv run python meta/monitor.py path.log   # analyzes an arbitrary log
    uv run python meta/monitor.py --json      # machine-readable summary

It prints a human-readable report and exits non-zero if any bottleneck is
active, so it can be wired into an automation that triggers expert review.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "game.log"

# --- Regexes for the per-tick line -----------------------------------------
RE_TICK = re.compile(r"^t(\d+)")
RE_RES = re.compile(r"r(\d+)/(\d+)")
RE_POP = re.compile(r"pop(\d+)\(W(\d+) V(\d+) R(\d+)\)")
RE_CORE = re.compile(r"core@([^ ]+) hp(\d+)/sh(\d+)/(\w+)")
RE_VIS = re.compile(r"vis(\d+)\[([^\]]*)\]")
RE_EV = re.compile(r"ev\[([^\]]*)\]")

# Bottleneck thresholds (tunable).
IDLE_GOLD_TICKS = 15        # resources == capacity for this many CONSECUTIVE ticks => under-investing
STUCK_MOVE_THRESHOLD = 0.10  # fraction of MOVE events that fail CELL_UNIT_LIMIT => clumping
CORE_HP_WARN = 4             # core hp at/below this (or shield < cap under threat) => defense failing
LOW_HARVEST_PER_TICK = 0.05  # harvests per tick below this => exploration stalled
RESOURCE_DROP_THRESHOLD = 10  # single-tick resource drop > this with no spawn => unexplained loss


@dataclass
class KPI:
    ticks: int = 0
    start_tick: int = 0
    end_tick: int = 0
    resources_min: int = 10**9
    resources_max: int = 0
    resources_last: int = 0
    capacity_last: int = 0
    # event tallies
    harvest: int = 0
    deposit: int = 0
    spawn: int = 0
    move_failed_cell: int = 0
    move_succeeded: int = 0
    unit_died: int = 0
    core_under_attack: int = 0
    core_died: int = 0
    enemy_core_destroyed: int = 0
    core_move_failed: int = 0
    resource_not_found: int = 0
    idle_gold_streak: int = 0       # longest run of consecutive capacity-sitting ticks
    resource_drops: int = 0         # ticks where resources fell > RESOURCE_DROP_THRESHOLD w/o spawn
    largest_drop: int = 0
    ticks_with_enemy_visible: int = 0
    # snapshots
    pop_last: int = 0
    workers_last: int = 0
    vanguards_last: int = 0
    rangers_last: int = 0
    core_hp_min: int = 10**9
    core_hp_last: int = 0
    core_shield_last: int = 0
    core_status_last: str = ""
    event_hist: Counter = field(default_factory=Counter)


def _parse_line(line: str) -> dict | None:
    if not line.startswith("t"):
        return None
    m = RE_TICK.search(line)
    if not m:
        return None
    rec: dict = {"tick": int(m.group(1))}
    m = RE_RES.search(line)
    if m:
        rec["res"], rec["cap"] = int(m.group(1)), int(m.group(2))
    m = RE_POP.search(line)
    if m:
        rec["pop"], rec["w"], rec["v"], rec["r"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        )
    m = RE_CORE.search(line)
    if m:
        rec["core_hp"], rec["core_shield"], rec["core_status"] = (
            int(m.group(2)), int(m.group(3)), m.group(4)
        )
    m = RE_VIS.search(line)
    if m:
        rec["vis_n"] = int(m.group(1))
        rec["vis_body"] = m.group(2)
    m = RE_EV.search(line)
    if m:
        rec["events"] = [e for e in m.group(1).split(";") if e]
    return rec


def analyze(path: str | Path) -> KPI:
    """Parse a game.log file into a KPI summary."""
    kpi = KPI()
    p = Path(path)
    if not p.exists():
        return kpi
    prev_res = None
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            rec = _parse_line(raw)
            if not rec:
                continue
            kpi.ticks += 1
            kpi.end_tick = rec["tick"]
            if kpi.start_tick == 0:
                kpi.start_tick = rec["tick"]
            if "res" in rec:
                res, cap = rec["res"], rec["cap"]
                kpi.resources_min = min(kpi.resources_min, res)
                kpi.resources_max = max(kpi.resources_max, res)
                kpi.resources_last = res
                kpi.capacity_last = cap
                # Consecutive capacity-sitting streak (under-investing signal).
                if res >= cap:
                    kpi.idle_gold_streak += 1
                else:
                    kpi.idle_gold_streak = 0
                # Unexplained resource loss: a big drop with no spawn that tick.
                spawned = any(
                    e.split("[")[0] == "SPAWN_SUCCEEDED"
                    for e in rec.get("events", [])
                )
                if prev_res is not None and not spawned and prev_res - res > RESOURCE_DROP_THRESHOLD:
                    kpi.resource_drops += 1
                    kpi.largest_drop = max(kpi.largest_drop, prev_res - res)
                prev_res = res
            if "pop" in rec:
                kpi.pop_last, kpi.workers_last = rec["pop"], rec["w"]
                kpi.vanguards_last, kpi.rangers_last = rec["v"], rec["r"]
            if "core_hp" in rec:
                kpi.core_hp_min = min(kpi.core_hp_min, rec["core_hp"])
                kpi.core_hp_last = rec["core_hp"]
                kpi.core_shield_last = rec["core_shield"]
                kpi.core_status_last = rec["core_status"]
            if "vis_n" in rec and rec["vis_n"] > 0:
                kpi.ticks_with_enemy_visible += 1
            for ev in rec.get("events", []):
                # strip trailing [n] payload
                base = ev.split("[")[0]
                kpi.event_hist[base] += 1
                if base == "HARVEST_SUCCEEDED":
                    kpi.harvest += 1
                elif base == "DEPOSIT_SUCCEEDED":
                    kpi.deposit += 1
                elif base == "SPAWN_SUCCEEDED":
                    kpi.spawn += 1
                elif base == "UNIT_MOVE_FAILED.CELL_UNIT_LIMIT":
                    kpi.move_failed_cell += 1
                elif base == "UNIT_MOVE_SUCCEEDED":
                    kpi.move_succeeded += 1
                elif base == "UNIT_DIED":
                    kpi.unit_died += 1
                elif base == "CORE_UNDER_ATTACK":
                    kpi.core_under_attack += 1
                elif base == "CORE_DIED":
                    kpi.core_died += 1
                elif base == "ENEMY_CORE_DESTROYED":
                    kpi.enemy_core_destroyed += 1
                elif base == "CORE_MOVE_FAILED":
                    kpi.core_move_failed += 1
                elif base == "RESOURCE_NOT_FOUND":
                    kpi.resource_not_found += 1
    return kpi


def detect_bottlenecks(kpi: KPI) -> list[str]:
    """Return a list of human-readable bottleneck alerts."""
    alerts: list[str] = []
    if kpi.ticks == 0:
        return ["no game data parsed (empty or missing log)"]

    harvest_per_tick = kpi.harvest / kpi.ticks
    if harvest_per_tick < LOW_HARVEST_PER_TICK:
        alerts.append(
            f"LOW_HARVEST: only {harvest_per_tick:.3f} harvests/tick "
            f"({kpi.harvest} total) — exploration/economy may be stalled"
        )
    if kpi.idle_gold_streak >= IDLE_GOLD_TICKS:
        alerts.append(
            f"IDLE_GOLD: resources sat at capacity for {kpi.idle_gold_streak} "
            f"consecutive ticks (last {kpi.resources_last}/{kpi.capacity_last}) "
            f"— capital not working"
        )
    moves = kpi.move_failed_cell + kpi.move_succeeded
    if moves > 0 and kpi.move_failed_cell / moves > STUCK_MOVE_THRESHOLD:
        alerts.append(
            f"UNIT_CLUMPING: {kpi.move_failed_cell}/{moves} move events blocked "
            f"({kpi.move_failed_cell / moves:.1%}) — units blocked (often a few "
            f"deadlocked Workers, not a full-team jam)"
        )
    if kpi.core_hp_min <= CORE_HP_WARN or kpi.core_died > 0:
        alerts.append(
            f"CORE_DEFENSE: core hp dipped to {kpi.core_hp_min} "
            f"(died {kpi.core_died}x) — defense failed, stored resources lost"
        )
    if kpi.enemy_core_destroyed == 0 and kpi.ticks_with_enemy_visible > 0:
        alerts.append(
            f"NO_RAID: enemies were visible for {kpi.ticks_with_enemy_visible} ticks "
            f"but 0 enemy Cores destroyed — missed +6 resource opportunity"
        )
    if kpi.resource_drops > 0:
        alerts.append(
            f"RESOURCE_LOSS: {kpi.resource_drops} ticks with a drop >"
            f"{RESOURCE_DROP_THRESHOLD} and no spawn (largest {kpi.largest_drop}) "
            f"— unexplained resource evaporation (overflow / manual spend)"
        )
    if kpi.unit_died > 0 and kpi.deposit == 0:
        alerts.append(
            "UNIT_LOSS_NO_DEPOSIT: units died without deposits — cargo wasted"
        )
    return alerts


def report(kpi: KPI, alerts: list[str]) -> str:
    if kpi.ticks == 0:
        return "No game data to report."
    lines = []
    lines.append("=" * 60)
    lines.append("ARENA HERO — BATTLE TELEMETRY")
    lines.append("=" * 60)
    lines.append(f"Ticks analyzed : {kpi.ticks} (t{kpi.start_tick}..t{kpi.end_tick})")
    lines.append(
        f"Resources      : last {kpi.resources_last}/{kpi.capacity_last} "
        f"(min {kpi.resources_min}, max {kpi.resources_max})"
    )
    lines.append(
        f"Army/Pop      : pop {kpi.pop_last} "
        f"(W{kpi.workers_last} V{kpi.vanguards_last} R{kpi.rangers_last})"
    )
    lines.append(
        f"Core           : hp {kpi.core_hp_last} (min {kpi.core_hp_min}) "
        f"sh {kpi.core_shield_last} [{kpi.core_status_last}]"
    )
    lines.append(
        f"Economy        : {kpi.harvest} harvests, {kpi.deposit} deposits, "
        f"{kpi.spawn} spawns"
    )
    lines.append(
        f"Combat         : {kpi.unit_died} unit deaths, "
        f"{kpi.core_under_attack} core-under-attack, "
        f"{kpi.enemy_core_destroyed} enemy cores destroyed"
    )
    lines.append(
        f"Friction       : {kpi.move_failed_cell} blocked moves, "
        f"{kpi.resource_not_found} resource-not-found, "
        f"{kpi.core_move_failed} core-move-failed"
    )
    lines.append(
        f"Enemy visible  : {kpi.ticks_with_enemy_visible} ticks"
    )
    lines.append("-" * 60)
    if alerts:
        lines.append("BOTTLENECKS / ACTION NEEDED:")
        for a in alerts:
            lines.append(f"  ! {a}")
    else:
        lines.append("No bottlenecks detected — tactic is healthy.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    path = LOG_PATH
    for a in argv:
        if a != "--json" and not a.startswith("-"):
            path = Path(a)
            break
    kpi = analyze(path)
    alerts = detect_bottlenecks(kpi)
    if as_json:
        out = asdict(kpi)
        out["bottlenecks"] = alerts
        print(json.dumps(out, indent=2, default=str))
    else:
        print(report(kpi, alerts))
    # Non-zero exit when a bottleneck is active so automations can branch on it.
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
