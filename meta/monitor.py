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
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

from arena_hero import UnitType, unit_cost

LOG_PATH = Path(__file__).resolve().parent.parent / "game.log"

# --- Regexes for the per-tick line -----------------------------------------
RE_TICK = re.compile(r"^t(\d+)")
RE_RES = re.compile(r"r(\d+)/(\d+)")
RE_POP = re.compile(r"pop(\d+)\(W(\d+) V(\d+) R(\d+)\)")
RE_CORE = re.compile(r"core@([^ ]+) hp(\d+)/sh(\d+)/(\w+)")
RE_VIS = re.compile(r"vis(\d+)\[([^\]]*)\]")
RE_VIS_CORE = re.compile(r"(?:^|,)-?\d+,-?\d+C(?:,|$)")
RE_ECO = re.compile(r"eco\[([^\]]*)\]")
# Event payloads may contain their own amount brackets, e.g.
# ``ev[HARVEST_SUCCEEDED[1];CORE_SPAWN_SUCCEEDED]``. Match the outer event
# field by its following ``plan[`` delimiter instead of stopping at the first
# inner closing bracket.
RE_EV = re.compile(r"ev\[(.*?)\](?:\s+dt\[(.*?)\])?\s+plan\[")
RE_RECEIPT = re.compile(r"^t(\d+)\s+rcv\[([A-Z_]{1,16})\]\s+actions\[(\d+)\]")
RE_TM = re.compile(r"TM\[(\d+),(\d+),(\d+)\]")
RE_STATUS = re.compile(r"ST\[([A-Z_]+)\]")
RE_ERROR = re.compile(r"ER\[([A-Z0-9_.-]{1,64})\]")
RE_LEGACY_SKIP = re.compile(r"^t(\d+)\s+submit_skipped\s+\(([^)\r\n]+)\)")

# Bottleneck thresholds (tunable).
IDLE_GOLD_TICKS = 15        # resources == capacity for this many CONSECUTIVE ticks => under-investing
STUCK_MOVE_THRESHOLD = 0.10  # fraction of MOVE events that fail => clumping
CORE_HP_WARN = 4             # core hp at/below this (or shield < cap under threat) => defense failing
LOW_HARVEST_PER_TICK = 0.05  # harvests per tick below this => exploration stalled
RESOURCE_DROP_THRESHOLD = 10  # single-tick resource drop > this with no spawn => unexplained loss
# OBS-001 timing thresholds (ms). The contract uses strict upper bounds.
PLAN_P50_MAX = 250
PLAN_P95_MAX = 1000
PLAN_P99_MAX = 2000
RAID_REBUILD_RESERVE = 3


def _percentile(values: list[int], pct: float) -> float:
    """Return the interpolated ``pct``-th percentile without truncation."""
    if not values:
        return 0.0
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    dk = k - f
    return sorted_vals[f] + dk * (sorted_vals[c] - sorted_vals[f])


@dataclass
class KPI:
    # ``ticks`` counts successful, unique accepted ticks. ``records`` includes
    # accepted duplicates and failed submissions for data-quality accounting.
    ticks: int = 0
    records: int = 0
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
    move_failed: int = 0
    move_failed_cell: int = 0
    move_failed_contested: int = 0
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
    ticks_with_enemy_core_visible: int = 0
    resource_assignments: int = 0
    visible_resource_assignments: int = 0
    history_resource_assignments: int = 0
    blocked_resource_candidates: int = 0
    cooled_resource_candidates: int = 0
    far_resource_candidates: int = 0
    stale_resource_candidates: int = 0
    explore_reservations: int = 0
    refresh_reservations: int = 0
    unreachable_resource_targets: int = 0
    harvested_resources: int = 0
    deposited_resources: int = 0
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
    error_hist: Counter = field(default_factory=Counter)
    # Timing (OBS-001): decide_ms, submit_ms, total_local_ms per tick
    decide_ms_list: list[int] = field(default_factory=list)
    submit_ms_list: list[int] = field(default_factory=list)
    total_ms_list: list[int] = field(default_factory=list)
    # Unique-tick tracking (OBS-001)
    unique_ticks: int = 0
    duplicate_ticks: int = 0
    failed_ticks: int = 0
    window_errors: int = 0
    submit_errors: int = 0
    decision_errors: int = 0
    # 缺口按所有已观测 Tick（包括失败记录）计算，避免把失败提交误判成日志缺失。
    tick_gaps: int = 0
    missing_ticks: int = 0
    # 失败提交的耗时与成功计划门禁分开，避免失败较多时证据静默消失。
    failed_decide_ms_list: list[int] = field(default_factory=list)
    failed_submit_ms_list: list[int] = field(default_factory=list)
    failed_total_ms_list: list[int] = field(default_factory=list)
    # 规范 receipt 归属。含 MANUAL receipt 的 Tick 不进入下面的可比业务指标，
    # 但仍在此处单独呈现。
    receipt_records: int = 0
    agent_receipts: int = 0
    manual_receipts: int = 0
    manual_ticks: int = 0
    comparable_ticks: int = 0
    unclassified_ticks: int = 0
    manual_event_hist: Counter = field(default_factory=Counter)
    unclassified_event_hist: Counter = field(default_factory=Counter)
    # 基于 v0.14 事件值的资源流量核算。
    deposit_amount: int = 0
    captured_amount: int = 0
    spawn_cost: int = 0
    repair_cost: int = 0
    unit_heal_cost: int = 0
    core_heal_cost: int = 0
    overflow_loss: int = 0
    net_resources: int = 0
    ticks_with_raid_budget: int = 0
    # Worker 级资源兑现链路：只保留汇总值，不把事件 ID 长期留在 KPI 中。
    harvest_deposit_chains: int = 0
    unmatched_harvests: int = 0
    unmatched_deposits: int = 0
    harvest_deposit_p50: float = 0.0
    harvest_deposit_p95: float = 0.0


_ERROR_CODE_ALIASES = {
    "APIERROR": "API_ERROR",
    "TRANSPORTERROR": "TRANSPORT_ERROR",
    "PROTOCOLERROR": "PROTOCOL_ERROR",
    "AUTHENTICATIONERROR": "AUTHENTICATION_ERROR",
    "POLICYVIOLATIONERROR": "POLICY_VIOLATION",
    "TURN_CLOSEDERROR": "TURN_CLOSED",
}


def _normalize_error_code(value: object) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if RE_ERROR.fullmatch(f"ER[{candidate}]"):
            return _ERROR_CODE_ALIASES.get(candidate, candidate)
    return "UNKNOWN_ERROR"


def _is_spawn_event(event_name: str) -> bool:
    """同时识别当前和旧版的成功生产事件名。"""
    return event_name in {"SPAWN_SUCCEEDED", "CORE_SPAWN_SUCCEEDED"}


def _parse_line(line: str) -> dict | None:
    if not line.startswith("t"):
        return None
    m = RE_TICK.search(line)
    if not m:
        return None
    receipt_match = RE_RECEIPT.match(line.rstrip("\n"))
    if receipt_match:
        return {
            "record_kind": "RECEIPT",
            "tick": int(receipt_match.group(1)),
            "source": receipt_match.group(2),
            "actions": int(receipt_match.group(3)),
        }
    legacy_skip = RE_LEGACY_SKIP.match(line.rstrip("\n"))
    rec: dict = {
        "tick": int(m.group(1)),
        "status": "SUBMIT_FAILED" if legacy_skip else "ACCEPTED",
    }
    if legacy_skip:
        rec["error_code"] = _normalize_error_code(legacy_skip.group(2))
    status_match = RE_STATUS.search(line)
    if status_match:
        rec["status"] = status_match.group(1)
    error_match = RE_ERROR.search(line)
    if error_match:
        rec["error_code"] = _normalize_error_code(error_match.group(1))
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
        rec["vis_core"] = bool(RE_VIS_CORE.search(m.group(2)))
    m = RE_ECO.search(line)
    if m:
        rec["eco"] = {
            name: int(value)
            for token in m.group(1).split(",")
            if (match := re.fullmatch(r"([a-z]+)(\d+)", token))
            for name, value in [match.groups()]
        }
    m = RE_EV.search(line)
    if m:
        rec["events"] = [e for e in m.group(1).split(";") if e]
        details: list[dict[str, str | int]] = []
        for raw_detail in (m.group(2) or "").split(";"):
            fields = raw_detail.split("|")
            if not fields or not fields[0]:
                continue
            detail: dict[str, str | int] = {"event_type": fields[0]}
            for field in fields[1:]:
                key, separator, value = field.partition("=")
                if not separator or not key or not value:
                    continue
                if re.fullmatch(r"-?\d+", value):
                    detail[key] = int(value)
                else:
                    detail[key] = value
            details.append(detail)
        rec["event_details"] = details
    m = RE_TM.search(line)
    if m:
        rec["decide_ms"], rec["submit_ms"], rec["total_ms"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3))
        )
    return rec


def analyze(path: str | Path) -> KPI:
    """Parse a game.log file into a KPI summary."""
    kpi = KPI()
    p = Path(path)
    if not p.exists():
        return kpi
    # receipt 可能晚于 Turn 日志到达，先扫描归属，避免后到的 MANUAL
    # receipt 污染可比 Tick 集合。
    manual_ticks: set[int] = set()
    agent_ticks: set[int] = set()
    for raw in p.open("r", encoding="utf-8", errors="replace"):
        rec = _parse_line(raw)
        if rec and rec.get("record_kind") == "RECEIPT":
            kpi.receipt_records += 1
            source = rec.get("source")
            if source == "MANUAL":
                kpi.manual_receipts += 1
                manual_ticks.add(rec["tick"])
            elif source == "AGENT":
                kpi.agent_receipts += 1
                agent_ticks.add(rec["tick"])
    prev_res = None
    last_success_tick: int | None = None
    last_observed_tick: int | None = None
    pending_harvests: dict[str, deque[int]] = {}
    harvest_deposit_latencies: list[int] = []
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            rec = _parse_line(raw)
            if not rec:
                continue
            if rec.get("record_kind") == "RECEIPT":
                continue
            kpi.records += 1
            tick = rec["tick"]
            if last_observed_tick is not None and tick > last_observed_tick + 1:
                kpi.tick_gaps += 1
                kpi.missing_ticks += tick - last_observed_tick - 1
            if last_observed_tick is None or tick > last_observed_tick:
                last_observed_tick = tick
            if rec.get("status") != "ACCEPTED":
                kpi.failed_ticks += 1
                error_code = rec.get("error_code", "UNKNOWN_ERROR")
                kpi.error_hist[error_code] += 1
                if error_code == "COMMAND_WINDOW_CLOSED":
                    kpi.window_errors += 1
                if rec["status"] == "SUBMIT_FAILED":
                    kpi.submit_errors += 1
                elif rec["status"] == "DECISION_FAILED":
                    kpi.decision_errors += 1
                if "decide_ms" in rec:
                    kpi.failed_decide_ms_list.append(rec["decide_ms"])
                if "submit_ms" in rec:
                    kpi.failed_submit_ms_list.append(rec["submit_ms"])
                if "total_ms" in rec:
                    kpi.failed_total_ms_list.append(rec["total_ms"])
                continue

            # Logs are append-only and ticks are monotonic. This bounded-state
            # check avoids retaining every tick ID while rejecting repeats and
            # out-of-order accepted records from business KPI aggregation.
            if last_success_tick is not None and tick <= last_success_tick:
                kpi.duplicate_ticks += 1
                continue
            last_success_tick = tick
            kpi.ticks += 1
            kpi.unique_ticks += 1
            receipt_tracking = kpi.receipt_records > 0
            is_manual = tick in manual_ticks
            is_unclassified = receipt_tracking and tick not in manual_ticks and tick not in agent_ticks
            excluded_from_agent = is_manual or is_unclassified
            if is_manual:
                kpi.manual_ticks += 1
            elif is_unclassified:
                kpi.unclassified_ticks += 1
            else:
                kpi.comparable_ticks += 1
            kpi.end_tick = tick
            if kpi.unique_ticks == 1:
                kpi.start_tick = tick
            if "decide_ms" in rec:
                kpi.decide_ms_list.append(rec["decide_ms"])
            if "submit_ms" in rec:
                kpi.submit_ms_list.append(rec["submit_ms"])
            if "total_ms" in rec:
                kpi.total_ms_list.append(rec["total_ms"])
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
                spawned = any(_is_spawn_event(e.split("[")[0]) for e in rec.get("events", []))
                if (
                    not excluded_from_agent
                    and tick - 1 not in manual_ticks
                    and prev_res is not None
                    and not spawned
                    and prev_res - res > RESOURCE_DROP_THRESHOLD
                ):
                    kpi.resource_drops += 1
                    kpi.largest_drop = max(kpi.largest_drop, prev_res - res)
                prev_res = None if excluded_from_agent else res
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
                if rec.get("vis_core"):
                    kpi.ticks_with_enemy_core_visible += 1
                    population = rec.get("pop")
                    raid_threshold = (
                        unit_cost(UnitType.RANGER, population) + RAID_REBUILD_RESERVE
                        if isinstance(population, int)
                        else 60
                    )
                    if rec.get("res", 0) >= raid_threshold:
                        kpi.ticks_with_raid_budget += 1
            if not excluded_from_agent:
                eco = rec.get("eco", {})
                kpi.resource_assignments += eco.get("a", 0)
                kpi.visible_resource_assignments += eco.get("av", 0)
                kpi.history_resource_assignments += eco.get("ah", 0)
                kpi.blocked_resource_candidates += eco.get("blk", 0)
                kpi.cooled_resource_candidates += eco.get("cool", 0)
                kpi.far_resource_candidates += eco.get("far", 0)
                kpi.stale_resource_candidates += eco.get("stale", 0)
                kpi.explore_reservations += eco.get("exp", 0)
                kpi.refresh_reservations += eco.get("ref", 0)
                kpi.unreachable_resource_targets += eco.get("unr", 0)
                kpi.harvested_resources += eco.get("harv", 0)
                kpi.deposited_resources += eco.get("dep", 0)
            for index, ev in enumerate(rec.get("events", [])):
                detail = (
                    rec.get("event_details", [])[index]
                    if index < len(rec.get("event_details", []))
                    else {}
                )
                event_tick = detail.get("tick", tick)
                event_is_manual = type(event_tick) is int and event_tick in manual_ticks
                # 去除末尾的 [n] 数量载荷。
                base = _event_base(ev)
                name = _event_name(ev)
                if excluded_from_agent or event_is_manual:
                    if is_unclassified and not event_is_manual:
                        kpi.unclassified_event_hist[base] += 1
                    elif is_manual or event_is_manual:
                        kpi.manual_event_hist[base] += 1
                    continue
                kpi.event_hist[base] += 1
                if name == "HARVEST_SUCCEEDED":
                    kpi.harvest += 1
                    kpi.deposit_amount += 0
                elif name == "DEPOSIT_SUCCEEDED":
                    kpi.deposit += 1
                    kpi.deposit_amount += _event_amount(ev)
                elif _is_spawn_event(name):
                    kpi.spawn += 1
                elif name == "UNIT_MOVE_FAILED":
                    kpi.move_failed += 1
                    reason = base.partition(".")[2]
                    if reason == "CELL_UNIT_LIMIT":
                        kpi.move_failed_cell += 1
                    elif reason == "MOVE_CONTESTED":
                        kpi.move_failed_contested += 1
                elif name == "UNIT_MOVE_SUCCEEDED":
                    kpi.move_succeeded += 1
                elif name == "UNIT_DIED" or (
                    name == "UNIT_DAMAGED"
                    and type(detail.get("hp")) is int
                    and detail.get("hp") == 0
                ):
                    kpi.unit_died += 1
                elif name in {"CORE_UNDER_ATTACK", "CORE_DAMAGED"}:
                    kpi.core_under_attack += 1
                elif name in {"CORE_DIED", "CORE_DESTROYED"}:
                    kpi.core_died += 1
                elif name == "ENEMY_CORE_DESTROYED" or (
                    name == "DESTRUCTION_PARTICIPATION"
                    and _event_reason(ev) == "CORE"
                ):
                    kpi.enemy_core_destroyed += 1
                elif name == "CORE_MOVE_FAILED":
                    kpi.core_move_failed += 1
                elif name in {"RESOURCE_NOT_FOUND", "HARVEST_FAILED"}:
                    kpi.resource_not_found += 1
                if name == "CORE_RESOURCES_CAPTURED":
                    kpi.captured_amount += _event_amount(ev)
                elif name == "CORE_RESOURCE_OVERFLOW_DESTROYED":
                    kpi.overflow_loss += _event_amount(ev)
            for detail in rec.get("event_details", []):
                detail_tick = detail.get("tick", tick)
                detail_is_manual = (
                    type(detail_tick) is int and detail_tick in manual_ticks
                )
                detail_is_unclassified = (
                    receipt_tracking
                    and type(detail_tick) is int
                    and detail_tick not in manual_ticks
                    and detail_tick not in agent_ticks
                )
                if excluded_from_agent or detail_is_manual or detail_is_unclassified:
                    continue
                event_type = detail.get("event_type")
                actor = detail.get("actor")
                if event_type == "HARVEST_SUCCEEDED":
                    if isinstance(actor, str) and type(detail_tick) is int:
                        pending_harvests.setdefault(actor, deque()).append(detail_tick)
                elif event_type == "DEPOSIT_SUCCEEDED":
                    if (
                        isinstance(actor, str)
                        and actor in pending_harvests
                        and pending_harvests[actor]
                        and type(detail_tick) is int
                    ):
                        harvest_tick = pending_harvests[actor].popleft()
                        harvest_deposit_latencies.append(
                            max(0, detail_tick - harvest_tick)
                        )
                        kpi.harvest_deposit_chains += 1
                        if not pending_harvests[actor]:
                            del pending_harvests[actor]
                    else:
                        kpi.unmatched_deposits += 1
                cost = detail.get("cost")
                if type(cost) is not int:
                    continue
                if _is_spawn_event(str(event_type)):
                    kpi.spawn_cost += cost
                elif event_type == "CORE_REPAIR_SUCCEEDED":
                    kpi.repair_cost += cost
                elif event_type == "UNIT_HEAL_SUCCEEDED":
                    kpi.unit_heal_cost += cost
                elif event_type == "CORE_HEAL_SUCCEEDED":
                    kpi.core_heal_cost += cost
            kpi.net_resources = (
                kpi.deposit_amount
                + kpi.captured_amount
                - kpi.spawn_cost
                - kpi.repair_cost
                - kpi.unit_heal_cost
                - kpi.core_heal_cost
                - kpi.overflow_loss
            )
    kpi.unmatched_harvests = sum(len(queue) for queue in pending_harvests.values())
    kpi.harvest_deposit_p50 = _percentile(harvest_deposit_latencies, 50)
    kpi.harvest_deposit_p95 = _percentile(harvest_deposit_latencies, 95)
    return kpi


def _event_base(event: str) -> str:
    """返回不含数量或详情载荷的事件名称。"""
    return event.split("[", 1)[0].split("{", 1)[0]


def _event_name(event: str) -> str:
    """返回不含原因码的 v0.14 事件类型。"""
    return _event_base(event).partition(".")[0]


def _event_reason(event: str) -> str:
    base = _event_base(event)
    _name, separator, reason = base.partition(".")
    return reason if separator else ""


def _event_amount(event: str) -> int:
    match = re.search(r"\[(\d+)\]", event)
    return int(match.group(1)) if match else 0


def detect_bottlenecks(kpi: KPI) -> list[str]:
    """Return a list of human-readable bottleneck alerts."""
    alerts: list[str] = []
    if kpi.ticks == 0:
        if kpi.records == 0:
            return ["no game data parsed (empty or missing log)"]
        alerts = ["no successful unique ticks parsed"]
        if kpi.window_errors:
            alerts.append(
                f"COMMAND_WINDOW_CLOSED: {kpi.window_errors} failed submissions"
            )
        if kpi.submit_errors:
            alerts.append(f"SUBMIT_ERRORS: {kpi.submit_errors} submissions failed")
        if kpi.decision_errors:
            alerts.append(f"DECISION_ERRORS: {kpi.decision_errors} decisions failed")
        return alerts

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
    # ``move_failed_cell`` predates the reason-aware counter. Keep it as a
    # fallback for callers constructing KPI objects from the old schema.
    failed_moves = max(kpi.move_failed, kpi.move_failed_cell)
    moves = failed_moves + kpi.move_succeeded
    if moves > 0 and failed_moves / moves > STUCK_MOVE_THRESHOLD:
        alerts.append(
            f"UNIT_CLUMPING: {failed_moves}/{moves} move events blocked "
            f"({failed_moves / moves:.1%}; CELL_UNIT_LIMIT {kpi.move_failed_cell}, "
            f"MOVE_CONTESTED {kpi.move_failed_contested}) — units blocked "
            f"(often a few deadlocked Workers, not a full-team jam)"
        )
    if kpi.core_hp_min <= CORE_HP_WARN or kpi.core_died > 0:
        alerts.append(
            f"CORE_DEFENSE: core hp dipped to {kpi.core_hp_min} "
            f"(died {kpi.core_died}x) — defense failed, stored resources lost"
        )
    if kpi.enemy_core_destroyed == 0 and kpi.ticks_with_raid_budget > 0:
        alerts.append(
            f"NO_RAID: enemy Cores were visible for "
            f"{kpi.ticks_with_enemy_core_visible} ticks and resources met the "
            f"dynamic Ranger replacement budget for {kpi.ticks_with_raid_budget} ticks, "
            f"but 0 enemy Cores were destroyed — verify attack-unit count and "
            f"line of fire before changing the tactic "
            f"(note: enemy-core loot is variable via CORE_RESOURCES_CAPTURED, not a "
            f"flat +6; see LESSONS L10)"
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
    if kpi.window_errors:
        alerts.append(
            f"COMMAND_WINDOW_CLOSED: {kpi.window_errors} failed submissions"
        )
    if kpi.submit_errors:
        alerts.append(f"SUBMIT_ERRORS: {kpi.submit_errors} submissions failed")
    if kpi.decision_errors:
        alerts.append(f"DECISION_ERRORS: {kpi.decision_errors} decisions failed")
    if kpi.missing_ticks:
        alerts.append(
            f"TICK_GAP: {kpi.missing_ticks} Tick records missing across "
            f"{kpi.tick_gaps} gaps — inspect reconnects or multiple clients"
        )
    # OBS-001 gates algorithmic plan time. Submit and total timings remain in
    # the report because they include server/network latency outside decide().
    if kpi.decide_ms_list:
        p50 = _percentile(kpi.decide_ms_list, 50)
        p95 = _percentile(kpi.decide_ms_list, 95)
        p99 = _percentile(kpi.decide_ms_list, 99)
        limits = (
            (p50, PLAN_P50_MAX, "P50"),
            (p95, PLAN_P95_MAX, "P95"),
            (p99, PLAN_P99_MAX, "P99"),
        )
        exceeded = ", ".join(
            f"{name} {value:g}ms >= {limit}ms"
            for value, limit, name in limits
            if value >= limit
        )
        if exceeded:
            alerts.append(
                f"PLAN_TIMING: {exceeded} — local plan may risk the 15s command window"
            )
    if kpi.duplicate_ticks > 0:
        alerts.append(
            f"DUP_TICK: {kpi.duplicate_ticks} duplicate ticks detected "
            f"({kpi.unique_ticks} unique) — verify single process"
        )
    return alerts


def report(kpi: KPI, alerts: list[str]) -> str:
    if kpi.records == 0:
        return "No game data to report."
    failed_moves = max(kpi.move_failed, kpi.move_failed_cell)
    lines = []
    lines.append("=" * 60)
    lines.append("ARENA HERO — BATTLE TELEMETRY")
    lines.append("=" * 60)
    lines.append(
        f"Ticks analyzed : {kpi.ticks} successful unique / {kpi.records} records "
        f"(comparable {kpi.comparable_ticks}, manual {kpi.manual_ticks}, "
        f"unclassified {kpi.unclassified_ticks}; "
        f"t{kpi.start_tick}..t{kpi.end_tick})"
    )
    if kpi.receipt_records:
        lines.append(
            f"Receipts       : {kpi.receipt_records} total "
            f"(Agent {kpi.agent_receipts}, Manual {kpi.manual_receipts})"
        )
    lines.append(
        f"Failures       : {kpi.failed_ticks} (window {kpi.window_errors}, "
        f"submit {kpi.submit_errors}, decision {kpi.decision_errors})"
    )
    lines.append(
        f"Tick gaps      : {kpi.missing_ticks} missing across {kpi.tick_gaps} gaps"
    )
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
        f"Dispatch       : {kpi.resource_assignments} assignments "
        f"(visible {kpi.visible_resource_assignments}, history "
        f"{kpi.history_resource_assignments}), blocked "
        f"{kpi.blocked_resource_candidates}, cooled "
        f"{kpi.cooled_resource_candidates}, stale "
        f"{kpi.stale_resource_candidates}, explore-reserved "
        f"{kpi.explore_reservations}, refresh-patrol "
        f"{kpi.refresh_reservations}, unreachable "
        f"{kpi.unreachable_resource_targets}"
    )
    lines.append(
        f"Resource flow  : harvested {kpi.harvested_resources}, "
        f"deposited {kpi.deposited_resources}; event amounts in/out "
        f"{kpi.deposit_amount}/{kpi.captured_amount}; "
        f"cost spawn/heal/repair/overflow "
        f"{kpi.spawn_cost}/{kpi.unit_heal_cost + kpi.core_heal_cost}/"
        f"{kpi.repair_cost}/{kpi.overflow_loss}; "
        f"net {kpi.net_resources} "
        f"({kpi.net_resources / kpi.comparable_ticks:.3f}/comparable tick)"
        if kpi.comparable_ticks
        else "Resource flow  : no comparable ticks"
    )
    lines.append(
        f"Worker chain   : {kpi.harvest_deposit_chains} harvest->deposit chains, "
        f"latency P50/P95 {kpi.harvest_deposit_p50:g}/{kpi.harvest_deposit_p95:g} ticks, "
        f"unmatched harvest/deposit {kpi.unmatched_harvests}/{kpi.unmatched_deposits}"
    )
    lines.append(
        f"Combat         : {kpi.unit_died} unit deaths, "
        f"{kpi.core_under_attack} core-under-attack, "
        f"{kpi.enemy_core_destroyed} enemy cores destroyed"
    )
    lines.append(
        f"Friction       : {failed_moves} blocked moves "
        f"(CELL_UNIT_LIMIT {kpi.move_failed_cell}, "
        f"MOVE_CONTESTED {kpi.move_failed_contested}), "
        f"{kpi.resource_not_found} resource-not-found, "
        f"{kpi.core_move_failed} core-move-failed"
    )
    lines.append(
        f"Enemy visible  : {kpi.ticks_with_enemy_visible} ticks "
        f"(enemy Core {kpi.ticks_with_enemy_core_visible}, "
        f"raid budget {kpi.ticks_with_raid_budget})"
    )
    if kpi.decide_ms_list:
        n = len(kpi.total_ms_list)
        p50_d = _percentile(kpi.decide_ms_list, 50)
        p95_d = _percentile(kpi.decide_ms_list, 95)
        p99_d = _percentile(kpi.decide_ms_list, 99)
        p50_s = _percentile(kpi.submit_ms_list, 50)
        p95_s = _percentile(kpi.submit_ms_list, 95)
        p99_s = _percentile(kpi.submit_ms_list, 99)
        p50_t = _percentile(kpi.total_ms_list, 50)
        p95_t = _percentile(kpi.total_ms_list, 95)
        p99_t = _percentile(kpi.total_ms_list, 99)
        lines.append(
            f"Timing ({n} samples) : decide {p50_d:g}/{p95_d:g}/{p99_d:g}ms "
            f"(P50/P95/P99)  submit {p50_s:g}/{p95_s:g}/{p99_s:g}ms  "
            f"total {p50_t:g}/{p95_t:g}/{p99_t:g}ms"
        )
    if kpi.failed_total_ms_list:
        lines.append(
            f"Failed timing  : {len(kpi.failed_total_ms_list)} samples "
            f"(decide/submit/total recorded separately)"
        )
    lines.append(
        f"Tick quality   : {kpi.unique_ticks} unique + {kpi.duplicate_ticks} duplicate"
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


def _watch_loop(path: str | Path, interval: int, as_json: bool) -> None:
    """Continuously re-analyze ``path`` every ``interval`` seconds (real-time monitor).

    Prints a compact live status line every cycle and only re-prints the bottleneck
    block when the set of active bottlenecks CHANGES, so a long-running session stays
    readable instead of spamming the same alerts. This is the standing real-time
    watch (user requirement #5) — run it as a nohup background daemon.
    """
    import time

    prev: tuple[str, ...] = ()
    print(f"[monitor] watching {path} every {interval}s (ctrl-c to stop)", flush=True)
    while True:
        kpi = analyze(path)
        alerts = tuple(detect_bottlenecks(kpi))
        status = (
            f"t{kpi.end_tick} r{kpi.resources_last}/{kpi.capacity_last} "
            f"pop{kpi.pop_last}(W{kpi.workers_last} V{kpi.vanguards_last} "
            f"R{kpi.rangers_last}) hp{kpi.core_hp_last}/"
            f"sh{kpi.core_shield_last} harv{kpi.harvest} dep{kpi.deposit} "
            f"evis{kpi.ticks_with_enemy_visible} "
            f"u{kpi.unique_ticks}d{kpi.duplicate_ticks} "
            f"g{kpi.tick_gaps}/{kpi.missing_ticks} "
            f"{'ALERTS(' + str(len(alerts)) + ')' if alerts else 'ok'}"
        )
        print(status, flush=True)
        if alerts != prev:
            print("--- BOTTLENECKS (changed) ---", flush=True)
            for a in alerts:
                print(f"  ! {a}", flush=True)
            prev = alerts
        time.sleep(interval)


def main(argv: list[str]) -> int:
    as_json = False
    loop = 0
    args: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--json":
            as_json = True
        elif a == "--loop" or a.startswith("--loop"):
            # Accept both "--loop 30" and "--loop=30".
            if "=" in a:
                loop = int(a.split("=", 1)[1])
            elif i + 1 < len(argv):
                loop = int(argv[i + 1])
                i += 1
        elif not a.startswith("-"):
            # main 接收的已经是 sys.argv[1:]，当前位置参数就是日志路径。
            args.append(a)
        i += 1
    path = Path(args[0]) if args else LOG_PATH
    if loop > 0:
        _watch_loop(path, loop, as_json)
        return 0
    kpi = analyze(path)
    alerts = detect_bottlenecks(kpi)
    if as_json:
        out = asdict(kpi)
        # dataclasses.asdict dumps sets as-is (not JSON-serializable);
        # convert to list.
        # Counter -> plain dict; timing lists are already lists.
        out["event_hist"] = dict(kpi.event_hist)
        out["manual_event_hist"] = dict(kpi.manual_event_hist)
        out["unclassified_event_hist"] = dict(kpi.unclassified_event_hist)
        out["error_hist"] = dict(kpi.error_hist)
        out["bottlenecks"] = alerts
        print(json.dumps(out, indent=2, default=str))
    else:
        print(report(kpi, alerts))
    # Non-zero exit when a bottleneck is active so automations can branch on it.
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
