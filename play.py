"""Run the balanced tactic against the live Arena Hero server.

The decision logic lives in :mod:`tactic`; this module only connects, submits
the plan the tactic builds, and reports progress. The API key is read from
``ARENA_HERO_API_KEY``, a ``.env`` file, or a hidden prompt, and is never
printed.

Each Tick a compact structured line is written to stderr and appended to
``game.log`` so the run can be monitored and iterated on. The key never
appears in any log line.

Usage::

    python play.py                 # reads ARENA_HERO_API_KEY / .env / prompt
    python play.py --local         # point at a local server

Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path

from arena_hero import (
    APIError,
    ArenaHeroClient,
    AuthenticationError,
    Received,
    HarvestSource,
    PolicyViolationError,
    ProtocolError,
    Tick,
    Turn,
    TransportError,
    TurnClosedError,
)

from tactic import decide
import tactic

DEFAULT_BASE_URL = "https://api.arenahero.io"
LOCAL_BASE_URL = "http://localhost:8080"
LOCAL_WS_URL = "ws://localhost:8080/api/v1/game/ws"
API_KEY_ENV = "ARENA_HERO_API_KEY"
LOG_PATH = Path(__file__).resolve().parent / "game.log"


def _api_key_from_env_file(path: Path) -> str | None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == API_KEY_ENV:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value or None
    return None


def load_api_key(api_key_file: Path | None = None) -> str:
    if api_key_file is not None:
        key = _api_key_from_env_file(api_key_file)
        if key is None and "\n" not in api_key_file.read_text(encoding="utf-8-sig").strip():
            key = api_key_file.read_text(encoding="utf-8-sig").strip()
        if not key:
            raise ValueError(f"No API key found in {api_key_file}")
        return key
    if key := os.environ.get(API_KEY_ENV, "").strip():
        return key
    env_path = Path.cwd() / ".env"
    if env_path.is_file() and (key := _api_key_from_env_file(env_path)):
        return key
    if not sys.stdin.isatty():
        raise ValueError(f"Set {API_KEY_ENV}, add it to .env, or pass --api-key-file")
    key = getpass("Arena Hero API key: ").strip()
    if not key:
        raise ValueError("API key cannot be empty")
    return key


def _summarize_plan(plan, *, full_ids: bool = False) -> str:
    parts: list[str] = []
    for _uid, action in plan.unit_actions.items():
        name = type(action).__name__.replace("Action", "").lower()
        # Include direction/target hint where useful.
        extra = ""
        direction = getattr(action, "direction", None)
        if direction is not None:
            extra = f":{direction.value}"
        uid = str(_uid) if full_ids else str(_uid)[-6:]
        parts.append(f"U{uid}:{name}{extra}")
    if plan.core_action is not None:
        cname = type(plan.core_action).__name__.replace("Action", "").lower()
        unit_type = getattr(plan.core_action, "unit_type", None)
        extra = f":{unit_type.value}" if unit_type is not None else ""
        parts.append(f"C:{cname}{extra}")
    return ",".join(parts) if parts else "-"


def _summarize_events(events) -> str:
    if not events:
        return "-"
    out: list[str] = []
    for e in events:
        tag = e.event_type
        if e.reason_code:
            tag += f".{e.reason_code}"
        if e.event_type in {
            "HARVEST_SUCCEEDED",
            "DEPOSIT_SUCCEEDED",
            "WORKER_CARGO_DROPPED",
            "CORE_RESOURCE_OVERFLOW_DESTROYED",
            "CORE_RESOURCES_CAPTURED",
        } and e.resource_amount is not None:
            tag += f"[{e.resource_amount}]"
        if e.event_type == "HARVEST_SUCCEEDED" and e.harvest_source is HarvestSource.DROPPED_CARGO:
            tag += "(pile)"
        out.append(tag)
    return ";".join(out)


def _summarize_event_details(events) -> str:
    """序列化有界事件详情，避免把服务端错误文本写入日志。"""
    out: list[str] = []
    for event in events:
        fields = [event.event_type]
        fields.append(f"tick={event.tick}")
        if event.reason_code:
            fields.append(f"reason={event.reason_code}")
        if event.event_id:
            fields.append(f"event={event.event_id}")
        if event.actor_id:
            fields.append(f"actor={event.actor_id}")
        if event.target_id:
            fields.append(f"target={event.target_id}")
        if event.position is not None:
            fields.append(f"pos={event.position[0]},{event.position[1]}")
        values = event.values or {}
        # 只保留 v0.14 KPI 解析所需的数值与枚举样式字段。
        for key in ("amount", "available", "destroyed", "capacity", "cost", "required", "hp"):
            value = values.get(key)
            if type(value) is int:
                fields.append(f"{key}={value}")
        source = values.get("source")
        if isinstance(source, str) and re.fullmatch(r"[A-Z_]{1,32}", source):
            fields.append(f"source={source}")
        # 分号分隔事件、竖线分隔字段；上面的值均为受限 ASCII 标记，无需转义。
        out.append("|".join(fields))
    return ";".join(out)


_SAFE_ERROR_CODE = re.compile(r"[A-Z0-9_.-]{1,64}")
_SAFE_STATUS = re.compile(r"[A-Z_]{1,32}")


def _safe_log_token(value: object, pattern: re.Pattern[str], fallback: str) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if pattern.fullmatch(candidate):
            return candidate
    return fallback


def _safe_error_code(exc: BaseException, fallback: str = "UNKNOWN_ERROR") -> str:
    """Return a bounded code without copying server messages into logs."""
    known = (
        (TurnClosedError, "TURN_CLOSED"),
        (APIError, "API_ERROR"),
        (TransportError, "TRANSPORT_ERROR"),
        (ProtocolError, "PROTOCOL_ERROR"),
        (AuthenticationError, "AUTHENTICATION_ERROR"),
        (PolicyViolationError, "POLICY_VIOLATION"),
    )
    for error_type, code in known:
        if isinstance(exc, error_type):
            raw = getattr(exc, "error", None)
            candidate = _safe_log_token(raw, _SAFE_ERROR_CODE, "")
            if candidate:
                return candidate
            return code
    return fallback


def _append_log_line(line: str) -> None:
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _failure_log_line(
    tick: int,
    status: str,
    error_code: str,
    decide_ms: int,
    submit_ms: int,
    total_local_ms: int,
) -> str:
    status = _safe_log_token(status, _SAFE_STATUS, "SUBMIT_FAILED")
    error_code = _safe_log_token(error_code, _SAFE_ERROR_CODE, "UNKNOWN_ERROR")
    return (
        f"t{tick} ST[{status}] ER[{error_code}] "
        f"TM[{decide_ms},{submit_ms},{total_local_ms}]"
    )


def _log_line(
    turn,
    accepted=None,
    decide_ms: int = 0,
    submit_ms: int = 0,
    total_local_ms: int = 0,
    dup: int = 0,
    status: str = "ACCEPTED",
    error_code: str | None = None,
) -> str:
    s = turn.state
    core = turn.core
    core_desc = "respawn" if core is None else f"{core.position[0]},{core.position[1]} hp{core.hp}/sh{core.shield}/{core.view.state}"
    enemy_types = [getattr(e, "unit_type", None) or "CORE" for e in turn.visible_enemies]
    workers_desc = ",".join(
        f"{w.position[0]},{w.position[1]}c{w.cargo}{str(w.id)[-6:]}ty{tactic._worker_column(idx, len(turn.workers), turn.core.position if turn.core else (0,0))+(tactic._explore_state.get(str(w.id),[0,0])[0])}"
        for idx, w in enumerate(turn.workers)
    ) or "-"
    others_desc = ",".join(
        f"{u.position[0]},{u.position[1]}" for u in (*turn.vanguards, *turn.rangers)
    ) or "-"
    enemies_desc = ",".join(
        f"{e.position[0]},{e.position[1]}{'C' if e.kind == 'CORE' else getattr(e,'unit_type',None)}" for e in turn.visible_enemies
    ) or "-"
    status = _safe_log_token(status, _SAFE_STATUS, "ACCEPTED")
    safe_error_code = _safe_log_token(error_code, _SAFE_ERROR_CODE, "UNKNOWN_ERROR")
    dup_flag = f" dup" if dup else ""
    return (
        f"t{turn.tick} "
        f"r{turn.resources}/{turn.resource_capacity} "
        f"pop{s.population}(W{len(turn.workers)} V{len(turn.vanguards)} R{len(turn.rangers)}) "
        f"core@{core_desc} "
        f"W[{workers_desc}] O[{others_desc}] "
        f"vis{len(turn.visible_enemies)}[{enemies_desc}] "
        f"res{len(turn.resource_cells)}[{','.join(f'{c[0]},{c[1]}' for c in sorted(turn.resource_cells))}] "
        f"obs{len(turn.obstacle_cells)} "
        f"beacon{turn.beacon.position[0]},{turn.beacon.position[1]} "
        f"eco[{tactic._resource_telemetry_summary()}] "
        f"TM[{decide_ms},{submit_ms},{total_local_ms}] ST[{status}]"
        f"{f' ER[{safe_error_code}]' if error_code else ''}{dup_flag} "
        f"ev[{_summarize_events(turn.events)}] "
        f"dt[{_summarize_event_details(turn.events)}] "
        f"plan[{_summarize_plan(turn.plan)}]"
    )


def _receipt_log_line(receipt: Received) -> str:
    """记录 receipt Tick 的 Agent/Manual 规范计划归属。"""
    source = receipt.source.value
    action_count = len(receipt.plan.unit_actions) + (
        1 if receipt.plan.core_action is not None else 0
    )
    return (
        f"t{receipt.tick} rcv[{source}] actions[{action_count}] "
        f"plan[{_summarize_plan(receipt.plan, full_ids=True)}]"
    )


def play(api_key: str, base_url: str, websocket_url: str | None) -> int:
    try:
        with ArenaHeroClient(api_key=api_key, base_url=base_url, websocket_url=websocket_url) as game:
            last_turn_tick: int | None = None
            logged_receipts: set[tuple[int, str, str]] = set()
            event_stream = game.events() if hasattr(game, "events") else game.turns()
            for stream_event in event_stream:
                if isinstance(stream_event, Received):
                    receipt_plan = _summarize_plan(stream_event.plan, full_ids=True)
                    receipt_key = (
                        stream_event.tick,
                        stream_event.source.value,
                        receipt_plan,
                    )
                    if receipt_key in logged_receipts:
                        continue
                    logged_receipts.add(receipt_key)
                    # 重连或重放时限制集合大小，同时抑制当前 Tick 的重复 receipt。
                    if len(logged_receipts) > 256:
                        floor = stream_event.tick - 8
                        logged_receipts = {key for key in logged_receipts if key[0] >= floor}
                    receipt_line = _receipt_log_line(stream_event)
                    print(receipt_line, file=sys.stderr)
                    _append_log_line(receipt_line)
                    continue
                if isinstance(stream_event, Tick):
                    continue
                # 兼容离线失败路径测试中的简化 Turn 适配器。
                if not isinstance(stream_event, Turn) and not hasattr(stream_event, "tick"):
                    continue
                turn = stream_event
                if turn.tick == last_turn_tick:
                    continue
                last_turn_tick = turn.tick
                t0 = time.monotonic()
                try:
                    decide(turn)
                except Exception as exc:
                    t1 = time.monotonic()
                    decide_ms = int((t1 - t0) * 1000)
                    error_code = _safe_error_code(exc, "DECISION_ERROR")
                    line = _failure_log_line(
                        turn.tick, "DECISION_FAILED", error_code, decide_ms, 0, decide_ms
                    )
                    print(line, file=sys.stderr)
                    _append_log_line(line)
                    continue

                t1 = time.monotonic()
                try:
                    accepted = turn.submit()
                except Exception as exc:
                    t2 = time.monotonic()
                    decide_ms = int((t1 - t0) * 1000)
                    submit_ms = int((t2 - t1) * 1000)
                    total_local_ms = int((t2 - t0) * 1000)
                    error_code = _safe_error_code(exc)
                    line = _failure_log_line(
                        turn.tick,
                        "SUBMIT_FAILED",
                        error_code,
                        decide_ms,
                        submit_ms,
                        total_local_ms,
                    )
                    print(line, file=sys.stderr)
                    _append_log_line(line)
                    if isinstance(exc, (AuthenticationError, PolicyViolationError)):
                        raise
                    continue

                t2 = time.monotonic()
                decide_ms = int((t1 - t0) * 1000)
                submit_ms = int((t2 - t1) * 1000)
                total_local_ms = int((t2 - t0) * 1000)
                line = _log_line(
                    turn,
                    accepted,
                    decide_ms,
                    submit_ms,
                    total_local_ms,
                    status="ACCEPTED",
                )
                print(line, file=sys.stderr)
                _append_log_line(line)
    except (
        AuthenticationError,
        PolicyViolationError,
        ProtocolError,
        APIError,
        TransportError,
        TurnClosedError,
    ) as exc:
        print(f"game_stopped ({_safe_error_code(exc)})", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Arena Hero balanced tactic.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--websocket-url", default=None)
    parser.add_argument("--local", action="store_true", help="use a local server")
    parser.add_argument("--api-key-file", type=Path, default=None)
    args = parser.parse_args(argv)

    base_url = LOCAL_BASE_URL if args.local else args.base_url
    websocket_url = LOCAL_WS_URL if args.local else args.websocket_url

    try:
        api_key = load_api_key(args.api_key_file)
    except (OSError, ValueError) as exc:
        print(f"Cannot load API key: {exc}", file=sys.stderr)
        return 2

    # Truncate the per-run log so the latest run is easy to read.
    try:
        LOG_PATH.write_text("", encoding="utf-8")
    except OSError:
        pass

    print(
        f"Starting balanced tactic (logging to {LOG_PATH}). Ctrl-C to stop. "
        f"Watch at https://app.arenahero.io/arena",
        file=sys.stderr,
    )
    try:
        return play(api_key, base_url, websocket_url)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
