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
import sys
from datetime import datetime
from getpass import getpass
from pathlib import Path

from arena_hero import ArenaHeroClient, APIError, HarvestSource, TurnClosedError, TransportError

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


def _summarize_plan(plan) -> str:
    parts: list[str] = []
    for _uid, action in plan.unit_actions.items():
        name = type(action).__name__.replace("Action", "").lower()
        # Include direction/target hint where useful.
        extra = ""
        direction = getattr(action, "direction", None)
        if direction is not None:
            extra = f":{direction.value}"
        uid_short = str(_uid)[-6:]
        parts.append(f"U{uid_short}:{name}{extra}")
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


def _log_line(turn, accepted) -> str:
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
        f"ev[{_summarize_events(turn.events)}] "
        f"plan[{_summarize_plan(turn.plan)}]"
    )


def play(api_key: str, base_url: str, websocket_url: str | None) -> int:
    submitted = 0
    with ArenaHeroClient(api_key=api_key, base_url=base_url, websocket_url=websocket_url) as game:
        for turn in game.turns():
            decide(turn)
            try:
                accepted = turn.submit()
            except (TurnClosedError, APIError, TransportError) as exc:
                # A stale tick, a closed window, a rate-limit, or a transient
                # transport failure (network/server) is recoverable: the next
                # Turn carries fresh state. Log and keep going rather than
                # killing the run. COMMAND_WINDOW_CLOSED and TICK_MISMATCH
                # arrive as APIError here; TransportError covers submission
                # failures that exhausted safe retries.
                err = getattr(exc, "error", None) or type(exc).__name__
                print(f"t{turn.tick} submit_skipped ({err})", file=sys.stderr)
                try:
                    with LOG_PATH.open("a", encoding="utf-8") as fh:
                        fh.write(f"t{turn.tick} submit_skipped ({err})\n")
                except OSError:
                    pass
                continue
            submitted += 1
            line = _log_line(turn, accepted)
            print(line, file=sys.stderr)
            try:
                with LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
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
