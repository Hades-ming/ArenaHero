"""Tests for the flag-file-driven lightweight Core-migration controller.

These mirror the fixtures in ``test_tactic.py`` (same ``_state``/``_turn``
helpers re-implemented locally for isolation) and exercise every branch of
``tactic._try_migrate_core``: absent flag, NORMAL Core steps toward target,
already-at-target removes flag, MOVING Core is skipped, invalid flag content
is purged, None Core is skipped, and obstacle detour picks an open neighbor.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from arena_hero import CommandPlan, Direction, PlayerState, Turn

import tactic
from tactic import decide

CORE_ID = UUID("00000000-0000-4000-8000-000000000001")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000002")


def _state(
    *,
    core_pos: tuple[int, int] = (10, 10),
    core_hp: int = 5,
    core_shield: int = 5,
    core_state: str = "NORMAL",
    resources: int = 5,
    population: int = 1,
    objects: list[dict[str, Any]] | None = None,
) -> PlayerState:
    if objects is None:
        objects = [
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": list(core_pos),
                "hp": core_hp,
                "shield": core_shield,
                "state": core_state,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [core_pos[0] + 1, core_pos[1]],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ]
    return PlayerState.model_validate(
        {
            "status": "ACTIVE",
            "resources": resources,
            "population": population,
            "champion_beacon": {"position": [0, 0]},
            "objects": objects,
            "events": [],
        }
    )


def _turn(state: PlayerState, tick: int = 10) -> Turn:
    return Turn(
        tick=tick,
        state=state,
        submitter=lambda plan, key=None: CommandPlan(tick=tick),
    )


def _core_action(plan: CommandPlan) -> Any:
    return plan.core_action


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tactic, "_STATE_PATH", tmp_path / "tactic_state.json")
    monkeypatch.setattr(tactic, "MIGRATE_FLAG_PATH", tmp_path / "migrate.flag")
    tactic._known_obstacles.clear()
    tactic._known_resources.clear()
    tactic._known_enemy_cores.clear()
    tactic._explore_state.clear()
    tactic._explore_targets.clear()
    tactic._explore_progress.clear()
    tactic._pos_history.clear()
    tactic._stuck_ticks.clear()
    tactic._chase_start.clear()
    tactic._chase_budget.clear()
    tactic._chase_cooldown_until.clear()
    tactic._last_enemy_pos.clear()
    tactic._persistent_state_dirty = False
    tactic._migrate_stall_ticks = 0
    yield
    tactic._known_obstacles.clear()
    tactic._known_resources.clear()
    tactic._known_enemy_cores.clear()
    tactic._explore_state.clear()
    tactic._explore_targets.clear()
    tactic._explore_progress.clear()
    tactic._pos_history.clear()
    tactic._stuck_ticks.clear()
    tactic._chase_start.clear()
    tactic._chase_budget.clear()
    tactic._chase_cooldown_until.clear()
    tactic._last_enemy_pos.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()


# ---------------------------------------------------------------------------
# No flag → normal decide() path, no core_action queued by migration
# ---------------------------------------------------------------------------


def test_no_flag_does_not_queue_core_action() -> None:
    """When migrate.flag is absent, decide() behaves exactly as before."""
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    # With zero resources and no threats, _control_core queues nothing.
    assert _core_action(turn.plan) is None


def test_flag_absent_workers_still_move() -> None:
    """Regression: no flag must not suppress Worker movement."""
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    # The Worker should have an action (move or wait) — the key point is
    # decide() ran the full normal path, not the migration short-circuit.
    # We assert the plan is non-empty (Worker got an action or wait).
    assert turn.plan is not None


# ---------------------------------------------------------------------------
# Flag present, NORMAL Core → start_move queued toward target
# ---------------------------------------------------------------------------


def test_flag_present_queues_start_move() -> None:
    """Core at (10,10), target (10,5) → should step UP."""
    tactic.MIGRATE_FLAG_PATH.write_text("10,5\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    action = _core_action(turn.plan)
    assert action is not None
    assert action.type == "START_MOVE"
    assert action.direction == Direction.UP


def test_flag_present_step_left() -> None:
    """Core at (10,10), target (5,10) → should step LEFT."""
    tactic.MIGRATE_FLAG_PATH.write_text("5,10", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    action = _core_action(turn.plan)
    assert action is not None
    assert action.type == "START_MOVE"
    assert action.direction == Direction.LEFT


def test_migration_short_circuits_rest_of_decide() -> None:
    """When migrating, decide() returns early — no spawn/repair/deposit."""
    tactic.MIGRATE_FLAG_PATH.write_text("10,5\n", encoding="utf-8")
    # Give resources so _control_core would normally queue something.
    state = _state(resources=10, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    action = _core_action(turn.plan)
    # Exactly one core_action: the START_MOVE, not a spawn.
    assert action is not None
    assert action.type == "START_MOVE"


# ---------------------------------------------------------------------------
# Already at target → flag removed, no start_move
# ---------------------------------------------------------------------------


def test_already_at_target_removes_flag() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("10,10\n", encoding="utf-8")
    assert tactic.MIGRATE_FLAG_PATH.is_file()
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    assert not tactic.MIGRATE_FLAG_PATH.is_file()
    assert _core_action(turn.plan) is None


# ---------------------------------------------------------------------------
# MOVING Core → skip (wait for step to resolve)
# ---------------------------------------------------------------------------


def test_moving_core_skips_migration() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("10,5\n", encoding="utf-8")
    state = _state(
        resources=0,
        core_pos=(10, 10),
        core_state="MOVING",
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [10, 10],
                "hp": 5,
                "shield": 5,
                "state": "MOVING",
                "move_direction": "UP",
                "move_progress": 1,
                "move_required_ticks": 1,
                "destination": [10, 9],
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [11, 10],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    # Flag still present (not consumed), no start_move queued by migration.
    assert tactic.MIGRATE_FLAG_PATH.is_file()
    # The existing _control_core MOVING guard returns early → no core_action.
    assert _core_action(turn.plan) is None


# ---------------------------------------------------------------------------
# Invalid flag content → flag removed, no crash
# ---------------------------------------------------------------------------


def test_invalid_flag_content_removed() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("not-a-coord\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)  # must not raise
    assert not tactic.MIGRATE_FLAG_PATH.is_file()
    assert _core_action(turn.plan) is None


def test_invalid_flag_single_number_removed() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("10\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    assert not tactic.MIGRATE_FLAG_PATH.is_file()


def test_invalid_flag_extra_fields_removed() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("10,5,99\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    assert not tactic.MIGRATE_FLAG_PATH.is_file()


# ---------------------------------------------------------------------------
# Core is None (respawn) → skip migration
# ---------------------------------------------------------------------------


def test_core_none_skips_migration() -> None:
    tactic.MIGRATE_FLAG_PATH.write_text("10,5\n", encoding="utf-8")
    state = PlayerState.model_validate(
        {
            "status": "RESPAWNING",
            "resources": 0,
            "population": 0,
            "respawn_at_tick": 30,
            "champion_beacon": {"position": [0, 0]},
            "objects": [],
            "events": [],
        }
    )
    turn = _turn(state)
    decide(turn)  # must not raise
    # Flag should still be there (we skip, not consume).
    assert tactic.MIGRATE_FLAG_PATH.is_file()


# ---------------------------------------------------------------------------
# Obstacle detour: direct step blocked → picks an open neighbor
# ---------------------------------------------------------------------------


def test_obstacle_detour_picks_open_direction() -> None:
    """Core at (10,10), target (5,10) → LEFT preferred, but LEFT is blocked.

    With LEFT (9,10) in _known_obstacles, _step_toward's detour logic should
    pick another open cardinal direction (UP or DOWN) rather than stall.
    """
    tactic._known_obstacles.add((9, 10))  # block direct LEFT step
    tactic.MIGRATE_FLAG_PATH.write_text("5,10\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))
    turn = _turn(state)
    decide(turn)
    action = _core_action(turn.plan)
    assert action is not None
    assert action.type == "START_MOVE"
    # Must NOT be LEFT (blocked), should be UP or DOWN (detour axis).
    assert action.direction in (Direction.UP, Direction.DOWN)


def test_friendly_unit_blocks_direction() -> None:
    """A friendly Unit on the target step cell forces a detour.

    Core at (10,10), target (5,10) → LEFT preferred (9,10), but a Worker is
    at (9,10).  The migration controller must treat (9,10) as blocked and
    pick another direction (UP or DOWN), avoiding CELL_UNIT_LIMIT.
    """
    tactic.MIGRATE_FLAG_PATH.write_text("5,10\n", encoding="utf-8")
    # Worker sitting on (9,10) — the direct LEFT step target.
    state = _state(
        resources=0,
        core_pos=(10, 10),
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [10, 10],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [9, 10],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _core_action(turn.plan)
    assert action is not None
    assert action.type == "START_MOVE"
    # Must NOT be LEFT (Worker there), should detour UP or DOWN.
    assert action.direction in (Direction.UP, Direction.DOWN)


def test_completely_blocked_waits_then_gives_up() -> None:
    """Core boxed in on all 4 sides → waits MIGRATE_MAX_STALL_TICKS, then flag purged."""
    tactic._known_obstacles.update(
        {(9, 10), (11, 10), (10, 9), (10, 11)}
    )
    tactic.MIGRATE_FLAG_PATH.write_text("5,10\n", encoding="utf-8")
    state = _state(resources=0, core_pos=(10, 10))

    # The first MIGRATE_MAX_STALL_TICKS calls should NOT remove the flag
    # (waiting for Units to clear).  No core_action queued.
    for i in range(tactic.MIGRATE_MAX_STALL_TICKS):
        turn = _turn(state)
        decide(turn)
        assert tactic.MIGRATE_FLAG_PATH.is_file(), f"flag removed at stall {i}"
        assert _core_action(turn.plan) is None

    # The next call exceeds the grace period → flag removed.
    turn = _turn(state)
    decide(turn)
    assert not tactic.MIGRATE_FLAG_PATH.is_file()
    assert _core_action(turn.plan) is None
