"""Decision tests for the balanced tactic, without live credentials or network.

These build ``PlayerState`` fixtures and a ``Turn`` whose submitter is a stub,
then assert on the queued ``turn.plan``. They follow the coverage list in the
bundled ``references/tactic-authoring.md``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from arena_hero import (
    BeaconStatus,
    CommandPlan,
    CoreView,
    Direction,
    HarvestSource,
    PlayerState,
    Turn,
    UnitType,
    UnitView,
)

import tactic
from tactic import decide

CORE_ID = UUID("00000000-0000-4000-8000-000000000001")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000002")
WORKER2_ID = UUID("00000000-0000-4000-8000-000000000003")
VANGUARD_ID = UUID("00000000-0000-4000-8000-000000000004")
RANGER_ID = UUID("00000000-0000-4000-8000-000000000005")
ENEMY_CORE_ID = UUID("00000000-0000-4000-8000-000000000099")
ENEMY_UNIT_ID = UUID("00000000-0000-4000-8000-000000000098")


@pytest.fixture(autouse=True)
def _reset_explore_state(tmp_path, monkeypatch) -> None:
    """Each test is isolated, including the persistent map file."""
    # ``decide`` flushes dirty map state at the end of a Tick. Point tests at a
    # temporary file so the suite can never overwrite the live tactic's map.
    monkeypatch.setattr(tactic, "_STATE_PATH", tmp_path / "tactic_state.json")
    tactic._explore_state.clear()
    tactic._explore_targets.clear()
    tactic._explore_target_cooldown_until.clear()
    tactic._explore_target_failures.clear()
    tactic._explore_progress.clear()
    tactic._known_resources.clear()
    tactic._resource_claims.clear()
    tactic._resource_hints.clear()
    tactic._resource_telemetry.clear()
    tactic._resource_absence_streak = 0
    tactic._last_harvest_tick = None
    tactic._persistent_state_dirty = False
    tactic._known_obstacles.clear()
    tactic._known_enemy_cores.clear()
    tactic._explored_cells.clear()
    tactic._pos_history.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()
    tactic._stuck_ticks.clear()
    tactic._chase_start.clear()
    tactic._chase_budget.clear()
    tactic._chase_cooldown_until.clear()
    yield
    tactic._explore_state.clear()
    tactic._explore_targets.clear()
    tactic._explore_target_cooldown_until.clear()
    tactic._explore_target_failures.clear()
    tactic._explore_progress.clear()
    tactic._known_resources.clear()
    tactic._resource_claims.clear()
    tactic._resource_hints.clear()
    tactic._resource_telemetry.clear()
    tactic._resource_absence_streak = 0
    tactic._last_harvest_tick = None
    tactic._persistent_state_dirty = False
    tactic._known_obstacles.clear()
    tactic._known_enemy_cores.clear()
    tactic._explored_cells.clear()
    tactic._pos_history.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()
    tactic._stuck_ticks.clear()
    tactic._chase_start.clear()
    tactic._chase_budget.clear()
    tactic._chase_cooldown_until.clear()


def _state(
    *,
    core_pos: tuple[int, int] = (0, 0),
    core_hp: int = 5,
    core_shield: int = 5,
    core_state: str = "NORMAL",
    resources: int = 5,
    population: int = 1,
    objects: list[dict[str, Any]] | None = None,
    beacon_pos: tuple[int, int] = (0, 0),
    beacon_status: str | None = None,
    beacon_carrier_id: UUID | None = None,
    events: list[dict[str, Any]] | None = None,
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
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ]
    beacon: dict[str, Any] = {"position": list(beacon_pos)}
    if beacon_status is not None:
        beacon["status"] = beacon_status
    if beacon_carrier_id is not None:
        beacon["carrier_id"] = str(beacon_carrier_id)
    return PlayerState.model_validate(
        {
            "status": "ACTIVE",
            "resources": resources,
            "population": population,
            "champion_beacon": beacon,
            "objects": objects,
            "events": events or [],
        }
    )


def _turn(state: PlayerState, tick: int = 10) -> Turn:
    return Turn(
        tick=tick,
        state=state,
        submitter=lambda plan, key=None: CommandPlan(tick=tick),
    )


def _worker(state: PlayerState) -> dict[str, Any]:
    return next(o for o in state.objects if o.get("kind") == "UNIT")  # type: ignore[union-attr]


def _action(plan: CommandPlan, unit_id: UUID) -> Any:
    return plan.unit_actions.get(unit_id)


def _core_action(plan: CommandPlan) -> Any:
    return plan.core_action


def test_respawning_submits_no_actions() -> None:
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
    decide(turn)
    assert turn.plan.unit_actions == {}
    assert turn.plan.core_action is None


def test_active_empty_world_explores_not_waits() -> None:
    # No visible resources, no enemies: the lone Worker must not stall. With a
    # small vision radius, standing still never reveals resources, so the
    # tactic marches an exploration direction instead of WAITing.
    turn = _turn(_state())
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "MOVE"


def test_exploration_assigns_different_bands_to_workers() -> None:
    # Two Workers should own different chunk-relative columns so they cover
    # different strips of the chunk width for full coverage.
    core = (181, 149)
    col0 = tactic._worker_column(0, 2, core)
    col1 = tactic._worker_column(1, 2, core)
    # Distinct columns keep the Workers on distinct sweep strips.
    assert col0 != col1
    # Both columns should be inside the home chunk x[160,191].
    chunk_x0 = core[0] // 32 * 32
    assert chunk_x0 <= col0 <= chunk_x0 + 31
    assert chunk_x0 <= col1 <= chunk_x0 + 31


def _workers_state(
    positions: list[tuple[int, int]],
    *,
    cargo: list[int] | None = None,
    resources: list[tuple[int, int]] | None = None,
) -> PlayerState:
    """Build a deterministic Core plus Worker fleet for dispatcher tests."""
    cargo = cargo or [0] * len(positions)
    objects: list[dict[str, Any]] = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        }
    ]
    for index, (position, worker_cargo) in enumerate(zip(positions, cargo, strict=True)):
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x6000 + index)),
                "controlled": True,
                "position": list(position),
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": worker_cargo,
            }
        )
    if resources:
        objects.append(
            {"kind": "RESOURCE", "positions": [list(cell) for cell in resources]}
        )
    return _state(population=len(positions), objects=objects)


def test_distant_history_resource_is_retained_but_not_assigned_when_healthy() -> None:
    distant = (50, 0)
    tactic._known_resources.add(distant)
    tactic._resource_hints[distant] = tactic.ResourceHint(19, "history")
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    tactic._observe_resources(turn)

    assert distant in tactic._known_resources
    assert tactic._worker_resource_assignments(turn) == {}
    assert tactic._resource_telemetry["far"] == 1


def test_remote_history_claim_is_released_before_a_long_trip() -> None:
    distant = (50, 0)
    worker_id = str(UUID(int=0x6000))
    tactic._known_resources.add(distant)
    tactic._resource_hints[distant] = tactic.ResourceHint(19, "history")
    tactic._resource_claims[worker_id] = distant
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    assert tactic._worker_resource_assignments(turn) == {}
    assert worker_id not in tactic._resource_claims
    assert tactic._resource_telemetry["far"] == 1


def test_history_claim_near_target_is_kept_for_completion() -> None:
    target = (30, 0)
    worker_id = str(UUID(int=0x6000))
    tactic._known_resources.add(target)
    tactic._resource_hints[target] = tactic.ResourceHint(19, "history")
    tactic._resource_claims[worker_id] = target
    turn = _turn(_workers_state([(29, 0)]), tick=20)

    assert tactic._worker_resource_assignments(turn) == {worker_id: target}


def test_distant_history_resource_is_assigned_after_harvest_drought() -> None:
    distant = (50, 0)
    tactic._known_resources.add(distant)
    tactic._resource_hints[distant] = tactic.ResourceHint(1, "history")
    tactic._last_harvest_tick = 1
    turn = _turn(_workers_state([(1, 0)]), tick=30)

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments == {str(UUID(int=0x6000)): distant}
    assert tactic._resource_telemetry["far"] == 0


def test_visible_distant_resource_is_always_assignable() -> None:
    distant = (50, 0)
    turn = _turn(_workers_state([(1, 0)], resources=[distant]), tick=20)
    tactic._observe_resources(turn)

    assert tactic._worker_resource_assignments(turn) == {
        str(UUID(int=0x6000)): distant
    }


def test_worker_on_visible_distant_resource_harvests_immediately() -> None:
    distant = (50, 0)
    turn = _turn(_workers_state([distant], resources=[distant]), tick=20)

    tactic._control_workers(turn, turn.core.position)

    assert _action(turn.plan, UUID(int=0x6000)).type == "HARVEST"


def test_history_dispatch_reserves_a_worker_for_exploration() -> None:
    """没有可见资源时，陈旧地图提示不能占用所有空闲 Worker。"""
    tactic._known_resources.update({(10, 0), (20, 0)})
    tactic._resource_hints.update(
        {
            (10, 0): tactic.ResourceHint(19, "history"),
            (20, 0): tactic.ResourceHint(19, "history"),
        }
    )
    turn = _turn(_workers_state([(0, 1), (0, 2), (0, 3)]), tick=20)

    assignments = tactic._worker_resource_assignments(turn)

    assert len(assignments) == 2
    assert len(set(assignments.values())) == 2
    assert tactic._resource_telemetry["explore_reserved"] == 1


def test_history_dispatch_keeps_nearest_worker_on_known_resource() -> None:
    # With no visible node, the far-away Worker should explore while the Worker
    # already beside the historical hint continues the resource attempt.  The
    # old Core-distance tie-breaker reserved the near Worker instead.
    cell = (30, 0)
    near_id = str(UUID(int=0x6000))
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(19, "history")
    turn = _turn(_workers_state([(29, 0), (0, 1)]), tick=20)

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments == {near_id: cell}


def test_history_dispatch_keeps_exploration_slot_when_visible_resource_exists() -> None:
    """历史提示不能占满 Worker，当前可见资源仍必须优先。"""
    visible = (10, 0)
    remembered = (100, 0)
    tactic._known_resources.add(remembered)
    tactic._resource_hints[remembered] = tactic.ResourceHint(19, "history")
    turn = _turn(
        _workers_state([(0, 1), (0, 2)], resources=[visible]),
        tick=20,
    )
    tactic._observe_resources(turn)

    assignments = tactic._worker_resource_assignments(turn)

    assert set(assignments.values()) == {visible}
    assert tactic._resource_telemetry["explore_reserved"] == 1


def test_legacy_history_hint_also_reserves_exploration_before_expiry() -> None:
    visible = (10, 0)
    remembered = (100, 0)
    tactic._known_resources.add(remembered)
    tactic._resource_hints[remembered] = tactic.ResourceHint(0, "legacy")
    turn = _turn(
        _workers_state([(0, 1), (0, 2)], resources=[visible]),
        tick=20,
    )

    assignments = tactic._worker_resource_assignments(turn)

    assert set(assignments.values()) == {visible}
    assert tactic._resource_telemetry["explore_reserved"] == 1


def test_visible_resources_cover_workers_before_history_reservation() -> None:
    visible = {(10, 0), (20, 0)}
    remembered = (100, 0)
    tactic._known_resources.add(remembered)
    tactic._resource_hints[remembered] = tactic.ResourceHint(0, "history")
    turn = _turn(
        _workers_state([(0, 1), (0, 2)], resources=list(visible)),
        tick=20,
    )

    assignments = tactic._worker_resource_assignments(turn)

    assert set(assignments.values()) == visible
    assert tactic._resource_telemetry["explore_reserved"] == 0


def test_visible_resource_keeps_nearest_worker_out_of_exploration_slot() -> None:
    visible = (321, 98)
    remembered = (100, 0)
    tactic._known_resources.add(remembered)
    tactic._resource_hints[remembered] = tactic.ResourceHint(19, "history")
    turn = _turn(
        _workers_state([(321, 95), (345, 126)], resources=[visible]),
        tick=20,
    )

    assignments = tactic._worker_resource_assignments(turn)

    nearest_worker = str(turn.workers[0].id)
    distant_worker = str(turn.workers[1].id)
    assert assignments[nearest_worker] == visible
    assert distant_worker not in assignments
    assert tactic._resource_telemetry["explore_reserved"] == 1


def test_visible_resource_claim_stays_with_worker_until_resolution() -> None:
    """移动中的 Worker 不应因另一名 Worker 变近而丢失当前资源认领。"""
    cell = (10, 0)
    first_id = str(UUID(int=0x6000))
    second_id = str(UUID(int=0x6001))

    first_turn = _turn(
        _workers_state([(8, 0), (4, 0)], resources=[cell]),
        tick=20,
    )
    tactic._observe_resources(first_turn)
    first_assignments = tactic._worker_resource_assignments(first_turn)
    assert first_assignments == {first_id: cell}

    # 第二名 Worker 现在更接近资源。每 Tick 重新匹配会换主并制造绕点运动；
    # 短期认领应一直保留到采集成功或明确失败。
    second_turn = _turn(
        _workers_state([(7, 0), (9, 0)], resources=[cell]),
        tick=21,
    )
    tactic._observe_resources(second_turn)
    second_assignments = tactic._worker_resource_assignments(second_turn)
    assert second_assignments == {first_id: cell}
    assert second_id not in second_assignments


def test_history_claim_keeps_progress_until_visible_node_is_decisively_closer() -> None:
    history = (20, 0)
    visible = (15, 0)
    first_id = str(UUID(int=0x6000))
    tactic._known_resources.add(history)
    tactic._resource_hints[history] = tactic.ResourceHint(19, "history")
    tactic._resource_claims[first_id] = history
    turn = _turn(
        _workers_state([(12, 0), (0, 1)], resources=[visible]),
        tick=20,
    )

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments[first_id] == history


def test_history_claim_yields_when_visible_node_is_much_closer() -> None:
    history = (20, 0)
    visible = (13, 0)
    first_id = str(UUID(int=0x6000))
    tactic._known_resources.add(history)
    tactic._resource_hints[history] = tactic.ResourceHint(19, "history")
    tactic._resource_claims[first_id] = history
    turn = _turn(
        _workers_state([(12, 0), (0, 1)], resources=[visible]),
        tick=20,
    )

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments[first_id] == visible


def test_very_old_history_hint_is_not_an_active_resource_target() -> None:
    cell = (100, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(1, "history")
    turn = _turn(_workers_state([(0, 0)]), tick=300)

    assert tactic._worker_resource_assignments(turn) == {}
    assert tactic._resource_telemetry["stale"] == 1


def test_harvest_failure_invalidates_history_hint_immediately() -> None:
    cell = (10, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(19, "history")
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": list(cell),
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
        events=[
            {
                "event_id": "00000000-0000-4000-8000-0000000000ac",
                "tick": 19,
                "event_type": "HARVEST_FAILED",
                "actor_id": str(WORKER_ID),
                "position": list(cell),
                "reason_code": "NOT_RESOURCE_CELL",
            }
        ],
    )
    turn = _turn(state, tick=20)

    decide(turn)

    assert cell not in tactic._known_resources
    assert cell not in tactic._resource_hints
    assert tactic._resource_telemetry["resource_failures"] == 1


def test_legacy_resource_hint_ages_out_after_restart() -> None:
    cell = (100, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(0, "legacy")
    turn = _turn(_workers_state([(0, 0)]), tick=300)

    assert tactic._worker_resource_assignments(turn) == {}
    assert tactic._resource_telemetry["stale"] == 1


def test_control_workers_reports_exploration_reservation_without_resources() -> None:
    turn = _turn(_workers_state([(0, 1), (0, 2), (0, 3)]), tick=20)

    tactic._control_workers(turn, turn.core.position)

    assert tactic._resource_telemetry["explore_reserved"] == 1
    assert len(turn.plan.unit_actions) == 3


def test_control_workers_reserves_two_refresh_patrols_without_visible_resources(
    monkeypatch,
) -> None:
    turn = _turn(_workers_state([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]), tick=20)
    captured: list[list[tuple[str, tuple[int, int]]]] = []

    def fake_assign(workers, *args, **kwargs):
        captured.append(list(workers))
        return {}

    monkeypatch.setattr(tactic, "_assign_explore_targets", fake_assign)
    tactic._control_workers(turn, turn.core.position)

    assert len(captured) == 1
    assert len(captured[0]) == 4
    assert tactic._resource_telemetry["refresh_reservations"] == 2
    assert len(turn.plan.unit_actions) == 6


def test_frontier_reassignment_keeps_refresh_patrols_reserved(monkeypatch) -> None:
    """前沿停滞重派不能抢走两个 Chunk 回扫 Worker。"""
    turn = _turn(
        _workers_state(
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        ),
        tick=20,
    )
    worker_ids = [str(UUID(int=0x6000 + index)) for index in range(6)]
    calls: list[list[str]] = []

    def fake_assign(workers, *args, **kwargs):
        calls.append([worker_id for worker_id, _ in workers])
        return {
            worker_id: (1, 1)
            for worker_id, _ in workers
            if worker_id == worker_ids[2]
        }

    monkeypatch.setattr(tactic, "_assign_explore_targets", fake_assign)
    monkeypatch.setattr(
        tactic,
        "_explore_target_has_stalled",
        lambda worker_id, *args, **kwargs: worker_id == worker_ids[2],
    )
    tactic._explore_targets[worker_ids[2]] = (1, 1)

    tactic._control_workers(turn, turn.core.position)

    assert calls
    assert all(worker_ids[index] not in call for call in calls for index in (0, 1))


def test_worker_destination_is_reserved_within_one_tick(monkeypatch) -> None:
    """同 Tick 的两个 Worker 不应预约同一个下一格。"""
    turn = _turn(_workers_state([(0, 2), (0, 2)]))

    monkeypatch.setattr(
        tactic,
        "_assign_explore_targets",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        tactic,
        "_explore_step",
        lambda *args, **kwargs: Direction.RIGHT,
    )

    tactic._control_workers(turn, turn.core.position)

    first_action = _action(turn.plan, UUID(int=0x6000))
    second_action = _action(turn.plan, UUID(int=0x6001))
    assert first_action is not None
    assert first_action.direction == Direction.RIGHT
    assert second_action is None


def test_visible_resource_priority_over_nearer_history_hint() -> None:
    remembered = (1, 0)
    visible = (10, 0)
    tactic._known_resources.add(remembered)
    turn = _turn(_workers_state([(0, 1)], resources=[visible]))
    tactic._observe_resources(turn)

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments[str(UUID(int=0x6000))] == visible


def test_global_resource_matching_beats_worker_order_greedy() -> None:
    # Greedy gives A->(1,0), B->(0,100), total 103. The global optimum is
    # A->(0,100), B->(1,0), total 101.
    resources = {(1, 0), (0, 100)}
    tactic._known_resources.update(resources)
    turn = _turn(_workers_state([(0, 0), (2, 0)], resources=list(resources)))

    assignments = tactic._worker_resource_assignments(turn)

    assert assignments == {
        str(UUID(int=0x6000)): (0, 100),
        str(UUID(int=0x6001)): (1, 0),
    }


def test_global_resource_matching_ignores_unit_input_order() -> None:
    tactic._known_resources.update({(1, 0), (0, 100)})
    state = _workers_state([(0, 0), (2, 0)], resources=[(1, 0), (0, 100)])
    turn = _turn(state)
    expected = tactic._worker_resource_assignments(turn)

    reversed_objects = [state.objects[0], *reversed(state.objects[1:])]
    reordered = state.model_copy(update={"objects": tuple(reversed_objects)})

    assert tactic._worker_resource_assignments(_turn(reordered)) == expected


def test_global_resource_matching_excludes_laden_workers_and_is_unique() -> None:
    tactic._known_resources.update({(1, 0), (2, 0)})
    turn = _turn(
        _workers_state(
            [(0, 0), (0, 1), (0, 2)],
            cargo=[1, 0, 0],
            resources=[(1, 0), (2, 0)],
        )
    )

    assignments = tactic._worker_resource_assignments(turn)

    assert str(UUID(int=0x6000)) not in assignments
    assert len(assignments) == 2
    assert len(set(assignments.values())) == 2


def test_blocked_resource_is_excluded_before_global_matching() -> None:
    blocked = (1, 0)
    available = (0, 3)
    turn = _turn(_workers_state([(0, 0)], resources=[blocked, available]))

    assignments = tactic._worker_resource_assignments(
        turn, blocked_resources=frozenset({blocked})
    )

    assert assignments == {str(UUID(int=0x6000)): available}


def test_friendly_full_resource_retargets_worker_to_available_resource() -> None:
    worker_id = UUID(int=0x6000)
    first_guard = UUID(int=0x7000)
    second_guard = UUID(int=0x7001)
    state = _state(
        population=3,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(worker_id),
                "controlled": True,
                "position": [0, 1],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(first_guard),
                "controlled": True,
                "position": [1, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
            {
                "kind": "UNIT",
                "id": str(second_guard),
                "controlled": True,
                "position": [1, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
            {"kind": "RESOURCE", "positions": [[1, 1], [0, 3]]},
        ],
    )
    turn = _turn(state)

    decide(turn)

    action = _action(turn.plan, worker_id)
    assert action.type == "MOVE"
    assert action.direction == Direction.DOWN


def test_enemy_occupied_resource_retargets_worker_to_available_resource() -> None:
    state = _workers_state([(0, 1)], resources=[(1, 1), (0, 3)])
    enemy = UnitView.model_validate(
        {
            "kind": "UNIT",
            "id": str(ENEMY_UNIT_ID),
            "controlled": False,
            "position": [1, 1],
            "hp": 4,
            "unit_type": "VANGUARD",
        }
    )
    state = state.model_copy(update={"objects": (*state.objects, enemy)})
    turn = _turn(state)

    decide(turn)

    action = _action(turn.plan, UUID(int=0x6000))
    assert action.type == "MOVE"
    assert action.direction == Direction.DOWN


def test_legacy_resource_state_loads_with_default_hint_metadata(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "tactic_state.json"
    state_path.write_text(
        '{"known_resources":[[7,8]],"known_obstacles":[],"known_enemy_cores":[],"explored_cells":[]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(tactic, "_STATE_PATH", state_path)

    tactic._load_persistent_state()

    assert tactic._known_resources == {(7, 8)}
    hint = tactic._resource_hints[(7, 8)]
    assert hint.last_confirmed_tick == 0
    assert hint.source == "legacy"
    assert hint.failure_count == 0
    assert hint.cooldown_until == 0


def test_resource_hint_metadata_round_trips_to_persistent_state(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "tactic_state.json"
    monkeypatch.setattr(tactic, "_STATE_PATH", state_path)
    cell = (11, 12)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=42,
        source="history",
        failure_count=2,
        cooldown_until=50,
    )

    tactic._save_persistent_state()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    tactic._known_resources.clear()
    tactic._resource_hints.clear()
    tactic._load_persistent_state()

    assert saved["resource_hints"] == [
        {
            "position": [11, 12],
            "last_confirmed_tick": 42,
            "source": "history",
            "failure_count": 2,
            "cooldown_until": 50,
        }
    ]
    assert tactic._resource_hints[cell].cooldown_until == 50


def test_bad_or_orphan_resource_hint_does_not_corrupt_persistent_map(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "tactic_state.json"
    state_path.write_text(
        json.dumps(
            {
                "known_resources": [[7, 8]],
                "resource_hints": [
                    {
                        "position": [7, 8],
                        "failure_count": [],
                        "cooldown_until": None,
                    },
                    {"position": [99, 99], "failure_count": 1},
                ],
                "known_obstacles": [[4, 4]],
                "known_enemy_cores": [[20, 20]],
                "explored_cells": [[1, 1]],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tactic, "_STATE_PATH", state_path)

    tactic._load_persistent_state()

    assert tactic._known_resources == {(7, 8)}
    assert tactic._resource_hints[(7, 8)].failure_count == 0
    assert tactic._resource_hints[(7, 8)].cooldown_until == 0
    assert tactic._known_obstacles == {(4, 4)}
    assert tactic._known_enemy_cores == {(20, 20)}
    assert tactic._explored_cells == {(1, 1)}


def test_resource_cooldown_failure_count_is_bounded() -> None:
    cell = (100, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=1,
        source="history",
        failure_count=10**6,
    )

    tactic._cooldown_resource(cell, tick=20)

    hint = tactic._resource_hints[cell]
    assert hint.failure_count == tactic._RESOURCE_FAILURE_CAP
    assert hint.cooldown_until == 20 + tactic._RESOURCE_COOLDOWN_CAP


def test_multiple_resource_cooldowns_flush_persistent_state_once(monkeypatch) -> None:
    writes = 0

    def fake_save() -> None:
        nonlocal writes
        writes += 1
        tactic._persistent_state_dirty = False

    monkeypatch.setattr(tactic, "_save_persistent_state", fake_save)
    for cell in ((100, 0), (101, 0)):
        tactic._known_resources.add(cell)
        tactic._cooldown_resource(cell, tick=20)

    assert writes == 0
    tactic._flush_persistent_state()
    tactic._flush_persistent_state()
    assert writes == 1


def test_reloaded_cooldown_expires_and_restores_assignment(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "tactic_state.json"
    monkeypatch.setattr(tactic, "_STATE_PATH", state_path)
    cell = (25, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=5,
        source="history",
        failure_count=2,
        cooldown_until=30,
    )
    tactic._save_persistent_state()
    tactic._known_resources.clear()
    tactic._resource_hints.clear()
    tactic._load_persistent_state()

    assert tactic._worker_resource_assignments(
        _turn(_workers_state([(0, 0)]), tick=29)
    ) == {}
    assert tactic._worker_resource_assignments(
        _turn(_workers_state([(0, 0)]), tick=30)
    ) == {str(UUID(int=0x6000)): cell}


def test_visible_resource_refreshes_hint_and_clears_cooldown() -> None:
    cell = (3, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=1,
        source="history",
        failure_count=3,
        cooldown_until=50,
    )
    turn = _turn(_workers_state([(0, 0)], resources=[cell]), tick=20)

    tactic._observe_resources(turn)

    hint = tactic._resource_hints[cell]
    assert hint.last_confirmed_tick == 20
    assert hint.source == "visible"
    assert hint.failure_count == 0
    assert hint.cooldown_until == 0
    assert tactic._worker_resource_assignments(turn) == {
        str(UUID(int=0x6000)): cell
    }


def test_cooled_history_hint_is_retained_but_not_assigned() -> None:
    cell = (100, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=5,
        source="history",
        failure_count=2,
        cooldown_until=30,
    )
    turn = _turn(_workers_state([(0, 0)]), tick=20)

    tactic._observe_resources(turn)

    assert cell in tactic._known_resources
    assert tactic._worker_resource_assignments(turn) == {}


def test_cooled_history_hint_recovers_immediately_when_visible() -> None:
    cell = (100, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=5,
        source="history",
        failure_count=2,
        cooldown_until=30,
    )
    turn = _turn(_workers_state([(0, 0)], resources=[cell]), tick=20)

    tactic._observe_resources(turn)

    assert tactic._resource_hints[cell].cooldown_until == 0
    assert tactic._worker_resource_assignments(turn) == {
        str(UUID(int=0x6000)): cell
    }


def test_unreachable_history_hint_is_cooled_and_worker_returns_to_frontier() -> None:
    cell = (10, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=5,
        source="history",
    )
    tactic._known_obstacles.update({(9, 0), (11, 0), (10, -1), (10, 1)})
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    decide(turn)

    hint = tactic._resource_hints[cell]
    assert cell in tactic._known_resources
    assert hint.failure_count == 1
    assert hint.cooldown_until == 24
    assert tactic._resource_telemetry["unreachable"] == 1
    assert _action(turn.plan, UUID(int=0x6000)).type == "MOVE"


def test_astar_budget_exhaustion_does_not_cool_reachable_history_hint() -> None:
    cell = (5000, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(
        last_confirmed_tick=5,
        source="history",
    )
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    decide(turn)

    hint = tactic._resource_hints[cell]
    assert hint.failure_count == 0
    assert hint.cooldown_until == 0
    assert tactic._resource_telemetry.get("unreachable", 0) == 0
    assert _action(turn.plan, UUID(int=0x6000)).direction == Direction.RIGHT


def test_unreachable_resource_fallback_preserves_other_frontier_targets() -> None:
    cell = (10, 0)
    first_worker = str(UUID(int=0x6000))
    second_worker = str(UUID(int=0x6001))
    stable_target = (0, 15)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(5, "history")
    tactic._known_obstacles.update({(9, 0), (11, 0), (10, -1), (10, 1)})
    tactic._explore_targets[second_worker] = stable_target
    turn = _turn(_workers_state([(1, 0), (0, 10)]), tick=20)

    decide(turn)

    assert first_worker in tactic._explore_targets
    assert tactic._explore_targets[second_worker] == stable_target


def test_unreachable_frontier_target_is_cooled_and_reassigned() -> None:
    worker_id = str(UUID(int=0x6000))
    target = (10, 0)
    tactic._explore_targets[worker_id] = target
    tactic._known_obstacles.update({(9, 0), (11, 0), (10, -1), (10, 1)})
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    decide(turn)

    assert tactic._explore_target_cooldown_until[target] == 24
    assert tactic._explore_targets.get(worker_id) != target
    assert _action(turn.plan, UUID(int=0x6000)).type == "MOVE"


def test_frontier_astar_budget_exhaustion_keeps_target_for_retry(monkeypatch) -> None:
    worker_id = str(UUID(int=0x6000))
    target = (10, 0)
    tactic._explore_targets[worker_id] = target
    turn = _turn(_workers_state([(1, 0)]), tick=20)

    monkeypatch.setattr(
        tactic,
        "_astar_step_result",
        lambda *args, **kwargs: (None, True),
    )
    decide(turn)

    assert tactic._explore_targets[worker_id] == target
    assert target not in tactic._explore_target_cooldown_until
    assert _action(turn.plan, UUID(int=0x6000)).direction == Direction.RIGHT


def test_frontier_progress_counter_retargets_after_stall() -> None:
    worker_id = str(UUID(int=0x6000))
    target = (10, 0)
    blocked = frozenset()
    tactic._explore_progress[worker_id] = tactic.ExploreProgress(
        target=target,
        position=(1, 0),
        distance=9,
        frontier_gain=tactic._frontier_gain(target, blocked),
        stalled_ticks=tactic._EXPLORE_STALL_TICKS - 1,
    )

    assert tactic._explore_target_has_stalled(worker_id, (1, 0), target, blocked)


def test_unchanged_visible_hint_does_not_dirty_full_persistent_map() -> None:
    cell = (3, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(10, "visible")
    turn = _turn(_workers_state([(0, 0)], resources=[cell]), tick=11)

    tactic._observe_resources(turn)

    assert tactic._resource_hints[cell].last_confirmed_tick == 10
    assert tactic._persistent_state_dirty is False


def test_resource_telemetry_log_is_monitorable_and_never_contains_api_key(
    tmp_path, monkeypatch, capsys
) -> None:
    import play
    from meta import monitor

    fake_key = "arena-secret-that-must-not-be-logged"
    monkeypatch.setenv(play.API_KEY_ENV, fake_key)
    turn = _turn(_workers_state([(0, 0)], resources=[(3, 0)]))
    decide(turn)

    # Legacy callers may still omit the optional timing/status arguments.
    line = play._log_line(turn, accepted=None)

    assert "eco[a1,av1,ah0,stale0,far0,exp0,ref0,blk0,cool0,unr0,harv0,dep0]" in line
    assert fake_key not in line
    record = monitor._parse_line(line)
    assert record is not None
    assert record["eco"] == {
        "a": 1,
        "av": 1,
        "ah": 0,
        "stale": 0,
        "far": 0,
        "exp": 0,
        "ref": 0,
        "blk": 0,
        "cool": 0,
        "unr": 0,
        "harv": 0,
        "dep": 0,
    }
    log_path = tmp_path / "game.log"
    log_path.write_text(line + "\n", encoding="utf-8")
    kpi = monitor.analyze(log_path)
    assert kpi.resource_assignments == 1
    assert kpi.visible_resource_assignments == 1
    assert monitor.main(["--json", str(log_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["resource_assignments"] == 1
    assert report["event_hist"] == {"-": 1}


def test_dispersed_exploration_targets_cover_multiple_directions() -> None:
    # Simulate a fully explored inner diamond. New targets should lie around
    # different parts of its frontier rather than all inheriting southeast.
    tactic._explored_cells.update(
        (x, y)
        for x in range(-10, 11)
        for y in range(-10, 11)
        if abs(x) + abs(y) <= 10
    )
    workers = [
        (str(UUID(int=0x6000 + index)), (2 + index, 2))
        for index in range(4)
    ]

    targets = tactic._assign_explore_targets(workers, (0, 0), frozenset())

    quadrants = {
        (1 if x >= 0 else -1, 1 if y >= 0 else -1)
        for x, y in targets.values()
    }
    assert len(targets) == 4
    assert len(quadrants) >= 3
    assert min(
        tactic._manhattan(a, b)
        for index, a in enumerate(targets.values())
        for b in list(targets.values())[index + 1 :]
    ) >= 6


def test_frontier_assignment_prefers_reachable_target_over_high_gain(
    monkeypatch,
) -> None:
    """先兑现近处前沿，不能为未知格数量追逐远端目标。"""
    worker_id = str(UUID(int=0x6000))
    candidates = [(1, 0), (10, 0), (0, 1), (0, 10)]
    monkeypatch.setattr(
        tactic,
        "_frontier_candidates",
        lambda *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        tactic,
        "_frontier_gain",
        lambda target, blocked: 100 if target == (10, 0) else 1,
    )

    targets = tactic._assign_explore_targets(
        [(worker_id, (0, 0))],
        (0, 0),
        frozenset(),
        tick=20,
        radius=20,
    )

    target = targets[worker_id]
    assert target != (10, 0)
    assert tactic._manhattan((0, 0), target) == 1


def test_exploration_target_stays_stable_until_frontier_is_lit() -> None:
    tactic._explored_cells.update(
        (x, y)
        for x in range(-6, 7)
        for y in range(-6, 7)
        if abs(x) + abs(y) <= 6
    )
    workers = [(str(UUID(int=0x6000)), (1, 1))]
    first = tactic._assign_explore_targets(workers, (0, 0), frozenset())
    second = tactic._assign_explore_targets(workers, (0, 0), frozenset())

    assert second == first


def test_unknown_cells_behind_obstacle_remain_exploration_frontier() -> None:
    tactic._explored_cells.update({(0, 0), (1, 0), (0, 1), (0, -1), (-1, 0)})
    workers = [(str(UUID(int=0x6000)), (0, 0))]

    targets = tactic._assign_explore_targets(workers, (0, 0), frozenset({(1, 0)}))

    assert targets[str(UUID(int=0x6000))] != (1, 0)
    assert targets[str(UUID(int=0x6000))] not in tactic._explored_cells


def test_frontier_fallback_does_not_reuse_fully_explored_cells() -> None:
    tactic._explored_cells.update(
        (x, y)
        for x in range(-tactic.MAX_SWEEP_RADIUS, tactic.MAX_SWEEP_RADIUS + 1)
        for y in range(-tactic.MAX_SWEEP_RADIUS, tactic.MAX_SWEEP_RADIUS + 1)
        if abs(x) + abs(y) <= tactic.MAX_SWEEP_RADIUS
    )

    candidates = tactic._frontier_candidates((0, 0), frozenset())

    assert candidates == []


def test_frontier_targets_stay_near_current_core_after_migration() -> None:
    # Historical explored terrain near an old Core must not pull idle Workers
    # back across the map after a manual migration.
    tactic._explored_cells.update(
        (100 + x, 100 + y)
        for x in range(-5, 6)
        for y in range(-5, 6)
    )
    worker_id = str(UUID(int=0x6000))

    targets = tactic._assign_explore_targets(
        [(worker_id, (1, 0))], (0, 0), frozenset()
    )

    assert tactic._manhattan(targets[worker_id], (0, 0)) <= tactic.MAX_SWEEP_RADIUS


def test_resource_drought_expands_empty_worker_frontier() -> None:
    # A fully scanned home radius must not trap every Worker in a dead ring
    # while refill nodes are being placed in a neighbouring chunk.
    tactic._explored_cells.update(
        (x, y)
        for x in range(-tactic.MAX_SWEEP_RADIUS, tactic.MAX_SWEEP_RADIUS + 1)
        for y in range(-tactic.MAX_SWEEP_RADIUS, tactic.MAX_SWEEP_RADIUS + 1)
        if abs(x) + abs(y) <= tactic.MAX_SWEEP_RADIUS
    )
    tactic._resource_absence_streak = tactic.DROUGHT_EXPAND_EVERY * 2 + 1

    radius = tactic._exploration_radius()
    assert radius > tactic.MAX_SWEEP_RADIUS
    candidates = tactic._frontier_candidates(
        (0, 0), frozenset(), tick=20, radius=radius
    )
    assert any(tactic._manhattan(cell, (0, 0)) > tactic.MAX_SWEEP_RADIUS for cell in candidates)


def test_visible_resource_resets_drought_expansion() -> None:
    tactic._resource_absence_streak = tactic.MAX_DROUGHT_SWEEP_RADIUS
    turn = _turn(_workers_state([(0, 1)], resources=[(3, 0)]), tick=20)

    tactic._observe_resources(turn)

    assert tactic._resource_absence_streak == 0
    assert tactic._exploration_radius() == tactic.MAX_SWEEP_RADIUS


def test_historical_resource_hint_does_not_mask_drought_expansion() -> None:
    # A remembered node is not current visibility.  When it remains unseen,
    # the empty frontier must still expand instead of staying inside the
    # 40-cell economic radius forever.
    cell = (20, 0)
    tactic._known_resources.add(cell)
    tactic._resource_hints[cell] = tactic.ResourceHint(1, "history")
    turn = _turn(_workers_state([(0, 1)]), tick=20)

    tactic._observe_resources(turn)

    assert tactic._resource_absence_streak == 1
    tactic._resource_absence_streak = tactic.DROUGHT_EXPAND_EVERY + 1
    assert tactic._exploration_radius() > tactic.MAX_SWEEP_RADIUS


def test_astar_rejects_blocked_goal_unless_explicitly_allowed() -> None:
    start = (0, 0)
    goal = (1, 0)
    blocked = frozenset({goal})

    assert tactic._astar_step(start, goal, frozenset(), blocked) is None
    assert (
        tactic._astar_step(
            start, goal, frozenset(), blocked, allow_blocked_goal=True
        )
        == Direction.RIGHT
    )


def test_astar_telemetry_counts_calls_expansions_and_paths() -> None:
    start = (0, 0)
    goal = (2, 0)

    assert tactic._astar_step(start, goal, frozenset(), frozenset()) == Direction.RIGHT
    assert tactic._resource_telemetry["astar_calls"] == 1
    assert tactic._resource_telemetry["astar_expansions"] > 0
    assert tactic._resource_telemetry["astar_paths"] == 1
    assert tactic._resource_telemetry.get("astar_budget_hits", 0) == 0

    step, exhausted = tactic._astar_step_result(
        start, (20, 0), frozenset(), frozenset(), max_expansions=1
    )
    assert step is None
    assert exhausted is True
    assert tactic._resource_telemetry["astar_calls"] == 2
    assert tactic._resource_telemetry["astar_budget_hits"] == 1


def test_ranger_chase_avoids_full_friendly_cell(monkeypatch) -> None:
    ranger2_id = UUID("00000000-0000-4000-8000-000000000006")
    state = _state(
        resources=0,
        population=5,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 1],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ranger2_id),
                "controlled": True,
                "position": [2, 1],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [3, 1],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER2_ID),
                "controlled": True,
                "position": [3, 1],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [8, 1],
                "hp": 2,
                "unit_type": "VANGUARD",
            },
        ],
    )
    monkeypatch.setattr(tactic, "_chase_target", lambda *args, **kwargs: (8, 1))
    turn = _turn(state)

    decide(turn)

    action = _action(turn.plan, ranger2_id)
    assert action is not None
    assert action.type == "MOVE"
    assert action.direction in {Direction.UP, Direction.DOWN}


def test_complete_worker_plan_is_independent_of_object_order() -> None:
    state = _workers_state([(2, 2), (3, 2), (4, 2), (5, 2)])
    first = _turn(state)
    decide(first)
    expected = first.plan.unit_actions

    tactic._explore_state.clear()
    tactic._explore_targets.clear()
    tactic._explored_cells.clear()
    tactic._pos_history.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()
    tactic._stuck_ticks.clear()
    reordered = state.model_copy(
        update={"objects": tuple([state.objects[0], *reversed(state.objects[1:])])}
    )
    second = _turn(reordered)
    decide(second)

    assert second.plan.unit_actions == expected


# ---------------------------------------------------------------------------
# Economy: harvest and deposit
# ---------------------------------------------------------------------------


def test_worker_on_resource_harvests() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[2, 0], [5, 5]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, WORKER_ID).type == "HARVEST"


def test_nearest_worker_claims_resource_independent_of_unit_order() -> None:
    """A far, earlier Worker must not steal a node from a nearby Worker."""
    resource = (6, 0)
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [-6, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER2_ID),
                "controlled": True,
                "position": [5, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [list(resource)]},
        ],
    )

    turn = _turn(state)
    decide(turn)

    assert tactic._worker_resource_assignments(turn) == {str(WORKER2_ID): resource}
    assert _action(turn.plan, WORKER2_ID).direction == Direction.RIGHT


def test_nearest_worker_takes_over_stale_far_lock() -> None:
    resource = (6, 0)
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [-6, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER2_ID),
                "controlled": True,
                "position": [5, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [list(resource)]},
        ],
    )
    tactic._explore_state[str(WORKER_ID)] = [0, 0, *resource]
    turn = _turn(state)
    decide(turn)
    assert tactic._worker_resource_assignments(turn) == {str(WORKER2_ID): resource}
    assert _action(turn.plan, WORKER2_ID).direction == Direction.RIGHT


def test_worker_with_cargo_home_deposits() -> None:
    state = _state(
        resources=5,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, WORKER_ID).type == "DEPOSIT"


def test_boxed_history_cannot_override_ready_deposit() -> None:
    tactic._pos_history[str(WORKER_ID)] = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    ]
    state = _state(
        resources=5,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ],
    )
    turn = _turn(state)

    decide(turn)

    assert _action(turn.plan, WORKER_ID).type == "DEPOSIT"


def test_worker_with_cargo_away_from_home_moves_home() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [3, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action.type == "MOVE"
    assert action.direction == Direction.LEFT  # toward Core at (0,0)


def test_worker_moves_toward_nearest_visible_resource() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[3, 0], [0, 5]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action.type == "MOVE"
    # Nearest is (3,0) at distance 3; step right.
    assert action.direction == Direction.RIGHT


# ---------------------------------------------------------------------------
# Resource disappearance / fog invalidation / retargeting
# ---------------------------------------------------------------------------


def test_worker_retargets_when_old_resource_disappears() -> None:
    # Worker was heading to (5,0) last Tick but that node was harvested; only
    # (0,4) is visible now. The tactic must choose from current cells only.
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[0, 4]]},
        ],
        events=[
            {
                "event_id": "00000000-0000-4000-8000-0000000000aa",
                "tick": 9,
                "event_type": "HARVEST_FAILED",
                "actor_id": str(WORKER_ID),
                "position": [5, 0],
                "reason_code": "RESOURCE_DEPLETED",
            }
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action.type == "MOVE"
    # From (2,0) toward (0,4): both LEFT (dx-1) and DOWN (dy+1) reduce Manhattan
    # distance. The old greedy _step_toward picked DOWN by axis-preference; A* (9th
    # review rank 1) picks whichever shortest-path tie-breaker wins. Either is valid
    # and reduces the gap.
    assert action.direction in (Direction.LEFT, Direction.DOWN)


def test_dropped_cargo_pile_is_treated_as_resource_cell() -> None:
    # A cargo pile left by a dead Worker shows up in resource_cells and can be
    # recovered by an empty Worker standing on it.
    state = _state(
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 1],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[1, 1]]},
        ],
        events=[
            {
                "event_id": "00000000-0000-4000-8000-0000000000bb",
                "tick": 9,
                "event_type": "WORKER_CARGO_DROPPED",
                "actor_id": str(WORKER2_ID),
                "position": [1, 1],
                "values": {"amount": 2},
            }
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, WORKER_ID).type == "HARVEST"


def test_out_of_sight_dropped_cargo_is_persisted_as_resource_hint() -> None:
    state = _state(
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
        events=[
            {
                "event_id": "00000000-0000-4000-8000-0000000000bc",
                "tick": 9,
                "event_type": "WORKER_CARGO_DROPPED",
                "actor_id": str(WORKER2_ID),
                "position": [20, 20],
                "values": {"amount": 2},
            }
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert (20, 20) in tactic._known_resources
    assert tactic._resource_hints[(20, 20)].source == "dropped"


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------


def test_ranger_shoots_visible_legal_target() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, RANGER_ID)
    assert action.type == "SHOOT"
    assert action.target_id == ENEMY_UNIT_ID
    assert tuple(action.expected_cell) == (0, 2)


def test_low_hp_ranger_shoots_before_heal_when_target_is_legal() -> None:
    state = _state(
        resources=1,
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 1,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, RANGER_ID)
    assert action.type == "SHOOT"
    assert _core_action(turn.plan) is None


def test_distant_enemy_core_raid_uses_dynamic_reserve_and_eta_budget() -> None:
    second_ranger_id = UUID("00000000-0000-4000-8000-000000000006")
    state = _state(
        resources=15,
        population=3,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(second_ranger_id),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "CORE",
                "id": str(ENEMY_CORE_ID),
                "controlled": False,
                "owner_username": "rival",
                "position": [0, 34],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, second_ranger_id)
    assert action.type == "MOVE"
    assert tactic._chase_budget[str(second_ranger_id)] >= 31


def test_ranger_prioritizes_near_core_raider_over_distant_enemy_core() -> None:
    state = _state(
        resources=0,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "CORE",
                "id": str(ENEMY_CORE_ID),
                "controlled": False,
                "owner_username": "rival",
                "position": [0, 3],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, RANGER_ID)
    assert action.type == "SHOOT"
    assert action.target_id == ENEMY_UNIT_ID


def test_ranger_shoots_visible_diagonal_target() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [2, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, RANGER_ID)
    assert action.type == "SHOOT"
    assert tuple(action.expected_cell) == (2, 2)


def test_ranger_does_not_shoot_diagonal_through_obstacle() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {"kind": "OBSTACLE", "positions": [[1, 1]]},
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [2, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, RANGER_ID).type != "SHOOT"


def test_ranger_diagonal_is_not_blocked_by_adjacent_obstacle() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {"kind": "OBSTACLE", "positions": [[1, 0]]},
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [2, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, RANGER_ID).type == "SHOOT"


def test_ranger_does_not_shoot_obstructed_target() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {"kind": "OBSTACLE", "positions": [[0, 1]]},
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 2],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    # Obstacle at (0,1) blocks the line to (0,2); no legal shot. The Ranger
    # does not shoot and instead explores (a MOVE), never WAITing on a bare
    # view since its vision is the best scout.
    action = _action(turn.plan, RANGER_ID)
    assert action.type != "SHOOT"


def test_ranger_out_of_range_does_not_shoot() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 5],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    # Range 5 > 3; the Ranger cannot shoot and explores instead of WAITing.
    action = _action(turn.plan, RANGER_ID)
    assert action.type != "SHOOT"


def test_vanguard_sweeps_adjacent_enemy_cell() -> None:
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(VANGUARD_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 4,
                "unit_type": "VANGUARD",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, VANGUARD_ID)
    assert action.type == "SWEEP"
    assert action.direction == Direction.RIGHT


def test_vanguard_steps_off_core_cell_to_unblock_deposit() -> None:
    # A Vanguard that spawned on the Core cell (Core + Vanguard = 2/2 capacity)
    # blocks laden Workers from entering the Core cell to deposit. With nothing
    # to sweep, the Vanguard must step off the Core cell so deposits can resume.
    state = _state(
        resources=0,
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(VANGUARD_ID),
                "controlled": True,
                "position": [0, 0],  # on the Core cell
                "hp": 4,
                "unit_type": "VANGUARD",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 1],  # laden Worker adjacent, wants to deposit
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    vaction = _action(turn.plan, VANGUARD_ID)
    # The Vanguard must move (not WAIT), stepping off the Core cell.
    assert vaction is not None
    assert vaction.type == "MOVE"
    ddx, ddy = vaction.direction.delta
    # Its destination must not be the laden Worker's cell (would re-block it).
    assert (ddx, ddy) != (0, 1)


def test_vanguard_guard_targets_keep_delivery_lanes_open() -> None:
    second_vanguard = UUID(int=0x7001)
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(VANGUARD_ID),
                "controlled": True,
                "position": [0, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
            {
                "kind": "UNIT",
                "id": str(second_vanguard),
                "controlled": True,
                "position": [0, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
        ],
    )
    turn = _turn(state)

    targets = tactic._vanguard_guard_targets(turn, (0, 0))

    assert len(set(targets.values())) == 2
    assert sum(tactic._manhattan(target, (0, 0)) == 1 for target in targets.values()) == 1
    assert sum(tactic._manhattan(target, (0, 0)) == 2 for target in targets.values()) == 1


def test_stacked_vanguards_split_instead_of_blocking_one_entrance() -> None:
    second_vanguard = UUID(int=0x7001)
    state = _state(
        population=2,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(VANGUARD_ID),
                "controlled": True,
                "position": [0, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
            {
                "kind": "UNIT",
                "id": str(second_vanguard),
                "controlled": True,
                "position": [0, 1],
                "hp": 4,
                "unit_type": "VANGUARD",
            },
        ],
    )
    turn = _turn(state)
    decide(turn)

    actions = [
        _action(turn.plan, VANGUARD_ID),
        _action(turn.plan, second_vanguard),
    ]
    moves = [action for action in actions if action is not None and action.type == "MOVE"]
    assert len(moves) == 1
    destination = (
        moves[0].direction.delta[0],
        1 + moves[0].direction.delta[1],
    )
    assert tactic._manhattan(destination, (0, 0)) == 2


def test_friendly_full_invalidates_stable_frontier_target() -> None:
    worker_id = str(UUID(int=0x6000))
    tactic._explored_cells.update({(0, 0), (1, 0), (0, 1), (0, -1), (-1, 0)})
    tactic._explore_targets[worker_id] = (2, 0)

    targets = tactic._assign_explore_targets(
        [(worker_id, (0, 0))], (0, 0), frozenset({(2, 0)})
    )

    assert targets[worker_id] != (2, 0)


# ---------------------------------------------------------------------------
# Obstacle-aware movement
# ---------------------------------------------------------------------------


def test_worker_avoids_obstacle_when_moving() -> None:
    # Worker at (0,0), resource at (2,0), obstacle at (1,0). The direct step
    # RIGHT is blocked, so the tactic should pick another reducing step.
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "OBSTACLE", "positions": [[1, 0]]},
            {"kind": "RESOURCE", "positions": [[2, 0]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action.type == "MOVE"
    assert action.direction != Direction.RIGHT


def test_out_of_radius_return_does_not_backtrack_into_recent_cell() -> None:
    worker_id = str(WORKER_ID)
    tactic._pos_history[worker_id] = [(0, 1)]
    blocked = frozenset({(1, 0), (0, -1)})

    step = tactic._explore_step(
        0,
        worker_id,
        (0, 0),
        (50, 0),
        blocked,
    )

    assert step is not None
    assert step != Direction.DOWN


def test_laden_worker_return_does_not_backtrack_into_recent_cell() -> None:
    state = _state(
        core_pos=(50, 0),
        beacon_pos=(100, 100),
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [50, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
            {"kind": "OBSTACLE", "positions": [[1, 0], [0, -1]]},
        ],
    )
    tactic._pos_history[str(WORKER_ID)] = [(0, 1)]
    turn = _turn(state)

    decide(turn)

    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "MOVE"
    assert action.direction != Direction.DOWN


# ---------------------------------------------------------------------------
# Core actions: repair and spawn
# ---------------------------------------------------------------------------


def test_core_heals_hp_before_repairing_shield_when_threatened() -> None:
    state = _state(
        core_hp=4,
        core_shield=2,
        resources=5,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 4,
                "shield": 2,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [1, 1],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _core_action(turn.plan).type == "HEAL"


def test_core_does_not_repair_when_shield_full() -> None:
    state = _state(
        core_hp=5,
        core_shield=5,
        resources=5,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [1, 1],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    # Shield full -> no repair. Should consider spawning instead.
    core_act = _core_action(turn.plan)
    assert core_act is None or core_act.type != "REPAIR_SHIELD"


def test_unit_heal_reservation_prevents_core_overspend() -> None:
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(RANGER_ID),
            "controlled": True,
            "position": [0, 0],
            "hp": 1,
            "unit_type": "RANGER",
            "cargo": None,
        },
    ]
    for index, position in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x6000 + index)),
                "controlled": True,
                "position": list(position),
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            }
        )
    state = _state(resources=10, population=5, objects=objects)
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, RANGER_ID).type == "HEAL"
    # The Ranger consumes the one resource needed to reach full HP before the
    # Core action. The remaining 9 cannot fund the 10-cost Vanguard spawn.
    assert _core_action(turn.plan) is None


def test_unit_heals_are_reserved_in_uuid_order() -> None:
    state = _state(
        resources=3,
        population=3,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(VANGUARD_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 1,
                "unit_type": "VANGUARD",
                "cargo": None,
            },
            {
                "kind": "UNIT",
                "id": str(RANGER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 1,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, VANGUARD_ID).type == "HEAL"
    assert _action(turn.plan, RANGER_ID).type != "HEAL"


def test_core_spawns_worker_below_target() -> None:
    # Lone Worker, plenty of resources, visible resource to work, no threat:
    # spawn toward target.
    state = _state(
        resources=10,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[2, 0]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _core_action(turn.plan).type == "SPAWN"
    assert _core_action(turn.plan).unit_type.value == "WORKER"


def test_core_does_not_spawn_into_full_cell() -> None:
    # Core cell already holds Core + one Unit; with a visible resource the
    # tactic would want to spawn, but the full cell (Core + Worker) means a
    # spawn would fail with CELL_UNIT_LIMIT, so it should not queue it.
    state = _state(
        resources=10,
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[3, 0]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _core_action(turn.plan) is None


def test_core_spawns_when_full_and_laden_worker_steps_off() -> None:
    # Deadlock regression: a laden Worker parked on the Core cell at full
    # capacity cannot deposit (CORE_RESOURCE_FULL) and its occupancy blocks
    # every spawn (CELL_UNIT_LIMIT) — a permanent freeze. The tactic must break
    # it by stepping the Worker OFF the Core cell AND spawning in the same Tick:
    # Unit movement resolves before Core spawn, so the spawn lands on a clear
    # cell and spends 5 resources, opening room for the Worker's deposit next
    # Tick. Empty-Worker occupancy must NOT trigger this (spawn would still
    # fail CELL_UNIT_LIMIT), which test_core_does_not_spawn_into_full_cell locks.
    state = _state(
        resources=10,  # == capacity max(10, pop*5) with pop=1 -> full
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [0, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    core_action = _core_action(turn.plan)
    assert core_action is not None
    assert core_action.unit_type == UnitType.WORKER
    # The laden Worker must MOVE off the Core cell, not deposit (it would fail).
    assert _action(turn.plan, WORKER_ID) is not None
    assert getattr(_action(turn.plan, WORKER_ID), "direction", None) is not None


def test_migrating_core_does_not_act() -> None:
    state = _state(
        resources=10,
        core_state="MOVING",
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "MOVING",
                "move_direction": "RIGHT",
                "move_progress": 2,
                "move_required_ticks": 4,
                "destination": [1, 0],
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _core_action(turn.plan) is None


# ---------------------------------------------------------------------------
# No stale controller reuse: decide reads turn only
# ---------------------------------------------------------------------------


def test_decide_is_pure_over_fresh_turns() -> None:
    """Calling decide on two fresh turns with the same state yields the same plan."""
    s1 = _state(resources=10)
    s2 = _state(resources=10)
    t1 = _turn(s1, tick=10)
    t2 = _turn(s2, tick=11)
    decide(t1)
    decide(t2)
    assert t1.plan.unit_actions == t2.plan.unit_actions
    assert t1.plan.core_action == t2.plan.core_action


# ---------------------------------------------------------------------------
# Beacon awareness
# ---------------------------------------------------------------------------


def test_holds_beacon_does_not_break_decisions() -> None:
    # Beacon is carried by a friendly Worker (status CARRIED + carrier_id).
    # The tactic should still function and not crash reading beacon fields.
    state = _state(
        resources=5,
        population=1,
        beacon_pos=(1, 0),
        beacon_status="CARRIED",
        beacon_carrier_id=WORKER_ID,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[1, 0]]},
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, WORKER_ID).type == "HARVEST"


def test_enemy_beacon_does_not_raise_our_shield_cap() -> None:
    state = _state(
        resources=12,
        population=2,
        beacon_pos=(1, 0),
        beacon_status="CARRIED",
        beacon_carrier_id=ENEMY_UNIT_ID,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is None or act.type != "REPAIR_SHIELD"


def test_worker_picks_up_ground_beacon_on_same_cell() -> None:
    state = _state(
        beacon_pos=(1, 0),
        beacon_status="GROUND",
    )
    turn = _turn(state)
    decide(turn)
    assert _action(turn.plan, WORKER_ID).type == "PICKUP_BEACON"


def test_core_repairs_before_picking_up_ground_beacon_under_threat() -> None:
    state = _state(
        resources=5,
        population=1,
        beacon_pos=(0, 0),
        beacon_status="GROUND",
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 4,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 3],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            },
        ],
    )
    turn = _turn(state)
    decide(turn)
    assert _core_action(turn.plan).type == "REPAIR_SHIELD"


def test_revisited_empty_enemy_core_position_is_forgotten() -> None:
    tactic._known_enemy_cores.add((2, 0))
    turn = _turn(_state())
    tactic._observe_enemies(turn)
    assert (2, 0) not in tactic._known_enemy_cores


def test_visible_enemy_core_is_kept_in_known_set() -> None:
    # A CORE visible this Tick stays remembered (the ghost pass must not drop
    # a live Core just because it also iterates the memory set).
    state = _state(
        population=1,
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {
                "kind": "CORE",
                "id": str(ENEMY_CORE_ID),
                "controlled": False,
                "owner_username": "rival",
                "position": [2, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
        ],
    )
    turn = _turn(state)
    tactic._observe_enemies(turn)
    assert (2, 0) in tactic._known_enemy_cores
    assert tactic._last_enemy_pos.get(str(ENEMY_CORE_ID)) is not None


def test_obstacle_shadow_is_not_recorded_as_explored() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {"kind": "OBSTACLE", "positions": [[1, 0]]},
        ],
        population=0,
    )
    tactic._observe_terrain(_turn(state))
    assert (1, 0) in tactic._explored_cells
    assert (2, 0) not in tactic._explored_cells


# ---------------------------------------------------------------------------
# Standing army: combat Units are maintained even in peacetime
# ---------------------------------------------------------------------------


def _state_with_workers(
    *,
    n_workers: int,
    resources: int,
    n_vanguards: int = 0,
    n_rangers: int = 0,
    threat: bool = False,
) -> PlayerState:
    """Build a state with a Core, ``n_workers`` Workers on distinct cells, and
    optional existing combat Units / a visible enemy threat."""
    objects: list[dict[str, Any]] = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        }
    ]
    # Place Workers on cells around the Core, none on the Core cell (so the
    # cell has room for a spawn).
    worker_offsets = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
        *(
            (10 + i % 10, 10 + i // 10)
            for i in range(20)
        ),
    ]
    for i in range(n_workers):
        ox, oy = worker_offsets[i]
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x2000 + i)),
                "controlled": True,
                "position": [ox, oy],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            }
        )
    for i in range(n_vanguards):
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x3000 + i)),
                "controlled": True,
                "position": [3 + i, 0],
                "hp": 4,
                "unit_type": "VANGUARD",
                "cargo": None,
            }
        )
    for i in range(n_rangers):
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x4000 + i)),
                "controlled": True,
                "position": [3 + i, 3],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            }
        )
    if threat:
        objects.append(
            {
                "kind": "UNIT",
                "id": str(ENEMY_UNIT_ID),
                "controlled": False,
                "position": [0, 3],
                "hp": 2,
                "unit_type": "RANGER",
                "cargo": None,
            }
        )
    return _state(
        resources=resources,
        population=n_workers + n_vanguards + n_rangers,
        objects=objects,
    )


def test_standing_army_spawns_vanguard_above_economy_floor() -> None:
    # 4 Workers (the economy floor), 10 resources, no threat: the tactic must
    # build the standing Vanguard reserve BEFORE growing the Worker fleet, so a
    # surprise raid always meets return fire.
    state = _state_with_workers(n_workers=4, resources=10)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "VANGUARD"


def test_peacetime_worker_bridge_preserves_ranger_reserve() -> None:
    # 已有 Vanguard 防线时，W12 仍可走有限的 Worker 桥接；支付 5 点后还会
    # 保留 7 点库存，Ranger 的和平期 5 点储备闸门仍不会被绕过。
    state = _state_with_workers(n_workers=12, resources=12, n_vanguards=2)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_standing_army_spawns_ranger_after_peacetime_reserve() -> None:
    state = _state_with_workers(n_workers=12, resources=17, n_vanguards=2)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "RANGER"


def test_visible_threat_overrides_peacetime_ranger_reserve() -> None:
    # 可见敌人时即使未达到和平库存下限，也必须立即生产战斗单位。
    state = _state_with_workers(
        n_workers=4,
        resources=12,
        n_vanguards=2,
        threat=True,
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "RANGER"


def test_standing_army_banks_resources_when_short_not_worker() -> None:
    # 4 Workers, reserve short (0 Vanguard), only 7 resources: cannot afford
    # the 10-cost Vanguard. Must NOT spend 5 on another Worker — bank toward
    # the Vanguard so the reserve completes on a later Tick.
    state = _state_with_workers(n_workers=4, resources=7)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is None, "should bank resources for the combat Unit, not spawn"


def test_partial_army_allows_one_economy_bridge_worker() -> None:
    # Once a Vanguard already protects the Core, a temporary Ranger shortfall
    # must not freeze the economy at the four-Worker floor.  At 8 resources a
    # Worker can be paid while preserving the three-resource bank reserve.
    state = _state_with_workers(
        n_workers=4, resources=8, n_vanguards=1, n_rangers=0
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_peacetime_bridge_reaches_second_extra_worker() -> None:
    # A mature W5/V2 economy should not freeze while it banks for its first
    # Ranger. The bounded bridge allows two further scouts, then the Ranger
    # remains the next combat priority once resources reach its price.
    state = _state_with_workers(
        n_workers=5, resources=8, n_vanguards=2, n_rangers=0
    )
    turn = _turn(state)

    decide(turn)

    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_peacetime_bridge_reaches_third_extra_worker() -> None:
    # With no visible threat, keep additional scouts active while the missing
    # Ranger is being funded.  W7 -> W8 is still below the first dynamic-price
    # boundary and raises discovery capacity by another worker.
    state = _state_with_workers(
        n_workers=7, resources=8, n_vanguards=2, n_rangers=0
    )
    turn = _turn(state)

    decide(turn)

    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_peacetime_bridge_reaches_fifteenth_worker() -> None:
    # A quiet W14/V2 economy may continue the bounded discovery bridge to W15.
    # Population remains 17 before the spawn, well below the first dynamic
    # price tier, while visible enemy signals still take the combat-first path.
    state = _state_with_workers(
        n_workers=14, resources=8, n_vanguards=2, n_rangers=0
    )
    turn = _turn(state)

    decide(turn)

    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_peacetime_bridge_stops_after_fifteenth_worker() -> None:
    # Once W15 is reached, bank for the missing Ranger instead of growing
    # indefinitely; this preserves the bounded bridge and standing army floor.
    state = _state_with_workers(
        n_workers=15, resources=8, n_vanguards=2, n_rangers=0
    )
    turn = _turn(state)

    decide(turn)

    assert _core_action(turn.plan) is None


def test_full_peacetime_army_does_not_grow_workers_past_bridge_cap() -> None:
    # Combat reserve complete should not reopen the larger population budget:
    # the discovery bridge remains capped at W15 until a later policy changes it.
    state = _state_with_workers(
        n_workers=15, resources=8, n_vanguards=1, n_rangers=1
    )
    turn = _turn(state)

    decide(turn)

    assert _core_action(turn.plan) is None


def test_visible_enemy_core_disables_worker_bridge() -> None:
    # A visible Core is an attack signal even when it is outside the local
    # threat radius.  Do not spend the bank on a Worker while the strike force
    # is still short.
    state = _state_with_workers(
        n_workers=6, resources=8, n_vanguards=2, n_rangers=0
    )
    objects = [obj.model_dump(mode="json") for obj in state.objects]
    objects.append(
        {
            "kind": "CORE",
            "id": str(ENEMY_CORE_ID),
            "controlled": False,
            "owner_username": "rival",
            "position": [0, 20],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        }
    )
    state = _state(
        resources=8,
        population=8,
        objects=objects,
    )
    turn = _turn(state)

    decide(turn)

    assert _core_action(turn.plan) is None


def test_partial_army_bridge_is_disabled_under_threat() -> None:
    # A nearby enemy keeps the combat-first gate: the bridge Worker must not
    # consume the bank that should fund the next defender.
    state = _state_with_workers(
        n_workers=4, resources=8, n_vanguards=1, n_rangers=0, threat=True
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is None


def test_standing_army_grows_workers_when_reserve_full() -> None:
    # Standing reserve satisfied (1 Vanguard + 1 Ranger), resources above the
    # spawn+reserve threshold: now grow the Worker fleet toward TARGET_WORKERS.
    # The bank reserve (WORKER_SPAWN_RESERVE) means a spawn must leave the Core
    # non-empty, so 5 alone is not enough — use 8 (5 cost + 3 reserve).
    state = _state_with_workers(n_workers=4, resources=8, n_vanguards=1, n_rangers=1)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_remembered_enemy_core_does_not_freeze_economy() -> None:
    tactic._known_enemy_cores.add((30, 30))
    state = _state_with_workers(
        n_workers=4, resources=8, n_vanguards=1, n_rangers=1
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_standing_army_banks_worker_spawn_when_below_reserve() -> None:
    # Reserve satisfied, only 5 resources: a Worker costs 5 but the reserve
    # requires 3 left over, so 5 is NOT enough — bank instead of draining to 0.
    state = _state_with_workers(n_workers=4, resources=5, n_vanguards=1, n_rangers=1)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is None, "should bank toward the reserve, not spawn to zero"


def test_standing_army_expands_defense_line_under_threat() -> None:
    # Under visible threat with the peacetime reserve already in place, grow
    # the defensive line toward DEFENSE_VANGUARDS (2) before more Workers.
    state = _state_with_workers(
        n_workers=4, resources=10, n_vanguards=1, n_rangers=1, threat=True
    )
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "VANGUARD"


def test_visible_enemy_triggers_attack_spawn_with_one_worker() -> None:
    state = _state_with_workers(n_workers=1, resources=10, threat=True)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "VANGUARD"


def test_population_19_can_spawn_the_twentieth_unit() -> None:
    # v0.14 keeps the base Vanguard price for the twentieth Unit. The old
    # population >= 19 guard incorrectly suppressed this defensive spawn.
    state = _state_with_workers(
        n_workers=18, n_vanguards=1, resources=10, threat=True
    )
    assert state.population == 19
    turn = _turn(state)

    decide(turn)

    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "VANGUARD"


@pytest.mark.parametrize(
    ("resources", "spawns"),
    [(12, False), (13, True)],
)
def test_population_20_uses_first_dynamic_price_for_combat_spawn(
    resources: int, spawns: bool
) -> None:
    # At population 20 the next Vanguard costs round(10 * 1.3) = 13. A
    # population gate must not hide the dynamic-price decision or block a
    # needed replacement when the bank can afford it.
    state = _state_with_workers(
        n_workers=18,
        n_vanguards=1,
        n_rangers=1,
        resources=resources,
        threat=True,
    )
    assert state.population == 20
    turn = _turn(state)

    decide(turn)

    act = _core_action(turn.plan)
    assert (act is not None) is spawns
    if spawns:
        assert act.type == "SPAWN"
        assert act.unit_type.value == "VANGUARD"


def test_cold_start_workers_before_army() -> None:
    # Below the economy floor (2 Workers), plenty of resources, no threat:
    # build the economy first — spawn a Worker, not a combat Unit.
    state = _state_with_workers(n_workers=2, resources=10)
    turn = _turn(state)
    decide(turn)
    act = _core_action(turn.plan)
    assert act is not None
    assert act.type == "SPAWN"
    assert act.unit_type.value == "WORKER"


def test_standing_army_scales_with_worker_fleet() -> None:
    # User requirement: the standing army grows with the Worker economy, so a
    # raid meets more return fire the larger the fleet. 8 Workers -> V1 R1;
    # 16 Workers -> V2 R2 (population 20 is still allowed; the next production
    # is the first dynamic-price unit).
    #
    # 10th review (rank 1): a HARD floor of V>=1,R>=1 prevents the ratchet
    # where a raid that kills combat units never triggers a rebuild because
    # the worker count is unchanged and the target matches the current count.
    # When the budget would overflow, the hard V/R floor remains while Worker
    # growth is held at the population target. Combat replacements are still
    # allowed above that target when the standing reserve is short.
    assert tactic._standing_army_targets(4) == (1, 1)
    assert tactic._standing_army_targets(8) == (1, 1)
    assert tactic._standing_army_targets(12) == (1, 1)
    assert tactic._standing_army_targets(16) == (2, 2)
    assert tactic._standing_army_targets(17) == (2, 1)
    # W=18: hard floor forces (1,1) after fitting the population target.
    assert tactic._standing_army_targets(18) == (1, 1)
    # W=19: hard floor keeps V=1,R=1 even though the floor itself makes the
    # combined count one above the soft target.
    assert tactic._standing_army_targets(19) == (1, 1)
    for w in range(4, tactic.FREE_UPKEEP_CAP):
        v, r = tactic._standing_army_targets(w)
        # Floor guarantee: no Worker count should ever suggest zero combat
        # units. 10th review overturned the old "W=19 → V=0,R=0" edge case.
        assert v >= 1 and r >= 1, (
            f"army floor broken at {w}: V{v}R{r}"
        )


def test_pop_over_budget_does_not_destroy_units_or_capacity() -> None:
    # A manual expansion above the soft target must not destroy units or shrink
    # capacity merely to avoid a dynamic production price.
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        }
    ]
    for i in range(19):
        objects.append(
            {
                "kind": "UNIT",
                "id": str(UUID(int=0x5000 + i)),
                "controlled": True,
                "position": [10 + (i % 8), 2 + (i // 8)],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            }
        )
    objects.append(
        {
            "kind": "UNIT",
            "id": str(VANGUARD_ID),
            "controlled": True,
            "position": [3, 0],
            "hp": 4,
            "unit_type": "VANGUARD",
        }
    )
    objects.append(
        {
            "kind": "UNIT",
            "id": str(RANGER_ID),
            "controlled": True,
            "position": [3, 3],
            "hp": 2,
            "unit_type": "RANGER",
        }
    )
    state = _state(resources=105, population=21, objects=objects)
    turn = _turn(state)
    decide(turn)
    culled = [
        uid
        for uid, a in turn.plan.unit_actions.items()
        if getattr(a, "type", "") == "SELF_DESTRUCT"
    ]
    assert culled == []


def test_resource_memory_pool_remembers_and_confirms() -> None:
    # A resource seen once enters the memory pool and survives after leaving
    # vision; when a friendly vision source sees the cell empty (not in
    # turn.resource_cells), it is confirmed bare and removed. Without this,
    # Workers re-scan already-depleted cells (the 5th-review discovery
    # bottleneck). Learned from the reference agent's known_resources.
    core_worker = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [1, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 0,
        },
    ]
    # Tick 1: resource at (4,0), Worker at (1,0) sees it (Manhattan 3 = vision).
    s1 = _state(
        resources=5,
        population=1,
        objects=core_worker + [{"kind": "RESOURCE", "positions": [[4, 0]]}],
    )
    t1 = _turn(s1)
    tactic._observe_resources(t1)
    assert (4, 0) in tactic._known_resources

    # Tick 2: resource gone; Worker still sees the cell (dist 3) and it is NOT
    # a resource -> confirmed bare, removed from the pool.
    s2 = _state(resources=5, population=1, objects=core_worker)
    t2 = _turn(s2)
    tactic._observe_resources(t2)
    assert (4, 0) not in tactic._known_resources

    # A resource too far from any vision source is KEPT (can't confirm empty).
    s3 = _state(
        resources=5,
        population=1,
        objects=core_worker + [{"kind": "RESOURCE", "positions": [[8, 0]]}],
    )
    t3 = _turn(s3)
    tactic._observe_resources(t3)
    assert (8, 0) in tactic._known_resources


def test_resource_hidden_behind_persistent_obstacle_is_not_forgotten() -> None:
    # A Worker can lose line of sight while stepping beside a wall that was
    # observed on an earlier Tick.  The persistent obstacle must still block
    # the empty-cell confirmation, otherwise the dispatcher drops the resource
    # and sends the Worker back and forth between historical targets.
    cell = (383, 126)
    obstacle = (383, 125)
    core_worker = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [347, 124],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [381, 125],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 0,
        },
    ]
    visible = _state(
        core_pos=(347, 124),
        population=1,
        objects=core_worker + [{"kind": "RESOURCE", "positions": [list(cell)]}],
    )
    tactic._observe_resources(_turn(visible, tick=1))
    assert cell in tactic._known_resources

    # The next authoritative state omits the resource and the currently
    # visible obstacle, but the worker's line to the cell crosses the persisted
    # obstacle at (383,125).
    tactic._known_obstacles.add(obstacle)
    hidden = _state(
        core_pos=(347, 124),
        population=1,
        objects=[
            core_worker[0],
            {**core_worker[1], "position": [382, 125]},
        ],
    )
    tactic._observe_resources(_turn(hidden, tick=2))
    assert cell in tactic._known_resources
    assert tactic._worker_resource_assignments(_turn(hidden, tick=2)) == {
        str(WORKER_ID): cell
    }


def test_current_resource_state_overrides_previous_harvest_event() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [1, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[3, 0]]},
        ],
        events=[
            {
                "event_id": "00000000-0000-4000-8000-0000000000cc",
                "tick": 9,
                "event_type": "HARVEST_SUCCEEDED",
                "actor_id": str(WORKER2_ID),
                "position": [3, 0],
                "values": {"amount": 1, "source": "DROPPED_CARGO"},
            }
        ],
    )
    turn = _turn(state)
    tactic._observe_resources(turn)
    assert (3, 0) in tactic._known_resources


def test_worker_sweep_directions_alternate_by_index() -> None:
    # 10th review: half-zone bias — even-index Workers sweep the NORTH half
    # first (start north), odd the south half, so both halves get covered.
    tactic._begin_outbound("worker0", 0, (0, 0), (0, 0))
    tactic._begin_outbound("worker1", 1, (0, 0), (0, 0))
    assert tactic._explore_state["worker0"][1] == 0  # even -> north first
    assert tactic._explore_state["worker1"][1] == 1  # odd  -> south first


def test_guard_ranger_holds_near_core() -> None:
    # 7th review DEF-2-1: the first Ranger is the dedicated Core guard — it
    # returns toward the Core when far instead of roaming the whole chunk, so
    # a raid always meets ranged return fire. Ranger at (3,3), Core at (0,0),
    # distance 6 -> it must MOVE toward the Core (no enemies to shoot).
    state = _state_with_workers(n_workers=4, resources=10, n_rangers=1)
    turn = _turn(state)
    decide(turn)
    ranger_uid = UUID(int=0x4000)
    action = _action(turn.plan, ranger_uid)
    assert action is not None
    assert action.type == "MOVE"
    pos = (3, 3)
    nxt = (pos[0] + action.direction.delta[0], pos[1] + action.direction.delta[1])
    assert tactic._manhattan(nxt, (0, 0)) < tactic._manhattan(pos, (0, 0))


def test_laden_worker_waits_when_core_cell_occupied() -> None:
    # Delivery congestion: two laden Workers, one ON the Core cell depositing,
    # one ADJACENT to it. The adjacent Worker must WAIT (no action) rather than
    # shove into the full 2/2 Core cell and spam CELL_UNIT_LIMIT (502/300 ticks
    # observed). Once the occupant leaves, it steps in next Tick.
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [0, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 1,
        },
        {
            "kind": "UNIT",
            "id": str(WORKER2_ID),
            "controlled": True,
            "position": [1, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 1,
        },
    ]
    state = _state(resources=5, population=2, objects=objects)
    turn = _turn(state)
    decide(turn)
    # The Worker ON the Core cell deposits.
    assert _action(turn.plan, WORKER_ID).type == "DEPOSIT"
    # The ADJACENT Worker WAITs (no MOVE into the occupied Core cell).
    assert _action(turn.plan, WORKER2_ID) is None


def test_empty_worker_yields_core_cell_to_laden_queue() -> None:
    # Delivery deadlock: an EMPTY Worker parked on the Core cell (2/2 full)
    # blocks every laden Worker from depositing — 414e50 sat on Core for 25+
    # ticks while laden workers queued at distance 1, r frozen. The empty
    # Worker must MOVE outward to an open adjacent cell to yield the Core.
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [0, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 0,
        },
        {
            "kind": "UNIT",
            "id": str(WORKER2_ID),
            "controlled": True,
            "position": [1, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 1,
        },
    ]
    state = _state(resources=5, population=2, objects=objects)
    turn = _turn(state)
    decide(turn)
    # The empty Worker on the Core cell must MOVE off (yield the 2/2 cell).
    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "MOVE"


def test_empty_worker_fallback_never_steps_into_core(monkeypatch) -> None:
    # 当前沿 A* 和扫描暂时都无法给出下一步时，最终脱困兜底仍必须遵守空载
    # Worker 不进 Core 的不变量。修复前第二次兜底会把 Core 当目标，导致同
    # Tick 生产因 CELL_UNIT_LIMIT 失败。
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [1, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 0,
        },
        {"kind": "OBSTACLE", "positions": [[1, -1], [1, 1], [2, 0]]},
    ]
    state = _state(resources=0, population=1, objects=objects)
    turn = _turn(state)
    monkeypatch.setattr(
        tactic,
        "_astar_step_result",
        lambda *args, **kwargs: (None, False),
    )
    monkeypatch.setattr(tactic, "_explore_step", lambda *args, **kwargs: None)

    decide(turn)

    assert _action(turn.plan, WORKER_ID) is None


def test_laden_worker_backs_off_when_ring_would_wall_in_core() -> None:
    # User suggestion: when the Core cell is occupied, keep at least one
    # adjacent slot open so the occupant can always leave. Here the Core cell
    # is taken by a laden worker and the other 3 adjacent cells are already
    # full (occ 2/2); the 4th laden worker at (0,-1) must BACK OFF (it is the
    # last exit) rather than take it and wall in the Core-cell worker.
    objects = [
        {
            "kind": "CORE",
            "id": str(CORE_ID),
            "controlled": True,
            "owner_username": "arena_hero",
            "position": [0, 0],
            "hp": 5,
            "shield": 5,
            "state": "NORMAL",
        },
        {
            "kind": "UNIT",
            "id": str(WORKER_ID),
            "controlled": True,
            "position": [0, 0],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 1,
        },
    ]
    # Full the 3 other adjacent cells to 2/2 and put the 4th laden worker at
    # the last adjacent slot (0,-1).
    for idx, (ox, oy) in enumerate([(1, 0), (-1, 0), (0, 1)]):
        for slot in range(2):
            objects.append(
                {
                    "kind": "UNIT",
                    "id": str(UUID(int=0x6000 + idx * 2 + slot)),
                    "controlled": True,
                    "position": [ox, oy],
                    "hp": 2,
                    "unit_type": "WORKER",
                    "cargo": 1 if slot == 0 else 0,
                }
            )
    objects.append(
        {
            "kind": "UNIT",
            "id": str(WORKER2_ID),
            "controlled": True,
            "position": [0, -1],
            "hp": 2,
            "unit_type": "WORKER",
            "cargo": 1,
        }
    )
    state = _state(resources=5, population=9, objects=objects)
    turn = _turn(state)
    decide(turn)
    # The 4th laden worker must NOT stay on the last exit slot: it either moves
    # off or WAITs, but never MOVE in the direction of the Core.
    action = _action(turn.plan, WORKER2_ID)
    if action is not None:
        assert action.type == "MOVE"
        # It must back away from the Core (increase distance), not hold ground.
        nxt = (0, -1 + action.direction.delta[1])
        assert tactic._manhattan(nxt, (0, 0)) > 1


def test_boxed_in_worker_escapes_pocket() -> None:
    # A Worker cycling between a few cells in an obstacle pocket (recent
    # positions all fit a tiny box) never triggers the STUCK check (which keys
    # on stillness) and spins forever. It must break out by moving away from
    # the Core instead. (Observed: ce6788 cycled between (12,215)/(12,216)/
    # (13,216) for 20+ ticks.)
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [5, 5],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
        ],
    )
    tactic._pos_history[str(WORKER_ID)] = [(5, 5), (5, 6), (5, 5), (5, 6)]
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "MOVE"
    nxt = (5 + action.direction.delta[0], 5 + action.direction.delta[1])
    # It must not walk toward the Core (that would re-enter the pocket).
    assert tactic._manhattan(nxt, (0, 0)) >= tactic._manhattan((5, 5), (0, 0))


def test_worker_harvests_resource_before_boxed_escape() -> None:
    # A worker standing on a visible resource cell must HARVEST even when its
    # recent positions look boxed-in. Observed: 68a41e parked on (11,247), a
    # visible resource, yet the boxed-in escape shuttled it into a move and it
    # drifted off without collecting. Harvest must win over the escape logic.
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [2, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 0,
            },
            {"kind": "RESOURCE", "positions": [[2, 0]]},
        ],
    )
    # Make the worker look boxed-in so the escape logic WOULD fire.
    tactic._pos_history[str(WORKER_ID)] = [(2, 0), (1, 0), (2, 0), (1, 0)]
    turn = _turn(state)
    decide(turn)
    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "HARVEST"


def test_laden_worker_uses_return_path_before_boxed_escape() -> None:
    state = _state(
        objects=[
            {
                "kind": "CORE",
                "id": str(CORE_ID),
                "controlled": True,
                "owner_username": "arena_hero",
                "position": [0, 0],
                "hp": 5,
                "shield": 5,
                "state": "NORMAL",
            },
            {
                "kind": "UNIT",
                "id": str(WORKER_ID),
                "controlled": True,
                "position": [3, 0],
                "hp": 2,
                "unit_type": "WORKER",
                "cargo": 1,
            },
        ]
    )
    tactic._pos_history[str(WORKER_ID)] = [(3, 0), (4, 0), (3, 0), (4, 0)]

    turn = _turn(state)
    decide(turn)

    action = _action(turn.plan, WORKER_ID)
    assert action is not None
    assert action.type == "MOVE"
    assert action.direction == Direction.LEFT


# ---------------------------------------------------------------------------
# OBS-001: decision timing and unique-tick observation
# ---------------------------------------------------------------------------


def test_percentile_sorted_values() -> None:
    from meta.monitor import _percentile

    assert _percentile([], 50) == 0
    assert _percentile([100], 50) == 100
    assert _percentile([100], 0) == 100
    assert _percentile([100], 100) == 100
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    p50 = _percentile(vals, 50)
    assert 45 <= p50 <= 55, f"P50 {p50} out of expected range"
    p99 = _percentile(vals, 99)
    assert p99 >= 90, f"P99 {p99} out of expected range"
    p0 = _percentile(vals, 0)
    assert p0 == 10


def test_monitor_parses_timing_field() -> None:
    from meta.monitor import _parse_line

    rec = _parse_line(
        "t100 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[80,12,92] "
        "ev[] plan[-]"
    )
    assert rec is not None
    assert rec["tick"] == 100
    assert rec["decide_ms"] == 80
    assert rec["submit_ms"] == 12
    assert rec["total_ms"] == 92


def test_monitor_aggregates_astar_path_telemetry(tmp_path) -> None:
    from meta import monitor

    lines = [
        "t100 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "path[c12,e40267,b10,p2] TM[518,994,1513] ST[ACCEPTED] "
        "ev[] plan[-]",
        "t101 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "path[c1,e176,b0,p1] TM[47,979,1026] ST[ACCEPTED] "
        "ev[] plan[-]",
    ]
    log_path = tmp_path / "game.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    record = monitor._parse_line(lines[0])
    assert record is not None
    assert record["path"] == {"c": 12, "e": 40267, "b": 10, "p": 2}

    kpi = monitor.analyze(log_path)
    assert kpi.astar_calls == 13
    assert kpi.astar_expansions == 40443
    assert kpi.astar_budget_hits == 10
    assert kpi.astar_paths == 3
    assert kpi.astar_calls_list == [12, 1]
    assert kpi.astar_expansions_list == [40267, 176]

    text = monitor.report(kpi, monitor.detect_bottlenecks(kpi))
    assert "A* path" in text
    assert "calls 13" in text
    assert "expansions 40443" in text


def test_monitor_backward_compat_no_timing() -> None:
    from meta.monitor import _parse_line

    rec = _parse_line(
        "t99 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "ev[] plan[-]"
    )
    assert rec is not None
    assert rec["tick"] == 99
    assert "decide_ms" not in rec
    assert "submit_ms" not in rec
    assert "total_ms" not in rec


def test_monitor_classifies_move_failures_and_enemy_core_visibility(tmp_path) -> None:
    from meta import monitor

    line = (
        "t100 r15/20 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
        "W[-] O[-] vis2[4,4C,5,5WORKER] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[1,2,3] ST[ACCEPTED] "
        "ev[UNIT_MOVE_FAILED.MOVE_CONTESTED;UNIT_MOVE_FAILED.CELL_UNIT_LIMIT;"
        "UNIT_MOVE_SUCCEEDED] plan[-]\n"
    )
    log_path = tmp_path / "game.log"
    log_path.write_text(line, encoding="utf-8")

    record = monitor._parse_line(line)
    assert record is not None
    assert record["vis_core"] is True

    kpi = monitor.analyze(log_path)
    assert kpi.move_failed == 2
    assert kpi.move_failed_contested == 1
    assert kpi.move_failed_cell == 1
    assert kpi.move_succeeded == 1
    assert kpi.ticks_with_enemy_visible == 1
    assert kpi.ticks_with_enemy_core_visible == 1
    alerts = monitor.detect_bottlenecks(kpi)
    assert any("UNIT_CLUMPING" in alert for alert in alerts)
    assert any("NO_RAID" in alert for alert in alerts)


def test_monitor_does_not_call_worker_visibility_no_raid(tmp_path) -> None:
    from meta import monitor

    line = (
        "t100 r5/10 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
        "W[-] O[-] vis1[5,5WORKER] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[1,2,3] ST[ACCEPTED] ev[] plan[-]\n"
    )
    log_path = tmp_path / "game.log"
    log_path.write_text(line, encoding="utf-8")

    kpi = monitor.analyze(log_path)
    assert kpi.ticks_with_enemy_visible == 1
    assert kpi.ticks_with_enemy_core_visible == 0
    assert not any("NO_RAID" in alert for alert in monitor.detect_bottlenecks(kpi))


def test_unique_tick_dedup() -> None:
    from meta.monitor import analyze, KPI
    import tempfile

    # Two unique ticks, t100 appears twice (duplicate).
    log = (
        "t100 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[80,12,92] ev[] plan[-]\n"
        "t100 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[85,15,100] ev[] plan[-]\n"
        "t101 r51/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[90,10,100] ev[] plan[-]\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(log)
        f.flush()
        kpi = analyze(f.name)
    assert kpi.unique_ticks == 2
    assert kpi.duplicate_ticks == 1
    assert kpi.records == 3
    assert kpi.ticks == 2
    # Only unique ticks' timing is collected (t100 first occurrence, t101).
    assert len(kpi.decide_ms_list) == 2
    assert kpi.decide_ms_list == [80, 90]


def test_timing_format_is_compact_integers() -> None:
    """TM field must be three comma-separated non-negative integers, no keys."""
    import re

    tm_re = re.compile(r"TM\[(\d+),(\d+),(\d+)\]")
    # Confirm regex only captures positive integer groups.
    m = tm_re.search("TM[0,1,9999]")
    assert m is not None
    assert m.group(1) == "0"
    assert m.group(3) == "9999"
    # Reject non-integer.
    assert tm_re.search("TM[1.5,2,3]") is None
    # Reject trailing content (no path/headers).
    line = "t100 TM[80,12,92] ev[]"
    m = tm_re.search(line)
    assert m is not None
    # The field sits between eco and ev; no sensitive content.
    sensitive = {"key", "secret", "token", "auth", "bearer"}
    assert not any(w in line.lower() for w in sensitive)


def test_duplicate_accepted_tick_does_not_double_count_business_kpis(tmp_path) -> None:
    from meta.monitor import analyze

    accepted = (
        "r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a1,av1,ah0,blk0,cool0,unr0,harv1,dep1] "
        "TM[80,12,92] ST[ACCEPTED] "
        "ev[HARVEST_SUCCEEDED;DEPOSIT_SUCCEEDED] plan[-]"
    )
    duplicate = accepted.replace("a1,av1", "a9,av9").replace(
        "harv1,dep1", "harv9,dep9"
    ).replace("TM[80,12,92]", "TM[85,15,100]")
    next_tick = accepted.replace("r50/95", "r51/95").replace(
        "a1,av1", "a2,av2"
    ).replace("TM[80,12,92]", "TM[90,10,100]")
    log_path = tmp_path / "game.log"
    log_path.write_text(
        f"t100 {accepted}\n"
        f"t100 {duplicate}\n"
        f"t101 {next_tick}\n",
        encoding="utf-8",
    )

    kpi = analyze(log_path)

    assert kpi.records == 3
    assert kpi.ticks == 2
    assert kpi.unique_ticks == 2
    assert kpi.duplicate_ticks == 1
    assert kpi.resource_assignments == 3
    assert kpi.visible_resource_assignments == 3
    assert kpi.harvest == 2
    assert kpi.deposit == 2
    assert kpi.event_hist["HARVEST_SUCCEEDED"] == 2
    assert kpi.event_hist["DEPOSIT_SUCCEEDED"] == 2
    assert kpi.decide_ms_list == [80, 90]


def test_failed_ticks_are_separate_from_successful_kpis(tmp_path) -> None:
    from meta.monitor import analyze, detect_bottlenecks

    accepted = (
        "t102 r50/95 pop19(W16 V2 R1) core@15,234 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[80,12,92] ST[ACCEPTED] ev[] plan[-]\n"
    )
    log_path = tmp_path / "game.log"
    log_path.write_text(
        "t99 submit_skipped (COMMAND_WINDOW_CLOSED)\n"
        "t100 ST[SUBMIT_FAILED] ER[COMMAND_WINDOW_CLOSED] TM[80,1,81]\n"
        "t101 ST[DECISION_FAILED] ER[DECISION_ERROR] TM[300,0,300]\n"
        + accepted,
        encoding="utf-8",
    )

    kpi = analyze(log_path)
    alerts = detect_bottlenecks(kpi)

    assert kpi.records == 4
    assert kpi.ticks == 1
    assert kpi.failed_ticks == 3
    assert kpi.window_errors == 2
    assert kpi.submit_errors == 2
    assert kpi.decision_errors == 1
    assert kpi.error_hist == {
        "COMMAND_WINDOW_CLOSED": 2,
        "DECISION_ERROR": 1,
    }
    assert kpi.resource_assignments == 0
    assert kpi.decide_ms_list == [80]
    assert any("COMMAND_WINDOW_CLOSED" in alert for alert in alerts)
    assert any("SUBMIT_ERRORS" in alert for alert in alerts)
    assert any("DECISION_ERRORS" in alert for alert in alerts)


def test_monitor_detects_tick_gaps_and_counts_core_spawns(tmp_path) -> None:
    from meta import monitor

    def line(tick: int, resources: int, events: str) -> str:
        return (
            f"t{tick} r{resources}/10 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
            "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
            "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
            "TM[1,2,3] ST[ACCEPTED] ev["
            f"{events}] plan[-]\n"
        )

    log_path = tmp_path / "game.log"
    log_path.write_text(
        line(10, 8, "")
        + line(13, 0, "CORE_SPAWN_SUCCEEDED"),
        encoding="utf-8",
    )

    kpi = monitor.analyze(log_path)

    assert kpi.ticks == 2
    assert kpi.tick_gaps == 1
    assert kpi.missing_ticks == 2
    assert kpi.spawn == 1
    assert kpi.resource_drops == 0
    assert any("TICK_GAP" in alert for alert in monitor.detect_bottlenecks(kpi))


def test_monitor_parses_compound_events_with_amount_payloads(tmp_path) -> None:
    from meta import monitor

    line = (
        "t10 r5/10 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv1,dep1] "
        "TM[1,2,3] ST[ACCEPTED] "
        "ev[HARVEST_SUCCEEDED[1];CORE_SPAWN_SUCCEEDED;DEPOSIT_SUCCEEDED[1]] plan[-]\n"
    )
    log_path = tmp_path / "game.log"
    log_path.write_text(line, encoding="utf-8")

    record = monitor._parse_line(line)
    assert record is not None
    assert record["events"] == [
        "HARVEST_SUCCEEDED[1]",
        "CORE_SPAWN_SUCCEEDED",
        "DEPOSIT_SUCCEEDED[1]",
    ]
    kpi = monitor.analyze(log_path)
    assert kpi.harvest == 1
    assert kpi.deposit == 1
    assert kpi.spawn == 1


def test_monitor_excludes_manual_ticks_and_accounts_v014_costs(tmp_path) -> None:
    from meta import monitor

    def state_line(tick: int, events: str, details: str) -> str:
        return (
            f"t{tick} r7/20 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
            "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
            "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
            "TM[1,2,3] ST[ACCEPTED] "
            f"ev[{events}] dt[{details}] plan[-]\n"
        )

    log_path = tmp_path / "game.log"
    log_path.write_text(
        state_line(
            10,
            "DEPOSIT_SUCCEEDED[2];CORE_SPAWN_SUCCEEDED",
            "DEPOSIT_SUCCEEDED|tick=10|amount=2;CORE_SPAWN_SUCCEEDED|tick=10|cost=5",
        )
        + "t10 rcv[MANUAL] actions[1] plan[C:spawn:worker]\n"
        + state_line(
            11,
            "DEPOSIT_SUCCEEDED[3];CORE_RESOURCES_CAPTURED[4];CORE_RESOURCE_OVERFLOW_DESTROYED[1]",
            "DEPOSIT_SUCCEEDED|tick=11|amount=3;CORE_RESOURCES_CAPTURED|tick=11|amount=4;"
            "CORE_RESOURCE_OVERFLOW_DESTROYED|tick=11|amount=1;CORE_REPAIR_SUCCEEDED|tick=11|cost=1",
        )
        + "t11 rcv[AGENT] actions[0] plan[-]\n",
        encoding="utf-8",
    )

    kpi = monitor.analyze(log_path)

    assert kpi.manual_ticks == 1
    assert kpi.comparable_ticks == 1
    assert kpi.manual_receipts == 1
    assert kpi.agent_receipts == 1
    assert kpi.deposit_amount == 3
    assert kpi.captured_amount == 4
    assert kpi.overflow_loss == 1
    assert kpi.repair_cost == 1
    assert kpi.spawn_cost == 0
    assert kpi.net_resources == 5


def test_monitor_correlates_agent_harvest_to_deposit_latency(tmp_path) -> None:
    from meta import monitor

    def state_line(tick: int, resources: int, events: str, details: str) -> str:
        return (
            f"t{tick} r{resources}/10 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
            "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
            "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
            "TM[1,2,3] ST[ACCEPTED] "
            f"ev[{events}] dt[{details}] plan[-]\n"
        )

    log_path = tmp_path / "game.log"
    log_path.write_text(
        state_line(
            10,
            5,
            "HARVEST_SUCCEEDED[1]",
            "HARVEST_SUCCEEDED|tick=10|event=h|actor=worker-1|pos=1,1|amount=1|source=RESOURCE_NODE",
        )
        + "t10 rcv[AGENT] actions[1] plan[Uworker-1:harvest]\n"
        + state_line(
            13,
            6,
            "DEPOSIT_SUCCEEDED[1]",
            "DEPOSIT_SUCCEEDED|tick=13|event=d|actor=worker-1|target=core-1|pos=0,0|amount=1|capacity=10",
        )
        + "t13 rcv[AGENT] actions[1] plan[Uworker-1:deposit]\n",
        encoding="utf-8",
    )

    kpi = monitor.analyze(log_path)

    assert kpi.harvest_deposit_chains == 1
    assert kpi.unmatched_harvests == 0
    assert kpi.unmatched_deposits == 0
    assert kpi.harvest_deposit_p50 == 3
    assert kpi.harvest_deposit_p95 == 3


def test_monitor_excludes_manual_harvest_deposit_chain(tmp_path) -> None:
    from meta import monitor

    def state_line(tick: int, events: str, details: str) -> str:
        return (
            f"t{tick} r5/10 pop2(W2 V0 R0) core@0,0 hp5/sh5/NORMAL "
            "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
            "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
            "TM[1,2,3] ST[ACCEPTED] "
            f"ev[{events}] dt[{details}] plan[-]\n"
        )

    log_path = tmp_path / "game.log"
    log_path.write_text(
        state_line(
            20,
            "HARVEST_SUCCEEDED[1]",
            "HARVEST_SUCCEEDED|tick=20|event=h|actor=worker-1|pos=1,1|amount=1|source=RESOURCE_NODE",
        )
        + "t20 rcv[MANUAL] actions[1] plan[Uworker-1:harvest]\n"
        + state_line(
            22,
            "DEPOSIT_SUCCEEDED[1]",
            "DEPOSIT_SUCCEEDED|tick=22|event=d|actor=worker-1|target=core-1|pos=0,0|amount=1|capacity=10",
        )
        + "t22 rcv[MANUAL] actions[1] plan[Uworker-1:deposit]\n",
        encoding="utf-8",
    )

    kpi = monitor.analyze(log_path)

    assert kpi.manual_ticks == 2
    assert kpi.harvest_deposit_chains == 0
    assert kpi.unmatched_harvests == 0
    assert kpi.unmatched_deposits == 0


@pytest.mark.parametrize(
    ("limit", "label"),
    [(250, "P50"), (1000, "P95"), (2000, "P99")],
)
def test_plan_timing_thresholds_are_strict(limit: int, label: str) -> None:
    from meta.monitor import KPI, detect_bottlenecks

    if label == "P50":
        at_values = [limit]
        below_values = [limit - 1]
    elif label == "P95":
        at_values = [0] * 18 + [limit, limit]
        below_values = [0] * 18 + [limit - 1, limit - 1]
    else:
        at_values = [0] * 98 + [limit, limit]
        below_values = [0] * 98 + [limit - 1, limit - 1]
    at_limit = KPI(ticks=1, records=1, unique_ticks=1, decide_ms_list=at_values)
    below_limit = KPI(
        ticks=1, records=1, unique_ticks=1, decide_ms_list=below_values
    )

    at_alerts = detect_bottlenecks(at_limit)
    below_alerts = detect_bottlenecks(below_limit)

    assert any(label in alert for alert in at_alerts if alert.startswith("PLAN_TIMING"))
    assert not any("PLAN_TIMING" in alert for alert in below_alerts)


def test_percentile_keeps_fractional_boundary() -> None:
    from meta.monitor import KPI, _percentile, detect_bottlenecks

    assert _percentile([0, 501], 50) == 250.5
    kpi = KPI(ticks=1, records=1, unique_ticks=1, decide_ms_list=[0, 501])
    assert any("P50 250.5ms" in alert for alert in detect_bottlenecks(kpi))


def test_plan_gate_excludes_submit_network_latency() -> None:
    from meta.monitor import KPI, detect_bottlenecks

    kpi = KPI(
        ticks=1,
        records=1,
        unique_ticks=1,
        decide_ms_list=[100],
        submit_ms_list=[5000],
        total_ms_list=[5100],
    )

    assert not any("PLAN_TIMING" in alert for alert in detect_bottlenecks(kpi))


def test_monitor_json_excludes_unbounded_tick_id_state(tmp_path, capsys) -> None:
    from meta import monitor

    log_path = tmp_path / "game.log"
    log_path.write_text(
        "t1 r1/10 pop1(W1 V0 R0) core@0,0 hp5/sh5/NORMAL "
        "W[-] O[-] vis0[-] res0[] obs0 beacon0,0 "
        "eco[a0,av0,ah0,blk0,cool0,unr0,harv0,dep0] "
        "TM[1,2,3] ST[ACCEPTED] ev[] plan[-]\n",
        encoding="utf-8",
    )

    monitor.main(["--json", str(log_path)])
    output = json.loads(capsys.readouterr().out)

    assert "seen_ticks" not in output
    assert output["unique_ticks"] == 1
    assert output["records"] == 1


def test_error_codes_are_bounded_and_do_not_echo_messages(tmp_path) -> None:
    import play
    from arena_hero import APIError, TransportError

    unsafe = APIError(
        status_code=400,
        error="COMMAND_WINDOW_CLOSED\nAuthorization: bearer secret-token",
    )
    assert play._safe_error_code(unsafe) == "API_ERROR"
    assert play._safe_error_code(TransportError()) == "TRANSPORT_ERROR"
    line = play._failure_log_line(
        7, "SUBMIT_FAILED\nLEAK", "bad code\nsecret", 1, 2, 3
    )
    assert "\n" not in line
    assert "secret" not in line.lower()
    assert "ST[SUBMIT_FAILED]" in line
    assert "ER[UNKNOWN_ERROR]" in line


def test_play_logs_decision_and_submit_failures_without_exception_text(
    tmp_path, monkeypatch
) -> None:
    import play
    from arena_hero import APIError

    class FakeTurn:
        def __init__(self, tick: int, submit_error: Exception | None = None) -> None:
            self.tick = tick
            self.submit_error = submit_error

        def submit(self):
            if self.submit_error is not None:
                raise self.submit_error
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self._turns = [
                FakeTurn(1),
                FakeTurn(
                    2,
                    APIError(
                        status_code=409,
                        error="TICK_MISMATCH\nsecret-response",
                    ),
                ),
                FakeTurn(3, RuntimeError("unstructured secret-response")),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def turns(self):
            return iter(self._turns)

    log_path = tmp_path / "game.log"
    monkeypatch.setattr(play, "ArenaHeroClient", FakeClient)
    monkeypatch.setattr(play, "LOG_PATH", log_path)

    def fake_decide(turn) -> None:
        if turn.tick == 1:
            raise RuntimeError("decision secret\nresponse body")

    monkeypatch.setattr(play, "decide", fake_decide)

    assert play.play("not-a-real-key", "https://example.invalid", None) == 0
    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert lines[0].startswith("t1 ST[DECISION_FAILED] ER[DECISION_ERROR]")
    assert lines[1].startswith("t2 ST[SUBMIT_FAILED] ER[API_ERROR]")
    assert lines[2].startswith("t3 ST[SUBMIT_FAILED] ER[UNKNOWN_ERROR]")
    assert all("secret" not in line.lower() for line in lines)
