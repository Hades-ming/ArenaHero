"""Decision tests for the balanced tactic, without live credentials or network.

These build ``PlayerState`` fixtures and a ``Turn`` whose submitter is a stub,
then assert on the queued ``turn.plan``. They follow the coverage list in the
bundled ``references/tactic-authoring.md``.
"""

from __future__ import annotations

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
def _reset_explore_state() -> None:
    """Each test sees a clean per-Worker exploration memory."""
    tactic._explore_state.clear()
    tactic._known_resources.clear()
    tactic._known_obstacles.clear()
    tactic._known_enemy_cores.clear()
    tactic._explored_cells.clear()
    tactic._pos_history.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()
    tactic._stuck_ticks.clear()
    tactic._lock_meta.clear()
    yield
    tactic._explore_state.clear()
    tactic._known_resources.clear()
    tactic._known_obstacles.clear()
    tactic._known_enemy_cores.clear()
    tactic._explored_cells.clear()
    tactic._pos_history.clear()
    tactic._prev_pos.clear()
    tactic._last_pos.clear()
    tactic._stuck_ticks.clear()
    tactic._lock_meta.clear()


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
            "population_tier": 0,
            "upkeep_next_tick": 0,
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
            "population_tier": 0,
            "upkeep_next_tick": 0,
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

    near_state = tactic._explore_state[str(WORKER2_ID)]
    assert tuple(near_state[2:4]) == resource
    far_state = tactic._explore_state.get(str(WORKER_ID), [])
    assert len(far_state) < 4 or tuple(far_state[2:4]) != resource


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


# ---------------------------------------------------------------------------
# Core actions: repair and spawn
# ---------------------------------------------------------------------------


def test_core_repairs_shield_when_threatened_and_damaged() -> None:
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
    assert _core_action(turn.plan).type == "REPAIR_SHIELD"


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
    worker_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
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


def test_standing_army_spawns_ranger_after_vanguard() -> None:
    # Vanguard reserve exists, 4 Workers, 12 resources: build the standing
    # Ranger (range-3 return fire) next.
    state = _state_with_workers(n_workers=4, resources=12, n_vanguards=1)
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
    # 16 Workers -> V2 R1 (fits the free-upkeep pop budget 16+2+1=19, favoring
    # the cheap Vanguard body-block over the pricier Ranger when the budget
    # must shrink). Never overflows the budget for any fleet size.
    assert tactic._standing_army_targets(4) == (1, 1)
    assert tactic._standing_army_targets(8) == (1, 1)
    assert tactic._standing_army_targets(12) == (1, 1)
    assert tactic._standing_army_targets(16) == (2, 1)
    assert tactic._standing_army_targets(17) == (1, 1)
    assert tactic._standing_army_targets(18) == (1, 0)
    # W=19 has no room for any army under the budget; Vanguard floor gives way
    # so the invariant W+V+R <= 19 holds (6th review, strategy STRAT-4).
    assert tactic._standing_army_targets(19) == (0, 0)
    for w in range(4, tactic.FREE_UPKEEP_CAP):
        v, r = tactic._standing_army_targets(w)
        # Free-upkeep guarantee: the total never reaches the upkeep tier-1
        # boundary (20). 6th review corrected the old <=20 bound.
        assert w + v + r <= tactic.FREE_UPKEEP_CAP - 1, (
            f"pop budget overflow at {w}: W{v}R{r}"
        )
        if w <= 17:
            assert v >= 1 and r >= 1, f"army floor broken at {w}"
        elif w == 18:
            assert v >= 1, f"vanguard floor broken at {w}"


def test_pop_over_budget_culls_empty_worker() -> None:
    # A fleet that grew past the free-upkeep cap (population 21) must
    # self-destruct a surplus EMPTY Worker each Tick to drop back under 20,
    # stopping the upkeep tier-1 drain. Combat Units are never culled.
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
    state = _state(resources=50, population=21, objects=objects)
    turn = _turn(state)
    decide(turn)
    culled = [
        uid
        for uid, a in turn.plan.unit_actions.items()
        if getattr(a, "type", "") == "SELF_DESTRUCT"
    ]
    # pop21 - budget19 = 2 surplus Workers culled in ONE Tick (SELF_DESTRUCT
    # resolves before upkeep, so the drain stops immediately).
    assert len(culled) == 2
    for uid in culled:
        culled_unit = next(u for u in turn.units if str(u.id) == str(uid))
        assert culled_unit.unit_type == "WORKER"


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
