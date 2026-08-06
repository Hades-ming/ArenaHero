"""Goal-driven tactic for Arena Hero — FINAL OBJECTIVE: accumulate the most resources.

Decisions are separated from connection setup so :func:`decide` can be tested
without a live credential. See ``references/tactic-authoring.md`` and
``references/game-rules.md`` (bundled with the arena-hero skill) for the rules
this tactic follows; no numeric rule is inferred from memory.

FINAL GOAL — maximize resource accumulation
-------------------------------------------
Score is not exposed in the player state, so every behavior is justified by how
it grows or protects the resource economy ("most resources" = highest durable
resource throughput and net stockpile, not idle hoarding):

* **Find & harvest** — Workers fan out to *discover* resources (active
  exploration when none are visible) and haul them home. Discovery is the #1
  lever on income; standing still earns nothing.
* **Deposit & reinvest** — carried cargo is deposited on the Core cell; the Core
  spends into more Workers so income compounds, while respecting v0.14 pricing.
* **Protect the economy** — losing the Core forfeits ALL stored resources, and
  losing Units forfeits their cargo. Defense (shield repair, a standing army,
  a Wall) is therefore an *investment in the goal*, never waste.
* **Build attack units in time (vs. other enemies)** — when enemies are visible,
  the army is raised immediately (even below the economy floor) so a raid meets
  return fire and the economy is not raided.
* **Raid enemy Cores for loot + elimination** — destroying a visible enemy
  Core removes their whole fleet and captures part of their stockpiled
  resources. The bounty is NOT a flat +6: per the ``CORE_RESOURCES_CAPTURED``
  event you gain ``amount`` of the enemy Core's stored ``available`` resources
  while ``destroyed`` (``available - amount``) is lost, and your own capacity can
  destroy the overflow. So a raided Core may yield zero if it spent everything.
  When one is in vision the tactic forms a strike force to capture it instead
  of passively defending.
* **Avoid idle gold** — resources on hand earn nothing; the reserve/wall logic
  spends surplus into economy, defense, or army so capital is always working.

Policy:

* deposit carried Worker cargo when sharing the Core cell;
* harvest when an empty Worker stands on a currently visible resource cell;
* move empty Workers toward the nearest visible resource;
* when NO resource is visible, explore: each Worker keeps a marching direction
  and turns periodically to sweep new ground, so it does not stall on a bare
  view (vision radius is small, so standing still never reveals resources);
* Rangers shoot visible legal targets (enemy Core prioritized), else explore/kite;
* Vanguards sweep the adjacent cell with the most enemies, else hold near Core;
* repair Core shield only when under visible threat;
* spawn Workers toward a soft target while the first dynamic-price tier is active
  and the Core cell has room;
* when enemies or an enemy Core are visible, prioritize attack-unit production;
* leave an object on WAIT when no legal useful action is known.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arena_hero import (
    BeaconStatus,
    Direction,
    HarvestSource,
    UnitType,
    unit_cost,
)

if TYPE_CHECKING:
    from arena_hero import Core, CoreView, Turn, Unit, UnitView

# v0.14 has no per-Tick upkeep. Keep the first dynamic-price boundary as a
# conservative soft population target: population 20 is allowed, while the
# next production (at population 20) is the first dynamically priced unit.
FREE_UPKEEP_CAP = 20
# Comfortable Worker count the tactic tries to maintain. The fourth review
# found TARGET_WORKERS=16 was unreachable at the observed harvest rate AND
# counterproductive: every deposit was immediately spent on a Worker spawn,
# so Core storage never accumulated (the durable score proxy). The fourth
# review set this to 8 to let deposits bank. The FIFTH review (economy +
# skeptic personas) re-measured and found the binding constraint is node
# DISCOVERY rate (~0.033 res/tick) not the chunk-quota ceiling (2.0/tick,
# 34-64x headroom): more Workers find nodes faster, directly raising
# throughput, and the first dynamic-price tier starts at production population 20. Raised 8 -> 12, then
# 12 -> 15 after r hit the pop-14 capacity ceiling (70): more Workers both
# raise discovery AND raise Core capacity (each Unit +5), letting r bank
# past 70. Still below the first price step (pop 17 < 20). The bank reserve +
# army-short gate still prevent draining deposits to r0.
TARGET_WORKERS = FREE_UPKEEP_CAP
MAX_WORKERS = FREE_UPKEEP_CAP + 1
# Bank reserve: never spend down to zero on a Worker spawn. A spawn must leave
# the Core with at least this many resources afterward, so the economy keeps a
# positive balance and the standing-army bank (toward the 10/12 combat Unit)
# is not reset by every Worker. The fourth review (economy persona) showed the
# prior "spend every r5 on a Worker" policy left Core storage oscillating
# 0->5->0 forever — the single largest economy drain.
WORKER_SPAWN_RESERVE = 3
# Standing-army policy (defends against the raid that destroyed the Core on
# 2026-08-02). The prior tactic spawned combat Units ONLY when a threat was
# already visible, so a surprise raid met an economy at r0 that could not bank
# the 10-resource Vanguard cost in time and the Core burned with zero return
# fire. We now maintain a combat reserve even in peacetime.
#
# Base Vanguard/Ranger prices are 10/12; v0.14 raises them with population. The
# fleet builds the standing reserve as soon as a minimal Worker economy exists,
# and BEFORE growing Workers past that floor — combat readiness outranks a
# larger Worker fleet once the economy can sustain the smallest army.
MIN_WORKERS_BEFORE_ARMY = 4
# If a combat Unit already exists, permit a bounded Worker bridge while the
# Core is not under immediate threat.  The old one-Worker bridge still left a
# healthy Core at W5/V2/R0 waiting for a Ranger; two additional scouts raise
# discovery without allowing an unbounded peaceful army deficit.
ECONOMY_BRIDGE_MAX_WORKERS = MIN_WORKERS_BEFORE_ARMY + 3
# Peacetime standing reserve now SCALES with the Worker fleet via
# _standing_army_targets (a floor of V1/R1, growing ~one combat pair per 8
# Workers up to the first-price-step population budget). These legacy constants document
# the floor that scaling starts from. A Vanguard is the cheapest return-fire
# Unit (1 damage to an adjacent cell, 4 HP body-block) and is built first.
STANDING_VANGUARDS = 1
STANDING_RANGERS = 1
# When a threat is visible, grow the defensive line up to these caps before any
# further Worker growth. A Ranger's range-3 shot can hit a raider before it
# reaches the Core cell, so it is the preferred defender once the reserve exists.
DEFENSE_VANGUARDS = 2
DEFENSE_RANGERS = 2
# Ranger range, straight from the rules: a shot is legal at Manhattan distance
# 1, 2, or 3 on a shared cardinal line with no obstacle between.
RANGER_MAX_RANGE = 3

DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
# Clockwise turn order (kept for Vanguard/Ranger fallback movement).
_TURN_ORDER = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
}
# Chunk is 32x32 (rules). Workers sweep edge-to-edge across the Core's own
# chunk: coverage matters far more than short deposit round-trips while the
# harvest rate is zero (0 * fast_round_trip is still 0).
CHUNK_SIZE = 32
# Horizontal column step between successive boustrophedon sweeps. Worker vision
# radius is 3, so a 6-step column spacing tiles the chunk with one-cell overlap
# and no unscanned gap.
_SWEEP_COL_STEP = 6
# How far north/south a Worker marches before advancing to the next column.
# A value of 16 covers the full 32-cell chunk height, but Workers rarely reach
# the boundary (obstacles, anti-backtrack), so col_off never advances and they
# re-sweep the same column forever. A shorter march span (10) reaches the
# turn-around far more often, so the Worker tiles multiple columns over time
# and actually discovers resources spread across the chunk width.
_SWEEP_HALF_WIDTH = 10
# Column offset wraps monotonically (east-wrap) rather than reflecting, so each
# Worker tiles the full chunk width over successive legs instead of oscillating
# on two columns. The wrap range spans [-HALF, +HALF].
_SWEEP_COL_WRAP = _SWEEP_HALF_WIDTH
# A Worker only harvests a resource it can reach in a step or two from its own
# vision. Chasing a node seen by another Worker makes it leave its band and
# 2-cycle back when that node leaves the other Worker's vision. Worker vision
# radius is 3, so a 4-cell reach covers "I can see it and step onto it".


def _advance_col_off(col_off: int) -> int:
    """Step a Worker's column offset east by ``_SWEEP_COL_STEP`` and wrap.

    A monotonic east-wrap tiles the full chunk width over successive legs
    instead of reflecting back and oscillating on two columns. The offset
    wraps from +_SWEEP_COL_WRAP to -_SWEEP_COL_WRAP so every column in the
    range is swept eventually.
    """
    col_off += _SWEEP_COL_STEP
    if col_off > _SWEEP_COL_WRAP:
        col_off = -_SWEEP_COL_WRAP + (col_off - _SWEEP_COL_WRAP - 1)
    return col_off


def _next_explore_col_off(
    col_off: int,
    base_col: int,
    sweep_y_lo: int,
    sweep_y_hi: int,
    chunk_x_lo: int,
    chunk_x_hi: int,
) -> int:
    """Advance a Worker's explore column, finally *consuming* the persistent map.

    ``_explored_cells`` (written every tick in ``_observe_terrain`` and persisted
    to disk) was being saved but never read back — exploration steered purely on
    the chunk boustrophedon + per-Worker column offsets, so the user's persisted
    map was dead weight. This helper reads it: when the next boustrophedon column
    is already fully lit in memory, jump to the nearest reachable column that
    still holds unseen cells instead of re-sweeping dead ground. If *every*
    reachable column is explored, fall back to the plain boustrophedon so Workers
    still revisit ground for respawned resources (purpose (b) of the map). With
    no explored memory yet it is byte-for-byte identical to ``_advance_col_off``.
    """
    natural = _advance_col_off(col_off)
    if not _explored_cells:
        return natural

    def _unexplored(off: int) -> int:
        x = base_col + off
        if x < chunk_x_lo or x > chunk_x_hi:
            return -1  # outside this Worker's reachable band
        return sum(
            1 for y in range(sweep_y_lo, sweep_y_hi + 1) if (x, y) not in _explored_cells
        )

    # Keep the natural next column whenever it still has any fresh cells, so the
    # steady-state sweep is unchanged.
    if _unexplored(natural) > 0:
        return natural

    # Every reachable column is explored: fall back to the plain boustrophedon
    # so Workers still revisit ground for respawned resources (purpose (b)).
    best_off, best_dist = natural, 10**9
    for off in range(chunk_x_lo - base_col, chunk_x_hi - base_col + 1):
        if _unexplored(off) <= 0:
            continue
        d = abs(off - col_off)  # jump to the NEAREST fresh ground, not the
        if d < best_dist:       # farthest, to avoid cross-map marches.
            best_dist, best_off = d, off
    return best_off


# Sweep radius cap: a worker normally explores within this Manhattan radius of
# the Core. Beyond it the deposit round trip is a net loss and workers were
# observed stranding 60-88 cells out (never returning). A prolonged resource
# drought temporarily expands the *empty-worker* frontier so refill nodes in a
# neighbouring chunk can be discovered; laden workers still use the fixed
# return path below.
MAX_SWEEP_RADIUS = 40
MAX_DROUGHT_SWEEP_RADIUS = 96
DROUGHT_EXPAND_EVERY = 8
DROUGHT_EXPAND_STEP = 16


def _exploration_radius() -> int:
    """Return the current empty-worker discovery radius.

    The server refills natural nodes every four Ticks, so a zero-resource
    window is meaningful only after several refill opportunities. Expanding in
    16-cell steps keeps normal delivery trips short while eventually reaching
    adjacent chunks when the home chunk remains empty.
    """
    if _resource_absence_streak < DROUGHT_EXPAND_EVERY + 1:
        return MAX_SWEEP_RADIUS
    steps = (_resource_absence_streak - 1) // DROUGHT_EXPAND_EVERY
    return min(
        MAX_DROUGHT_SWEEP_RADIUS,
        MAX_SWEEP_RADIUS + steps * DROUGHT_EXPAND_STEP,
    )

# Per-Worker exploration memory, keyed by the Unit UUID string. Each entry is
# [direction_index_into_DIRECTIONS, steps_taken_in_this_leg]. This is not a
# retained controller object (the skill forbids that); it is a small intent
# hint that is rebuilt from the current live Units each Tick.
_explore_state: dict[str, list[int]] = {}
# Stable per-Worker frontier targets. These are coordinate intents only; live
# SDK controllers are always read from the current Turn.
_explore_targets: dict[str, tuple[int, int]] = {}
# Frontier targets are coordinate intents, not persistent terrain facts. A
# target that A* proves unreachable is cooled briefly so every idle Worker does
# not repeatedly select the same sealed pocket. A* budget exhaustion is kept as
# a retryable condition and never enters this cooldown map.
_explore_target_cooldown_until: dict[tuple[int, int], int] = {}
_explore_target_failures: dict[tuple[int, int], int] = {}
_EXPLORE_COOLDOWN_BASE = 4
_EXPLORE_COOLDOWN_CAP = 32
_EXPLORE_STALL_TICKS = 6


@dataclass
class ExploreProgress:
    """Track whether a frontier target is making real progress."""

    target: tuple[int, int]
    position: tuple[int, int]
    distance: int
    frontier_gain: int
    stalled_ticks: int = 0


_explore_progress: dict[str, ExploreProgress] = {}
# Recent-positions history per Worker, used to break multi-cell backtracking
# cycles on the laden deposit return path through obstacle corridors. The
# fifth review (skeptic persona) found a laden Worker trapped in a permanent
# 4-cycle (A->B->C->D->A) for 362+ ticks — a single-cell _prev_pos only
# prevents the immediately-prior cell, not the 3-cells-ago cell that closes a
# longer cycle. A deque of the last AVOID_HISTORY positions catches longer
# cycles. Capped at AVOID_HISTORY entries; oldest drops when full.
_prev_pos: dict[str, tuple[int, int]] = {}  # last pos (kept for stuck detection)
_pos_history: dict[str, list[tuple[int, int]]] = {}
AVOID_HISTORY = 4

# Bounded drive-off state for non-guard Rangers (8th review, rank 2): the fleet
# was 100% passive — it only shot what was already in range, so a raider parked
# just outside range was never driven off. Record last-seen enemy positions and
# chase them briefly (bounded ticks + cooldown), never pulling the guard Ranger
# or the home Vanguard off defense.
_last_enemy_pos: dict[str, tuple[tuple[int, int], int]] = {}  # enemy id -> (pos, tick)
_chase_start: dict[str, int] = {}  # ranger id -> tick the chase began
_chase_cooldown_until: dict[str, int] = {}  # ranger id -> tick it may chase again
CHASE_RADIUS = 15         # drive off enemies within this many cells of the Core
CHASE_MAX_TICKS = 8       # give up after chasing this many ticks
CHASE_COOLDOWN_TICKS = 12
ENEMY_MEMORY_TICKS = 6    # forget a last-seen enemy position after this many ticks

# Local resource memory pool: remember resource cells even after they leave
# vision, so Workers don't re-sweep bare ground they already confirmed empty.
# A cell is added when first seen as a resource. It is removed when:
# 1. a HARVEST_SUCCEEDED event confirms it was collected (immediate), or
# 2. any friendly vision source sees the cell and it is NOT in the current
#    turn.resource_cells (the cell is genuinely bare, not just out of view).
# Without this, Workers re-scan already-depleted columns (the 5th review
# identified discovery rate ~0.033 res/tick as the binding constraint —
# memory directly raises effective discovery by avoiding re-scan of empty
# cells already known from earlier sweeps).
_known_resources: set[tuple[int, int]] = set()


@dataclass
class ResourceHint:
    """一个历史资源坐标的持久可信度元数据。"""

    last_confirmed_tick: int
    source: str
    failure_count: int = 0
    cooldown_until: int = 0


_resource_hints: dict[tuple[int, int], ResourceHint] = {}
_resource_telemetry: dict[str, int] = {}
_resource_absence_streak = 0
_RESOURCE_COOLDOWN_BASE = 4
_RESOURCE_COOLDOWN_CAP = 64
_RESOURCE_FAILURE_CAP = 5
_RESOURCE_CONFIRM_SAVE_STEP = 64
# 记忆节点只是提示而非静态地形。超过该逻辑 Tick 数未被确认后，停止把
# Worker 派往该点，交给前沿扫描重新确认它是否仍然存在。
_MAX_HISTORY_RESOURCE_AGE = 256
_HISTORY_RESOURCE_AGE_WEIGHT = 1
_PERSISTED_TICK_CAP = 2**63 - 1
_persistent_state_dirty = False
# Persistent obstacle-terrain memory. turn.obstacle_cells only exposes obstacles
# in current vision; a wall seen once and forgotten makes A* pathing collide
# with it again and again (workers re-route around the same walls every trip).
# Obstacles are static terrain, so persisting them is safe and makes pathing
# near-optimal (user asked: persist the terrain of known cells).
_known_obstacles: set[tuple[int, int]] = set()
# Persistent enemy-Core positions: a Core seen once is remembered forever so a
# hunt (or a post-restart re-acquisition) knows where the rival lives. Enemy
# Units come and go, but a Core is a durable target worth keeping.
_known_enemy_cores: set[tuple[int, int]] = set()
# Persistent explored-region set: every cell a friendly vision source has
# covered (Manhattan within radius). Lets the tactic (a) avoid re-sweeping
# ground it has already lit, and (b) know which areas to periodically revisit
# for respawned resources (community Player D pattern). Large but bounded by
# the explored map; saved with the other state on change.
_explored_cells: set[tuple[int, int]] = set()
# How many NEW explored cells accrue before we snapshot the map to disk again.
# Obstacles are saved on appearance; the explored set grows every tick, so we
# persist it in coarse batches to keep a restart from losing the whole map
# without thrashing disk each tick.
_EXPLORED_SAVE_STEP = 1000
_last_saved_explored: int = 0

# Persistent resource + terrain + enemy + explored memory across restarts.
# Every play.py restart re-imports tactic.py and resets in-memory state, so
# without persistence the resource/obstacle pool is wiped on each deploy and
# Workers re-scan the map from scratch (observed: user saw map resources like
# (15,208) that the tactic had forgotten). Mirrors the reference agent's
# load_persistent_state/save_state.
_STATE_PATH = Path(__file__).resolve().parent / "tactic_state.json"


def _load_persistent_state() -> None:
    """Restore the persisted resource/obstacle/enemy-core/explored memory."""
    global _known_resources, _resource_hints, _persistent_state_dirty
    global _known_obstacles, _known_enemy_cores, _explored_cells
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        raw = data.get("known_resources", [])
        if isinstance(raw, list):
            _known_resources = {
                (int(a), int(b))
                for a, b in raw
                if isinstance(a, int) and isinstance(b, int)
            }
        _resource_hints = {}
        raw_hints = data.get("resource_hints", [])
        if isinstance(raw_hints, list):
            for item in raw_hints:
                if not isinstance(item, dict):
                    continue
                position = item.get("position")
                if (
                    not isinstance(position, list)
                    or len(position) != 2
                    or not all(isinstance(value, int) for value in position)
                ):
                    continue
                cell = (position[0], position[1])
                if cell not in _known_resources:
                    continue

                def bounded_int(name: str, cap: int) -> int:
                    value = item.get(name, 0)
                    if isinstance(value, bool) or not isinstance(value, int):
                        return 0
                    return min(max(0, value), cap)

                _resource_hints[cell] = ResourceHint(
                    last_confirmed_tick=bounded_int(
                        "last_confirmed_tick", _PERSISTED_TICK_CAP
                    ),
                    source=str(item.get("source", "history")),
                    failure_count=bounded_int(
                        "failure_count", _RESOURCE_FAILURE_CAP
                    ),
                    cooldown_until=bounded_int(
                        "cooldown_until", _PERSISTED_TICK_CAP
                    ),
                )
        for cell in _known_resources:
            _resource_hints.setdefault(cell, ResourceHint(0, "legacy"))
        raw_obs = data.get("known_obstacles", [])
        if isinstance(raw_obs, list):
            _known_obstacles = {
                (int(a), int(b))
                for a, b in raw_obs
                if isinstance(a, int) and isinstance(b, int)
            }
        raw_cores = data.get("known_enemy_cores", [])
        if isinstance(raw_cores, list):
            _known_enemy_cores = {
                (int(a), int(b))
                for a, b in raw_cores
                if isinstance(a, int) and isinstance(b, int)
            }
        raw_expl = data.get("explored_cells", [])
        if isinstance(raw_expl, list):
            _explored_cells = {
                (int(a), int(b))
                for a, b in raw_expl
                if isinstance(a, int) and isinstance(b, int)
            }
        _persistent_state_dirty = False
    except (OSError, ValueError, TypeError):
        _known_resources = set()
        _resource_hints = {}
        _known_obstacles = set()
        _known_enemy_cores = set()
        _explored_cells = set()
        _persistent_state_dirty = False


def _save_persistent_state() -> None:
    """Persist resource/obstacle/enemy-core/explored memory (on change)."""
    global _persistent_state_dirty
    try:
        payload = {
            "known_resources": [list(c) for c in sorted(_known_resources)],
            "resource_hints": [
                {
                    "position": list(cell),
                    "last_confirmed_tick": hint.last_confirmed_tick,
                    "source": hint.source,
                    "failure_count": hint.failure_count,
                    "cooldown_until": hint.cooldown_until,
                }
                for cell in sorted(_known_resources)
                for hint in [_resource_hints.setdefault(cell, ResourceHint(0, "legacy"))]
            ],
            "known_obstacles": [list(c) for c in sorted(_known_obstacles)],
            "known_enemy_cores": [list(c) for c in sorted(_known_enemy_cores)],
            "explored_cells": [list(c) for c in sorted(_explored_cells)],
        }
        _STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        _persistent_state_dirty = False
    except OSError:
        pass


def _mark_persistent_state_dirty() -> None:
    """标记状态需要落盘，由当前 Tick 末尾统一写入。"""
    global _persistent_state_dirty
    _persistent_state_dirty = True


def _flush_persistent_state() -> None:
    """每个 Tick 最多合并写入一次完整持久状态。"""
    if _persistent_state_dirty:
        _save_persistent_state()


def _observe_terrain(turn: "Turn") -> None:
    """Merge currently-visible obstacles into the persistent obstacle memory.

    Runs each Tick so walls seen once are remembered forever (obstacles are
    static), making A* pathing and exploration near-optimal. Persists only when
    new walls appear.
    """
    global _known_obstacles, _explored_cells, _last_saved_explored
    before = len(_known_obstacles)
    _known_obstacles.update(turn.obstacle_cells)
    # Record only cells actually visible through the obstacle layout. A plain
    # Manhattan diamond incorrectly marks cells behind walls as explored.
    sources = _vision_sources(turn)
    for src, radius in sources:
        sx, sy = src
        for dx in range(-radius, radius + 1):
            for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
                cell = (sx + dx, sy + dy)
                if _any_vision_sees(cell, [(src, radius)], _known_obstacles):
                    _explored_cells.add(cell)
    if len(_known_obstacles) != before:
        _mark_persistent_state_dirty()
        _last_saved_explored = len(_explored_cells)
    elif len(_explored_cells) - _last_saved_explored >= _EXPLORED_SAVE_STEP:
        # The explored map is the part the user actually wanted persisted; flush
        # it in coarse batches so a play.py restart resumes near where it left off
        # instead of re-scanning from scratch.
        _mark_persistent_state_dirty()
        _last_saved_explored = len(_explored_cells)


_load_persistent_state()


def _record_pos(wid: str, pos: tuple[int, int]) -> None:
    """Append ``pos`` to the Worker's recent-position history (capped).

    Deduplicates consecutive duplicates so a Worker that WAITs in place does
    not fill its history with the same cell and lose the cycle-breaking
    memory. Used to build the avoid-set for pathing.
    """
    hist = _pos_history.get(wid)
    if hist and hist[-1] == pos:
        return
    if hist is None:
        hist = []
        _pos_history[wid] = hist
    hist.append(pos)
    if len(hist) > AVOID_HISTORY:
        del hist[0]


def _is_boxed_in(wid: str) -> bool:
    """True if the Worker's recent positions all fit in a tiny box.

    A Worker trapped in an obstacle pocket cycles between 2-3 cells and never
    stays still, so the STUCK check (which keys on stillness) never fires. If
    the last AVOID_HISTORY positions span <=2 cells on both axes, it is stuck
    in a pocket and should break out (see the boxed-in escape in _control_workers).
    """
    hist = _pos_history.get(wid)
    if not hist or len(hist) < AVOID_HISTORY:
        return False
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    return (max(xs) - min(xs)) <= 2 and (max(ys) - min(ys)) <= 2


def _avoid_set(wid: str) -> frozenset[tuple[int, int]] | None:
    """Build the avoid-set from the Worker's recent-position history."""
    hist = _pos_history.get(wid)
    if not hist:
        return None
    return frozenset(hist)
# Last known position per Worker and how many consecutive ticks it has been
# unchanged. A Worker stuck in place (boxed in by obstacles + the anti-backtrack
# cell) would WAIT forever; after STUCK_TICKS we reset its exploration state so
# it can pick a fresh direction and escape the deadlock.
_last_pos: dict[str, tuple[int, int]] = {}
_stuck_ticks: dict[str, int] = {}
STUCK_TICKS = 4


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _minimum_assignment(costs: list[list[int]]) -> list[int]:
    """Return the minimum-cost column for each row of a rectangular matrix.

    This is the O(rows^2 * columns) Hungarian algorithm for ``rows <= columns``.
    Iteration order is deterministic, so sorted Workers and coordinates also
    give deterministic tie-breaking without an external optimization package.
    """
    if not costs:
        return []
    row_count = len(costs)
    column_count = len(costs[0])
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment matrix must be rectangular with rows <= columns")

    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    infinity = 10**30

    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        current_column = 0
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = infinity
            next_column = 0
            for column_index in range(1, column_count + 1):
                if used[column_index]:
                    continue
                reduced = (
                    costs[current_row - 1][column_index - 1]
                    - row_potential[current_row]
                    - column_potential[column_index]
                )
                if reduced < minimum[column_index]:
                    minimum[column_index] = reduced
                    predecessor[column_index] = current_column
                if minimum[column_index] < delta:
                    delta = minimum[column_index]
                    next_column = column_index
            for column_index in range(column_count + 1):
                if used[column_index]:
                    row_potential[matched_row[column_index]] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum[column_index] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    result = [-1] * row_count
    for column_index in range(1, column_count + 1):
        if matched_row[column_index] != 0:
            result[matched_row[column_index] - 1] = column_index - 1
    return result


def _cooldown_resource(cell: tuple[int, int], tick: int) -> None:
    """暂时停用不可达提示，但不删除持久地图事实。"""
    hint = _resource_hints.setdefault(cell, ResourceHint(0, "legacy"))
    hint.failure_count = min(hint.failure_count + 1, _RESOURCE_FAILURE_CAP)
    delay = (
        _RESOURCE_COOLDOWN_CAP
        if hint.failure_count >= _RESOURCE_FAILURE_CAP
        else _RESOURCE_COOLDOWN_BASE * 2 ** (hint.failure_count - 1)
    )
    hint.cooldown_until = tick + delay
    hint.source = "history"
    _resource_telemetry["unreachable"] = _resource_telemetry.get("unreachable", 0) + 1
    _mark_persistent_state_dirty()


def _resource_telemetry_summary() -> str:
    """返回不含凭据的紧凑经济指标，供 Tick 日志使用。"""
    keys = (
        ("assignments", "a"),
        ("visible_assignments", "av"),
        ("history_assignments", "ah"),
        ("stale", "stale"),
        ("explore_reserved", "exp"),
        ("blocked", "blk"),
        ("cooled", "cool"),
        ("unreachable", "unr"),
        ("harvested", "harv"),
        ("deposited", "dep"),
    )
    return ",".join(
        f"{short}{_resource_telemetry.get(name, 0)}" for name, short in keys
    )


def _resource_age(cell: tuple[int, int], tick: int) -> int:
    """返回历史资源提示的逻辑 Tick 年龄。"""
    hint = _resource_hints.get(cell)
    if hint is None:
        return max(0, tick)
    return max(0, tick - hint.last_confirmed_tick)


def _worker_resource_assignments(
    turn: "Turn",
    blocked_resources: frozenset[tuple[int, int]] = frozenset(),
) -> dict[str, tuple[int, int]]:
    """Globally match empty Workers to all trustworthy known resources.

    The lexicographic objective is: assign as many resources as possible,
    prefer currently visible cells when resources outnumber Workers, then
    minimize total Manhattan travel. The input collections are sorted so Unit
    order in the authoritative state cannot change the result.
    """
    workers = sorted(
        (
            worker
            for worker in turn.workers
            if worker.cargo == 0
            and not (
                turn.beacon.status == BeaconStatus.GROUND
                and turn.beacon.position == worker.position
            )
        ),
        key=lambda worker: str(worker.id),
    )
    visible_resources = set(turn.resource_cells)
    all_resources = _known_resources | visible_resources
    cooled_resources = {
        resource
        for resource in all_resources - visible_resources
        if _resource_hints.setdefault(resource, ResourceHint(0, "legacy")).cooldown_until
        > turn.tick
    }
    stale_resources = {
        resource
        for resource in all_resources - visible_resources - cooled_resources
        if _resource_age(resource, turn.tick) > _MAX_HISTORY_RESOURCE_AGE
    }
    eligible_resources = all_resources - cooled_resources - stale_resources
    _resource_telemetry.update(
        {
            "stale": len(stale_resources),
            "explore_reserved": 0,
        }
    )
    fixed: dict[str, tuple[int, int]] = {}
    # A Worker already standing on a currently visible resource is the highest
    # confidence assignment. Pin it before the global matcher so history
    # targets, stale age penalties, or dynamic blockers can never pull that
    # Worker away and waste the harvest window.
    fixed_resources = eligible_resources & (visible_resources | blocked_resources)
    for resource in sorted(fixed_resources):
        occupant = next(
            (worker for worker in workers if tuple(worker.position) == resource),
            None,
        )
        if occupant is not None:
            fixed[str(occupant.id)] = resource
    fixed_worker_ids = set(fixed)
    workers = [worker for worker in workers if str(worker.id) not in fixed_worker_ids]
    resources = sorted(eligible_resources - blocked_resources - set(fixed.values()))
    _resource_telemetry.update(
        {
            "blocked": len((all_resources & blocked_resources) - set(fixed.values())),
            "cooled": len(cooled_resources),
        }
    )
    # 只要候选里混有历史提示，就保留一个没有站在可见资源上的 Worker 做前沿
    # 扫描。这样历史点不能在可见资源出现时把全队锁死，同时站在可见资源格的
    # Worker 仍会优先采集。
    dispatch_workers = workers
    history_resources = set(resources) - visible_resources
    explorer_candidates = [
        worker for worker in workers if tuple(worker.position) not in visible_resources
    ]
    if (
        history_resources
        and len(workers) >= 2
        and len(visible_resources) < len(workers)
        and explorer_candidates
    ):
        core_pos = turn.core.position if turn.core is not None else (0, 0)
        visible_targets = visible_resources & set(resources)
        priority_targets = visible_targets or set(resources)

        def explorer_score(worker: "Worker") -> tuple[int, int, str]:
            # 有可见资源时保护其最近采集者；只有历史提示时，则保护离历史
            # 资源最近的 Worker，避免把已经走到节点旁的采集者派去探索。
            resource_distance = min(
                (_manhattan(worker.position, resource) for resource in priority_targets),
                default=0,
            )
            return (
                resource_distance,
                _manhattan(worker.position, core_pos),
                str(worker.id),
            )

        explorer = max(
            explorer_candidates,
            key=explorer_score,
        )
        dispatch_workers = [worker for worker in workers if worker is not explorer]
        _resource_telemetry["explore_reserved"] = 1

    if not dispatch_workers or not resources:
        assignments = fixed
        _resource_telemetry.update(
            {
                "assignments": len(assignments),
                "visible_assignments": sum(
                    target in visible_resources for target in assignments.values()
                ),
                "history_assignments": sum(
                    target not in visible_resources for target in assignments.values()
                ),
            }
        )
        return assignments

    max_distance = max(
        _manhattan(worker.position, resource)
        for worker in dispatch_workers
        for resource in resources
    )
    matched_count = min(len(dispatch_workers), len(resources))
    # One non-visible assignment must cost more than every possible total
    # distance difference, making visible-resource coverage a strict priority.
    history_penalty = max_distance * matched_count + 1

    if len(dispatch_workers) <= len(resources):
        costs = [
            [
                _manhattan(worker.position, resource)
                + (
                    0
                    if resource in visible_resources
                    else history_penalty
                    + _resource_age(resource, turn.tick) * _HISTORY_RESOURCE_AGE_WEIGHT
                )
                for resource in resources
            ]
            for worker in dispatch_workers
        ]
        columns = _minimum_assignment(costs)
        assignments = fixed | {
            str(worker.id): resources[column]
            for worker, column in zip(dispatch_workers, columns, strict=True)
        }
    else:
        # Worker 多于资源时，每个资源都能分配；可见资源已全部覆盖，
        # 此时只需继续最小化总路程。
        costs = [
            [
                _manhattan(resource, worker.position)
                + (
                    0
                    if resource in visible_resources
                    else history_penalty
                    + _resource_age(resource, turn.tick) * _HISTORY_RESOURCE_AGE_WEIGHT
                )
                for worker in dispatch_workers
            ]
            for resource in resources
        ]
        columns = _minimum_assignment(costs)
        assignments = fixed | {
            str(dispatch_workers[column].id): resource
            for resource, column in zip(resources, columns, strict=True)
        }
    _resource_telemetry.update(
        {
            "assignments": len(assignments),
            "visible_assignments": sum(
                target in visible_resources for target in assignments.values()
            ),
            "history_assignments": sum(
                target not in visible_resources for target in assignments.values()
            ),
        }
    )
    return assignments


def _frontier_gain(
    target: tuple[int, int], blocked: frozenset[tuple[int, int]]
) -> int:
    """Count still-unknown cells a Worker could reveal near ``target``."""
    tx, ty = target
    return sum(
        1
        for dx in range(-3, 4)
        for dy in range(-(3 - abs(dx)), 3 - abs(dx) + 1)
        if (tx + dx, ty + dy) not in _explored_cells
        and (tx + dx, ty + dy) not in blocked
    )


def _cooldown_explore_target(target: tuple[int, int], tick: int) -> None:
    """Temporarily remove a frontier target that A* proved unreachable."""
    failures = min(_explore_target_failures.get(target, 0) + 1, 5)
    _explore_target_failures[target] = failures
    delay = min(
        _EXPLORE_COOLDOWN_CAP,
        _EXPLORE_COOLDOWN_BASE * 2 ** (failures - 1),
    )
    _explore_target_cooldown_until[target] = tick + delay


def _prune_explore_target_cooldowns(tick: int) -> None:
    """Drop expired frontier cooldowns so the intent cache stays bounded."""
    for target, until in list(_explore_target_cooldown_until.items()):
        if until <= tick:
            del _explore_target_cooldown_until[target]


def _explore_target_is_cooled(target: tuple[int, int], tick: int) -> bool:
    """Return whether a frontier target is still in its retry cooldown."""
    return _explore_target_cooldown_until.get(target, 0) > tick


def _frontier_candidates(
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    tick: int | None = None,
    radius: int | None = None,
) -> list[tuple[int, int]]:
    """Build unknown boundary cells, with a radial cold-start fallback."""
    effective_tick = 0 if tick is None else tick
    radius_limit = MAX_SWEEP_RADIUS if radius is None else radius
    _prune_explore_target_cooldowns(effective_tick)

    def eligible(cell: tuple[int, int]) -> bool:
        return (
            cell not in blocked
            and cell not in _explored_cells
            and not _explore_target_is_cooled(cell, effective_tick)
        )

    candidates: set[tuple[int, int]] = set()
    for x, y in _explored_cells:
        for direction in DIRECTIONS:
            dx, dy = direction.delta
            cell = (x + dx, y + dy)
            if (
                eligible(cell)
                and cell != core_pos
                and _manhattan(cell, core_pos) <= radius_limit
            ):
                candidates.add(cell)
    if not candidates:
        cx, cy = core_pos
        radius = 12
        candidates.update(
            {
                (cx + radius, cy),
                (cx - radius, cy),
                (cx, cy + radius),
                (cx, cy - radius),
                (cx + radius // 2, cy + radius // 2),
                (cx + radius // 2, cy - radius // 2),
                (cx - radius // 2, cy + radius // 2),
                (cx - radius // 2, cy - radius // 2),
            }
        )
        candidates = {
            cell for cell in candidates
            if eligible(cell) and cell != core_pos
        }
    return sorted(candidates)


def _explore_sector(
    cell: tuple[int, int], core_pos: tuple[int, int]
) -> tuple[int, int]:
    """Return the signed compass sector of a frontier cell.

    Eight sectors (including the cardinal axes) are enough to prevent several
    idle Workers from selecting the same side of the frontier while keeping
    the assignment deterministic and independent of map iteration order.
    """
    dx = cell[0] - core_pos[0]
    dy = cell[1] - core_pos[1]
    return (
        1 if dx > 0 else -1 if dx < 0 else 0,
        1 if dy > 0 else -1 if dy < 0 else 0,
    )


def _assign_explore_targets(
    workers: list[tuple[str, tuple[int, int]]],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    tick: int = 0,
    radius: int | None = None,
) -> dict[str, tuple[int, int]]:
    """Assign stable, mutually separated frontier targets to idle Workers."""
    radius_limit = _exploration_radius() if radius is None else radius
    _prune_explore_target_cooldowns(tick)
    live_ids = {worker_id for worker_id, _ in workers}
    for worker_id in list(_explore_targets):
        target = _explore_targets[worker_id]
        if (
            worker_id not in live_ids
            or target in blocked
            or _explore_target_is_cooled(target, tick)
            or _frontier_gain(target, blocked) == 0
            or _manhattan(target, core_pos) > radius_limit
        ):
            _explore_progress.pop(worker_id, None)
            del _explore_targets[worker_id]

    candidates = _frontier_candidates(
        core_pos, blocked, tick=tick, radius=radius_limit
    )
    selected = list(_explore_targets.values())
    sector_counts: Counter[tuple[int, int]] = Counter(
        _explore_sector(target, core_pos) for target in selected
    )
    result: dict[str, tuple[int, int]] = {}
    for worker_id, position in sorted(workers):
        existing = _explore_targets.get(worker_id)
        if existing is not None:
            result[worker_id] = existing
            continue
        available = [candidate for candidate in candidates if candidate not in selected]
        if not available:
            break

        def score(candidate: tuple[int, int]) -> tuple[int, int, int, int, int, int]:
            separation = (
                min(_manhattan(candidate, other) for other in selected)
                if selected
                else _manhattan(candidate, core_pos)
            )
            # Prefer an unexplored direction before taking another target on a
            # saturated side.  The old score only maximized local frontier gain,
            # so a dense east edge could claim every idle Worker and recreate
            # the user's observed southeast cluster.
            sector = _explore_sector(candidate, core_pos)
            return (
                -sector_counts.get(sector, 0),
                _frontier_gain(candidate, blocked),
                separation,
                -_manhattan(position, candidate),
                -candidate[0],
                -candidate[1],
            )

        target = max(available, key=score)
        _explore_targets[worker_id] = target
        result[worker_id] = target
        selected.append(target)
        sector_counts[_explore_sector(target, core_pos)] += 1
    return result


def _explore_target_has_stalled(
    worker_id: str,
    position: tuple[int, int],
    target: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
) -> bool:
    """Detect repeated frontier ticks without distance or discovery progress."""
    distance = _manhattan(position, target)
    frontier_gain = _frontier_gain(target, blocked)
    previous = _explore_progress.get(worker_id)
    if previous is None or previous.target != target:
        stalled_ticks = 0
    else:
        progressed = (
            distance < previous.distance
            or frontier_gain < previous.frontier_gain
        )
        stalled_ticks = 0 if progressed else previous.stalled_ticks + 1
    _explore_progress[worker_id] = ExploreProgress(
        target=target,
        position=position,
        distance=distance,
        frontier_gain=frontier_gain,
        stalled_ticks=stalled_ticks,
    )
    return stalled_ticks >= _EXPLORE_STALL_TICKS


def _same_fire_line(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two cells share a cardinal or exact-diagonal fire line."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx == 0 and dy > 0) or (dy == 0 and dx > 0) or dx == dy


def _obstacles_between(
    a: tuple[int, int], b: tuple[int, int], obstacles: frozenset[tuple[int, int]]
) -> bool:
    """True if an obstacle lies between cells on a legal Ranger fire line.

    Only obstacles in the cardinal or exact-diagonal shot cells block Ranger
    fire. Obstacles beside an exact diagonal do not block it.
    """
    if not _same_fire_line(a, b):
        return True
    if a == b:
        return False
    ax, ay = a
    bx, by = b
    dx = 0 if ax == bx else (1 if bx > ax else -1)
    dy = 0 if ay == by else (1 if by > ay else -1)
    distance = max(abs(bx - ax), abs(by - ay))
    return any(
        (ax + dx * step, ay + dy * step) in obstacles
        for step in range(1, distance)
    )


def _step_toward(
    start: tuple[int, int],
    target: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    avoid: frozenset[tuple[int, int]] | None = None,
) -> Direction | None:
    """Pick a single cardinal step toward ``target`` that avoids blocked cells.

    Prefers the axis with the larger remaining gap so progress is monotonic on
    open ground. Ties break to the x-axis. Falls back to a detour step when
    the direct route is blocked, so a Worker can route around an obstacle.

    ``avoid`` is a set of cells the Worker recently left (its position
    history); a step that would return to ANY of them is skipped unless it is
    the only open cell, so the Worker does not re-enter a cell it visited
    recently and close a multi-cell backtracking cycle (A->B->C->D->A).
    """
    sx, sy = start
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    options: list[Direction] = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            options.append(Direction.RIGHT)
        elif dx < 0:
            options.append(Direction.LEFT)
        if dy > 0:
            options.append(Direction.DOWN)
        elif dy < 0:
            options.append(Direction.UP)
    else:
        if dy > 0:
            options.append(Direction.DOWN)
        elif dy < 0:
            options.append(Direction.UP)
        if dx > 0:
            options.append(Direction.RIGHT)
        elif dx < 0:
            options.append(Direction.LEFT)
    for direction in options:
        ddx, ddy = direction.delta
        nxt = (sx + ddx, sy + ddy)
        if nxt in blocked:
            continue
        if avoid is not None and nxt in avoid:
            continue
        return direction
    # The reducing steps were all blocked. Detour: try any open neighbor even
    # if it does not strictly reduce distance, so a Worker can route around an
    # obstacle instead of stalling. Prefer steps on the perpendicular axis,
    # then any remaining open cell.
    detour: list[Direction] = []
    perp = (Direction.UP, Direction.DOWN) if abs(dx) >= abs(dy) else (Direction.LEFT, Direction.RIGHT)
    for direction in (*perp, *DIRECTIONS):
        if direction not in detour:
            detour.append(direction)
    for direction in detour:
        ddx, ddy = direction.delta
        nxt = (sx + ddx, sy + ddy)
        if nxt in blocked:
            continue
        if avoid is not None and nxt in avoid:
            continue
        return direction
    # Genuinely boxed in: WAIT rather than step back into the cell just left,
    # which would 2-cycle and waste every tick. An obstacle or enemy may clear
    # next tick, letting progress resume.
    return None


def _astar_step_result(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: frozenset[tuple[int, int]],
    blocked: frozenset[tuple[int, int]],
    max_expansions: int = 4000,
    allow_blocked_goal: bool = False,
) -> tuple[Direction | None, bool]:
    """Return the first cardinal step on an A* path from ``start`` to ``goal``.

    The greedy _step_toward wedges in obstacle-dense terrain: a laden Worker
    returning to a Core parked in a stone corner can spin 20+ cells out, never
    finding the gap (observed 67512f/d48045). A* guarantees a path when one
    exists. ``obstacles`` are permanent terrain; ``blocked`` are dynamic
    (friendly-full/enemy cells). ``allow_blocked_goal`` is reserved for a laden
    Worker entering its own Core cell to deposit.

    返回值的第二项表示搜索是否仅因展开预算耗尽而停止；这与已确认
    开放集耗尽的“不可达”不同。
    """
    if start == goal:
        return None, False
    frontier: list[tuple[int, int, int, tuple[int, int]]] = [
        (_manhattan(start, goal), 0, 0, start)
    ]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    best_cost: dict[tuple[int, int], int] = {start: 0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _, cost, _, current = heapq.heappop(frontier)
        if cost != best_cost.get(current):
            continue
        expansions += 1
        if current == goal:
            break
        for d in DIRECTIONS:
            ddx, ddy = d.delta
            nxt = (current[0] + ddx, current[1] + ddy)
            if nxt in obstacles or (
                nxt in blocked and not (allow_blocked_goal and nxt == goal)
            ):
                continue
            new_cost = cost + 1
            if new_cost >= best_cost.get(nxt, 10**9):
                continue
            best_cost[nxt] = new_cost
            came_from[nxt] = current
            heapq.heappush(
                frontier,
                (new_cost + _manhattan(nxt, goal), new_cost, expansions, nxt),
            )
    if goal not in came_from:
        return None, bool(frontier) and expansions >= max_expansions
    cursor = goal
    while came_from.get(cursor) != start:
        parent = came_from.get(cursor)
        if parent is None:
            return None, False
        cursor = parent
    ddx = cursor[0] - start[0]
    ddy = cursor[1] - start[1]
    for d in DIRECTIONS:
        if d.delta == (ddx, ddy):
            return d, False
    return None, False


def _astar_step(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: frozenset[tuple[int, int]],
    blocked: frozenset[tuple[int, int]],
    max_expansions: int = 4000,
    allow_blocked_goal: bool = False,
) -> Direction | None:
    """返回 A* 路径第一步；保留现有调用方的简单接口。"""
    step, _ = _astar_step_result(
        start,
        goal,
        obstacles,
        blocked,
        max_expansions=max_expansions,
        allow_blocked_goal=allow_blocked_goal,
    )
    return step


def _step_away_from(
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    avoid: frozenset[tuple[int, int]] | None = None,
) -> Direction | None:
    """Pick a cardinal step that moves AWAY from ``core_pos``.

    Used to disperse laden Workers when the Core is at full resource capacity:
    they cannot deposit (CORE_RESOURCE_FULL), so clustering at home only fills
    every Core-adjacent cell to 2/2 and deadlocks the Core's spawn. Drifting
    outward clears the cell so a spawn can spend resources and open room for
    the deposit next Tick. Prefers the step that maximizes the gain in
    Manhattan distance; falls back to any open cell when none increases it.
    """
    from_here = _manhattan(pos, core_pos)
    best_dir: Direction | None = None
    best_gain = -1
    for d in DIRECTIONS:
        ddx, ddy = d.delta
        nxt = (pos[0] + ddx, pos[1] + ddy)
        if nxt in blocked:
            continue
        if avoid is not None and nxt in avoid:
            continue
        gain = _manhattan(nxt, core_pos) - from_here
        if gain > best_gain:
            best_gain = gain
            best_dir = d
    if best_dir is not None:
        return best_dir
    # Nothing strictly increases distance (ring of obstacles): take any open
    # cell rather than stall — lateral drift still thins the cluster.
    for d in DIRECTIONS:
        ddx, ddy = d.delta
        nxt = (pos[0] + ddx, pos[1] + ddy)
        if nxt in blocked:
            continue
        if avoid is not None and nxt in avoid:
            continue
        return d
    return None


def _step_direction(direction: Direction, pos: tuple[int, int], blocked: frozenset[tuple[int, int]]) -> Direction | None:
    """Try to take one step in ``direction``; detour around an obstacle if blocked."""
    ddx, ddy = direction.delta
    if (pos[0] + ddx, pos[1] + ddy) not in blocked:
        return direction
    # Blocked: try the two perpendicular directions, then the reverse, so the
    # Worker skirts the obstacle instead of stalling.
    perp = (Direction.UP, Direction.DOWN) if direction in (Direction.LEFT, Direction.RIGHT) else (Direction.LEFT, Direction.RIGHT)
    for alt in (*perp, _TURN_ORDER.get(direction, direction)):
        adx, ady = alt.delta
        if (pos[0] + adx, pos[1] + ady) not in blocked:
            return alt
    return None


def _chunk_origin(core_pos: tuple[int, int]) -> tuple[int, int]:
    """North-west corner of the chunk containing ``core_pos`` (32x32 grid)."""
    return (core_pos[0] // 32 * 32, core_pos[1] // 32 * 32)


def _worker_column(index: int, fleet_size: int, core_pos: tuple[int, int]) -> int:
    """Chunk-relative column assigned by Worker index, spanning the full chunk.

    The third multi-perspective review found that Core-anchored bands blind the
    half of the chunk opposite the Core's offset within its chunk (the Core at
    (181,149) sits at chunk offset (21,21) -- the SE quadrant -- so a symmetric
    Core-centered sweep never sees the NW half). Anchoring columns to the CHUNK
    guarantees full-width coverage regardless of where the Core sits. With
    vision radius 3, evenly-spaced columns tile the 32-cell width with overlap.
    """
    chunk_x0, _ = _chunk_origin(core_pos)
    if fleet_size <= 0:
        return chunk_x0 + 16
    # Evenly distribute columns across the full chunk width [chunk_x0+1, chunk_x0+30].
    # The outermost workers sit near the chunk edges so their vision (radius 3)
    # covers x=0 and x=31.
    return chunk_x0 + 1 + int((index) * 29 / max(fleet_size - 1, 1)) if fleet_size > 1 else chunk_x0 + 16


def _step_keep_col(
    direction: Direction,
    pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    target_col: int,
    avoid: frozenset[tuple[int, int]] | None = None,
) -> Direction | None:
    """Step toward ``target_col`` without detouring off the column.

    If the horizontal step is blocked, detour vertically (which keeps progress
    along the column) rather than perpendicular-horizontal. ``avoid`` skips the
    cell just left so the vertical detour does not 2-cycle back into it.

    Fourth-review fix: the original only tried the primary horizontal step and
    the two vertical perpendiculars, then WAITed. A Worker caught in a 3-walled
    pocket one step short of its column (both perpendiculars are permanent
    obstacles) would WAIT forever — the STUCK_TICKS reset re-initialized the
    same state and re-deadlocked because the obstacle geometry is permanent.
    3 of 7 Workers were lost this way (cbf157 stuck 2780 ticks, 3e47b8 1457
    ticks). Now, after the vertical detour fails, try the REVERSE horizontal
    step (step away from the column to escape the pocket) and then any open
    neighbor, so a Worker can break out of a pocket instead of stalling.
    """
    ddx, ddy = direction.delta
    nxt = (pos[0] + ddx, pos[1] + ddy)
    if nxt not in blocked and (avoid is None or nxt not in avoid):
        return direction
    for alt in (Direction.UP, Direction.DOWN):
        adx, ady = alt.delta
        anxt = (pos[0] + adx, pos[1] + ady)
        if anxt not in blocked and (avoid is None or anxt not in avoid):
            return alt
    # Vertical detour blocked: try the REVERSE horizontal step to escape the
    # pocket (stepping away from the target column is better than WAITing
    # forever; the Worker re-approaches on a later tick via a different row).
    rev = _TURN_ORDER.get(_TURN_ORDER.get(direction, direction), direction)
    rdx, rdy = rev.delta
    rnxt = (pos[0] + rdx, pos[1] + rdy)
    if rnxt not in blocked and (avoid is None or rnxt not in avoid):
        return rev
    # Last resort: any open neighbor at all, so a boxed-in Worker still moves
    # rather than 2-cycle or stall. Prefer non-avoid cells.
    for cand in DIRECTIONS:
        cdx, cdy = cand.delta
        cnxt = (pos[0] + cdx, pos[1] + cdy)
        if cnxt not in blocked and (avoid is None or cnxt not in avoid):
            return cand
    return None


def _explore_step(
    worker_index: int,
    worker_id: str,
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    target_col: int | None = None,
    avoid: frozenset[tuple[int, int]] | None = None,
    force_band: int | None = None,
    fleet_size: int = 1,
    sweep_radius: int | None = None,
) -> Direction | None:
    """Return one chunk-anchored boustrophedon step for a Worker.

    The Worker owns a vertical column (an x coordinate) anchored to the CHUNK,
    not the Core, and sweeps it north/south edge-to-edge across the chunk,
    shifting one column at the y-boundary. The third review found that
    Core-anchored bands blind the chunk half opposite the Core's offset; chunk
    anchoring guarantees full-width coverage regardless of Core position.
    Turning is dictated by ABSOLUTE chunk-y position. State:
    [column_offset, going_south_flag].

    ``target_col`` overrides the Worker's column when set (chemotaxis).
    ``force_band`` overrides the chunk column assignment.
    """
    chunk_x0, chunk_y0 = _chunk_origin(core_pos)
    chunk_y1 = chunk_y0 + CHUNK_SIZE - 1
    # Extend the sweep beyond the home chunk so edge workers tile the reachable
    # aprons of neighboring chunks when no frontier target is available.
    # Half-zone bias: even-index workers sweep the NORTH half mainly, odd the
    # SOUTH half, so both halves stay covered. Without it, workers returning
    # to deposit cluster south of the Core (Core y=234 is near the chunk's
    # north edge, and the south apron is deeper), leaving north-refilled
    # resources undiscovered (observed: 13/15 workers south, north nearly
    # empty).
    if worker_index % 2 == 0:
        sweep_y_lo = chunk_y0 - 12
        sweep_y_hi = core_pos[1] + 10   # north half + a little past the midline
    else:
        sweep_y_lo = core_pos[1] - 10
        sweep_y_hi = chunk_y1 + 12      # south half + a little past the midline
    # Fourth-review fix: clamp the Worker's target column to the home chunk
    # x-range. _advance_col_off wraps col_off within [-_SWEEP_COL_WRAP,
    # +_SWEEP_COL_WRAP], which for an edge-assigned Worker (base_col near a
    # chunk boundary) produces target_x OUTSIDE the home chunk (095d133 drifted
    # to x=-43, 11 cells into the neighboring poor chunk). Clamp here so a
    # Worker never marches off the chunk it is supposed to scan.
    chunk_x_lo = chunk_x0 - 12
    chunk_x_hi = chunk_x0 + CHUNK_SIZE + 12 - 1
    # The Worker's base column is chunk-relative (full-width coverage); the
    # mutable col_off steps it sideways at each y-boundary to tile the chunk.
    if target_col is not None:
        base_col = target_col
    elif force_band is not None:
        base_col = core_pos[0] + force_band
    else:
        base_col = _worker_column(worker_index, fleet_size, core_pos)

    state = _explore_state.get(worker_id)
    if state is None or len(state) < 2:
        # Half-zone initial direction: even-index workers start NORTH (matching
        # their north-half sweep band), odd start south. This is load-bearing —
        # starting even workers south (the old cov-3) sent them into the south
        # apron first and north coverage never happened (observed 13-14/15
        # workers stuck south, north half nearly empty).
        south_init = 0 if worker_index % 2 == 0 else 1
        state = [0, south_init, None, None]
        _explore_state[worker_id] = state
    # Sweep-radius cap: a worker must not wander beyond the current discovery
    # radius. Empty workers may expand this radius during a resource drought;
    # laden workers never call this path unless the Core is full and use the
    # fixed default instead.
    # Core (the deposit sink). Past ~40 cells the delivery round trip is a net
    # loss AND the worker never returns — observed workers stranded 60-88 cells
    # out (x 30-44, y 276-297) while north resources near the Core sat
    # uncollected. Steer back toward the Core until back in range.
    # 9th review rank 3: use A* (same laden-return pattern). Keep the
    # anti-backtrack history while returning: when a visible enemy or obstacle
    # changes the locally preferred A* branch, clearing history every Tick can
    # make the Worker oscillate between the same two cells outside the scan
    # radius.
    radius_limit = MAX_SWEEP_RADIUS if sweep_radius is None else sweep_radius
    if _manhattan(pos, core_pos) > radius_limit:
        step = _astar_step(pos, core_pos, blocked, blocked)
        avoid = _avoid_set(worker_id)
        if step is not None and avoid is not None:
            ddx, ddy = step.delta
            next_cell = (pos[0] + ddx, pos[1] + ddy)
            if next_cell in avoid:
                step = _step_toward(pos, core_pos, blocked, avoid=avoid)
        if step is None:
            step = _step_toward(pos, core_pos, blocked, avoid=avoid)
        if step is None:
            step = _step_toward(pos, core_pos, blocked)
        if step is not None:
            return step
    col_off, south = state[0], state[1]
    target_x = max(chunk_x_lo, min(chunk_x_hi, base_col + col_off))

    # First reach the assigned column, detouring vertically (never off-column)
    # if the horizontal step is blocked, and never stepping back onto the cell
    # just left (which would 2-cycle the vertical detour).
    if pos[0] != target_x:
        primary = Direction.RIGHT if target_x > pos[0] else Direction.LEFT
        return _step_keep_col(primary, pos, blocked, target_col=target_x, avoid=avoid)

    # On the column: march north/south to the CHUNK edge (not Core-relative),
    # then step the column and reverse.
    if south and pos[1] >= sweep_y_hi:
        south = 0
        col_off = _next_explore_col_off(col_off, base_col, sweep_y_lo, sweep_y_hi, chunk_x_lo, chunk_x_hi)
    elif not south and pos[1] <= sweep_y_lo:
        south = 1
        col_off = _next_explore_col_off(col_off, base_col, sweep_y_lo, sweep_y_hi, chunk_x_lo, chunk_x_hi)

    _explore_state[worker_id] = [col_off, south, None, None]
    new_target_x = max(chunk_x_lo, min(chunk_x_hi, base_col + col_off))
    # If we just advanced the column, step horizontally onto it.
    if pos[0] != new_target_x:
        primary = Direction.RIGHT if new_target_x > pos[0] else Direction.LEFT
        return _step_keep_col(primary, pos, blocked, target_col=new_target_x, avoid=avoid)
    # March north/south. If the march cell is blocked, ADVANCE the column
    # (wrap) and step onto it — do NOT silently flip south, which traps the
    # Worker in a wall-bounce between two one-sided blockers. Only fall back to
    # the opposite vertical march if the new column is also blocked.
    march = Direction.DOWN if south else Direction.UP
    mdx, mdy = march.delta
    if (pos[0] + mdx, pos[1] + mdy) not in blocked and (
        avoid is None or (pos[0] + mdx, pos[1] + mdy) not in avoid
    ):
        return march
    # March blocked (or would backtrack): advance the column and step onto it.
    col_off = _advance_col_off(col_off)
    _explore_state[worker_id] = [col_off, south, None, None]
    target_col_now = base_col + col_off
    if pos[0] != target_col_now:
        primary = Direction.RIGHT if target_col_now > pos[0] else Direction.LEFT
        step = _step_keep_col(primary, pos, blocked, target_col=target_col_now, avoid=avoid)
        if step is not None:
            return step
    # New column also unreachable: try the opposite vertical march as a last
    # resort, skipping the avoided cell (wait rather than 2-cycle).
    opp = Direction.UP if south else Direction.DOWN
    odx, ody = opp.delta
    onxt = (pos[0] + odx, pos[1] + ody)
    if onxt not in blocked and (avoid is None or onxt not in avoid):
        _explore_state[worker_id] = [col_off, 0 if south else 1, None, None]
        return opp
    return None


def _begin_outbound(worker_id: str, worker_index: int, pos: tuple[int, int], core_pos: tuple[int, int]) -> None:
    """Resume a Worker's band sweep after a deposit, clearing any harvest lock.

    Preserve the column offset so the Worker resumes in the outer band it had
    reached rather than re-scanning the empty inner columns from scratch.
    """
    state = _explore_state.get(worker_id)
    col_off = state[0] if state is not None and len(state) >= 1 else 0
    # Resume toward the worker's half zone: even -> north, odd -> south (must
    # match the south_init in _explore_step, or a deposit pulls the worker
    # south again and north coverage collapses).
    south = 1 if worker_index % 2 == 1 else 0
    _explore_state[worker_id] = [col_off, south, None, None]


def _select_ranger_target(
    ranger_pos: tuple[int, int],
    enemies: tuple[UnitView | CoreView, ...],
    obstacles: frozenset[tuple[int, int]],
    core_pos: tuple[int, int],
) -> UnitView | CoreView | None:
    """Choose a visible enemy the Ranger can legally shoot this Tick.

    Rules: shared cardinal or exact-diagonal line, range 1-3, no obstacle strictly
    between. An enemy within two cells of our Core is prioritized before a
    distant enemy Core: keeping the stored economy alive dominates a speculative
    raid. Among equally urgent targets, enemy Cores still win because destroying
    one removes the enemy fleet and can capture its stockpiled resources
    (variable loot, not a flat +6; see ``CORE_RESOURCES_CAPTURED``). Units then
    prefer a one-shot-killable (hp==1) and a FLEEING target (farther from the
    Core than its last-known position) so driven-off raiders are finished, not
    let to escape (8th review, rank 3).
    """
    best: UnitView | CoreView | None = None
    best_key: tuple[int, int, int, int, int, str] | None = None
    for enemy in enemies:
        cell = enemy.position
        if not _same_fire_line(ranger_pos, cell):
            continue
        dist = max(abs(ranger_pos[0] - cell[0]), abs(ranger_pos[1] - cell[1]))
        if dist < 1 or dist > RANGER_MAX_RANGE:
            continue
        if _obstacles_between(ranger_pos, cell, obstacles):
            continue
        is_core = enemy.kind == "CORE"
        hp = getattr(enemy, "hp", None)
        finishable = 0 if hp == 1 else 1
        last = _last_enemy_pos.get(str(enemy.id))
        fleeing = 1  # lowest = best here (0 beats 1)
        if last is not None:
            last_pos, _ = last
            if _manhattan(cell, core_pos) > _manhattan(last_pos, core_pos):
                fleeing = 0  # moving away from the Core = escaping
        key = (
            0 if _manhattan(cell, core_pos) <= 2 else 1,
            0 if is_core else 1,
            finishable,
            fleeing,
            dist,
            str(enemy.id),
        )
        if best_key is None or key < best_key:
            best, best_key = enemy, key
    return best


def _vanguard_sweep_target(
    vanguard_pos: tuple[int, int],
    enemies: tuple[UnitView | CoreView, ...],
) -> Direction | None:
    """Sweep the adjacent cell holding the most enemy objects.

    A sweep hits every enemy Unit and any enemy Core in the one adjacent cell,
    so a contested adjacent cell is the best use of the action.
    """
    best_dir: Direction | None = None
    best_hits = 0
    for direction in DIRECTIONS:
        ddx, ddy = direction.delta
        cell = (vanguard_pos[0] + ddx, vanguard_pos[1] + ddy)
        hits = sum(enemy.position == cell for enemy in enemies)
        if hits > best_hits or (
            hits == best_hits and hits > 0 and best_dir is not None and direction < best_dir
        ):
            best_dir, best_hits = direction, hits
    return best_dir if best_hits > 0 else None


def _threats_to_core(
    core_pos: tuple[int, int],
    enemies: tuple[UnitView | CoreView, ...],
) -> list[UnitView | CoreView]:
    """Visible enemies within raiding range of the Core are treated as threats.

    A Ranger can hit the Core or its Workers from range 3, and a raiding party
    often sits a few cells out before striking. Treating anything within 6 as a
    threat lets the Core start spawning defenders and repairing before the
    enemy is on the Core cell.
    """
    return [e for e in enemies if _manhattan(core_pos, e.position) <= 6]


def _control_workers(turn: "Turn", core_pos: tuple[int, int]) -> None:
    resource_cells = turn.resource_cells
    explore_radius = _exploration_radius()
    # Base blocked set: obstacle terrain plus visible enemy Core/Unit cells.
    # A Worker that steps onto an enemy cell fails with MOVE_DESTINATION_OCCUPIED
    # every tick and deadlocks, so enemy positions must be routed around.
    # Merge persistent obstacle memory so pathing avoids walls seen in the
    # past as well as the current view (obstacles are static terrain).
    base_blocked = frozenset(turn.obstacle_cells) | _known_obstacles | frozenset(
        e.position for e in turn.visible_enemies
    )
    # Fourth-review fix: do NOT blanket-block every friendly Unit cell. The
    # rules (game-rules.md) say a cell holds up to two occupying entities and
    # same-player objects may co-occupy; the prior code treated every friendly
    # cell as impassable, forcing laden Workers returning home to detour around
    # outbound Workers and contributing to the Core-proximity orbit. Only block
    # a friendly cell that is genuinely FULL (2 occupants), so a Worker can
    # path through a singly-occupied friendly cell as the rules allow.
    from collections import Counter as _Counter
    friendly_occupancy: dict[tuple[int, int], int] = _Counter(
        tuple(u.position) for u in turn.units
    )
    friendly_full = frozenset(
        cell for cell, count in friendly_occupancy.items() if count >= 2
    )
    dynamically_blocked = base_blocked | friendly_full
    resource_assignments = _worker_resource_assignments(
        turn, blocked_resources=dynamically_blocked
    )
    for worker_id in resource_assignments:
        _explore_targets.pop(worker_id, None)
        _explore_progress.pop(worker_id, None)
    idle_workers = [
        (str(worker.id), tuple(worker.position))
        for worker in turn.workers
        if worker.cargo == 0
        and str(worker.id) not in resource_assignments
        and not (
            turn.beacon.status == BeaconStatus.GROUND
            and turn.beacon.position == worker.position
        )
    ]
    # 保留探索名额是最终控制器的行为，而不只是资源分配器的中间状态。
    # 当没有可见资源且至少有三个空闲 Worker 时，即使历史提示全部陈旧，
    # 这些 Worker 仍会进入前沿扫描；在此处再次记录，避免提前返回或遥测
    # 汇总把“正在探索”误报成没有探索保留。
    if not resource_cells and len(idle_workers) >= 3:
        _resource_telemetry["explore_reserved"] = 1
    explore_targets = _assign_explore_targets(
        idle_workers, core_pos, dynamically_blocked, tick=turn.tick
    )
    fallback_workers = list(idle_workers)
    sorted_workers = sorted(turn.workers, key=lambda worker: str(worker.id))
    for orig_index, worker in enumerate(sorted_workers):
        pos = worker.position
        wid = str(worker.id)
        # An EMPTY Worker standing on a visible resource cell harvests
        # IMMEDIATELY, before the boxed-in/STUCK logic that could otherwise
        # shuttle it away in a move (observed: worker 68a41e parked on
        # (11,247) — a visible resource — yet got boxed-escaped into a move
        # instead of harvesting, and drifted off uncollected).
        if (
            worker.cargo == 0
            and turn.beacon.status == BeaconStatus.GROUND
            and turn.beacon.position == pos
        ):
            worker.pickup_beacon()
            continue
        if (
            worker.cargo == 0
            and pos in resource_cells
            and resource_assignments.get(wid) == pos
        ):
            worker.harvest()
            continue
        if (
            worker.cargo > 0
            and pos == core_pos
            and turn.core is not None
            and turn.core.view.state == "NORMAL"
            and turn.resources < turn.resource_capacity
        ):
            # 已经到核且存在容量时，交付必须先于循环/卡死恢复；历史轨迹
            # 只能影响移动，不能把可立即兑现的货物再次带离 Core。
            worker.deposit()
            _begin_outbound(wid, orig_index, pos, core_pos)
            _prev_pos[wid] = pos
            _pos_history.pop(wid, None)
            continue
        # Boxed-in detection: the STUCK check below only fires when a Worker
        # stays STILL. A Worker trapped in an obstacle pocket CYCLES between a
        # few cells (A-B-A-B...) — it never stays still, so STUCK never fires
        # and it spins forever (observed: worker ce6788 cycled between
        # (12,215)/(12,216)/(13,216) for 20+ ticks). Detect the pocket by the
        # recent-position history all fitting in a tiny box; break the loop by
        # switching the sweep to a DIFFERENT column (clearing explore state
        # alone re-initializes to the SAME column and re-enters the pocket) and
        # stepping outward.
        # 带货 Worker 必须优先回 Core；探索脱困会把它反向带离交付路线。
        if worker.cargo == 0 and _is_boxed_in(wid):
            st = _explore_state.get(wid)
            col_off = st[0] if st is not None and len(st) >= 1 else 0
            # Keep the half-zone direction on escape (even -> north, odd ->
            # south), otherwise a boxed escape re-sends even workers south.
            _explore_state[wid] = [col_off + _SWEEP_COL_STEP, 1 if orig_index % 2 == 1 else 0, None, None]
            _pos_history.pop(wid, None)
            _prev_pos.pop(wid, None)
            _last_pos.pop(wid, None)
            _stuck_ticks[wid] = 0
            step = _step_away_from(pos, core_pos, base_blocked, avoid=None)
            if step is not None:
                _prev_pos[wid] = pos
                _record_pos(wid, pos)
                worker.move(step)
                continue
        # Detect a stuck Worker (no movement for STUCK_TICKS): reset its
        # exploration state and anti-backtrack memory so it can pick a fresh
        # direction instead of WAITing forever in a self-made deadlock.
        if _last_pos.get(wid) == pos:
            _stuck_ticks[wid] = _stuck_ticks.get(wid, 0) + 1
        else:
            _stuck_ticks[wid] = 0
            _last_pos[wid] = pos
        if _stuck_ticks.get(wid, 0) >= STUCK_TICKS:
            _explore_state.pop(wid, None)
            _prev_pos.pop(wid, None)
            _pos_history.pop(wid, None)  # clear cycle memory too
            # Fourth-review fix (#5): also clear _last_pos so the next tick does
            # not immediately re-increment stuck_ticks against the stale value
            # and re-trigger a reset that produces no move (the reset was
            # incomplete, leaving cbf157 deadlocked for 2780 ticks).
            _last_pos.pop(wid, None)
            _stuck_ticks[wid] = 0
        # Each Worker paths around obstacles, enemies, and FULL friendly cells
        # (2 occupants), but may step onto a singly-occupied friendly cell or
        # its own current cell or the Core (to deposit).
        blocked = (base_blocked | friendly_full) - {pos, core_pos}
        # A full Worker deposits if it is home, else heads home.
        if worker.cargo > 0:
            core_full = turn.resources >= turn.resource_capacity
            if pos == core_pos and turn.core is not None and turn.core.view.state == "NORMAL":
                if core_full:
                    # Deadlock escape: Core at full capacity. This laden Worker
                    # on the Core cell cannot deposit (CORE_RESOURCE_FULL) but
                    # its occupancy blocks Core spawn (CELL_UNIT_LIMIT). Step
                    # outward into any open cell rather than deposit, so the
                    # spawn fires and spends resources — then next Tick the cap
                    # has room and this Worker (still laden) returns to deposit.
                    avoid = _avoid_set(wid)
                    step = _step_away_from(pos, core_pos, blocked, avoid=avoid)
                    if step is None:
                        step = _step_away_from(pos, core_pos, blocked, avoid=None)
                    if step is not None:
                        _prev_pos[wid] = pos
                        _record_pos(wid, pos)
                        worker.move(step)
                        continue
                worker.deposit()
                # Deposited: resume the assigned scan row, clearing any lock.
                _begin_outbound(wid, orig_index, pos, core_pos)
                _prev_pos[wid] = pos
                _pos_history.pop(wid, None)  # fresh start after deposit
            else:
                if core_full:
                    # Core full: laden Workers cannot deposit. Instead of
                    # orbiting the Core (dead time), resume the boustrophedon
                    # sweep to keep discovering nodes; once a slot frees
                    # (spawn/casualty) the Worker returns to deposit (9th
                    # review rank 1). This preserves discovery during the
                    # saturation window.
                    step = _explore_step(
                        orig_index, wid, pos, core_pos, blocked,
                        target_col=None, avoid=_avoid_set(wid),
                        fleet_size=len(turn.workers),
                    )
                    if step is None:
                        step = _step_away_from(pos, core_pos, blocked, avoid=None)
                    if step is not None:
                        _prev_pos[wid] = pos
                        _record_pos(wid, pos)
                        worker.move(step)
                    continue
                # Deposit admission (user suggestion): when the Core cell is
                # occupied, keep at least ONE adjacent cell open so the occupant
                # can always leave. If laden workers filled the whole adjacent
                # ring (occ 2/2), the Core-cell unit is walled in and r freezes
                # (observed: empty worker 414e50 trapped by 8 laden workers).
                # When the ring has only 1 free slot left, BACK OFF to distance 2
                # instead of taking the last exit; when there are >=2 free slots,
                # WAIT in place is harmless (the occupant still has a way out).
                core_colocated = sum(
                    1 for u in turn.units if u.position == core_pos
                )
                if core_colocated >= 1 and _manhattan(pos, core_pos) <= 1:
                    free_adjacent = 0
                    for d in DIRECTIONS:
                        nxt = (core_pos[0] + d.delta[0], core_pos[1] + d.delta[1])
                        if nxt == pos:
                            continue  # the slot we occupy is not an exit
                        if nxt in base_blocked:
                            continue
                        if friendly_occupancy.get(nxt, 0) < 2:
                            free_adjacent += 1
                    if free_adjacent <= 1:
                        step = _step_away_from(pos, core_pos, blocked, avoid=_avoid_set(wid))
                        if step is None:
                            step = _step_away_from(pos, core_pos, blocked, avoid=None)
                        if step is not None:
                            _prev_pos[wid] = pos
                            _record_pos(wid, pos)
                            worker.move(step)
                    continue
                # A* path back to the Core. Greedy _step_toward wedges in
                # obstacle-dense stone-corner terrain: a laden Worker 20+ cells
                # out can spin in circles and never find the gap to the Core
                # (observed 67512f/d48045). A* guarantees a route when one
                # exists; fall back to the greedy step if A* finds none.
                step = _astar_step(
                    pos, core_pos, base_blocked, blocked,
                    allow_blocked_goal=True,
                )
                avoid = _avoid_set(wid)
                if step is not None and avoid is not None:
                    ddx, ddy = step.delta
                    next_cell = (pos[0] + ddx, pos[1] + ddy)
                    if next_cell in avoid:
                        step = _step_toward(pos, core_pos, blocked, avoid=avoid)
                if step is None:
                    step = _step_toward(pos, core_pos, blocked, avoid=avoid)
                    if step is None:
                        step = _step_toward(pos, core_pos, blocked, avoid=None)
                if step is not None:
                    _prev_pos[wid] = pos
                    _record_pos(wid, pos)
                    worker.move(step)
            continue
        # An EMPTY Worker must not enter the Core cell: it only occupies the
        # 2/2 slot and deadlocks laden deposits (observed: empty worker 414e50
        # parked on the Core for 25+ ticks, r frozen). Keep core_pos in the
        # blocked set for empty Workers so their harvest-lock and sweep paths
        # never step onto it. Only laden Workers may enter to deposit.
        blocked_empty = blocked | {core_pos}
        # An empty Worker on a visible resource cell harvests now. The cell set
        # is the current visible view, not permanent terrain.
        if pos in resource_cells and resource_assignments.get(wid) == pos:
            worker.harvest()
            continue

        target = resource_assignments.get(wid)
        if target is not None:
            step, budget_exhausted = _astar_step_result(
                pos, target, base_blocked, blocked_empty
            )
            target_sealed = all(
                (
                    target[0] + direction.delta[0],
                    target[1] + direction.delta[1],
                )
                in blocked_empty
                for direction in DIRECTIONS
            )
            if (
                step is None
                and target not in resource_cells
                and (target_sealed or not budget_exhausted)
            ):
                # 历史提示无法通过完整已知地图到达时，暂停追逐并回到前沿。
                # 当前可见资源是真实目标，仍允许下面的贪心兜底。
                _cooldown_resource(target, turn.tick)
                _explore_targets.pop(wid, None)
                _explore_progress.pop(wid, None)
                fallback_workers.append((wid, tuple(pos)))
                explore_targets = _assign_explore_targets(
                    fallback_workers,
                    core_pos,
                    dynamically_blocked,
                    tick=turn.tick,
                )
            elif step is None:
                step = _step_toward(pos, target, blocked_empty, avoid=_avoid_set(wid))
            if step is not None:
                _prev_pos[wid] = pos
                _record_pos(wid, pos)
                worker.move(step)
                continue

        # No resource task: move toward this Worker's stable, globally dispersed
        # frontier target. The persistent map now actively drives exploration.
        target = explore_targets.get(wid)
        step = None
        retargeted_frontier = False
        if target is not None:
            if _explore_target_has_stalled(
                wid, tuple(pos), target, dynamically_blocked
            ):
                _cooldown_explore_target(target, turn.tick)
                _explore_targets.pop(wid, None)
                _explore_progress.pop(wid, None)
                explore_targets = _assign_explore_targets(
                    idle_workers,
                    core_pos,
                    dynamically_blocked,
                    tick=turn.tick,
                )
                target = explore_targets.get(wid)
                retargeted_frontier = True
            if target is not None and not retargeted_frontier:
                step, budget_exhausted = _astar_step_result(
                    pos, target, base_blocked, blocked_empty
                )
                target_sealed = all(
                    (
                        target[0] + direction.delta[0],
                        target[1] + direction.delta[1],
                    )
                    in blocked_empty
                    for direction in DIRECTIONS
                )
                if step is None and (target_sealed or not budget_exhausted):
                    # A fully drained open set proves that this frontier cell
                    # is unreachable. Do not follow it with a greedy detour;
                    # cool it and select a fresh frontier instead.
                    _cooldown_explore_target(target, turn.tick)
                    _explore_targets.pop(wid, None)
                    _explore_progress.pop(wid, None)
                    explore_targets = _assign_explore_targets(
                        idle_workers,
                        core_pos,
                        dynamically_blocked,
                        tick=turn.tick,
                    )
                    target = None
                    retargeted_frontier = True
                elif step is None:
                    # The expansion budget is a retryable condition, not proof
                    # of a sealed target. Keep the intent and try again later.
                    step = _step_toward(
                        pos, target, blocked_empty, avoid=_avoid_set(wid)
                    )
        if step is None:
            step = _explore_step(
                orig_index, wid, pos, core_pos, blocked_empty,
                target_col=None, avoid=_avoid_set(wid),
                fleet_size=len(turn.workers),
                sweep_radius=explore_radius,
            )
        if step is not None:
            _prev_pos[wid] = pos
            _record_pos(wid, pos)
            worker.move(step)
        else:
            # The sweep found no legal step (trapped in an obstacle pocket or
            # parked on the Core cell). WAITing here loops forever: STUCK
            # resets the explore state every 4 ticks but the pocket still has no
            # column-reachable step (observed: ce6788 waited-then-moved between
            # (12,215)/(12,216)/(13,216) for 20+ ticks). Force a step toward any
            # open adjacent cell using base_blocked (not blocked_empty), so the
            # Core cell is reachable as a last-resort exit — moving beats a
            # permanent stall.
            step = _step_away_from(pos, core_pos, base_blocked, avoid=None)
            if step is None:
                step = _step_toward(pos, core_pos, base_blocked, avoid=None)
            if step is not None:
                _prev_pos[wid] = pos
                _record_pos(wid, pos)
                worker.move(step)


def _vanguard_guard_targets(
    turn: "Turn", core_pos: tuple[int, int]
) -> dict[str, tuple[int, int]]:
    """Give one Vanguard a close guard cell and spread the rest at distance 2.

    A single adjacent Vanguard can sweep a raider entering the Core. Stacking
    two there fills a delivery entrance to 2/2, so additional Vanguards use
    unique support cells and leave at least three Core-adjacent lanes empty.
    """
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    enemy_positions = frozenset(e.position for e in turn.visible_enemies)
    friendly_occ = Counter(tuple(unit.position) for unit in turn.units)
    vanguard_occ = Counter(tuple(unit.position) for unit in turn.vanguards)

    def available(cell: tuple[int, int]) -> bool:
        non_vanguards = friendly_occ.get(cell, 0) - vanguard_occ.get(cell, 0)
        return (
            cell not in obstacles
            and cell not in enemy_positions
            and non_vanguards < 2
        )

    cx, cy = core_pos
    adjacent = [
        (cx + direction.delta[0], cy + direction.delta[1])
        for direction in DIRECTIONS
        if available((cx + direction.delta[0], cy + direction.delta[1]))
    ]
    support = sorted(
        (cx + dx, cy + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if abs(dx) + abs(dy) == 2 and available((cx + dx, cy + dy))
    )

    targets: dict[str, tuple[int, int]] = {}
    selected: set[tuple[int, int]] = set()
    for index, vanguard in enumerate(
        sorted(turn.vanguards, key=lambda unit: str(unit.id))
    ):
        pools = [adjacent, support] if index == 0 else [support, adjacent]
        candidates = [
            cell for pool in pools for cell in pool if cell not in selected
        ]
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda cell: (
                0 if cell in pools[0] else 1,
                _manhattan(vanguard.position, cell),
                friendly_occ.get(cell, 0),
                cell,
            ),
        )
        targets[str(vanguard.id)] = target
        selected.add(target)
    return targets


def _pending_deposit_amount(turn: "Turn") -> int:
    """计算本 Tick 进入 Core 的可用 Worker 货物上限。"""
    core = turn.core
    if core is None or core.view.state != "NORMAL":
        return 0
    space = max(0, turn.resource_capacity - turn.resources)
    if space == 0:
        return 0
    cargo = sum(
        max(0, worker.cargo)
        for worker in turn.workers
        if worker.position == core.position and worker.cargo > 0
    )
    return min(space, cargo)


def _reserve_unit_heals(
    turn: "Turn",
    core_pos: tuple[int, int],
    available_resources: int,
) -> tuple[frozenset[str], int]:
    """按实际结算顺序预留 Unit HEAL 资源，避免 Core 动作透支。"""
    core = turn.core
    if core is None or core.view.state != "NORMAL":
        return frozenset(), available_resources

    candidates: list[tuple[str, int]] = []
    for unit in sorted(turn.units, key=lambda item: str(item.id)):
        if unit.position != core_pos:
            continue
        if unit.unit_type == "VANGUARD" and unit.hp <= 1:
            missing = 4 - unit.hp
        elif unit.unit_type == "RANGER" and unit.hp <= 1:
            missing = 2 - unit.hp
        else:
            continue
        if missing > 0:
            candidates.append((str(unit.id), missing))

    reserved: set[str] = set()
    remaining = max(0, available_resources)
    for unit_id, cost in candidates:
        if cost > remaining:
            continue
        reserved.add(unit_id)
        remaining -= cost
    return frozenset(reserved), remaining


def _control_vanguards(
    turn: "Turn",
    core_pos: tuple[int, int],
    heal_ids: frozenset[str] | None = None,
) -> None:
    enemies = turn.visible_enemies
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    friendly_occ = Counter(tuple(unit.position) for unit in turn.units)
    friendly_full = frozenset(
        cell for cell, count in friendly_occ.items() if count >= 2
    )
    enemy_positions = frozenset(enemy.position for enemy in enemies)
    targets = _vanguard_guard_targets(turn, core_pos)
    resources = turn.resources
    core_normal = turn.core is not None and turn.core.view.state == "NORMAL"
    for vanguard in sorted(turn.vanguards, key=lambda unit: str(unit.id)):
        # HEAL at the Core: post-combat HP recovery costs 1 resource per HP
        # (SDK arena-hero 0.2.8, v0.8 rules). A 1-HP Vanguard (max 4) recovers
        # 3 HP for 3 resources vs 10 to rebuild — 3.3x ROI.
        should_heal = (
            str(vanguard.id) in heal_ids
            if heal_ids is not None
            else (
                core_normal
                and vanguard.position == core_pos
                and vanguard.hp <= 1
                and resources >= 3
            )
        )
        if should_heal:
            vanguard.heal()
            continue
        sweep_dir = _vanguard_sweep_target(vanguard.position, enemies)
        if sweep_dir is not None:
            vanguard.sweep(sweep_dir)
            continue
        target = targets.get(str(vanguard.id))
        # A Vanguard at full HP (4) can absorb one more hit before dying; a 1-HP
        # Vanguard is dead after one more sweep. Regroup toward the Core when
        # critically damaged so a raider cannot one-shot our body-block.
        hp = getattr(vanguard, "hp", 4)
        if hp <= 1 and target is None:
            step = _step_toward(
                vanguard.position, core_pos,
                (obstacles | enemy_positions | friendly_full) - {vanguard.position},
                avoid=None,
            )
            if step is not None:
                vanguard.move(step)
            continue
        if target is None or target == vanguard.position:
            continue
        blocked = (
            obstacles | enemy_positions | friendly_full | {core_pos}
        ) - {vanguard.position}
        step = _astar_step(vanguard.position, target, obstacles, blocked)
        if step is None:
            step = _step_toward(vanguard.position, target, blocked)
        if step is not None:
            vanguard.move(step)


def _ring_patrol_step(
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    inner: int = 8,
    outer: int = 16,
) -> Direction | None:
    """Step to keep a non-guard Ranger patrolling a home band around the Core.

    8th review rank 5: roaming Rangers were full-chunk scouts, often far from
    the Core when a raid hit. Keeping one Ranger patrolling a band 8-16 cells
    out makes it a close-in interceptor for the drive-off while the others
    still scout far. Too close -> drift out; too far -> return; inside the
    band -> step to a neighbor that keeps the distance (ring patrol).
    """
    dist = _manhattan(pos, core_pos)
    if dist < inner:
        return _step_away_from(pos, core_pos, blocked, avoid=None)
    if dist > outer:
        return _step_toward(pos, core_pos, blocked, avoid=None)
    best: Direction | None = None
    best_keep = 10**9
    # Within the band: prefer a step that KEEPS the distance roughly constant
    # (tangential ring-walk), scoring |change| only against band-members. This
    # avoids the perverse preference where a large |change| that jumps across
    # the band (e.g. dist-16 -> dist-2) beat an exact hold.
    for d in DIRECTIONS:
        nxt = (pos[0] + d.delta[0], pos[1] + d.delta[1])
        if nxt in blocked:
            continue
        nd = _manhattan(nxt, core_pos)
        if inner <= nd <= outer:
            change = abs(nd - dist)
            if change < best_keep:
                best_keep = change
                best = d
    return best


def _can_shoot(
    shooter: tuple[int, int],
    target: tuple[int, int],
    obstacles: frozenset[tuple[int, int]],
) -> bool:
    """True if a Ranger at ``shooter`` can legally shoot ``target``.

    Rules (v0.14): shared cardinal or exact-diagonal line, range 1-3, with no
    obstacle in an intermediate shot cell.
    """
    if not _same_fire_line(shooter, target):
        return False
    dist = max(abs(shooter[0] - target[0]), abs(shooter[1] - target[1]))
    if dist < 1 or dist > 3:
        return False
    return not _obstacles_between(shooter, target, obstacles)


def _guard_reposition_step(
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    obstacles: frozenset[tuple[int, int]],
    friendly_full: frozenset[tuple[int, int]],
    avoid: frozenset[tuple[int, int]] | None = None,
) -> Direction | None:
    """Return one step toward the nearest legal dist-2/3 LOS choke for the guard.

    9th review rank 5: when the guard sits out of Ranger range (dist 4-5) with
    no adjacent cell that lands on a dist-2/3 LOS choke, it previously froze
    (observed at (13,237), dist 5, zero return fire for 43+ ticks). This finds
    the nearest passable cell within Manhattan 2-3 of the Core that has a clear
    line to the Core cell, then A*-routes toward it (falling back to a greedy
    step), so the guard re-enters range instead of standing still.
    """
    # Enumerate all dist-2/3 cells around the Core with a clear shot to it.
    candidates: list[tuple[int, tuple[int, int]]] = []
    for ddx in range(-3, 4):
        for ddy in range(-3, 4):
            cell = (core_pos[0] + ddx, core_pos[1] + ddy)
            nd = abs(ddx) + abs(ddy)
            if nd < 2 or nd > 3:
                continue
            if cell in blocked or cell in friendly_full:
                continue
            if not _can_shoot(cell, core_pos, obstacles):
                continue
            candidates.append((_manhattan(pos, cell), cell))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    goal = candidates[0][1]
    step = _astar_step(pos, goal, obstacles, blocked)
    if step is None:
        step = _step_toward(pos, goal, blocked, avoid=avoid)
    return step


def _guard_step(
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
    obstacles: frozenset[tuple[int, int]],
    enemies: tuple[UnitView | CoreView, ...],
    friendly_full: frozenset[tuple[int, int]],
    avoid: frozenset[tuple[int, int]] | None = None,
) -> Direction | None:
    """Step to keep a guard Ranger at a defensive choke near the Core.

    The guard's goal is to ALWAYS have a legal shot to the Core cell (so any
    raider stepping onto the Core meets return fire) AND toward the likely
    approach vector. 8th review found the old ring-orbit guard was
    direction-blind: it scored nothing but Manhattan distance, allowed dist-4
    cells that cannot reach the Core cell (range cap 3), and had no obstacle
    / occupancy awareness. New design:
    - Far (>5): return toward Core.
    - Adjacent (<3): drift outward UNLESS every dist-3 cell is blocked (in an
      obstacle corner the adjacent cell may be the only legal spot — hold it).
    - At dist 2-3: score candidate cells by (i) clear cardinal line to Core,
      (ii) clear line toward a visible enemy (threat-side bias), (iii) avoid
      laden workers and full cells (deposit leave-one-exit rule).
    - Never accept a cell at dist-4 or with no LOS to the Core.
    """
    dist = _manhattan(pos, core_pos)
    if dist > 3:
        # 9th review rank 5: at dist 4-5 the adjacent-cell scoring below
        # returns None (no neighbor lands at dist 2-3), so the guard froze at
        # (13,237), dist 5, with ZERO ranged return fire for 43+ ticks. When
        # out of range and no adjacent dist-2/3 LOS step exists, A* toward the
        # nearest legal dist-2/3 LOS choke instead of standing still.
        step = _guard_reposition_step(pos, core_pos, blocked, obstacles, friendly_full, avoid)
        if step is not None:
            return step
        if dist > 5:
            return _step_toward(pos, core_pos, blocked, avoid=avoid)
        return None
    if dist < 2:
        # Adjacent to the Core: drift outward to a dist-2/3 cell that can
        # shoot the Core; in a corner the adjacent cell may be the only legal
        # spot, so hold it if nothing reachable has a line to the Core.
        step = _step_away_from(pos, core_pos, blocked, avoid=avoid)
        if step is not None:
            nxt = (pos[0] + step.delta[0], pos[1] + step.delta[1])
            if _manhattan(nxt, core_pos) <= 3 and _can_shoot(nxt, core_pos, obstacles):
                return step
        return None
    # At dist 2-3: score candidate cells.
    best_dir: Direction | None = None
    best_score: int | None = None
    for d in DIRECTIONS:
        nxt = (pos[0] + d.delta[0], pos[1] + d.delta[1])
        if nxt in blocked or nxt in friendly_full:
            continue
        if avoid is not None and nxt in avoid:
            continue
        nd = _manhattan(nxt, core_pos)
        if nd < 2 or nd > 3:
            continue  # too close or too far — cannot cover the Core
        if not _can_shoot(nxt, core_pos, obstacles):
            continue  # cannot shoot the Core cell -> useless guard position
        # Score: count enemies within range 3 of this cell (on cardinals).
        enemy_score = sum(
            1 for e in enemies if _can_shoot(nxt, e.position, obstacles)
        )
        score = (enemy_score, nd)  # prefer more enemy coverage, then closer
        if best_score is None or score > best_score:
            best_score = score
            best_dir = d
    # Fall back to any dist-3 cell with LOS to the Core if no shootable
    # enemy exists.
    if best_dir is None:
        for d in DIRECTIONS:
            nxt = (pos[0] + d.delta[0], pos[1] + d.delta[1])
            if nxt in blocked or nxt in friendly_full:
                continue
            if avoid is not None and nxt in avoid:
                continue
            nd = _manhattan(nxt, core_pos)
            if nd < 2 or nd > 3:
                continue
            if _can_shoot(nxt, core_pos, obstacles):
                best_dir = d
                break
    return best_dir


def _control_rangers(
    turn: "Turn",
    core_pos: tuple[int, int],
    heal_ids: frozenset[str] | None = None,
) -> None:
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    enemies = turn.visible_enemies
    resources = turn.resources
    core_normal = turn.core is not None and turn.core.view.state == "NORMAL"
    for index, ranger in enumerate(turn.rangers):
        # HEAL at the Core: post-combat HP recovery costs 1 resource per HP
        # (SDK arena-hero 0.2.8, v0.8 rules). A 1-HP Ranger (max 2) recovers
        # to full for 1 resource vs 12 to rebuild — 12x ROI.
        should_heal = (
            str(ranger.id) in heal_ids
            if heal_ids is not None
            else (
                core_normal
                and ranger.position == core_pos
                and ranger.hp <= 1
                and resources >= 1
            )
        )
        if should_heal:
            ranger.heal()
            continue
        target = _select_ranger_target(ranger.position, enemies, obstacles, core_pos)
        if target is not None:
            ranger.shoot(target)
            continue
        # The FIRST Ranger is the dedicated Core guard: hold a choke near the
        # Core with a clear cardinal line to the Core cell, preferring cells
        # that cover visible enemies too. 8th review: old ring-orbit guard
        # scored nothing but Manhattan distance, allowed dist-4 cells that
        # cannot shoot the Core cell (range cap 3), and had no obstacle or
        # occupancy awareness — the new _guard_step is LOS/choke-aware.
        if index == 0:
            # Friendly-full cells block the guard's ring, but the guard must
            # not block the deposit lane (leave-one-exit rule).
            friendly_full = frozenset(
                cell for cell, count
                in Counter(tuple(u.position) for u in turn.units).items()
                if count >= 2
            )
            enemy_positions = frozenset(e.position for e in enemies)
            step = _guard_step(
                ranger.position, core_pos,
                obstacles | enemy_positions, obstacles,
                turn.visible_enemies, friendly_full,
                avoid=_avoid_set(str(ranger.id)),
            )
            if step is not None:
                ranger.move(step)
            continue
        # Otherwise explore like a Worker (a Ranger's vision radius of 5 is the
        # best scout): use the deterministic scan-row sweep so it covers ground
        # instead of milling near the Core.
        pos = ranger.position
        rid = str(ranger.id)
        # Bounded drive-off (8th review, rank 2): a non-guard Ranger chases a
        # visible or recently-seen enemy near the Core, so inbound raiders are
        # intercepted instead of ignored. Guard (index 0) never chases.
        chase = _chase_target(pos, core_pos, enemies, rid, turn.tick, turn.resources)
        if chase is not None:
            enemy_cells = frozenset(e.position for e in enemies)
            # Chase to a cell that can SHOOT the target, not onto it. Moving
            # onto an enemy cell always fails MOVE_DESTINATION_OCCUPIED; a
            # Vanguard/Ranger reached cell fights (we want distance), a Worker
            # reached cell is useless (cannot attack). _step_toward supports
            # `toward_exact`; passing a "blocked" goal makes it step toward it
            # while treating the exact goal cell as allowed-only-if-reached.
            blocked = obstacles | enemy_cells
            goal_exact = pos == chase  # already there -> just look for a shot
            if not goal_exact:
                step = _astar_step(pos, chase, obstacles, blocked)
                if step is None:
                    step = _step_toward(pos, chase, obstacles | (enemy_cells - {chase}),
                                        avoid=_avoid_set(rid))
            else:
                step = None
            if step is None:
                step = _step_toward(pos, chase, obstacles, avoid=_avoid_set(rid))
            if step is not None:
                _record_pos(rid, pos)
                ranger.move(step)
            continue
        # Boxed-in escape (same pocket-cycle trap as workers): if the Ranger's
        # recent positions fit a tiny box, it is spinning in an obstacle pocket
        # — break out by stepping away from the Core.
        if _is_boxed_in(rid):
            _explore_state.pop(rid, None)
            _pos_history.pop(rid, None)
            step = _step_away_from(pos, core_pos, obstacles, avoid=None)
            if step is not None:
                _record_pos(rid, pos)
                ranger.move(step)
            continue
        step = _explore_step(index + 10, rid, pos, core_pos, obstacles)
        if step is not None:
            _record_pos(rid, pos)
            ranger.move(step)
        elif index == 1:
            # The 2nd non-guard Ranger holds a home-band patrol ring so a raid
            # always has a close-in interceptor; only the OTHER roaming Ranger
            # (index >= 2) scouts far (8th review, rank 5).
            step = _ring_patrol_step(pos, core_pos, obstacles)
            if step is not None:
                _record_pos(rid, pos)
                ranger.move(step)


def _standing_army_targets(n_workers: int) -> tuple[int, int]:
    """Scale the standing combat reserve with the Worker economy.

    User requirement: as the Worker fleet grows, the standing army must grow
    with it, so a raid meets more return fire the more valuable the Core is
    (the 2026-08-02 Core loss was a raid against an economy with no army).
    Roughly one combat pair (Vanguard + Ranger) per 8 Workers, a floor of 1,
    then shrunk to fit the conservative population target (W + V + R <= 20,
    i.e. budget = FREE_UPKEEP_CAP). Population 20 is allowed; only the next
    production pays the first dynamic-price premium.

    Ratchet-proof (10th review, rank 1): a raid that kills a Vanguard/Ranger
    does NOT lower the Worker count, so the old formula returned the same
    target and the dead combat Unit was never rebuilt. The floor V>=1,R>=1 is
    now HARD — the budget loop only shrinks above it, and the worker_target in
    _control_core absorbs the budget pressure by lowering the Worker ceiling
    instead. The W=19 V=0,R=0 edge case is removed.
    """
    pairs = max(1, n_workers // 8)
    vanguards = pairs
    rangers = pairs
    floor_v, floor_r = 1, 1  # hard floor — a dead combat Unit must be rebuilt
    budget = FREE_UPKEEP_CAP
    while n_workers + vanguards + rangers > budget:
        if vanguards > max(rangers, floor_v):
            vanguards -= 1
        elif rangers > floor_r:
            rangers -= 1
        else:
            break
    return max(vanguards, floor_v), max(rangers, floor_r)


def _control_core(
    turn: "Turn",
    threats: list[UnitView | CoreView],
    enemy_core_visible: bool = False,
    available_resources: int | None = None,
) -> None:
    core = turn.core
    if core is None:
        return
    # A migrating Core cannot spawn, repair, or receive deposits; let the move
    # resolve rather than queue an action that would fail with CORE_ALREADY_MOVING.
    if core.view.state == "MOVING":
        return

    resources = turn.resources
    if available_resources is None:
        available_resources = resources + _pending_deposit_amount(turn)
    # Repair shield first when under threat and there is space and a spare
    # resource. Holding the Beacon raises the cap to 10, so use the live cap.
    if threats and available_resources >= 1:
        friendly_ids = {u.id for u in turn.units}
        friendly_ids.add(core.id)
        owns_beacon = (
            turn.beacon.status == BeaconStatus.CARRIED
            and turn.beacon.carrier_id in friendly_ids
        )
        cap = 10 if owns_beacon else 5
        if core.shield < cap:
            core.repair_shield()
            return
    if (
        turn.beacon.status == BeaconStatus.GROUND
        and turn.beacon.position == core.position
    ):
        core.pickup_beacon()
        return

    # Spawn Workers toward the target fleet so the economy grows and explores
    # faster, keeping the soft population target at 20. Only spawn
    # when the Core cell has room (Core + at most one colocated Unit) so it
    # does not fail with CELL_UNIT_LIMIT.
    #
    # EXCEPTION: when the Core is at full resource capacity AND the only
    # occupant is a LADEN Worker, skip the colocated guard: the Worker's
    # deposit would fail (CORE_RESOURCE_FULL) so _control_workers steps it OFF
    # the Core cell this same Tick, and Unit movement resolves BEFORE Core
    # spawn (rules: moves step 5, spawn step 9), clearing the cell so the spawn
    # succeeds. The spawn spends 5 resources, opening room for the Worker's
    # deposit next Tick. An empty Worker or a combat Unit stays put, so
    # spawning would only fail CELL_UNIT_LIMIT — keep the guard for those.
    population = turn.state.population
    core_full = resources >= turn.resource_capacity
    colocated = sum(u.position == core.position for u in turn.units)
    laden_on_core = sum(
        w.cargo > 0 for w in turn.workers if w.position == core.position
    )
    can_clear_occupied = core_full and colocated == laden_on_core and laden_on_core > 0
    if colocated >= 1 and not can_clear_occupied:
        return
    # Deposits resolve before Unit heals and the Core action in the same Tick;
    # ``available_resources`` already includes only the amount that fits and
    # excludes resources reserved for Unit HEAL actions.
    effective_resources = available_resources

    # Standing-army priorities. Combat Units are built BEFORE growing the Worker
    # fleet past the economy floor, so a surprise raid always meets return fire.
    # The target combat counts step up when a threat is actually visible.
    #
    # Order: below MIN_WORKERS_BEFORE_ARMY, Workers come first so the economy
    # can sustain the army; at or above the floor, combat Units take priority
    # (and Worker growth is bank-throttled — see the army_short gate below) so
    # the standing reserve completes before the fleet grows past the floor.
    threatened = bool(threats)
    if threatened:
        # Under a visible threat, escalate the defensive line to the combat
        # caps regardless of the peacetime standing scale.
        target_vanguards = DEFENSE_VANGUARDS
        target_rangers = DEFENSE_RANGERS
    else:
        # Peacetime standing reserve scales with the Worker economy (user
        # requirement): a bigger fleet must field a bigger army so a raid
        # meets more return fire the more valuable the Core is. Growth is
        # capped by the conservative population target at 20.
        target_vanguards, target_rangers = _standing_army_targets(
            len(turn.workers)
        )
    # FINAL GOAL: a visible enemy Core is a strike target — destroying it
    # removes the enemy fleet and may capture its stockpiled resources. The loot
    # is variable (per CORE_RESOURCES_CAPTURED, not a flat +6), so the raid's real
    # value is elimination + uncertain loot, not a guaranteed resource windfall.
    # Form a strike force (extra Vanguards to tank + Rangers to snipe the Core)
    # to raid it instead of only defending.
    if enemy_core_visible:
        target_vanguards = max(target_vanguards, DEFENSE_VANGUARDS + 2)
        target_rangers = max(target_rangers, DEFENSE_RANGERS + 2)

    # FINAL GOAL (build attack units in time): when ANY enemy is visible, the
    # army must form immediately — drop the economy floor so a Vanguard/Ranger
    # is built even with a young Worker fleet, instead of waiting for 4 Workers
    # while the enemy raids unchecked.
    enemy_present = threatened or enemy_core_visible
    economy_floor_met = len(turn.workers) >= MIN_WORKERS_BEFORE_ARMY
    army_floor_met = enemy_present or len(turn.workers) >= MIN_WORKERS_BEFORE_ARMY

    if economy_floor_met or army_floor_met:
        # Rangers are the strongest defender (range-3 return fire from a
        # 5-vision scout) but have a higher dynamic price, so build the cheap
        # Vanguard body-block first, then the Ranger.
        wants_vanguard = len(turn.vanguards) < target_vanguards
        wants_ranger = len(turn.rangers) < target_rangers
        vanguard_price = unit_cost(UnitType.VANGUARD, population)
        ranger_price = unit_cost(UnitType.RANGER, population)
        if wants_vanguard and effective_resources >= vanguard_price:
            core.spawn(UnitType.VANGUARD)
            return
        if wants_ranger and effective_resources >= ranger_price:
            core.spawn(UnitType.RANGER)
            return

    # Budget-aware Worker target: the effective ceiling is the population
    # target minus the standing army's population cost. A fixed TARGET_WORKERS
    # ignored the army and pushed the fleet into the next price step.
    # TARGET_WORKERS remains as a public compatibility constant; the live
    # ceiling is derived from the current standing-army population cost.
    worker_target = FREE_UPKEEP_CAP - (
        len(turn.vanguards) + len(turn.rangers)
    )
    wants_worker = len(turn.workers) < worker_target
    army_short = (
        len(turn.vanguards) < target_vanguards
        or len(turn.rangers) < target_rangers
    )
    # Under a visible enemy, if the combat reserve is still short, bank toward
    # the next combat Unit.  In peaceful play a bounded economy bridge is
    # allowed whenever at least one combat Unit already exists: the extra
    # Workers increase discovery and Core capacity while the bridge cap keeps
    # a missing Ranger from turning into unbounded Worker spending.
    bridge_worker_allowed = (
        not threatened
        and len(turn.workers) < ECONOMY_BRIDGE_MAX_WORKERS
        and (len(turn.vanguards) + len(turn.rangers)) > 0
    )
    combat_count = len(turn.vanguards) + len(turn.rangers)
    if army_short and not bridge_worker_allowed:
        # A cold start still builds its first combat pair before growing past
        # the economy floor. Once a combat Unit exists, stop the bridge at its
        # cap and bank for the missing Ranger/Vanguard; a visible enemy always
        # takes that combat-first path regardless of the bridge state.
        if (
            enemy_present
            or (combat_count == 0 and economy_floor_met)
            or (
                combat_count > 0
                and len(turn.workers) >= ECONOMY_BRIDGE_MAX_WORKERS
            )
        ):
            return
    # Bank reserve: only spawn a Worker if the Core keeps at least
    # WORKER_SPAWN_RESERVE resources afterward, so the economy never drains to
    # zero and the standing-army bank is not reset each spawn.
    worker_price = unit_cost(UnitType.WORKER, population)
    if wants_worker and effective_resources >= worker_price + WORKER_SPAWN_RESERVE:
        core.spawn(UnitType.WORKER)
        return


def _sync_explore_state(turn: "Turn") -> None:
    """Drop exploration hints for Units that are no longer alive.

    A Worker that died or a fresh respawn (new UUIDs) must not keep stale state.
    This keeps the intent dict aligned with the current live Units.
    """
    live = {str(u.id) for u in turn.units}
    for uid in list(_explore_state):
        if uid not in live:
            del _explore_state[uid]
    for uid in list(_explore_targets):
        if uid not in live:
            del _explore_targets[uid]
    for uid in list(_explore_progress):
        if uid not in live:
            del _explore_progress[uid]
    for uid in list(_prev_pos):
        if uid not in live:
            del _prev_pos[uid]
    for uid in list(_pos_history):
        if uid not in live:
            del _pos_history[uid]
    for uid in list(_last_pos):
        if uid not in live:
            del _last_pos[uid]
            del _stuck_ticks[uid]


def _vision_sources(turn: "Turn") -> list[tuple[tuple[int, int], int]]:
    """Yield (position, vision_radius) pairs for all friendly vision sources.

    Core vision radius 5, Worker 3, Vanguard 4, Ranger 5 (game rules).
    """
    sources: list[tuple[tuple[int, int], int]] = []
    core = turn.core
    if core is not None:
        sources.append((core.position, 5))
    for u in turn.units:
        if u.unit_type == "WORKER":
            sources.append((u.position, 3))
        elif u.unit_type == "VANGUARD":
            sources.append((u.position, 4))
        elif u.unit_type == "RANGER":
            sources.append((u.position, 5))
    return sources


def _supercover_line(
    start: tuple[int, int], target: tuple[int, int]
) -> list[tuple[int, int]]:
    """Integer supercover line cells from ``start`` to ``target`` inclusive.

    Same as the reference agent's implementation: when the ray passes exactly
    through a cell corner, both adjacent cells are considered traversed, so an
    obstacle on either blocks the line.
    """
    x, y = start
    tx, ty = target
    dx = abs(tx - x)
    dy = abs(ty - y)
    sx = 1 if tx > x else -1
    sy = 1 if ty > y else -1
    covered = [(x, y)]
    px = 0
    py = 0
    while px < dx or py < dy:
        horizontal = (1 + 2 * px) * dy
        vertical = (1 + 2 * py) * dx
        if horizontal == vertical:
            prev_x, prev_y = x, y
            x += sx
            px += 1
            covered.append((x, y))
            covered.append((prev_x, prev_y + sy))
            y += sy
            py += 1
            covered.append((x, y))
        elif horizontal < vertical:
            x += sx
            px += 1
            covered.append((x, y))
        else:
            y += sy
            py += 1
            covered.append((x, y))
    return covered


def _any_vision_sees(
    cell: tuple[int, int],
    sources: list[tuple[tuple[int, int], int]],
    obstacles: frozenset[tuple[int, int]],
) -> bool:
    """True if any vision source can see ``cell`` (within radius and with a
    clear supercover line — an obstacle on any traversed cell blocks vision).
    """
    for pos, radius in sources:
        if _manhattan(pos, cell) > radius:
            continue
        if _manhattan(pos, cell) == 0:
            return True
        blocked = any(
            (x, y) in obstacles
            for (x, y) in _supercover_line(pos, cell)[1:-1]
        )
        if not blocked:
            return True
    return False


def _observe_resources(turn: "Turn") -> None:
    """Update the local resource memory pool from the current Tick's state.

    Add newly-visible resources; remove cells confirmed bare (visible and NOT
    a resource); remove cells harvested successfully this Tick.
    """
    global _known_resources, _resource_telemetry, _resource_absence_streak
    _resource_telemetry = {
        "resource_failures": sum(
            1
            for event in turn.events
            if event.event_type == "HARVEST_FAILED"
            and event.reason_code in {
                "RESOURCE_DEPLETED",
                "NOT_RESOURCE_CELL",
                "RESOURCE_NOT_FOUND",
            }
        ),
        "harvested": sum(
            event.resource_amount or 0
            for event in turn.events
            if event.event_type == "HARVEST_SUCCEEDED"
        ),
        "deposited": sum(
            event.resource_amount or 0
            for event in turn.events
            if event.event_type == "DEPOSIT_SUCCEEDED"
        ),
    }
    changed = False
    # Remove cells harvested this Tick (the event carries the harvest cell).
    for event in turn.events:
        if event.event_type == "HARVEST_SUCCEEDED" and event.position is not None:
            cell = tuple(event.position)
            changed = changed or cell in _known_resources or cell in _resource_hints
            _known_resources.discard(cell)
            _resource_hints.pop(cell, None)
    # The complete current state is authoritative over previous-Tick events.
    # A dropped-cargo pile can remain after a partial harvest, and a natural
    # node can refill at the same coordinate.
    for cell in turn.resource_cells:
        if cell not in _known_resources:
            _known_resources.add(cell)
            changed = True
        hint = _resource_hints.get(cell)
        if (
            hint is None
            or hint.source != "visible"
            or hint.failure_count > 0
            or hint.cooldown_until > 0
            or turn.tick - hint.last_confirmed_tick >= _RESOURCE_CONFIRM_SAVE_STEP
        ):
            _resource_hints[cell] = ResourceHint(
                last_confirmed_tick=turn.tick,
                source="visible",
            )
            changed = True
    # Remove cells that a friendly vision source can see and are confirmed
    # not to hold a resource (visible but not in turn.resource_cells).
    sources = _vision_sources(turn)
    # The current state may omit a wall that was outside this Tick's view, but
    # the wall is still real terrain.  Keep it in the line-of-sight test or a
    # resource behind that remembered wall is incorrectly declared empty and
    # the next Tick sends its Worker toward a different historical target.
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    for cell in list(_known_resources):
        if cell in turn.resource_cells:
            continue  # still a resource, keep it
        if _any_vision_sees(cell, sources, obstacles):
            _known_resources.discard(cell)
            _resource_hints.pop(cell, None)
            changed = True
    active_hints = any(
        _resource_age(cell, turn.tick) <= _MAX_HISTORY_RESOURCE_AGE
        and _resource_hints.get(cell, ResourceHint(0, "legacy")).cooldown_until
        <= turn.tick
        for cell in _known_resources
    )
    if turn.resource_cells or active_hints:
        _resource_absence_streak = 0
    else:
        _resource_absence_streak += 1
    # Keep this internal value available to live diagnostics without changing
    # the compact log contract consumed by existing monitor tests.
    _resource_telemetry["drought"] = _resource_absence_streak
    if changed:
        _mark_persistent_state_dirty()


def _observe_enemies(turn: "Turn") -> None:
    """Record last-seen enemy positions and prune stale memory.

    Feeds the bounded drive-off for non-guard Rangers (8th review, rank 2):
    with only current-visible enemies, a raider parked just outside range is
    never driven off. Remembering positions lets Rangers chase briefly.
    """
    tick = turn.tick
    cores_before = set(_known_enemy_cores)
    visible_core_positions = {
        tuple(e.position) for e in turn.visible_enemies if e.kind == "CORE"
    }
    sources = _vision_sources(turn)
    for e in turn.visible_enemies:
        _last_enemy_pos[str(e.id)] = (tuple(e.position), tick)
        # Persist enemy Cores permanently (a Core is a durable hunt target).
        if e.kind == "CORE":
            _known_enemy_cores.add(tuple(e.position))
    # Full-ghost pass: a CORE cell that a friendly vision source can see but
    # that does not hold a Core this Tick has moved or been destroyed — drop it
    # so a remembered coordinate never sends Rangers to a dead cell.
    for position in list(_known_enemy_cores):
        if (
            position not in visible_core_positions
            and _any_vision_sees(position, sources, turn.obstacle_cells)
        ):
            _known_enemy_cores.discard(position)
    for eid in list(_last_enemy_pos):
        _, seen = _last_enemy_pos[eid]
        if tick - seen > ENEMY_MEMORY_TICKS:
            del _last_enemy_pos[eid]
    if _known_enemy_cores != cores_before:
        _mark_persistent_state_dirty()


def _chase_target(
    pos: tuple[int, int],
    core_pos: tuple[int, int],
    enemies: tuple[UnitView | CoreView, ...],
    rid: str,
    tick: int,
    resources: int,
) -> tuple[int, int] | None:
    """Pick an enemy cell for a non-guard Ranger to drive off or hunt, or None.

    Priority: (1) a VISIBLE enemy Core within HUNT range (when resources allow
    rebuilding) — destroying it removes the enemy fleet and may capture its
    stockpiled resources (variable loot, not a flat +6); (2) any enemy
    within CHASE_RADIUS of the Core (drive off inbound raiders); (3) a
    recently-seen enemy within the radius. Bounded: gives up after
    CHASE_MAX_TICKS, then cools down before chasing again.
    """
    if _chase_cooldown_until.get(rid, 0) > tick:
        return None
    start = _chase_start.get(rid)
    if start is not None and tick - start > CHASE_MAX_TICKS:
        _chase_start.pop(rid, None)
        _chase_cooldown_until[rid] = tick + CHASE_COOLDOWN_TICKS
        return None
    # Priority 1: hunt a visible OR remembered enemy Core near home when we
    # can afford to rebuild a Ranger if one falls. The known_enemy_cores set
    # persists across restarts so we don't forget the rival after a deploy.
    core_target = None
    if resources >= 60:
        visible_core = next(
            (e for e in enemies
             if e.kind == "CORE" and _manhattan(core_pos, e.position) < 40),
            None,
        )
        if visible_core is not None:
            core_target = tuple(visible_core.position)
        else:
            remembered = [
                cpos for cpos in _known_enemy_cores
                if _manhattan(core_pos, cpos) < 40
            ]
            if remembered:
                core_target = min(remembered, key=lambda c: _manhattan(pos, c))
    if core_target is not None:
        _chase_start[rid] = _chase_start.get(rid, tick)
        return core_target
    # Priority 2: drive off any visible enemy within CHASE_RADIUS of the Core.
    candidates = [
        e for e in enemies
        if _manhattan(core_pos, e.position) < CHASE_RADIUS
    ]
    if candidates:
        nearest = min(candidates, key=lambda e: _manhattan(pos, e.position))
        _chase_start[rid] = _chase_start.get(rid, tick)
        return tuple(nearest.position)
    # Priority 3: re-acquire a recently-seen enemy within the radius + slack.
    for epos, seen in _last_enemy_pos.values():
        if _manhattan(core_pos, epos) < CHASE_RADIUS + 6:
            _chase_start[rid] = _chase_start.get(rid, tick)
            return epos
    return None


def _process_events(turn: "Turn") -> None:
    """Read the previous Tick's resolution events.

    Per the bundled rules, ``turn.events`` explains how the new state came
    about and must not be replayed as patches. We use them only to inform the
    run observer (see play.py) and to surface notable outcomes; they do not
    change the queued plan.
    """
    resource_invalidated = False
    for event in turn.events:
        if (
            event.event_type == "HARVEST_FAILED"
            and event.position is not None
            and event.reason_code
            in {"RESOURCE_DEPLETED", "NOT_RESOURCE_CELL", "RESOURCE_NOT_FOUND"}
        ):
            # A failed harvest is stronger evidence than a remembered hint. It
            # confirms that the node is gone at resolution; if the next state
            # still shows a cargo pile or a refill at that coordinate,
            # _observe_resources will add it back as fresh current truth.
            cell = tuple(event.position)
            if cell in _known_resources or cell in _resource_hints:
                _known_resources.discard(cell)
                _resource_hints.pop(cell, None)
                resource_invalidated = True
        if event.event_type == "WORKER_CARGO_DROPPED":
            continue
        if event.harvest_source is HarvestSource.DROPPED_CARGO:
            continue
        if event.event_type == "CORE_DAMAGED":
            continue
    if resource_invalidated:
        _mark_persistent_state_dirty()


def _clear_exploration_state() -> None:
    """Reset per-worker route intents after a manual Core migration.

    Called when the Core finishes migrating. Route and frontier intents refer to
    the old neighborhood, but persistent resource and map facts remain valid
    until a fresh Turn explicitly disproves them.
    """
    _explore_state.clear()
    _explore_targets.clear()
    _explore_progress.clear()
    _explore_target_cooldown_until.clear()
    _explore_target_failures.clear()
    _pos_history.clear()
    _prev_pos.clear()
    _last_pos.clear()
    _stuck_ticks.clear()
def decide(turn: "Turn") -> None:
    """Queue a complete plan for one Turn based only on its authoritative state.

    This reads ``turn`` only and never retains controller objects across
    Turns. At most one action is queued per object; objects with no useful
    legal action are left to resolve as ``WAIT``.
    """
    _process_events(turn)
    # The user migrates the Core manually (often into obstacle corners). When a
    # move resolves, every sweep column and remembered resource points at the
    # old neighborhood — clear so workers re-sweep the new one instead of
    # clustering at stale cells near the moved Core.
    if any(e.event_type == "CORE_MOVE_SUCCEEDED" for e in turn.events):
        _clear_exploration_state()
    _sync_explore_state(turn)
    _observe_terrain(turn)
    _observe_resources(turn)
    _observe_enemies(turn)

    core = turn.core
    if core is None:
        # Respawning: submit no invented actions. An empty plan is a valid
        # complete replacement that leaves every object on WAIT.
        _flush_persistent_state()
        return

    core_pos = core.position
    threats = _threats_to_core(core_pos, turn.visible_enemies)
    # FINAL GOAL signal: a visible enemy Core is a strike target (elimination +
    # uncertain loot via CORE_RESOURCES_CAPTURED, not a flat +6). Mark it so the
    # Core controller forms a strike force (extra Vanguards/Rangers) to raid
    # it instead of only defending. Any visible enemy triggers prompt attacker
    # production (req: build attack units in time vs. other enemies).
    #
    # Enemy Cores expose `kind == "CORE"` (not `unit_type`); `_observe_enemies`
    # already latches seen Cores into `_known_enemy_cores` for bounded Ranger
    # tracking. Extra production is intentionally limited to a currently visible
    # Core: a stale permanent coordinate must not freeze Worker investment.
    enemy_core_visible = any(
        getattr(e, "kind", "") == "CORE" for e in turn.visible_enemies
    )
    # Defense first: react to visible nearby enemies and Core damage before
    # pursuing the economy or distant goals.
    # FINAL GOAL: hold the Beacon (champion-bonus harvest + Core shield cap 10)
    # when it is in vision — a worker/Ranger passing its cell picks it up on the
    # way through. The Core itself never travels (a migrating Core cannot act and
    # the Beacon is far from our cell), so Units are the picker-uppers.
    if (
        turn.beacon.status == BeaconStatus.GROUND
        and turn.beacon.position is not None
    ):
        beacon_pos = tuple(turn.beacon.position)
        for unit in turn.units:
            if unit.position == beacon_pos and str(unit.id) not in turn.plan:
                unit.pickup_beacon()
                break
    # Worker deposits happen before Unit HEAL and the Core action. Reserve the
    # exact full-heal costs in raw UUID order so later actions cannot knowingly
    # spend the same resources twice.
    available_resources = turn.resources + _pending_deposit_amount(turn)
    heal_ids, available_resources = _reserve_unit_heals(
        turn, core_pos, available_resources
    )
    _control_rangers(turn, core_pos, heal_ids=heal_ids)
    _control_vanguards(turn, core_pos, heal_ids=heal_ids)
    _control_core(
        turn,
        threats,
        enemy_core_visible=enemy_core_visible,
        available_resources=available_resources,
    )
    _control_workers(turn, core_pos)
    _flush_persistent_state()
