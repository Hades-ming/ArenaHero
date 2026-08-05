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
  spends into more Workers (free-upkeep fleet) so income compounds.
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
* spawn Workers toward a soft target while upkeep stays free (population < 20)
  and the Core cell has room;
* when enemies or an enemy Core are visible, prioritize attack-unit production;
* leave an object on WAIT when no legal useful action is known.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from arena_hero import (
    BeaconStatus,
    Direction,
    HarvestSource,
    UnitType,
)

if TYPE_CHECKING:
    from arena_hero import Core, CoreView, Turn, Unit, UnitView

# Population at and above which upkeep becomes a real cost. Upkeep is
# ``tier * (tier + 1) / 2`` where ``tier = floor(population / 20)``; the first
# tier (0) is free, so staying below 20 Units costs nothing.
FREE_UPKEEP_CAP = 20
# Comfortable Worker count the tactic tries to maintain. The fourth review
# found TARGET_WORKERS=16 was unreachable at the observed harvest rate AND
# counterproductive: every deposit was immediately spent on a Worker spawn,
# so Core storage never accumulated (the durable score proxy). The fourth
# review set this to 8 to let deposits bank. The FIFTH review (economy +
# skeptic personas) re-measured and found the binding constraint is node
# DISCOVERY rate (~0.033 res/tick) not the chunk-quota ceiling (2.0/tick,
# 34-64x headroom): more Workers find nodes faster, directly raising
# throughput, and upkeep is still 0 below pop 20. Raised 8 -> 12, then
# 12 -> 15 after r hit the pop-14 capacity ceiling (70): more Workers both
# raise discovery AND raise Core capacity (each Unit +5), letting r bank
# past 70. Still free-upkeep (pop 17 < 20). The bank reserve +
# army-short gate still prevent draining deposits to r0.
TARGET_WORKERS = 19
MAX_WORKERS = 21
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
# A Vanguard costs 10 and a Ranger 12 (rules). The fleet builds the standing
# reserve as soon as a minimal Worker economy (MIN_WORKERS_BEFORE_ARMY) exists,
# and BEFORE growing Workers past that floor — combat readiness outranks a
# larger Worker fleet once the economy can sustain the smallest army.
MIN_WORKERS_BEFORE_ARMY = 4
# Peacetime standing reserve now SCALES with the Worker fleet via
# _standing_army_targets (a floor of V1/R1, growing ~one combat pair per 8
# Workers up to the free-upkeep pop budget). These legacy constants document
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
HARVEST_REACH = 4


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


# Distance within which a Worker locks onto a visible resource and commits to
# walking onto it even across ticks where it leaves the small vision radius.
# Slightly larger than vision (3) so an edge node can be pursued.
HARVEST_LOCK_RANGE = 6
# Distance within which a Worker commits to a REMEMBERED resource (from the
# persistent _known_resources pool). The user can see map-wide resources the
# tactic has only partial vision of; a Worker parked 8-16 cells from a known
# node used to just keep exploring (observed: workers idled while user-visible
# resources sat uncollected). Commit to a known node within reach instead.
MAX_HARVEST_REACH = 30
# Max Core-to-resource distance worth harvesting. The alternative to locking a
# known node is IDLE EXPLORATION, which yields ~0 resources (workers only hit a
# node by chance). So committing to a known node within this radius is strictly
# better than milling around. Workers pick the NEAREST known node first, so
# close nodes are always taken before any distant one; a distant node is only
# approached by a worker that happens to be near it. Raised from 30 (user
# complaint: workers loitered near the Core while known resources sat
# uncollected — the old 30 radius + a 30-cell worker lock gate meant most
# workers were too far to ever commit and just swept aimlessly).
MAX_HARVEST_FROM_CORE = 40
# Wider band used when the economy is STARVED: nearby-but-distant nodes that are
# a net loss in a healthy economy become worth taking when nothing closer exists
# (expert review L2: late-game the Core sat at r0/95 because the only visible
# nodes were d=41-46 and were hard-filtered). Keeps income alive instead of idling.
MAX_HARVEST_FROM_CORE_STARVED = 65
# If no harvest succeeds for this many ticks, treat the economy as starved and
# permit harvesting out to MAX_HARVEST_FROM_CORE_STARVED.
STARVE_TICKS = 50
# Tick of the most recent successful harvest (module-level, reset on Core move).
_last_harvest_tick: int = -10**9
# Sweep radius cap: a worker explores within this Manhattan radius of the Core
# only. Beyond it the deposit round trip is a net loss and workers were observed
# stranding 60-88 cells out (never returning). Steers back toward the Core.
MAX_SWEEP_RADIUS = 40

# Per-Worker exploration memory, keyed by the Unit UUID string. Each entry is
# [direction_index_into_DIRECTIONS, steps_taken_in_this_leg]. This is not a
# retained controller object (the skill forbids that); it is a small intent
# hint that is rebuilt from the current live Units each Tick.
_explore_state: dict[str, list[int]] = {}
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

# 9th review rank 1: per-Worker lock-progress watchdog. The anti-backtrack
# avoid-set + an obstacle box can defeat greedy _step_toward pursuit of a valid
# harvest lock (c8c1c2 4-cycled 20+ ticks around (28,247), never reaching it,
# r frozen 42 ticks). Track the current lock, the Manhattan distance when the
# worker locked it, and the lock tick. If the SAME lock is held >
# LOCK_STALL_TICKS without the worker getting closer, drop the lock and clear
# _pos_history so the A* path can form a fresh route instead of re-cycling.
_lock_meta: dict[str, tuple[tuple[int, int], int, int]] = {}  # wid -> (lock, dist_at_lock, tick)
LOCK_STALL_TICKS = 15

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
    global _known_resources, _known_obstacles, _known_enemy_cores, _explored_cells
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        raw = data.get("known_resources", [])
        if isinstance(raw, list):
            _known_resources = {
                (int(a), int(b))
                for a, b in raw
                if isinstance(a, int) and isinstance(b, int)
            }
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
    except (OSError, ValueError, TypeError):
        _known_resources = set()
        _known_obstacles = set()
        _known_enemy_cores = set()
        _explored_cells = set()


def _save_persistent_state() -> None:
    """Persist resource/obstacle/enemy-core/explored memory (on change)."""
    try:
        payload = {
            "known_resources": [list(c) for c in _known_resources],
            "known_obstacles": [list(c) for c in _known_obstacles],
            "known_enemy_cores": [list(c) for c in _known_enemy_cores],
            "explored_cells": [list(c) for c in _explored_cells],
        }
        _STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


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
        _save_persistent_state()
        _last_saved_explored = len(_explored_cells)
    elif len(_explored_cells) - _last_saved_explored >= _EXPLORED_SAVE_STEP:
        # The explored map is the part the user actually wanted persisted; flush
        # it in coarse batches so a play.py restart resumes near where it left off
        # instead of re-scanning from scratch.
        _save_persistent_state()
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


def _astar_step(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: frozenset[tuple[int, int]],
    blocked: frozenset[tuple[int, int]],
    max_expansions: int = 4000,
) -> Direction | None:
    """Return the first cardinal step on an A* path from ``start`` to ``goal``.

    The greedy _step_toward wedges in obstacle-dense terrain: a laden Worker
    returning to a Core parked in a stone corner can spin 20+ cells out, never
    finding the gap (observed 67512f/d48045). A* guarantees a path when one
    exists. ``obstacles`` are permanent terrain; ``blocked`` are dynamic
    (friendly-full/enemy cells) and the ``goal`` is always enterable (a laden
    Worker must step onto the Core cell to deposit).
    """
    if start == goal:
        return None
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
            if nxt in obstacles or (nxt in blocked and nxt != goal):
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
        return None
    cursor = goal
    while came_from.get(cursor) != start:
        parent = came_from.get(cursor)
        if parent is None:
            return None
        cursor = parent
    ddx = cursor[0] - start[0]
    ddy = cursor[1] - start[1]
    for d in DIRECTIONS:
        if d.delta == (ddx, ddy):
            return d
    return None


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
) -> Direction | None:
    """Return one chunk-anchored boustrophedon step for a Worker.

    The Worker owns a vertical column (an x coordinate) anchored to the CHUNK,
    not the Core, and sweeps it north/south edge-to-edge across the chunk,
    shifting one column at the y-boundary. The third review found that
    Core-anchored bands blind the chunk half opposite the Core's offset; chunk
    anchoring guarantees full-width coverage regardless of Core position.
    Turning is dictated by ABSOLUTE chunk-y position. State:
    [col_offset, going_south_flag, lock_x, lock_y].

    ``target_col`` overrides the Worker's column when set (chemotaxis).
    ``force_band`` overrides the chunk column assignment.
    """
    chunk_x0, chunk_y0 = _chunk_origin(core_pos)
    chunk_y1 = chunk_y0 + CHUNK_SIZE - 1
    # 10th review rank 1: extend the sweep beyond the home chunk so edge
    # workers tile the reachable aprons of the four neighbor chunks (virgin
    # ground; the home chunk is exhausted). Collecting still respects
    # MAX_HARVEST_FROM_CORE, so far-apron cells are lit but not locked.
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
    # Sweep-radius cap: a worker must not wander beyond economic reach of the
    # Core (the deposit sink). Past ~40 cells the delivery round trip is a net
    # loss AND the worker never returns — observed workers stranded 60-88 cells
    # out (x 30-44, y 276-297) while north resources near the Core sat
    # uncollected. Steer back toward the Core until back in range.
    # 9th review rank 3: use A* (same laden-return pattern) + clear anti-
    # backtrack history so the avoid-set cannot re-enter the obstacle corner trap
    # (67512f oscillated 70+ ticks at d=67-71 around the enemy Core at (38,278)).
    if _manhattan(pos, core_pos) > MAX_SWEEP_RADIUS:
        _pos_history.pop(worker_id, None)
        _prev_pos.pop(worker_id, None)
        step = _astar_step(pos, core_pos, blocked, blocked)
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
    between. Enemy Cores are prioritized: destroying one removes the enemy fleet
    and can capture its stockpiled resources (variable loot, not a flat +6; see
    ``CORE_RESOURCES_CAPTURED``). Among Units, prefer a one-shot-killable (hp==1)
    and a FLEEING
    target (farther from the Core than its last-known position) so driven-off
    raiders are finished, not let to escape (8th review, rank 3).
    """
    best: UnitView | CoreView | None = None
    best_key: tuple[int, int, int, int, str] | None = None
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
        fleeing = 1
        if last is not None:
            last_pos, _ = last
            if _manhattan(cell, core_pos) > _manhattan(last_pos, core_pos):
                fleeing = 0  # moving away from the Core = escaping
        key = (0 if is_core else 1, finishable, fleeing, dist, str(enemy.id))
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
    # Chemotaxis DISABLED (fourth review, coverage+strategy personas): the
    # chunk-anchored column assignment already tiles the full chunk width, and
    # chemotaxis collapsed the fleet onto the single nearest visible node's
    # column whenever one appeared, abandoning coverage of the far half (where
    # most nodes actually sat). The node is usually harvested by the lowest-UUID
    # Worker before others arrive, so convergence wastes the followers' ticks.
    # Workers now stay on their assigned columns and harvest nodes they
    # encounter via the sweep + the persistent harvest lock.
    chemotaxis_col: int | None = None
    # Distributed resource claims: each KNOWN resource is claimed by at most
    # one empty Worker this Tick, so workers fan out across the pool instead of
    # all converging on the single nearest node (observed: workers clustered on
    # the south resource while north nodes (13,214)/(16,211) went uncollected).
    claims: dict[tuple[int, int], int] = {}
    # L12: sort Workers by distance to their nearest KNOWN resource so that
    # closer Workers are processed first and claim nearby nodes ahead of far
    # competitors — prevents first-come-first-served where an early, distant
    # Worker (idx 0 at -6,0) locks a node at a near Worker's feet (idx 1 at 5,0
    # for resource at 6,0).  orig_index is preserved for column/direction parity
    # (_explore_step / _begin_outbound) so a Worker's sweep band stays stable.
    _known_snap = frozenset(_known_resources)
    _sorted_workers: list[tuple[int, object]] = sorted(
        enumerate(turn.workers),
        key=lambda x: (
            x[1].cargo > 0,
            min(_manhattan(x[1].position, r) for r in _known_snap)
            if _known_snap and x[1].cargo == 0
            else 10**18,
            x[0],
        ),
    )
    for _srt_pos, (orig_index, worker) in enumerate(_sorted_workers):
        pos = worker.position
        wid = str(worker.id)
        # An EMPTY Worker standing on a visible resource cell harvests
        # IMMEDIATELY, before the boxed-in/STUCK logic that could otherwise
        # shuttle it away in a move (observed: worker 68a41e parked on
        # (11,247) — a visible resource — yet got boxed-escaped into a move
        # instead of harvesting, and drifted off uncollected).
        # 9th review rank 2: gate the on-cell harvest by Core distance. The
        # lock path already refuses nodes beyond MAX_HARVEST_FROM_CORE, but
        # this on-cell path fired unconditionally — 3ae099 harvested (35,251)
        # at d=37 and spent 53+ ticks hauling cargo 1 home (a net loss). A
        # worker standing on a node beyond the economic radius skips it and
        # stays in the sweep fleet; a closer worker finds it or it refills.
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
            and _manhattan(core_pos, pos) <= _harvest_radius(turn)
        ):
            worker.harvest()
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
        if _is_boxed_in(wid):
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
                        target_col=chemotaxis_col, avoid=_avoid_set(wid),
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
                )
                if step is None:
                    step = _step_toward(pos, core_pos, blocked, avoid=_avoid_set(wid))
                    if step is None:
                        step = _step_toward(pos, core_pos, blocked, avoid=None)
                        _pos_history.pop(wid, None)
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
        # 9th review rank 2: gate by Core distance (same as the immediate
        # harvest at L940) so a distant node is left for discovery/sweep.
        if pos in resource_cells and _manhattan(core_pos, pos) <= _harvest_radius(turn):
            worker.harvest()
            continue

        # Persistent harvest lock: commit to the nearest KNOWN resource and
        # keep walking toward it even on ticks where it leaves this Worker's
        # small vision, until the Worker stands on the cell. This kills the
        # flicker 2-cycle where a Worker steps toward an edge-of-vision node,
        # loses sight of it next tick, and steps back. State index 2,3 hold the
        # locked target coordinates (None when unlocked).
        st = _explore_state.get(wid)
        lock = (st[2], st[3]) if st is not None and len(st) >= 4 and st[2] is not None else None
        # Candidate targets: currently-visible resources PLUS the local memory
        # pool (remembered cells that left vision). The lock distance is larger
        # for KNOWN resources (20 cells) than for currently-visible ones (6),
        # so a Worker parked ~8-16 cells from a known resource commits to it
        # rather than blindly exploring (observed: workers idled while the user
        # saw map-wide resources uncollected).
        known = frozenset(_known_resources)
        # Claim exemption: if this worker ALREADY holds a valid lock on a known
        # resource (and it's still economically reachable), keep it — another
        # worker must not yank a mid-walk target (community Player C's pacing
        # anti-pattern: "a far worker steals the node at a near worker's feet").
        if (
            lock is not None
            and lock in known
            and lock not in claims
            and _manhattan(core_pos, lock) <= _harvest_radius(turn)
        ):
            claims[lock] = orig_index + 1
        else:
            if lock in claims:
                lock = None
                if st is not None and len(st) >= 4:
                    st[2] = None
                    st[3] = None
            # Sort KNOWN resources by distance to this worker. Pick the nearest
            # one that has NOT already been claimed by another empty worker —
            # this fans the fleet across the pool instead of converging on the
            # single closest node.
            visible = sorted(
                known if known else resource_cells,
                key=lambda c: _manhattan(pos, c),
            )
            if visible:
                for candidate in visible:
                    if candidate in claims:
                        continue
                    nearest = candidate
                    # For a KNOWN resource, commit from anywhere within the
                    # economic harvest radius (the worker may be far; A* walks it
                    # there). For a merely VISIBLE resource, keep the tight
                    # HARVEST_LOCK_RANGE so a Worker does not detour for an
                    # edge-of-vision node it may lose next tick. The old code used
                    # MAX_HARVEST_REACH (30) as the worker-distance cap, which left
                    # most of the fleet too far to ever commit — they just swept
                    # near the Core while known resources sat uncollected.
                    lock_dist = (
                        max(MAX_HARVEST_REACH, _harvest_radius(turn))
                        if nearest in known
                        else HARVEST_LOCK_RANGE
                    )
                    # Skip nodes too far from the Core — the round trip is a net
                    # loss and stalls the economy in a healthy state. When starved,
                    # the radius widens so a distant node is still taken rather
                    # than idling.
                    if _manhattan(core_pos, nearest) > _harvest_radius(turn):
                        continue
                    if _manhattan(pos, nearest) <= lock_dist:
                        lock = nearest
                        claims[lock] = orig_index + 1
                        break
        # If we have a lock, keep moving toward it; abandon it only once the
        # Worker is on the cell but it is not a resource (harvested or gone).
        if lock is not None:
            if pos == lock:
                # On the locked cell: if it's still a visible resource, harvest;
                # otherwise the node was consumed/contested — clear and explore.
                if pos in resource_cells:
                    worker.harvest()
                    continue
                lock = None
            else:
                # Revalidate the lock each tick (9th review, rank 4): if the
                # locked cell is no longer a known/visible resource (e.g.
                # harvested by another worker), drop it immediately instead of
                # walking up to 20 cells into a dead end.
                if lock not in _known_resources and lock not in resource_cells:
                    if st is not None and len(st) >= 4:
                        st[2] = None
                        st[3] = None
                        _explore_state[wid] = st
                    lock = None
                else:
                    if st is None or len(st) < 4:
                        st = [0, 1, lock[0], lock[1]]
                    else:
                        st[2], st[3] = lock[0], lock[1]
                    _explore_state[wid] = st
                    # 9th review rank 1: the anti-backtrack avoid-set + obstacle
                    # box defeats greedy _step_toward on the lock approach — c8c1c2
                    # 4-cycled 20+ ticks around (28,247), never reaching it, and r
                    # froze 42 ticks while the node stayed visible. Use the proven
                    # laden-return A* pattern (L1067) with a greedy fallback.
                    step = _astar_step(pos, lock, base_blocked, blocked_empty)
                    if step is None:
                        step = _step_toward(pos, lock, blocked_empty, avoid=_avoid_set(wid))
                    if step is not None:
                        # Watchdog: if this lock has been held > LOCK_STALL_TICKS
                        # without the worker getting closer, the avoid-set is
                        # re-cycling the same box. Drop the lock AND clear the
                        # position history so a fresh A* route can form next tick.
                        meta = _lock_meta.get(wid)
                        if meta is None or meta[0] != lock:
                            _lock_meta[wid] = (lock, _manhattan(pos, lock), turn.tick)
                        else:
                            _, dist_at_lock, lock_tick = meta
                            if (
                                turn.tick - lock_tick > LOCK_STALL_TICKS
                                and _manhattan(pos, lock) >= dist_at_lock
                            ):
                                _lock_meta.pop(wid, None)
                                _pos_history.pop(wid, None)
                                _explore_state.pop(wid, None)
                                lock = None
                                step = None
                    if step is not None:
                        _prev_pos[wid] = pos
                        _record_pos(wid, pos)
                        worker.move(step)
                    continue
        # No lock and no nearby resource: explore using the chunk-anchored
        # boustrophedon. Pass the fleet size and Worker index so columns are
        # distributed evenly across the full chunk width.
        step = _explore_step(
            orig_index, wid, pos, core_pos, blocked_empty,
            target_col=chemotaxis_col, avoid=_avoid_set(wid),
            fleet_size=len(turn.workers),
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


def _control_vanguards(turn: "Turn", core_pos: tuple[int, int]) -> None:
    enemies = turn.visible_enemies
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    # Count friendly occupants per cell so a Vanguard steps to a cell that is
    # not already at capacity (2/2) and, when possible, not onto a laden Worker
    # that needs the Core-cell path clear to deposit.
    from collections import Counter as _Counter
    friendly_occ: dict[tuple[int, int], int] = _Counter(
        tuple(u.position) for u in turn.units
    )
    laden_positions = frozenset(
        tuple(w.position) for w in turn.workers if w.cargo > 0
    )
    enemy_positions = frozenset(e.position for e in enemies)
    for vanguard in turn.vanguards:
        sweep_dir = _vanguard_sweep_target(vanguard.position, enemies)
        if sweep_dir is not None:
            vanguard.sweep(sweep_dir)
            continue
        # With nothing to sweep, hold near the Core to body-block raiders.
        if vanguard.position == core_pos:
            # The Vanguard spawned on the Core cell (Core + Vanguard = 2/2
            # capacity), which BLOCKS laden Workers from stepping onto the
            # Core cell to deposit (UNIT_MOVE_FAILED.CELL_UNIT_LIMIT). Step
            # off to an adjacent open cell so deposits can resume while still
            # body-blocking raiders next to the Core. Prefer a cell that is
            # empty (no friendly occupant) and not occupied by a laden Worker,
            # so the Vanguard does not re-block the deposit path.
            best: Direction | None = None
            best_key: tuple[int, int] | None = None
            for direction in DIRECTIONS:
                ddx, ddy = direction.delta
                nxt = (core_pos[0] + ddx, core_pos[1] + ddy)
                if nxt in obstacles or nxt in enemy_positions:
                    continue
                if friendly_occ.get(nxt, 0) >= 2:
                    continue  # cell full, cannot enter
                is_laden_cell = nxt in laden_positions
                occ = friendly_occ.get(nxt, 0)
                # Rank: prefer non-laden, then fewer occupants.
                key = (1 if is_laden_cell else 0, occ)
                if best_key is None or key < best_key:
                    best_key, best = key, direction
            if best is not None:
                vanguard.move(best)
                continue
        if _manhattan(vanguard.position, core_pos) > 1 and turn.core is not None:
            # Approach the Core body-block WITHOUT stepping onto the Core cell
            # (blocks deposits) or into a full 2/2 friendly cell (8th review,
            # rank 4). friendly_full is computed from the same occupancy
            # counter above.
            friendly_full = frozenset(
                cell for cell, count in friendly_occ.items() if count >= 2
            )
            step = _step_toward(
                vanguard.position, core_pos,
                obstacles | friendly_full | {core_pos},
            )
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
    for d in DIRECTIONS:
        nxt = (pos[0] + d.delta[0], pos[1] + d.delta[1])
        if nxt in blocked:
            continue
        nd = _manhattan(nxt, core_pos)
        if inner <= nd <= outer:
            if abs(nd - dist) < best_keep:
                best_keep = abs(nd - dist)
                best = d
    return best


def _can_shoot(
    shooter: tuple[int, int],
    target: tuple[int, int],
    obstacles: frozenset[tuple[int, int]],
) -> bool:
    """True if a Ranger at ``shooter`` can legally shoot ``target``.

    Rules (v0.13): shared cardinal or exact-diagonal line, range 1-3, with no
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


def _control_rangers(turn: "Turn", core_pos: tuple[int, int]) -> None:
    obstacles = frozenset(turn.obstacle_cells) | _known_obstacles
    enemies = turn.visible_enemies
    for index, ranger in enumerate(turn.rangers):
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
            step = _guard_step(
                ranger.position, core_pos,
                obstacles, obstacles,
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
            step = _step_toward(
                pos, chase,
                obstacles | enemy_cells,
                avoid=_avoid_set(rid),
            )
            if step is None:
                step = _step_toward(pos, chase, obstacles, avoid=None)
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
    then shrunk to fit the free-upkeep population budget (W + V + R <= 19,
    i.e. budget = FREE_UPKEEP_CAP - 1) so growth never overflows into upkeep
    tier 1.
    """
    pairs = max(1, n_workers // 8)
    vanguards = pairs
    rangers = pairs
    budget = FREE_UPKEEP_CAP - 1
    while n_workers + vanguards + rangers > budget:
        if vanguards > rangers and vanguards > 1:
            vanguards -= 1
        elif rangers > 0:
            rangers -= 1
        else:
            # At the W=19 edge with V=1, R=0: total = 20, exceeds budget.
            # Clamp vanguard floor to 0 for this single edge case so the
            # invariant W+V+R <= 19 holds; the army will be rebuilt as soon
            # as a Worker dies or later when the army target is applied
            # dynamically (budget-aware wants_worker in _control_core).
            if vanguards == 1 and rangers == 0:
                vanguards = 0
            break
    return vanguards, rangers


def _control_core(
    turn: "Turn",
    threats: list[UnitView | CoreView],
    enemy_core_visible: bool = False,
) -> None:
    core = turn.core
    if core is None:
        return
    # A migrating Core cannot spawn, repair, or receive deposits; let the move
    # resolve rather than queue an action that would fail with CORE_ALREADY_MOVING.
    if core.view.state == "MOVING":
        return

    resources = turn.resources
    if (
        turn.beacon.status == BeaconStatus.GROUND
        and turn.beacon.position == core.position
    ):
        core.pickup_beacon()
        return
    # Repair shield first when under threat and there is space and a spare
    # resource. Holding the Beacon raises the cap to 10, so use the live cap.
    if threats and resources >= 1:
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

    # Spawn Workers toward the target fleet so the economy grows and explores
    # faster, staying in the free-upkeep band (population < 20). Only spawn
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
    if population >= FREE_UPKEEP_CAP - 1:
        return
    # Deposits resolve before spawn in the same Tick (rules: resolution order),
    # so a colocated Worker's cargo counts toward the spawn cost.
    pending_deposit = sum(
        w.cargo for w in turn.workers if w.position == core.position
    )
    effective_resources = resources + pending_deposit

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
        # capped by the free-upkeep population budget below 20.
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
    army_floor_met = len(turn.workers) >= (
        MIN_WORKERS_BEFORE_ARMY if not enemy_present else 2
    )

    if economy_floor_met or army_floor_met:
        # Rangers are the strongest defender (range-3 return fire from a
        # 5-vision scout) but cost 12, so build the cheap Vanguard body-block
        # first, then the Ranger.
        wants_vanguard = len(turn.vanguards) < target_vanguards
        wants_ranger = len(turn.rangers) < target_rangers
        if wants_vanguard and effective_resources >= 10:
            core.spawn(UnitType.VANGUARD)
            return
        if wants_ranger and effective_resources >= 12:
            core.spawn(UnitType.RANGER)
            return

    # Budget-aware Worker target (6th review, strategy STRAT-5): the effective
    # ceiling is the free-upkeep population budget minus the standing army's pop
    # cost. A fixed TARGET_WORKERS=19 ignored the army (e.g. V1R3 leaves room
    # for only 15 Workers) and pushed the fleet into tier-1 upkeep. TARGET_WORKERS
    # remains as a legacy constant for the never-reached pre-army ceiling.
    worker_target = FREE_UPKEEP_CAP - 1 - (
        len(turn.vanguards) + len(turn.rangers)
    )
    wants_worker = len(turn.workers) < worker_target
    army_short = (
        len(turn.vanguards) < target_vanguards
        or len(turn.rangers) < target_rangers
    )
    # Above the economy floor (and whenever an enemy is present), if the combat
    # reserve is STILL short, do NOT spend 5 on another Worker — bank the
    # resource toward the 10/12 combat Unit instead, or the economy stalls at
    # ~5 and never affords return fire (the exact trap that lost the Core on
    # 2026-08-02).
    if (economy_floor_met or enemy_present) and army_short:
        return
    # Bank reserve: only spawn a Worker if the Core keeps at least
    # WORKER_SPAWN_RESERVE resources afterward, so the economy never drains to
    # zero and the standing-army bank is not reset each spawn.
    if wants_worker and effective_resources >= 5 + WORKER_SPAWN_RESERVE:
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
    for uid in list(_lock_meta):
        if uid not in live:
            del _lock_meta[uid]


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
    global _known_resources
    # Remove cells harvested this Tick (the event carries the harvest cell).
    for event in turn.events:
        if event.event_type == "HARVEST_SUCCEEDED" and event.position is not None:
            _known_resources.discard(tuple(event.position))
    # The complete current state is authoritative over previous-Tick events.
    # A dropped-cargo pile can remain after a partial harvest, and a natural
    # node can refill at the same coordinate.
    _known_resources.update(turn.resource_cells)
    # Remove cells that a friendly vision source can see and are confirmed
    # not to hold a resource (visible but not in turn.resource_cells).
    sources = _vision_sources(turn)
    obstacles = turn.obstacle_cells
    for cell in list(_known_resources):
        if cell in turn.resource_cells:
            continue  # still a resource, keep it
        if _any_vision_sees(cell, sources, obstacles):
            _known_resources.discard(cell)
    # Prune cells far beyond the Core (Chebyshev > 80). When the Core migrates
    # (the user does this manually), stale cells from the abandoned chunk
    # persist in the pool and workers detour to them instead of sweeping fresh
    # territory (7th review, MIG-3). Loose bound: a normal chunk-anchored sweep
    # never triggers it, a migration does.
    core = turn.core
    if core is not None:
        cx, cy = core.position
        _known_resources = {
            cell
            for cell in _known_resources
            if max(abs(cell[0] - cx), abs(cell[1] - cy)) < 80
        }
        # Economy-pool hygiene (9th review, rank 5): drop nodes so far from the
        # Core they are never collectable (the round trip is a net loss). They
        # are re-added if a vision source re-lights them later. Prevents a
        # sparse pool from pulling an edge Worker into a wasted approach. Bound
        # matches the widest harvest radius (starved) so a node the tactic WILL
        # target is never pruned from memory before a Worker can reach it.
        _known_resources = {
            cell
            for cell in _known_resources
            if _manhattan((cx, cy), cell) <= MAX_HARVEST_FROM_CORE_STARVED
        }
    _save_persistent_state()


def _observe_enemies(turn: "Turn") -> None:
    """Record last-seen enemy positions and prune stale memory.

    Feeds the bounded drive-off for non-guard Rangers (8th review, rank 2):
    with only current-visible enemies, a raider parked just outside range is
    never driven off. Remembering positions lets Rangers chase briefly.
    """
    tick = turn.tick
    for e in turn.visible_enemies:
        _last_enemy_pos[str(e.id)] = (tuple(e.position), tick)
        # Persist enemy Cores permanently (a Core is a durable hunt target).
        if e.kind == "CORE":
            _known_enemy_cores.add(tuple(e.position))
    for eid in list(_last_enemy_pos):
        _, seen = _last_enemy_pos[eid]
        if tick - seen > ENEMY_MEMORY_TICKS:
            del _last_enemy_pos[eid]
    if _known_enemy_cores:
        _save_persistent_state()


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
    for event in turn.events:
        if event.event_type == "WORKER_CARGO_DROPPED":
            continue
        if event.harvest_source is HarvestSource.DROPPED_CARGO:
            continue
        if event.event_type == "CORE_DAMAGED":
            continue
        # Track the last successful harvest so the harvest radius can widen when
        # the economy goes starved (no income for a while).
        if event.event_type == "HARVEST_SUCCEEDED":
            global _last_harvest_tick
            _last_harvest_tick = turn.tick


def _clear_exploration_state() -> None:
    """Reset all per-worker exploration memory and the resource pool.

    Called when the Core finishes migrating (the user moves it manually into
    obstacle corners). After a move, the memory pool and sweep state refer to
    the abandoned neighborhood — workers detour to stale harvested cells and
    cluster near the Core instead of sweeping fresh territory (observed: 15
    workers parked within a few cells of a moved Core, r stuck for hundreds
    of ticks). Clearing forces a clean re-sweep of the new neighborhood.
    """
    _known_resources.clear()
    _explore_state.clear()
    _pos_history.clear()
    _prev_pos.clear()
    _last_pos.clear()
    _stuck_ticks.clear()
    _lock_meta.clear()
    global _explored_cells, _last_saved_explored, _last_harvest_tick
    # The explored map belongs to the abandoned neighborhood too; drop it so the
    # next sweep re-lights fresh ground instead of trusting stale cells, and
    # write the cleared state through so a later restart can't reload it.
    _explored_cells.clear()
    _last_saved_explored = 0
    _last_harvest_tick = -10**9
    _save_persistent_state()


def _harvest_radius(turn: "Turn") -> int:
    """Effective Core-to-resource harvest radius for this Turn.

    Healthy economy: ``MAX_HARVEST_FROM_CORE`` (40) — distant nodes are a net
    loss. Starved economy (no harvest for ``STARVE_TICKS`` ticks, no known
    resources at all, OR every known resource already beyond the non-starved
    radius): widen to ``MAX_HARVEST_FROM_CORE_STARVED`` (65) so the fleet
    keeps earning instead of idling at r0 while visible resources sit at
    40 < d <= 65 (L12 — occasional harvests from sweep collisions kept
    _last_harvest_tick fresh, preventing starved mode, but all lockable
    resources were beyond the non-starved gate).
    """
    if not _known_resources:
        return MAX_HARVEST_FROM_CORE_STARVED
    if turn.tick - _last_harvest_tick > STARVE_TICKS:
        return MAX_HARVEST_FROM_CORE_STARVED
    # L12: if every known resource is beyond the non-starved radius, treat
    # the economy as effectively starved — workers idly sweep while the only
    # nodes they could lock sit at 43-48 cells and get filtered by the
    # core-distance gate below. The starved radius lets the lock-claim loop
    # (L1253-1280) actually commit to them instead of skipping.
    core = turn.core
    if core is not None:
        if not any(
            _manhattan(core.position, r) <= MAX_HARVEST_FROM_CORE
            for r in _known_resources
        ):
            return MAX_HARVEST_FROM_CORE_STARVED
    return MAX_HARVEST_FROM_CORE


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
    _control_rangers(turn, core_pos)
    _control_vanguards(turn, core_pos)
    _control_core(turn, threats, enemy_core_visible=enemy_core_visible)
    _control_workers(turn, core_pos)
