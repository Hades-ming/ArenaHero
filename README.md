# Arena Hero — balanced tactic

A resource-first balanced tactic for [Arena Hero](https://doc.arenahero.io/) v0.14,
built with the official [`arena-hero`](https://pypi.org/project/arena-hero/)
Python SDK. Tactics decisions are separated from the connection loop so they
can be tested without a live credential.

## Files

- `tactic.py` — `decide(turn)` queues one complete plan per Turn.
- `play.py` — connects to the live server, calls `decide`, and submits.
- `tests/test_tactic.py` — decision tests using `PlayerState` fixtures.
- `pyproject.toml` — pins `arena-hero>=0.2.9,<0.3`.

## Install

```bash
python -m pip install 'arena-hero>=0.2.9,<0.3'
python -m pip install pytest
```

## Run

```bash
# Set your key one of these ways, then run:
export ARENA_HERO_API_KEY="your-key"
python play.py

# or put it in .env:
echo 'ARENA_HERO_API_KEY="your-key"' > .env
python play.py

# or point at a file / local server:
python play.py --api-key-file ./arena-key.txt
python play.py --local
```

Stop with `Ctrl-C`. Watch the game at <https://app.arenahero.io/arena> signed
in with the same account that owns the key.

## Test

```bash
python -m pytest -q
```

## Policy

The tactic is continuously measured and is not claimed to be globally optimal:

- deposit carried Worker cargo when sharing the Core cell;
- harvest when an empty Worker stands on a currently visible resource cell;
- assign known resources to nearby empty Workers, while unassigned Workers explore;
- Rangers shoot visible legal cardinal or diagonal targets (Core prioritized);
- Vanguards sweep the adjacent cell with the most enemies, else hold near Core;
- repair Core shield only when under visible threat;
- maintain a Worker economy and standing combat reserve while treating population
  20 as the first dynamic-price boundary;
- leave an object on WAIT when no legal useful action is known.

Every numeric rule comes from the bundled Arena Hero v0.14 references; none is
inferred from memory.
