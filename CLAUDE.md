# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

RoboChess simulates a Fetch robotic arm playing chess against a human inside a MuJoCo physics environment. The human plays White via a Tkinter GUI; the AI plays Black using Stockfish for move selection and a pretrained TQC (Soft Actor-Critic variant) policy for physical arm control.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive game (opens MuJoCo viewer + Tkinter GUI)
python main.py

# Headless smoke test (single manipulation op, no viewer)
python scripts/headless_run.py

# Verify MuJoCo installation
python scripts/verify_mujoco.py

# Check piece stability
python scripts/drift_env_test.py
```

## Tests

```bash
# Run all tests
export PYTHONPATH=.
pytest

# Run a single test file
pytest tests/test_chess_manager.py
```

## Architecture

The system has a modular 6-layer architecture:

### 1. Entrypoint & Runtime (`main.py`, `src/app.py`, `src/bootstrap.py`)
`main.py` is a thin wrapper calling `src.app.main()`. `src/bootstrap.py` handles the startup sequence and returns a `GameSystems` dataclass. The GUI runs on the main thread; MuJoCo and the game worker run on a daemon thread.

### 2. GUI (`src/gui/board_panel.py`)
`BoardPanel` renders the board and handles click-based input. It supports a **Step Mode** (wait for user before AI moves) and **Hint** requests.

### 3. Game Orchestration (`src/game_loop.py`, `src/turn_executor.py`)
`GameLoop` owns the turn cycle and turn-boundary health checks. `src/turn_executor.py` handles the physical execution of chess operations (`PickPlaceOp`).

### 4. Chess Logic (`src/logic/`)
- `ChessGameManager` — owns the `python-chess` board and Stockfish engine.
- `OperationPlanner` — translates a `chess.Move` into ordered `PickPlaceOp`s.

### 5. Physical Execution (`src/control/execution_controller.py`, `src/env/`)
`ExecutionController` runs a consolidated 11-stage Cartesian pipeline using three execution modes:
- `SCRIPTED`: Pure Cartesian waypoints.
- `RL_PRIMARY`: RL policy drives completely with scripted fallback on stall.
- `RL_CLAMPED`: RL drives with XY displacement clamped for vertical descent.

### 6. Health Checks & Guards (`src/health_checks.py`, `src/runtime_guard.py`)
`RuntimeCheckRegistry` runs pluggable checks (integrity, mapping, stability) at lifecycle hooks. `freeze_on_exception` logs fatal errors and freezes the sim for inspection with a configurable timeout.

## Physics Configuration (Final Production)
To ensure piece stability while maintaining robot control, the simulation uses:
- **Timestep:** 1ms (`physics_timestep: 0.001`) with 40 substeps.
- **Integrator:** Euler (for robot stability) + `noslip_iterations=100` (for piece stability).
- **Piece Profile:** 0.25kg mass with 10.0 freejoint damping.

## Code Conventions
- All tunable parameters live in `src/settings/runtime.yaml`.
- Never hardcode values that are in the YAML; use `src/config.py` exports.
- Piece mesh collisions are disabled by default (`contype=0`); pieces use cylinder hitboxes for board interaction.
