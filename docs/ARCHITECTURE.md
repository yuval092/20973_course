# RoboChess Architecture

## Overview

RoboChess is split into five layers:

1. Runtime orchestration
   - `src/game_runtime.py`
   - Boots the scene, loads the policy, owns the game loop, and dispatches runtime checks.
2. Manipulation control
   - `src/control/execution_controller.py`
   - Executes staged pick-and-place operations and emits arm stage lifecycle events.
3. Environment and scene access
   - `src/env/chess_pick_place_env.py`
   - Wraps the MuJoCo scene using Fetch-style observation/action semantics.
4. Validation and health monitoring
   - `src/health_checks.py`
   - `src/runtime_guard.py`
   - `src/observation/board_observer.py`
5. YAML-backed configuration and training prep
   - `src/config.py`
   - `src/settings/runtime.yaml`
   - `src/settings/training.yaml`
   - `src/training/*`
6. Project utilities
   - `scripts/generate_xml.py`
   - `scripts/verify_mujoco.py`
   - `scripts/headless_run.py`

`main.py` is only a CLI entrypoint.

## Runtime Flow

Startup sequence:

1. `load_scene()`
2. `ChessPickPlaceEnv(...)`
3. `POST_SCENE_LOAD` checks
4. `load_rl_policy()`
5. `ChessGameManager`, `OperationPlanner`, `ExecutionController`
6. `initialize_square_to_piece()`
7. `PROGRAM_START` checks
8. viewer launch

Turn sequence:

1. `TURN_START` checks
2. human turn or AI turn
3. arm stage callbacks during AI execution
4. `TURN_END` checks
5. viewer sync

If any exception escapes, `freeze_on_exception()` logs it and freezes the sim for inspection.

## Runtime Checks

Runtime checks are pluggable classes registered in `RuntimeCheckRegistry`.

Current default checks:

- scene assets present
- mapping integrity
- board agreement
- arm home pose
- workspace containment
- finite simulation state
- piece observer stability
- stage goal reachability
- stage outcome success
- grip attachment after close/lift

Checks raise typed exceptions, so they are testable and fail-fast.

## Logging

Runtime logging now follows a stricter split:

- `DEBUG` for step-level controller and environment diagnostics
- `INFO` for boot, move, and stage lifecycle events
- `WARNING` for degraded-but-continuing conditions such as workspace issues
- `ERROR` for fatal runtime failures

## Fine-Tuning Prep

Fine-tuning preparation code lives in:

- `src/training/config.py`
- `src/training/tasks.py`
- `src/env/chess_manipulation_train_env.py`

These modules do not run training yet. They provide task definitions, curriculum configuration, and an environment wrapper for binding manipulation tasks onto the current MuJoCo scene.
