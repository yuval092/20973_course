# RoboChess Project Guide

## 1. Project Purpose

RoboChess is a MuJoCo-based robotic chess system in which:

- White is controlled by a human through UCI move input
- Black is controlled by an AI chess engine
- Black's physical piece movement is executed by a Fetch-style robotic arm controller
- The simulation is validated continuously through runtime checks

The project combines:

- a logical chess engine layer
- a MuJoCo scene and physics layer
- a robotic manipulation controller
- runtime validation infrastructure
- test coverage for startup, gameplay, arm execution, and failure handling

## 2. Core Modules

### Entry point

- `main.py`
  Thin CLI entrypoint. It delegates all real runtime behavior to the runtime module.

### Runtime orchestration

- `src/game_runtime.py`
  Owns:
  - scene boot
  - model boot
  - environment creation
  - runtime checks
  - turn processing
  - human move execution
  - AI move execution
  - viewer sync
  - game-loop orchestration

### Runtime validation

- `src/health_checks.py`
  Defines the runtime-check framework, hook points, default checks, and dispatch registry.

- `src/runtime_guard.py`
  Defines lower-level validation helpers and freeze-on-exception behavior.

- `src/observation/board_observer.py`
  Implements physical scene stability checks for tipped, buried, or displaced pieces.

### Chess logic

- `src/logic/chess_manager.py`
  Owns the `python-chess` board and Stockfish integration.

- `src/logic/operation_planner.py`
  Converts chess moves into physical pick/place operations, including capture, castling, and promotion.

### Manipulation control

- `src/control/execution_controller.py`
  Executes staged arm manipulation and emits lifecycle events for stage-level checks.

### Environment

- `src/env/chess_pick_place_env.py`
  Adapts the chess MuJoCo scene to Fetch-style observation/action semantics.

### Fine-tuning preparation

- `src/training/config.py`
- `src/training/tasks.py`
- `src/env/chess_manipulation_train_env.py`
- `scripts/prepare_finetune_config.py`
- `scripts/generate_xml.py`
- `scripts/verify_mujoco.py`
- `scripts/headless_run.py`

These provide curriculum configuration, task definitions, and a training-facing environment wrapper. They do not run actual training yet.

### Configuration

- `src/settings/runtime.yaml`
- `src/settings/training.yaml`
- `src/config.py`

Runtime and training defaults are YAML-backed. `config.py` loads runtime settings and exposes them as Python constants for the rest of the codebase.

Utility scripts live under `scripts/`. The scene generator writes
`src/assets/chess_world.xml`, the MuJoCo verifier sanity-checks the generated
scene, and the headless runner executes a single scripted manipulation op.

## 3. Runtime Flow

### Startup

1. `load_scene()`
2. create `ChessPickPlaceEnv`
3. run `POST_SCENE_LOAD` checks
4. load RL checkpoint
5. create `ChessGameManager`, `OperationPlanner`, `ExecutionController`
6. infer `square_to_piece`
7. run `PROGRAM_START` checks
8. launch viewer

### Per turn

1. run `TURN_START` checks
2. run human or AI turn
3. for AI movement, execute staged arm motion
4. stage hooks emit `ARM_STAGE_START` and `ARM_STAGE_END`
5. run `TURN_END` checks
6. sync viewer and sleep

### Failure behavior

If any exception escapes the runtime flow:

- it is logged
- the simulator is frozen in-place
- the application does not silently recover by teleporting pieces or masking the fault

## 4. Runtime Check System

Runtime checks are modular classes registered in a `RuntimeCheckRegistry`.

### Current hook points

- `program_start`
- `post_scene_load`
- `turn_start`
- `turn_end`
- `arm_stage_start`
- `arm_stage_end`

### Current default checks

- `SceneAssetsCheck`
  Verifies required MuJoCo assets exist

- `MappingIntegrityCheck`
  Verifies square mapping uniqueness and count consistency

- `BoardAgreementCheck`
  Verifies physical piece positions and logical board state agree

- `ArmHomePoseCheck`
  Verifies the gripper returns to home position

- `RobotWorkspaceCheck`
  Verifies the gripper remains within expected workspace bounds

- `FiniteStateCheck`
  Verifies MuJoCo state arrays contain only finite values

- `PieceObserverCheck`
  Verifies no tipped, buried, or strongly displaced pieces remain

- `StageGoalReachabilityCheck`
  Verifies each arm-stage goal is reachable before execution

- `StageOutcomeCheck`
  Verifies each arm stage reports successful completion

- `StageGripAttachmentCheck`
  Verifies close/lift stages keep the target piece attached to the gripper

### Extending the check system

To add a new check:

1. subclass `RuntimeCheck`
2. assign one or more `CheckHook` values
3. implement `run(context)`
4. register the check in `build_default_check_registry()`

Checks should:

- raise explicit exceptions
- be deterministic
- run cheaply enough at their hook point
- be covered by tests

## 5. Exception Model

Important exception types:

- `SceneLoadError`
- `RLModelError`
- `AIEngineError`
- `ExecutionError`
- `BoardStateError`
- `StabilityError`
- `RuntimeCheckError`
- `ArmStateError`
- `MappingIntegrityError`
- `NumericalStabilityError`
- `SceneIntegrityError`
- `PieceLookupError`

The project is designed so invariants fail loudly and are testable.

## 6. Fine-Tuning Preparation

The codebase is now prepared for the tuning phase, but not yet running training.

### What already exists

- curriculum config model
- task model for move-derived manipulation tasks
- deterministic task sampling helpers
- training-facing environment wrapper for binding a task onto the scene
- detailed fine-tuning plan document in `docs/FINETUNE_FETCH_CHESS_PLAN.md`

### What still needs to be implemented by the training phase

- reset pipeline from arbitrary legal FEN positions
- clutter/capture curriculum resets
- reward shaping logic for training
- evaluation harness for sequential physical manipulation
- checkpoint save/load flow for fine-tuned models

## 7. Tests

The project includes:

- startup/runtime tests
- health-check tests
- training-prep tests
- headless gameplay integration tests
- headless arm execution tests
- operation planner tests
- board observer tests
- chess manager tests

The suite validates:

- startup and shutdown paths
- scene and model boot
- viewer handling
- move planning
- human turn handling
- AI turn handling
- runtime checks
- stage-event dispatch
- headless physical move execution
- mapping integrity
- failure propagation
- fine-tuning preparation APIs

## 8. Running The Project

### Interactive

```bash
python3 main.py
```

### Headless smoke run

```bash
python3 scripts/headless_run.py
```

### Regenerate and validate the scene

```bash
python3 scripts/generate_xml.py
python3 scripts/verify_mujoco.py
```

### Full test suite

```bash
pytest -q
```

## 9. Current Limits

The project is more robust now, but several limits remain:

- the manipulation controller is still largely scripted
- the loaded pretrained Fetch checkpoint is not yet genuinely adapted to the chess scene
- some board areas remain out-of-policy and less reliable
- graveyard capture handling now uses reachable raised trays, but still needs tuned manipulation reliability
- fine-tuning infrastructure is prepared but not yet fully implemented

## 10. Logging Levels

- `DEBUG` is intended for environment-step and controller-stage diagnostics
- `INFO` is the default operational level used by `main.py`
- `WARNING` flags degraded execution that still permits the run to continue
- `ERROR` is reserved for fatal runtime faults

## 11. Recommended Next Engineering Steps

Before or alongside fine-tuning:

1. add a board-reset pipeline for arbitrary legal FEN positions
2. define structured reward terms and evaluation metrics for training
3. add persistent run artifacts for failures: JSON summaries, screenshots, and move traces
4. add a replayable incident format for debugging failed arm operations
5. add configuration files for runtime and training instead of relying only on module constants
6. add CI execution for the fast and medium-weight subsets of the test suite
7. split simulator-heavy tests into explicit test markers such as `unit`, `headless`, and `integration`
