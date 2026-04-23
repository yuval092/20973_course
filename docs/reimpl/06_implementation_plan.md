# 06 — Step-by-Step Implementation Plan

## Rules

1. **Do not proceed to the next stage until the current stage is complete.** Each stage has explicit completion criteria. All listed tests for that stage must pass before moving on.
2. **Never hardcode values** that belong in `runtime.yaml`.
3. **No HuggingFace dependency.** The model is loaded from a local path.
4. **Tests must pass before moving on.** If a test fails, diagnose and fix before proceeding.
5. **Run validation scripts** after stages that touch physics or model loading.
6. **Keep code minimal.** Implement exactly what is specified. No extra abstractions.

---

## Stage 0: Repository Scaffold

### Goal
Create the clean project directory structure with empty packages and stubs.

### Scope
- Directory layout
- `requirements.txt`
- `main.py` stub
- Empty `__init__.py` files

### Deliverables

**Directory tree**:
```
robo_chess/
├── main.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── assets/           (empty; XML and STL files added in Stage 6)
│   ├── control/
│   │   └── __init__.py
│   ├── env/
│   │   └── __init__.py
│   ├── gui/
│   │   └── __init__.py
│   ├── logic/
│   │   └── __init__.py
│   ├── observation/
│   │   └── __init__.py
│   ├── settings/         (empty; YAML added in Stage 1)
│   └── training/
│       └── __init__.py
├── scripts/
├── tests/
│   └── conftest.py       (empty initially)
└── docs/
    └── reimpl/           (these documents)
```

**`requirements.txt`** (minimum):
```
mujoco>=3.0
gymnasium>=0.29
gymnasium-robotics>=1.2
stable-baselines3>=2.0
python-chess>=1.10
numpy>=1.24
pyyaml>=6.0
```

Note: `sb3-contrib` is NOT needed. The model is SAC (from `stable_baselines3`), not TQC (from `sb3_contrib`).

**`main.py`**:
```python
from src.app import main

if __name__ == "__main__":
    main()
```

### Completion Criteria
- `python -c "import src"` exits with code 0.
- `python -c "import src.control"` exits with code 0.
- Directory tree matches the layout above.

---

## Stage 1: Configuration Layer

### Goal
YAML-backed configuration with full schema validation and module-level constant exports.

### Scope
- `src/settings/runtime.yaml`
- `src/config.py` (`ConfigManager` class + module exports)
- `tests/test_config.py`

### Deliverables

**`src/settings/runtime.yaml`**: Create fresh. Key values derived from confirmed environment geometry:
```yaml
local_model_path: "chess_fetch_20260421_095337.zip"
board_center: [0.88, 0.2641, 0.4]
square_size: 0.08
# Robot: base at X=0.2869, Y=0.2641; board near edge X=0.56, far edge X=1.20
spare_staging_origin: [0.60, 0.65, 0.4]   # off the +Y edge of the board, reachable
spare_staging_spacing: 0.09
```
Review `REACHABLE_X_MIN/MAX`, `REACHABLE_Y_MIN/MAX` to match the actual arm reach from the new robot position.

**`src/config.py`**: Implement `ConfigManager`. Export all constants including `LOCAL_MODEL_PATH`. Add a check: if `LOCAL_MODEL_PATH` does not point to an existing file, raise `FileNotFoundError` with a clear message.

**Schema must include** (among others):
```yaml
local_model_path: str    # path to .zip model file
policy_xy_radius: float  # (increased for local model)
stockfish_paths: list    # List of paths to search for the stockfish binary
```

### Tests
All tests in `tests/test_config.py` must pass (see Testing Plan).

### Completion Criteria
- `from src.config import BOARD_CENTER, LOCAL_MODEL_PATH` succeeds.
- Invalid YAML raises `ValueError` with a helpful message listing the offending keys.
- `pytest tests/test_config.py` passes.

---

## Stage 2: Exception Hierarchy

### Goal
All domain-specific exceptions defined.

### Scope
- `src/exceptions.py`
- `tests/test_exceptions.py`

### Deliverables
13 exception classes as specified in `02_low_level_design.md`. All plain subclasses; no additional logic.

### Completion Criteria
- `pytest tests/test_exceptions.py` passes.
- `from src.exceptions import RoboChessError, ExecutionError, ArmStateError` succeeds.

---

## Stage 3: Logging Utilities

### Goal
Structured event logging helper.

### Scope
- `src/logging_utils.py`
- `tests/test_logging_utils.py`

### Deliverables
`EventLogger` with `format_value`, `format_fields`, `log_event` static methods. Module-level `log_event` function wrapper.

### Completion Criteria
- `pytest tests/test_logging_utils.py` passes.

---

## Stage 4: Data Models

### Goal
All data container classes.

### Scope
- `src/models.py` (`GameSystems`, `MoveResult`, `GameState`)
- `src/logic/operation_planner.py` (`PickPlaceOp` only — `OperationPlanner` comes in Stage 5)
- `src/control/execution_controller.py` (`OpResult`, `OpPreflight` only — full class comes in Stage 9)
- `tests/test_models.py`

### Deliverables
Plain data classes with constructors and field access only. No logic.

### Completion Criteria
- `pytest tests/test_models.py` passes.

---

## Stage 5: Chess Logic Layer

### Goal
Complete chess board management and move-to-operation translation.

### Scope
- `src/logic/chess_manager.py` (full `ChessGameManager`)
- `src/logic/operation_planner.py` (full `OperationPlanner`)
- `tests/test_chess_manager.py`
- `tests/test_operation_planner.py`

### Key Notes
- Stockfish is optional. Tests using `get_ai_move` must use `pytest.importorskip` or `try/except AIEngineError` to skip gracefully.
- En passant: the captured pawn is on `(dest_file, src_rank)`, NOT on `(dest_file, dest_rank)`.
- Castling: only two ops (king move, rook move). The rook square indices are computed from the rank of the king's source square (not hardcoded rank 0).

### Completion Criteria
- `pytest tests/test_chess_manager.py tests/test_operation_planner.py` passes.

---

## Stage 6: MuJoCo Scene Assets

### Goal
Working physics scene that loads without error and contains all 32 pieces in the correct positions.

### Scope
- `src/assets/chess_world.xml` (and all referenced assets)
- `src/assets/fetch.xml`, `src/assets/shared.xml`
- All `.stl` mesh files
- `scripts/generate_xml.py` (the scene generator)
- `scripts/verify_mujoco.py`
- `scripts/verify_scene.py`
- `scripts/verify_board_mapping.py`

### Key Notes
- The board geometry in XML must exactly match `BOARD_CENTER` and `SQUARE_SIZE` from `runtime.yaml`.
- Piece body names must follow the convention: `w_pawn_1` through `w_pawn_8`, `w_rook_1`, `w_rook_2`, ..., `w_king`. Same for black (`b_`).
- Spare promotion pieces: `w_spare_queen_1`, `w_spare_queen_2`, `w_spare_rook_1`, ... (2 per type per color). Placed on top of the table in the **spare staging area** (at `SPARE_STAGING_ORIGIN` from config), NOT hidden below the table — the robot arm must physically reach them and the arm cannot reach through the table. Set spare piece `contype=0` and `conaffinity=0` initially (no contact until activated by promotion). Set high freejoint damping (200 N·s/m) so they stay put.
- Graveyard platforms at positions matching `WHITE_GRAVEYARD_ORIGIN` and `BLACK_GRAVEYARD_ORIGIN`.
- Piece freejoints: each piece body must have a `freejoint`.
- Piece sites: each piece body must have a site named `{piece_name}_site` for observation.
- Piece hitboxes: use **box** geoms (not cylinders). A box geom with half-extents matching the piece's footprint cannot roll when the gripper contacts it off-centre, whereas a cylinder can. The SAC policy was trained on a cube object (`FetchPickAndPlace-v4`); box hitboxes provide flat contact surfaces consistent with training dynamics.
- Collision masking: piece box hitboxes enabled (`contype=1`); mesh geoms disabled by default (`contype=0`).
- Board collision: use invisible `board_collision` box above the table surface.
- Piece sites: place each `{piece_name}_site` at the **geometric centre** (CoM) of each piece body, not at the top. The RL policy was trained with a site at the object's CoM; a top-placed site adds lever-arm contributions to velocity features that the policy was not trained on.
- Robot: include `fetch.xml`; it provides the mocap body, joints, and gripper actuators named `robot0:l_gripper_finger_joint` and `robot0:r_gripper_finger_joint`.
- Grip site: the robot must have a site named `robot0:grip`.

### Completion Criteria
- `python scripts/verify_mujoco.py` prints "MuJoCo OK".
- `python scripts/verify_scene.py` prints "Scene OK: 32 pieces mapped."
- `python scripts/verify_board_mapping.py` shows all 32 pieces on correct squares.

---

## Stage 7: Physics Environment

### Goal
Gym-compatible MuJoCo wrapper for the Fetch robot.

### Scope
- `src/env/obs_wrapper.py` (`ObservationReconstructor`)
- `src/env/chess_pick_place_env.py` (`ChessPickPlaceEnv`)
- `tests/test_obs_wrapper.py`
- `tests/test_chess_pick_place_env.py`
- `scripts/verify_home_pose.py`
- `scripts/verify_obs_shape.py`

### Key Notes
- The robot home joint values are `robot0:slide0=0.405`, `robot0:slide1=0.48`, `robot0:slide2=0.0`, with mocap driven to `FETCH_INIT_GRIP`.
- Piece damping is a realistic low value set once in XML.
- Piece mesh collision uses realistic fully-enabled physics meshes/hitboxes always.
- The 26D observation must exactly follow the layout in `03_class_hierarchy.md`.

### Completion Criteria
- `pytest tests/test_obs_wrapper.py tests/test_chess_pick_place_env.py` passes.
- `python scripts/verify_home_pose.py` prints "Home pose OK."
- `python scripts/verify_obs_shape.py` prints "Observation OK."

---

## Stage 8: Observation Layer and Board Observer

### Goal
Physical piece stability checking.

### Scope
- `src/observation/board_observer.py` (`BoardObserver`)
- `tests/test_board_observer.py`
- `scripts/drift_env_test.py`

### Completion Criteria
- `pytest tests/test_board_observer.py` passes.
- `python scripts/drift_env_test.py` prints "Drift test OK."

---

## Stage 9: Execution Controller

### Goal
10-stage pick-and-place controller with RL-primary transit stages and scripted precision descents.

### Scope
- `src/control/execution_controller.py` (full implementation: `OpResult`, `OpPreflight`, `ExecutionController`)
- `tests/test_execution_controller.py` (all unit tests with `DummyEnv`)
- `tests/test_headless_execution.py` (integration tests)
- `scripts/headless_run.py`

### Key Notes

**10-stage structure**: The controller uses 10 stages: `HOME_RESET`, `PREHOVER_SRC`, `DESCEND_SRC`, `CLOSE_GRIPPER_ONLY`, `LIFT_VERIFY`, `PREHOVER_DEST`, `DESCEND_DEST`, `OPEN_GRIPPER_ONLY`, `RELEASE_AND_CLEAR`, `SETTLE_AND_HOME`. See `02_low_level_design.md` for the full dispatch table.

**Coordinate system**: With `board_center=[0.88, 0.2641, 0.4]` and `square_size=0.08`:
```
start_x = 0.88 - 4*0.08 = 0.56
start_y = 0.2641 + 4*0.08 = 0.5841
x = start_x + (7 - file_idx + 0.5) * 0.08   # 'a' is highest X (1.16), 'h' is lowest (0.60)
y = start_y - (rank_idx + 0.5) * 0.08        # rank 1 is highest Y (0.544), rank 8 is lowest (-0.016)
```
After implementing `get_square_pos`, add a unit test asserting `get_square_pos('a1') ≈ [1.16, 0.544, Z_GRASP]`. Do not rely on prose alone.

**Stage waypoint routing** in `_move_gripper_to`: always approach via a 3-waypoint path: rise to `transit_z`, translate at `transit_z`, then descend to target. This prevents sweeping through piece-occupied space.

**Stall detection**: Track `best_dist`. If no improvement of 1mm after 50 consecutive steps, declare stall and return failure.

**RL usage**: When `use_rl=True` and a model is loaded, give the RL model 100% control over the XYZ motion (`rl_model.predict(obs)`). The scripted gripper target overrides the RL's gripper action (dim 3). Three stages are RL-primary: `PREHOVER_SRC` (approach), `LIFT_VERIFY` (lift with grasped piece), and `PREHOVER_DEST` (carry to destination). These match the three-phase FetchPickAndPlace behavior the policy was trained on. Scripted control is reserved for `DESCEND_SRC` and `DESCEND_DEST` (the final ~3cm precision descents) and the fixed-target home/clear/settle stages.

**Dynamic desired_goal**: The RL model relies on the observation vector's `desired_goal` to navigate. Call `env.set_target(piece_name, goal)` and `env.reset_elapsed_steps()` at the start of each RL stage (`PREHOVER_SRC`, `LIFT_VERIFY`, `PREHOVER_DEST`) so the RL policy knows exactly which waypoint to reach and starts with a fresh 50-step budget.

**Time feature**: `FETCH_POLICY_HORIZON = 50`. Reset `_elapsed_steps` at the start of each RL stage. Without this, the time feature reaches 0 after 50 steps and stays there for the remainder of the operation.

**Pick retry from PREHOVER_SRC**: On `CLOSE_GRIPPER_ONLY` or `LIFT_VERIFY` failure, reset `stage_index` to `PREHOVER_SRC` (not HOME_RESET). Re-hover above the piece and retry the descent without travelling all the way home.

**Policy workspace check**: Compute `_policy_compatibility_issues` and log warnings. Do NOT block execution based on these. Only `_preflight_op`'s reachability check (hard bounds) blocks execution.

**DESCEND_DEST outcome logic** (4 cases):
1. Catastrophic (piece below table or above Z_SAFE) → `False`
2. Piece tipped (`up_z < 0.95`) → `False`
3. Piece within XY+Z tolerance → `True`
4. Piece near table but slightly off → attempt lateral correction → check tolerance again → return result

### Completion Criteria
- `pytest tests/test_execution_controller.py` passes (unit tests).
- `pytest tests/test_headless_execution.py` passes (integration tests).
- `python scripts/headless_run.py` prints success with XY error < 20mm.

---

## Stage 10: Health Check Framework

### Goal
Pluggable runtime validation running at named lifecycle hooks, plus the physical board state guard.

> **Why before Bootstrap**: `SystemBootstrapper.bootstrap_game_systems()` calls `build_default_check_registry()` and immediately runs `POST_SCENE_LOAD` and `PROGRAM_START` checks. `BoardAgreementCheck` calls `RuntimeGuard.validate_board_state`. Both must exist before bootstrap is implemented.

### Scope
- `src/health_checks.py` (all 10 checks + `RuntimeCheckRegistry` + `build_default_check_registry()`)
- `src/runtime_guard.py` (`RuntimeGuard`, `freeze_on_exception`)
- `tests/test_health_checks.py`
- `tests/test_runtime_guard.py`

### Key Notes
- `CheckContext.extra` is a plain dict. Arm stage events populate it with `{"goal": ..., "success": ..., "steps_taken": ...}`.
- `StageOutcomeCheck` reads `context.extra["success"]`. Raise only if `success is False` (not if key missing).
- `StageGripAttachmentCheck` only fires when `context.stage in {"CLOSE_GRIPPER_ONLY", "LIFT_VERIFY"}` AND `success is True`.
- `ArmHomePoseCheck` fires at `PROGRAM_START`, `TURN_START`, and `TURN_END` — catching a displaced arm both before a turn begins and after it ends.

### Completion Criteria
- `pytest tests/test_health_checks.py tests/test_runtime_guard.py` passes.
- `build_default_check_registry()` returns a registry with exactly 10 checks.

---

## Stage 11: Bootstrap and Orchestration

### Goal
Full system initialization and move dispatch. Requires Stage 10 (health checks and RuntimeGuard) to be complete.

### Scope
- `src/bootstrap.py` (`SystemBootstrapper`)
- `src/game_runtime.py` (`GameOrchestrator`)
- `tests/test_startup_runtime.py`
- `tests/test_graveyard_operations.py`
- `scripts/verify_rl_model.py`

### Key Notes

**Local model loading** (`load_rl_policy`):
```python
from stable_baselines3 import SAC

rl_model = SAC.load(local_model_path, env=env)
```

The model is SAC (confirmed from zip metadata: `stable_baselines3.sac.policies.MultiInputPolicy`). No `sb3_contrib`, no `huggingface_sb3`, no custom_objects needed.

**No HuggingFace imports anywhere.** Do not add `huggingface_sb3` to `requirements.txt`.

**Policy workspace in `execute_ops`**: Log warnings from `_policy_compatibility_issues` but attempt physical execution. The `_preflight_op` inside `execute_op` will return a failure result if targets are outside hard reachability bounds.

**Promotion swap**: The spare piece must have its geom RGBA and contype restored. White spare: RGBA `[1,1,1,1]`; Black spare: `[0.1,0.1,0.1,1]`. Box hitbox: `contype=1, conaffinity=1`; mesh geom: `contype=0, conaffinity=0`. The mesh geom must remain collision-disabled (`contype=0`) — using `contype=2` would leave the promoted piece participating in contact resolution for the rest of the game, inconsistent with all other pieces which have mesh collision disabled by default.

### Completion Criteria
- `python scripts/verify_rl_model.py` prints "RL model OK."
- `pytest tests/test_startup_runtime.py tests/test_graveyard_operations.py` passes. This includes `test_bootstrap_all_checks_pass` which verifies that all 10 health checks run without error on a freshly bootstrapped scene.

---

## Stage 12: Game Loop and Logic

### Goal
Turn orchestration combining chess logic and physical execution.

### Scope
- `src/game_loop.py` (`GameLoop`)
- `tests/test_game_loop.py`
- `tests/test_headless_gameplay.py`

### Key Notes
- `submit_move` and `execute_ai_turn` should invoke TURN_START and TURN_END health checks via the registry before and after execution. This requires wiring: pass `check_registry` from `systems` and call `run(TURN_START, ...)` before and `run(TURN_END, ...)` after.
- `execute_ai_turn`: if Stockfish is not available (raised `AIEngineError`), return `MoveResult(False, ..., error=e)`.

### Completion Criteria
- `pytest tests/test_game_loop.py tests/test_headless_gameplay.py` passes.

---

## Stage 13: GUI Layer

### Goal
Tkinter chess board panel for human input.

### Scope
- `src/gui/board_panel.py` (`BoardPanel`)
- `tests/test_board_panel.py`

### Key Notes
- Two-click move: first click selects a piece (highlight yellow). Second click attempts the move. Invalid first click (no piece, or wrong color) should reset selection silently.
- Auto-queen promotion: if moving a pawn to rank 8 (white) or rank 1 (black), append "q" to UCI.
- `lock_board` / `unlock_board` should be called by the app before/after arm execution to prevent concurrent input.

### Completion Criteria
- `pytest tests/test_board_panel.py` passes.

---

## Stage 14: Application Entry Point

### Goal
Fully wired interactive application with GUI and physics running in separate threads.

### Scope
- `src/app.py` (`RoboChessApp`)
- `main.py` (already created in Stage 0)

### Key Notes
- `game_worker` is a daemon thread. On main thread exit it terminates automatically.
- All GUI updates from `game_worker` must use `root.after(0, callback)`.
- `stop_event` is the shutdown signal. Check it in the worker loop.
- On `PolicyWorkspaceError` (if it ever surfaces): show status bar message, do not crash.
- On other exceptions: `freeze_on_exception(exc, viewer=viewer, ...)` and break out of the loop.

### Completion Criteria
- `python main.py` opens a MuJoCo viewer and Tkinter window.
- Clicking a pawn and a destination square submits a move.
- The robot arm physically moves the piece to the destination.
- The AI plays Black without crashing.

---

## Stage 15: Full System Validation

### Goal
Confirm end-to-end correctness across all components.

### Scope
- Run full test suite
- Run all validation scripts
- Manual play session

### Steps

**1. Run full test suite**:
```bash
PYTHONPATH=. pytest tests/ -v --tb=short
```
Expected: all tests pass. Target coverage ≥ 80% overall.

**2. Run all validation scripts in sequence**:
```bash
python scripts/verify_mujoco.py
python scripts/verify_scene.py
python scripts/verify_board_mapping.py
python scripts/verify_home_pose.py
python scripts/verify_obs_shape.py
python scripts/verify_rl_model.py
python scripts/drift_env_test.py
python scripts/headless_run.py
```
Expected: all print their success message with no errors.

**3. Manual play session**:
- Run `python main.py`.
- Play 5 moves as White.
- Observe 5 AI responses.
- Verify: arm moves correctly, pieces stay upright, board panel updates, no crashes.

### Completion Criteria
- All pytest tests pass.
- All validation scripts succeed.
- Manual session completes 5 rounds without error.
- `runtime.yaml` has `local_model_path` pointing to the user's model file.

---

## Dependency Graph Between Stages

```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4
                                              ↓
                                         Stage 5 (chess logic)
                                              ↓
Stage 6 (scene assets)
    ↓
Stage 7 (physics env) → Stage 8 (board observer)
    ↓                         ↓
Stage 9 (execution controller) ← depends on 7 and 8
    ↓
Stage 10 (health checks + runtime guard) ← depends on 8, 9
    ↓
Stage 11 (bootstrap + orchestration) ← depends on 10, 9, 5, 1
    ↓
Stage 12 (game loop) ← depends on 11, 10, 5
    ↓
Stage 13 (GUI)
    ↓
Stage 14 (app)
    ↓
Stage 15 (full validation)
```

Stages 2, 3, 4, 5 can be done in parallel after Stage 1. All others are sequential.
