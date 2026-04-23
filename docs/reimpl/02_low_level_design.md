# 02 — Low-Level Design

This document gives a detailed specification for every class in the project. For each class the agent must understand its responsibilities, its public and private API, its fields, its dependencies, and edge cases to handle. The agent should implement the described logic from first principles — do not replicate the exact code structure of the existing codebase.

---

## `src/exceptions.py`

### Exception Hierarchy

```
RoboChessError (base, extends Exception)
├── RuntimeCheckError
│   ├── ArmStateError
│   ├── MappingIntegrityError
│   ├── NumericalStabilityError
│   └── SceneIntegrityError
├── AIEngineError
├── RLModelError
├── SceneLoadError
├── ExecutionError
├── StabilityError
├── BoardStateError
├── PieceLookupError
└── PolicyWorkspaceError
```

All exceptions are simple subclasses with no additional logic. Their purpose is to carry semantic meaning to callers.

| Exception | When raised |
|-----------|-------------|
| `RoboChessError` | Base; never raised directly |
| `RuntimeCheckError` | Base for check failures; never raised directly |
| `ArmStateError` | Gripper is out of workspace or wrong position |
| `MappingIntegrityError` | `square_to_piece` is internally inconsistent |
| `NumericalStabilityError` | NaN or Inf found in simulation arrays |
| `SceneIntegrityError` | Required MuJoCo asset (body/site/actuator) missing |
| `AIEngineError` | Stockfish not found or returned no move |
| `RLModelError` | Local RL model file not found or fails to load |
| `SceneLoadError` | MuJoCo XML not found or fails to parse |
| `ExecutionError` | Robot arm failed to complete a physical op |
| `StabilityError` | Piece tipped, buried, or drifted off square |
| `BoardStateError` | Physical positions don't match logical board |
| `PieceLookupError` | Expected MuJoCo body or joint not found |
| `PolicyWorkspaceError` | Target is outside RL policy trained workspace (informational) |

---

## `src/logging_utils.py`

### `EventLogger`

A stateless helper for emitting structured log messages of the form `event | key=value key=value`.

**`format_value(value) → str`**: If float, format to 4 decimal places. Otherwise `str(value)`.

**`format_fields(**fields) → str`**: Sort keys alphabetically. Skip `None` values. Join as `key=value` pairs separated by spaces.

**`log_event(logger, level, event, **fields)`**: Build message as `event` (if no fields) or `event | <fields>`. Call `logger.log(level, message)`.

Also expose a module-level `log_event(...)` function as a convenience wrapper.

---

## `src/config.py`

### `ConfigManager`

Loads `src/settings/runtime.yaml` at import time and validates it against a hard-coded schema.

**Constructor `__init__(package_dir=None)`**:
- Default `package_dir` is the directory containing `config.py`.
- Derive `assets_dir` and `settings_dir` from it.
- Load `runtime.yaml` via `_load_yaml_settings("runtime.yaml")`.
- Call `_validate_config(settings)` and raise `ValueError` if validation fails.

**`_load_yaml_settings(name) → dict`**: Open `settings_dir / name` and return `yaml.safe_load(...)`.

**`_validate_config(config)`**: Check every required key exists and has the correct Python type. Raise `ValueError` listing all missing keys and all type mismatches. Schema must include `local_model_path: str` (not `hf_repo_id`/`hf_filename`).

**`get(key) → Any`**: Return `runtime_settings.get(key)`.

**Module-level exports**: After creating a singleton `_MANAGER = ConfigManager()`, export every config value as a named constant (e.g. `BOARD_CENTER = _MANAGER.get("board_center")`). Any value with no reasonable default that is missing from YAML should raise at import time. `LOCAL_MODEL_PATH` should be exported; if the path does not exist on disk, raise a clear `FileNotFoundError` at startup (not silently continue).

---

## `src/models.py`

### `GameSystems`

A plain data container. Holds references to every subsystem. Created once at bootstrap; passed by reference everywhere.

**Fields**: `mj_model`, `mj_data`, `env`, `rl_model`, `manager`, `planner`, `controller`, `square_to_piece`, `arm_home_grip`, `check_registry`, `captured_count` (dict, default `{"white": 0, "black": 0}`), `spare_piece_states` (dict mapping each spare body name to one of `"staging"` | `"board"` | `"captured"`, initialized to `"staging"` for all spares at bootstrap).

**Why `spare_piece_states`**: `square_to_piece` only tracks pieces currently on the board. A spare queen that was promoted and then captured is absent from `square_to_piece.values()`, indistinguishable from a spare still at the staging area. Without an explicit three-state tracker, `_find_available_spare` would return the captured spare and the arm would try to pick it up from the staging origin where it no longer exists.

No methods; no logic. Simply store and expose the references.

### `MoveResult`

**Fields**: `success: bool`, `message: str`, `error: Exception | None`.

Represents the outcome of one full turn execution (submit or AI). The `error` field carries the original exception for error-type dispatch in the app layer.

### `GameState`

A snapshot of the logical game state for the GUI to read. Created on demand by `GameLoop.get_state()`.

**Fields**: `board` (chess.Board), `is_game_over`, `is_check`, `is_checkmate`, `is_stalemate`, `is_insufficient_material`, `is_fifty_moves`, `is_threefold`.

All boolean fields are computed from the board at creation time.

---

## `src/logic/operation_planner.py`

### `PickPlaceOp`

**Fields**: `target_square: str`, `dest_square: str`, `piece_name: str`, `is_capture: bool = False`, `promotion: int | None = None`, `dest_pos_override: np.ndarray | None = None`.

The `promotion` field holds a `chess.PieceType` integer (e.g. `chess.QUEEN = 5`). It is `None` for non-promotion moves.

The `dest_pos_override` field carries a pre-resolved XYZ destination. When set, `_preflight_op` uses it directly instead of calling `get_pos(dest_square)`. This avoids mutating `dest_square` to a non-string type when the destination is a computed grid position (e.g., graveyard slot).

### `OperationPlanner`

Translates a validated `chess.Move` into an ordered list of `PickPlaceOp`s. Must inspect the **pre-move** board state.

**`generate_operations(move, board, square_to_piece, all_piece_names, spare_piece_states) → list[PickPlaceOp]`**:

`all_piece_names` is the complete set of all piece body names in the scene (including spares), used for type/color filtering. `square_to_piece` contains only active board pieces. `spare_piece_states` maps each spare body name to `"staging"` | `"board"` | `"captured"` and is used by `_find_available_spare`.

The method inspects `board` before the move is pushed. The logic tree:

1. **Castling** (`board.is_castling(move)`):
   - Op 1: King from `move.from_square` to `move.to_square`.
   - Determine rook squares: kingside rook at file 7, dest at file 5; queenside rook at file 0, dest at file 3 (same rank as king).
   - Op 2: Rook from rook source to rook dest.
   - Return both ops. Do not fall through to capture/normal logic.

2. **Capture** (`board.is_capture(move)`):
   - **En passant** (`board.is_en_passant(move)`): the captured pawn is on the **same rank as the moving pawn's source**, at the destination file. Compute captured square as `chess.square(dest_file, src_rank)`.
   - **Normal capture**: captured piece is on `move.to_square`.
   - Look up the captured piece on the logical board (`board.piece_at(captured_square)`). If `None`, raise `ValueError`.
   - Determine graveyard: `"white_graveyard"` if captured piece is white, `"black_graveyard"` if black.
   - Op 1: Move captured piece to graveyard (`is_capture=True`).

3. **Promotion** (`move.promotion` is not `None`):
   - Determine graveyard: `"white_graveyard"` if moving piece is white, `"black_graveyard"` if black.
   - Op 2 (if capture occurred, this is Op 2, else Op 1): Move pawn from `move.from_square` to graveyard.
   - Op 3: Call `_find_available_spare(board, move, all_piece_names, spare_piece_states)` to identify the first spare body of the promoted type and color with state `"staging"`. Set `piece_name` to that spare body name, `target_square="spare_staging"`. Move spare from staging area to `move.to_square`.
   - Return all ops. Do not fall through to normal move.

4. **Normal move** (always appended, unless castling or promotion already returned):
   - Op: Move piece from `move.from_square` to `move.to_square`.

**`_find_available_spare(board, move, all_piece_names, spare_piece_states) → str`**: Find the first spare body name of the promoted type and color whose state in `spare_piece_states` is `"staging"`. Naming convention: `w_spare_queen_1`, `w_spare_queen_2`, etc. Raise `PieceLookupError` if no spare of that type has state `"staging"`. Do NOT use `square_to_piece.values()` for this check — a captured spare is absent from `square_to_piece` but its state is `"captured"`, not `"staging"`, and the arm cannot retrieve it from the graveyard.

**`_require_piece(square_to_piece, square_name) → str`**: Look up `square_to_piece[square_name]`. Raise `KeyError` with a descriptive message if not found.

---

## `src/logic/chess_manager.py`

### `ChessGameManager`

Owns a `python-chess` board and a Stockfish engine process.

**Constructor**: Initialize `self.board = chess.Board()` (standard starting position). Call `self.engine = self._open_engine()`. Raise `AIEngineError` if no engine found.

**`_open_engine() → chess.engine.SimpleEngine`**: Try each path in `stockfish_paths` with `chess.engine.SimpleEngine.popen_uci(path)`. Catch `FileNotFoundError`, `OSError`, `chess.engine.EngineError`. Return the first success. If all fail, raise `AIEngineError("Stockfish not found in any configured path.")`.

**`validate_move(uci) → bool`**: Parse with `chess.Move.from_uci(uci)` (catch `ValueError` → return `False`). Return `move in self.board.legal_moves`.

**`push_move(uci)`**: Call `self.board.push_uci(uci)`. Raises `ValueError` on illegal or invalid UCI (let it propagate).

**`is_game_over() → bool`**: Return `self.board.is_game_over()`.

**`get_ai_move(limit_time=STOCKFISH_TIME_LIMIT) → chess.Move`**: Call `self.engine.play(self.board, chess.engine.Limit(time=limit_time))`. If `result.move` is `None`, raise `AIEngineError`. On `chess.engine.EngineError`, raise `AIEngineError`.

**`close()`**: Call `self.engine.quit()` if engine is not `None`. Swallow all exceptions. Set `self.engine = None`.

**Context manager**: `__enter__` returns `self`. `__exit__` calls `self.close()`.

---

## `src/env/obs_wrapper.py`

### `ObservationReconstructor`

Stateless. Contains one static method.

**`reconstruct_obs(model, data, target_body_name, target_site_name, n_substeps, time_feature) → np.ndarray[float32, (26,)]`**:

> **Implementation warning**: `grip_velp` must be computed early (it is needed for `obj_velp`), but its **position in the final concatenation is [20:23]**, not at the front. The numbered steps below are the *concatenation order*. Intermediate quantities needed before their slice position are noted explicitly.

Compute first (needed as intermediate values before their slice position):
- `dt = n_substeps * model.opt.timestep`
- `grip_velp_raw` = `mujoco_utils.get_site_xvelp(model, data, GRIP_SITE) * dt` (used in step 6; placed in step 8)
- `robot_qpos`, `robot_qvel` via `mujoco_utils.robot_get_obs(model, data, all_joint_names)` (used in steps 4 and 9)

Concatenation order (matches `FetchPickAndPlace-v1` observation layout):
1. `grip_pos` [0:3]: `data.site(GRIP_SITE).xpos.copy()`
2. `obj_pos` [3:6]: `data.site(target_site_name).xpos.copy()`
3. `obj_rel_pos` [6:9]: `obj_pos - grip_pos`
4. `gripper_state` [9:11]: look up `robot0:l_gripper_finger_joint` and `robot0:r_gripper_finger_joint` by name from `robot_qpos`. Do NOT use `robot_qpos[-2:]` — this is a fragile positional assumption that breaks if joints are reordered in the XML. Explicit name lookups are robust to XML changes.
5. `obj_rot` [11:14]: `rotations.mat2euler(data.site(target_site_name).xmat.reshape(3,3))`
6. `obj_velp` [14:17]: `mujoco_utils.get_site_xvelp(model, data, target_site_name) * dt - grip_velp_raw`
7. `obj_velr` [17:20]: `mujoco_utils.get_site_xvelr(model, data, target_site_name) * dt`
8. `grip_velp` [20:23]: `grip_velp_raw` (computed above; placed here in the concatenation)
9. `gripper_vel` [23:25]: look up `robot0:l_gripper_finger_joint` and `robot0:r_gripper_finger_joint` velocities from `robot_qvel` by name, then multiply by `dt`
10. `time_feature` [25]: `float(np.clip(time_feature, 0.0, 1.0))`

Return `np.concatenate([grip_pos, obj_pos, obj_rel_pos, gripper_state, obj_rot, obj_velp, obj_velr, grip_velp, gripper_vel, [time_feature]]).astype(np.float32)`

Concatenate all into a single array and cast to `np.float32`. The resulting shape must be `(26,)`.

---

## `src/env/chess_pick_place_env.py`

### `ChessPickPlaceEnv`

A `gym.Env` subclass wrapping the MuJoCo chess scene. Acts as the robot's interface: stepping physics, reading state, controlling the gripper.

**Constructor `__init__(mj_model, mj_data, n_substeps=N_SUBSTEPS)`**:
1. Store `self.model`, `self.data`, `self.n_substeps`.
2. Resolve left/right gripper actuator IDs by name.
3. Initialize state tracking fields: `_target_body_name=None`, `_target_site_name=None`, `_goal=None`, `_elapsed_steps=0`, `_policy_horizon=FETCH_POLICY_HORIZON`, and home-pose caches (initially `None`).
4. Build `_piece_mesh_geom_ids`: scan all bodies starting with `w_` or `b_` (including spare pieces), find their mesh geom ID. Dict mapping body name → geom ID.
5. Build `_piece_joint_dofadrs`: scan same bodies (including spare pieces), find freejoint DOF address. Dict mapping body name → int. Spare pieces must be included so that `get_piece_linear_velocity` works correctly after a pawn promotion.
6. Build `_piece_mesh_contact_masks`: for each piece, snapshot current `contype`/`conaffinity` from the model.
7. (Removed: Pieces retain standard physically simulated mesh collisions and damping at all times).
8. Run 200 `mujoco.mj_step(model, data, n_substeps)` calls to settle the scene.
9. Compute `_mocap_min` and `_mocap_max` from `BOARD_CENTER`, `SQUARE_SIZE`, workspace margin.
10. Define `observation_space` (Dict with `observation` Box(26,), `achieved_goal` Box(3,), `desired_goal` Box(3,)) and `action_space` (Box(4,)).
11. Call `reset_robot_pose()`.

**`reset_robot_pose()`**: If `_robot_home_joint_state` is not `None`, restore saved joint positions/velocities and mocap pose, then call `mj_forward`. Otherwise, set the three Fetch slider joints to canonical values, reset mocap welds, call `mj_forward`, then drive mocap to `FETCH_INIT_GRIP` for 10 steps. Snapshot the resulting robot joint state, mocap pose, and grip position.

Important: Capture piece joint states before resetting robot, restore them after, so pieces don't jump.

**`set_target(piece_name, goal_pos)`**: Set `_target_body_name`, compute `_target_site_name = f"{piece_name}_site"`, copy `goal_pos` to `_goal`, and reset `_elapsed_steps`. This updates the observation vector so the RL policy knows the current goal.

**`reset_elapsed_steps()`**: Reset `_elapsed_steps` to 0. Call this at the start of each RL-active stage (PREHOVER_SRC, PREHOVER_DEST) so the policy sees a fresh 50-step horizon (matching the `FETCH_POLICY_HORIZON = 50` training episode length) rather than a time feature stuck near 0 for hundreds of steps. The SAC model was fine-tuned from `sac-FetchPickAndPlace-v4` which uses a 50-step episode horizon; resetting per RL stage matches that distribution.

(Removed `set_piece_mesh_collision_enabled` and `set_piece_active_damping`: Physics modifications are strictly forbidden to ensure a pure simulation.)

**`step(action, viewer=None, debug=False, gripper_target=None) → (obs, reward, False, info)`**:
1. Copy and convert `action` to float64.
2. Expand to Fetch action format via `_expand_fetch_action`: `pos_ctrl = action[:3] * ACTION_SCALE`, `rot_ctrl = [1,0,1,0]`, `gripper_ctrl = [action[3], action[3]]`.
3. Apply via `mujoco_utils.ctrl_set_action` and `mujoco_utils.mocap_set_action`.
4. Clip `mocap_pos[0]` to workspace bounds.
5. If `gripper_target` is not `None`, call `set_gripper_target(gripper_target)`.
6. Step physics `n_substeps` times.
7. Increment `_elapsed_steps`.
8. Sync viewer if provided (with optional delay).
9. Build and return obs dict, sparse reward, `terminated=False`, info dict.

Note: `terminated` is always `False` in this implementation; the controller manages its own termination logic.

**`compute_reward(achieved_goal, desired_goal) → float`**: Return `0.0` if `np.linalg.norm(achieved - desired) < GOAL_TOLERANCE` else `-1.0`.

**`get_obs() → dict`**: Assert `_target_body_name` and `_goal` are set. Call `ObservationReconstructor.reconstruct_obs(...)`. Return dict with `observation`, `achieved_goal` (piece site xpos), `desired_goal` (copy of `_goal`).

**`force_gripper_open()`** / **`force_gripper_closed()`**: Call `set_gripper_target(GRIPPER_OPEN)` / `set_gripper_target(GRIPPER_CLOSED)`.

**`set_gripper_target(opening)`**: Clip `opening` to `[0.0, 0.05]`. Set both finger actuator control values.

**`get_grip_pos() → np.ndarray`**: Return `data.site(GRIP_SITE).xpos.copy()`.

**`get_piece_pos() → np.ndarray`**: Assert target set. Return `data.body(_target_body_name).xpos.copy()`.

**`get_piece_linear_velocity() → np.ndarray`**: Return `data.qvel[qvel_addr:qvel_addr+3].copy()` where `qvel_addr` comes from the target piece's freejoint.

**`get_gripper_finger_qpos() → np.ndarray`**: Return joint positions of `robot0:l_gripper_finger_joint` and `robot0:r_gripper_finger_joint`.

**Properties**: `target_body_name`, `goal`, `home_grip_pos` (assert not None), `remaining_time_feature` (`1 - elapsed/horizon`, clipped), `policy_horizon`.

---

## `src/observation/board_observer.py`

### `BoardObserver`

Scans all MuJoCo bodies for physical instability.

**Constructor**: Store `self.model`, `self.data`.

**`verify_stability(active_piece_names=None)`**: Iterate all bodies by index. For each:
1. Get name via `mujoco.mj_id2name(...)`. Skip ignored bodies (empty name, or contains any of: `world`, `table`, `spare`, `graveyard`, `base`, `fetch`, `robot`, `mocap`).
2. If `active_piece_names` is provided and name not in it, skip.

Important: all callers should pass `active_piece_names=set(square_to_piece.values())` to exclude pieces that have left the board (captured pieces now in the graveyard). Graveyard pieces may be within 35cm of the board center, and since they are no longer on a chess square their displacement from the nearest square center will trigger a false `StabilityError`.
3. Compute `up_z = 1 - 2*(quat[1]² + quat[2]²)`. If `up_z < TILT_THRESHOLD_COS`, raise `StabilityError(f"Piece '{name}' has tipped over (up_z={up_z:.3f})")`.
4. If `pos[2] < TABLE_HEIGHT - 0.01`, raise `StabilityError(f"Piece '{name}' is below the table level (z={pos[2]:.3f})")`.
5. For pieces near board center (`|dx| < 0.35 and |dy| < 0.35`) and near table level (`pos[2] < TABLE_HEIGHT + 0.05`): compute distance from nearest square center in meters. If `> SQUARE_SIZE * DISPLACEMENT_THRESHOLD`, raise `StabilityError`.

**`_up_z(quat) → float`** (static): `1 - 2*(quat[1]² + quat[2]²)`.

**`_is_ignored_body(name) → bool`** (static): Check if name is empty or contains any ignored token.

---

## `src/control/execution_controller.py`

### `OpResult`

**Fields**: `success: bool`, `steps_taken: int`, `final_error_mm: float`, `phases_completed: int`, `details: str`.

### `OpPreflight`

**Fields**: `actual_piece_pos`, `src_pos`, `dest_pos`, `stage_targets`, `policy_issues`, `failure_result`.

### `ExecutionController`

The heart of the robotic manipulation system. Implements the 10-stage pick-and-place state machine.

**Class constants**:
- `STAGES`: list of 10 stage name strings: `HOME_RESET`, `PREHOVER_SRC`, `DESCEND_SRC`, `CLOSE_GRIPPER_ONLY`, `LIFT_VERIFY`, `PREHOVER_DEST`, `DESCEND_DEST`, `OPEN_GRIPPER_ONLY`, `RELEASE_AND_CLEAR`, `SETTLE_AND_HOME`
- `_reach_min`, `_reach_max`: numpy arrays from config
- `_source_descend_offset_z = 0.03`
- `_progress_log_interval = 10`

**Constructor `__init__(rl_model, env)`**:
- Store `self.rl_model`, `self.env`.
- Set `_policy_center` from `env.home_grip_pos` if available, else `FETCH_INIT_GRIP`.
- Set `_pregrasp_piece_rest_z = Z_GRASP`.
- Set `_placement_goal = None`, `_stage_event_handler = None`.
- Note: graveyard position tracking belongs entirely to `GameOrchestrator` (using `GameSystems.captured_count`). The controller does not maintain graveyard counters.

**`set_stage_event_handler(handler)`** / **`_emit_stage_event(phase, **payload)`**: Register/invoke a callback at stage lifecycle events.

---

### Position Computation

**`get_square_pos(square_name) → np.ndarray`** (static): Convert chess square string to world XYZ.
- `file_idx = ord(square_name[0]) - ord('a')` (0–7)
- `rank_idx = int(square_name[1:]) - 1` (0–7)
- `start_x = BOARD_CENTER[0] - 4*SQUARE_SIZE`
- `start_y = BOARD_CENTER[1] + 4*SQUARE_SIZE` (high-Y corner, rank 1 side)
- File increases in the negative X direction: `x = start_x + (7 - file_idx + 0.5) * SQUARE_SIZE`
- Rank increases in the **negative** Y direction: `y = start_y - (rank_idx + 0.5) * SQUARE_SIZE`
- Z = `Z_GRASP`

Implementation note: after implementing `get_square_pos`, add a unit test that asserts `get_square_pos('a1')` equals the expected XYZ derived from the actual XML. Do not rely on the prose description alone — the board corner is the most important invariant to lock down.

**`get_pos(square_name) → np.ndarray`**: Convert square name to XYZ. If square name is not a valid chess square, raise `PieceLookupError`. Else → `get_square_pos(square_name)`.

---

### Operation Execution

**`execute_op(op, viewer=None) → OpResult`**: Top-level entry point.

1. Call `_preflight_op(op)`. Extract `actual_piece_pos`, `src_pos`, `dest_pos`, `stage_targets`. Set `_placement_goal = dest_pos.copy()`.
2. Log policy warnings. If `preflight.failure_result` is not `None`, return it.
3. Setup: Extract actual piece position and build stage targets. (Note: `env.set_target` will be called dynamically by the stages that delegate to the RL policy, ensuring the policy observation matches the immediate waypoint).
4. Loop through `STAGES` using an index (allows non-sequential advancement):
   - Resolve the stage goal via `_resolve_stage_goal`.
   - Emit `ARM_STAGE_START`.
   - Call `_execute_stage(stage, op.piece_name, goal, viewer)` → `(success, steps_taken)`.
   - Emit `ARM_STAGE_END`.
   - If success: increment `stages_completed`, advance `stage_index`.
   - If failure in `CLOSE_GRIPPER_ONLY` or `LIFT_VERIFY` and `pick_retry_count < 2`: force gripper open, settle, re-read piece position, rebuild stage targets, reset `stage_index` to `PREHOVER_SRC` (not HOME_RESET — re-hover above the source piece and retry the descent without travelling all the way home), increment retry count.
   - If failure in any other stage: record `failed_stage`, break.
5. If `failed_stage` is not `None` and not `RETURN_HOME`: call `_retract_arm(viewer)`.
6. Evaluate final placement: XY error, Z error, `_placement_stability_issue`.
7. Build and return `OpResult`.

**`_preflight_op(op) → OpPreflight`**:
1. Read actual piece position from `env.data.body(op.piece_name).xpos`.
2. `src_pos`: use actual position, unless target_square contains "graveyard" or is "spare_staging" (use `get_pos` for graveyard/staging source).
3. `dest_pos`: if `op.dest_pos_override` is not None, use it directly. Otherwise call `get_pos(op.dest_square)`. This handles graveyard destinations where `dest_square` remains a string but the resolved XYZ is pre-computed by the orchestrator.
4. Set `_pregrasp_piece_rest_z`.
5. Build `stage_targets` via `_build_stage_targets(src_pos, dest_pos, actual_piece_pos)`.
6. Compute `policy_issues` via `_policy_compatibility_issues(src_pos, dest_pos)`.
7. Check reachability of all non-None stage targets. If any unreachable: create `failure_result = OpResult(success=False, ...)`.
8. Return `OpPreflight(...)`.

**`_resolve_stage_goal(stage, stage_targets, src_pos, dest_pos) → np.ndarray | None`**:
- `PREHOVER_DEST`: target XY = `dest_pos[:2] - _current_carry_xy_offset()`, Z from stage_targets. Note: this offset is computed once at stage entry and used as a fixed waypoint; the carry position may shift during transit (see note on carry offset staleness below).
- `DESCEND_SRC`: use current live piece position + `_source_descend_offset_z` in Z.
- `DESCEND_DEST`: `[dest_pos[0], dest_pos[1], dest_pos[2] - CLOSE_DESCEND_OFFSET]`.
- All others: look up from `stage_targets`.

**`_current_carry_xy_offset() → np.ndarray`**: If piece Z > `Z_GRASP + 0.004` (it's in the air), return `piece_pos[:2] - grip_pos[:2]`. Else return zero vector. Used to compensate for carry drift when navigating to the destination hover.

**`_build_stage_targets(src_pos, dest_pos, live_piece_pos=None) → dict`**:
Compute targets for all 10 stages. Key targets:
- `PREHOVER_SRC`: `[src_x, src_y, Z_SAFE]`
- `DESCEND_SRC`: `[src_x, src_y, piece_z + _source_descend_offset_z]`
- `LIFT_VERIFY`: `[src_x, src_y, Z_SAFE]`
- `PREHOVER_DEST`: `[dest_x, dest_y, Z_SAFE]`
- `DESCEND_DEST`: `[dest_x, dest_y, dest_z - CLOSE_DESCEND_OFFSET]`
- `RELEASE_AND_CLEAR`: `[dest_x, dest_y, clearance_z]` where `clearance_z = _release_clearance_z(dest_pos, home_goal)`
- `SETTLE_AND_HOME`: `home_goal`
- Stages with no positional target (`HOME_RESET`, `CLOSE_GRIPPER_ONLY`, `OPEN_GRIPPER_ONLY`): `None`

**`_placement_stability_issue(piece_name, dest_pos) → str | None`**:
Check: `up_z < 0.966` → "tipped"; `piece_pos[2] > dest_pos[2] + PLACEMENT_TOLERANCE` → "elevated"; `piece_pos[2] < dest_pos[2] - PLACEMENT_TOLERANCE` → "sunk". Return `None` if all pass.

**`_is_reachable(goal) → bool`** (classmethod): Check `goal >= _reach_min` and `goal <= _reach_max`.

**`_policy_compatibility_issues(src_pos, dest_pos) → list[str]`**: Check src, dest, approach-over-src, approach-over-dest against `POLICY_XY_RADIUS` and hard reachability bounds. Return list of issue strings. These are warnings only; they do not block execution.

---

### Stage Execution

**`_execute_stage(stage, piece_name, goal, viewer) → (bool, int)`**: Dispatcher. Each stage calls a specialized method or inline logic. Key behaviors:

| Stage | Action |
|-------|--------|
| `HOME_RESET` | `force_gripper_open()`; if arm not near home, `_move_gripper_to(home_goal)`; settle |
| `PREHOVER_SRC` | `env.set_target(piece_name, goal)`; `env.reset_elapsed_steps()`; `_move_gripper_to(goal, use_rl=True, gripper_opening=GRIPPER_OPEN)` |
| `DESCEND_SRC` | `_set_gripper_aperture(PREGRASP_GRIPPER_OPENING)` at entry (absorbs PREGRASP_NARROW); then `_move_gripper_to(goal, use_rl=False, max_cartesian_action=0.25, max_piece_drift=PIECE_DRIFT_TOLERANCE*4)` |
| `CLOSE_GRIPPER_ONLY` | `_actuate_gripper(close=True, stabilize_piece=False)` |
| `LIFT_VERIFY` | `env.set_target(piece_name, goal)`; `env.reset_elapsed_steps()`; `_move_gripper_to(goal, use_rl=True, require_piece_follow=True, gripper_opening=GRIPPER_CLOSED)` |
| `PREHOVER_DEST` | `env.set_target(piece_name, goal)`; `env.reset_elapsed_steps()`; `_move_gripper_to(goal, use_rl=True, require_piece_follow=True, gripper_opening=GRIPPER_CLOSED)` |
| `DESCEND_DEST` | `_handle_descend_dest(piece_name, goal, viewer)` |
| `OPEN_GRIPPER_ONLY` | `_actuate_gripper(close=False)` |
| `RELEASE_AND_CLEAR` | Begin raising arm immediately after gripper opens (using `tolerate_released_piece_jitter=True`); check settle conditions after clearing. Combines former POST_RELEASE_SETTLE + POST_RELEASE_CLEARANCE: the piece settles whether or not the arm is above it, and measuring settlement after arm clears avoids gripper-proximity artifacts. |
| `SETTLE_AND_HOME` | Move arm to home while interleaving physics settle steps for final placement verification. Combines former RETURN_HOME + FINAL_PLACEMENT_SETTLE: by the time the arm reaches home the piece will have settled. RETURN_HOME failure is non-fatal; proceed to final check regardless. |

Note on RL usage: the RL model's gripper output (action dim 3) is effectively zeroed by explicit `gripper_target` overrides in all stages. This is intentional — the scripted controller manages gripper state explicitly. The RL has 100% XYZ control during PREHOVER_SRC, LIFT_VERIFY, and PREHOVER_DEST — these are the three transit stages the SAC policy was trained to execute (approach, lift, carry). Scripted motion is used only for DESCEND_SRC and DESCEND_DEST (the final ~3cm vertical descents) where millimeter-precision rigid body contact is required, and for HOME_RESET, RELEASE_AND_CLEAR, and SETTLE_AND_HOME where RL provides no benefit over a simple proportional controller.

**`_handle_descend_dest(piece_name, goal, viewer) → (bool, int)`**: Descend with `use_rl=False`. After motion:
1. If piece Z < `TABLE_HEIGHT - 0.01` or > `Z_SAFE`: return `(False, steps)`.
2. If `up_z < 0.95`: return `(False, steps)`.
3. If XY and Z errors within `PLACEMENT_TOLERANCE`: return `(True, steps)`.
4. If piece near table level (`piece_z <= dest_z + 0.03`): attempt lateral correction by shifting gripper XY by `dest_xy - piece_xy`. Re-check tolerance. Return result.
5. Else: return `(False, steps)`.

---

### Motion Primitives

**`_move_gripper_to(target_pos, viewer, gripper_opening, require_piece_follow, max_piece_drift, transit_z, released_piece_goal, tolerate_released_piece_jitter, max_cartesian_action, rl_vertical_only, use_rl) → (bool, int)`**:

Drives gripper through 3 waypoints: `[current_x, current_y, transit_z]` → `[target_x, target_y, transit_z]` → `target_pos`.

For each waypoint, loop up to `RETRACT_STEPS * 3` times:
- Compute `delta = waypoint - grip_pos`. If `|delta| < 0.006`: break (waypoint reached).
- Track `best_dist`; if no improvement after 50 steps: return `(False, steps)` (stall detection).
- Compose action via `_compose_guided_action(delta, max_cartesian_action, rl_vertical_only, use_rl)`.
- Call `env.step(action, viewer=viewer, gripper_target=gripper_opening)`.
- If `max_piece_drift` set: check XY piece drift from initial; return `(False, steps)` if exceeded.
- If `released_piece_goal` set and not tolerating jitter: check released piece via `_released_piece_issue`; return `(False, steps)` if disturbed.

After all waypoints: validate via `_validate_motion_stage(...)`. Return `(success, steps)`.

**`_compose_guided_action(delta, max_cartesian_action, use_rl) → np.ndarray[4]`**:
1. If `use_rl=True` and `rl_model` is present: We fully trust the RL policy to execute the transit. Call `action, _ = rl_model.predict(env.get_obs(), deterministic=True)`. The controller enforces safety by zeroing out the gripper dimension (`action[3] = 0`) to maintain explicit grasp control, but the XYZ motion is 100% RL-driven.
2. Fallback (if RL not available or `use_rl=False`): Build a purely scripted proportional controller action: `np.clip(delta / ACTION_SCALE, -max_cartesian_action, max_cartesian_action)`, setting gripper dim 0.

Rationale: The RL model is trusted to solve the inverse kinematics and continuous control for transits. By explicitly setting `env.set_target(goal)` at the start of the RL stage, the policy's objective aligns with the waypoint, allowing 100% RL control without drifting to the final destination.

**`_actuate_gripper(close, viewer, stabilize_piece) → (bool, int)`**:
- If closing: call `_close_gripper_with_descent(viewer, stabilize_piece)`.
- If opening: call `_set_gripper_aperture(GRIPPER_OPEN, viewer)`. Check `_gripper_is_open()`.

**`_close_gripper_with_descent(viewer, stabilize_piece) → (bool, int)`**:
1. Proximity check: if `|grip_xy - piece_xy| < threshold` and `grip_z` is already near `piece_z + 0.05`, skip the re-alignment step and proceed directly to the closing loop. Only perform the re-lift if significant XY drift is detected.
2. Otherwise: align gripper above piece XY at `max(grip_z, piece_z + 0.05)`.
3. Loop `CLOSE_DESCEND_STEPS` times: if piece-to-grip distance > `GRIP_CONTACT_TOLERANCE * 0.95`, apply small Z-down action while commanding `GRIPPER_CLOSED`. Break when XY close enough and piece-to-grip < threshold.
4. Settle. Check `xy_dist < GRIP_CONTACT_TOLERANCE` and `piece_to_grip < desired_gap`. Do NOT add a `fingers_closed` condition based on `max |finger_qpos| < 0.002` — that threshold checks for an empty grasp (fingers nearly fully shut). A successful grasp on a piece stops the fingers partway (at the piece radius, typically 0.010–0.020). The `piece_to_grip` check already captures the relevant physical state.

**`_set_gripper_aperture(target_opening, viewer) → (bool, int)`**: Set gripper target. Settle for `GRIPPER_ACTUATION_STEPS` (more if fully opening: `max(GRIPPER_ACTUATION_STEPS, RELEASE_SETTLE_STEPS * 3)`). Return `(True, steps)`.

**`_gripper_is_open() → bool`**: `min(finger_qpos) >= GRIPPER_OPEN - RELEASE_GRIPPER_OPEN_TOLERANCE`.

**`_settle_released_piece(goal_pos, viewer, max_steps) → (bool, int)`**: Step physics (via `mujoco.mj_step`) for up to `max_steps` steps. After each step, check XY error, Z error, piece speed. Return `(True, steps)` when all within tolerance. Return `(xy_ok and z_ok, max_steps)` if timeout.

**`_finalize_placement(goal_pos, viewer, max_steps) → (bool, int)`**: Similar to `_settle_released_piece` but also checks `_placement_stability_issue`. Must satisfy XY error, Z error, speed, and upright orientation.

**`_settle(viewer, steps)`**: Run `mujoco.mj_step` `steps` times; sync viewer each step.

**`_retract_arm(viewer) → (bool, int)`**: Force gripper open. Compute `retreat_z = _release_clearance_z(...)`. Move up to `retreat_z` then to `home_goal`. Settle.

**`_return_home_after_release(home_goal, viewer, released_piece_goal) → (bool, int)`**: Lift clear of placed piece, then move to home, tolerating jitter in released piece position.

**Implementation note — carry offset staleness**: `_current_carry_xy_offset()` is computed once at PREHOVER_DEST entry and used as a fixed waypoint. During long horizontal transits, the piece may swing relative to the gripper (especially with low active damping). The existing lateral correction in DESCEND_DEST partially compensates for this. Document this as a known small-error source. For future improvement, recompute the offset at intervals during transit.

**Implementation note — hidden piece physics**: Pawn pieces moved to the graveyard during promotion are positioned at `PIECE_HIDE_Z` (below the table). Because the robot cannot reach below the table, hidden pieces must have `contype=0` and `conaffinity=0` to disable contact, preventing physics instability. Also set high freejoint damping (200 N·s/m) to prevent indefinite gravity-driven acceleration.

**Spare piece staging**: Spare promotion pieces must NOT be hidden below the table. The robot arm physically picks them up during promotion — it cannot reach through the table. Place all spare pieces in a **spare staging area** on TOP of the table surface, off to the side of the board (at `SPARE_STAGING_ORIGIN`, defined in `runtime.yaml`). The staging area should be within the arm's reachable workspace, outside the board footprint. Spare pieces are laid out in a grid at `SPARE_STAGING_ORIGIN` with `SPARE_STAGING_SPACING` between them. The `get_square_pos` equivalent for spare staging is `get_spare_pos(piece_name)` which computes the spare's position in the staging grid.

**`_release_clearance_z(dest_pos, home_goal) → float`**: `min(REACHABLE_Z_MAX - 0.01, max(Z_SAFE + RELEASE_CLEARANCE_MARGIN, dest_z + RELEASE_CLEARANCE_MARGIN, home_z + 0.02))`.

**`_validate_motion_stage(target_pos, require_piece_follow, initial_piece_pos, max_piece_drift) → bool`**:
- `grip_ok`: `|grip_pos - target_pos| < 0.02`
- `drift_ok`: if drift checking, `|piece_xy - initial_xy| <= max_piece_drift`
- If `require_piece_follow`: also check `piece_xy` close to `grip_xy`, `piece_z` within 10cm of `grip_z`, and piece either lifted or at goal.

---

## `src/runtime_guard.py`

### `RuntimeGuard`

**`expected_board_squares(board) → dict[str, chess.Piece]`** (static): Map all squares from `board.piece_map()` to piece objects.

**`validate_board_state(mj_model, mj_data, board, square_to_piece, controller, position_tolerance)`** (classmethod):
1. Compare expected squares vs mapped squares. Raise `BoardStateError` if sets differ.
2. For each square: call `validate_piece_identity`. Read piece position and quaternion. Compare XY and Z against `get_pos(square_name)`. If any error exceeds `position_tolerance`: raise `BoardStateError`.
3. Check `up_z` against `TILT_THRESHOLD_COS`. Raise `StabilityError` if tilted.
4. Call `BoardObserver(mj_model, mj_data).verify_stability(active_piece_names=...)`.

**`validate_piece_identity(square_name, piece_name, piece)`** (classmethod): Check color prefix (`w_`/`b_`). Check that piece type token (`pawn`, `knight`, etc.) appears in body name. Raise `BoardStateError` on mismatch.

**`freeze_on_exception(exc, viewer, mj_model, mj_data)`** (classmethod): Log fatal error with traceback. Call `run_freeze_loop(...)`.

**`run_freeze_loop(viewer, mj_model, mj_data, sleep_sec, max_cycles)`** (static): Loop calling `mj_forward` and `viewer.sync()` at `sleep_sec` intervals. Optional `max_cycles` limit for testability.

**Module-level functions**: Expose `expected_board_squares`, `validate_board_state`, `detect_physical_instability`, `run_freeze_loop`, `freeze_on_exception` as top-level wrappers (backward compatibility).

---

## `src/health_checks.py`

### `CheckHook` (str Enum)

Values: `PROGRAM_START`, `POST_SCENE_LOAD`, `TURN_START`, `TURN_END`, `ARM_STAGE_START`, `ARM_STAGE_END`.

### `CheckContext`

**Fields**: `hook`, `systems`, `viewer`, `move_uci`, `stage`, `op`, `extra` (dict).

### `RuntimeCheck` (abstract base)

**Fields**: `name: str` (class var), `hooks: tuple` (class var).

**`run(context)`**: Abstract. Must raise a domain exception on failure, or return normally on success.

### `RuntimeCheckRegistry`

**Constructor**: Store checks as tuple.

**`run(hook, context)`**: For each check where `hook in check.hooks`, call `check.run(context)`. Log debug events before and after.

### 10 Built-in Checks

| Check | Hooks | Logic |
|-------|-------|-------|
| `SceneAssetsCheck` | POST_SCENE_LOAD | Verify `w_king`, `b_king` bodies; `robot0:grip` site; two finger actuators exist |
| `MappingIntegrityCheck` | PROGRAM_START, TURN_START, TURN_END | No duplicate piece assignments; count matches `board.piece_map()` |
| `BoardAgreementCheck` | PROGRAM_START, TURN_START, TURN_END | Call `validate_board_state(...)` |
| `ArmHomePoseCheck` | PROGRAM_START, TURN_START, TURN_END | Gripper distance from `arm_home_grip` < tolerance |
| `RobotWorkspaceCheck` | PROGRAM_START, TURN_START, TURN_END | Gripper within reachability bounds |
| `FiniteStateCheck` | POST_SCENE_LOAD, PROGRAM_START, TURN_START, TURN_END, ARM_STAGE_START, ARM_STAGE_END | No NaN/Inf in `qpos`, `qvel`, `ctrl` |
| `PieceObserverCheck` | PROGRAM_START, TURN_START, TURN_END | `BoardObserver.verify_stability(active_piece_names=...)` |
| `StageGoalReachabilityCheck` | ARM_STAGE_START | If `context.extra["goal"]` is set, verify it's reachable |
| `StageOutcomeCheck` | ARM_STAGE_END | If `context.extra["success"] is False`, raise `ExecutionError` |
| `StageGripAttachmentCheck` | ARM_STAGE_END | After CLOSE_GRIPPER_ONLY or LIFT_VERIFY, check piece-to-grip distance ≤ `PIECE_FOLLOW_TOLERANCE * 1.5` |

**`build_default_check_registry() → RuntimeCheckRegistry`**: Instantiate all 10 checks and return registry.

---

## `src/bootstrap.py`

### `SystemBootstrapper`

**`load_scene(scene_xml) → (MjModel, MjData)`**: Check file exists (raise `SceneLoadError` if not). Load with `mujoco.MjModel.from_xml_path(...)`. Create `MjData`. Call `mj_forward`. Return both. Wrap errors in `SceneLoadError`.

**`load_rl_policy(env, local_model_path) → SAC`**: Load using `SAC.load(local_model_path, env=env)` from `stable_baselines3`. Raise `RLModelError` on failure. The model is a fine-tuned SAC (Soft Actor-Critic) policy, not TQC — confirmed from the model zip metadata. No `sb3_contrib` dependency needed.

**`_is_piece_body(body_name) → bool`** (static): Return `True` if name starts with `w_` or `b_` AND does not contain `"spare"` as a substring AND does not contain any of `"world"`, `"table"`, `"graveyard"`, `"robot"`, `"square"`. Use substring containment (`"spare" in name`), not prefix matching — spare pieces are named `w_spare_queen_1` which starts with `w_`, so a prefix check for `"spare"` would miss them and incorrectly include them in the square mapping.

**`initialize_square_to_piece(mj_model, mj_data) → dict`** (classmethod): Scan all bodies. For each piece body, read its XYZ position. Compute file and rank indices by comparing position to `a1` and `b1` positions from `get_square_pos`. Round to nearest integer. If `0 <= file < 8` and `0 <= rank < 8`, map `square_name → body_name`.

**`validate_initial_mapping(square_to_piece)`** (static): Assert `len == 32`. Raise `BoardStateError` otherwise.

**`bootstrap_game_systems(scene_xml, local_model_path) → GameSystems`** (classmethod):
1. Load scene → `(mj_model, mj_data)`.
2. Create `ChessPickPlaceEnv(mj_model, mj_data, n_substeps=N_SUBSTEPS)`.
3. Snapshot `arm_home_grip = env.get_grip_pos().copy()`.
4. Create preliminary `GameSystems` with `rl_model=None`.
5. Run `POST_SCENE_LOAD` checks.
6. Load RL policy from `local_model_path`.
7. Build `square_to_piece` mapping; validate 32 pieces.
8. Create final `GameSystems` with all components.
9. Wire stage event handler: `controller.set_stage_event_handler(lambda phase, **kw: check_registry.run(phase, CheckContext(...)))`.
10. Run `PROGRAM_START` checks.
11. Return `GameSystems`.

---

## `src/game_runtime.py`

### `GameOrchestrator`

**`sync_viewer(viewer, mj_model, mj_data)`** (static): If viewer provided, call `mj_forward` then `viewer.sync()`.

**`execute_ops(mj_model, mj_data, ops, controller, square_to_piece, systems, viewer, physical_arm)`** (classmethod): For each op:
1. If physical_arm is True:
   - If `op.dest_square` is a graveyard name: compute grid XYZ using `systems.captured_count` and `_get_graveyard_grid_pos`. Set `op.dest_pos_override` to this value. Do NOT mutate `op.dest_square` — it stays as `"white_graveyard"` / `"black_graveyard"` (a string) so that `_preflight_op` can use `dest_pos_override` rather than calling `get_pos(dest_square)`, which would fail on a non-square-name string.
   - Call `controller.execute_op(op, viewer)`. The arm physically performs the move.
   - If `OpResult.success=False`, raise `ExecutionError`.
   - If destination was graveyard, increment `captured_count`.
2. (No teleport mode. Execution is strictly physical).
3. Update `square_to_piece` via `_update_square_mapping`.
4. Call `mj_forward` and `sync_viewer`.

**`_update_square_mapping(systems, target_square, dest_square, piece_name)`** (static):
- If `target_square == "spare_staging"`: the piece did not come from a board square. Do not pop anything. Simply add `dest_square → piece_name` to the map.
- If destination is graveyard: pop `target_square` from map (piece leaves the board).
- Else (normal move): move entry from `target_square` to `dest_square`.

**`_get_graveyard_grid_pos(origin, count) → list[float]`** (static): Compute grid position from `GRAVEYARD_COLS` and `GRAVEYARD_SPACING`.

---

## `src/game_loop.py`

### `GameLoop`

**Constructor**: Store `self.systems`.

**`get_state() → GameState`**: Create a deepcopy of `systems.manager.board` (e.g., `board_copy = self.systems.manager.board.copy()`) to ensure thread-safe iteration by the GUI. Build `GameState` using this copy, including all boolean flags.

**`request_hint() → str`**: Call `systems.manager.get_ai_move()` and return `.uci()`.

**`submit_move(uci, viewer=None) → MoveResult`**:
1. `systems.manager.validate_move(uci)`. If False, return `MoveResult(False, f"Illegal move: {uci}")`.
2. Parse move, generate ops, call `GameOrchestrator.execute_ops(physical_arm=True)`.
3. On success: push move to board, return `MoveResult(True, ...)`.
4. On exception: return `MoveResult(False, ..., error=e)`.

**`execute_ai_turn(viewer=None) → MoveResult`**:
1. Get AI move from Stockfish.
2. Generate ops, execute.
3. Push move to board.
4. Return `MoveResult`.
5. Wrap all exceptions into `MoveResult(False, ..., error=e)`.

Note: Both methods run TURN_START and TURN_END health checks if health check integration is desired. (These can be added as wrappers around the execution.)

---

## `src/gui/board_panel.py`

### `BoardPanel`

A `tk.Frame` subclass.

**Constructor**: Store `game_loop`, callbacks, and flags. Initialize `selected_square = None`. Build UI. Call `refresh()`.

**`_build_ui()`**: Create toolbar with "Hint", "Step (Logic)", "Step (Physics)" buttons, and "Step-by-Step Mode" checkbox. Create 8×8 grid of `tk.Button` widgets (one per square). Create status label (bottom). Create move history `Listbox` (bottom).

**`_on_square_click(square)`**:
- If game over: return.
- If no square selected: if clicked square has own-color piece, select it (highlight yellow).
- If square already selected: build UCI string. Auto-append "q" for pawn promotion moves. Check legal. If legal, call `on_move_callback(move_uci)`. Reset `selected_square`. Refresh.

**`refresh()`**: For each square, reset to base color. If piece present, show Unicode symbol. Update status bar with turn info, check indicator, or game over.

**`lock_board()` / `unlock_board()`**: Set all square buttons to `DISABLED` / `NORMAL`.

**`append_history(msg)`**: Insert into listbox and scroll to bottom.

**`set_status(msg)`**: Update `status_var`.

**`_request_hint()`**: Call `game_loop.request_hint()`. Show result in status bar.

**`_step_auto()`**: Call `on_move_callback(None)` to trigger AI step.

**`_step_physics()`**: Call `physics_step_callback()`.

**`_on_step_mode_toggle()`**: Set or clear `step_mode_event` based on checkbox state.

---

## `src/app.py`

### `RoboChessApp`

**Constructor**: Initialize all fields to None/empty: `systems`, `game_loop`, `root`, `panel`, `move_queue`, `physics_step_queue`, `stop_event`, `step_mode_event`.

**`run()`**:
1. Call `SystemBootstrapper.bootstrap_game_systems()` → `self.systems`.
2. Create `GameLoop(systems)`.
3. Create `tk.Tk()` root, set title.
4. Create `BoardPanel(root, game_loop, ...)`.
5. Register `WM_DELETE_WINDOW` → `on_close`.
6. Start daemon thread running `game_worker`.
7. Call `root.mainloop()`.

**`game_worker()`**:
1. Open `mujoco.viewer.launch_passive(mj_model, mj_data)`.
2. Loop while not stopping, viewer running, and game not over:
   a. Drain physics_step_queue.
   b. Get game state.
   c. If White's turn: block on `move_queue.get(timeout=0.1)`. If UCI: `submit_move`. If None and step_mode: `execute_ai_turn`.
   d. If Black's turn: if step_mode, wait for None from queue. Else immediately `execute_ai_turn`.
   e. Post-turn: `root.after(0, panel.refresh)`. Handle errors.
3. Show game-over dialog.
4. Keep viewer open in a passive loop.
5. In `finally`: call `systems.manager.close()`.

**`on_gui_move(uci)`**: `move_queue.put(uci)`.

**`on_physics_step()`**: `physics_step_queue.put("step")`.

**`on_close()`**: `stop_event.set()`. `root.destroy()`.
