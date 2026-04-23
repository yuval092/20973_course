# 05 — Validation and Sanity-Check Scripts

These scripts are standalone tools for verifying correctness during development. They are not part of the test suite (`pytest`). Run them manually after each implementation stage to catch configuration errors, integration problems, and physics regressions early.

All scripts must:
- Add the project root to `sys.path` if run directly.
- Use structured logging (`logging_utils.log_event`) for output.
- Exit with code 0 on success, code 1 on failure.
- Be runnable headlessly (no GUI, no interactive input).

---

## `scripts/verify_mujoco.py`

**Purpose**: Confirm MuJoCo is installed and the Python binding works.

**Usage**:
```bash
python scripts/verify_mujoco.py
```

**What it does**:
1. Import `mujoco` and print the version string.
2. Create a minimal inline XML model (a box body with gravity).
3. Create `MjData`, run 10 `mj_step` calls.
4. Assert `data.time > 0`.
5. Print "MuJoCo OK".

**Expected output**:
```
MuJoCo version: 3.x.x
Ran 10 steps. Time: 0.0100
MuJoCo OK
```

**Failure indicates**: MuJoCo not installed, wrong version, or broken Python binding.

---

## `scripts/verify_scene.py`

**Purpose**: Confirm `chess_world.xml` loads correctly and all 32 pieces are placed on the board.

**Usage**:
```bash
python scripts/verify_scene.py
```

**What it does**:
1. Load `SCENE_XML` with `mujoco.MjModel.from_xml_path(...)`.
2. Create `MjData`, call `mj_forward`.
3. Call `SystemBootstrapper.initialize_square_to_piece(model, data)`.
4. Print a table of all mapped squares and body names.
5. Assert `len(square_to_piece) == 32`.
6. Assert no duplicate body names.
7. Print "Scene OK: 32 pieces mapped."

**Expected output**:
```
a1 → w_rook_1     at [1.160, 0.544, 0.413]
a2 → w_pawn_1     at [1.160, 0.464, 0.413]
...
Scene OK: 32 pieces mapped.
```

Note: exact coordinates depend on `BOARD_CENTER` and `SQUARE_SIZE` in `runtime.yaml`. With `board_center=[0.88, 0.2641, 0.4]` and `square_size=0.08`: `start_x=0.56`, `start_y=0.5841`; a1 (file 0, rank 0) → `x=1.16, y=0.5441`; rank increases in the **negative** Y direction.

**Failure indicates**: XML parse error, wrong board geometry in YAML, or piece body naming mismatch.

---

## `scripts/verify_home_pose.py`

**Purpose**: Confirm the robot arm initializes to the expected home pose.

**Usage**:
```bash
python scripts/verify_home_pose.py
```

**What it does**:
1. Load scene, create `ChessPickPlaceEnv`.
2. Print `env.home_grip_pos`.
3. Compare to `FETCH_INIT_GRIP` from config; assert within 0.05m.
4. Assert `home_grip_pos` is within `[REACHABLE_X_MIN/MAX, REACHABLE_Y_MIN/MAX, REACHABLE_Z_MIN/MAX]`.
5. Print "Home pose OK."

**Expected output**:
```
Home grip pos: [1.3419, 0.7491, 0.5750]
Config FETCH_INIT_GRIP: [1.3419, 0.7491, 0.5750]
Distance from config: 0.0000m
Home pose OK.
```

**Failure indicates**: Scene XML or Fetch robot initialization not matching expected grip position.

---

## `scripts/verify_rl_model.py`

**Purpose**: Confirm the local RL model loads and produces valid predictions.

**Usage**:
```bash
python scripts/verify_rl_model.py
```

**What it does**:
1. Load scene, create `ChessPickPlaceEnv`.
2. Call `SystemBootstrapper.load_rl_policy(env, LOCAL_MODEL_PATH)`.
3. Print model type and policy type.
4. Create a zero observation dict with correct shapes.
5. Call `model.predict(obs, deterministic=True)`.
6. Assert output action shape == `(4,)`.
7. Assert all action values in `[-1.5, 1.5]` (slightly relaxed from [-1,1] for clipped output).
8. Print "RL model OK."

**Expected output**:
```
Loading model from: /path/to/model.zip
Model type: SAC
Action: [0.123, -0.045, 0.678, 0.000]
Action shape: (4,)
RL model OK.
```

**Failure indicates**: Model file not found, format incompatible with SAC, or observation shape mismatch.

---

## `scripts/headless_run.py`

**Purpose**: Execute a single pick-and-place operation without a GUI. Primary smoke test for the execution pipeline.

**Usage**:
```bash
python scripts/headless_run.py
```

**What it does**:
1. Load scene, create `ChessPickPlaceEnv` and `ExecutionController(rl_model=None, env)`.
2. Create a `PickPlaceOp` for a mid-board move (e.g., `f4 → f5`, piece `w_pawn_6`).
3. Call `controller.execute_op(op, viewer=None)`.
4. Print `result.details` and full piece state (XYZ, quaternion, up_z).
5. Assert `result.success == True`.
6. Print "Headless run OK."

**Expected output**:
```
Starting headless op | piece=w_pawn_6 | src=f4 | dest=f5
SUCCESS: xy_error=3.2mm, z_error=1.1mm, stages=10/10, steps=847
Post-op piece state | pos=[...] | up_z=0.9998
Headless run OK.
```

**Failure indicates**: Execution controller broken, piece not reachable, physics unstable, or stage logic error.

---

## `scripts/verify_board_mapping.py`

**Purpose**: Print a human-readable view of the full physical board layout at startup.

**Usage**:
```bash
python scripts/verify_board_mapping.py
```

**What it does**:
1. Load scene and build `square_to_piece`.
2. Print an 8×8 grid showing which piece body is at each square, with its physical XYZ position.
3. Print a summary: count of white pieces, black pieces, and any unmapped squares.

**Expected output**:
```
Board layout (file a-h, rank 1-8):
  a8: b_rook_1       [1.160, -0.016, 0.413]
  b8: b_knight_1     [1.080, -0.016, 0.413]
  ...
  e2: w_pawn_5       [0.840, 0.464, 0.413]
  ...
White: 16, Black: 16, Unmapped: 0
Mapping OK.
```

Note: with `board_center=[0.88, 0.2641, 0.4]` and `square_size=0.08`, rank 1 is at Y≈0.544 and rank 8 is at Y≈-0.016. Rank increases in the negative Y direction.

**Failure indicates**: Piece body positions don't align with expected squares; board geometry mismatch.

---

## `scripts/drift_env_test.py`

**Purpose**: Verify that idle physics simulation does not cause piece drift or instability over time.

**Usage**:
```bash
python scripts/drift_env_test.py
```

**What it does**:
1. Load scene, create `ChessPickPlaceEnv`.
2. Snapshot initial piece positions.
3. Run 500 idle `mujoco.mj_step` calls (no actions).
4. For each piece, compare final position to initial position.
5. Assert maximum drift < 0.001m (1mm) for any piece.
6. Call `BoardObserver.verify_stability(active_piece_names=all_piece_names)`.
7. Print per-piece drift summary.
8. Print "Drift test OK."

**Expected output**:
```
Running 500 idle physics steps...
Max piece drift: 0.0002m (w_pawn_3)
All pieces stable.
Drift test OK.
```

**Failure indicates**: Physics configuration causing piece creep (check damping, noslip settings, timestep).

---

## `scripts/verify_obs_shape.py`

**Purpose**: Verify the observation vector shape and content are correct.

**Usage**:
```bash
python scripts/verify_obs_shape.py
```

**What it does**:
1. Load scene, create `ChessPickPlaceEnv`.
2. Call `env.set_target("w_pawn_1", env.get_grip_pos())`.
3. Call `env.get_obs()`.
4. Print each feature slice with its values.
5. Assert `obs["observation"].shape == (26,)`.
6. Assert `obs["observation"].dtype == np.float32`.
7. Assert `obs["achieved_goal"].shape == (3,)`.
8. Assert `0.0 <= obs["observation"][25] <= 1.0` (time feature).
9. Print "Observation OK."

**Expected output**:
```
observation shape: (26,)
observation dtype: float32
grip_pos:          [1.342, 0.749, 0.575]
obj_pos:           [1.342, 0.749, 0.575]
obj_rel_pos:       [0.000, 0.000, 0.000]
gripper_state:     [0.050, 0.050]
obj_rot:           [0.000, 0.000, 0.000]
time_feature:      1.0000
achieved_goal:     [1.342, 0.749, 0.575]
desired_goal:      [1.342, 0.749, 0.575]
Observation OK.
```

**Failure indicates**: Observation vector wrong shape/dtype; GRIP_SITE name wrong; n_substeps/timestep configuration issue.

---

## `scripts/generate_xml.py`

**Purpose**: Programmatically generate `src/assets/chess_world.xml` from the config values. This ensures the XML geometry is always consistent with `runtime.yaml`.

**Usage**:
```bash
python scripts/generate_xml.py
```

**What it does**:
1. Read `BOARD_CENTER`, `SQUARE_SIZE`, `TABLE_HEIGHT`, `Z_GRASP`, graveyard config from `src/config.py`.
2. Generate 64 square geom boxes using the computed positions.
3. Generate 32 piece bodies at their starting positions, each with correct freejoint, box hitbox (not cylinder), mesh geom, and site at geometric CoM.
4. Generate 2 spare promotion pieces per type per color, placed in the **spare staging area** on top of the table (at `SPARE_STAGING_ORIGIN`, off the +Y edge of the board). Do NOT place spares below the table — the arm cannot physically reach them there.
5. Generate graveyard platform geoms.
6. Include `<include file="fetch.xml"/>`.
7. Write the complete XML to `src/assets/chess_world.xml`.
8. Print "XML generated: N bodies, M geoms."

**After running**: Immediately run `scripts/verify_scene.py` to confirm the generated scene is valid.

**Failure indicates**: Config values inconsistent, XML template error, or body naming convention mismatch.

---

## Development Stage Checklist

Use these scripts in this order as you implement each stage:

| Stage | Scripts to run |
|-------|----------------|
| Stage 0 (Scaffold) | (none yet) |
| Stage 1 (Config) | `verify_mujoco.py` (confirms environment setup) |
| Stage 5 (Chess logic) | (run chess manager tests only) |
| Stage 6 (Scene assets) | `verify_mujoco.py`, `verify_scene.py`, `verify_board_mapping.py` |
| Stage 7 (Physics env) | `verify_home_pose.py`, `verify_obs_shape.py` |
| Stage 8 (Board observer) | `drift_env_test.py` |
| Stage 9 (Execution controller) | `headless_run.py` |
| Stage 10 (Bootstrap + RL model) | `verify_rl_model.py`, then `headless_run.py` again with real model |
| Stage 15 (Full validation) | All scripts in sequence |
