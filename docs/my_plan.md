# RoboChess MuJoCo Implementation Plan (v2)

## 1. Objectives

This implementation plan refines the RoboChess MuJoCo scene and controller so that:

- The chess pieces behave physically realistically (no sinking through the board/table, no excessive jitter) under MuJoCo’s contact model.
- All robot motions remain within the pretrained FetchPickAndPlace TQC policy’s effective workspace, with accurate, meaningful `OUT_OF_POLICY_WORKSPACE` diagnostics rather than hard failures.
- Pick–place stages are robust to minor pose errors but fail fast and clearly when a catastrophic contact issue (e.g., piece falling off the board) occurs.
- The scene XML is cleanly generated from code, so future geometry or parameter changes are one‑source‑of‑truth in `scripts/generate_xml.py` and `src/settings/runtime.yaml`.

Downstream usage: this document is written for an automated refactoring agent that will edit specific files in the repository. Each task specifies concrete file paths, code locations, and expected changes.

***

## 2. New findings from logs8

### 2.1 Second move failure pattern

`logs8.txt` contains two AI moves: `e7e5` (b_pawn_5) and `d7d5` (b_pawn_4). The first completes successfully with small final errors (≈2.2 mm XY, 1.6 mm Z). The second fails in stage `DESCEND_DEST` with the piece dropping far below the board.

Key observations for the failing `d7d5` move:

- Source pose (before pick): `piece_current_xyz=[1.2151, 0.5745, 0.4120]` (Z close to `z_grasp=0.413`).
- Destination pose (square d5): `dest_xyz=[1.215, 0.715, 0.413]` (again, Z ≈ 0.413 at COM).
- During `DESCEND_DEST`, the stage target is `goal_xyz=[1.2116, 0.7146, 0.4547]` (COM plus `grasp_descend_offset`).
- Immediately before the catastrophic event, the piece is still near the destination above the board: `piece=[1.2154, 0.7147, 0.4611]` during the first `DESCEND_DEST` progress log, then `[1.2154, 0.7147, 0.4611]` in the same context.
- At the `DESCEND_DEST` stage validation, the piece has already fallen a long way: `piece=[1.2229, 0.7278, 0.1035]`, while the goal is `z≈0.4547`.
- A subsequent alignment shim then moves the gripper laterally while the piece continues falling, ending at `piece=[1.2249, 0.7315, -0.0986]` — well below the table height `0.4`.

This matches the earlier failure mode in `logs7.txt`: a pawn “falls through” the board/table contact geometry in the final descent and ends at a Z coordinate below any physical support.

### 2.2 OUT_OF_POLICY_WORKSPACE diagnostics

`ExecutionController._policy_compatibility_issues` computes XY distance and Z range violations relative to `_policy_center = np.array(FETCH_INIT_GRIP)`, with `policy_xy_radius=0.15`, `policy_z_min=0.42`, and `policy_z_max=0.75`.

In `logs8.txt`:

- For `e7e5` (b_pawn_5):
  - `src_xy_dist=0.263 m > 0.150 m`, `src_z=0.414 m < 0.420 m`.
  - `dest_xy_dist=0.200 m > 0.150 m`, `dest_z=0.413 m < 0.420 m`.
  - Acquire/transit poses at `Z_SAFE=0.56` are within Z range but outside the XY radius.
  - Despite these warnings, the operation succeeds with small final errors.

- For `d7d5` (b_pawn_4):
  - `src_xy_dist=0.216 m > 0.150 m`, `src_z=0.412 m < 0.420 m`.
  - `dest_xy` is *inside* the 0.15 m radius (≈0.128 m), so it is not flagged; `dest_z=0.413 m < 0.420 m` is flagged.
  - Acquire pose at `Z_SAFE` is again outside XY radius.

The warning message therefore correctly reflects the configured policy workspace: sources on ranks 7 and 8 are outside the conservative 0.15 m XY radius around `FETCH_INIT_GRIP`, and the pawn COM Z≈0.413 lies slightly below the configured minimum 0.42.

The **semantic mismatch** is that the TQC FetchPickAndPlace policy was likely trained on an axis-aligned rectangular workspace (object/goal anywhere within a box), whereas the code models a conservative **cylindrical** workspace centered at the home grip pose. This is not a logic bug, but it is more conservative than necessary and may generate noisy warnings even for successfully executed moves.

***

## 3. High-level changes

### 3.1 Scene and contact hardening

Goals:

- Prevent pieces from falling through the board/table near the end of `DESCEND_DEST` by ensuring adequate contact geometry and solvers.
- Provide an optional invisible safety floor slightly above the table plane to catch numerical edge cases without affecting normal gameplay.

Planned changes:

1. Increase board thickness and its collision representation.
2. Ensure the board’s collision geometry extends slightly above the visible top surface.
3. Confirm that table and board `geom.type`, `contype`, `conaffinity`, and `solimp`/`solref` settings are robust for small, tall pieces.
4. Optionally add a thin, invisible `plane` or `box` just below Z≈0.40 as a last-resort safety floor.

### 3.2 Policy workspace diagnostics

Goals:

- Keep `OUT_OF_POLICY_WORKSPACE` as a **non-fatal diagnostic**, but make it more faithful to the true TQC training distribution.
- Reduce noise: treat many early-game pawn moves as “edge-of-distribution” rather than catastrophically out-of-distribution.

Planned changes:

1. Use a rectangular workspace check aligned with `ChessPickPlaceEnv._mocap_min`/`_mocap_max`, not just a cylinder around `FETCH_INIT_GRIP`.
2. Separate warnings for:
   - “Definitely out of training box” (object/goal outside known workspace bounds).
   - “Near edge of training workspace” (inside XY bounds but close to margins, or slightly below Z min due to COM differences).
3. Optionally base `_policy_center` on `env.home_grip_pos` instead of the hard-coded `FETCH_INIT_GRIP` to reflect the actual initialized grip pose.

### 3.3 Stage-level failure detection

Goals:

- If a piece drops far below the board (e.g., Z < table height + small epsilon), treat this as a *hard failure* and stop further alignment shims in `DESCEND_DEST`.

Planned changes:

1. In `DESCEND_DEST`, after each `Waypoint progress` and at stage validation, check for catastrophic Z deviation.
2. Fail immediately (return `False`) if the piece Z is more than, e.g., 3–4 cm below the intended board COM height or below the table plane.
3. Make the failure reason explicit in logs to aid debugging.

***

## 4. Concrete tasks by file

### 4.1 Hardening the MuJoCo scene generation

**Files:**

- `scripts/generate_xml.py` (scene generator)
- `src/assets/chess_world.xml` (generated; do not hand-edit once generator is updated)

#### 4.1.1 Board and table collision geometry

**Goal:** make sure the board and table provide a thick, continuous collision volume under all squares.

**Steps:**

1. Open `scripts/generate_xml.py` and locate the board/table geometry creation functions (search for `board`, `table`, and `chess_world` generation code). Use the existing `TABLE_HEIGHT` and `Z_GRASP` from `src/config.py`.

2. Ensure the board is modeled as a `geom` of type `box` (or `plane` plus a box) with:

   - Center Z around `TABLE_HEIGHT + board_thickness / 2`.
   - `size` or `halfsize` such that the **top surface** is at or slightly above `z_grasp` (e.g., top at `Z_GRASP - 0.002`).
   - Thickness of at least `0.02`–`0.03` m to provide a robust collision.

   Example pseudo-code (do not copy literally; adapt to generator’s actual API):

   ```python
   board_thickness = 0.024  # 2.4 cm
   board_half_extents = [4 * SQUARE_SIZE + 0.01, 4 * SQUARE_SIZE + 0.01, board_thickness / 2]
   board_z = TABLE_HEIGHT + board_thickness / 2

   add_geom(
       name="board_collision",
       type="box",
       size=board_half_extents,
       pos=[BOARD_CENTER, BOARD_CENTER, board_z],
       contype="1",
       conaffinity="1",
       friction=[0.8, 0.1, 0.002],
   )
   ```

3. Ensure the table top below the board is also a `box` or `plane` that extends under the entire board region, with a Z height equal to `TABLE_HEIGHT` and sufficient thickness (e.g., 3–5 cm) and a compatible `contype`/`conaffinity`.

4. Regenerate `src/assets/chess_world.xml` by running the generation script as documented in the repo (likely `python scripts/generate_xml.py` with appropriate arguments). Commit the generated XML.

#### 4.1.2 Optional invisible safety floor

**Goal:** provide a small guard against numerical contact failures by catching pieces that fall below the board.

**Steps:**

1. In `scripts/generate_xml.py`, after defining table and board geoms, add an *invisible* safety floor just below the table.

   - Type: `plane` or a large `box`.
   - Z position: slightly above 0, but below `TABLE_HEIGHT` (e.g., `TABLE_HEIGHT - 0.05`).
   - Set `rgba="0 0 0 0"` (no visual) and `contype`/`conaffinity` so that it interacts with pieces.

   Example:

   ```python
   add_geom(
       name="safety_floor",
       type="plane",
       pos=[BOARD_CENTER, BOARD_CENTER, TABLE_HEIGHT - 0.05],
       size=[1.0, 1.0, 0.0],
       rgba=[0, 0, 0, 0],  # invisible
       contype="1",
       conaffinity="1",
   )
   ```

2. Keep this guard optional by placing the creation behind a flag in `runtime.yaml` (e.g., `enable_safety_floor: true`) if desired. For an AI agent, this can be a static `True` for now.

3. Regenerate the XML and confirm pieces cannot fall below the table in simple tests.

### 4.2 Policy workspace diagnostics improvements

**Files:**

- `src/control/execution_controller.py` (`_policy_compatibility_issues` and class-level attributes)
- `src/env/chess_pick_place_env.py` (for workspace bounds used by the env)

#### 4.2.1 Use rectangular workspace bounds

**Goal:** replace the purely cylindrical check with a combination of rectangular and radial checks.

**Steps:**

1. In `ExecutionController`, add class-level copies of the mocap bounds from the environment. For an automated agent, it is sufficient to reuse the existing reachable bounds and workspace Z limits from `src.config`:

   ```python
   from src.config import WORKSPACE_Z_MIN, WORKSPACE_Z_MAX

   _policy_xy_min = np.array([REACHABLE_X_MIN, REACHABLE_Y_MIN], dtype=np.float64)
   _policy_xy_max = np.array([REACHABLE_X_MAX, REACHABLE_Y_MAX], dtype=np.float64)
   _policy_z_min = POLICY_Z_MIN
   _policy_z_max = POLICY_Z_MAX
   ```

2. Modify `_policy_compatibility_issues` to:

   - Keep the existing radial check as a *soft* warning.
   - Add a rectangular check that compares `pos[:2]` against `_policy_xy_min/_policy_xy_max`.

   Example structure (preserve log text style):

   ```python
   @classmethod
   def _policy_compatibility_issues(cls, src_pos, dest_pos):
       issues = []
       grip_pos = cls._policy_center

       def append_if_outside(name, pos):
           xy = np.asarray(pos[:2])
           z = float(pos)

           # Rectangular workspace check
           if not (cls._policy_xy_min <= xy <= cls._policy_xy_max and
                   cls._policy_xy_min <= xy <= cls._policy_xy_max):
               issues.append(
                   f"{name}_xy={xy.round(3).tolist()} outside trained XY box "
                   f"x∈[{cls._policy_xy_min:.3f},{cls._policy_xy_max:.3f}], "
                   f"y∈[{cls._policy_xy_min:.3f},{cls._policy_xy_max:.3f}]"
               )

           # Radial check (soft warning)
           xy_dist = np.linalg.norm(xy - grip_pos[:2])
           if xy_dist > POLICY_XY_RADIUS:
               issues.append(
                   f"{name}_xy_dist={xy_dist:.3f}m exceeds trained radius {POLICY_XY_RADIUS:.3f}m"
               )

           # Z-range check
           if z < cls._policy_z_min or z > cls._policy_z_max:
               issues.append(
                   f"{name}_z={z:.3f}m outside trained z-range "
                   f"[{cls._policy_z_min:.3f}, {cls._policy_z_max:.3f}]"
               )

       append_if_outside("src", src_pos)
       append_if_outside("dest", dest_pos)
       append_if_outside("acquire", np.array([src_pos, src_pos, Z_SAFE], dtype=np.float64))
       append_if_outside("transit", np.array([dest_pos, dest_pos, Z_SAFE], dtype=np.float64))
       return issues
   ```

3. Optionally, **relax** `POLICY_Z_MIN` slightly (e.g., from 0.42 to 0.41) in `runtime.yaml` to avoid flagging the nominal pawn COM at `z_grasp=0.413` as out-of-policy. This is a configuration change, not a code change.

#### 4.2.2 Align policy center with actual home grip pose

**Goal:** ensure `OUT_OF_POLICY_WORKSPACE` distances are measured around the *actual* reset grip pose, not a slightly idealized value.

**Steps:**

1. In `ExecutionController`, replace the class-level `_policy_center = np.array(FETCH_INIT_GRIP, ...)` with a per-instance copy initialized from the environment:

   ```python
   class ExecutionController:
       _policy_center = None  # class default

       def __init__(self, rl_model, env: ChessPickPlaceEnv):
           self.rl_model = rl_model
           self.env = env
           self.__class__._white_graveyard_count = 0
           self.__class__._black_graveyard_count = 0
           self._pregrasp_piece_rest_z = Z_GRASP
           self._placement_goal = None
           self._stage_event_handler = None

           # Initialize policy center from the actual home grip pose
           grip_home = getattr(env, "home_grip_pos", None)
           if grip_home is not None:
               self.__class__._policy_center = np.asarray(grip_home, dtype=np.float64)
           else:
               self.__class__._policy_center = np.array(FETCH_INIT_GRIP, dtype=np.float64)
   ```

2. Ensure `_policy_compatibility_issues` gracefully handles `_policy_center` being `None` (e.g., skip checks if unset), although in normal usage it will always be set during initialization.


### 4.3 Stage-level catastrophic failure detection

**File:** `src/control/execution_controller.py` (`_execute_stage` → `DESCEND_DEST` and `_validate_motion_stage`).

#### 4.3.1 Add Z-based catastrophic-failure guard

**Goal:** if a piece falls significantly below the board plane or destination COM during `DESCEND_DEST`, stop trying to align and mark the stage as failed.

**Steps:**

1. In `_execute_stage`, inside the `"DESCEND_DEST"` branch, after the initial `success, steps_taken = self._move_gripper_to(...)`, insert a check on the piece Z before computing XY/Z errors:

   ```python
   if stage == "DESCEND_DEST":
       self.env.set_target(piece_name, goal)
       success, steps_taken = self._move_gripper_to(
           goal,
           viewer=viewer,
           gripper_opening=GRIPPER_CLOSED,
           require_piece_follow=True,
           max_piece_drift=PIECE_DRIFT_TOLERANCE * 3,
           max_cartesian_action=0.25,
           rl_vertical_only=True,
           use_rl=False,
       )

       piece_pos = self.env.get_piece_pos()

       # Catastrophic Z guard: piece fell well below the board/table
       table_z = TABLE_HEIGHT  # import TABLE_HEIGHT from src.config
       if piece_pos < table_z + 0.01:
           logger.warning(
               "    Catastrophic Z deviation in DESCEND_DEST | "
               f"piece_z={piece_pos:.3f} < table_z+1cm ({table_z + 0.01:.3f})"
           )
           return False, steps_taken

       xy_error = np.linalg.norm(piece_pos[:2] - self._placement_goal[:2])
       z_error = abs(float(piece_pos) - float(self._placement_goal))
       # existing logic continues...
   ```

   Make sure to import `TABLE_HEIGHT` from `src.config` at the top of the file.

2. Similarly, before running the alignment shim, add a guard so that alignment is only attempted when the piece is roughly at the intended height (within, say, ±3 cm):

   ```python
       if xy_error <= PLACEMENT_TOLERANCE and z_error <= PLACEMENT_TOLERANCE:
           return True, steps_taken

       # Only attempt lateral alignment if the piece is near the board surface
       if abs(piece_pos - self._placement_goal) <= 0.03:
           align_goal = self.env.get_grip_pos().copy()
           align_goal[:2] += self._placement_goal[:2] - piece_pos[:2]
           # existing align_success logic...
       return success, steps_taken
   ```

3. This prevents the situation seen in `logs8.txt` where the alignment shim is executed while the piece has already fallen far below the board, exacerbating the failure.


### 4.4 Minor robustness improvements and cleanups

**Files:**

- `src/env/chess_pick_place_env.py`
- `src/control/execution_controller.py`

#### 4.4.1 Ensure idle-stable damping for non-target pieces

This is already implemented via `set_piece_active_damping` and is used correctly in `set_target` to switch only the active piece to low damping. No change needed, but an AI agent should confirm that:

- All piece names starting with `"w_"` or `"b_"` (excluding spares) are included in `_piece_joint_dofadrs` and `_piece_mesh_geom_ids`.
- No piece is left in `ACTIVE_PIECE_FREEJOINT_DAMPING` at the end of an operation.

If any discrepancy is found, fix the iteration logic so all relevant piece bodies are covered.

#### 4.4.2 Keep OUT_OF_POLICY_WORKSPACE non-fatal

Do **not** convert `OUT_OF_POLICY_WORKSPACE` into a hard failure. The logs from `logs8.txt` show that out-of-policy moves like `e7e5` can still succeed with small errors. The only planned change is to make the diagnostic more informative and better aligned with the training workspace (Section 4.2).


***

## 5. Implementation order for an AI agent

This subsection gives a recommended execution order with clear checkpoints.

1. **Scene hardening (board/table and safety floor)**
   1. Edit `scripts/generate_xml.py` to adjust board/table geoms as in 4.1.
   2. Regenerate `src/assets/chess_world.xml`.
   3. Run a short script or manual test where a pawn is teleported to various squares and lightly dropped from above to confirm it never sinks below the board/table.

2. **Policy workspace diagnostics**
   1. Modify `ExecutionController._policy_compatibility_issues` to add rectangular workspace checks and keep radial checks as soft warnings.
   2. Initialize `_policy_center` from `env.home_grip_pos`.
   3. Optionally relax `policy_z_min` in `runtime.yaml` from 0.42 to 0.41 to better match `z_grasp=0.413`.

3. **Stage-level catastrophic Z guard**
   1. Add the `TABLE_HEIGHT`-based Z guard inside `DESCEND_DEST` as described in 4.3.1.
   2. Restrict the alignment shim to cases where `|piece_z - placement_goal_z| <= 0.03`.

4. **Regression tests against logs7/logs8 scenarios**
   1. Re-run the moves from `logs7.txt` and `logs8.txt` (e.g., initial `e7e5`, `d7d5`) and confirm:
      - No piece sinks below the board/table.
      - `OUT_OF_POLICY_WORKSPACE` warnings are still emitted but now show rectangular bounds.
      - `DESCEND_DEST` either succeeds or fails with a clear “catastrophic Z deviation” warning if contact is still broken.

5. **Optional future work** (not required for this iteration):
   1. Add a simple scripted test harness that sweeps all 32 initial pieces through a pick–place cycle and records success/failure statistics.
   2. If failures remain clustered at specific squares, adjust the board placement or REACHABLE_* bounds slightly so that all reachable squares lie comfortably within both the robot workspace and the policy’s training region.

***

## 6. Summary of key bug fixes

- **Board/table collision:** strengthen and thicken collision geometry so pieces cannot fall through during `DESCEND_DEST`, addressing failures like the `d7d5` pawn in `logs8.txt` that ended at `z≈-0.0986`.
- **Policy diagnostics:** refine `OUT_OF_POLICY_WORKSPACE` checks to combine rectangular and radial constraints, and base the policy center on the actual home grip pose.
- **Catastrophic Z guard:** add explicit Z-based failure detection in `DESCEND_DEST` to prevent alignment shims from running when a piece has already dropped far below the board.

These changes directly target the newly observed failure in `logs8.txt` while remaining consistent with the existing staged controller architecture.