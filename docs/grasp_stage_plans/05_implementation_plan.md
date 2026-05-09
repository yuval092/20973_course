# Implementation Plan: Grasp Stage

## Guiding Principles

1. **Physics first** — verify physics changes in isolation before any code changes
2. **Scripted GRASP, RL everything else** — do not retrain the model
3. **Cube never falls** — abort the chain on any cube drop; do not continue
4. **Explicit preconditions** — every phase checks its entry conditions; no silent assumptions
5. **Full observability** — every step logs cube position, gripper state, and contact forces

---

## Files Overview

| File | Action | Description |
|:---|:---|:---|
| `chess_env/assets/pick_and_place.xml` | **Modify** | Cube physics: mass, condim, solref, solimp, actuator Kp, ctrlrange |
| `chess_env/assets/robot.xml` | **Modify** | Finger condim=6, solref/solimp on finger geoms |
| `chess_env/assets/shared.xml` | **Modify** | Weld solref=0.02 |
| `configs/env.yaml` | **Modify** | grasp_z=0.425, home_position_xy, new grasp thresholds |
| `src/chess_env/simulation.py` | **Modify** | Add grasp_mode branch in `_set_action` |
| `src/chess_env/task.py` | **Modify** | Add `execute_grasp()`, `get_cube_position()`, `get_cube_quat()`, `_check_cube_held()`, force_start_pos/force_cube_pos support |
| `scripts/test_grasp_physics.py` | **Create** | Physics verification script (run BEFORE anything else) |
| `scripts/eval_grasp.py` | **Create** | Full sequence evaluation script |

---

## Step 1: XML Physics Hardening

### 1a. `chess_env/assets/pick_and_place.xml`

Apply the following changes (see doc 02 for full rationale):

**Cube geom** — change this exact element:
```xml
<!-- FROM: -->
<geom size="0.015 0.015 0.015" type="box" condim="4" name="object0"
      material="block_mat" mass="0.5" friction="2.0 0.5 0.01"/>

<!-- TO: -->
<geom size="0.015 0.015 0.015" type="box" condim="6" name="object0"
      material="block_mat" mass="0.05"
      friction="2.0 0.005 0.0001"
      solref="0.002 1" solimp="0.99 0.999 0.001"/>
```

**Actuators** — change both:
```xml
<!-- FROM: -->
<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint"
          kp="60000" name="robot0:l_gripper_finger_joint" user="1"/>
<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint"
          kp="60000" name="robot0:r_gripper_finger_joint" user="1"/>

<!-- TO: -->
<position ctrllimited="true" ctrlrange="0 0.05" joint="robot0:l_gripper_finger_joint"
          kp="150000" name="robot0:l_gripper_finger_joint" user="1"/>
<position ctrllimited="true" ctrlrange="0 0.05" joint="robot0:r_gripper_finger_joint"
          kp="150000" name="robot0:r_gripper_finger_joint" user="1"/>
```

### 1b. `chess_env/assets/robot.xml`

**Both finger geoms** — add condim, solref, solimp; adjust friction:
```xml
<!-- l_gripper_finger_link geom, FROM: -->
<geom pos="0 0.008 0" size="0.0385 0.007 0.0135" type="box"
      name="robot0:l_gripper_finger_link" material="robot0:gripper_finger_mat"
      condim="4" friction="10.0 0.5 0.01"/>

<!-- TO: -->
<geom pos="0 0.008 0" size="0.0385 0.007 0.0135" type="box"
      name="robot0:l_gripper_finger_link" material="robot0:gripper_finger_mat"
      condim="6" friction="10.0 0.005 0.0001"
      solref="0.002 1" solimp="0.99 0.999 0.001"/>

<!-- Same change for r_gripper_finger_link -->
```

### 1c. `chess_env/assets/shared.xml`

**Weld constraint**:
```xml
<!-- FROM: -->
<weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.01 1"/>

<!-- TO: -->
<weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.02 1"/>
```

### 1d. `configs/env.yaml`

Add/modify:
```yaml
grasp_z: 0.425              # CHANGED from 0.430 — 78% cube overlap, 6.5mm table clearance (safe minimum)

# Home Position (new)
home_position_xy: [0.680, 0.2641]   # within transit training range [0.640, 1.120]; NOT 0.600 (outside range)

# Grasp Phase Thresholds (new)
grasp_contact_approach_tolerance: 0.001   # 1mm fine-descent tolerance
grasp_close_steps: 150                    # Steps for actuator-driven finger close
grasp_hold_steps: 50                      # Hold steps after close to let physics settle
grasp_verify_xy_threshold: 0.015          # 15mm — max cube-to-grip XY error to verify hold
grasp_verify_z_threshold: 0.020           # 20mm — max cube-to-grip Z error to verify hold
grasp_verify_finger_threshold: 0.012      # finger stalls at j≈0.0143; threshold BELOW stall (0.016 falsely rejects valid grasps)

# Cube Hold Monitoring (new)
cube_held_xy_limit: 0.030                 # 30mm — if cube exceeds this from grip, it's dropped
cube_held_z_limit: 0.020                  # 20mm — tighter limit: 40mm allows 45mm of ascent before detecting a drop
```

---

## Step 2: `simulation.py` — Grasp Mode in `_set_action`

Modify `_set_action` to add a `grasp_mode` branch:

```python
def _set_action(self, action):
    assert action.shape == (4,)
    action = action.copy()
    pos_ctrl, gripper_ctrl = action[:3], action[3]
    pos_ctrl *= self.POS_CTRL_SCALE
    rot_ctrl = np.zeros(4)

    target_qpos = getattr(self, "finger_target_joint", 0.0)

    if getattr(self, 'grasp_mode', False):
        # Actuator-driven mode: set ctrl target only.
        # Let MuJoCo's Kp controller drive the fingers with contact physics enabled.
        # The cube can push back against the fingers; contact forces are computed.
        self.data.ctrl[0] = target_qpos
        self.data.ctrl[1] = target_qpos
        # IMPORTANT: do NOT call set_joint_qpos or set_joint_qvel here.
    else:
        # Teleport mode: absolute enforcement (no cube contact during movement phases).
        self.data.ctrl[0] = target_qpos
        self.data.ctrl[1] = target_qpos
        self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target_qpos)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target_qpos)
        self._utils.set_joint_qvel(self.model, self.data, "robot0:l_gripper_finger_joint", 0.0)
        self._utils.set_joint_qvel(self.model, self.data, "robot0:r_gripper_finger_joint", 0.0)

    mocap_action = np.concatenate([pos_ctrl, rot_ctrl])
    self._utils.mocap_set_action(self.model, self.data, mocap_action)
```

---

## Step 3: `task.py` — New Methods and Modified Logic

### 3a. New constants in `__init__`

```python
# Grasp control
self.grasp_mode = False

# Home position
home_xy = self.env_cfg.get("home_position_xy", [0.680, 0.2641])
self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

# Grasp thresholds
self.GRASP_CONTACT_APPROACH_TOLERANCE = self.env_cfg.get("grasp_contact_approach_tolerance", 0.001)
self.GRASP_CLOSE_STEPS = self.env_cfg.get("grasp_close_steps", 150)
self.GRASP_HOLD_STEPS = self.env_cfg.get("grasp_hold_steps", 50)
self.GRASP_VERIFY_XY_THRESHOLD = self.env_cfg.get("grasp_verify_xy_threshold", 0.015)
self.GRASP_VERIFY_Z_THRESHOLD = self.env_cfg.get("grasp_verify_z_threshold", 0.020)
self.GRASP_VERIFY_FINGER_THRESHOLD = self.env_cfg.get("grasp_verify_finger_threshold", 0.012)
self.CUBE_HELD_XY_LIMIT = self.env_cfg.get("cube_held_xy_limit", 0.030)
self.CUBE_HELD_Z_LIMIT = self.env_cfg.get("cube_held_z_limit", 0.020)

# Start position override (used by eval_grasp.py)
self.force_start_pos = None
self.force_cube_pos = None
```

### 3b. `_reset_sim` changes

```python
def _reset_sim(self) -> bool:
    # ... existing scenario selection code ...
    
    # NEW: reset grasp mode on every reset
    self.grasp_mode = False
    
    # NEW: support force_start_pos (Home Position override)
    # CRITICAL: force_start_pos MUST be applied here inside _reset_sim.
    # Declaring it in __init__ is NOT enough — _reset_sim always computes arm_start_pos
    # from _sample_board_position() unless explicitly overridden here.
    # Failure to implement this override means eval_grasp.py's HOME→src transit
    # is never actually tested from HOME — the arm starts at a random board position.
    if self.current_scenario == "transit":
        # ... existing goal sampling ...
        if self.force_start_pos is not None:
            arm_start_pos = self.force_start_pos.copy()
        else:
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
    
    # NEW: support force_cube_pos.
    # The actual _reset_sim writes directly to self.data.qpos — there is no
    # intermediate "cube_pos" variable. The override must be structured as an
    # elif branch inside the existing hide_object block (lines 288–299 of task.py):
    #
    #   if self.hide_object:
    #       self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
    #       ...
    #   elif self.force_cube_pos is not None:          ← NEW elif branch
    #       self.data.qpos[qpos_start : qpos_start + 3] = self.force_cube_pos[:3]
    #       self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
    #       self.data.qvel[dof_start  : dof_start  + 6] = 0.0
    #   else:
    #       self.data.qpos[qpos_start : qpos_start + 2] = start_xy
    #       ...
    #   mujoco.mj_forward(self.model, self.data)
    #
    # DO NOT assign to a local "cube_pos" variable and expect it to be used later —
    # the existing code never reads such a variable. The write must go to qpos directly.
```

### 3c. `get_cube_position()` and `get_cube_quat()`

```python
def get_cube_position(self) -> np.ndarray:
    """Returns the current 3D world position of the cube."""
    obj_joint_id = self.model.joint("object0:joint").id
    qpos_start = self.model.jnt_qposadr[obj_joint_id]
    return self.data.qpos[qpos_start : qpos_start + 3].copy()

def get_cube_quat(self) -> np.ndarray:
    """Returns the quaternion orientation of the cube (w, x, y, z)."""
    obj_joint_id = self.model.joint("object0:joint").id
    qpos_start = self.model.jnt_qposadr[obj_joint_id]
    return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()
```

### 3d. `_check_cube_held()`

```python
def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
    """
    Checks that the cube is still within the gripper.
    Only called when grasp_mode=True and current_scenario is 'ascend' or 'transit'.
    """
    cube_pos = self.get_cube_position()
    xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
    # Cube CoM is ~15mm below grip site when held at center; allow 40mm total Z tolerance
    z_error = abs(cube_pos[2] - (grip_pos[2] - 0.015))
    
    if xy_error > self.CUBE_HELD_XY_LIMIT:
        return False, f"CUBE_DROPPED_XY (err={xy_error*1000:.1f}mm > limit={self.CUBE_HELD_XY_LIMIT*1000:.0f}mm)"
    if z_error > self.CUBE_HELD_Z_LIMIT:
        return False, f"CUBE_DROPPED_Z (err={z_error*1000:.1f}mm > limit={self.CUBE_HELD_Z_LIMIT*1000:.0f}mm)"
    return True, None
```

### 3e. `execute_grasp()`

**AMENDED — see doc 10 for full improvement roadmap.**

Critical ordering bug found empirically: the halt loop's zero-velocity step must happen BEFORE
the 50 simulation steps, not after. During the halt loop, the arm oscillates near GRASP_Z with
residual velocity from the RL policy. At GRASP_Z, the finger bottom is already at 0.4065m
(below the cube top at 0.430m), so oscillation can trigger proximity contact explosions that
launch the cube BEFORE finger close. This accounts for ~5% of full-pipeline failures.

Also: add early-abort FINGER_CLOSED_EMPTY detection in the close loop.

```python
def execute_grasp(self) -> dict:
    """
    Scripted GRASP phase. Executes immediately after DESCEND success.
    Returns a plain dict (NOT a dataclass). All call sites use result["success"].
    
    Phases:
      0: Zero velocity (IMMEDIATELY) + short settle (15 steps)
      0.5: Pre-conditions + cube sanity + cube rotation check
      1: Contact Approach (fine-descent to GRASP_Z, 1mm/step)
      2: Finger Close (actuator-driven, 150 steps, with early-abort)
      3: Hold Settle (50 steps)
      4: Final Verification
    """
    ROBOT_DOF = 15
    dummy_action = np.zeros(4)
    
    # ── Phase 0: ZERO VELOCITY FIRST, then short settle ──────────────────
    # CRITICAL: Zero qvel/qacc BEFORE running any steps. The arm arrives with
    # residual velocity from the RL policy. Running 50 simulation steps BEFORE
    # zeroing velocity causes the arm to oscillate near GRASP_Z, triggering
    # proximity contact explosions that launch the cube (~5% of full pipeline).
    # By zeroing first, the arm is stationary from step 1 of settle.
    self.data.qvel[:ROBOT_DOF] = 0.0
    self.data.qacc[:ROBOT_DOF] = 0.0
    mujoco.mj_forward(self.model, self.data)
    
    zero_action = np.zeros(4)
    for _ in range(15):  # Short settle (arm already stationary — just equilibrate weld)
        self._set_action(zero_action)
        self._mujoco_step(None)
    
    # ── Precondition Checks ───────────────────────────────────────────────
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
    speed = float(np.linalg.norm(grip_vel))
    if speed > 0.005:  # > 5mm/s after halt — something is wrong
        result["reason"] = f"PRECONDITION_SPEED ({speed*1000:.2f}mm/s after halt)"
        return result
    
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - self.GRASP_Z) > 0.010:
        result["reason"] = f"PRECONDITION_Z (grip at {grip_pos[2]*1000:.1f}mm, GRASP_Z={self.GRASP_Z*1000:.1f}mm)"
        return result
    
    cube_pos = self.get_cube_position()
    result["pre_grasp_cube_xy"] = cube_pos[:2].copy()
    xy_to_cube = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
    if xy_to_cube > 0.010:
        result["reason"] = f"PRECONDITION_XY (cube is {xy_to_cube*1000:.1f}mm from grip site)"
        return result
    
    l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
    if l_finger < self.FINGER_OPEN_JOINT - 0.002:
        result["reason"] = f"PRECONDITION_FINGERS_NOT_OPEN (finger={l_finger:.4f})"
        return result
    
    # ── Sub-Phase 1: Contact Approach (Fine-Descent) ──────────────────────
    # CRITICAL: DO NOT SKIP THIS PHASE. DESCEND success threshold is 10mm, so
    # the arm can succeed at GRASP_Z + 9mm = 0.434m. Without fine-descent, the
    # cube overlap is only 48% instead of 78%, reducing pipeline grasp success
    # from ~84% to ~78%. This phase is the single most impactful step.
    #
    # CRITICAL ORDER: _set_action calls mocap_set_action → reset_mocap2body_xpos,
    # which resets mocap_pos to the current body position (erasing any direct
    # modification). step_vec MUST be applied AFTER _set_action or it will be
    # silently erased before the physics step.
    # Correct order: (1) _set_action(zero_action), (2) mocap_pos += step_vec, (3) _mujoco_step()
    target_z = self.GRASP_Z
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - target_z) > self.GRASP_CONTACT_APPROACH_TOLERANCE:
        fine_target = np.array([grip_pos[0], grip_pos[1], target_z])
        for _ in range(50):
            grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = fine_target - grip_pos_now
            if np.linalg.norm(error) < self.GRASP_CONTACT_APPROACH_TOLERANCE:
                break
            step_vec = 0.5 * error
            if np.linalg.norm(step_vec) > 0.003:
                step_vec = step_vec / np.linalg.norm(step_vec) * 0.003  # 3mm/step max
            # CRITICAL ORDER: _set_action calls mocap_set_action → reset_mocap2body_xpos,
            # which resets mocap_pos to the current *body* position (erasing any direct
            # modification). step_vec MUST be applied AFTER _set_action or it will be
            # silently erased before the physics step, causing the arm to never move.
            self._set_action(zero_action)           # 1. Reset mocap_pos → body pos
            self.data.mocap_pos[0][:3] += step_vec  # 2. Add step_vec on top
            self._mujoco_step(None)                 # 3. Physics sees correct target
        else:
            grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            if abs(grip_pos_now[2] - target_z) > 0.005:
                result["reason"] = f"CONTACT_APPROACH_FAILED (z_err={abs(grip_pos_now[2]-target_z)*1000:.1f}mm)"
                return result
    
    # ── Sub-Phase 2: Finger Close (Actuator-Driven) ────────────────────────
    self.grasp_mode = True
    self.finger_target_joint = self.FINGER_CLOSED_JOINT
    close_steps = 0
    
    for step in range(self.GRASP_CLOSE_STEPS):
        self._set_action(zero_action)
        self._mujoco_step(None)
        close_steps += 1
        
        if step % 10 == 0:
            cube_now = self.get_cube_position()
            if cube_now[2] > self.GRASP_Z + 0.050:
                result["reason"] = f"CUBE_EXPLOSION_DURING_CLOSE (cube_z={cube_now[2]*1000:.1f}mm)"
                return result
        
        # Early-abort FINGER_CLOSED_EMPTY: if fingers nearly fully closed after >30 steps
        # with no cube resistance, the cube is not in the finger path → abort immediately.
        # Without early abort, the full 150 steps elapse uselessly.
        if step > 30:
            l_finger_now = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if l_finger_now < 0.003:  # Fingers closed fully — no cube contact
                result["reason"] = f"FINGER_CLOSED_EMPTY_EARLY (finger={l_finger_now:.4f} after {step} steps)"
                result["close_steps_used"] = step + 1
                return result
        
        # NOTE: fingers WITH cube stall at j≈0.0143, never reach 0.012 or below.
    
    result["close_steps_used"] = close_steps
    
    # ── Sub-Phase 3: Grasp Verification ───────────────────────────────────
    cube_pos = self.get_cube_position()
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
    
    xy_error = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
    # Z check: cube is on table during grasp (CoM ≈ 0.415m), grip site at GRASP_Z=0.425.
    # Gap is ~10mm. GRASP_VERIFY_Z_THRESHOLD=20mm covers this. Do NOT use (grip_pos[2]-0.015)
    # here — the 15mm hang only occurs when cube is in mid-air, not resting on table.
    z_error  = float(abs(cube_pos[2] - grip_pos[2])) * 1000
    
    result["final_xy_error_mm"] = xy_error
    result["final_z_error_mm"] = z_error
    result["final_finger_pos"] = l_finger
    
    if xy_error > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
        result["reason"] = f"VERIFY_XY_FAILED ({xy_error:.1f}mm > {self.GRASP_VERIFY_XY_THRESHOLD*1000:.0f}mm)"
        return result
    if z_error > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
        result["reason"] = f"VERIFY_Z_FAILED ({z_error:.1f}mm > {self.GRASP_VERIFY_Z_THRESHOLD*1000:.0f}mm)"
        return result
    if l_finger > self.GRASP_VERIFY_FINGER_THRESHOLD:
        result["reason"] = f"VERIFY_FINGERS_FAILED (finger={l_finger:.4f} > threshold={self.GRASP_VERIFY_FINGER_THRESHOLD})"
        return result
    
    # ── Sub-Phase 4: Hold Settle ───────────────────────────────────────────
    for _ in range(self.GRASP_HOLD_STEPS):
        self._set_action(zero_action)
        self._mujoco_step(None)
    
    cube_pos_final = self.get_cube_position()
    grip_pos_final = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    final_xy = float(np.linalg.norm(cube_pos_final[:2] - grip_pos_final[:2])) * 1000
    
    if final_xy > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
        result["reason"] = f"CUBE_DRIFTED_AFTER_HOLD ({final_xy:.1f}mm)"
        return result
    
    result["success"] = True
    result["post_grasp_cube_pos"] = cube_pos_final.tolist()
    result["post_grasp_cube_quat"] = self.get_cube_quat().tolist()
    return result
```

### 3f. `step()` — Add cube-held check and grasp_mode guard

In the crash detection section of `step()`, add after existing crash checks:

```python
# NEW: Cube drop check (only during grasp_mode in ascend/transit)
if self.grasp_mode and self.current_scenario in {"ascend", "transit"}:
    cube_held, drop_reason = self._check_cube_held(gripper_pos)
    if not cube_held:
        crashed = True
        crash_reason = drop_reason
```

**Finger state machine guard (required for future PLACE stage compatibility):**

The `step()` finger state machine currently sets `finger_target_joint = FINGER_OPEN_JOINT`
whenever `current_scenario == "descend"`. During the PICK stage this is correct
(descend always has `grasp_mode=False`). However, during the future PLACE stage,
`current_scenario == "descend"` with `grasp_mode=True` — the open-finger command
would apply 2,715N outward force and drop the cube.

Guard the finger state machine now to prevent this future bug:
```python
# In step() finger state machine — GUARD with grasp_mode check:
if self.current_scenario == "descend" and not self.grasp_mode:
    self.finger_target_joint = self.FINGER_OPEN_JOINT
# else: grasp_mode=True during descend (PLACE leg) — do NOT change finger target
```

This change is safe for the current PICK stage: during PICK descend, `grasp_mode`
is always False, so the guard condition is equivalent to the unguarded version.

---

## Step 4: Create `scripts/test_grasp_physics.py`

This script MUST be run and pass before `eval_grasp.py` is attempted.

```python
#!/usr/bin/env python3
"""
Physics verification for the grasp stage.
Run this after applying XML changes to verify the cube can be held.

Usage:
    python scripts/test_grasp_physics.py --n-trials 20 --visualize --debug
"""
import argparse
import numpy as np
import gymnasium as gym
import src.chess_env

def test_static_grasp(env, debug=False):
    """Test 1: Place arm at GRASP_Z, close fingers, check cube stability."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    obs, _ = env.reset()
    
    # Settle arm exactly at GRASP_Z above cube
    src_xy = uw.get_cube_position()[:2]
    target = np.array([src_xy[0], src_xy[1], uw.GRASP_Z])
    # ... use align_to_waypoint to position arm ...
    
    # Execute grasp
    result = uw.execute_grasp()
    
    cube_pos = uw.get_cube_position()
    cube_quat = uw.get_cube_quat()
    z_displacement = abs(cube_pos[2] - (uw.GRASP_Z - 0.015))
    
    return {
        "test": "static_grasp",
        "grasp_success": result["success"],
        "grasp_reason": result.get("reason"),
        "cube_z_displacement_mm": z_displacement * 1000,
        "close_steps": result.get("close_steps_used"),
    }

def test_lift(env, debug=False):
    """Test 2: Grasp cube and lift 120mm. Verify no drop."""
    # ... similar structure ...

def test_transit_held(env, debug=False):
    """Test 3: Transit 100mm while holding cube. Verify no drop, no rotation."""
    # ...

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    
    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode,
                   hide_object=False)
    
    results = {"static_grasp": [], "lift": [], "transit_held": []}
    
    for trial in range(args.n_trials):
        results["static_grasp"].append(test_static_grasp(env, args.debug))
        results["lift"].append(test_lift(env, args.debug))
        results["transit_held"].append(test_transit_held(env, args.debug))
    
    # Print summary: success rates, average errors, failure breakdown
    print_physics_test_summary(results)
    env.close()
```

**Pass criteria for test_grasp_physics.py:**
- Test 1 (static grasp): ≥ 80% success (empirical ceiling ~92%; ~8% explosions are expected)
- Test 2 (lift): ≥ 90% success (no drop during ascent)
- Test 3 (transit): ≥ 85% success (no drop, rotation < 15°)

If any test fails below its threshold, do NOT proceed to `eval_grasp.py`. Instead,
re-examine the physics parameters for the failing test.

**Note for `scripts/verify_physics.py` Gate 0:**
The kp assertion must check for 150000, not any other value. A common agent mistake
is writing `if kp != 500:` (wrong) instead of `if kp != 150000:` (correct). This
check will silently block Gate 0 forever if the wrong value is used.

---

## Step 5: Create `scripts/eval_grasp.py`

The full sequence evaluation: home → transit → descend → grasp → ascend → transit(home).

**Critical architecture note:** `ChainEpisodeRunner` (from `eval_sequence.py`) only
exposes `run_one_chain(waypoints: list[np.ndarray]) -> list`. It has no `run_scenario()`,
`transition()`, or `home_pos` property — those methods do not exist. `eval_grasp.py`
cannot use `ChainEpisodeRunner` because the GRASP step requires pausing mid-chain.
Instead, use **explicit `while not done:` step loops** with manual `soft_reset` calls.

```python
#!/usr/bin/env python3
"""
Full pick sequence evaluation: home → transit → descend → GRASP → ascend → transit(home).

IMPORTANT: This script does NOT use ChainEpisodeRunner. It uses explicit while-loops
because the GRASP step must be inserted between DESCEND and ASCEND.

Usage:
    python scripts/eval_grasp.py \
        --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
        --n-episodes 50 --drift-limit 0.010 --debug --visualize --wait
"""
import argparse
import numpy as np
import gymnasium as gym
import src.chess_env
from stable_baselines3 import SAC

SAFE_Z   = 0.550
GRASP_Z  = 0.425
TABLE_Z  = 0.400
CUBE_H   = 0.030


def reset_episode_timelimit(env):
    """Reset the TimeLimit wrapper step counter. No helper function exists — traverse inline."""
    curr = env
    while hasattr(curr, "env"):
        if hasattr(curr, "_elapsed_steps"):
            curr._elapsed_steps = 0
            break
        curr = curr.env


def run_scenario_loop(env, model, initial_obs, max_steps=500, debug=False, label="", delay=0.0):
    """
    Run one RL scenario with an explicit step loop.
    Returns: {"outcome": "success"|"crash"|"timeout", "steps": int, "obs": np.ndarray,
              "crash_reason": str|None}

    IMPORTANT: ChessTaskEnv.step() overrides the parent and never calls self.render().
    render_mode="human" creates the viewer but nothing animates without an explicit
    env.render() call after each step. This function handles that correctly.
    """
    obs = initial_obs
    done = False
    steps = 0
    outcome = "timeout"
    crash_reason = None
    visualize = (env.unwrapped.render_mode == "human")

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        # Render after every step — required because ChessTaskEnv.step() does not
        # call self.render() internally, so the viewer would freeze without this.
        if visualize:
            env.render()

        if delay > 0:
            time.sleep(delay)

        if debug:
            uw = env.unwrapped
            g = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
            print(f"  [{label} step {steps}] grip=({g[0]*1000:.1f}, {g[1]*1000:.1f}, {g[2]*1000:.1f})mm")

        if info.get("is_success"):
            outcome = "success"
            break
        if terminated and not info.get("is_success"):
            outcome = "crash"
            crash_reason = info.get("crash_reason", "UNKNOWN_CRASH")
            break

    return {"outcome": outcome, "steps": steps, "obs": obs, "crash_reason": crash_reason}


def run_one_pick_sequence(env, model, home_pos, debug=False, delay=0.0, drift_limit=0.010):
    """
    Runs the full pick sequence for one episode.
    Returns a dict with per-step results and grasp quality metrics.

    NOTE ON VISUALIZATION: execute_grasp() and soft_reset() call _mujoco_step() directly
    and do NOT call env.render() internally. The viewer will freeze during those phases.
    To animate them, add `if self.render_mode == "human": self.render()` inside the
    inner loops of execute_grasp() and soft_reset() in task.py (future improvement).
    """
    uw = env.unwrapped
    results = {}

    # Choose src_xy for this episode
    src_xy = uw._sample_board_position()[:2]

    # ── Step 1: Initialize ─────────────────────────────────────────────────────
    # NOTE: force_start_pos and force_cube_pos are consumed by _reset_sim and then
    # must be cleared so subsequent resets (if any) don't reuse stale overrides.
    uw.force_start_pos = home_pos.copy()
    uw.force_cube_pos  = np.array([src_xy[0], src_xy[1], TABLE_Z + CUBE_H / 2])
    uw.force_scenario  = "transit"
    uw.hide_object     = False
    obs, _ = env.reset()
    # Clear overrides immediately after reset so they don't persist into future episodes
    uw.force_start_pos = None
    uw.force_cube_pos  = None
    # Apply drift limit. Must be set after every reset (reset() reads force_drift_limit on
    # the first step, not at reset time). Without this, --drift-limit is silently ignored.
    uw.force_drift_limit = drift_limit

    if debug:
        print(f"\n[Episode] src_xy=({src_xy[0]*1000:.1f}, {src_xy[1]*1000:.1f})mm")

    # ── Step 2: Transit (home → src_xy at SAFE_Z) ─────────────────────────────
    transit_res = run_scenario_loop(env, model, obs, label="TRANSIT", delay=delay)
    results["transit_to_src"] = transit_res
    if transit_res["outcome"] != "success":
        return results

    # ── Transition: transit → descend ─────────────────────────────────────────
    # soft_reset() returns (obs, info) — must unpack as tuple, NOT single variable
    obs, trans_info = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], GRASP_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    # ── Step 3: Descend (SAFE_Z → GRASP_Z at src_xy) ─────────────────────────
    descend_res = run_scenario_loop(env, model, obs, label="DESCEND", delay=delay)
    results["descend"] = descend_res
    if descend_res["outcome"] != "success":
        return results

    # ── Step 4: GRASP (scripted — no soft_reset needed) ───────────────────────
    # execute_grasp() runs its own Phase 0 halt loop internally.
    # Returns a plain dict; use result["success"], not result.success.
    grasp_result = uw.execute_grasp()
    results["grasp"] = grasp_result
    if not grasp_result["success"]:
        if debug:
            print(f"  [GRASP FAILED] {grasp_result['reason']}")
        return results

    # ── Transition: GRASP → ascend (grasp_mode=True preserved) ───────────────
    obs, trans_info = uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], GRASP_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)
    # Sanity check: grasp_mode must survive soft_reset
    assert uw.grasp_mode, "grasp_mode was reset during soft_reset (Bug C not fixed)"

    # ── Step 5: Ascend (GRASP_Z → SAFE_Z, cube held) ─────────────────────────
    ascend_res = run_scenario_loop(env, model, obs, label="ASCEND", delay=delay)
    results["ascend"] = ascend_res
    if ascend_res["outcome"] != "success":
        return results

    # ── Transition: ascend → transit (toward home) ────────────────────────────
    obs, trans_info = uw.soft_reset(
        new_scenario="transit",
        new_goal_pos=np.array([home_pos[0], home_pos[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    # ── Step 6: Transit to Home (carrying cube) ───────────────────────────────
    transit_home_res = run_scenario_loop(env, model, obs, label="TRANSIT_HOME", delay=delay)
    results["transit_to_home"] = transit_home_res

    # ── Evaluate grasp quality ─────────────────────────────────────────────────
    if transit_home_res["outcome"] == "success":
        results["grasp_quality"] = evaluate_grasp_quality(uw, src_xy)

    return results


def evaluate_grasp_quality(uw, src_xy: np.ndarray) -> dict:
    """Measures cube position/orientation drift at the end of the sequence."""
    cube_pos  = uw.get_cube_position()
    grip_pos  = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    cube_quat = uw.get_cube_quat()

    xy_drift  = float(np.linalg.norm(cube_pos[:2] - src_xy)) * 1000
    # Cube hangs ~15mm below grip site while held in mid-air
    z_error   = float(abs(cube_pos[2] - (grip_pos[2] - 0.015))) * 1000

    import scipy.spatial.transform
    r = scipy.spatial.transform.Rotation.from_quat(
        [cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]]  # scipy: xyzw
    )
    euler_deg = r.as_euler("xyz", degrees=True)

    return {
        "cube_xy_drift_mm": xy_drift,
        "cube_z_error_mm": z_error,
        "cube_max_rotation_deg": float(np.max(np.abs(euler_deg))),
        "cube_euler_xyz_deg": euler_deg.tolist(),
    }


def print_summary(all_results: list):
    """Print the standardised EVAL_GRASP report."""
    n = len(all_results)
    keys = ["transit_to_src", "descend", "grasp", "ascend", "transit_to_home"]
    print(f"\n=== EVAL_GRASP RESULTS ({n} episodes) ===\n")
    print("Step success rates:")
    for k in keys:
        success = sum(
            1 for r in all_results
            if k in r and (r[k].get("outcome") == "success" or r[k].get("success"))
        )
        print(f"  {k:<22}: {success}/{n} ({100*success//n}%)")

    quality_episodes = [r["grasp_quality"] for r in all_results if "grasp_quality" in r]
    if quality_episodes:
        print(f"\nGrasp quality (successful episodes, n={len(quality_episodes)}):")
        for metric in ["cube_xy_drift_mm", "cube_z_error_mm", "cube_max_rotation_deg"]:
            vals = [q[metric] for q in quality_episodes]
            print(f"  {metric:<30}: mean={np.mean(vals):.1f}  max={np.max(vals):.1f}")
    print("\n=== DONE ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, help="Path to SAC model .zip")
    p.add_argument("--n-episodes",  type=int, default=50)
    p.add_argument("--drift-limit", type=float, default=0.010,
                   help="Must match env.yaml eval_drift_limit (default 0.010)")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--delay",       type=float, default=0.0,
                   help="Sleep seconds between steps (slow-motion visualization)")
    p.add_argument("--wait",        action="store_true",
                   help="Pause between episodes (press Enter to continue)")
    args = p.parse_args()

    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode, hide_object=False)
    model = SAC.load(args.model, env=env)

    uw = env.unwrapped
    # Verify GRASP_Z before any run
    assert abs(uw.GRASP_Z - 0.425) < 0.001, \
        f"GRASP_Z={uw.GRASP_Z}, expected 0.425. Restart process after env.yaml change."

    home_xy  = np.array(uw.env_cfg.get("home_position_xy", [0.680, 0.2641]))
    home_pos = np.array([home_xy[0], home_xy[1], SAFE_Z])

    all_results = []
    for ep in range(args.n_episodes):
        result = run_one_pick_sequence(env, model, home_pos, debug=args.debug,
                                       delay=args.delay, drift_limit=args.drift_limit)
        all_results.append(result)
        if args.wait:
            input(f"  Episode {ep+1}/{args.n_episodes} done. Press Enter...")

    print_summary(all_results)
    env.close()


if __name__ == "__main__":
    main()
```

**CLI flags:**
- `--model` (required): path to SAC model `.zip`
- `--n-episodes` (default: 50): number of pick sequences to run
- `--drift-limit` (default: 0.010): must match `env.yaml`'s `eval_drift_limit`
- `--debug`: verbose per-step logging including cube position
- `--visualize`: open MuJoCo viewer
- `--wait`: pause between episodes (keypress to continue)

---

## Execution Order

### Original Implementation (completed)
1. Apply XML changes (Step 1)
2. Run `test_grasp_physics.py --n-trials 20 --visualize` — verify physics
3. Implement `simulation.py` grasp_mode (Step 2)
4. Implement `task.py` changes (Step 3a–3f)
5. Run `test_grasp_physics.py --n-trials 100`
6. Implement `eval_grasp.py` (Step 5)
7. Run `eval_grasp.py --n-episodes 100` → **88% achieved**

### Phase 1: Scripted Improvements (no retraining) — target: ≥ 94%
Apply these changes to the existing implementation:

8. **Fix execute_grasp halt loop** (doc 10 P1.1 — HIGHEST PRIORITY):
   - `src/chess_env/task.py`: Move `qvel=0; qacc=0; mj_forward()` to BEFORE the 50-step loop.
   - Reduce loop from 50 to 15 steps (arm is already stationary).
   - Expected: ~5% proximity explosion failures eliminated.

9. **TUBE_BREACH grace buffer** (doc 10 P1.2):
   - `src/chess_env/task.py`, step(): `TUBE_BREACH_GRACE = 0.0015` → `0.0020`

10. **Cube rotation abort threshold** (doc 10 P1.3):
    - `execute_grasp()` pre-conditions: `max_rotation_deg > 35.0` → `> 25.0`

11. **Early FINGER_CLOSED_EMPTY abort** (doc 10 P1.4):
    - `execute_grasp()` finger close loop: add early abort at `l_finger < 0.003` after step 30

12. **Fix test_grasp_physics.py position bounds** (doc 10 P1.5):
    - Add 10-retry loop with board bounds check (X∈[0.64,1.12], Y∈[0.02,0.50])

13. Run `eval_grasp.py --n-episodes 100` → target ≥ 94%

### Phase 2: RL Retraining — target: ≥ 98%
14. Retrain descend model with `drift_limit_end=0.008` (warm start from best_model_combined.zip)
15. Run `eval_grasp.py --n-episodes 100` → target ≥ 98%
16. (Optional) Add cube XY to descend observation + XY alignment reward component

See **doc 10** for full Phase 1-3 roadmap.

---

---

## Scripts Folder Audit

**User Note 5: Current state of `scripts/` and what must change for cube integration.**

### `scripts/eval.py` — Single Scenario Evaluator

**Current state:** Evaluates one RL scenario (transit, descend, or ascend) over N
episodes. No cube, no grasp. Tracks success rate, step count, crash reasons.

**Required changes:**
1. `eval_drift_limit=0.005` in the CLI default — WRONG. Current training uses
   `drift_limit_end=0.010`. The 5mm default rejects episodes that would pass with
   the correct limit. **Change default to 0.010.**
2. No `hide_object` flag — the env defaults to True. For cube-enabled testing, add
   `--hide-object / --no-hide-object` flag.
3. No cube-position logging even when `hide_object=False`. Add optional cube tracking.

**Status: Needs 1 fix before release (drift_limit), 2 additions for cube use.**

### `scripts/eval_sequence.py` — Chain Evaluator

**Current state:** Runs the full transit→descend→ascend chain over N episodes.
No cube. No grasp. Uses `ChainEpisodeRunner` class internally.

**Required changes:**
1. No `hide_object` support — always uses no cube. Add flag for cube-enabled runs.
2. No GRASP step between descend and ascend — cannot be used for cube evaluation as-is.
3. No `force_start_pos` support — always randomizes start position.
4. No `force_cube_pos` support — cannot set cube at known src_xy.
5. `force_drift_limit` hardcoded at episode start — no validation that this matches
   `env.yaml`'s `eval_drift_limit`. **Add assertion: `assert force_drift_limit == uw.EVAL_DRIFT_LIMIT`.**

**Status: Cannot be used for cube sequences. eval_grasp.py (Step 5) replaces it
for cube evaluation. eval_sequence.py remains for pure-movement certification.**

### `scripts/verify_physics.py` — Physics Health Check

**Current state:** Checks XML integrity, static stability (no drift at rest), basic
kinematic reachability, and teleport verification. Does NOT test grasp physics.

**Required changes after XML modifications:**
1. Add assertion: `model.body_mass[cube_body_id] == 0.05` (was 0.5)
2. Add assertion: finger actuator Kp matches expected (`kp=150000`)
3. Add assertion: finger ctrlrange = [0, 0.05]
4. Add assertion: weld solref second component ≥ 0.02
5. Add warning if `condim` of cube geom ≠ 6 (verify from XML diff, MuJoCo API varies)
6. Add `grasp_z` value assertion: `uw.GRASP_Z == 0.425`

Add these checks to a new `verify_grasp_xml_changes()` function in verify_physics.py.

**Status: Must add post-XML assertions before running test_grasp_physics.py.**

### `scripts/test_corners.py` — Corner Stress Test

**Current state:** Runs 5 board-corner transits × N episodes. Tests the transit
model at extreme XY positions. No cube.

**Required changes:** None for this stage. The corner test validates transit only.
After cube integration, add a cube-carried corner transit test (future stage).

**Status: No changes needed.**

### `scripts/test_grasp_physics.py` — NEW (create)

**Required:** Physics verification for the grasp stage. See doc 09 for full spec.
Must pass before any `eval_grasp.py` run.

### `scripts/eval_grasp.py` — NEW (create)

**Required:** Full pick sequence evaluator. Implements the six-step home→transit→
descend→GRASP→ascend→transit chain. See Step 5 above for full spec.

---

## Training Validation Requirements

**User Note 6: Training and scripts must perform correct validations.**

### What the Current Training Does NOT Validate

1. **`eval_drift_limit` mismatch:** `train.py` trains with `drift_limit_end=0.010`
   but `eval.py` defaults to `eval_drift_limit=0.005`. An episode that passes
   training criteria will fail eval criteria. This creates false negatives during
   model certification.
   - **Fix:** `eval.py` default must match `env.yaml`'s `eval_drift_limit`.
   - **Fix:** `train.py` should assert `BRAKING_DIST == SUCCESS_THRESHOLD` (dead zone check)
     at startup and refuse to train if `BRAKING_DIST > SUCCESS_THRESHOLD`.

2. **No GRASP_Z validation at training time:** The descend scenario trains toward
   `GRASP_Z`. If `env.yaml` has the wrong value (e.g., old 0.430), the model trains
   toward the wrong height. The descend model will never generalize to GRASP_Z=0.425.
   - **Fix:** Add to `train.py` startup: `assert abs(cfg.GRASP_Z - 0.425) < 0.001`
     for descend/ascend training runs.

3. **No cube-held validation during ascend training:** The ascend model is trained
   without a cube. When run with a cube (grasp_mode=True), the arm behavior is
   identical (Phase 9 trick zeros finger state). But there is no test confirming the
   policy output is identical in both modes.
   - **Fix:** In `eval_grasp.py`, log arm action statistics during ascend-with-cube
     and compare to descend-only eval. Flag if policy outputs diverge.

4. **`test_grasp_physics.py` is not run as part of any CI pipeline.** Physics
   validation is a one-time manual check. Add to README: "After XML changes, run
   `python scripts/test_grasp_physics.py` before any training or evaluation."

### Certification Gate: Before `eval_grasp.py`

```
1. verify_physics.py: XML assertions pass
2. test_grasp_physics.py: Test 1 ≥ 95%, Test 2 ≥ 90%, Test 3 ≥ 85%
3. eval.py --model <checkpoint> --scenario ascend: ≥ 95% success
4. eval_sequence.py --model <checkpoint> --chain transit+descend+ascend: ≥ 85% success
5. eval_grasp.py --n-episodes 20 --debug --visualize: inspect manually
6. eval_grasp.py --n-episodes 100: ≥ 75% full success
```

Steps 1-4 are blocking gates. Steps 5-6 are reporting gates.

---

## What This Stage Does NOT Cover

- Placing the cube at a destination (PLACE sequence)
- Handling piece captures (removing an existing piece before placing)
- Integration with a chess engine (move selection)
- Real-hardware deployment
- Training a grasp-specific policy (the grasp is scripted in this stage)

---
*Next: [06 — Risk Analysis and Failure Modes](./06_risk_analysis.md)*
