# `execute_grasp()` Complete Rewrite

**File:** `src/chess_env/task.py`  
**Scope:** (1) New `_move_mocap_to()` helper method, (2) Full rewrite of `execute_grasp()`.

---

## Critical Pre-condition: The Mocap Reset Loophole

`_set_action(action)` internally calls `gymnasium_robotics.utils.mujoco_utils.mocap_set_action`, which immediately begins with:
```python
reset_mocap2body_xpos(model, data)   # ← teleports mocap back to current body position AND rotation
data.mocap_pos[:] = data.mocap_pos + pos_delta
data.mocap_quat[:] = data.mocap_quat + quat_delta
```

With `zero_action`, both deltas are zero. Therefore `_set_action(zero_action)` = "reset mocap to wherever the physical body currently is, both position AND quaternion."

**Consequence:** Any `set_mocap_quat(VERTICAL_QUAT)` call is immediately erased the next time `_set_action` is called. Any direct `mocap_pos`/`mocap_quat` write is overwritten on the next `_set_action` call.

**The Solution — Assert Every Step:** After every `_set_action(zero_action)` call, immediately overwrite BOTH `mocap_pos` AND `mocap_quat` with the desired absolute targets before calling `_mujoco_step`. The physics engine then drives the body toward these locked targets. This must be done every single step of every loop.

---

## Part A: Add `_move_mocap_to()` Helper

Add this method to `ChessTaskEnv` (place before `execute_grasp`):

```python
def _move_mocap_to(self, target_pos: np.ndarray, target_quat: np.ndarray,
                   max_steps: int = 80, tolerance: float = 0.001) -> bool:
    """
    Drives the physical arm to target_pos/target_quat by asserting the target
    on EVERY step. Required because _set_action resets mocap to body state
    (both position and rotation) before applying deltas. Without re-asserting
    every step, set_mocap_quat calls are silently overwritten.

    Returns True if the target was reached within tolerance, False if max_steps
    was exhausted (kinematic limit or obstacle). Callers must check the return
    value and abort if False.
    """
    zero_action = np.zeros(4)
    for _ in range(max_steps):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if np.linalg.norm(grip_pos - target_pos) < tolerance:
            return True
        self._set_action(zero_action)               # 1. Resets mocap → body pos+quat
        self.data.mocap_pos[0][:3] = target_pos     # 2. OVERRIDE position (absolute)
        self.data.mocap_quat[0][:] = target_quat    # 3. OVERRIDE rotation (absolute)
        self._mujoco_step(None)                     # 4. Physics drives body to target
    # Max steps exhausted — check final distance
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    return bool(np.linalg.norm(grip_pos - target_pos) < tolerance)
```

> **Note on tolerance:** 1mm tolerance is safe. 0.5mm risks a perpetual loop at kinematic limits (the arm physically cannot reach certain positions exactly). Do not lower this value.

---

## Part B: Rewrite `execute_grasp()`

Replace the current `execute_grasp()` method entirely.

### Phase Overview

```
Phase 0  — Halt & Settle:      Zero velocity, 15 settle steps (re-assert pos+quat each step)
Phase 1+2 — Align & Verticalize: Move to [cube_xy, HOVER_Z] + VERTICAL_QUAT simultaneously
Phase 3  — Plunge:             HOVER_Z → GRASP_Z at 1mm/step (re-assert xy+quat every step)
Phase 4  — Grasp:              Linear ramp close, 150 steps (re-assert pos+quat every step)
Phase 5  — Hold & Verify:      50 settle steps + cube proximity check
Phase 6  — Retract:            GRASP_Z → HOVER_Z at 1mm/step (re-assert xy+quat every step)
```

### Full Implementation

```python
def execute_grasp(self) -> dict:
    """
    Scripted GRASP pipeline. Runs after DESCEND succeeds at HOVER_Z.
    Returns dict with 'success' bool. Arm exits at HOVER_Z with cube held.

    KEY INVARIANT: After every _set_action call, re-assert BOTH mocap_pos AND
    mocap_quat before _mujoco_step. _set_action resets mocap to physical body
    state (via reset_mocap2body_xpos), so assertions must be repeated every step.
    """
    import math

    zero_action = np.zeros(4)
    result = {
        "success": False,
        "reason": None,
        "pre_grasp_cube_xy": None,
        "post_grasp_cube_pos": None,
        "final_xy_error_mm": 0.0,
        "final_z_error_mm": 0.0,
        "final_finger_pos": 0.0,
        "close_steps_used": 0,
    }

    # ── Phase 0: Halt & Settle ────────────────────────────────────────────────
    # Zero velocity FIRST. Residual RL momentum causes oscillation if not stopped.
    # Zero the full state ([:]) not just the robot DOFs [:15]. DOFs 15-20 are the
    # cube's free joint. In execute_grasp the cube is on the table (harmless), but
    # the full zero is consistent and safe.
    self.data.qvel[:] = 0.0
    self.data.qacc[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

    settle_pos  = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    settle_quat = self.data.mocap_quat[0].copy()
    for _ in range(15):
        self._set_action(zero_action)                    # resets mocap → body
        self.data.mocap_pos[0][:3] = settle_pos          # re-lock position
        self.data.mocap_quat[0][:] = settle_quat         # re-lock rotation
        self._mujoco_step(None)

    # Speed check: arm should be stationary
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
    if float(np.linalg.norm(grip_vel)) > 0.005:
        result["reason"] = "PRECONDITION_SPEED"
        return result

    # Z check: RL must have stopped at HOVER_Z (within ±15mm)
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - self.HOVER_Z) > 0.015:
        result["reason"] = (f"PRECONDITION_Z (grip={grip_pos[2]*1000:.1f}mm, "
                            f"HOVER_Z={self.HOVER_Z*1000:.1f}mm)")
        return result

    # Fingers must be open
    l_finger = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()
    if l_finger < self.FINGER_OPEN_JOINT - 0.003:
        result["reason"] = f"PRECONDITION_FINGERS_NOT_OPEN (j={l_finger:.4f})"
        return result

    # ── Phase 1+2: Rotation Abort Check + Perfect Align & Verticalize ────────
    # Read cube position and check for dangerous diagonal orientation.
    cube_pos = self.get_cube_position()
    result["pre_grasp_cube_xy"] = cube_pos[:2].copy()

    # Cube rotation check: diagonal cube (yaw >25°) has effective width 42.4mm,
    # exceeding max finger opening (38mm). Abort before plunge to prevent stub.
    cube_quat = self.get_cube_quat()  # (w, x, y, z)
    w, x, y, z = cube_quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = abs(math.atan2(siny_cosp, cosy_cosp))  # → [0, π]
    # Correct 4-fold symmetry: fold [0,π] into [0,π/2] quadrant, then find
    # distance to nearest square axis. A cube at 175° is equivalent to 5°.
    # The simple formula `min(yaw, abs(yaw - π/2))` FAILS above 90° —
    # e.g. 175° gives 85° (false abort) instead of the correct 5°.
    yaw_modulo   = yaw % (math.pi / 2)                        # fold → [0, π/2)
    effective_yaw = min(yaw_modulo, (math.pi / 2) - yaw_modulo)  # dist to nearest axis
    if effective_yaw > 0.436:  # > 25 degrees
        result["reason"] = f"CUBE_ROTATED (yaw={math.degrees(effective_yaw):.1f}°)"
        return result

    # Drive to [cube_xy, HOVER_Z] with VERTICAL_QUAT in one combined move.
    # _move_mocap_to asserts both position and quat every step, so vertical
    # correction and XY alignment happen simultaneously without either drifting.
    align_target = np.array([cube_pos[0], cube_pos[1], self.HOVER_Z])
    if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
        result["reason"] = "ROTATION_FAILED (kinematic limit — arm cannot reach vertical at this position)"
        return result

    # ── Phase 3: Plunge (HOVER_Z → GRASP_Z) ──────────────────────────────────
    # The arm is perfectly vertical and centered over the cube.
    # Lower 1mm/step while re-asserting XY and VERTICAL_QUAT every step.
    # This makes the arm act as a rigid vertical piston — no XY drift, no tilt.
    plunge_target = np.array([cube_pos[0], cube_pos[1], self.HOVER_Z])

    for _ in range(int(round((self.HOVER_Z - self.GRASP_Z) / 0.001)) + 15):
        grip_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if grip_now[2] <= self.GRASP_Z + 0.001:
            break
        plunge_target[2] = max(self.GRASP_Z, plunge_target[2] - 0.001)
        self._set_action(zero_action)                       # 1. reset mocap → body
        self.data.mocap_pos[0][:3] = plunge_target          # 2. set XY + new Z
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT    # 3. re-lock vertical
        self._mujoco_step(None)                             # 4. step physics

    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - self.GRASP_Z) > 0.008:
        result["reason"] = (f"PLUNGE_FAILED (z={grip_pos[2]*1000:.1f}mm, "
                            f"target={self.GRASP_Z*1000:.1f}mm)")
        return result

    # ── Phase 4: Grasp (Finger Close, Linear Ramp) ───────────────────────────
    # Ramp finger target from OPEN (0.0181) to CLOSED (0.012) over 150 steps.
    # Direct jump → 2000N impulse → physics explosion. Ramp → ~300N stable hold.
    # Re-assert plunge_target + VERTICAL_QUAT every step to prevent gravity sag.
    self.grasp_mode = True
    close_target  = plunge_target.copy()  # already at [cube_xy, GRASP_Z]
    ramp_start    = self.FINGER_OPEN_JOINT    # 0.0181
    ramp_end      = self.FINGER_CLOSED_JOINT  # 0.012
    ramp_delta    = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS

    steps_used = 0
    for step in range(self.GRASP_CLOSE_STEPS):
        self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
        self._set_action(zero_action)                       # 1. reset mocap → body
        self.data.mocap_pos[0][:3] = close_target          # 2. re-lock XY + Z
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT    # 3. re-lock vertical
        self._mujoco_step(None)                             # 4. step physics
        steps_used += 1

        # Explosion guard
        if step % 10 == 0:
            cube_now = self.get_cube_position()
            if cube_now[2] > self.GRASP_Z + 0.050:
                result["reason"] = f"CUBE_EXPLOSION (z={cube_now[2]*1000:.1f}mm)"
                return result

        # Early abort: fingers fully closed after 30 steps = no cube contact.
        # With cube: fingers stall at j≈0.0141, never drop below 0.003.
        if step > 30:
            l_now = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if l_now < 0.003:
                result["reason"] = f"FINGER_CLOSED_EMPTY (j={l_now:.4f} at step {step})"
                result["close_steps_used"] = steps_used
                return result

    result["close_steps_used"] = steps_used

    # ── Phase 5: Hold & Verify ────────────────────────────────────────────────
    # 50 settle steps to let contact impulses stabilize (prevent ringing).
    # Continue re-asserting position and quat to hold the arm perfectly still.
    for _ in range(50):
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = close_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

    cube_pos  = self.get_cube_position()
    grip_pos  = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    l_finger  = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()

    xy_error = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
    z_error  = float(abs(cube_pos[2] - grip_pos[2])) * 1000

    result["post_grasp_cube_pos"]  = cube_pos.copy()
    result["final_xy_error_mm"]    = xy_error
    result["final_z_error_mm"]     = z_error
    result["final_finger_pos"]     = l_finger

    if xy_error > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
        result["reason"] = (f"VERIFY_XY_FAILED ({xy_error:.1f}mm > "
                            f"{self.GRASP_VERIFY_XY_THRESHOLD*1000:.0f}mm)")
        return result

    if z_error > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
        result["reason"] = (f"VERIFY_Z_FAILED ({z_error:.1f}mm > "
                            f"{self.GRASP_VERIFY_Z_THRESHOLD*1000:.0f}mm)")
        return result

    if l_finger < 0.003:
        result["reason"] = f"VERIFY_FINGERS_CLOSED_EMPTY (j={l_finger:.4f})"
        return result

    # ── Phase 6: Retract (GRASP_Z → HOVER_Z) ─────────────────────────────────
    # Lift 1mm/step while re-asserting XY and VERTICAL_QUAT. The smooth upward
    # acceleration eliminates the "wrist snap" shock from suddenly yanking the
    # cube upward (previously caused drift/destabilization with soft weld).
    retract_target = close_target.copy()  # [cube_xy, GRASP_Z]

    for _ in range(int(round((self.HOVER_Z - self.GRASP_Z) / 0.001)) + 15):
        grip_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if grip_now[2] >= self.HOVER_Z - 0.001:
            break
        retract_target[2] = min(self.HOVER_Z, retract_target[2] + 0.001)
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = retract_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

        # Cube drop check: use RELATIVE distance between grip and cube (not absolute altitude).
        # The cube starts at TABLE_Z + CUBE_H/2 = 0.415m (on the table). An absolute
        # threshold of TABLE_Z + CUBE_H/2 + 0.005 = 0.420m would always fire on step 1
        # because 0.415 < 0.420. Instead, verify the cube travels with the arm:
        # when held, the cube CoM hangs ~15mm below the grip site.
        cube_now = self.get_cube_position()
        grip_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if abs(cube_now[2] - (grip_now[2] - 0.015)) > self.CUBE_HELD_Z_LIMIT:
            result["reason"] = "CUBE_DROPPED_DURING_RETRACT"
            return result

    result["success"] = True
    result["post_grasp_cube_pos"] = self.get_cube_position().copy()
    return result
```

---

## Additional `__init__` Change

In `ChessTaskEnv.__init__`, add after the existing grasp threshold block:

```python
self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)
```

---

## Key Invariants This Rewrite Preserves

1. **`grasp_mode` lifecycle:** Set `True` in Phase 4, persists through `soft_reset` into ASCEND/TRANSIT.
2. **`finger_target_joint` lifecycle:** After Phase 4, set to `FINGER_CLOSED_JOINT (0.012)`, persists through ASCEND and TRANSIT.
3. **`grasp_mode = False` on reset:** Already in `_reset_sim`.
4. **Render visibility:** Add `self.render()` inside phase loops when `render_mode="human"`.
