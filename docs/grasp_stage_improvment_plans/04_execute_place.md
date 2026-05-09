# `execute_place()` — Place & Release Specification

The place operation is the mirror of the grasp. After the arm arrives at HOVER_Z over the destination square, `execute_place()` aligns over the destination, plunges, releases, and retracts. Uses the same `_move_mocap_to` "assert every step" paradigm as `execute_grasp`.

---

## Prerequisites

`execute_place()` is called after:
1. `[TRANSIT]` — arm has flown to destination XY at SAFE_Z
2. `[DESCEND]` — RL has lowered to HOVER_Z over destination XY
3. `grasp_mode = True`, `finger_target_joint = FINGER_CLOSED_JOINT (0.012)`, cube held

---

## Full Implementation

```python
def execute_place(self, dst_xy: np.ndarray) -> dict:
    """
    Scripted PLACE pipeline. Runs after DESCEND to HOVER_Z over destination.
    dst_xy: 2D destination XY (board square center).
    Returns dict with 'success' bool. Arm exits at HOVER_Z, cube placed, grasp_mode=False.
    """
    zero_action = np.zeros(4)
    result = {
        "success": False,
        "reason": None,
        "final_cube_pos": None,
        "final_xy_error_mm": 0.0,
    }

    # ── Phase 0: Halt & Settle ────────────────────────────────────────────────
    # CRITICAL: Zero the full simulation state ([:]), not just robot DOFs [:15].
    # During execute_place, the arm is airborne holding the cube. DOFs 15-20 are
    # the cube's free joint. If the cube retains transit momentum and only the
    # robot is frozen, the cube jerks violently inside the closed fingers,
    # risking a physics solver spike or drop.
    self.data.qvel[:] = 0.0
    self.data.qacc[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

    # Lock to CURRENT quat, not VERTICAL_QUAT. The arm arrived from RL DESCEND
    # and may be tilted. Forcing VERTICAL_QUAT instantly while holding the cube
    # causes a violent physics snap, dropping the piece. Vertical correction
    # happens gradually in Phase 1+2 via _move_mocap_to.
    settle_pos  = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    settle_quat = self.data.mocap_quat[0].copy()
    for _ in range(15):
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = settle_pos
        self.data.mocap_quat[0][:] = settle_quat
        self._mujoco_step(None)

    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
    if float(np.linalg.norm(grip_vel)) > 0.005:
        result["reason"] = "PRECONDITION_SPEED"
        return result

    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - self.HOVER_Z) > 0.015:
        result["reason"] = f"PRECONDITION_Z (grip={grip_pos[2]*1000:.1f}mm)"
        return result

    # ── Phase 1+2: Vertical Correction + XY Align Over Destination ───────────
    # Simultaneously correct wrist orientation and move over destination.
    align_target = np.array([dst_xy[0], dst_xy[1], self.HOVER_Z])
    if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
        result["reason"] = "ROTATION_FAILED (kinematic limit)"
        return result

    # ── Phase 3: Plunge (HOVER_Z → PLACE_Z) ──────────────────────────────────
    # Use GRASP_Z as place Z (grip site at same height as during the original pick).
    # Re-assert XY and VERTICAL_QUAT every step.
    place_z = self.GRASP_Z
    plunge_target = np.array([dst_xy[0], dst_xy[1], self.HOVER_Z])

    for _ in range(int(round((self.HOVER_Z - place_z) / 0.001)) + 15):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if grip_pos[2] <= place_z + 0.001:
            break
        plunge_target[2] = max(place_z, plunge_target[2] - 0.001)
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = plunge_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

    # Verify plunge reached place_z before releasing the cube. If the arm stalled
    # mid-descent and we open fingers here, the cube falls from height.
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    if abs(grip_pos[2] - place_z) > 0.008:
        result["reason"] = (f"PLUNGE_FAILED (z={grip_pos[2]*1000:.1f}mm, "
                            f"target={place_z*1000:.1f}mm)")
        return result

    # ── Phase 4: Release (Linear Ramp Open) ───────────────────────────────────
    # Ramp from CLOSED (0.012) to OPEN (0.0181) over 80 steps.
    # Slow release prevents the sudden opening force from launching the cube sideways.
    release_target = plunge_target.copy()
    ramp_start = self.FINGER_CLOSED_JOINT  # 0.012
    ramp_end   = self.FINGER_OPEN_JOINT    # 0.0181
    ramp_delta = (ramp_end - ramp_start) / 80

    for step in range(80):
        self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = release_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

    # Full open settle (30 steps)
    self.finger_target_joint = self.FINGER_OPEN_JOINT
    for _ in range(30):
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = release_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

    # Disable grasp mode — fingers can be teleported again in subsequent RL phases
    self.grasp_mode = False

    # ── Phase 5: Verify Placement ─────────────────────────────────────────────
    cube_pos = self.get_cube_position()
    xy_error = float(np.linalg.norm(cube_pos[:2] - dst_xy[:2])) * 1000
    z_error  = float(abs(cube_pos[2] - (self.TABLE_Z + self.CUBE_HEIGHT / 2.0))) * 1000

    result["final_cube_pos"] = cube_pos.copy()
    result["final_xy_error_mm"] = xy_error

    if xy_error > 20.0:
        result["reason"] = f"PLACE_XY_FAILED ({xy_error:.1f}mm drift from target)"
        return result

    if z_error > 10.0:
        result["reason"] = f"PLACE_Z_FAILED ({z_error:.1f}mm — cube not flat on table)"
        return result

    # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ──────────────────────────────────
    retract_target = release_target.copy()
    for _ in range(int(round((self.HOVER_Z - place_z) / 0.001)) + 15):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        if grip_pos[2] >= self.HOVER_Z - 0.001:
            break
        retract_target[2] = min(self.HOVER_Z, retract_target[2] + 0.001)
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] = retract_target
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)

    result["success"] = True
    return result
```

---

## State Transition After `execute_place`

After `execute_place()` returns success:
- `grasp_mode = False`
- `finger_target_joint = FINGER_OPEN_JOINT` (fingers open)
- Arm is at HOVER_Z over destination XY

Call `soft_reset("ascend", np.array([dst_xy[0], dst_xy[1], env.SAFE_Z]), nominal_xy=dst_xy)` to hand off to the RL ASCEND policy.

---

## Full Pick-and-Place Sequence

```python
# Step 1: RL Transit home → over source
obs, _ = env.reset(options={"scenario": "transit", "target_xy": src_xy})
done = False
while not done:
    action, _ = model_transit.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Step 2: Descend over source
obs, _ = env.soft_reset("descend", np.array([src_xy[0], src_xy[1], env.HOVER_Z]),
                         nominal_xy=src_xy)
done = False
while not done:
    action, _ = model_descend.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Step 3: Scripted Grasp
result = env.execute_grasp()
assert result["success"], result["reason"]

# Step 4: Ascend from source
obs, _ = env.soft_reset("ascend", np.array([src_xy[0], src_xy[1], env.SAFE_Z]),
                         nominal_xy=src_xy)
done = False
while not done:
    action, _ = model_ascend.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Step 5: Transit over destination
obs, _ = env.soft_reset("transit", np.array([dst_xy[0], dst_xy[1], env.SAFE_Z]))
done = False
while not done:
    action, _ = model_transit.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Step 6: Descend over destination
obs, _ = env.soft_reset("descend", np.array([dst_xy[0], dst_xy[1], env.HOVER_Z]),
                         nominal_xy=dst_xy)
done = False
while not done:
    action, _ = model_descend.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Step 7: Scripted Place
result = env.execute_place(dst_xy)
assert result["success"], result["reason"]

# Step 8: Ascend from destination
obs, _ = env.soft_reset("ascend", np.array([dst_xy[0], dst_xy[1], env.SAFE_Z]),
                         nominal_xy=dst_xy)
done = False
while not done:
    action, _ = model_ascend.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Move complete. Cube is at dst_xy.
```
