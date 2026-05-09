# GRASP Scenario: Implementation Specification

## Overview

The GRASP scenario is a new **scripted phase** (not RL-driven) inserted between
DESCEND and ASCEND. It executes after the arm has settled at GRASP_Z directly above
the cube, and before the arm lifts the cube upward.

The GRASP phase consists of four sub-phases:
1. **Contact approach** — lower the arm the final 2mm to ensure the fingers are at
   cube height, not floating above.
2. **Finger close** — actuator-driven close with contact physics enabled.
3. **Grasp verification** — confirm the cube is inside the gripper.
4. **Hold settle** — allow the physics to stabilize before the ascent begins.

---

## Preconditions (Invariants Before GRASP Begins)

The GRASP phase may only begin if ALL of the following hold:

```
1. self.current_scenario == "descend" just completed with is_success = True
2. HALT LOOP has run (50 zero-action steps inside execute_grasp) — arm is stationary
3. arm velocity < 5mm/s after halt loop — confirmed by explicit speed check
4. grip site Z is within 10mm of GRASP_Z (same as DESCEND success tolerance)
5. cube XY is within 10mm of grip site XY (arm is above the cube)
6. fingers are OPEN (finger_target_joint == FINGER_OPEN_JOINT)
```

**Critical:** `HALT_VEL_THRESHOLD = 0.0005m/s = 0.5mm/s`. The DESCEND scenario can
succeed at speeds up to 50mm/s (STABILITY_VEL_THRESHOLD). Do NOT check speed against
HALT_VEL_THRESHOLD at the start of execute_grasp — the arm will never pass this
check on the first call. Instead, run a 50-step halt loop FIRST to dissipate velocity,
then check that speed has dropped below an operational threshold of 5mm/s.

Any precondition failure aborts the GRASP and records the failure reason.
It does NOT crash the chain — it records `GRASP_PRECONDITION_FAILED` and
returns so the caller can log and skip the episode.

---

## Sub-Phase 1: Contact Approach (Fine-Descent)

The arm may have stopped 3–9mm above the exact GRASP_Z due to success-threshold
tolerance. A short scripted settle closes this gap. This uses the same `align_to_waypoint`
logic from `soft_reset`, but targets only the Z axis.

```python
CONTACT_APPROACH_TARGET_Z = GRASP_Z  # 0.425m — selected GRASP_Z
CONTACT_APPROACH_TOLERANCE = 0.001   # 1mm — tighter than ascend settle

# Scripted settle toward GRASP_Z
grip_pos = get_site_xpos("robot0:grip")
if abs(grip_pos[2] - CONTACT_APPROACH_TARGET_Z) > CONTACT_APPROACH_TOLERANCE:
    target = np.array([grip_pos[0], grip_pos[1], CONTACT_APPROACH_TARGET_Z])
    align_result = align_to_waypoint(target, tolerance=CONTACT_APPROACH_TOLERANCE,
                                     max_steps=50, gain=0.5)
    # Note: lower gain (0.5 vs 0.8) for this final approach to avoid overshoot
    if not align_result["converged"]:
        return GraspResult(success=False, reason="CONTACT_APPROACH_FAILED")
```

**Why necessary:** The descend success threshold is 10mm. The arm may be anywhere
from GRASP_Z to GRASP_Z-10mm at success. If it's 9mm above, the fingers will attempt
to close on air above the cube. The contact approach guarantees the fingers are at
exactly the cube's vertical center.

---

## Sub-Phase 2: Finger Close (Actuator-Driven)

This is the most critical sub-phase. The fingers must be driven by the actuator
(Kp controller), NOT by direct position teleportation, so MuJoCo can compute
contact forces.

```python
GRASP_CLOSE_STEPS = 150       # Steps for finger close (at 20ms/step = 3 seconds scripted)
GRASP_CLOSE_TARGET = FINGER_CLOSED_JOINT   # 0.0 — full close

# Switch to grasp mode: actuator-driven, not teleported
self.grasp_mode = True
self.finger_target_joint = FINGER_CLOSED_JOINT

# Zero arm velocity (hold position) while fingers close
zero_arm_action = np.zeros(4)   # [dx=0, dy=0, dz=0, gripper_ignored]

for step in range(GRASP_CLOSE_STEPS):
    # _set_action in grasp_mode: sets ctrl target only, does NOT teleport qpos
    self._set_action(zero_arm_action)
    self._mujoco_step(None)
    
    # Monitor finger progress every 10 steps
    if step % 10 == 0:
        l_pos = get_joint_qpos("robot0:l_gripper_finger_joint")
        cube_pos = get_cube_position()
        grip_pos = get_site_xpos("robot0:grip")
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
        
        # Early exit: detect cube explosion (Z sudden jump)
        if cube_pos[2] > GRASP_Z + 0.050:
            return GraspResult(success=False, reason="CUBE_EXPLOSION_DURING_CLOSE")
        
        # Early exit: fingers stopped moving (stalled against cube)
        # NOTE: with cube present, fingers physically stop at j≈0.014 (gap≈30mm),
        # so l_pos will NEVER reach FINGER_CLOSED_JOINT=0.0. Check for velocity=0
        # instead, or simply run all GRASP_CLOSE_STEPS to ensure full settle.
        # The 150-step budget handles this — no early exit needed.
```

**Why 150 steps?** The actuator Kp=150,000 drives the finger at rate proportional to
error. At 0.0181 (full open), the finger must travel 18.1mm. At max force, this happens
in ~20–50 steps, but we give 150 steps for the physics to fully settle, especially
when the cube resistance slows convergence.

**Why NOT teleport?** If we teleport the finger to 0.0 while the cube is in the way,
MuJoCo will see an interpenetration and generate a huge repulsive force, popping the
cube out. Actuator-driven control allows the finger to press against the cube and
generate a stable contact force.

---

## Sub-Phase 3: Grasp Verification

After the finger close loop, verify the cube is actually in the gripper:

```python
GRASP_VERIFY_XY_THRESHOLD = 0.015    # 15mm — cube must be within 15mm of grip site XY
GRASP_VERIFY_Z_THRESHOLD  = 0.020    # 20mm — cube must be within 20mm of grip site Z
GRASP_VERIFY_FINGER_THRESHOLD = 0.016 # Fingers stalled against cube at j≈0.014; must be ≤ 0.016

cube_pos = get_cube_position()
grip_pos = get_site_xpos("robot0:grip")
l_finger = get_joint_qpos("robot0:l_gripper_finger_joint")

xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
z_error  = abs(cube_pos[2] - grip_pos[2])

if xy_error > GRASP_VERIFY_XY_THRESHOLD:
    return GraspResult(success=False, reason=f"CUBE_XY_TOO_FAR ({xy_error*1000:.1f}mm)")

if z_error > GRASP_VERIFY_Z_THRESHOLD:
    return GraspResult(success=False, reason=f"CUBE_Z_TOO_FAR ({z_error*1000:.1f}mm)")

if l_finger > GRASP_VERIFY_FINGER_THRESHOLD:
    # Fingers above 0.016 means they haven't reached the cube — grasp failed
    return GraspResult(success=False, reason=f"FINGERS_NOT_CLOSED ({l_finger:.4f})")

# GRASP VERIFIED
log.info(f"[GRASP] Verified. xy_err={xy_error*1000:.1f}mm z_err={z_error*1000:.1f}mm "
         f"finger={l_finger:.4f}")
```

---

## Sub-Phase 4: Hold Settle

Allow the physics to stabilize for 50 steps while holding the cube:

```python
GRASP_HOLD_STEPS = 50

for _ in range(GRASP_HOLD_STEPS):
    self._set_action(zero_arm_action)  # still in grasp_mode
    self._mujoco_step(None)

# Final cube position check after settle
cube_pos_final = get_cube_position()
grip_pos_final = get_site_xpos("robot0:grip")
final_xy_error = np.linalg.norm(cube_pos_final[:2] - grip_pos_final[:2])

if final_xy_error > GRASP_VERIFY_XY_THRESHOLD:
    return GraspResult(success=False, reason=f"CUBE_DRIFTED_AFTER_HOLD ({final_xy_error*1000:.1f}mm)")
```

---

## GraspResult Data Structure

`execute_grasp()` returns a plain Python **dict** (not a dataclass). All call sites
must use dict access (`result["success"]`, not `result.success`).

```python
result = {
    "success": bool,
    "reason": str | None,               # Failure reason code, or None on success
    "pre_grasp_cube_xy": np.ndarray,    # Cube XY before finger close
    "post_grasp_cube_pos": list,        # Cube 3D position after hold settle
    "post_grasp_cube_quat": list,       # Cube orientation (w, x, y, z)
    "final_xy_error_mm": float,         # Cube-to-grip XY error at verification
    "final_z_error_mm": float,          # Cube-to-grip Z error at verification
    "final_finger_pos": float,          # l_finger joint qpos at verification
    "close_steps_used": int,            # Steps run in the close loop
}
```

---

## Home Position Definition

The **Home Position** is a fixed 3D coordinate above the edge of the board, where the
arm parks between moves. It is not a board square — it is a safe zone outside the
play area.

```python
# In configs/env.yaml:
home_position_xy: [0.680, 0.2641]   # within transit training range [0.640, 1.120]
# home Z is always SAFE_Z = 0.550
```

The Home Position is at `(0.680, 0.264, 0.550)` — at the left edge of the transit
training range (low_x = 0.640). X=0.600 is 40mm OUTSIDE the training range and must
NOT be used — the transit policy was never trained to navigate there. See doc 04 for
full rationale.

**Home Position transit:** Uses the existing TRANSIT scenario with `force_scenario="transit"`.
The model already handles any-to-any transit at SAFE_Z.

---

## `task.py` Changes

### New method: `execute_grasp()`

```python
def execute_grasp(self) -> GraspResult:
    """
    Scripted GRASP phase. Must be called when the arm has just completed DESCEND.
    
    Preconditions:
        - Arm is stationary at approximately GRASP_Z
        - Fingers are open (FINGER_OPEN_JOINT)
        - Cube is within 10mm of grip site XY
    
    Returns:
        GraspResult with success=True if cube is securely held, False otherwise.
    """
```

### New method: `get_cube_position()` → `np.ndarray`

```python
def get_cube_position(self) -> np.ndarray:
    """Returns the current 3D position of the cube (object0 body)."""
    obj_joint_id = self.model.joint("object0:joint").id
    qpos_start = self.model.jnt_qposadr[obj_joint_id]
    return self.data.qpos[qpos_start : qpos_start + 3].copy()
```

### New method: `get_cube_quat()` → `np.ndarray`

```python
def get_cube_quat(self) -> np.ndarray:
    """Returns the quaternion orientation of the cube (w, x, y, z)."""
    obj_joint_id = self.model.joint("object0:joint").id
    qpos_start = self.model.jnt_qposadr[obj_joint_id]
    return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()
```

### Modified: `__init__` — new flags

```python
self.grasp_mode = False   # When True: actuator-driven fingers (contact physics active)
```

### Modified: `_set_action` in `simulation.py`

```python
def _set_action(self, action):
    # ... existing pos_ctrl code ...
    
    target_qpos = getattr(self, "finger_target_joint", 0.0)
    
    if getattr(self, 'grasp_mode', False):
        # Actuator-driven: let MuJoCo Kp controller handle fingers
        # Contact forces are computed normally — cube can push back
        self.data.ctrl[0] = target_qpos
        self.data.ctrl[1] = target_qpos
        # NO set_joint_qpos, NO set_joint_qvel
    else:
        # Teleport mode: absolute enforcement (used during no-cube phases)
        self.data.ctrl[0] = target_qpos
        self.data.ctrl[1] = target_qpos
        self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", target_qpos)
        self._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", target_qpos)
        self._utils.set_joint_qvel(model, data, "robot0:l_gripper_finger_joint", 0.0)
        self._utils.set_joint_qvel(model, data, "robot0:r_gripper_finger_joint", 0.0)
```

### Modified: `_reset_sim`

On every reset, reset grasp_mode:
```python
self.grasp_mode = False  # Reset to teleport mode on every reset
```

---

## Cube Drift Monitoring During ASCEND and TRANSIT

Once the cube is grasped, every call to `step()` should check that the cube has not
been dropped. This is done via a helper called at the end of `step()` when
`grasp_mode=True`:

```python
def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
    """
    Returns (is_held, reason) — checks that the cube is still within acceptable 
    distance of the grip site during the held-cube phase.
    """
    cube_pos = self.get_cube_position()
    xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
    z_error  = abs(cube_pos[2] - (grip_pos[2] - 0.015))  # cube CoM should be ~15mm below grip
    
    if xy_error > CUBE_HELD_XY_LIMIT:  # 0.030m = 30mm
        return False, f"CUBE_DROPPED_XY ({xy_error*1000:.1f}mm)"
    if z_error > CUBE_HELD_Z_LIMIT:    # 0.020m = 20mm (40mm is too loose: allows 45mm ascent before detection)
        return False, f"CUBE_DROPPED_Z ({z_error*1000:.1f}mm)"
    return True, None
```

This check runs only when `self.grasp_mode=True` and `self.current_scenario` is
`ascend` or `transit`. It adds a new crash type `CUBE_DROPPED` that aborts the
chain immediately.

---

## Contact Approach Compatibility Analysis

**User Note 4: Does the arm position after DESCEND allow contact approach to work?**

After DESCEND succeeds, the arm can be anywhere within the success volume: within 10mm
of the goal `[src_xy, src_xy, GRASP_Z]` and moving at < 50mm/s. The contact approach
must handle this entire range cleanly.

### Worst Case: Arm 10mm Above GRASP_Z

```
Grip site Z: GRASP_Z + 0.010 = 0.435m
Finger bottom: 0.435 - 0.0185 = 0.4165m
Cube top: 0.400 + 0.030 = 0.430m
Finger-cube gap: 0.430 - 0.4165 = 13.5mm (fingers above cube)
```

The contact approach must close 10mm of vertical gap. At 3mm/step max and 50 step
budget, the approach has a 150mm travel budget — more than enough.

### Worst Case: Arm 10mm Ahead of GRASP_Z (Too Low)

Can this happen? The DESCEND success condition checks `dist_to_goal < 10mm`. If
GRASP_Z=0.425 and the arm is at 0.415, that's exactly on the table-collision boundary.
The contact approach will not descend further — it only moves the arm if
`grip_pos[2] > GRASP_Z`. If the arm is at or below GRASP_Z, the contact approach
terminates immediately. This is correct behavior: the arm is already at or below
the target.

### Worst Case: 9mm XY Offset from Cube

After DESCEND, the arm may be 9mm off in XY from the cube. The contact approach
only adjusts Z, not XY. The finger face is 77mm wide (size_x = 0.0385m → 38.5mm
each side of center). With a 9mm XY offset, both fingers still contact the 30mm
cube — no misalignment risk.

```
Left edge coverage:   38.5mm - 9mm = 29.5mm (cube width 30mm → just covers)
Right edge coverage:  38.5mm + 9mm = 47.5mm (ample)
```

A 9mm XY miss is within the finger coverage envelope. The precondition check
(`cube_xy_dist < 10mm`) ensures this is never exceeded.

### Contact Approach Step Limit

The approach uses `mocap_pos` adjustment directly (not the RL action space). Each
step moves at most 3mm toward GRASP_Z. The 50-step budget covers up to 150mm of
vertical travel — the maximum descent from DESCEND tolerance is 10mm, so the budget
is 15× oversized. No timeout risk under any normal DESCEND outcome.

### What If Contact Approach Fails?

If the arm cannot reach GRASP_Z within 5mm tolerance in 50 steps, the grasp aborts
with `CONTACT_APPROACH_FAILED`. This can occur if:
1. The mocap weld is at its stiffness limit (physically prevented)
2. The arm's torso joint is at its upper limit, blocking downward motion
   (note: at GRASP_Z=0.425, the arm is descending, not ascending — torso limit is
   not an issue here)

In practice, CONTACT_APPROACH_FAILED should not occur if DESCEND succeeds within
10mm. The approach is a 5mm fine-tune, not a large movement.

---

## Cube Z-Offset During Hold: Full Effect Analysis

**User Note 7: The cube hangs 15mm below the fingers when held. Does this affect anything?**

### What causes the 15mm offset?

When the cube is grasped at GRASP_Z=0.425:
- Grip site Z = 0.425m
- Cube CoM Z under gravity ≈ 0.410m (15mm below grip)

The grip force is entirely from friction (F_friction = μ × F_normal = 10.0 × 2,100N).
Gravity (0.49N) pulls the cube down, and friction resists it. The cube settles at
the point where friction force equals gravity force — which is still extremely close
to the grip site center. The 15mm offset is MuJoCo's equilibrium under soft contacts.

### Effect 1: SAFE_Z Transit Clearance

At SAFE_Z=0.550m, the cube CoM is at ~0.535m and cube bottom at ~0.520m.

```
Table surface:          0.400m
Cube bottom (held):     0.520m
Clearance over table:   120mm  ← no collision risk
Max chess piece height: ~30mm (assumption)
Clearance over pieces:  490mm  ← ample
```

**Conclusion: no effect on transit clearance.**

### Effect 2: Grasp Verification Z Check

The verification checks `abs(cube_pos[2] - grip_pos[2]) < GRASP_VERIFY_Z_THRESHOLD`.
At verification time, the cube is at GRASP_Z - 0.015 = 0.410m while the grip site
is at GRASP_Z = 0.425m. The gap is 15mm.

`GRASP_VERIFY_Z_THRESHOLD = 0.020` (20mm) covers this 15mm gap with 5mm margin.
**No adjustment needed — the threshold is already set correctly.**

### Effect 3: _check_cube_held During Ascent/Transit

The held-cube monitor explicitly accounts for the offset:
```python
z_error = abs(cube_pos[2] - (grip_pos[2] - 0.015))
```
The 15mm offset is baked into the reference. This is correct — a stable grasp will
have `z_error ≈ 0` once physics settles.

### Effect 4: evaluate_grasp_quality Z Reference

The quality evaluator uses `expected_cube_z = grip_pos[2] - 0.015`. At Home Position
with arm at SAFE_Z, `grip_pos[2] ≈ SAFE_Z = 0.550m`, so `expected_cube_z ≈ 0.535m`.
This is the correct reference for quality reporting.

### Effect 5: Descent → Contact Approach Z Reference

The contact approach targets `GRASP_Z` for the grip site, which puts the finger
centers at `GRASP_Z`. The cube top is at `table_z + cube_height = 0.400 + 0.030 = 0.430m`.
The finger bottom at GRASP_Z=0.425 is at `0.425 - 0.0185 = 0.4065m`, which is
**23.5mm above the table** and **23.5mm below the cube top** → 78% cube overlap.

The 15mm "hang" only manifests once the cube is picked up and gravity acts. During
the approach, the cube is on the table and does not hang.

### Summary

| Effect | Impact | Action |
|:---|:---|:---|
| Transit clearance | None (120mm above table) | No change |
| Grasp verify Z | Covered by 20mm threshold | No change |
| Cube held monitor | Explicitly corrected | Already in code |
| Grasp quality metric | Explicitly corrected | Already in evaluator |
| Contact approach | Not applicable (cube on table) | No change |

**The 15mm hang is handled correctly everywhere. No additional code changes required.**

---
*Next: [04 — Home Position and Full Sequence](./04_home_position_sequence.md)*
