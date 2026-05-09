# Transition Architecture: The Soft Reset

## Overview

The "soft reset" is the mechanism that connects one scenario to the next in a chain.
It replaces `env.reset()` at scenario boundaries. The arm stays where it is; only the
task description (scenario, goal, tube) changes. The arm must be **completely stationary
and precisely positioned** at the nominal waypoint before the next scenario begins.

The transition executes in four strict phases, in order:

```
Phase 1: COMPLETE HALT
Phase 2: WAYPOINT ALIGNMENT
Phase 3: GRIPPER STATE TRANSITION
Phase 4: SOFT RESET (state update + fresh observation)
```

---

## Phase 1: Complete Halt

The RL success condition requires `speed < 50mm/s`. After a success step, the arm can
still be coasting. Phase 1 drives the arm to a dead stop before anything else happens.

**Why dead stop?** A coasting arm entering a 5mm tube is a guaranteed TUBE_BREACH.
The alignment loop (Phase 2) also assumes the arm is stationary so its gain-based
convergence is stable.

### Step 1a — Hold Actions

Issue zero-movement actions until the arm settles:

```python
HALT_VEL_THRESHOLD  = 0.0005   # 0.5mm/s — tight enough to call it "stopped"
HALT_HOLD_MAX_STEPS = 100      # up to 100 hold steps before active zeroing

zero_action = np.zeros(4)      # zero dx, dy, dz, gripper (gripper is overridden anyway)

for step in range(HALT_HOLD_MAX_STEPS):
    grip_vel = env.unwrapped._utils.get_site_xvelp(
        env.unwrapped.model, env.unwrapped.data, "robot0:grip"
    )
    speed = np.linalg.norm(grip_vel)
    if speed < HALT_VEL_THRESHOLD:
        log.debug(f"[HALT] Settled in {step} hold steps. speed={speed*1000:.3f}mm/s")
        break
    obs, _, _, _, _ = env.step(zero_action)
```

In practice, episodes end at 1–3mm/s and settle within 5–10 hold steps. The 100-step
budget is a large safety margin.

### Step 1b — Active Velocity Zeroing (Robot Only)

After the hold loop (whether it settled or not), actively zero robot joint velocities:

```python
# Zero robot joint velocities — robot DOF only, NOT the object joint.
# (future scenarios will have a cube in the gripper; zeroing object qvel would cause a drop)
#
# Fetch MuJoCo joint layout (verified by querying the model dynamically):
#   qvel[ 0: 3] — slide0, slide1, slide2 (base translation)
#   qvel[ 3]    — torso_lift
#   qvel[ 4: 6] — head_pan, head_tilt
#   qvel[ 6:13] — arm joints (shoulder_pan through wrist_roll) — 7 DOF
#   qvel[13:15] — l_gripper_finger_joint, r_gripper_finger_joint
#   qvel[15+]   — object0:joint (free joint, 6 DOF)
ROBOT_DOF = 15   # slide(3) + torso(1) + head(2) + arm(7) + fingers(2)
env.unwrapped.data.qvel[:ROBOT_DOF] = 0.0
env.unwrapped.data.qacc[:ROBOT_DOF] = 0.0
mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
```

**Critical distinction — robot DOF only:** `data.qvel[:]` = 0 would also zero any
held cube. Once grasping is implemented, zeroing the cube's velocity would violently
halt a falling/rotating piece. Zero only the robot (indices 0–14) to leave object
physics intact. Note: `ARM_JOINT_COUNT = 7` is incorrect — the arm occupies indices
6–12, but the base and head joints (indices 0–5) also need to be zeroed to prevent
the torso and slides from drifting.

### Step 1c — Halt Confirmation

Verify the arm is truly stopped:

```python
grip_vel = env.unwrapped._utils.get_site_xvelp(
    env.unwrapped.model, env.unwrapped.data, "robot0:grip"
)
speed = np.linalg.norm(grip_vel)
if speed >= HALT_VEL_THRESHOLD:
    raise TransitionError(
        f"HALT_FAILED: arm still moving at {speed*1000:.3f}mm/s "
        f"after {HALT_HOLD_MAX_STEPS} hold steps + active zeroing"
    )
log.info(f"[HALT] Complete. speed={speed*1000:.3f}mm/s ✓")
```

After Step 1b (active zeroing), a `mj_forward` pass immediately brings the arm
to rest. The confirmation check is a sanity guard; it should not fail in practice.

**Why ROBOT_DOF = 15, not 7:** The first 7 `qvel` indices in Fetch MuJoCo are the
base slides (3), torso (1), and head (2) — only the last of the 7 is the first arm joint.
The full arm occupies indices 6–12; fingers occupy 13–14. Using only 7 would leave the
elbow, wrist, and forearm joints still moving. The object joint starts at index 15.

---

## Phase 2: Waypoint Alignment

After complete halt, the arm is within ±10mm of the nominal waypoint (success_threshold
= 10mm). This is not precise enough for a safe transition in a 10mm tube. Phase 2 physically moves the arm
to the exact nominal waypoint using a scripted gain-based settle loop.

**This is NOT a teleport.** It is a physics-simulated smooth movement. The arm moves
through physics, which is safe for future cube scenarios because the gripper is closed
and the cube moves with it.

### Why Not Use Actual-Position as Tube Center?

The chess cell position does not change. If we set tube_center = arm's actual landing
position, descend will be aimed at a slightly wrong point on the board. A transit that
ends 9mm off would place the chess piece 9mm from the correct square — that is
physically incorrect.

The correct solution is to move the arm to the right place, not redefine "the right
place" to be wherever the arm happened to end up.

### The Alignment Loop

The settle loop is the same pattern already proven in `_settle_arm_to_start`:

```python
ALIGN_TOLERANCE_M   = 0.003    # 3mm — tighter than the 10mm drift limit; 7mm breathing room
ALIGN_MAX_STEPS     = 200      # budget for the scripted settle
ALIGN_GAIN          = 0.8      # proportional gain (same as _settle_arm_to_start)

def align_to_waypoint(env_unwrapped, target_pos: np.ndarray) -> dict:
    """
    Physically moves the arm to target_pos using the settle loop.
    Returns diagnostic dict: {steps, final_error_mm, converged}.
    """
    uw = env_unwrapped
    for step in range(ALIGN_MAX_STEPS):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        error = target_pos - grip_pos
        dist = np.linalg.norm(error)
        if dist < ALIGN_TOLERANCE_M:
            # Zero robot velocities after settling — robot DOF only, not object
            uw.data.qvel[:ROBOT_DOF] = 0.0
            uw.data.qacc[:ROBOT_DOF] = 0.0
            mujoco.mj_forward(uw.model, uw.data)
            log.info(f"[ALIGN] Converged in {step} steps. error={dist*1000:.2f}mm ✓")
            return {"steps": step, "final_error_mm": dist * 1000, "converged": True}

        # Proportional step toward target
        step_vec = ALIGN_GAIN * error
        # Clamp to max single-step size (same as _settle_arm_to_start)
        max_step = 0.005   # 5mm per step
        if np.linalg.norm(step_vec) > max_step:
            step_vec = step_vec / np.linalg.norm(step_vec) * max_step
        uw.data.mocap_pos[0][:3] += step_vec
        uw._mujoco.mj_step(uw.model, uw.data, nstep=uw.n_substeps)

    # Did not converge
    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    dist = np.linalg.norm(target_pos - grip_pos) * 1000
    log.error(f"[ALIGN] FAILED to converge after {ALIGN_MAX_STEPS} steps. "
              f"final_error={dist:.1f}mm, target={target_pos}")
    return {"steps": ALIGN_MAX_STEPS, "final_error_mm": dist, "converged": False}
```

### Alignment Targets by Scenario

```
Transit ends at:  nominal_waypoint = [dst_xy[0], dst_xy[1], SAFE_Z]
Descend ends at:  nominal_waypoint = [tube_center[0], tube_center[1], GRASP_Z]
Ascend ends at:   nominal_waypoint = [tube_center[0], tube_center[1], SAFE_Z]
```

The alignment target is always the **nominal** endpoint of the current scenario — the
exact position the arm was supposed to reach. After alignment, `tube_center = nominal_xy`
for the next descend/ascend scenario.

### Abort if Alignment Fails

```python
align_result = align_to_waypoint(env_unwrapped, current_scenario_exit_waypoint)
if not align_result["converged"]:
    raise TransitionError(
        f"ALIGN_FAILED: arm at {align_result['final_error_mm']:.1f}mm from "
        f"waypoint after {ALIGN_MAX_STEPS} steps"
    )
```

---

## Phase 3: Gripper State Transition

After the arm is stationary and precisely positioned, transition the finger state if the
next scenario requires it. This uses the same scripted 50-step loop as `_reset_sim`
Phase 2. The arm holds position during this phase.

| From Scenario | To Scenario | Finger Transition | Steps |
|:---|:---|:---|:---|
| Transit (closed) | Descend | Closed → Open | 50 scripted steps |
| Descend (open) | Ascend | Open → Closed | 50 scripted steps |
| Ascend (closed) | Transit | No change | — |
| Transit (closed) | Transit | No change | — |
| Descend (open) | Transit | Open → Closed | 50 scripted steps |
| Ascend (closed) | Descend | Closed → Open | 50 scripted steps |

After the scripted transition, perform the same finger validation used in `_reset_sim`
Phase 3:

```python
l_pos = _utils.get_joint_qpos(model, data, "robot0:l_gripper_finger_joint").item()
assert abs(l_pos - target_qpos) < 0.0005, \
    f"Finger transition failed: actual={l_pos:.6f}, target={target_qpos:.6f}"
```

---

## Phase 4: Soft Reset (State Update)

The environment state is updated without any physics reset:

```python
env_unwrapped.current_scenario = new_scenario
env_unwrapped.goal_pos = derive_goal_for_scenario(new_scenario, nominal_xy)
env_unwrapped.goal = env_unwrapped.goal_pos.copy()  # Critical: _build_phase9_observation() uses self.goal
env_unwrapped.episode_steps = 0

# For descend/ascend: tube_center = nominal_xy (arm is now aligned there)
if new_scenario in {"descend", "ascend"}:
    env_unwrapped.tube_center_xy = nominal_xy.copy()  # NOT grip_pos[:2]
else:
    env_unwrapped.tube_center_xy = None

# Re-enforce vertical orientation
env_unwrapped._utils.set_mocap_quat(
    env_unwrapped.model, env_unwrapped.data, "robot0:mocap", VERTICAL_QUAT
)

return env_unwrapped._get_obs()
```

**tube_center = nominal_xy always.** After Phase 2 alignment, the arm IS at nominal_xy
within 3mm (well inside the 5mm drift limit), so setting tube_center = nominal_xy is
physically correct.

---

## Goal Position Derivation Rules

For each scenario in a chain, the goal position is determined from the **nominal** board
position — the exact chess cell coordinates, not the arm's achieved position.

```
Transit(src_xy → dst_xy):
    goal_pos = np.array([dst_xy[0], dst_xy[1], SAFE_Z])

Descend(at dst_xy):
    tube_center = dst_xy   ← NOMINAL, not actual arm position
    goal_pos = np.array([dst_xy[0], dst_xy[1], GRASP_Z])

Ascend(at dst_xy):
    tube_center = dst_xy   ← NOMINAL, not actual arm position
    goal_pos = np.array([dst_xy[0], dst_xy[1], SAFE_Z])
```

The XY of descend/ascend goals is always the chess cell center. The arm is aligned to
that position during Phase 2. There is no "close enough" fallback.

---

## Complete Transition Sequence (Summary)

```
[RL SUCCESS] speed < 50mm/s, dist < 10mm
        ↓
[PHASE 1: COMPLETE HALT]
  1a. Issue hold actions until speed < 0.5mm/s (up to 100 steps)
  1b. Zero arm joint qvel/qacc (arm joints only, not object)
  1c. Confirm speed < 0.5mm/s → abort chain if not
        ↓
[PHASE 2: WAYPOINT ALIGNMENT]
  Run scripted settle loop toward nominal exit waypoint
  Arm physically moves from ±10mm → within 3mm of exact waypoint
  Zero arm qvel after convergence
  Abort chain if not converged within 200 steps
        ↓
[PHASE 3: GRIPPER TRANSITION]
  Execute scripted 50-step open/close if needed
  Validate finger position (< 0.0005 joint error)
        ↓
[PHASE 4: SOFT RESET]
  Update: scenario, goal_pos, tube_center = nominal_xy
  Reset episode_steps = 0
  Re-enforce vertical orientation
  Return fresh observation
        ↓
[NEXT SCENARIO BEGINS]
  Model sees: new goal, new scenario_id, near-zero velocity, arm at nominal position
```

---

## Chain Failure Policy

If any phase of the transition fails, or any scenario in a chain crashes or times out,
the chain **stops immediately**. The remaining scenarios are not attempted.

```
Chain [transit, descend, ascend]:
  Step 1 [transit]:  SUCCESS  (dist=7.3mm, speed=1.9mm/s, steps=18)
    HALT:            OK       (settled in 6 hold steps, speed=0.3mm/s)
    ALIGN:           OK       (converged in 34 steps, error=1.8mm)
    GRIPPER:         OK       (open transition validated)
  Step 2 [descend]:  CRASH    (TUBE_BREACH at step 4, drift=0.0063 > limit=0.0050)
  Step 3 [ascend]:   SKIPPED  (chain aborted at step 2)
  Chain result: FAILED at scenario 2/3
```

---
*Next: [03 — Evaluation Script Specification](./03_eval_sequence_script.md)*
4, drift=0.0063 > limit=0.0050)
  Step 3 [ascend]:   SKIPPED  (chain aborted at step 2)
  Chain result: FAILED at scenario 2/3
```

---
*Next: [03 — Evaluation Script Specification](./03_eval_sequence_script.md)*
