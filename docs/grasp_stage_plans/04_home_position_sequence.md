# Home Position and Full Sequence Architecture

## Home Position

### Definition

The Home Position is a fixed 3D point in world space. It is the arm's resting
position before and after a chess move. It must satisfy:
1. **Safe height** — at SAFE_Z (0.550m), clear of all board objects.
2. **Outside the board** — not above any chess square, so there is no collision risk
   when the arm parks here.
3. **Reachable in one transit** — within the board's XY sampling range so the
   existing transit model can navigate to it.

```yaml
# configs/env.yaml — new entry:
home_position_xy: [0.680, 0.2641]   # within transit training range [0.640, 1.120]; left-edge region
                                      # Same Y as table center
```

At `(0.680, 0.264)`, the home position is 20cm from table center in X — 4cm inside
the left training boundary. The transit model trains over `low_x = TABLE_CENTER_X -
TABLE_HALF_X + EDGE_MARGIN = 0.88 - 0.28 + 0.04 = 0.640`. HOME_X=0.680 is within
this range, so the policy has been trained to navigate to and from this position.

**Why not X=0.600?** X=0.600 equals the raw table edge (`0.88 - 0.28`), which is
40mm outside the training distribution. The transit policy has never seen a goal or
start at X=0.600 during training. Placing Home there causes out-of-distribution
behavior — the arm may stall, overshoot, or oscillate.

The home position 3D point used for transit goals:
```python
HOME_POS = np.array([0.680, 0.2641, SAFE_Z])  # [home_xy[0], home_xy[1], SAFE_Z]
```

### Home Position Transit

Home is reached via the existing TRANSIT scenario. The transit model handles any
SAFE_Z → SAFE_Z movement. No changes needed to the RL model.

```python
# From anywhere at SAFE_Z → Home:
goal = np.array([HOME_XY[0], HOME_XY[1], SAFE_Z])
# Use force_scenario="transit" or just set current_scenario="transit"
```

---

## Full Sequence Architecture

### The Six-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: INITIALIZE                                               │
│   env.reset(force_scenario="transit")                            │
│   Arm settles at Home Position (SAFE_Z, home_xy)                 │
│   Cube placed at src_xy on table                                  │
│   hide_object = False                                             │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: TRANSIT (home → src_xy at SAFE_Z)                        │
│   RL model drives transit scenario                               │
│   Goal: [src_xy[0], src_xy[1], SAFE_Z]                          │
│   Success: dist < 10mm, speed < 50mm/s                           │
│   Transition: soft_reset → DESCEND                               │
│   (halt + align to [src_xy[0], src_xy[1], SAFE_Z])              │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: DESCEND (SAFE_Z → GRASP_Z at src_xy)                     │
│   RL model drives descend scenario                               │
│   Goal: [src_xy[0], src_xy[1], GRASP_Z=0.425]                   │
│   Tube: ±10mm around src_xy                                      │
│   Success: dist < 10mm, speed < 50mm/s                           │
│   On success: transition to GRASP (not via soft_reset)           │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: GRASP (scripted)                                         │
│   execute_grasp() — see doc 03                                   │
│   Sub-phases: contact approach → finger close → verify → hold    │
│   grasp_mode = True set here                                     │
│   Success: cube verified in gripper                              │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: ASCEND (GRASP_Z → SAFE_Z at src_xy, cube held)          │
│   RL model drives ascend scenario                                │
│   grasp_mode remains True (actuator-driven fingers)             │
│   Goal: [src_xy[0], src_xy[1], SAFE_Z]                          │
│   Additional check: CUBE_DROPPED crash if cube leaves gripper    │
│   Transition: soft_reset → TRANSIT                               │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: TRANSIT (src_xy → home, cube held)                       │
│   RL model drives transit scenario                               │
│   grasp_mode remains True                                        │
│   Goal: HOME_POS                                                 │
│   CUBE_DROPPED check active                                      │
│   End: evaluate grasp quality (cube position/rotation/XY drift)  │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Design Decisions

**1. DESCEND → GRASP Transition: No `soft_reset` needed.**

The GRASP phase starts from the arm's current position at GRASP_Z. Unlike RL-to-RL
transitions, the GRASP is a scripted phase that begins immediately after DESCEND's
success condition. The transition is:

```python
if descend_result["outcome"] == "success":
    # No halt phase needed — arm already satisfied speed < 50mm/s
    # No align phase needed — GRASP contact approach handles the final mm
    # No state update needed — grasp is not an RL scenario
    grasp_result = env.unwrapped.execute_grasp()
```

**2. GRASP → ASCEND Transition: Also no `soft_reset`.**

After a successful grasp, the arm is stationary at GRASP_Z with the cube in the
gripper. The ASCEND scenario starts from exactly this position:

```python
if grasp_result["success"]:
    # Switch to ASCEND via soft_reset — but with grasp_mode=True preserved
    obs = env.unwrapped.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], GRASP_Z]),
        nominal_xy=src_xy,
    )
    # grasp_mode was set by execute_grasp; soft_reset must NOT reset it
```

**3. `soft_reset` must NOT reset `grasp_mode`.**

Add this note to `soft_reset`:
```python
# Do NOT reset self.grasp_mode here.
# The grasp persists through ASCEND and TRANSIT.
# grasp_mode is only reset by env.reset() (full reset).
```

**4. First scenario: TRANSIT from Home.**

The first scenario in the chain is always TRANSIT from Home Position to src_xy.
The arm must start at Home Position, not a random position. This requires passing
`force_scenario="transit"` AND resetting the arm to Home Position.

```python
# Before env.reset(), override force_scenario and home position:
env_unwrapped = env.unwrapped
env_unwrapped.force_scenario = "transit"
env_unwrapped.force_start_pos = HOME_POS  # new override flag
obs, _ = env.reset()
```

A new `force_start_pos` parameter in `_reset_sim` will override the random start
position sampling with the fixed home position.

---

## `_reset_sim` Changes for Home Position Support

```python
def _reset_sim(self) -> bool:
    # ... existing code ...
    
    if self.current_scenario == "transit":
        goal_pos = self._sample_board_position()
        while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
            goal_pos = self._sample_board_position()
        
        # NEW: if force_start_pos is set, override the start position
        if hasattr(self, 'force_start_pos') and self.force_start_pos is not None:
            arm_start_pos = self.force_start_pos.copy()
        else:
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
        
        self.tube_center_xy = None
        self.goal_pos = np.array([goal_pos[0], goal_pos[1], self.SAFE_Z])
    # ... rest unchanged ...
```

---

## Cube Placement Strategy

For the grasp evaluation, the cube is placed at a known `src_xy` position and the
arm transits to that position. The cube placement must be consistent with the
target used for transit/descend.

```python
# In eval_grasp.py, before env.reset():
src_xy = env.unwrapped._sample_board_position()[:2]
env.unwrapped.force_start_pos = HOME_POS
env.unwrapped.force_cube_pos = np.array([src_xy[0], src_xy[1], TABLE_Z + CUBE_HEIGHT/2])
```

A new `force_cube_pos` parameter in `_reset_sim` (within `ChessTaskEnv._reset_sim`)
will override the random cube placement:

```python
# In ChessTaskEnv._reset_sim, replace the cube placement code:
if hasattr(self, 'force_cube_pos') and self.force_cube_pos is not None:
    cube_pos = self.force_cube_pos.copy()
else:
    cube_pos = self._sample_board_position()
    cube_pos[2] = self.TABLE_SURFACE_Z + self.CUBE_HEIGHT / 2.0
```

---

## Waypoint Assignment for the Full Sequence

```python
# eval_grasp.py: waypoint and chain assignment

HOME_XY = np.array(env_cfg["home_position_xy"])
src_xy = env.unwrapped._sample_board_position()[:2]

# Chain: transit, descend, (grasp — scripted, not RL), ascend, transit
rl_chain = ["transit", "descend", "ascend", "transit"]
waypoints = [
    src_xy,      # transit: home → src_xy
    src_xy,      # descend: SAFE_Z → GRASP_Z at src_xy
    src_xy,      # ascend:  GRASP_Z → SAFE_Z at src_xy
    HOME_XY,     # transit: src_xy → home (carrying cube)
]
# The GRASP step is handled outside the rl_chain loop
```

---

## Post-Sequence Grasp Quality Metrics

After the arm returns to Home Position, evaluate the grasp quality:

```python
def evaluate_grasp_quality(env_unwrapped, src_xy: np.ndarray) -> dict:
    """Evaluates how well the cube was held during the full sequence."""
    cube_pos = env_unwrapped.get_cube_position()
    cube_quat = env_unwrapped.get_cube_quat()
    grip_pos = env_unwrapped._utils.get_site_xpos(
        env_unwrapped.model, env_unwrapped.data, "robot0:grip"
    )
    
    # XY drift from original placement
    xy_drift = np.linalg.norm(cube_pos[:2] - src_xy) * 1000  # mm
    
    # Z height — should be at SAFE_Z - 15mm (held at arm height)
    expected_cube_z = grip_pos[2] - 0.015  # 15mm below grip site
    z_error = abs(cube_pos[2] - expected_cube_z) * 1000  # mm
    
    # Rotation — convert quaternion to euler angles, check roll/pitch/yaw
    import scipy.spatial.transform
    r = scipy.spatial.transform.Rotation.from_quat(
        [cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]]  # scipy: xyzw
    )
    euler_deg = r.as_euler('xyz', degrees=True)
    max_rotation = np.max(np.abs(euler_deg))
    
    return {
        "cube_xy_drift_mm": xy_drift,
        "cube_z_error_mm": z_error,
        "cube_max_rotation_deg": max_rotation,
        "cube_euler_xyz_deg": euler_deg.tolist(),
        "cube_pos": cube_pos.tolist(),
        "grip_pos": grip_pos.tolist(),
    }
```

---

## Full Chess Move: Pick + Place Sequence

**User Note 2: Detailed explanation of all waypoints and stages for a complete chess move.**

> **⚠️ SCOPE NOTE — Steps 7–11 (PLACE and beyond) are forward-looking architecture only.**
> This stage (Grasp Stage) ends at Step 6: the arm returns home with the cube held.
> Steps 7–11 document the intended design for the PLACE stage but **cannot be
> executed yet**. The `execute_place()` function, PLACE_Z support in `_reset_sim`,
> and the place-leg descend scenario (with grasp_mode=True during descent) are all
> out of scope. Do not implement these in this stage.

A complete chess move requires picking a piece from `src_xy` and placing it at
`dst_xy`. This extends the six-step pick sequence with five additional steps.

### The Eleven-Step Full Move Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: INITIALIZE                                                │
│   Arm at Home Position (SAFE_Z). Cube at src_xy.                 │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: TRANSIT — home → src_xy (SAFE_Z)                         │
│   Waypoint: [src_xy[0], src_xy[1], SAFE_Z]                       │
│   Soft-reset → DESCEND on success                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: DESCEND — SAFE_Z → GRASP_Z at src_xy (fingers open)      │
│   Waypoint: [src_xy[0], src_xy[1], GRASP_Z=0.425]               │
│   On success → GRASP (no soft_reset)                             │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: GRASP — scripted close + verify (at src_xy, GRASP_Z)     │
│   execute_grasp() — contact approach + close + verify + settle   │
│   grasp_mode = True set here                                     │
│   On success → ASCEND via soft_reset (grasp_mode preserved)      │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: ASCEND — GRASP_Z → SAFE_Z at src_xy (cube held)          │
│   Waypoint: [src_xy[0], src_xy[1], SAFE_Z]                       │
│   grasp_mode = True; CUBE_DROPPED check active                   │
│   Soft-reset → TRANSIT on success                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: TRANSIT — src_xy → dst_xy (SAFE_Z, cube held)            │
│   Waypoint: [dst_xy[0], dst_xy[1], SAFE_Z]                       │
│   grasp_mode = True; CUBE_DROPPED check active                   │
│   Soft-reset → DESCEND on success                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 7: DESCEND — SAFE_Z → PLACE_Z at dst_xy (cube held)         │
│   PLACE_Z = table_z + cube_height/2 + small_margin ≈ 0.430m     │
│   Lower the cube to just above the board surface                 │
│   grasp_mode = True during descent                               │
│   On arrival → PLACE (no soft_reset)                             │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 8: PLACE — scripted release (at dst_xy, PLACE_Z)            │
│   Open fingers: grasp_mode = False, finger_target = OPEN         │
│   Scripted retract: arm moves up 5mm while fingers open          │
│   Verify: cube at dst_xy ± 10mm, cube Z = table level            │
│   On success → ASCEND (no soft_reset needed)                     │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 9: ASCEND — PLACE_Z → SAFE_Z at dst_xy (fingers open)       │
│   Waypoint: [dst_xy[0], dst_xy[1], SAFE_Z]                       │
│   grasp_mode = False                                             │
│   Soft-reset → TRANSIT on success                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 10: TRANSIT — dst_xy → home (SAFE_Z)                        │
│   Waypoint: HOME_POS                                             │
│   Arm returns to Home Position                                   │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 11: EVALUATE                                                 │
│   Measure: cube final XY vs dst_xy, cube rotation, step counts   │
│   Record: success/failure at each step, total steps used         │
└──────────────────────────────────────────────────────────────────┘
```

### Waypoint Table for a Complete Move

| Step | Scenario | Waypoint XY | Waypoint Z | grasp_mode | Notes |
|:---|:---|:---|:---|:---|:---|
| 1 | — | home_xy | SAFE_Z | False | Start state |
| 2 | transit | src_xy | SAFE_Z | False | Move over piece |
| 3 | descend | src_xy | GRASP_Z=0.425 | False | Lower to piece |
| 4 | GRASP (scripted) | src_xy | GRASP_Z | False→True | Close fingers |
| 5 | ascend | src_xy | SAFE_Z | True | Lift piece |
| 6 | transit | dst_xy | SAFE_Z | True | Move to target |
| 7 | descend | dst_xy | PLACE_Z≈0.430 | True | Lower to board |
| 8 | PLACE (scripted) | dst_xy | PLACE_Z | True→False | Release piece |
| 9 | ascend | dst_xy | SAFE_Z | False | Retract |
| 10 | transit | home_xy | SAFE_Z | False | Return home |
| 11 | — | home_xy | SAFE_Z | False | Done |

### PLACE_Z vs GRASP_Z

PLACE_Z ≈ 0.430m. This is the Z at which the cube bottom just touches the board
surface, with the arm slightly higher to avoid pushing the cube into the board.

```
Cube bottom at PLACE_Z: PLACE_Z - cube_height/2 - grip_site_to_cube_offset
                       ≈ 0.430 - 0.015 - 0.015 = 0.400m (table surface)
```

GRASP_Z (0.425) < PLACE_Z (0.430): the arm descends slightly lower for picking
than placing. This is intentional — picking requires maximum finger overlap; placing
just needs the cube to touch the surface.

### Key Transitions in the Full Move

**Step 3 → 4 (DESCEND → GRASP):** No soft_reset. The grasp starts from wherever
DESCEND succeeded (within 10mm of GRASP_Z). The contact approach closes the gap.

**Step 4 → 5 (GRASP → ASCEND):** soft_reset is needed to switch the RL scenario
from descend to ascend. grasp_mode must be preserved (not reset). The soft_reset
halt phase keeps the arm stationary while the scenario context changes.

**Step 5 → 6 (ASCEND → TRANSIT):** soft_reset switches ascend→transit. The new
goal is dst_xy at SAFE_Z. grasp_mode stays True.

**Step 6 → 7 (TRANSIT → DESCEND):** soft_reset switches transit→descend. Now
descending toward dst_xy at PLACE_Z. grasp_mode stays True. Note: descend scenario
with cube held is **new behavior** (cube integration step 2, not this stage).

**Step 7 → 8 (DESCEND → PLACE):** No soft_reset. Same pattern as DESCEND→GRASP.
The scripted PLACE phase opens fingers and retracts.

**Step 8 → 9 (PLACE → ASCEND):** After fingers open, switch to ascend scenario
with grasp_mode=False. The cube is now on the board. soft_reset halt keeps arm
stationary during scenario switch.

---

## Soft_Reset Reference: All Scenario Transitions

**User Note 3: Detailed soft_reset explanation for each possible scenario transition.**

Soft_reset is a controlled handoff between two RL scenarios. It prevents the abrupt
state change from confusing the new scenario's policy. The four phases are:

### Phase 1: HALT (100 steps, gain=0)

The arm's velocity is zeroed by overriding the action with zero and letting MuJoCo
dissipate kinetic energy. During halt:
- `action = [0, 0, 0, 0]`
- Finger teleportation continues (unless grasp_mode=True, in which case fingers
  remain actuator-driven at their current target)
- The physics integrator runs normally — contact forces and gravity apply

**grasp_mode interaction:** If grasp_mode=True, the halt phase must NOT reset
finger targets. The cube hangs in the gripper during halt, which is correct.

### Phase 2: ALIGN (up to 200 steps, gain=0.8)

The arm uses the mocap controller to move to the exact nominal exit position for
the outgoing scenario. The align target is the waypoint where the previous scenario
should have ended.

```python
align_target = nominal_exit_pos  # Where the old scenario's ideal endpoint is
error = align_target - current_grip_pos
mocap_delta = 0.8 * error
apply_mocap_delta(mocap_delta)
```

If the arm converges within 1mm of `nominal_exit_pos`, the phase ends early.
If it doesn't converge in 200 steps, soft_reset fails (the chain is aborted).

### Phase 3: STATE UPDATE

The environment's internal scenario is switched. The new scenario's goal is set.
The new reward function, success condition, and observation construction take effect.

### Phase 4: GRIPPER TRANSITION (50 steps)

The finger target is set for the new scenario. If the new scenario requires open
fingers, they are teleported open. If closed — teleported closed.

**grasp_mode interaction:** This phase is SKIPPED when grasp_mode=True. The fingers
remain actuator-driven at their current target (FINGER_CLOSED_JOINT). There is no
teleportation. Bug C fix (doc 05) guards this.

---

### Transition: transit → descend

**When:** Home → src_xy transit succeeds; arm is at [src_xy, SAFE_Z].

```
Phase 1 (HALT):   Stop arm at [src_xy, SAFE_Z]. grasp_mode=False.
Phase 2 (ALIGN):  Align to nominal exit = [src_xy, SAFE_Z]. Typically ≤ 5mm.
Phase 3 (STATE):  Switch to descend. New goal = [src_xy, GRASP_Z].
Phase 4 (GRIP):   finger_target = FINGER_OPEN_JOINT (teleport open). Fine.
```

**Risk:** If transit arm position drifted in XY (tube drift), the align step corrects
up to 200 steps × 0.8 gain. The descend goal is fixed at src_xy, so any XY correction
here propagates correctly.

---

### Transition: DESCEND → GRASP (no soft_reset)

**When:** Descend succeeds. Arm is within 10mm of [src_xy, GRASP_Z].

No soft_reset is performed. The scripted `execute_grasp()` begins immediately:
1. Contact approach closes the 0–10mm vertical gap
2. Fingers close actuator-driven (grasp_mode=True set inside execute_grasp)

**Why no soft_reset:** Grasp is a scripted phase, not an RL scenario. The policy
is not called during grasp. The phase transition is direct.

---

### Transition: GRASP → ascend

**When:** execute_grasp() returns success. Arm is at [src_xy, GRASP_Z], cube held.
grasp_mode=True.

```
Phase 1 (HALT):   Arm already stationary (execute_grasp ends with hold settle).
                  However, halt is still run to ensure clean state. grasp_mode=True,
                  so fingers are NOT teleported — actuator holds cube.
Phase 2 (ALIGN):  Align to nominal exit = [src_xy, GRASP_Z].
                  The arm is already there (±1mm), so this phase is nearly instant.
Phase 3 (STATE):  Switch to ascend. New goal = [src_xy, SAFE_Z].
Phase 4 (GRIP):   SKIPPED (grasp_mode=True). Fingers stay actuator-driven.
```

**Critical:** `soft_reset` must not touch `self.grasp_mode`. The ascend scenario
inherits grasp_mode=True from the grasp phase.

---

### Transition: ascend → transit (after pick)

**When:** Ascend succeeds. Arm is at [src_xy, SAFE_Z], cube held.
grasp_mode=True.

```
Phase 1 (HALT):   Stop arm at [src_xy, SAFE_Z]. grasp_mode=True, cube still held.
Phase 2 (ALIGN):  Align to nominal exit = [src_xy, SAFE_Z]. Typically instant.
Phase 3 (STATE):  Switch to transit. New goal = [dst_xy, SAFE_Z] (or home_xy).
Phase 4 (GRIP):   SKIPPED (grasp_mode=True). No finger teleportation.
```

The cube hangs in the gripper throughout. _check_cube_held() runs during transit.

---

### Transition: transit → descend (place leg, cube held)

**When:** Transit to dst_xy succeeds. Arm is at [dst_xy, SAFE_Z], cube held.
grasp_mode=True.

```
Phase 1 (HALT):   Stop arm. Cube held.
Phase 2 (ALIGN):  Align to [dst_xy, SAFE_Z]. Already there.
Phase 3 (STATE):  Switch to descend. New goal = [dst_xy, PLACE_Z≈0.430].
Phase 4 (GRIP):   SKIPPED (grasp_mode=True). Fingers stay closed around cube.
```

The descend scenario now descends toward PLACE_Z rather than GRASP_Z. A new
`force_target_z` override is needed in `_reset_sim` to set PLACE_Z instead of
GRASP_Z as the descend goal.

---

### Transition: DESCEND → PLACE (no soft_reset, cube held)

**When:** Place-leg descend succeeds. Arm at [dst_xy, PLACE_Z], cube touching board.

No soft_reset. scripted `execute_place()` runs:
1. Open fingers: `finger_target = FINGER_OPEN_JOINT`, grasp_mode=False
2. Retract 5mm upward (scripted mocap move) to break contact cleanly
3. Verify cube at dst_xy ± 10mm on table surface

---

### Transition: PLACE → ascend (after release)

**When:** execute_place() succeeds. Arm at [dst_xy, PLACE_Z], fingers open.
grasp_mode=False.

```
Phase 1 (HALT):   Arm already stationary.
Phase 2 (ALIGN):  Align to [dst_xy, PLACE_Z].
Phase 3 (STATE):  Switch to ascend. New goal = [dst_xy, SAFE_Z].
Phase 4 (GRIP):   finger_target = FINGER_OPEN_JOINT (teleport open). Fine.
```

---

### Transition: transit → transit (home return, no cube)

**When:** Post-place transit ends. Arm at [dst_xy or intermediate, SAFE_Z], no cube.

Standard transit→transit soft_reset. Both scenarios are identical in structure.
The new goal is HOME_POS[:2]. grasp_mode=False throughout.

---

*Next: [05 — Implementation Plan](./05_implementation_plan.md)*
