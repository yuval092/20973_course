# Risk Analysis: Grasp Stage Failure Modes

## Risk Summary Table

| Risk | Probability | Severity | Phase | Mitigation |
|:---|:---|:---|:---|:---|
| Cube Z-explosion on finger close | High (without physics fix) | Critical | GRASP | condim=6, solref=0.005, mass=0.05 |
| Cube spins in gripper during transit | High (without condim fix) | High | ASCEND/TRANSIT | condim=6 on both cube and finger geoms |
| Cube slides down during ascent | Medium | High | ASCEND | Kp=150000, friction=10.0 maintained |
| GRASP_Z still wrong (cached config) | Medium | High | DESCEND/GRASP | Restart env after yaml change |
| Arm still teleporting in grasp_mode | Medium | High | GRASP | Confirm grasp_mode branch in _set_action |
| Descend ends 9mm above cube | Medium | Medium | GRASP | Contact approach sub-phase handles this |
| Cube not under gripper (XY miss) | Low | High | GRASP | Precondition check + abort |
| Wrist snap on cube pickup | Low | Medium | ASCEND | Weld solref=0.02 |
| Torso limit hit during transit | Low | High | TRANSIT | Monitor torso_lift_joint; known risk |
| grasp_mode not reset on new reset | Low | Medium | All | Explicit reset in _reset_sim |
| "Too-perfect start" at Home | Low | Low | TRANSIT | Same mitigation as before |
| **soft_reset FINGER_VALIDATION crash** | **Certain (unfixed)** | **Critical** | **GRASP→ASCEND** | **Guard with `if not self.grasp_mode:`** |
| step() opens fingers during PLACE descent | Low (future stage) | High | DESCEND (place leg) | Guard finger state machine with `if not self.grasp_mode:` |
| HOME_XY outside training range | Certain (unfixed) | High | TRANSIT | Change HOME_X from 0.600 → 0.680 |

---

## Risk 1: Z-Explosion on Finger Close

**The Problem:**
In the current XML (`mass=0.5`, `condim=4`, no `solref/solimp`), when the actuator
tries to close the fingers against the cube, MuJoCo computes a hard-wall contact with
massive repulsive force. The cube gets "punched" upward by the closing fingers.
This is the most common failure mode for pick-and-place in MuJoCo.

**Root Cause:**
Two hard geoms (finger + cube) interpenetrating → MuJoCo generates impulse proportional
to interpenetration depth and mass. At Kp=60,000 and mass=0.5kg, this impulse is large
enough to pop the cube out of the finger zone.

**Mitigation Chain (all must be applied):**
1. `mass=0.05` → 10× less inertia → 10× less impulse from same contact force
2. `solref="0.002 1"` on cube AND finger geoms → contact force builds within 1 timestep.
   Using 0.005 is too slow — fingers penetrate the 50g cube before contact develops.
3. `solimp="0.99 0.999 0.001"` on cube AND finger geoms → very stiff constraint prevents
   ghosting. Less stiff values (e.g. "0.9 0.95 0.001") allow penetration at Kp=150,000.
4. Actuator-driven close (grasp_mode=True) → gradual force, not instant position
5. 150-step close budget → allows physics to equilibrate

**Test:** Test 1 in `test_grasp_physics.py`. Expected pass rate: ~92% (8% physics
explosions are normal at this mass/Kp ratio). Gate criterion: ≥ 80%. If cube Z moves
> 5mm during close in >20% of trials, physics hardening was not applied. Check that
the XML was saved and the environment was restarted (not using a cached compiled model).

---

## Risk 2: Cube Spin During Transit

**The Problem:**
A square cube in a cylindrical grip (two flat faces pressing two flat finger surfaces)
can rotate along the vertical axis (yaw) during horizontal acceleration. Without
torsional friction (`condim=4`), MuJoCo computes zero resistance to this rotation.
At transit speeds, 30–90° of yaw rotation during a single transit is possible.

**Root Cause:**
`condim=4` provides friction in 4D: 2 translational + 2 for the rolling constraint.
It does NOT provide torsional (yaw) resistance. A cube spinning on its vertical axis
is a torsional motion — it requires `condim=6`.

**Mitigation:**
Apply `condim=6` to both the cube geom AND both finger geoms. The torsional friction
coefficient is set by `friction[1]` (second value in the friction triple):
```xml
friction="10.0 0.005 0.0001"
```
- `friction[0]=10.0`: Translational (sliding) friction — the main hold force
- `friction[1]=0.005`: Torsional friction — prevents spinning
- `friction[2]=0.0001`: Rolling friction — prevents rolling

**Test:** Test 3 in `test_grasp_physics.py`. Run 100 transits with cube held and
measure cube yaw rotation. Target: <15° per transit.

---

## Risk 3: Cube Slides Down During Ascent

**The Problem:**
During ASCEND, the arm accelerates upward. The cube must be held against gravity
PLUS inertial force. The hold force is `F_hold = friction × F_normal`. If `F_normal`
is too small, the cube slides down.

**Physics of the hold:**

The finger geometry means a 30mm cube blocks closure at joint≈0.014 (gap≈30mm).
The actuator commands ctrl=0.0 (CLOSED), so the actuator error = 0.014m.

- `F_normal = Kp × joint_error = 150,000 × 0.014 = 2,100N` (per finger)
- Effective contact friction = min(cube_friction, finger_friction) = min(2.0, 10.0) = 2.0
  (MuJoCo uses the minimum of the two contacting geoms' friction coefficients)
- `F_friction_max = effective_friction × F_normal = 2.0 × 2,100N = 4,200N` (per finger)
- Cube weight: `F_gravity = 0.05 × 9.81 = 0.49N`
- Safety factor: 4,200N / 0.49N ≈ **8,571× — still an enormous margin**

Kp=150,000 is critical. The "compression" is not 0.2mm of material strain — it is
the 14mm joint error between ctrl target (0) and the physically blocked finger
position (0.014). This large error drives a very high contact force.
Finger geom friction=10.0 is irrelevant to the safety margin since the cube
geom's friction=2.0 is the limiting surface.

**Test:** Test 2 in `test_grasp_physics.py`. Measure cube Z position relative to
grip site throughout the full ascent.

---

## Risk 4: grasp_z Config Cache

**The Problem:**
After changing `grasp_z` from 0.430 to 0.425 in `env.yaml`, if the Python
environment has cached the old config (e.g., module-level `_cfg = load_config("env")`
in `waypoints.py`), the old value will still be used.

**Mitigation:**
- `waypoints.py` loads config at module level (`_cfg = load_config("env")`). After
  changing `env.yaml`, the Python process must be restarted — no in-process reload.
- Add an assertion in `test_grasp_physics.py` that verifies `uw.GRASP_Z == 0.425`
  before running tests.

---

## Risk 5: Teleport Mode Active During Grasp

**The Problem:**
If `grasp_mode=False` when `execute_grasp()` runs the finger close loop, `_set_action`
will teleport the finger to `FINGER_CLOSED_JOINT=0.0` — which means instant 100%
closure against the cube. This causes the Z-explosion described in Risk 1 even with
the soft physics, because the finger position is forced regardless of contact.

**Mitigation:**
`execute_grasp()` sets `self.grasp_mode = True` before the close loop. An assertion
at the start of the close loop verifies this:
```python
assert self.grasp_mode, "execute_grasp must set grasp_mode=True before finger close"
```

**Additional check:** Add a test in `test_grasp_physics.py` that deliberately runs
`execute_grasp()` with `grasp_mode=False` and verifies it detects the explosion.

---

## Risk 6: Descend Ends 9mm Above Cube

**The Problem:**
The DESCEND success threshold is 10mm. The arm can succeed with the grip site at
GRASP_Z + 9mm = 0.429m. At this height, the finger center is at 0.449m; the finger
bottom edge is at 0.449 - 0.0385 = 0.4105m. The cube top is at 0.430m.
Overlap = 0.430 - 0.4105 = 19.5mm out of 30mm (65%) — usable but not optimal.

**Mitigation:**
The Contact Approach sub-phase (Sub-Phase 1) of `execute_grasp()` scripted-descends
the final 9mm to GRASP_Z=0.425, ensuring 78% cube overlap and 6.5mm table clearance.
The grip site is guaranteed within 1mm of GRASP_Z before close begins.

---

## Risk 7: Arm Above Wrong XY Position

**The Problem:**
After DESCEND succeeds, the arm is within 10mm XY of the tube_center. The cube
is at src_xy. If the arm's XY is 9mm off from src_xy, and the cube is exactly at
src_xy, the gripper may be partially off-center. The cube may be grasped off-center,
leading to asymmetric hold and rotation during transit.

**Mitigation:**
- The GRASP precondition check requires `cube_xy_dist < 10mm` from grip site.
- A 9mm off-center grasp will still close both fingers on the cube (finger face width
  = 77mm, cube = 30mm, so fingers cover the cube even with 9mm offset).
- Post-grasp metrics track the offset and flag episodes where centering was poor.

---

## Risk 8: Wrist Snap on Cube Pickup

**The Problem:**
When the arm begins ascending with a 50g cube, the cube's inertia briefly resists
the upward force. This creates a momentary downward impulse on the wrist via the
contact force. The weld constraint must absorb this jerk. If `solref="0.01 1"` (10ms),
the impulse propagates to the wrist before the weld can absorb it, causing a snap.

**Mitigation:** `solref="0.02 1"` on the weld doubles the damping time constant.
The impulse is spread over 20ms instead of 10ms — below the perceptible jerk threshold.

---

## Risk 9: Torso Limit Hit During Transit

**The Problem:**
The `robot0:torso_lift_joint` has a hard limit at 0.3861m. When the Fetch arm
reaches its upper extent, hitting this limit generates a massive repulsive force
in MuJoCo that can destabilize the entire arm. This is a known issue documented
in the physics audit.

**When it can happen:** During transit at SAFE_Z=0.550, if the arm's kinematics
require the torso to be near its upper limit. This depends on the board position.

**Mitigation:**
- Add torso joint monitoring to `eval_grasp.py`:
  ```python
  torso_pos = env.unwrapped.data.qpos[
      env.unwrapped.model.jnt_qposadr[
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:torso_lift_joint")
      ]
  ]
  if torso_pos > 0.38:  # 1mm safety margin
      log.warning(f"Torso near limit: {torso_pos:.4f}m (limit=0.3861m)")
  ```
- No board position is excluded currently. If torso limit collisions become frequent,
  add a board region exclusion zone for the high-torso positions.

---

## Recommended Debugging Workflow If Physics Tests Fail

### Failure: Cube explodes (Z > GRASP_Z + 50mm)

1. Check `pick_and_place.xml` was saved and the process was restarted
2. Verify `uw.data.geom("object0").condim == 6` in the live model
3. Verify mass: `uw.model.body_mass[body_id] == 0.05`
4. If XML confirmed correct: reduce `GRASP_CLOSE_STEPS` velocity by adding a
   slower first-pass at 50% Kp (experimental tuning phase)

### Failure: Cube slides down during ascent

1. Check `grasp_mode=True` is set before ascend
2. Verify `_set_action` is NOT teleporting fingers when `grasp_mode=True`
3. Check that Kp=150,000 in the loaded model (not 60,000)
4. Add `print(uw.data.ctrl)` at each step to see if the actuator target is correct

### Failure: Cube rotates >15° during transit

1. Confirm `condim=6` on both cube and finger geoms in the live model
2. Check `friction[1]` (torsional) is non-zero (0.005)
3. If still spinning: increase torsional friction coefficient to 0.01

---

## Risk 13: Proximity Explosion During Halt Loop (Critical — ~5% of Full Pipeline)

**Status: PRESENT in current implementation (2026-05-05). Fix specified in doc 10 P1.1.**

**The Problem:**
After DESCEND succeeds, `execute_grasp()` runs a 50-step halt loop to dissipate residual arm
velocity. During those 50 steps, the arm is near GRASP_Z with residual velocity and oscillates
±2-3mm. At GRASP_Z, the finger bottom is at 0.4065m — already below the cube top (0.430m).
The finger overlaps the cube in Z while the fingers are still OPEN (gap=38mm).

MuJoCo detects proximity between the finger geoms and the cube geom. With `solref="0.002 1"` (very
stiff contact), even a brief proximity event generates an impulse. With residual arm velocity, this
impulse launches the cube before finger close begins.

**Evidence:** 3 of 5 explosion episodes in the 100-episode triage showed cube at >100mm from grip
site when `execute_grasp()` was called — cube was already launched DURING the halt loop, not during
finger close. `XY_MISALIGNMENT > 100mm` at pre-condition check is the symptom.

**Root Cause in Code:**
```python
# CURRENT (WRONG) ORDER:
for _ in range(50):  # Halt loop — arm oscillates with residual velocity
    self._set_action(dummy_action)
    self._mujoco_step(None)
# Then zero velocity AFTER the damage is done:
self.data.qvel[:ROBOT_DOF] = 0.0
self.data.qacc[:ROBOT_DOF] = 0.0
mujoco.mj_forward(self.model, self.data)
```

**The Fix (see doc 10 P1.1):**
```python
# CORRECT ORDER:
# 1. Zero velocity FIRST — arm is now stationary
self.data.qvel[:ROBOT_DOF] = 0.0
self.data.qacc[:ROBOT_DOF] = 0.0
mujoco.mj_forward(self.model, self.data)
# 2. Short settle — arm already stationary, just equilibrate weld (15 steps is enough)
for _ in range(15):
    self._set_action(dummy_action)
    self._mujoco_step(None)
# 3. Then contact approach (no oscillation → no proximity explosion)
```

**Why test_grasp_physics.py showed 100% success:** The `move_arm` function in the test script
uses smooth interpolation and runs until convergence. The arm is fully settled BEFORE execute_grasp
is called. There is no residual velocity. The halt loop has nothing to dissipate, so no oscillation
occurs and no proximity explosions happen. This explains why the physics test shows 100% on valid
positions while the full pipeline shows 88%.

**Distinction from CUBE_EXPLOSION_DURING_CLOSE:** That risk is finger-contact explosions during
the close loop itself. This risk (13) is proximity explosions during the halt phase BEFORE close.
Both contribute to the ~5% explosion rate in the full pipeline.

---

## Risk 14: Test Infrastructure Bug — Invalid Board Positions (Low — Test Accuracy Only)

**The Problem:**
`_sample_board_position()[:2]` in `test_grasp_physics.py` occasionally returns XY coordinates
outside the robot's reachable workspace (observed: X=4.13, X=-0.876, Y=-1.715, etc.). The cube
is placed there, falls off the table during arm settling, and the test fails with XY_MISALIGNMENT.

**Impact on Test Results:**
- In 30 trials: ~7 test-function failures from invalid positions (~7/90 = 7.8% rate)
- Reported: 27/30 static, 27/30 lift, 29/30 transit
- Reality: 100% physics success rate on valid positions
- The test is OVERSTATING the failure rate by ~10%

**Impact on Pipeline:** NONE. eval_grasp.py uses `force_cube_pos` from explicit XY coordinates,
not from `_sample_board_position()`. This bug only affects test_grasp_physics.py.

**Fix (see doc 10 P1.5):**
```python
BOARD_X_MIN, BOARD_X_MAX = 0.64, 1.12
BOARD_Y_MIN, BOARD_Y_MAX = 0.02, 0.50

for _ in range(10):
    src_xy = uw._sample_board_position()[:2]
    if BOARD_X_MIN <= src_xy[0] <= BOARD_X_MAX and BOARD_Y_MIN <= src_xy[1] <= BOARD_Y_MAX:
        break
```

---

## Risk 10: soft_reset FINGER_VALIDATION_FAILED (Critical — Will Crash Every Grasp)

**The Problem:**
The existing `soft_reset` implementation (task.py lines 496–503) ends with a
`FINGER_VALIDATION` block that verifies the finger position matches `finger_target_joint`
within 0.0005 tolerance. This is a sanity check for teleport mode. The code:
```python
if abs(l_pos - self.finger_target_joint) > 0.0005:
    raise RuntimeError("FINGER_VALIDATION_FAILED: ...")
```

**Why it crashes:**
After `execute_grasp()` sets `grasp_mode=True` and `finger_target_joint=0.0` (CLOSED),
the fingers physically stall at j≈0.014 (blocked by the cube). When
`soft_reset(GRASP→ascend)` runs its FINGER_VALIDATION:

```
abs(0.014 - 0.000) = 0.014 >> 0.0005 → RuntimeError on every episode
```

The cube is held correctly — the finger position is physically blocked by the cube,
not a software bug. The validation is checking the wrong condition for `grasp_mode`.

**The Fix:**
```python
# In soft_reset(), after Phase 4:
if not self.grasp_mode:  # Only validate in teleport mode
    if abs(l_pos - self.finger_target_joint) > 0.0005:
        raise RuntimeError(
            f"FINGER_VALIDATION_FAILED: finger={l_pos:.5f}, "
            f"target={self.finger_target_joint:.5f}"
        )
```

**Test:** `test_soft_reset_grasp_mode.py` verifies that soft_reset with grasp_mode=True
does NOT raise RuntimeError.

---

## Risk 11: step() Finger State Machine Opens Fingers During PLACE Descent

**The Problem (Future Stage — Document Now):**
During the PLACE sequence (Step 7 in the full chess move), the arm descends toward
`PLACE_Z` with `grasp_mode=True` (cube still held). The `step()` function has a
finger state machine:
```python
if self.current_scenario == "descend":
    self.finger_target_joint = self.FINGER_OPEN_JOINT
```
When `current_scenario == "descend"` AND `grasp_mode=True`, this sets
`finger_target_joint = FINGER_OPEN_JOINT = 0.0181` (open). In `grasp_mode`, `_set_action`
uses this as the ctrl target, commanding Kp × 0.0181 ≈ 2,715N OUTWARD per finger.
The cube drops immediately.

**Why it's safe for the current stage (PICK):**
During the PICK descend (`current_scenario="descend"`), `grasp_mode=False`. The
finger state machine opens the fingers (correct behavior for descend). No conflict.
The conflict only arises during PLACE descent where `grasp_mode=True`.

**The Fix (for the PLACE stage, not this stage):**
```python
if self.current_scenario == "descend" and not self.grasp_mode:
    self.finger_target_joint = self.FINGER_OPEN_JOINT
elif self.current_scenario != "descend":
    # ... existing logic for other scenarios
```

**Action:** Document this fix in the PLACE stage design. Do NOT implement in this stage.
The current code is correct for the PICK stage.

---

## Risk 12: HOME_XY Outside Transit Training Range

**The Problem:**
The transit model trains with start/goal positions sampled from:
```
low_x  = TABLE_CENTER_X - TABLE_HALF_X + EDGE_MARGIN = 0.88 - 0.28 + 0.04 = 0.640
high_x = TABLE_CENTER_X + TABLE_HALF_X - EDGE_MARGIN = 0.88 + 0.28 - 0.04 = 1.120
```
The original HOME_XY=`[0.600, 0.264]` has X=0.600, which is **40mm outside** this
training range. The transit policy has never been rewarded for navigating to or from
X=0.600. Placing Home there causes out-of-distribution behavior — the policy may
stall, overshoot, or fail to reach the goal.

**The Fix:**
Change `home_position_xy: [0.600, 0.2641]` → `[0.680, 0.2641]` in `configs/env.yaml`.
X=0.680 is 40mm inside the training boundary — well within the distribution.

**Verification:** After changing `env.yaml`, run `eval.py --scenario transit --n-episodes 50`
with `force_start_pos` set to HOME_POS and goals pointing into the board. Success rate
should match the nominal transit rate (≥95%).

---

## Recommended Debugging Workflow: New Risks

### Failure: RuntimeError "FINGER_VALIDATION_FAILED" during soft_reset

1. Confirm the error occurs at the GRASP→ASCEND transition (grasp_mode=True)
2. Apply the `if not self.grasp_mode:` guard to the FINGER_VALIDATION block
3. Re-run `test_soft_reset_grasp_mode.py` — must pass without exception
4. If error occurs at a DIFFERENT transition: check whether grasp_mode was
   accidentally left True from a previous episode (confirm `_reset_sim` resets it)

### Failure: Cube drops immediately at start of ASCEND

Possible cause: `step()` finger state machine opened fingers during the final
DESCEND step (Race condition: last descend step sets FINGER_OPEN, then execute_grasp
starts). Check:
1. `execute_grasp()` PRECONDITION_FINGERS_NOT_OPEN — if this fires, the finger
   state machine ran AFTER the grasp check
2. Ensure the execute_grasp() precondition check for finger position fires BEFORE
   the close loop (it does — but verify the order in your implementation)

---
*Next: [05 — Implementation Plan](./05_implementation_plan.md) (already written — refer back for code)*
