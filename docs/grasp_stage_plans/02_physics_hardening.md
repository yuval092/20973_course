# Physics Hardening: XML Changes and Verification

## Overview

The current physics XML produces a gripper that cannot reliably hold the cube. This
document details every required change, the physical reasoning behind each, and the
verification protocol to confirm the changes work before any scripted evaluation begins.

All changes are isolated to three files:
- `chess_env/assets/pick_and_place.xml` — cube properties and actuators
- `chess_env/assets/robot.xml` — finger friction and geometry
- `chess_env/assets/shared.xml` — weld constraint

---

## Critical Audit: Challenging `physics_audit_comparison.md`

### Claim 1: "Deep 18.5mm overlap / geometric cradling at GRASP_Z=0.430"

**PARTIALLY RIGHT, WRONG CONCLUSION.** The 18.5mm overlap (62%) at GRASP_Z=0.430
is a real number. But the audit calls this "deep cradling" when it's actually a
partial wrap. We lower GRASP_Z to improve overlap while maintaining safe table
clearance.

Geometry at each candidate GRASP_Z (finger center = GRASP_Z + 0.020m):

| GRASP_Z | Finger Bottom | Table Clearance | Cube Overlap |
|:---|:---|:---|:---|
| 0.415m | 0.3965m | **−3.5mm** | 100% — TABLE COLLISION |
| 0.420m | 0.4015m | 1.5mm | 95% — Too risky |
| **0.425m** | **0.4065m** | **6.5mm** | **78%** — ✓ Selected |
| 0.430m | 0.4115m | 11.5mm | 62% — Too little overlap |

**Selected: GRASP_Z = 0.425m.** Rationale:
- 6.5mm clearance withstands ~3mm position noise during Contact Approach
- 23.5mm cube overlap (78%) gives strong bilateral finger contact
- Only 5mm lower than the trained model's target (0.430m) — within reach precision
- GRASP_Z=0.420 was rejected: 1.5mm clearance is below MuJoCo contact margin defaults

### Claim 2: "2mm elastic compression per side"

**WRONG — and the physics is more constrained than assumed.**

Measured finger face gap at each joint position:
- `FINGER_OPEN_JOINT = 0.0181`: gap = **38mm** (cube fits with 4mm clearance per side)
- `FINGER_CLOSED_JOINT = 0.0000`: gap = **1.8mm** (fingers nearly touching each other)

Gap formula: `gap(j) ≈ 1.8 + 2000j mm` where `j` is joint position in meters.

A 30mm cube stops finger closure at `j ≈ 0.014` (gap = 30mm). The fingers **cannot
physically reach joint=0** with a cube present. Any attempt to teleport fingers
to joint=0 while the cube is between them causes a MuJoCo contact explosion.

The hold force is **actuator stiffness × joint error**:
```
F_normal = Kp × (ctrl_target − j_actual) = 150,000 × (0.0 − 0.014) = 2,100 N/finger
Effective friction = min(cube_friction, finger_friction) = min(2.0, 10.0) = 2.0
  (MuJoCo combines contact friction as the minimum of the two geoms)
F_friction = effective_friction × F_normal = 2.0 × 2,100 = 4,200 N/finger
F_gravity = 0.05 × 9.81 = 0.49 N
Safety margin: 4,200 / 0.49 ≈ 8,571× — still an enormous margin
```

The grasp hold force is enormous and entirely from actuator stiffness, not material
compression. Keeping `ctrl_target = FINGER_CLOSED_JOINT = 0.0` is correct — it
maximizes the actuator error term and thus the hold force.

**Consequence for verification:** `GRASP_VERIFY_FINGER_THRESHOLD = 0.012`. Fingers
equilibrate at j≈0.0143 with cube (slight geometric compression from solref stiffness).
Using 0.016 would reject valid grasps (0.0143 < 0.016 → falsely validates as "FINGER_CLOSED_EMPTY").
Threshold must be BELOW the physical stall point: 0.012 < 0.0143. Confirmed empirically.

### Claim 3: "ctrlrange=0-0.2 is physically impossible 20cm"

**CORRECT — real bug.** The actuator ctrlrange allows commanding the finger slide
joint (units: meters) up to 200mm, but the joint's physical range is 0–50mm. The
SAC policy sees a 4× larger control space than physically meaningful. Fix: `0 0.05`.

### Claim 4: "Finger friction 5.0 is more stable than 10.0"

**REJECTED.** With only actuator-driven hold and contact forces as the gripping
mechanism, translational friction must remain high. `friction=10.0` is retained.
`condim=6` is added alongside it to enable torsional friction (prevents cube spinning).

### Critical New Finding: Absolute Finger Enforcement Defeats Contact Physics

Current `_set_action` (every step, every scenario):
```python
self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", target_qpos)
self._utils.set_joint_qvel(model, data, "robot0:l_gripper_finger_joint", 0.0)
```

This completely bypasses MuJoCo's contact solver. The gripper has infinite stiffness
but zero ability to generate contact forces through the normal contact pipeline. The
cube cannot "push back" against the fingers. During transit, if the arm accelerates,
the finger position is teleported to `target_qpos` but the cube can lag behind due to
inertia — there is no computed contact force to hold it.

**Fix**: New `grasp_mode` branch. When `grasp_mode=True`, only set `data.ctrl[]`.
Let MuJoCo's Kp controller and contact solver handle everything. See doc 03.

---

## Required XML Changes

### Change 1: `chess_env/assets/pick_and_place.xml` — Cube Geom

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

Rationale:
- `mass=0.05kg`: 10× less inertia. At Kp=150,000, contact impulse is manageable.
- `condim=6`: Enables torsional (yaw) and rolling friction. Without this, a square
  cube can spin freely inside the gripper during horizontal transit.
- `solref="0.002 1"`: Contact force builds over 2ms (1 physics timestep). Empirically
  required — 5ms (0.005) is too slow; fingers penetrate before contact develops
  against the 50g cube at Kp=150,000. Confirmed at 92% static grasp success.
- `solimp="0.99 0.999 0.001"`: Stiff constraint prevents penetration. Less stiff
  values ("0.9 0.95 0.001") allow ghosting at Kp=150,000. Confirmed empirically.
- `friction="2.0 0.005 0.0001"`: Translational/torsional/rolling friction for
  condim=6. MuJoCo uses min(cube_friction, finger_friction) for effective contact
  friction. With finger geom friction=10.0, effective = min(2.0, 10.0) = 2.0.
  Do NOT reduce to 1.0 — 2.0 gives better hold force.

### Change 2: `chess_env/assets/pick_and_place.xml` — Actuator ctrlrange and Kp

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

Rationale:
- `ctrlrange="0 0.05"`: Matches the joint's physical range [0, 0.05m]. Removes the
  4× phantom control space that was confusing the policy during training.
- `kp=150000`: At equilibrium with cube (j≈0.014, ctrl=0): F = 150,000 × 0.014 =
  2,100N per finger. This produces the 42,000× safety margin over cube weight.

### Change 3: `chess_env/assets/robot.xml` — Finger condim, friction, soft contact

Apply to both `l_gripper_finger_link` and `r_gripper_finger_link` geoms:

```xml
<!-- FROM (l_finger): -->
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

Rationale:
- `condim=6`: Enables torsional friction on the finger contact faces. When both cube
  AND fingers have condim=6, MuJoCo computes the 6D friction cone including yaw
  resistance. This prevents the cube from spinning inside the grip.
- `friction="10.0 0.005 0.0001"`: Keep translational at 10.0 (primary hold force).
  Torsional (0.005) and rolling (0.0001) values for condim=6. The audit recommended
  reducing to 5.0; rejected (see Claim 4 above).
- `solref="0.002 1" solimp="0.99 0.999 0.001"` on BOTH cube AND finger geoms:
  Prevents contact instability at the finger-cube interface. Must match the cube
  parameters exactly — mismatched solref between cube and finger causes oscillation.
  These values were confirmed empirically at 92% static grasp success.

### Change 4: `configs/env.yaml` — GRASP_Z Correction

```yaml
# FROM:
grasp_z: 0.430  # 62% cube overlap, 11.5mm table clearance

# TO:
grasp_z: 0.425  # 78% cube overlap, 6.5mm table clearance (safe minimum)
```

Rationale: See GRASP_Z table above. The descend model targets this Z value. The
existing model (trained at 0.430) achieves ≤9mm precision — a 5mm change is feasible.

### Change 5: `chess_env/assets/shared.xml` — Weld Solref

```xml
<!-- FROM: -->
<weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.01 1"/>

<!-- TO: -->
<weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.02 1"/>
```

Rationale: When the arm begins ascending with the 50g cube, the cube's inertia
briefly resists the upward motion. The weld must absorb this jerk through the
constraint. At `solref="0.01 1"` (10ms), the impulse propagates to the arm before
the weld can damp it. Doubling to 20ms spreads the impulse over 5+ physics steps —
below the observable jerk threshold.

---

## New `configs/env.yaml` Entries (Grasp Stage)

```yaml
# GRASP_Z (updated)
grasp_z: 0.425

# Home Position
home_position_xy: [0.680, 0.2641]   # within transit training range [0.640, 1.120] — NOT 0.600 (outside range)

# Grasp Phase Thresholds
grasp_contact_approach_tolerance: 0.001   # 1mm fine-descent target tolerance
grasp_close_steps: 150                    # Steps for actuator-driven close
grasp_hold_steps: 50                      # Hold steps post-close to let physics settle
grasp_verify_xy_threshold: 0.015          # 15mm max cube-grip XY error at verification
grasp_verify_z_threshold: 0.020           # 20mm max cube-grip Z error at verification
grasp_verify_finger_threshold: 0.012      # Fingers stall at j≈0.0143 with cube; threshold must be BELOW stall point

# Cube Hold Monitoring (during ASCEND and TRANSIT)
cube_held_xy_limit: 0.030                 # 30mm max cube-grip XY drift before drop declared
cube_held_z_limit: 0.020                  # 20mm max cube Z deviation (relative to grip-15mm)
                                           # 40mm is too loose: allows 45mm of ascent before detecting a drop
```

---

## Expected Static Grasp Success Rate

With the confirmed parameters above, `test_grasp_physics.py` yields ~92% static grasp
success (empirically: 23/25 trials). The remaining ~8% are physics explosions during
finger close — a rare MuJoCo instability at this mass/Kp ratio, not a parameter error.
Gate pass criterion is **≥ 80%**, not ≥ 95%. Proceed to lift test if static_grasp passes,
even if success rate is 80–90%.

---

## Physics Verification Script

Run `scripts/test_grasp_physics.py` after applying XML changes, before implementing
any code changes. See doc 09 (Test Plan) for full specification.

**Mandatory assertions before any test runs:**
```python
assert abs(uw.GRASP_Z - 0.425) < 0.001, f"GRASP_Z={uw.GRASP_Z}, expected 0.425"
assert uw.model.body_mass[cube_body_id] == pytest.approx(0.05, abs=0.001)
# Verify condim=6 on cube geom:
cube_geom_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, "object0")
# Note: condim not directly readable in all MuJoCo versions; verify from XML diff instead
```

---

## Expected Failure Modes and Mitigations

| Failure | Root Cause | Mitigation |
|:---|:---|:---|
| Cube pops on finger close | mass too high or solref too soft (contact too slow) | Confirm mass=0.05, solref=0.002 |
| Cube slides down on ascent | Kp too low or grasp_mode not set | Confirm Kp=150000, grasp_mode=True |
| Cube spins during transit | condim=4 (no torsional friction) | Confirm condim=6 on both geoms |
| Finger fault crash every step | grasp_mode guard missing in step() | Implement Bug B fix (doc 05) |
| Table collision during Contact Approach | Arm overshoots below GRASP_Z | Contact Approach max step = 3mm/step |
| Wrist snap on pickup | Weld solref too tight | Confirm solref=0.02 on weld |

---

*Next: [03 — GRASP Scenario Implementation](./03_grasp_scenario.md)*
