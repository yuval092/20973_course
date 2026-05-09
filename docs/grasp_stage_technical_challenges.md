# Grasp Stage: Technical Challenges & Solver Dynamics

## 1. The "Ghosting" Phenomenon

### 1.1 Root Cause Analysis
The most significant challenge was the fingers passing through the cube (ghosting) during the initial `execute_grasp` phase. 

*   **Mechanical Cause:** High Actuator Force. With `Kp=150,000`, the fingers attempt to close the 18mm gap (open to cube surface) almost instantly.
*   **Solver Cause:** Large Penetration Depth. If the finger travels 5mm in a single physics timestep, it penetrates the cube by 4.5mm before MuJoCo's contact solver can detect the collision. 
*   **Result:** Once the penetration depth exceeds a certain ratio of the geom size, the solver's repulsive force can become inconsistent or even flip direction, causing the geoms to "tunnel" through each other.

### 1.2 Solution: Linear Control Ramping
We implemented a linear ramp for the `finger_target_joint` setpoint. 
*   Instead of jumping from 0.0181 to 0.012 in Step 1, the script increments the target by only `(0.0181 - 0.012) / 150 ≈ 0.00004m` per step.
*   This keeps the finger velocity extremely low (~2mm/s), allowing the contact solver to detect the collision at a depth of micrometers rather than millimeters.

---

## 2. The "Mocap Sag" Bug

### 2.1 Mechanical Analysis
During the `execute_grasp` phase, the arm would consistently sag ~10mm below the target `GRASP_Z`, often resulting in a table collision.

*   **Cause:** The standard `_set_action` logic was resetting the `mocap` position to the current *body* position at every step.
*   **Dynamics:** Gravity pulls the physical arm down. The `mocap` controller sees the body is lower, so it moves its target lower to "match" the body. This created a positive feedback loop of sagging.
*   **Correction:** Refactored `execute_grasp` to avoid `_set_action` during positional settle. By setting `data.ctrl` directly and leaving `data.mocap_pos` at the desired world coordinate, the mocap weld constraint pulls the arm *up* against gravity, maintaining a precise altitude.

---

## 3. Deviations from Implementation Plan

| Feature | Original Plan | Final As-Built | Rationale |
|:---|:---|:---|:---|
| **Close Target** | `0.0` (Full Close) | `0.012` (`HOLD_TARGET`) | Target 0.0 with 150k Kp created ~2000N force, which caused "jitter" and numerical explosions. 0.012 gives a stable, firm 300N hold. |
| **Drift Grace** | 1.5mm | **3.0mm** | The RL policy for `DESCEND` often lands exactly at the 10mm training boundary. 3mm grace prevents false-negative `TUBE_BREACH` crashes. |
| **Finger Damping** | Default (50) | **5000** | Required to critically damp the high-stiffness position actuators. Without this, fingers would vibrate against the cube. |
| **Halt Duration** | 50 Steps | **15 Steps** | Combined with **Immediate Velocity Zeroing**, 15 steps is more than enough for the weld constraint to equilibrate. |
| **Verify Finger** | 0.012 | **0.016** | The 1.5cm cube half-width + 2mm finger overlap stalls joints at ~0.0141. 0.016 is the correct "passed" threshold. |

---

## 4. Stability Observations

### 4.1 Numerical Oscillations
At `Kp=150,000`, the MuJoCo integrator is sensitive. We observed that increasing `solref` from `0.002` to `0.01` (slower contact) actually *worsened* stability because the geoms would overlap more before being pushed back. The "stiff" configuration (`0.002 1`) is the most robust, despite the light 0.05kg mass.

### 4.2 Torsional Friction
The transition to `condim=6` was critical. With the default `condim=4`, the cube was free to rotate along its Z-axis (yaw) during transit. Since the fingers are flat, any rotation reduces contact area and leads to a drop. `condim=6` enables a torsional friction component that locks the cube's orientation.
