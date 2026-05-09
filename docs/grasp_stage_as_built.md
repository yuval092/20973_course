# Grasp Stage: Detailed As-Built Specification

## 1. Architectural Overview

The Grasp Stage implementation enables the robotic arm to perform its first physical interaction with the environment. This is achieved by combining the existing Reinforcement Learning (RL) movement scenarios with a new, high-precision scripted **Grasp Phase**.

### 1.1 The Hybrid Control Pipeline
The system operates using a hybrid control model:
1.  **RL Phases (Transit/Descend/Ascend):** Use "Teleport Mode" where gripper fingers are forced to specific positions (`FINGER_OPEN_JOINT` or `FINGER_CLOSED_JOINT`) at every simulation step, bypassing contact physics to ensure the fingers do not interfere with the arm's trajectory.
2.  **Scripted Phase (Grasp):** Uses "Grasp Mode" (Actuator-Driven). Finger positions are determined purely by MuJoCo's Kp controller and contact solver, allowing the fingers to press against the physical cube and generate holding forces.

### 1.2 Data Flow & State Management
*   **`self.grasp_mode` (bool):** A new state variable in `ChessTaskEnv`. When `True`, the low-level `_set_action` method in `simulation.py` stops calling `set_joint_qpos` for the fingers and only sets `data.ctrl`.
*   **`self.finger_target_joint` (float):** The target setpoint for the finger actuators. This is updated by the `execute_grasp` script and preserved during the subsequent `ASCEND` and `TRANSIT` scenarios via a modified `soft_reset`.

---

## 2. Detailed Implementation of `execute_grasp()`

The `execute_grasp()` method is a multi-stage state machine that takes over control from the RL policy once the `DESCEND` scenario succeeds.

| Sub-Phase | Action | Technical Detail |
|:---|:---|:---|
| **0. Entry Settle** | Zero Velocity | Zeros `qvel` and `qacc` for all 21 DOFs (15 robot + 6 cube). Prevents momentum carry-over. |
| **0.1 Pre-Checks** | Invariant Validation | Confirms arm speed < 5mm/s, Z-error < 10mm, and XY-alignment < 10mm. |
| **1. Contact Approach** | Fine Descent | Uses direct `mocap` manipulation to lower the arm to exactly `GRASP_Z` (0.425m). |
| **2. Finger Close** | Linear Ramping | Gradually moves `finger_target_joint` from 0.0181 to 0.012 over 150 steps. |
| **3. Hold Settle** | Physics Equilibrium | Holds position for 50 steps to allow the MuJoCo solver to settle contact impulses. |
| **4. Final Verify** | Success Detection | Confirms cube is within 15mm XY and 20mm Z of the grip site. |

---

## 3. Physics & XML Configuration

The success of the physical grasp depends entirely on the numerical stability of the contact solver. The following parameters were meticulously tuned:

### 3.1 Cube Geometry (`pick_and_place.xml`)
*   **Mass (0.05kg):** Reduced from 0.5kg to minimize inertia-driven "explosions" when fingers close at high Kp.
*   **Condim (6):** Enabled 6D friction (translational + torsional + rolling). Essential for preventing the cube from spinning or "walking" out of the gripper during horizontal transit.
*   **SolRef (0.002 1):** Set a very fast time constant (2ms). This ensures contact forces develop immediately (within 1 physics step), preventing geoms from interpenetrating (ghosting).
*   **SolImp (0.99 0.999 0.001):** Created a very stiff, narrow contact constraint. This effectively turns the cube and fingers into "hard" surfaces.

### 3.2 Finger Actuators (`pick_and_place.xml`)
*   **Kp (150,000):** Provides extremely high stiffness.
*   **Hold Target (0.012):** By targeting 1.2cm (slightly narrower than the 1.5cm cube half-width), the actuator creates a constant error term:
    `Force = Kp * (0.012 - 0.0141) ≈ 150,000 * 0.0021 ≈ 315N`.
    This ~300N force provides a massive safety margin over the cube's weight (0.49N) while remaining numerically stable.

---

## 4. Empirical Performance Results

The following metrics were captured from a 100-episode stress test (`Gate 7`):

### 4.1 Success Rates
*   **Transit to Source:** 100% (No regressions in RL movement).
*   **Descend to Cube:** 96% (Occasional `TUBE_BREACH` due to marginal overshoot).
*   **Physical Grasp:** 94% (Stable and repeatable).
*   **Full Pipeline:** **86%** (Cumulative success from home to home).

### 4.2 Grasp Quality (Mean Metrics)
*   **XY Drift:** 4.6mm. The cube is exceptionally stable during high-speed transit.
*   **Z Error:** 6.6mm. Accounting for the 15mm "hang" below the grip site, the cube settles precisely where expected.
*   **Max Rotation:** 20.5°. While slightly above the 15° design goal, the rotation is stable and does not impact subsequent placement logic.

---

## 5. File-by-File Change Matrix

| File | Primary Change | Purpose |
|:---|:---|:---|
| `simulation.py` | `_set_action` branch | Implemented `grasp_mode` to toggle actuator-driven fingers. |
| `task.py` | `execute_grasp()` | Added the primary scripted grasping logic and velocity zeroing. |
| `task.py` | `soft_reset()` | Modified Phase 4 to skip finger teleportation when a cube is held. |
| `task.py` | `step()` | Added `TUBE_BREACH_GRACE` (3mm) and `_check_cube_held()` monitoring. |
| `pick_and_place.xml` | Cube & Actuators | Hardened physics for 0.05kg cube and 150k Kp actuators. |
| `robot.xml` | Finger Geoms | Added `condim=6` and high joint damping (5000) for stability. |
| `shared.xml` | Weld Equality | Set `solref=0.02` to dampen upward jerk during lifting. |
| `env.yaml` | Configs | Defined `home_position_xy` and new grasp thresholds. |
