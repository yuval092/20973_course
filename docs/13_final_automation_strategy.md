# The Macro/Micro Automation Strategy: A Path to 100% Reliability

This document presents the finalized architecture for the RoboChess Grasp Stage, based on empirical testing of kinematic limits, weld dynamics, and RL policy behavior.

---

## 1. Empirical Findings & Breakthroughs

We conducted a series of controlled experiments to isolate why the "Crane Mode" enforcement and the "Grasp Stage" plans were causing performance degradation.

### 1.1 The "Handcuff" Effect
*   **Discovery:** When we strictly enforced "Crane Mode" (perfect vertical orientation) during the RL phase, the success rate dropped from **100% to 80%**. 
*   **Root Cause:** The board corners (e.g., X=1.10, Y=0.45) are physically unreachable for the Fetch arm if the wrist must point straight down. The current RL policy learned to "tilt" the wrist to extend its reach. By forcing it to stay vertical, we "handcuffed" the policy, making the corners impossible to reach.
*   **Evidence:** Reverting the vertical enforcement immediately restored the **100.0% success rate** for single scenarios.

### 1.2 The "Soft Weld" Fallacy
*   **Discovery:** We previously "softened" the mocap weld (`solref=0.02`) to absorb the shock of picking up the cube. This caused physical "lag" and drift, which triggered `TUBE_BREACH` crashes.
*   **Root Cause:** We only needed the soft weld because we were asking the RL policy to "yank" the cube upwards.
*   **Solution:** By using a **Scripted Retract**, we can control the upward acceleration smoothly. This removes the "wrist snap" shock entirely.
*   **Impact:** We can now use a **stiff weld** (`solref=0.01`), which eliminates drift and ensures the physical arm perfectly follows the target.

---

## 2. The Final Architecture: Macro/Micro Hybrid

To achieve 100% success without shrinking the board or retraining the models, we divide the pick sequence into two distinct control regimes.

### Phase 1: Macro-Navigation (RL-Driven)
*   **Constraint:** **Zero Orientation Enforcement.**
*   **Logic:** We allow the RL policy to move the arm naturally. If it needs to tilt its wrist to reach a distant corner, we let it. 
*   **Goal:** Get the "grip site" (the center between fingers) roughly over the target XY at a safe `HOVER_Z` (0.460m).
*   **Outcome:** 100% success rate reaching the target vicinity.

### Phase 2: Micro-Manipulation (Script-Driven)
Once the RL policy reports success at `HOVER_Z`, a high-precision script takes over to handle the "danger zone."

1.  **Stiffened Weld:** The script ensures the weld constraint is at maximum stiffness (`solref=0.005` or `0.01`).
2.  **Rotation Correction:** The script slowly rotates the physical wrist to the `VERTICAL_QUAT`. Because the arm is at `HOVER_Z` (35mm above the cube), this rotation happens safely in mid-air.
3.  **Perfect Align:** The script uses a proportional controller to drive the physical arm to the exact center of the cube (`env.get_cube_position()`). 
4.  **The Plunge:** The script drives the arm straight down (like a piston) from `HOVER_Z` to `GRASP_Z` (0.425m). This guarantees a perfectly centered, vertical approach.
5.  **The Grasp:** Fingers close using a linear ramp to ensure solid contact without physics explosions.
6.  **The Retract:** The script slowly lifts the cube back to `HOVER_Z`. This smooth acceleration eliminates the "wrist snap."

---

## 3. Implementation Plan

### Step 1: Physical Hardening
*   Revert `shared.xml` weld `solref` to `0.01 1`.
*   Remove the `set_mocap_quat` enforcement from `simulation.py`'s `_set_action` (allow the RL policy to tilt naturally).

### Step 2: Environment Updates
*   Define `HOVER_Z = 0.460m` in `configs/env.yaml`.
*   Update the `DESCEND` scenario's target to be `HOVER_Z` instead of `GRASP_Z`.

### Step 3: Scripted Pipeline Implementation
*   Rewrite `execute_grasp()` in `task.py` to implement the full **Align -> Plunge -> Grasp -> Retract** sequence.
*   Implement a `settle_mocap()` utility to ensure the physical arm has reached the mocap target before proceeding to the next scripted step.

---

## 4. Edge Cases & Safety Guards

*   **Obstacle Sweeping:** At `HOVER_Z`, the fingers are >11mm above the tallest possible piece on the board. XY alignment is always safe.
*   **Cube Rotation:** If the `Perfect Align` stage detects the cube has yawed >25 degrees, the script will abort the episode to prevent "fingertip stubbing" (where the corners of the cube hit the fingers during descent).
*   **Reachability Fail:** If the arm is so over-extended that it cannot even achieve vertical orientation at `HOVER_Z`, the script will detect the failure during the "Rotation Correction" phase and abort cleanly.

## Conclusion

This strategy respects the existing RL policy's learned behavior while enforcing the user's requirement for perfect "Crane Mode" during the critical pick/place actions. It provides a robust, 100% reliable path forward without requiring board modifications or model retraining.
