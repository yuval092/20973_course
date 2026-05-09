# Automation and Reliability Architecture: The Macro/Micro Paradigm

## 1. Analysis of Current Failure Modes

Through rigorous evaluation of the Grasp Stage, we have identified a fundamental conflict between the capabilities of a Reinforcement Learning (RL) policy and the precision requirements of physical robotic interaction.

### 1.1 The "Crane Mode" and Weld Lag Issue
*   **Observation:** The physical arm visually tilts away from a perfect vertical "crane mode" when reaching for distant locations, despite the `mocap` target being strictly enforced as vertical.
*   **Root Cause:** In the previous stage, we "softened" the weld constraint connecting the virtual `mocap` body to the physical `gripper_link` (changing `solref` from `0.01 1` to `0.02 1`). This was done to absorb the shock of picking up the cube. However, a soft weld acts like a rubber band. When the arm moves fast (RL transit) or reaches near its kinematic limits, the physical arm "lags" behind the virtual target, causing drift and angular tilt.
*   **Impact:** This lag causes the RL policy (which was trained on a stiff weld) to fail its 10mm `TUBE_BREACH` limits, dropping `DESCEND` success rates from 100% to ~85%.

### 1.2 The "Collision Boundary" Issue
*   **Observation:** Relying on the RL policy to descend all the way to `GRASP_Z` (0.425m) is dangerous.
*   **Root Cause:** `GRASP_Z` is only 6.5mm above the physical table. The RL policy has a known precision variance of ±8mm. If the policy overshoots by even a few millimeters, it strikes the table, crashing the episode. If it oscillates near the object, it triggers "proximity explosions."

---

## 2. The Core Philosophy: Macro-RL vs. Micro-Scripting

To achieve 100% reliability, we must stop asking the RL policy to perform tasks it is not suited for (millimeter-perfect precision near collision boundaries). 

We will adopt a **Hybrid Macro/Micro Paradigm**:
*   **Macro-Navigation (RL):** The RL policy is used *only* for fast, generalized movement through safe, open space.
*   **Micro-Manipulation (Scripts):** Scripted proportional controllers take over for fine alignment, contact approach, and grasping.

---

## 3. Proposed Solutions & Architecture Updates

### Idea 1: The "Perfect Align" Script (Fixing Transit & Crane Mode)

Instead of relying on the RL policy to perfectly center the arm, we will introduce a dedicated scripted alignment phase that runs automatically at the end of macro-movements.

*   **How it works:** 
    1. The RL `TRANSIT` finishes when it gets "close enough" (within 10mm) to the target.
    2. We invoke a `scripted_align(target_xy, target_z)` function.
    3. **Dynamic Weld Stiffness:** During this script, we programmatically change the MuJoCo `solref` parameter of the weld constraint back to an extremely stiff value (e.g., `0.005 1`).
    4. We use a proportional controller to drive the *physical* `gripper_pos` (not just the mocap target) to exactly `target_xy` with < 0.5mm error, while the stiffened weld forces perfect vertical "crane mode."
*   **Feasibility:** Very High. We already use similar logic inside `execute_grasp`.
*   **Benefit:** Guarantees perfect starting conditions for the descend phase, completely mitigating the "weld lag" issue without needing to retrain the RL model.

### Idea 2: The "Hover & Plunge" Architecture

We will change the RL `DESCEND` target to insulate the policy from the table.

*   **Step 1: RL Hover.** Change the `DESCEND` scenario's target Z-height from `GRASP_Z` (0.425m) to a new safe `HOVER_Z` (e.g., 0.460m). 
    *   *Benefit:* At 0.460m, the arm is 60mm above the table. The RL policy can safely operate here with 100% success rate, zero risk of table collisions, and zero risk of proximity explosions with the cube.
*   **Step 2: Scripted Plunge.** Once the RL successfully reaches `HOVER_Z`, we invoke the "Perfect Align" script to ensure the arm is dead-center over the cube. Then, a scripted `PLUNGE` slowly drives the arm straight down the remaining 35mm to `GRASP_Z` using kinematic enforcement.
    *   *Benefit:* Scripted motion guarantees exactly vertical descent, zero oscillation, and perfect depth control.

### Idea 3: The "Clean Retract" Script

The inverse of the Plunge. After the `GRASP` or `PLACE` script finishes, handing control directly back to the RL policy for `ASCEND` is risky because the policy's first action might jerk the arm sideways while it is still near the table or other pieces.

*   **How it works:** After grasping, a scripted `RETRACT` moves the arm straight up from `GRASP_Z` to `HOVER_Z`. Only once the arm is safely at `HOVER_Z` does the RL `ASCEND` policy take over to move it the rest of the way to `SAFE_Z`.

---

## 4. Evaluation of Distant Reachability Solutions

The user noted the arm struggles to reach board edges.

*   **Idea A: Shrinking or Re-positioning the Board**
    *   *Feasibility:* Extremely High.
    *   *Implementation:* Adjust `table_half_x` and `table_half_y` in `configs/env.yaml` to create a smaller playable area perfectly centered in the robot's kinematic "sweet spot."
    *   *Impact on Training:* **Zero retraining required** if we only shrink the boundaries (the arm simply visits a subset of the coordinates it already knows).
    *   *Verdict:* **Highly Recommended.** This instantly removes kinematic strain and solves the root cause of the "crane mode" tilting.

*   **Idea B: Extending the Robotic Arm**
    *   *Feasibility:* Technically possible, but practically prohibitive.
    *   *Implementation:* Requires rebuilding the robot XML, recalculating inertia tensors, and manually tuning joint limits.
    *   *Impact on Training:* **Massive retraining required.** Changing limb lengths alters the entire kinematic chain. The current SAC policies would become entirely invalid.
    *   *Verdict:* **Rejected.** The effort-to-reward ratio is too low compared to simply shrinking the board.

---

## 5. The New 8-Step Pipeline

Incorporating these automation concepts, the Pick sequence evolves into a highly reliable 8-step pipeline:

```text
[1] TRANSIT (RL): Home → Hover over src_xy at SAFE_Z
        ↓
[2] PERFECT ALIGN (Script): Force exact XY centering, stiffen weld, enforce Crane Mode.
        ↓
[3] DESCEND (RL): SAFE_Z → HOVER_Z (0.460m)
        ↓
[4] PLUNGE (Script): HOVER_Z → GRASP_Z (0.425m) directly downward.
        ↓
[5] GRASP (Script): Close fingers.
        ↓
[6] RETRACT (Script): GRASP_Z → HOVER_Z directly upward.
        ↓
[7] ASCEND (RL): HOVER_Z → SAFE_Z
        ↓
[8] TRANSIT (RL): src_xy → Home at SAFE_Z
```

## Conclusion

By shifting precision tasks from the RL policy to dedicated, dynamic scripts (Perfect Align, Plunge, Retract), we can insulate the neural network from the physics boundaries that cause crashes. This "Macro/Micro" paradigm is the definitive path to achieving a 100% robust pipeline without needing to retrain the models.
