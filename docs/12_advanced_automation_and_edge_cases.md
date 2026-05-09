# Advanced Automation & Edge Case Analysis

This document provides a deep-dive re-evaluation of the Grasp Stage failure modes, exploring how scripted macro/micro automation can natively solve orientation, velocity, and reachability issues *without* altering the physical environment (i.e., no shrinking the board, no extending the arm).

---

## 1. The Core Breakthrough: The "Soft Weld" Fallacy

In the previous analysis, we identified that the arm drifting (causing `TUBE_BREACH`) and the wrist tilting (breaking "Crane Mode" at distant edges) were both symptoms of the softened `mocap` weld constraint (`solref="0.02 1"`).

**Why did we soften it in the first place?**
To mitigate "Risk 8: Wrist Snap on Pickup." When the RL policy suddenly applied an upward velocity vector to lift the 0.05kg cube, the instant change in inertia caused the wrist to snap. Softening the weld absorbed this jerk.

**The Automation Solution:**
If we implement a **Scripted Retract** phase immediately after grasping, we control the exact velocity profile of the lift. By using a smooth mathematical easing function (e.g., a cosine curve or small linear steps) to slowly lift the cube the first 35mm, we mathematically eliminate the sudden jerk/acceleration spike.

**The Result:**
Because the scripted retract removes the shock, **we no longer need the soft weld.** We can revert the weld in `shared.xml` back to being extremely stiff (`solref="0.005 1"` or `0.01 1`).

**Cascading Benefits (Fixing problems without changing the board):**
1.  **Crane Mode Restored:** With a stiff weld, the physical wrist cannot yield to kinematic strain. It will be forced to obey the strictly vertical `mocap` target. At extreme edges, the arm will fully extend rather than tilting the wrist to compromise.
2.  **RL Drift Eliminated:** A stiff weld eliminates the "lag" between the virtual target and physical arm. The RL `DESCEND` policy will return to its 100% success rate without hitting the 10mm `TUBE_BREACH` boundaries, because it was originally trained on a stiff weld.

---

## 2. Deep Dive: The New Automated Stages

To safely reach the edges of the board and handle the cube flawlessly, we replace the single `DESCEND` scenario with a 4-part automated sequence.

### Stage A: Macro-RL Hover
*   **Action:** The RL policy targets `HOVER_Z` (0.460m) instead of `GRASP_Z` (0.425m).
*   **Nuance:** The policy is kept 35mm above the physical cube. 
*   **Edge Case Addressed:** The RL policy is fundamentally noisy (±8mm precision). By keeping it at `HOVER_Z`, we eliminate any chance of a "proximity explosion" (the arm striking the cube before the fingers are ready) or a table collision, regardless of how fast or erratically the policy moves.

### Stage B: Scripted Perfect Align
*   **Action:** A proportional controller script takes over, driving the XY coordinates to perfectly match the cube's *actual* observed center, while maintaining `HOVER_Z`.
*   **Nuance:** The RL policy only aims for the abstract board coordinate. The `Perfect Align` script uses `env.get_cube_position()[:2]` to dynamically correct for any minor drift the cube experienced during the transit phase.
*   **Edge Case Addressed:** What if the arm sweeps other pieces off the board during this XY alignment? Because we are at `HOVER_Z` (0.460m), the finger bottoms are at 0.4415m. The tallest chess pieces (assuming 30mm height + 400mm table) reach 0.430m. The arm has >11mm of pure vertical clearance over *every* piece on the board. The XY sweep is 100% safe.

### Stage C: Scripted Plunge
*   **Action:** The script drives the `mocap` target straight down from `HOVER_Z` (0.460m) to `GRASP_Z` (0.425m) in 1mm increments.
*   **Nuance:** We strictly enforce the `VERTICAL_QUAT` at every step of this plunge.
*   **Edge Case Addressed:** Because the weld is now stiff, and the XY alignment was perfected in Stage B, the arm drops like a perfect, rigid piston over the cube. There is no horizontal drift during the plunge.

### Stage D: Scripted Grasp & Retract
*   **Action:** Close the fingers using the linear ramp (developed during testing), hold for 50 steps, then slowly reverse the Plunge (move `mocap` up 1mm per step) back to `HOVER_Z`.
*   **Nuance:** Once the arm reaches `HOVER_Z` with the cube in hand, we trigger `soft_reset` to hand control back to the RL `ASCEND` policy.
*   **Edge Case Addressed:** Handing control to the RL policy while touching the table is highly unstable. By completing the dangerous "breakaway" lift purely via script, we hand the RL policy an arm that is already floating safely in mid-air.

---

## 3. Problems & Blind Spots to Consider

Before finalizing this architecture, we must analyze potential weaknesses.

### Blind Spot 1: Kinematic Singularities at the Absolute Edges
*   **The Problem:** We decided *not* to shrink the board. This means the arm still has to reach X=1.12m. While the stiff weld forces the wrist straight (fixing Crane Mode), what happens if the arm physically *cannot* reach 1.12m without tearing itself apart in MuJoCo?
*   **The Nuance:** The Fetch arm is mounted at X=0.0. The center of the board is X=0.88m. The far edge is X=1.16m. The maximum physical reach of the Fetch arm (fully extended horizontally) is approximately 1.05m from its shoulder joint.
*   **The Critical Revelation:** *The arm literally cannot reach X=1.12m.* The reason it succeeded in previous stages is because the soft weld allowed the wrist to tilt forward, effectively "throwing" the virtual mocap target further out while the physical arm hung back.
*   **Conclusion:** If we enforce a stiff weld and perfect Crane Mode, the physics simulation *will* fail at the far edges of the board. **We have no choice but to shift the board closer to the robot.** (e.g., changing `table_center_xy` from `[0.88, 0.26]` to `[0.75, 0.0]`). We do not need to *shrink* the board, but we must translate its center to be within the true kinematic workspace.

### Blind Spot 2: Mocap-to-Body Desync during Scripting
*   **The Problem:** In our scripts, we update `data.mocap_pos`. The MuJoCo physics engine then applies forces to pull the physical body to that position. 
*   **The Nuance:** If our script moves `mocap_pos` by 5mm in a single step, the physical body doesn't instantly appear there. It accelerates towards it. If we run `mj_step` only once per script loop, the body will lag behind the mocap.
*   **The Solution:** We must use a **settle loop**. Whenever we change `mocap_pos` during a critical script (like Plunge), we must run a small `while` loop that checks the physical `grip_pos` and runs `mj_step` until the physical body actually arrives at the mocap target before moving the mocap target again.

### Blind Spot 3: The "Fingertip Stub" Edge Case
*   **The Problem:** During the `PLUNGE` stage, if the cube rotated heavily (e.g., 45 degrees) due to a previous collision, the diagonal width of a 30mm square cube is ~42.4mm.
*   **The Nuance:** The maximum opening of our gripper fingers is 38mm (`FINGER_OPEN_JOINT = 0.0181`). 
*   **The Result:** If the cube is rotated perfectly diagonally, the descending fingers will "stub" their toes on the top corners of the cube. The scripted plunge will push the arm down, crushing the cube into the table and likely exploding the physics.
*   **The Solution:** We *must* implement the **Rotation Abort Threshold** (P1.3 from the roadmap) during the `PERFECT ALIGN` stage. If the script detects the cube has yawed more than ~25 degrees, it must instantly abort the episode *before* the Plunge begins.

---

## 4. Final Architectural Verdict

By relying on scripted micro-manipulation for the dangerous phases, we can achieve 100% success without retraining the RL policy. 

However, your constraint to "avoid changing the board" is physically incompatible with enforcing perfect "Crane Mode." If the arm must point straight down, its reach is mathematically limited. 

**The Revised Plan of Action:**
1.  **Translate the Board:** We must update `env.yaml` to move the `table_center_xy` closer to the robot base to bring the far edge within the ~1.0m maximum vertical-reach radius. (No retraining is needed if we just shift the coordinates, as the observation space is relative to the arm).
2.  **Stiffen the Weld:** Revert `shared.xml` weld `solref` to `0.01 1` (or stiffer) to fix the RL drift and enforce rigidity.
3.  **Implement the 4-Stage Script:** Update `execute_grasp()` to perform the Hover, Perfect Align (with rotation abort), Plunge, Grasp, and Retract sequence.