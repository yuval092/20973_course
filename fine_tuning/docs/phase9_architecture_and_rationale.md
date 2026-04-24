# Phase 9: Architecture, Rationale, and Q&A

This document captures the theoretical background, discussions, assumptions, and physics constraints that led to the creation of the Phase 9 RL model training plan.

## 1. End-to-End RL vs. Hybrid Architecture
**Question:** Should we train the RL model end-to-end to handle the entire flow from cell A to cell B, including opening/closing the gripper, or split it into stages?
**Answer:** We must split it into stages (Hybrid Architecture).
**Rationale:** Teaching an RL agent to reliably learn a multi-stage sequence while avoiding 31 other dynamic obstacles (chess pieces) is notoriously difficult. It requires massive training time, and the model is prone to finding "reward hacks" or failing in edge cases. By using a Python script to enforce a deterministic sequence (Hover -> Descend -> Grasp -> Ascend -> Transit) and using RL *only* for point-to-point movement within that sequence, we guarantee 100% collision avoidance in transit.

## 2. Dynamic Constraints via Observation Space vs. "Virtual Tubes"
**Question:** Can we provide the RL model with another parameter that specifies constraints (e.g. `[min_x, max_x, min_y, max_y]`) as an input so it knows when it's constrained?
**Answer:** While brilliant (known as Contextual RL), changing the observation space breaks our existing, highly successful pretrained weights (`final_grandmaster_model.zip`). We would have to train from scratch.
**Rationale:** Instead, we implement "Virtual Tubes" and "Virtual Floors" implicitly in the environment's reward function. Because we explicitly set the Start and Goal states during `reset()`, the environment intrinsically knows if it's an "Air Move" or an "Elevator" move, and applies massive penalties for straying out of bounds. The RL model learns these constraints naturally without changing its neural network architecture.

## 3. The Gripper Conundrum
**Question:** When should the gripper be open or closed? Who changes its state? Can we train the arm without regard to the gripper?
**Answer:** The gripper is purely scripted by the Python controller (not the RL model). 
**Rationale:** The RL model will output a 4D action `[dx, dy, dz, gripper]`. We will intentionally **ignore** the 4th value. During training, the environment will force the simulated gripper to be open or closed depending on the scenario. This allows the RL model to learn to move the arm's mass perfectly, completely abstracted away from the finger mechanics.

## 4. Physics and Math Assumptions
### Board and Pieces
*   **Chess Cell Size:** 7x7 cm.
*   **Piece Size:** 2x2x2 cm. (Reduced from 5cm to allow the gripper to fit into a 7cm cell without knocking neighboring pieces).
*   **Clearance:** The gripper fingers are 1.4cm thick each (2.8cm total). Opening to 3cm to grab a 2cm piece creates a 5.8cm wide assembly. This leaves 0.6cm clearance on each side within a 7cm cell.
*   **Friction:** The cube's friction is increased to `2.0` (acting like felt on wood) so it doesn't drift when placed.

### Absolute Heights
*   `TABLE_Z = 0.400 m` (The immutable physical floor).
*   `CUBE_Z = 0.410 m` (Center of mass of the 2cm piece).
*   `GRASP_Z = 0.410 m` (Gripper center perfectly aligned with piece center).
*   `RELEASE_Z = 0.415 m` (5mm above resting height to ensure a "soft drop").
*   `SAFE_Z = 0.550 m` (Clearance altitude. Clears all standing pieces).

## 5. Edge Cases Addressed
1.  **The "Phantom Settle" (PID Swing Collision):** Setting the arm's target position (`mocap_pos`) doesn't instantly teleport the joints; the arm swings to it via a PID controller. If we spawn the arm and object together, the arm will violently smash the object during this initial swing, crashing the simulation. *Solution:* We must teleport the object off-board, let the arm settle into position for 10-20 dummy physics steps, zero its velocities, and *then* teleport the object back into the correct starting configuration.
2.  **"Elevator Up" Physics Explosion:** If we spawn the arm at `GRASP_Z` inside the cube, MuJoCo will throw a `NaN` error as the objects intersect. *Solution:* We must explicitly set the gripper finger joint `qpos` to be closed *around* the cube before calling `mj_forward`.
3.  **Inertial Overshoot:** When teleporting the arm to start an "Elevator Down" task, momentum from the previous episode could cause it to drift sideways and immediately fail the Virtual Tube constraint. *Solution:* We must explicitly zero out all `self.data.qvel` values during `_reset_sim()`.
4.  **Averaged Evaluation:** A generic 95% success rate is useless if the model is 100% good at Transit but 80% at Descend. *Solution:* We must use Stratified Evaluation—three separate evaluation environments locking the model into testing only Scenario A, Scenario B, or Scenario C independently.

## 6. Rewards and Penalties
*   **Success:** `+10.0`
*   **Smoothness:** `-0.01 * ||action||^2`
*   **Virtual Floor (Transit):** Continuous pain (`-50 * depth`) if dipping below `SAFE_Z - 0.01`. Terminal death (`-10.0`) if dipping below `0.500m`.
*   **Virtual Tube (Elevator):** Continuous pain (`-20 * drift`) pulling back to dead-center. Terminal death (`-10.0`) if drifting horizontally `> 0.025m` (breaching the 7x7 cell bounds).
