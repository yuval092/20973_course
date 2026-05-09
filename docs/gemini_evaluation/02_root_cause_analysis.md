# Root Cause Analysis: The Mechanics of Failure

Our investigation identified four distinct, interacting root causes for the failure of the chess piece lift.

## 1. Geometric Miscalculation (The "Gap" Bug)
The most critical blocker is the calculation of `GRASP_WIDTH_CLOSED` in `src/chess_env/task.py`.
*   **The Math:** The code targets a joint position of `0.0145m` to hold a 30mm cube.
*   **The Reality:** MuJoCo's finger geoms have an implicit offset. At a joint value of `0.0145m`, the actual distance from the finger surface to the center is `0.0154m`.
*   **Result:** A **0.4mm air gap** exists on both sides of the cube. Friction ($F_f = \mu \cdot N$) is mathematically zero because the normal force ($N$) is zero.

## 2. Settlement Precision (Initialization Noise)
The environment settlement loop uses a `settle_tolerance` of `0.003` (3mm). 
*   **Impact:** Because the model was trained on "Pure Movement" (where it always thinks it is holding the object), it expects perfect alignment at the start of an "Ascend" scenario.
*   **Result:** A 1-2mm error at step 0 means the gripper isn't even centered over the "ghost gap." As soon as the model applies its learned transport weights, it expects the object to follow its center. Since the physical cube is misaligned and not held, the gripper moves while the cube stays put.

## 3. High-Frequency Impulse (The "Hammer" Effect)
The SAC model produces "jitter" (high-frequency action changes). 
*   **Acceleration Spikes:** We measured lateral accelerations of **1.5 m/s²**. 
*   **Impulse Physics:** Because the simulation uses rigid contact parameters (`solref=0.005`), these accelerations act like a hammer. If the fingers *do* touch the cube (e.g., during a wiggle), they hit it with a high-velocity impulse rather than a steady grip.
*   **Result:** This explains why increasing friction in Document 02 led to "explosions." The rigid solver sees the overlap caused by jitter and resolves it with an infinite repulsive force.

## 4. Logical Misinterpretation (Coupled Weights Myth)
The previous investigation concluded that Z-movement is coupled to XY-jitter.
*   **Our Test:** By manually teleporting the cube to the gripper (Super Glue test), we proved the model **can reach 0.551m** and achieve "Success" with its current weights.
*   **Our Finding:** The "stalling" at 0.546m seen by the previous team was not a mathematical limit of the network; it was the result of the `is_success` condition failing (because the cube was left behind) and the episode terminating or the reward surface losing its gradient.
