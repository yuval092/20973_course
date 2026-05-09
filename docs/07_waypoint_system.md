# Waypoint System: Robotic Chess Move Execution

This document provides a detailed specification for the 10-stage waypoint system used to execute precise, collision-free chess moves. It coordinates high-level state logic with the low-level Reinforcement Learning (RL) movement engine.

## 1. Move Execution Stages

### Stage 1: Init in HOME_BASE
*   **Goal:** Establish a verified, stable starting state.
*   **Verification:**
    *   Confirm source and destination coordinates are valid and reachable.
    *   Verify the arm is at `HOME_BASE` coordinates.
    *   Verify **Vertical Orientation** is locked (Gripper pointing straight down).
    *   Verify the gripper is **Closed**.
    *   Monitor for several physics steps to ensure **Zero Velocity** and no drift.

### Stage 2: Transit to Source
*   **Goal:** Move horizontally to a position directly above the target piece.
*   **Scenario:** `transit`
*   **Target:** `[Source_X, Source_Y, SAFE_Z]`
*   **Verification:**
    *   **Timeout:** Exit with error if the target is not reached within the allotted steps.
    *   **On Arrival:** Open the gripper to **5.0 cm** (ensures safety clearance for 3cm pieces).

### Stage 3: Descend
*   **Goal:** Lower the open gripper to the picking altitude.
*   **Scenario:** `descend`
*   **Target:** `[Source_X, Source_Y, GRASP_Z]`
*   **Verification:**
    *   **Timeout:** Exit with error if destination is not reached.
    *   **Collision:** Exit with error if the arm touches the board, the target piece, or neighboring pieces.

### Stage 4: Grasp
*   **Goal:** Securely clamp the target piece.
*   **Action:** Close the gripper (apply and maintain constant maximum force).
*   **Verification:**
    *   **Stability:** Ensure the piece does not drift due to finger contact or excessive force.
    *   **Verification:** Compare the final finger joint positions against expected values for a 3cm cube to confirm a successful grasp.

### Stage 5: Ascend (with Piece)
*   **Goal:** Lift the secured piece to the cruise altitude.
*   **Scenario:** `ascend`
*   **Target:** `[Source_X, Source_Y, SAFE_Z]`
*   **Verification:**
    *   **Timeout/Collision:** Standard error handling.
    *   **Payload Check:** Periodically verify the piece is still held by checking its position relative to the `grip` site.

### Stage 6: Transit to Destination
*   **Goal:** Move the piece horizontally to the target cell.
*   **Scenario:** `transit`
*   **Target:** `[Dest_X, Dest_Y, SAFE_Z]`
*   **Verification:**
    *   **Timeout:** Exit with error if target is not reached.
    *   **Payload Check:** Confirm the piece hasn't fallen.

### Stage 7: Descend (Delivery)
*   **Goal:** Lower the piece to the board surface.
*   **Scenario:** `descend`
*   **Target:** `[Dest_X, Dest_Y, GRASP_Z + 0.005]` (The 5mm offset ensures a soft, collision-free delivery).
*   **Verification:**
    *   **Timeout/Collision:** Standard error handling.
    *   **Pre-Release Check:** Confirm the piece is still held and hasn't prematurely struck the board.

### Stage 8: Release
*   **Goal:** Place the piece on the board and disengage.
*   **Action:** Open the gripper to **5.0 cm**.
*   **Verification:**
    *   **Drift/Collision:** Ensure the piece stays centered in the destination cell during release.

### Stage 9: Ascend (Post-Release)
*   **Goal:** Clear the area for future moves.
*   **Scenario:** `ascend`
*   **Target:** `[Dest_X, Dest_Y, SAFE_Z]`
*   **Verification:**
    *   **Stability:** Verify the piece remains stationary in the destination cell and was not dragged or knocked during ascent.

### Stage 10: Transit to HOME_BASE
*   **Goal:** Return the arm to the parking position.
*   **Scenario:** `transit`
*   **Target:** `HOME_BASE`
*   **Action:** Close the gripper.
*   **Verification:**
    *   Standard timeout handling.

---

## 2. Technical Specs & Physics Analysis

### Gripper Geometry
Based on `robot.xml` analysis:
*   **Finger Length:** 7.7 cm (effective grasp range).
*   **Finger Thickness:** 1.4 cm.
*   **Tip Location:** 1.85 cm below the `grip` site center.

### Altitude Recommendations
*   **SAFE_Z (0.550m):** Provides 15 cm of clearance from the table, clearing all chess pieces.
*   **GRASP_Z (0.430m):**
    *   At this height, the gripper center is at the top of a 3cm cube.
    *   The fingers overlap with **1.5 cm** of the piece (top half).
    *   The fingertips are **1.15 cm** above the table, providing a robust safety buffer against board collisions.

### Handling Momentum and Stability
*   **Inter-Stage Settle:** Each stage MUST end with a "Physics Settle" phase. Hold the current position until the arm's linear velocity is **< 0.01 m/s** before transitioning to the next stage.
*   **Gripper Force:** Once Stage 4 starts, the system **must** send a continuous `closed` signal (-1.0) in every subsequent step to prevent the piece from slipping.

### Reward System Refinements
*   **Distance Reward:** The current weighting is optimal for stability.
*   **Universal Collision Penalty:** Colliding with *any* object (neighboring pieces, the board, or the target piece during transit) triggers a terminal `CRASH_PENALTY`.
*   **Transit Rigidity:** A Z-variance penalty (`-10.0 * abs(Z - SAFE_Z)`) is implemented during `transit` to enforce flat trajectories and prevent "U-shaped" moves.
*   **Stable Success:** The success bonus is granted once the arm is within 1cm of the goal and has stabilized below the velocity threshold.

### Solver Stability
*   **Friction Optimization:** To prevent pieces from "glitching" through the board, we use high-friction fingertips (`10.0`) but keep the board/table surface friction moderate. This ensures stable placement without solver-induced jitter.
