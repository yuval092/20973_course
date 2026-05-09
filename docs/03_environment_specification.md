# Environment Specification: Gymnasium Wrappers

This document details the software implementation of the `ChessFetchTask-v0` environment.

## 1. Class Hierarchy
1.  **`MujocoFetchPickAndPlaceEnv` (Base):** The standard Gymnasium Robotics environment.
2.  **`ChessSimulationEnv` (src/chess_env/simulation.py):** Extends the base to support the 64cm board, custom XML loading, and action scaling.
3.  **`ChessTaskEnv` (src/chess_env/task.py):** Implements the V6 movement logic, rewards, and scenario management.

## 2. The "Phantom Settle" Reset Sequence
Setting the arm's position directly in MuJoCo can lead to massive physics "explosions" if joints teleport through collision objects. To prevent this, `ChessTaskEnv` uses a multi-stage reset:

1.  **Object Teleportation:** The chess piece is moved to a stable hidden location (`[2, 2, 0.015]`) so it doesn't collide with the arm during its initial swing.
2.  **Gain-Based Settle:** The arm moves from its previous position to the new starting waypoint via a gain-based loop (`_settle_arm_to_start`). This runs for up to `500` steps until the error is `< 3mm`.
3.  **Velocity Purge:** Once settled, all joint velocities (`qvel`) and accelerations (`qacc`) are zeroed out.
4.  **Object Restoration:** The chess piece is moved back to the starting board coordinate (if required by the scenario).
5.  **Final Physics Settle:** The simulation runs for `25` steps with no actions to ensure gravity and contact forces are resolved.

## 3. Observation Space (25 Dimensions)
The environment returns a flat vector of 25 values, meticulously mapped to trick the model:
*   `[0-2]`: Gripper Pos (X, Y, Z).
*   `[3-5]`: **The Trick** - Object Pos (Forced to match Gripper Pos).
*   `[6-8]`: Object-to-Goal Relative Distance.
*   `[9-10]`: Gripper finger state (Masked).
*   `[11-13]`: Scenario One-Hot ID (`transit`, `descend`, `ascend`).
*   `[14-19]`: Object Velocity (Masked).
*   `[20-22]`: Gripper Velocity.
*   `[23-24]`: Gripper finger velocity (Masked).

## 4. Collision and Termination Logic
The environment monitors for two types of failures:
*   **Virtual Floor Crash:** In `transit`, the episode ends if the arm dips below `TABLE_Z`.
*   **Virtual Tube Breach:** In `descend/ascend`, the episode ends if the arm drifts laterally more than the current `DRIFT_LIMIT` or touches the table.

Success is defined as: `Distance to Goal < 1cm` **AND** `Gripper Velocity < 0.05 m/s`.
