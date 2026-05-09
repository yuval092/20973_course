# Physics and Kinematics: RoboChess Ground Truth

This document defines the physical constants and coordinate system used in the MuJoCo simulation. All values are sourced directly from `configs/env.yaml` and `configs/physics.yaml`.

## 1. Coordinate System
*   **Origin (0,0,0):** The center of the robot's base.
*   **X-Axis:** Forward movement towards the board.
*   **Y-Axis:** Lateral movement (left/right).
*   **Z-Axis:** Vertical altitude.

## 2. Table and Board Geometry
*   **Table Surface (TABLE_Z):** `0.400 m`. This is the physical floor of the simulation.
*   **Table Center:** `[0.88, 0.2641]`.
*   **Board Dimensions:** `64 cm x 64 cm` (implemented as half-sizes of `0.28m` with margins).
*   **Chess Piece (CUBE_HEIGHT):** `3 cm` cube.
*   **Piece Resting Z:** `0.415 m` (calculated as `TABLE_Z + CUBE_HEIGHT/2`).

## 3. Waypoint Altitudes
To ensure collision-free movement across the board, the system uses two primary altitudes:
*   **SAFE_Z (Cruise Altitude):** `0.550 m`. At this height, the arm is guaranteed to clear all standing chess pieces (even the King).
*   **GRASP_Z (Picking Altitude):** `0.430 m`. This is the height required for the gripper to safely surround a 3cm piece without smashing the table.

## 4. Operational Constraints
### 4.1 Verticality Constraint
The gripper must always point straight down to successfully grasp pieces. This is enforced via the `VERTICAL_QUAT` constant:
*   `VERTICAL_QUAT = [1.0, 0.0, 1.0, 0.0]` (Normalized in simulation).

### 4.2 The "Virtual Tube" (Drift Limit)
During `descend` and `ascend` scenarios, the arm is constrained to a radial cylinder (tube) to prevent it from knocking over neighboring pieces.
*   **Radial Limit:** `0.035 m` (3.5 cm).
*   **Rationale:** Standard chess cells are 7cm x 7cm. A 3.5cm radius ensures the gripper never leaves the boundaries of the target cell.

### 4.3 Movement Scaling
To prevent physics instability (PID overshoot), actions are scaled by `pos_ctrl_scale = 0.015`. This limits the maximum displacement to **1.5 cm per step**.
