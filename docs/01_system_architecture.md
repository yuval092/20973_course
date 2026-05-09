# System Architecture: RoboChess Fine-Tuning

This document describes the high-level design and architectural decisions of the RoboChess robotic movement engine.

## 1. Hybrid Control Strategy
The movement system is designed as a **Hybrid Architecture** that combines deterministic Python state logic with Reinforcement Learning (RL) for point-to-point movement.

*   **State Machine:** A high-level controller (Master State Machine) dictates the sequence of operations: `Hover -> Descend -> Grasp -> Ascend -> Transit -> Deliver`.
*   **RL Engine (Cerebellum):** The SAC (Soft Actor-Critic) model is responsible for the actual 3D movement between these waypoints. It handles the low-level physics, inertia, and precision required to move the arm without colliding with the board.

## 2. The "Holding Object" Trick
To leverage pretrained weights from standard robotic tasks (like `FetchPickAndPlace-v1`) without the complexity of training grasping from scratch, we employ an architectural mapping trick:

*   **Observation Mapping:** In the 25-dimensional observation space, the `object_pos` (indices 3-5) is explicitly mapped to the current `grip_pos`.
*   **Rationale:** This convinces the RL model that it is **always** carrying an object. Consequently, the model uses its "Phase 2" transport logic, which is highly stable and optimized for moving a payload, even during the `descend` and `ascend` phases where no piece is actually held.

## 3. Configuration-Driven Design
The entire system is governed by a centralized YAML configuration system (`configs/*.yaml`). This ensures that:
*   Physical constants (e.g., table height, gripper width) are synchronized between the simulation and the training scripts.
*   Reward weights and success thresholds are easily tunable without modifying core Python code.

## 4. Scenario-Based Training (Pure Movement)
Training is split into three distinct scenarios to ensure specialized performance:
1.  **Transit:** High-altitude horizontal movement between board positions.
2.  **Descend:** Vertical movement from cruise altitude (`SAFE_Z`) to picking altitude (`GRASP_Z`).
3.  **Ascend:** Vertical lift from picking altitude to cruise altitude.

By training on these specific vectors, we achieve near 100% success rates on the individual components of a chess move.
