# Reward Shaping and Curriculum

This document explains the dense reward formula and the curriculum strategy used to train the RoboChess movement engine.

## 1. Dense Reward Formula
The model is trained using a dense reward function that provides feedback at every timestep. The total reward is the sum of five components:

1.  **Distance Reward:** `-1.0 * Distance to Goal`. Encourages the arm to move towards the target.
2.  **Z-Error Penalty:** `-0.5 * |Achieved Z - Desired Z|`. Extra penalty for height inaccuracy, crucial for vertical scenarios.
3.  **Braking Reward:** `-1.0 * Velocity * BrakingMask`. Active only when distance is `< 3cm`. Penalizes high speed near the goal to ensure stability.
4.  **Jitter Penalty:** `-0.003 * ||action||^2`. Encourages smooth, low-acceleration movements.
5.  **Safety Penalty:** `-0.5` per step if the arm is within `2.5 cm` of the table/floor surface.

### Terminal Bonuses/Penalties:
*   **Success Bonus:** `+500.0`. Awarded once when the arm is within `1cm` of the goal and velocity is `< 0.05 m/s`.
*   **Crash Penalty:** `-500.0`. Applied if the arm breaches a virtual tube or hits the table during a non-picking phase.

## 2. Drift Curriculum
The "Virtual Tube" constraint (3.5cm radius) is very strict and difficult for a fresh model to learn from scratch. We use a linear curriculum to bootstrap learning:

*   **Start Limit:** `10 cm`. Allows the model to explore and receive distance rewards before crashing.
*   **End Limit:** `1.5 cm`. (Note: This is the training target; evaluation uses `3.5 cm` for robustness).
*   **Duration:** `500,000 steps`. The tube radius shrinks linearly from `10cm` to `1.5cm` over the first half of the training run.

## 3. Training Hyperparameters (SAC)
The system uses the Soft Actor-Critic (SAC) algorithm with stabilized parameters:
*   **Learning Rate:** `5e-5`
*   **Batch Size:** `512`
*   **Target Entropy:** `-4.0` (for a 4D action space)
*   **Entropy Coefficient:** Starts at `0.1` and is auto-tuned during training.
