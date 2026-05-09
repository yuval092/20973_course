# Architecture Overview: The "Staccato" Pipeline

## 1. The Design Philosophy
The goal of this project is high-precision chess piece manipulation. In robotic piece manipulation, **reliability and precision** are vastly more important than continuous, fluid motion. If the arm moves flawlessly 99% of the time but drops a piece 1% of the time, the entire chess game state is corrupted.

Therefore, we have chosen the **"Staccato" Pipeline** architecture. Instead of training a single, monolithic Reinforcement Learning (RL) model to perform an entire Pick-and-Place sequence, we decompose the sequence into isolated, specialized "Motor Primitives" (`Ascend`, `Transit`, `Descend`). 

These primitives are orchestrated by a deterministic **Master Controller**. The Master Controller enforces a "physical firewall" between each stage: the arm must come to a complete, motionless stop before the next stage is triggered. This prevents momentum and physics errors from bleeding over between tasks.

## 2. "Blind Actor, Omniscient Critic" Paradigm
The most critical architectural decision is how the RL model perceives the cube. We are using a "Blind Actor, Omniscient Critic" approach.

### The Blind Actor (The RL Model)
*   **Observation:** The RL model's observation space will **not** include the physical coordinates or velocity of the cube. The "Holding Object" trick is removed, and the model only sees its own gripper position, velocity, and the goal. 
*   **Why:** If the model sees the cube slipping, it will likely develop "twitchy", reactive behaviors to try and catch it. By keeping the Actor "blind," we force it to learn a single, smooth, open-loop motor trajectory that assumes the grasp is perfect.

### The Omniscient Critic (The Environment & Reward)
*   **Observation:** The simulation environment (`task.py`) always knows exactly where the cube is relative to the gripper.
*   **Why:** Even though the Actor is blind, the environment uses its omniscience to shape the reward. If the environment detects that the cube has slipped or dropped, it applies a massive terminal penalty. 
*   **The Result:** Over millions of steps, the SAC algorithm will backpropagate this penalty to the actions that caused the slip (e.g., high-frequency jitter or massive acceleration). The model learns that "fast, jerky movements equal sudden death," naturally converging on smooth, stable transport trajectories without ever needing to "look" at the payload.

## 3. System Components
1.  **MuJoCo Physics Engine:** Hardened and tuned to prioritize contact stability over absolute real-world accuracy.
2.  **RL Motor Primitives:** Three independently trained SAC models (or one multi-headed model) specializing in strictly vertical lifts (`Ascend`), horizontal moves (`Transit`), and strictly vertical drops (`Descend`).
3.  **The Sentry:** A background diagnostic layer running in the environment that constantly calculates the cube's position in the Gripper's Local Frame to detect microscopic shifts.
4.  **The Master Controller:** The Python script (`eval.py` or `game.py`) that manages the state machine, triggers the scripted grasp/release, and enforces "Stillness Validation" between RL stages.
