# Scenario Definitions: Training the Motor Primitives

The core of the "Staccato" pipeline is training three isolated SAC models, each specializing in one segment of a complete chess piece move. This approach drastically simplifies the observation space and speeds up convergence.

## 1. Domain Randomization: The "Clean Start" Delusion
A critical risk in specialized training is the "Clean Start" assumption. If the `Transit` model only ever trains starting from a perfectly centered, perfectly grasped cube, it will fail in production because the real `Ascend` phase will never deliver the cube perfectly.

**The Rule:** To make the "Blind Actor" robust, we must use Payload Domain Randomization. During `reset_sim` for `Transit` and `Descend`, the cube must be spawned inside the closed gripper with:
*   **Random XYZ Offset:** +/- 2mm from true center.
*   **Random Yaw Rotation:** +/- 5 degrees.

This forces the model to learn a stable transport path even if the grip is slightly "sloppy" from the previous stage.

---

## 2. Scenario 1: Ascend (The Lift)

**Goal:** Lift the piece straight up to the cruise altitude without lateral drift.

*   **Start State (`SAFE_Z` or `GRASP_Z`?):** The arm spawns perfectly settled at `GRASP_Z` directly over the cube. 
*   **The Scripted Grasp:** Before the RL model takes Step 0, a scripted ramp-up closes the fingers from `0.05m` to `0.0136m` over 20 physics steps. This is deterministic and invisible to the model.
*   **Target State:** The arm reaches `SAFE_Z` (0.55m).
*   **Success Condition:** The gripper $Z$ coordinate is within `0.01m` of `SAFE_Z`, and `is_held` remains True. (No velocity check required; vertical speed is fine).

---

## 3. Scenario 2: Transit (The Move)

**Goal:** Move horizontally across the board as fast as possible without losing the payload.

*   **Start State:** The arm spawns at `SAFE_Z`. The gripper is closed (`0.0136m`). The cube is teleported into the gripper using **Payload Domain Randomization** (see Rule 1).
*   **Target State:** A randomly sampled Goal XY coordinate at `SAFE_Z`.
*   **Success Condition:** The gripper XY coordinate is within `0.01m` of Goal XY, and `is_held` remains True.

---

## 4. Scenario 3: Descend (The Placement)

**Goal:** Lower the piece to the table surface and stop moving completely.

*   **Start State:** The arm spawns at `SAFE_Z` exactly over the target location. The gripper is closed. The cube uses **Payload Domain Randomization**.
*   **Target State:** The gripper reaches `GRASP_Z` (the piece touches the table).
*   **Success Condition (The "Stop" Rule):** This is the most dangerous phase. If the arm hits the table while moving fast, it triggers a MuJoCo explosion. Therefore, the episode is only successful if the arm reaches `GRASP_Z` **AND** its velocity is `< STABILITY_VEL_THRESHOLD` (0.05 m/s). 

---

## 5. The "Release" Phase (Un-Trained)
There is no `Release` RL model. Once `Descend` is successful, the Master Controller takes over. It initiates a scripted ramp-down, slowly opening the fingers from `0.0136m` to `0.05m` over 20 steps while maintaining `solref=0.02` to prevent "flicking" the cube. After release, the arm retracts to `SAFE_Z`.
