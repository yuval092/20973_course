# Phase 9: Contextual Hybrid RL Implementation Master Plan

This document is the definitive, step-by-step execution guide for implementing the Phase 9 RoboChess Fine-Tuning.

*For the theoretical background, Q&A, and physics assumptions, see `phase9_architecture_and_rationale.md`.*
*For how this maps to the overall chess movement flow, see `phase9_waypoint_mapping.md`.*

---

## STEP 1: Physical Environment Modifications (XML)
**Target File:** `chess_env/assets/pick_and_place.xml`
**Goal:** Adjust the physical properties of the object to match a 2x2x2 cm chess piece with high friction.

**Required Actions:**
1.  **Locate:** The `<geom>` tag inside `<body name="object0">`.
2.  **Modify Size:** Change `size="0.025 0.025 0.025"` to `size="0.01 0.01 0.01"`. (MuJoCo uses half-extents, so 0.01 = 2cm total width).
3.  **Modify Friction:** Add the attribute `friction="2.0 0.05 0.0001"` to the same `<geom>` tag to prevent the piece from sliding when placed.

---

## STEP 2: The Contextual Environment - Core Structure
**Target File:** `chess_env/chess_fetch_dense_env.py`
**Goal:** Create the new environment class that handles the 3 distinct training scenarios (Air Transit, Elevator Down, Elevator Up).

**Required Actions:**
1.  **Class Definition:** Create `ChessFetchDenseEnv` inheriting from `ChessFetchEnv`.
2.  **Define Constants:** Add the following class-level constants for absolute Z-heights:
    *   `TABLE_Z = 0.400`
    *   `CUBE_Z = 0.410`
    *   `GRASP_Z = 0.410`
    *   `RELEASE_Z = 0.415`
    *   `SAFE_Z = 0.550`
3.  **State Variables:** In `__init__`, initialize tracking variables:
    *   `self.current_scenario = None` (Will hold 'transit', 'descend', or 'ascend').
    *   `self.tube_center_xy = None` (Will hold the `[X, Y]` coordinate for tube constraints).
    *   `self.gripper_forced_state = None` (Will hold 'open', 'closed', or 'ignore').
    *   Add an optional `force_scenario=None` argument to `__init__` for evaluation purposes.

---

## STEP 3: The Contextual Environment - Scenario Generation (`_reset_sim`)
**Target File:** `chess_env/chess_fetch_dense_env.py`
**Goal:** Override the reset function to randomly select a scenario and perfectly teleport the arm and object into the correct starting positions.

**Required Actions:**
1.  **Override `_reset_sim(self)`:** 
    * If `self.force_scenario` is set, use it. Otherwise, randomly select from `['transit', 'descend', 'ascend']`.
2.  **Sample Coordinates:** Generate two random valid board coordinates: `start_xy` and `goal_xy`.
3.  **Scenario Logic Switching:**
    *   **IF 'transit':**
        *   `self.tube_center_xy = None`
        *   `arm_start_pos = [start_xy[0], start_xy[1], self.SAFE_Z]`
        *   `self.goal_pos = [goal_xy[0], goal_xy[1], self.SAFE_Z]`
        *   `object_pos = [goal_xy[0], goal_xy[1], self.CUBE_Z]` (Out of the way)
        *   `self.gripper_forced_state = self.np_random.choice(['open', 'closed'])`
    *   **IF 'descend':**
        *   `self.tube_center_xy = start_xy`
        *   `arm_start_pos = [start_xy[0], start_xy[1], self.SAFE_Z]`
        *   `self.goal_pos = [start_xy[0], start_xy[1], self.np_random.choice([self.GRASP_Z, self.RELEASE_Z])]`
        *   `object_pos = [start_xy[0], start_xy[1], self.CUBE_Z]`
        *   `self.gripper_forced_state = 'open'`
    *   **IF 'ascend':**
        *   `self.tube_center_xy = start_xy`
        *   `arm_start_pos = [start_xy[0], start_xy[1], self.GRASP_Z]`
        *   `self.goal_pos = [start_xy[0], start_xy[1], self.SAFE_Z]`
        *   `object_pos = [start_xy[0], start_xy[1], self.CUBE_Z]`
        *   `self.gripper_forced_state = self.np_random.choice(['open', 'closed'])`
4.  **Execute Teleportation (Physics Safe - The "Phantom Settle"):**
    *   **Hide Object:** Temporarily teleport the object far away (e.g., `[2.0, 2.0, self.TABLE_Z]`) to prevent collisions during arm swing.
    *   **Move Arm:** Set the arm's target position (`self.data.mocap_pos[0] = arm_start_pos`).
    *   **Phantom Settle:** Run 10-20 "dummy" steps (`self.sim.step()` or `self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)`) to let the PID controller physically swing the arm to the target in empty space.
    *   **Zero Velocities:** Explicitly zero out all joint velocities (`self.data.qvel[:] = 0.0`) to kill any residual momentum from the swing.
    *   **Set Gripper Fingers:** If `forced_state` is 'open', set finger joints to `0.05`. If 'closed', set to `0.0` (or `0.01` if grabbing piece).
    *   **Restore Object:** Teleport the object back to its true `object_pos`.
    *   **Final Settle:** Call `mujoco.mj_forward(self.model, self.data)` to finalize the observation state without advancing time.

---

## STEP 4: The Contextual Environment - Dynamic Constraints (`step`)
**Target File:** `chess_env/chess_fetch_dense_env.py`
**Goal:** Intercept the action to enforce gripper state, execute the step, and calculate phase-aware rewards/penalties.

**Required Actions:**
1.  **Override `step(self, action)`:**
2.  **Action Interception (Gripper Masking):**
    *   Modify `action[3]` before passing it to MuJoCo.
    *   If `self.gripper_forced_state == 'open'`, set `action[3] = 1.0`.
    *   If `self.gripper_forced_state == 'closed'`, set `action[3] = -1.0`.
3.  **Execute Step:** `obs, reward, terminated, truncated, info = super().step(action)`
4.  **Extract State:** `gripper_pos = obs['observation'][0:3]`
5.  **Calculate Base Reward:**
    *   `dist_to_goal = np.linalg.norm(gripper_pos - self.goal_pos)`
    *   If `dist_to_goal < 0.01`: `reward += 10.0`, `terminated = True`.
    *   `reward -= 0.01 * np.linalg.norm(action[0:3])**2` (Smoothness penalty).
6.  **Apply 'Transit' Constraints (Virtual Floor):**
    *   If `self.current_scenario == 'transit'`:
        *   If `gripper_pos[2] < (self.SAFE_Z - 0.01)`: `reward -= 50.0 * ((self.SAFE_Z - 0.01) - gripper_pos[2])`
        *   If `gripper_pos[2] < 0.500`: `reward -= 10.0`, `terminated = True`.
7.  **Apply 'Elevator' Constraints (Virtual Tube):**
    *   If `self.current_scenario in ['descend', 'ascend']`:
        *   `drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)`
        *   `reward -= 20.0 * drift`
        *   If `drift > 0.025`: `reward -= 10.0`, `terminated = True`.
8.  **Return:** `obs, reward, terminated, truncated, info`.

---

## STEP 5: Stratified Evaluation Setup
**Target File:** `scripts/fine_tune_dense.py`
**Goal:** Create three independent evaluation environments to strictly monitor the success rate of each scenario.

**Required Actions:**
1.  **Environment Factories:** Create three separate functions to instantiate the environment:
    *   `make_eval_transit()` -> Env forced to 'transit'.
    *   `make_eval_descend()` -> Env forced to 'descend'.
    *   `make_eval_ascend()` -> Env forced to 'ascend'.
2.  **Vectorize:** Create three `DummyVecEnv` instances.
3.  **Callbacks:** Create three separate `SuccessRateEvalCallback` instances, one for each VecEnv. Log them to separate TensorBoard folders (e.g., `eval_transit`, `eval_descend`, `eval_ascend`).
4.  **Callback List:** Pass a `CallbackList([cb_transit, cb_descend, cb_ascend])` to `model.learn()`.

---

## STEP 6: Execute and Validate
**Commands:**
1.  `python scripts/visual_inspect.py` (Verify reset teleportation works without explosions).
2.  `python scripts/fine_tune_dense.py` (Begin the 300k step style-transfer run).
