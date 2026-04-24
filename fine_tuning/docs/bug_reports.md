# RoboChess Fine-Tuning: Deviations and Bug Reports

During the execution of the `robochess_gameplan_v2.docx` plan, we encountered several discrepancies between the written plan and the reality of the software versions, physics engine, and provided model weights. 

This document provides an in-depth comparison of our implementation against the original plan, detailing every deviation, bug, and the corresponding resolution to ensure a stable, 100%-reachable training environment.

---

## Part 1: Strategic Deviations from the Game Plan

The following changes were made where following the original game plan exactly would have resulted in failure, poor performance, or an inability to use a full-sized chess board.

### 1. XML Assets and Dependencies
*   **Original Plan:** "Copy only the `pick_and_place.xml` asset file into your own project folder." (Phase 1, Step 1.2)
*   **Reality:** MuJoCo XML files heavily utilize `<include>` tags. `pick_and_place.xml` relies on `shared.xml` and `robot.xml`, which in turn rely on mesh files in `stls/` and materials in `textures/`.
*   **Action Taken:** Copied the entire suite of XML files, `stls/`, and `textures/` from the installed `gymnasium_robotics` package into `chess_env/assets/` to prevent `MjModel.from_xml_path()` from crashing with file-not-found errors.

### 2. Table Z-Axis (Height) Dimension
*   **Original Plan:** Stated the original table size was `0.25 0.35 0.02` and instructed to change it to `0.25 0.25 0.02`. (Phase 2, Step 2.3)
*   **Reality:** The actual `pick_and_place.xml` in version 1.4.2 defines the table with a Z half-extent of `0.2` (i.e., `size="0.25 0.35 0.2"`). Changing this to `0.02` would have drastically lowered the table surface, breaking the robot's pre-trained approach trajectory.
*   **Action Taken:** Modified only the X and Y dimensions, resulting in `size="0.25 0.25 0.2"`. `TABLE_SURFACE_Z` in Python was correctly calculated as `0.4` (body pos Z `0.2` + geom half-extent Z `0.2`).

### 3. Model Filename and Missing Replay Buffer
*   **Original Plan:** Instructed downloading `sac_her_pnp.zip` and `replay_buffer.pkl` to warm-start the training. (Prerequisites)
*   **Reality:** As noted by the user, the actual filename on Hugging Face is `sac-FetchPickAndPlace-v4.zip`, and no replay buffer is hosted in the repository.
*   **Action Taken:** Updated the download scripts and loading paths to use the correct `.zip` filename. Removed the `model.load_replay_buffer()` call entirely.

### 4. Environment Class Name Change
*   **Original Plan:** Instructed inheriting from `FetchPickAndPlaceEnv`. (Phase 3)
*   **Reality:** In modern `gymnasium-robotics` (v1.4.2), the base class has been refactored and renamed to `MujocoFetchPickAndPlaceEnv` (located in `gymnasium_robotics.envs.fetch.pick_and_place`).
*   **Action Taken:** Updated the import path and class inheritance in `chess_env/chess_fetch_env.py` to use `MujocoFetchPickAndPlaceEnv`.

### 5. Reachability Troubleshooting (The Table Position)
*   **Original Plan:** "If reachability is below 90%: Reduce `TABLE_HALF_X` and `TABLE_HALF_Y`... by 0.02 m at a time until it passes." (Phase 4, Step 4.3)
*   **Reality:** The table was originally offset far to the right (`Y=0.75` while the robot is at `Y=0.2641`) and far forward (`X=1.3`). Shrinking the board would have defeated the goal of creating a full-size chess board. Furthermore, reaching sideways to a distant table put the robot arm at a kinematic singularity (fully extended).
*   **Action Taken:** Instead of shrinking the board, we **repositioned the entire workspace**. 
    *   **Y-Axis:** Aligned the table center (`Y=0.2641`) perfectly with the robot's base.
    *   **X-Axis:** Brought the table closer. We iterated through `X=1.3` $\rightarrow$ `X=0.75` $\rightarrow$ `X=0.65` (which caused base collisions and restricted backwards movement, dropping reachability to 78%) $\rightarrow$ and finally settled on **`X=0.85`**. 
    *   This elegant solution preserved the 0.5m x 0.5m board size while achieving **100% reachability** without any physical collisions between the arm and the table edge.

---

## Part 2: Technical Bug Reports

These are runtime errors encountered during execution and how they were fixed.

### Bug 1: Environment Checker Fails on `np_random` Update
*   **Symptom:** `gymnasium.utils.env_checker` raised `AssertionError: The .np_random is not properly been updated after step.`
*   **Cause:** The game plan placed the `MIN_GOAL_DIST` validation loop inside `_reset_sim()`. Because `self.goal` persists across episodes, the random block spawn position in episode $N$ depended on the randomly sampled goal from episode $N-1$. This broke the determinism of `reset(seed=...)`, triggering the Gymnasium checker's warning because the RNG state diverged unexpectedly.
*   **Resolution:** Moved the `MIN_GOAL_DIST` validation logic from `_reset_sim()` into `_sample_goal()`. Now, `_reset_sim()` samples exactly one random position unconditionally, guaranteeing that a given seed always yields the identical initial object state. `_sample_goal()` then loops until it finds a goal far enough from the object.

### Bug 2: TensorBoard ImportError During Training
*   **Symptom:** `model.learn(...)` crashed with `ImportError: Trying to log data to tensorboard but tensorboard is not installed.`
*   **Cause:** The pretrained model `sac-FetchPickAndPlace-v4.zip` from Hugging Face was saved with TensorBoard logging enabled. When `SAC.load()` restored the model, it also restored this configuration. Since `tensorboard` was not in our `requirements.txt`, the logger initialization failed.
*   **Resolution:** Explicitly disabled TensorBoard logging immediately after loading the model by setting `model.tensorboard_log = None` before calling `model.learn()`.

### Bug 3: RuntimeError on Replay Buffer Sampling
*   **Symptom:** `model.learn(...)` crashed immediately with `RuntimeError: Unable to sample before the end of the first episode. We recommend choosing a value for learning_starts...`
*   **Cause:** We loaded a pretrained model (which already had ~250,000 `num_timesteps` stored internally) but did *not* load its corresponding replay buffer (because it doesn't exist). The default `learning_starts` parameter is typically 100. Because `250,000 > 100`, Stable-Baselines3 immediately attempted to sample from the empty replay buffer on step 1 of fine-tuning, instead of collecting new data first.
*   **Resolution:** Dynamically adjusted the `learning_starts` threshold to account for the pretrained history by setting `model.learning_starts = model.num_timesteps + 1000`. This forces the agent to collect 1,000 steps of new experience on the modified, centered board before resuming gradient updates.

### Bug 4: The "Phantom Table" / Hardcoded Home Position
*   **Symptom:** Even after moving the table, the robot would start every episode reaching out into thin air at `X=1.3, Y=0.75`. This wasted several seconds of every episode as the robot travelled to the new board location, causing many corner moves to fail due to time limits.
*   **Cause:** The `gymnasium_robotics` Fetch environments have a hardcoded `initial_qpos` (initial joint configuration) that anchors the robot's slides at the original table location.
*   **Resolution:** Overrode `self.initial_qpos` in the `ChessFetchEnv.__init__` method. Set the X-slide to `0.10` and the Y-slide to `0.00`. This centers the robot perfectly over our new board at the start of every episode. This single fix increased corner success from **66% to 100%**.
