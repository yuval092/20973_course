# Review Findings: Scenario Chaining Plan

After a deep-dive review comparing the proposed chaining plan against the MuJoCo physics engine and the existing `src/chess_env` codebase, the architecture is confirmed to be highly viable. The "Nominal Alignment" strategy is the correct physical approach to solve error propagation.

During the review, three critical issues were identified and have been proactively fixed in the documentation:

### 1. The Observation Vector Desync (`self.goal`)
*   **The Flaw**: The original plan updated `self.goal_pos` but forgot to update `self.goal`. Because `_build_phase9_observation()` relies on `self.goal` to calculate the relative distance to the target, failing to update it would cause the RL model to see the *previous scenario's* target as its current goal.
*   **The Fix**: Explicitly added `self.goal = self.goal_pos.copy()` to Phase 4 of `soft_reset`.

### 2. The Gymnasium `TimeLimit` Wrapper
*   **The Flaw**: `gym.make(..., max_episode_steps=200)` wraps the environment in a `TimeLimit` class. While `soft_reset` resets the internal `episode_steps` counter, it did not reset the wrapper's `_elapsed_steps`. This would cause a 3-scenario chain to unexpectedly timeout at step 200 (across all scenarios).
*   **The Fix**: Added a wrapper traversal loop to explicitly reset `_elapsed_steps = 0` during Phase 4 of `soft_reset`.

### 3. The Alignment Target Bug
*   **The Flaw**: In Phase 2, `align_target` was originally set to `new_goal_pos.copy()`. If transitioning from Transit to Descend, the `new_goal_pos` is at `GRASP_Z`. This would cause the scripted alignment loop to drag the arm down to the bottom of the tube *before* the RL episode started, skipping the entire descent!
*   **The Fix**: `align_target` must use the *exit* waypoint of the current scenario. The signature for `soft_reset` was updated to accept `nominal_exit_pos`, which is passed from the external script using `exit_waypoint()`.

### 4. MuJoCo Joint Safety (`ROBOT_DOF = 15`)
*   **Confirmation**: The decision to slice `data.qvel[:15]` to zero the robot's velocity is technically perfect. The Fetch robot occupies indices 0-14 (slides, torso, head, arm, fingers). Index 15 is the object joint. By stopping at 15, the soft reset safely halts the arm without teleporting or freezing the chess piece, preserving its physical momentum.

**Status**: The Markdown files in `docs/next_stage_plans/` have been updated with these fixes. The plan is now bullet-proof and ready for implementation.
