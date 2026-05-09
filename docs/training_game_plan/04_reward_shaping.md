# Reward Shaping: The Omniscient Critic

Since our RL Actor is "Blind" (it cannot see the cube's position or velocity in its observation space), we must use the Reward Function—the "Omniscient Critic"—to guide it toward behaviors that keep the cube stable.

## 1. Why Velocity Penalties Fail
The original investigation recommended penalizing lateral velocity (`reward -= 10.0 * lateral_speed`). 
*   **The Flaw:** This makes the entire arm sluggish. Fast, smooth movement is completely fine for a grasped object. What breaks the grasp is high-frequency changes in velocity (jitter), which create massive acceleration impulses (The "Hammer" effect). 

## 2. Action Variance Penalty (The "Smoothness" Curriculum)
Instead of penalizing speed, we will penalize **jerk/acceleration**. 

*   **The Math:** We calculate the difference between the current action and the previous action: 
    $\text{Variance} = \sum |a_t - a_{t-1}|^2$
*   **The Penalty:** `reward -= JITTER_PENALTY_WEIGHT * Variance`
*   **The Curriculum:** RL agents are lazy. If `JITTER_PENALTY_WEIGHT` starts too high, the model will learn that the easiest way to avoid the penalty is to stay perfectly still and never reach the goal. 
    *   **Phase 1 (0 - 50k steps):** `Weight = 0.0`. Let the model learn to reach the goal wildly.
    *   **Phase 2 (50k - 150k steps):** `Weight` ramps linearly to `5.0`. The model learns to "polish" its wild trajectory into a smooth arc to maximize its final reward.

## 3. Center-Line Drift Penalty (The "Virtual Tube")
During the `Ascend` and `Descend` scenarios, horizontal movement is not just unnecessary; it's dangerous. 

*   **The Math:** Calculate the L2 distance of the gripper's XY coordinates from the `tube_center_xy` (the starting or ending board coordinate).
*   **The Penalty:** `reward -= DRIFT_WEIGHT * drift_distance`.
*   **The Benefit:** This provides a smooth, continuous gradient pushing the model back to the center line at every step, replacing the binary "Crash" penalty used previously.

## 4. The Omniscient Penalties
The environment has access to the internal MuJoCo state and uses it to enforce the "Blind" model's discipline.

### A. The Drop Penalty (`is_held`)
*   **Detection:** Calculate the 3D distance between the `robot0:grip` site and the `object0` geom. If `dist > 0.03m` (3cm), the cube has fallen out.
*   **Penalty:** `reward -= 500`. 
*   **Termination:** The episode ends immediately (`terminated = True`). The model "feels" sudden death.

### B. The Shift Penalty (`has_shifted`)
*   **Detection:** This is for micro-slips. We transform the `object0` position into the **Gripper's Local Frame**. If the relative Y or X coordinate (sideways) of the cube moves more than `0.005m` (5mm) from its starting position within the fingers, a shift has occurred.
*   **Penalty:** `reward -= 10.0 * shift_distance`.
*   **Benefit:** The model feels a growing negative pressure as its actions cause the cube to vibrate toward the edge of the fingers, teaching it to stabilize *before* a drop occurs.
