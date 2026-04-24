# Phase 9: Refined Contextual RL Curriculum Plan

This document serves as the updated, high-level blueprint for transitioning the RoboChess movement engine from static training to a dynamic **Curriculum Learning** framework.

---

## 1. Analysis of Proposed Refinements

After rigorous internal review, the proposed refinements are assessed as follows:

| Refinement | Status | Rationale |
| :--- | :--- | :--- |
| **Observation Masking** | **Correct & Essential** | Removing irrelevant piece data in `transit` mode forces the policy to generalize solely on navigation constraints. |
| **Penalty Annealing** | **Correct & Powerful** | Essential to prevent "early termination" bias. By starting with loose constraints and tightening them, we prevent the model from dying before it even discovers the goal. |
| **Dynamic Reward Tuning** | **Correct & Recommended** | Velocity-based dampening is the industry standard for "soft landings" in robotic pick-and-place. |
| **Evaluation Refinement** | **Correct & Efficient** | Fixed-interval evaluation is computationally expensive. Switching to event-based (improvement-based) evaluation will increase training throughput. |
| **Hyperparameter Annealing** | **Correct & Standard** | Reducing entropy as the model matures is standard practice for stabilizing policies in SAC (Soft Actor-Critic). |

---

## 2. Updated Implementation Roadmap

### Phase A: Environment Masking (Transit Logic)
*   **Action:** Modify `step()` in `chess_fetch_dense_env.py`.
*   **Logic:** Detect `self.current_scenario == 'transit'`.
*   **Implementation:** Overwrite `obs['observation'][3:6]` (the object coordinates) with `0.0` before returning.
*   **Verification:** Confirm that the observation input to the model no longer tracks the cube during air transit.

### Phase B: Penalty Annealing (Curriculum)
*   **Action:** Modify constraint logic in `step()`.
*   **Implementation:**
    *   **Drift Limit:** Linearly interpolate `MAX_DRIFT` from `0.080m` (start) to `0.025m` (end).
    *   **Floor Limit:** Linearly interpolate `MIN_Z` from `0.450m` (start) to `0.480m` (end).
*   **Logic:** `self.current_drift_limit = start_limit + (end_limit - start_limit) * (current_steps / total_steps)`.

### Phase C: Velocity-Based Soft Landing
*   **Action:** Modify `reward` calculation in `step()`.
*   **Logic:** If `dist_to_goal < 0.05` and `velocity < 0.01`, add `+2.0` bonus.
*   **Rationale:** Rewards the model for actively slowing down as it reaches the goal, preventing the "smashing" behavior observed in early runs.

### Phase D: Evaluation Optimization
*   **Action:** Refactor `SuccessRateEvalCallback` in `fine_tune_dense.py`.
*   **Logic:** Instead of fixed `eval_freq=5000`, monitor the model's performance and only perform full stratified evaluation when the training-rollout success rate shows a statistically significant upward trend.

---

## 3. Potential Edge Cases & Traps

*   **Annealing Instability:** If the constraints tighten too fast, the model may experience "Policy Drift," where the weights collapse because they can't adapt to the shrinking safe zone. 
    *   *Mitigation:* Keep the annealing curve conservative (sigmoid or slow linear) to ensure the model has plenty of time to adapt.
*   **State-Dependency Bias:** By masking object coordinates in Transit, we are essentially training two separate "brains" (one that knows where the object is, and one that doesn't). 
    *   *Observation:* This is actually a feature, not a bug—the policy will learn to switch off its "perception" of the object when its task is purely navigational.
*   **Reward Scale Imbalance:** When we add a "Soft Landing" bonus, it might overpower the "Precision" penalty if not scaled carefully.
    *   *Mitigation:* Reward components will be normalized to ensure no single signal dominates the others during gradient updates.

---

## 4. Final Review
The plan to implement Curriculum Learning is correct and highly recommended. It transforms a "trial by fire" training approach into a structured educational curriculum for the robot arm. Proceeding with this plan will ensure the model masters the constraints precisely without failing during the early stages of training.
