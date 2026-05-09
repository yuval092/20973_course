# Grasp Stage Evaluation Report

## Commands Used for Evaluation

To evaluate the physics, single scenarios, and full pipeline, the following command suite was executed:

1. **Gate 1-3 (Physics Isolation Tests)**: Tests static grasp, lift, and transit with the exact physics parameters.
   ```bash
   PYTHONPATH=. python scripts/test_grasp_physics.py --n-trials 50
   ```

2. **Gate 4 (Single Scenario Certification)**: Tests the RL model purely on individual scenarios.
   ```bash
   PYTHONPATH=. python scripts/eval.py --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip --scenario transit --n-episodes 100
   PYTHONPATH=. python scripts/eval.py --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip --scenario descend --n-episodes 100
   PYTHONPATH=. python scripts/eval.py --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip --scenario ascend --n-episodes 100
   ```

3. **Gate 5 (3-Chain Certification)**: Tests the RL scenarios sequentially without the scripted grasp phase.
   ```bash
   PYTHONPATH=. python scripts/eval_sequence.py --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip --chain pick --n-episodes 100 --drift-limit 0.010
   ```

4. **Gate 7 (Full Sequence Pipeline)**: Tests the end-to-end performance including the scripted Grasp phase.
   ```bash
   PYTHONPATH=. python scripts/eval_grasp.py --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip --n-episodes 100 --drift-limit 0.010
   ```

## Test Results and Success Rates

As per instructions, the implementation was strictly adhered to the plans detailed in `@docs/grasp_stage_plans/**` (Documents 01 through 09), while deliberately excluding all solutions, bug fixes, and improvements outlined in Document 10 ("Path to 99%+").

| Test Phase | Sub-Task | Success Rate |
| :--- | :--- | :--- |
| **Physics Isolation (Gate 1-3)** | Static Grasp | **0.0%** (0/50) |
| | Lift | **0.0%** (0/50) |
| | Transit Held | **0.0%** (0/50) |
| **Single Scenarios (Gate 4)** | Transit | **100.0%** |
| | Descend | **81.0%** |
| | Ascend | **91.0%** |
| **Sequence Chain (Gate 5)** | 3-Chain | **93.0%** |
| **Full Pipeline (Gate 7)** | End-to-End | **0.0%** |

## Main Reasons of Failure

The evaluation logs show two primary categories of failure directly related to the strict adherence to the initial plan:

1. **`FINGER_CLOSED_EMPTY_EARLY` (100% of Physics Grasp Failures):**
   * **Observation:** In the physics tests (Gates 1-3) and the full pipeline (Gate 7), the fingers consistently trigger the early-abort check (e.g., `finger=0.0006 after 31 steps`), meaning they closed completely without encountering the cube.
   * **Root Cause:** By strictly following the plan (which dictates a high `Kp=150,000` combined with directly setting `finger_target_joint = 0.0`), the fingers snap shut at extreme velocities in a single simulation step. This causes them to bypass MuJoCo's contact solver entirely, resulting in "ghosting" where the fingers phase through the 0.05kg cube.

2. **`TUBE_BREACH` (Primary cause of Descend failures in Gates 4 and 5):**
   * **Observation:** The `descend` scenario success rate dropped to 81%, with crashes like `TUBE_BREACH (center=0.0123 > limit=0.0100)`.
   * **Root Cause:** The RL policy naturally operates near the 10mm limit boundary. Without the 3mm `TUBE_BREACH_GRACE` buffer (which was introduced as a fix in Document 10), marginal drifts immediately fail the episode, heavily degrading performance.

3. **`PRECONDITION_XY` and `PRECONDITION_SPEED` (Full Pipeline Failures):**
   * **Observation:** In Gate 7, several episodes failed before the grasp even began due to `PRECONDITION_XY (cube is 341.3mm from grip site)`.
   * **Root Cause:** Without the corrected velocity-zeroing order (which was prioritized in Document 10), the `execute_grasp` function begins with a 50-step "halt loop" while the arm still carries residual momentum from the RL policy. This causes the arm to oscillate at `GRASP_Z` and physically strike the cube, causing a "proximity explosion" that launches the cube off the table before the fingers can close.

## Insights Deduced

The critical insight from this evaluation is that **the original plan parameters (Documents 01-09) are physically incompatible without the stabilization techniques outlined in Document 10**. 

1. **High Forces Require Velocity Control:** The planned `Kp` of 150,000 provides excellent grip strength but introduces insurmountable solver instability (ghosting) when applied instantaneously to a 0.05kg object. A linear target ramp (as introduced in Phase 1 fixes) is mathematically necessary to slow the fingers down enough for the contact solver to register the collision.
2. **Residual Momentum Must Be Handled Explicitly:** Relying purely on a 0-action loop to halt the arm near fragile objects leads to proximity explosions. The `qvel` and `qacc` vectors must be manually zeroed *before* the simulation steps continue.
3. **Rigid Adherence vs. Practical Physics:** While the logic of the plan is structurally sound, MuJoCo's continuous-time simulation approximations mean that certain "ideal" parameters (like strict 10mm drift limits or instantaneous high-Kp target jumps) require engineered grace buffers and ramping to function in practice.

By strictly excluding Document 10, we empirically confirmed every single failure mode that the Document 10 roadmap was created to solve.