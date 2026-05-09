# Testing and Validation: The Mandatory Pre-Flight Suite

Before any retraining is allowed to commence, the environment modifications must be rigorously verified. Modifying physics engines is prone to hidden "Silent Killers" (e.g., clipping geom edges, infinite repulsive forces).

The following Automated Validation Suite must be implemented as scripts in `scripts/` and must pass with 100% reliability.

## 1. The "Static Grip" Test
**Goal:** Verify the geometric offset fix and the 30N Normal Force math.
*   **Procedure:** 
    1. Spawn the arm exactly over the cube at `GRASP_Z`.
    2. Close the fingers to `0.0136m`.
    3. Run the MuJoCo simulation for `500` steps with **NO actions** (`[0, 0, 0, grasp_width]`).
*   **Pass Criteria:** The cube's absolute X, Y, and Z coordinates must not change by more than `0.0001m` (0.1mm) from step 20 to step 500.
*   **Failure:** If the cube slowly slides or falls, the grasp width math is wrong, or friction is insufficient.

## 2. The "Impulse Stress" Test
**Goal:** Verify the "Rubber Finger" (`solref`, `solimp`) and `condim=6` modifications.
*   **Procedure:**
    1. Grasp the cube securely.
    2. Apply a synthetic "jitter" action sequence: `[+1, 0, 0]`, `[-1, 0, 0]`, `[+1, 0, 0]` at 50Hz (simulating the 1.5 m/s² accelerations seen from the SAC model).
*   **Pass Criteria:** 
    1. The simulation must not crash.
    2. The cube's Z-coordinate must not exceed `2.0m` (Detecting the "Explosion" bug).
    3. `is_held` must remain True throughout the vibration.
*   **Failure:** The physics hardening failed to absorb the impulsive energy.

## 3. The "Stillness Threshold" Test
**Goal:** Verify that the Master Controller's `0.05 m/s` Stillness Validation is achievable.
*   **Procedure:** 
    1. Load the *current* buggy SAC model.
    2. Run it in an empty space (no cube) and command it to reach a fixed point.
*   **Pass Criteria:** The arm's resting velocity must drop below `0.05 m/s` within 50 steps of reaching the goal coordinates.
*   **Failure:** If the SAC model's inherent "resting jitter" is higher than 0.05 m/s, the Master Controller will hang in an infinite loop waiting for stillness. The threshold must be raised, or the Action Variance Penalty must be applied.

## 4. The "Payload Noise" Test
**Goal:** Verify the Domain Randomization reset logic for `Transit` and `Descend`.
*   **Procedure:**
    1. Call `env.reset()` 100 times for the `Transit` scenario.
    2. The cube should be spawned slightly offset (+/- 2mm) within the closed fingers.
*   **Pass Criteria:** The MuJoCo solver must cleanly resolve the initial contacts in step 0 without applying an infinite repulsive force (i.e., the cube doesn't instantly shoot through the ceiling on reset).
*   **Failure:** The domain randomization is spawning the cube *inside* the physical volume of the finger geoms.

---

**Execution Gate:** Only once `test_static_grip.py`, `test_impulse_stress.py`, `test_stillness.py`, and `test_payload_noise.py` all return `True` should `trainer.py` be executed for the 200k step fine-tuning run.
