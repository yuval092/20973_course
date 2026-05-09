# 03. Diagnostic Masking Tests

Before committing to a long retraining process, we performed a definitive "Zero-Lateral" mask test to see if the current model was even capable of a straight vertical lift.

## 1. The Test: "Blinding" the Model
We modified `src/chess_env/task.py` to physically override the model's output. If the scenario was `ascend`, we forced the X and Y actions to **exactly zero**, allowing only Z-axis movement to reach the physics engine.

### The Hypotheses:
*   **A:** If the arm reaches `SAFE_Z` successfully, the wiggles are just "trash" noise. We can fix the problem with a simple code mask.
*   **B:** If the arm fails to reach the goal, it means the model's Z-axis logic is mathematically "coupled" to the X/Y jitter. It proves that the model **must** be retrained to understand vertical-only movement.

## 2. The Result: Definitive Stalling
The results were shocking and provided the final piece of our investigation.

### Observations:
1.  **Grasp Integrity:** With the mask active, the cube **never dropped**. The arm moved perfectly vertically without any slip.
2.  **The Stall:** Even though the jitter was gone, the arm **stopped at `z = 0.546m`** and hovered there indefinitely, failing to reach the `0.550m` success threshold.

## 3. Conclusion: Coupled Weights
This test proved that Hypothesis B is correct. The model's neural network weights have learned that a "lift" move *requires* those tiny lateral corrections to function. It cannot mathematically reach the final 4mm of altitude without the wiggles.

**What we learned:** We cannot simply "mask out" the bad behavior. The model's internal PID-like logic for vertical precision is broken. We must retrain the model with a new reward surface to teach it that lateral movement during a lift is forbidden.
