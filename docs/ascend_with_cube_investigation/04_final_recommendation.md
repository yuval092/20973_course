# 04. Final Recommendation and Path Forward

Based on our exhaustive analysis and testing, we have a clear path to achieve production-ready chess piece manipulation.

## 1. Why Retraining is the Only Solution
As proven in Document 03, the existing model's "Cerebellum" is fundamentally trained to move with high-frequency noise. This noise is incompatible with carrying a payload because it generates massive inertial forces that overcome any realistic physical friction. Furthermore, the vertical and horizontal axes are mathematically coupled in the current weights, making simple output masking impossible.

## 2. The Fix: "Payload Slip" Reward Shaping
We will fine-tune the model for **200,000 steps** with a new, aggressive reward constraint.

### Implementation:
We will add a new component to the reward function in `task.py`:
1.  **Calculate Lateral Speed:** We will measure the gripper's velocity in the X and Y planes (`grip_velp[:2]`).
2.  **Gated Penalty:** We will apply a massive negative reward (weight: `10.0`) based on this speed, **BUT ONLY** when the scenario is `ascend`.

```python
if self.current_scenario == "ascend":
    # Aggressively penalize lateral jitter
    reward -= 10.0 * lateral_speed
```

## 3. Why This Will Work
*   **Decoupling:** The SAC optimizer will see the massive penalty for lateral wiggles and will find a new set of weights that reach the vertical goal (`SAFE_Z`) using only the Z-axis.
*   **Smoothing:** Because we penalize velocity, the model will learn to move with the "gentleness" required to keep a 0.05kg cube stable within the fingers.
*   **Precision:** By resuming from our best 100% success model and applying these new constraints, we will fix the jitter without losing the speed and accuracy the model already possesses.
*   **No Regression:** Because the penalty is "gated" (only active in `ascend`), the `transit` scenario (which *must* move laterally) will be completely unaffected.

## 4. Summary of Strategy
We are moving from a model that **can move** to a model that **can carry**. Training the model to respect payload inertia is the standard industry approach for precision robotics and will yield the stable, reliable behavior required for chess manipulation.
