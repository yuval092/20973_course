# Training Analysis and Model Selection

## Overview

This document covers: (1) what went wrong in the latest training run, (2) the
correct fix, (3) model selection for the grasp stage, and (4) the criteria for
certifying that a model is ready for cube integration.

---

## The Stalling Bug: Root Cause Analysis

### What Was Observed

`models/latest_model_2.zip` was found to achieve only ~65% chain success, with the
arm stalling 11–13mm above the goal. The arm outputs a weak upward action
(`action[2] = +0.005`) rather than descending the final millimeters. A competing
agent attributed this to "Reward Gradient Collapse" and proposed doubling the
distance reward near the goal. That diagnosis is wrong.

### Actual Root Cause: BRAKING_DIST Dead Zone

The failure is a classic RL dead zone created by misconfigured hyperparameters.

```
BRAKING_DIST    = 0.015m  (15mm)   ← distance where arm decelerates
SUCCESS_THRESHOLD = 0.010m (10mm)  ← distance where episode terminates as success
```

When `dist_to_goal` is between 10mm and 15mm, the braking penalty applies but
success is not yet possible. The optimal policy in this zone is to **stop moving**:
- Moving toward goal: still penalized for being in the braking zone
- Moving away: increases distance penalty
- Stopping: minimizes penalty accumulation

The policy correctly learned to stop. The stalling is **not a bug in the policy**
— it is the expected optimal behavior for these hyperparameters.

### Evidence

From action traces of `latest_model_2.zip`:
- At 11.3mm from goal: `action[2] = +0.005` (opposing direction)
- At 13.2mm from goal: `action[2] = +0.003` (opposing direction)

The policy is actively moving away from the goal (upward) to stay at the edge of
the dead zone. This maximizes reward by avoiding both the braking penalty and the
boundary where moving toward goal increases braking penalty further.

### Why `best_model_combined.zip` Does Not Have This Bug

`checkpoints/pure_movement_v6_20260502_171928/best_model_combined.zip` was trained
with `BRAKING_DIST = 0.010m` (at this checkpoint, `BRAKING_DIST == SUCCESS_THRESHOLD =
0.010m`, so the dead zone has zero width). The policy approaches the goal directly.

From action traces of `best_model_combined.zip`:
- At 11.3mm from goal: `action[2] = -0.027` (toward goal)
- At 3.5mm from goal: `action[2] = -0.005` (braking, nearly stopped)
- Success rate: 100% (3-chain, 100 episodes)

---

## The Fix: Eliminate the Dead Zone

### Hyperparameter Change

```yaml
# configs/env.yaml — current training (FIXED):
braking_dist: 0.010     # Was: 0.015. Now equals success_threshold → dead zone = 0
success_threshold: 0.010
```

With `BRAKING_DIST = SUCCESS_THRESHOLD`, the braking zone begins exactly where the
success condition starts. The first step into the braking zone can also be the first
step where success is detected. There is no zone where "braking but not success" is
the only option.

### What NOT to Do

The competing agent proposed "doubling the distance reward near the goal." This is
wrong because:
1. The stalling is rational behavior for the given reward function — increasing the
   reward does not change the rational action
2. Changing reward shaping mid-training resets the policy's value estimates and can
   cause catastrophic forgetting
3. The correct fix is a one-line hyperparameter change, not reward surgery

---

## Model Selection for Grasp Stage

### Use: `checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip`

This is the **latest training session** (2026-05-04 09:06:39). It is the model to use
for all grasp stage evaluation and development. The earlier `20260502_171928` checkpoint
is retained for historical reference only.

The new training runs (`20260504_010553`, `20260504_084035`, `20260504_090639`) were
triggered to fix the `BRAKING_DIST` dead zone bug. Before running grasp stage
evaluation, certify the new checkpoint against the criteria below.

**Certification Protocol:**
1. `eval.py --scenario transit --n-episodes 100`: success ≥ 95%
2. `eval.py --scenario descend --n-episodes 100`: success ≥ 95%
3. `eval.py --scenario ascend --n-episodes 100`: success ≥ 95%
4. `eval_sequence.py --n-episodes 100`: chain success ≥ 85%
5. Terminal speed at goal ≤ 5mm/s (verify no stalling in 10–15mm approach zone)
6. `assert env.unwrapped.BRAKING_DIST == env.unwrapped.SUCCESS_THRESHOLD` (dead zone = 0)

If any criterion fails, check whether `braking_dist: 0.010` is correctly set in the
training config for that run.

---

## Training Configuration Reference

### Stable Training Configuration

```yaml
# configs/train.yaml (stable — do not change for this stage):
learning_rate: 0.0003
gamma: 0.99
buffer_size: 1000000
batch_size: 256
tau: 0.005
ent_coef: "auto"
target_entropy: "auto"
```

```yaml
# configs/env.yaml (grasp-stage values):
braking_dist: 0.010          # CRITICAL: must equal success_threshold
success_threshold: 0.010
stability_vel_threshold: 0.05
drift_limit_end: 0.010
eval_drift_limit: 0.010      # Must match drift_limit_end
grasp_z: 0.425               # Must be updated from 0.430
safe_z: 0.550
```

### Dead Zone Guard in `train.py`

Add this assertion at the start of any training run:

```python
assert cfg.BRAKING_DIST <= cfg.SUCCESS_THRESHOLD, (
    f"BRAKING_DIST={cfg.BRAKING_DIST} > SUCCESS_THRESHOLD={cfg.SUCCESS_THRESHOLD}: "
    f"creates a dead zone where stalling is the optimal policy. "
    f"Set braking_dist={cfg.SUCCESS_THRESHOLD} to fix."
)
```

This prevents inadvertently re-introducing the dead zone.

---

## Retraining: When Is It Needed?

### NOT needed for this stage

The Phase 9 trick (finger state zeroed in observation, object position mapped to
gripper position) makes the RL policy identical for:
- Cube hidden vs cube visible
- Fingers teleported vs fingers actuator-driven (grasp_mode)

The policy output is the same. No retraining required.

### Required if:

1. **GRASP_Z changes again** (e.g., from 0.425 to a new value more than ±10mm from
   the trained value). The descend model was trained toward a specific GRASP_Z. A
   ±5mm change is within the model's precision; a ±20mm change requires retraining.

2. **A new scenario is added** (e.g., a "place" scenario that descends to PLACE_Z
   while holding the cube). This is a new task that the current model was not trained
   for. The Phase 9 trick may not apply if the cube's weight visibly affects arm
   dynamics.

3. **Observation space changes** (e.g., adding cube-relative position to the
   observation). The current model was trained on a 25-D observation; changing this
   invalidates the policy.

4. **Physics changes cause arm dynamics to shift** (e.g., the weld solref change
   alters the arm's feel during ascent). If the new physics differs enough that the
   current policy no longer converges, targeted retraining of the affected scenario
   (ascend) may be needed.

---

## How to Diagnose Stalling in a Trained Model

Run this diagnostic script after training:

```python
# Quick stall check — run before any eval
import numpy as np

def check_dead_zone(env, model, n_episodes=20):
    """Checks whether the model stalls in the braking zone."""
    stall_count = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            
            dist = info.get("dist_to_goal", None)
            if dist is None:
                break
            
            braking_dist = env.unwrapped.BRAKING_DIST
            success_thresh = env.unwrapped.SUCCESS_THRESHOLD
            
            if success_thresh < dist < braking_dist:
                # In the dead zone
                arm_vel = np.linalg.norm(env.unwrapped.get_arm_velocity())
                if arm_vel < 0.003:  # < 3mm/s = effectively stopped
                    stall_count += 1
                    print(f"Ep {ep} step {step}: STALL at dist={dist*1000:.1f}mm "
                          f"vel={arm_vel*1000:.1f}mm/s action_z={action[2]:.3f}")
                    break
            
            if done or trunc:
                break
    
    stall_rate = stall_count / n_episodes
    print(f"Stall rate: {stall_rate:.0%} ({stall_count}/{n_episodes})")
    return stall_rate
```

A stall rate > 0% indicates the dead zone is active. Check `BRAKING_DIST` and
`SUCCESS_THRESHOLD` in `env.yaml`.

---

*Next: [08 — Soft Reset Reference](./08_soft_reset_reference.md)*
