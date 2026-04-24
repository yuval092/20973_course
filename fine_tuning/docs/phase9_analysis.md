# Phase 9 Training: Full Analysis, Problems & Recommendations

## Current State Summary

- **Active run:** `dense_refinement_20260423_222121` — started ~22:21, ~30k steps done as of analysis time
- **Previous run (`_214738`):** Ran to ~134k steps before being killed. Had promising Descend metrics (78% at step 98k) but Transit and Ascend were at 0%.
- **Training process:** PID 166388, running `scripts/fine_tune_dense.py`, loading `models/final_grandmaster_model.zip`

---

## 🚨 Critical Problems

### Problem 1: The New Run Was Killed For the Wrong Reason — "Catastrophic Reset Syndrome"

The AI agent killed the previous run (which reached 78% on Descend) by claiming the "observation space and reward logic changes would corrupt the weights with conflicting data."

**This is factually incorrect and a major mistake.**

The observation *masking* in Phase A only happens at inference time inside `step()` — the network architecture (input/output dimensions) didn't change. The reward tweak (Phase C, +2.0 soft landing bonus) is a minor additive scalar. Neither of these changes invalidates existing weights. SB3's replay buffer simply accumulates new experiences under the new rules. There was no architectural change. The correct action would have been to **continue the `_214738` run** with the new code injected, not abort it.

The current run (`_222121`) is now back to square one, showing 0% success across all scenarios at step 30k — which is exactly where `_214738` was before it built back up. This reset wasted several hours of compute.

---

### Problem 2: The `ent_coef` Is Not Being Set Properly

In `fine_tune_dense.py` line 165:
```python
model.ent_coef = 0.001 if 'FetchPickAndPlace' in base_model_path else 0.001
```
The condition is **always True vs. True** — both branches are `0.001`. This is dead code. More critically:

The `ent_coef` logs show it sitting at **~0.0057** (rising to match entropy target), not locked at `0.001` as intended. This means the `model.ent_coef = 0.001` line is **silently failing** — assigning to an attribute that SB3's SAC ignores in favor of its internal `log_ent_coef` tensor when `ent_coef='auto'`. The correct fix is:
```python
import torch
model.log_ent_coef.data = torch.tensor([np.log(0.001)], device=model.device)
```
Because `ent_coef` in SB3 is a `torch.nn.Parameter`, not a plain Python attribute. Setting it as a Python attribute shadows the PyTorch parameter but doesn't affect training.

**Impact:** The model is exploring ~5x more than intended (`ent_coef ≈ 0.006` vs target `0.001`), causing erratic, jittery behavior that causes tube-drift terminations.

---

### Problem 3: The Virtual Tube Is Too Tight At Episode Start — "Instant Death" on Reset

From the env debug logs, the single most common termination pattern is:
```
TERMINATED (Descend): Tube Drift = 0.0425m (Max 0.040m) at step 1
TERMINATED (Ascend): Tube Drift = 0.0407m (Max 0.040m) at step 3
TERMINATED (Descend): Tube Drift = 0.0456m (Max 0.040m) at step 1
```

The arm is **dying at step 1 or step 3** — before the policy has taken a single meaningful action. This is a **reset physics problem**, not a policy problem.

**Root Cause Analysis:**
The `_reset_sim` does a 100-step proportional feedback loop (the "Phantom Settle") to bring the arm to `arm_start_pos`. However, the loop has a fundamental flaw: it nudges `mocap_pos` by `error * 0.5` and runs the physics, but the *gripper tip* (`robot0:grip`) lags behind the mocap target due to joint compliance. After 100 steps, the arm may still have residual inertia, and when `qvel[:] = 0.0` is applied, the PID springs back slightly — especially with a stiff model — causing the reported 4-5cm horizontal displacements at step 1.

The 40mm tube limit is the "expanded wiggle room" from a previous iteration. The fact that the arm is still failing at step 1 means the reset is not landing the arm inside the tube before the episode actually starts.

**The plan's own Phase B (Penalty Annealing) was designed to fix exactly this**, but was never implemented. The roadmap specified starting at `MAX_DRIFT = 0.080m`. The code currently hardcodes `0.040m` with no annealing.

---

### Problem 4: Transit Floor Termination Is Misconfigured

From docs `phase9_architecture_and_rationale.md`, the spec says:
> Virtual Floor (Transit): Terminal death (`-10.0`) if dipping below `0.500m`.

But `chess_fetch_dense_env.py` line 170 says:
```python
if gripper_pos[2] < 0.450:
```
The threshold was silently lowered from `0.500m` to `0.450m` (without updating the docs or the debug log message, which still says `Min 0.450m`). This is inconsistent with the spec. More importantly, the debug logs show transit failures at `Z = 0.4249`, `0.4362`, `0.4371` — all well below even the 0.450 threshold. The arm is dipping ~10cm below SAFE_Z (0.550) before being killed. This is the expected "gravitational sweep" of the base model, which was never trained to hold altitude. The floor at 0.450 is arguably too harsh for a model that isn't being rewarded for altitude maintenance until it's already broken the floor.

---

### Problem 5: Observation Masking Is Masking the Wrong Indices AND Applied Post-Reward

In `chess_fetch_dense_env.py`, the masking is applied at lines 137-140:
```python
if self.current_scenario == 'transit':
    obs['observation'][3:9] = 0.0
    obs['observation'][11:20] = 0.0
```

The masking happens **before the reward is calculated on the next step**, but **after the current reward** that uses `gripper_pos = obs['observation'][0:3]`. This is correct for the policy input, but:

The Fetch observation layout is:
```
[0:3]   = gripper_pos
[3:6]   = object_pos (relative to gripper — CORRECT to mask)  
[6:9]   = object_rel_pos
[9:12]  = gripper_state
[12:15] = object_rot (Euler)
[15:18] = object_velp
[18:21] = object_velr
[21:24] = gripper_vel
[24:25] = gripper_fingers
```

The masking of `[11:20]` is **wrong**. Indices 11–20 include `gripper_state[2]` (index 11) and parts of object data. More critically, it misses `obs['achieved_goal']` and `obs['desired_goal']` keys that HER/the base env also uses. But the most important issue: **masking isn't needed at step 1** because at reset, observation is returned from the parent's `reset()` *before* `step()` is called. The first unmasked `obs` from `reset()` still contains the real object position, giving the policy a "peek" at object data before masking kicks in.

---

### Problem 6: Reward Signal Is Inconsistent Between Scenarios

Looking at the reward structure in `step()`:
- **Success:** `reward = 10.0` if `dist_to_goal < 0.01`
- **Failure:** `reward = -dist_to_goal` (continuous penalty)
- **Transit floor violation:** `reward -= 50.0 * depth` (proportional)
- **Tube violation:** `reward -= 10.0 * max(0, drift - 0.01)` (proportional)
- **Soft landing:** `reward += 2.0` (bonus)

For a **transit** scenario where the arm is 0.35m from goal (typical), the base reward = `−0.35`. But the tube/floor penalties can be `−50 * 0.1 = −5.0` in a single step. The sparse success bonus (`+10.0`) occurs only on the final step. The **reward scales are mismatched** by ~50x between the distance signal and the constraint signal. This drowns out the "go toward goal" signal and the policy learns to minimize penalty (stay still or die fast) rather than navigate.

---

### Problem 7: Phase B (Penalty Annealing) Was Never Implemented — And Is Desperately Needed

The roadmap explicitly says:
> **Penalty Annealing: Correct & Powerful** — Essential to prevent "early termination" bias.

The agent skipped implementing this, claiming "async multiprocessing complexity." This is a weak excuse — `SubprocVecEnv` doesn't require synchronous custom wrappers for annealing. The parent env simply needs a `current_step` that it receives from a callback.

The evidence clearly shows we need it: **step-1 tube terminations** are the primary training bottleneck. The model cannot learn from 1-3 step episodes.

---

### Problem 8: `model.learning_starts` Is Set Incorrectly

Line 170 in `fine_tune_dense.py`:
```python
model.learning_starts = 2000
```
The base model has `~1,055,000 n_updates` stored. After loading without a replay buffer and clearing it, the emptied buffer will trigger sampling after 2000 steps. But the model has complex priors built from 1M+ updates. Training on 2000 randomly initialized experiences against those priors is **catastrophically noisy**. The previous bug report explicitly noted that `learning_starts = model.num_timesteps + 1000` was the correct fix — but this was reverted to `2000` in the current script.

---

## ⚠️ Misconceptions in the Previous Agent Responses

1. **"78% Descend by step 135k is spectacular progress."** — This is *only* Descend. The model hasn't touched Transit (0%) or Ascend (0%) at all. The overall usable success rate is effectively 0% since you need all three scenarios to work to move a chess piece.

2. **"Phase A (Observation Masking) will drastically simplify Transit."** — It won't fix the root issue. The transit problem is the base model's natural tendency to sweep arcs (it was trained for pick-and-place, not horizontal transit), not its awareness of the object. The model needs a stronger altitude-maintenance reward signal.

3. **"The model is learning to navigate inside Virtual Tubes."** — The env debug logs directly contradict this. At step 30k of the new run, the arm is still dying on step 1-4 of elevator scenarios and step 4-11 of transit scenarios. Average survival doesn't meaningfully reflect learning when most of the survival time comes from a few lucky Transit episodes that run to timeout.

4. **"Training at 200+ FPS is fast."** — 82 FPS (from the actual logs) with 4 parallel envs is actually slow for this environment. Typical FetchPickAndPlace runs 300-500+ FPS. This suggests the 100-step phantom settle loop in `_reset_sim` is a significant bottleneck — it runs physics 100 times per episode reset.

---

## ✅ Recommended Actions (Priority Order)

### Fix 1 (CRITICAL — Do Now): Fix `ent_coef` Assignment
```python
import torch
model.log_ent_coef.data.fill_(np.log(0.001))
```
Replace line 165 in `fine_tune_dense.py`.

### Fix 2 (CRITICAL): Implement Phase B — Penalty Annealing
Add a `TOTAL_TIMESTEPS = 300_000` constant and in `step()`:
```python
progress = min(self.total_env_steps / TOTAL_TIMESTEPS, 1.0)
self.current_tube_limit = 0.080 + (0.040 - 0.080) * progress  # 8cm → 4cm
self.current_floor_limit = 0.480 + (0.450 - 0.480) * progress  # 48cm → 45cm
```
Pass the total_step counter via a shared value or an env attribute updated by a callback.

### Fix 3 (CRITICAL): Fix the Phantom Settle Reset — Stop Step-1 Deaths
The 100-step settle with proportional control needs a convergence check, not just a fixed count:
```python
for _ in range(500):  # max 500 steps
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    error = arm_start_pos - grip_pos
    if np.linalg.norm(error) < 0.005:  # 5mm convergence — stop early
        break
    self.data.mocap_pos[0] += error * 0.3
    self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
```

### Fix 4 (HIGH): Rebalance Reward Scales
The distance-based penalty `-dist_to_goal` per step over 175 steps means possible range is `-175 * 0.56 = -98` (board diagonal). The tube penalty is `-10.0 * drift`. They need normalization. Replace distance reward: `reward = -0.1 * dist_to_goal` (scale down by 10x).

### Fix 5 (HIGH): Restore Correct `learning_starts`
```python
model.learning_starts = model.num_timesteps + 2000
```

### Fix 6 (MEDIUM): Add Transit Altitude Reward — Not Just Penalties
Transit needs a *positive* signal for maintaining altitude, not just a cliff penalty:
```python
if self.current_scenario == 'transit':
    altitude_bonus = max(0, gripper_pos[2] - (self.SAFE_Z - 0.02))
    reward += 0.5 * altitude_bonus
```

### Fix 7 (MEDIUM): Clean Up Observation Masking
The masking should happen in `_get_obs()` override rather than post-step to ensure the reset obs is also masked. It should also mask the correct indices based on the actual Fetch obs layout.

### Fix 8 (LOW): Don't Kill Runs That Are Making Progress
The previous `_214738` run should be resumed (from `best_model_descend.zip` at 78%) rather than discarded. Load that checkpoint instead of `final_grandmaster_model.zip` and apply the fixes.

---

## TL;DR

The **current training run is suffering from 3 compounding issues** that prevent all learning:
1. `ent_coef` is silently broken (~5x too high), causing erratic movement
2. Virtual tubes are too tight, killing the arm at step 1 before it can learn anything (Phase B annealing was never implemented)
3. The reward scales are wildly imbalanced (constraint penalties dominate by 50x)

The "78% Descend" from the previous run was **real progress that was thrown away** for the wrong reasons. The correct path forward is to load `best_model_descend.zip`, apply the 3 critical fixes above, and resume training.
