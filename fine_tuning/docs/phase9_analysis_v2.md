# Phase 9 Code Review — Second Pass
> After the fixes from the previous analysis were applied.

---

## Executive Summary

The updated code is significantly better. The phantom-settle convergence fix, `ent_coef` fix, `learning_starts` fix, and curriculum annealing are all properly implemented and verified to work. The reset no longer produces step-1 deaths (confirmed: drifts at reset are 1-3mm, well within the 80mm starting limit).

**However, there are still 12 bugs/issues found — ranging from a critical curriculum counter that silently does nothing in production, to a reward structure that always fires two bonuses simultaneously at success, to a deeper root cause for why transit and ascend fail that the current reward signal can't address.**

---

## 🔴 Critical

### Bug 1: The Curriculum Counter Is Not Shared with SubprocVecEnv Workers

**File:** `scripts/fine_tune_dense.py` lines 145–146  
**Verification:** Confirmed via `pickle.dumps(Value(...))` — raises `RuntimeError: Synchronized objects should only be shared between processes through inheritance`.

```python
curriculum_counter = Value(c_longlong, 0)
train_env = SubprocVecEnv([make_env(curriculum_counter) for _ in range(num_envs)])
```

`SubprocVecEnv` pickles the `make_env` factory to send it to worker processes. `multiprocessing.Value` objects **cannot be pickled** — they can only be shared through `multiprocessing.Process` *inheritance* (i.e., forking). The factory closure captures `curriculum_counter`, and when the subprocess tries to deserialize it, it creates an **independent copy** with value 0 that is **never updated**.

**Result:** The curriculum is completely frozen at `progress = 0.0`. Every episode across the entire 300k-step training run uses `CURRICULUM_DRIFT_START = 0.08m` and `CURRICULUM_FLOOR_START = 0.45m`. The tightening never happens. The model never faces harder constraints. This defeats the entire purpose of curriculum learning.

**Fix:** Use `multiprocessing.Manager().Value()` which is proxy-based and picklable, OR fall back to env-local `total_env_steps` (which already exists in the env). The simplest fix: remove the shared counter entirely and just use the env's internal `self.total_env_steps`:

```python
# In _curriculum_progress(), just use self.total_env_steps
# (already the fallback when curriculum_steps is None)
# Remove CurriculumProgressCallback and curriculum_counter from fine_tune_dense.py
```

Alternatively, pass `curriculum_progress_override` as a float that gets updated — but that has the same pickling problem. The cleanest solution is having the env manage its own step count, which it already does.

---

### Bug 2: `compute_reward` Is Called Twice Per Step (Wasteful + Latent Risk)

**File:** `chess_env/chess_fetch_dense_env.py` lines 318, 335

The parent's `step()` in `BaseRobotEnv` (line 150 in robot_env.py):
```python
reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
```
...calls **our overridden `compute_reward`** via Python MRO. The first call happens inside `super().step()`, and the return value is discarded via `obs, _, terminated, truncated, info = super().step(action_copy)`. Then our `step()` calls it again on line 335.

The correct fix was done (discarding `_`), so this doesn't produce wrong rewards — but it **doubles the `compute_reward` CPU cost** and creates a subtle risk: the `info` dict passed to `compute_reward` the first time (from `super().step()`) does **not yet contain** the Phase 9 scenario metadata (which is only added on line 324 after `super().step()` returns). This means the first `compute_reward` call processes an `info` dict without `scenario`, `tube_center_xy`, or `current_drift_limit`. Since the second call (with full `info`) overwrites it, the actual output is correct — but the first call with empty `info` means all scenario-specific reward components return 0 for that call, which is wasteful.

**Fix:** Override `compute_terminated` to return `False` from our env (it already returns `False` from the base). More importantly, recognize this double-call pattern and restructure so `compute_reward` is only called once from our `step()`.

---

## 🟠 High Severity

### Bug 3: Soft Landing Bonus Always Fires at Success

**File:** `chess_fetch_dense_env.py` lines 337–338, 278

`compute_reward` already adds `SUCCESS_BONUS = +10.0` when `dist < SUCCESS_THRESHOLD (0.01m)`.  
Then in `step()`:
```python
if dist_to_goal < self.SOFT_LANDING_DIST and grip_velocity < self.SOFT_LANDING_VEL:
    reward += self.SOFT_LANDING_BONUS  # +2.0
```

Since `SUCCESS_THRESHOLD (0.01m) < SOFT_LANDING_DIST (0.05m)`, **every single success also triggers the soft-landing bonus**. Success reward = `10.0 + 2.0 = 12.0` always, not just when approaching slowly. This makes the soft-landing a guaranteed 20% bonus on top of success, rather than a distinct incentive for slow approach. The policy learns nothing different about *how* to arrive — it just gets 12.0 instead of 10.0 at success.

**Fix:** Gate the soft-landing bonus to the near-but-not-success zone:
```python
if self.SOFT_LANDING_DIST > dist_to_goal >= self.SUCCESS_THRESHOLD and grip_velocity < self.SOFT_LANDING_VEL:
    reward += self.SOFT_LANDING_BONUS
```

---

### Bug 4: Success Is Never Explicitly Terminated in `step()`

**File:** `chess_fetch_dense_env.py` lines 369–375

```python
if info["is_success"] == 1.0:
    terminated = True
```

This works correctly — but note that `info["is_success"]` is set on line 341 using our `SUCCESS_THRESHOLD (0.01m)`, while the parent's `_is_success()` (called from within `super().step()`) uses `self.distance_threshold = 0.05m` (the original FetchPickAndPlace default). The parent sets `info["is_success"]` on line 144 of robot_env.py, but we immediately overwrite it on line 341 of our env. This is fine for training, but **the EvalCallback's `evaluations_successes` list is populated from the Monitor wrapper's `is_success` field**, which should be our override value since it happens before the Monitor gets it.

However, if anything in the chain reads the parent's `is_success` (which uses `0.05m` threshold) before our override, success rates in evaluation logs would be inflated. This is low risk but worth documenting.

---

### Bug 5: Transit Altitude Reward Is Too Weak to Override the Descent Prior

**Root Cause Analysis of Transit Failure:**

This is the deeper reason transit has failed in *every* run. The grandmaster model was trained on `FetchPickAndPlace-v4` where the task is: go to object XY → descend → grasp → ascend → go to goal XY → descend → release. The model's policy has deeply encoded: **"when I arrive at the XY overhead, descend."**

In transit, we ask it to: **go to goal XY while staying at SAFE_Z**. Its prior says to descend when goal XY is reached. This creates a direct conflict.

The altitude reward analysis (run live) shows:

| Z Height | Altitude Bonus | Floor Warning | Net |
|---|---|---|---|
| 0.550 | +0.010 | 0 | +0.010 |
| 0.535 | +0.003 | 0 | +0.003 |
| 0.520 | 0 | 0 | 0 |
| 0.515 | 0 | **-0.100** | -0.100 |
| 0.500 | 0 | **-0.400** | -0.400 |
| 0.450 | 0 | **-1.400** | **-1.400** (+death -10) |

The "safe zone" between SAFE_Z and the warning threshold (0.550→0.520) produces only **+0.01 at most**. The success bonus is **+10.0**. So the model gains essentially nothing from altitude until it reaches the floor warning zone. It will naturally descend as its prior dictates, only feeling pain once it's already 3cm below SAFE_Z.

**Fix:** The altitude reward needs to be a cliff (or steep gradient) that immediately penalizes any deviation below SAFE_Z, not just a weak bonus for being at or above it:
```python
# Replace current altitude bonus with:
z_error = max(0.0, self.SAFE_Z - gripper_pos[2])  # 0 at/above SAFE_Z, positive below
reward -= 5.0 * z_error  # Proportional penalty below SAFE_Z
```
This mirrors exactly how the elevator tube works (proportional penalty for drift), creating symmetric pressure to stay at altitude.

---

### Bug 6: Descend Goal-Sampling Wastes RNG and Creates Invisible Inefficiency

**File:** `chess_fetch_dense_env.py` lines 185–191

For `descend` and `ascend`, `goal_xy` is sampled from the board **but never used** — the actual goal XY is `start_xy` in both cases:
```python
start_pos = self._sample_board_position()
goal_pos = self._sample_board_position()  # Wasted for descend/ascend
while np.linalg.norm(goal_pos[:2] - start_pos[:2]) < self.MIN_GOAL_DIST:
    goal_pos = self._sample_board_position()  # Potentially many wasted calls
```
This wastes RNG calls, diverges the random seed from a "pure" implementation, and the `while` loop could loop many times in a seeded environment (spending extra calls on the unused `goal_xy`).

**Fix:** Only sample `goal_pos` when `current_scenario == 'transit'`.

---

## 🟡 Medium Severity

### Bug 7: Ascend With Closed Gripper Has ~1mm Physics Clearance

**From XML analysis:**

When `gripper_forced_state='closed'` (set randomly in ascend, line 216), `CLOSED_FINGER_POS = 0.01`. Each finger has a geom half-size of 0.007m in Y. The inner gap is:
- Right inner edge: `(0.0159 + 0.010) - 0.007 = 0.0109m`
- Total gap: `2 * 0.0109 = 0.0218m = 2.18cm`
- Object width: `2 * 0.01 = 2.0cm`

**Clearance: 1.8mm per side** (0.9mm per side finger-to-object). This is dangerously tight. MuJoCo's contact detection with `margin=0.001` (from the XML defaults) means contact forces activate within 1mm of the surfaces. With 0.9mm clearance, the physics engine will fire contact forces on every physics step, creating oscillatory force noise that physically pushes the gripper sideways — directly causing the tube drift that kills ascend episodes.

**Fix:** Set `CLOSED_FINGER_POS = 0.008` (gives ~3.4mm clearance per side) or change the object's geom size to match.

---

### Bug 8: Object Is Placed Directly at Transit Goal XY (Potential Confusion)

**File:** `chess_fetch_dense_env.py` line 197

```python
object_pos = np.array([goal_xy[0], goal_xy[1], self.CUBE_Z])
```

In transit, the object is at the *destination* XY but 14cm below on the table. While this doesn't cause a collision (the arm reaches SAFE_Z=0.550, object is at 0.410), the masked observation zeros out object position. The model never sees where the object is, but the object IS there. If the model descends to pick up the block (its prior behavior), the object is physically at the destination — this is an unintended but partially correct "pick at destination" scenario that confuses the semantics of transit training.

**Recommendation:** For transit, place the object completely off-board (`HIDDEN_OBJECT_POS`) since transit doesn't semantically involve the object at all.

---

### Bug 9: `test_reset_phase9.py` Tests Are Too Weak to Catch Training Issues

**File:** `scripts/test_reset_phase9.py`

The test checks for NaN and confirms a single step doesn't crash. It does **not** test:
- Whether the arm is actually within the tube at reset (it should, but the test doesn't verify)
- Whether the ascend scenario has the gripper at the correct height
- Whether closed-gripper scenarios produce physics instability after several steps
- Whether the curriculum limit values are reasonable at reset

**Recommendation:** Add asserts:
```python
# For ascend: verify arm starts at GRASP_Z ± 5mm
assert abs(obs['observation'][2] - env.unwrapped.GRASP_Z) < 0.005
# For descend: verify arm starts at SAFE_Z ± 5mm
assert abs(obs['observation'][2] - env.unwrapped.SAFE_Z) < 0.005
# For elevator: verify tube drift is within limit
drift = np.linalg.norm(obs['observation'][:2] - env.unwrapped.tube_center_xy)
assert drift < env.unwrapped.current_drift_limit
```

---

### Bug 10: Observation Masking Indices Are Correct But "Velocity Used for Soft Landing" Is Grip_velp, Not Gripper Velocity

**File:** `chess_fetch_dense_env.py` line 321

```python
grip_velocity = np.linalg.norm(obs["observation"][20:23])
```

Indices `[20:23]` = `grip_velp` (gripper positional velocity in Cartesian space). This is **correct** for measuring how fast the arm is moving. However, `grip_velp` is measured in `m/step` (scaled by `dt * n_substeps`). With `timestep=0.002` and `n_substeps=20`, `dt = 0.04s`. The soft-landing threshold `SOFT_LANDING_VEL = 0.01` means `0.01 m/step = 0.25 m/s`. This is quite slow — the arm basically needs to be nearly stationary. This is fine intentionally, but worth documenting that the units are `m/step` not `m/s`.

---

### Bug 11: XML — `coordinate="local"` Is Deprecated in MuJoCo ≥ 2.3

**File:** `chess_env/assets/pick_and_place.xml` line 3

```xml
<compiler angle="radian" coordinate="local" meshdir=...>
```

The `coordinate="local"` attribute was deprecated in MuJoCo 2.3 and removed in later versions. Modern mujoco (3.x) always uses local coordinates. This will emit a warning on every environment creation and may cause errors in future versions. It's harmless now but should be cleaned up.

**Fix:** Remove `coordinate="local"` from the compiler tag.

---

### Bug 12: Object Mass Is Too High, Creating Phantom Forces in Ascend

**File:** `chess_env/assets/pick_and_place.xml` line 24

```xml
<geom size="0.01 0.01 0.01" type="box" ... mass="2" friction="2.0 0.05 0.0001">
```

The object is a 2cm cube (volume ≈ 8 cm³) with **mass = 2kg**. This gives a density of ~250 kg/m³ — lighter than water (1000 kg/m³) but surprisingly heavy for a chess piece (which should be ~20-50g). More importantly, this means **the arm has to support 2kg** when it "grips" in the ascend scenario. The finger joints have `armature=100` and `damping=1000`, which is very stiff. The arm mocap weld `solimp="0.9 0.95 0.001" solref="0.02 1"` with a 2kg hanging object will create oscillatory settling forces that push the gripper sideways, contributing to tube drift in ascend.

**Fix:** Reduce object mass to `0.050` (50g, realistic for a chess piece). This also makes the physics more physically meaningful.

---

## Summary Table

| # | Bug | Severity | Impact |
|---|---|---|---|
| 1 | Curriculum counter not shared with subprocess workers (silently broken) | 🔴 Critical | Curriculum never anneals |
| 2 | `compute_reward` called twice per step | 🔴 Critical | 2x CPU cost + latent confusion |
| 3 | Soft landing bonus always fires at success | 🟠 High | Policy learns nothing from approach style |
| 4 | Parent `_is_success` threshold mismatch (0.05 vs 0.01) | 🟠 High | Inflated eval metrics possible |
| 5 | Transit altitude reward too weak vs descent prior | 🟠 High | Root cause of all transit failures |
| 6 | Descend/ascend goal_pos sampling wastes RNG | 🟡 Medium | Seed divergence, wasted calls |
| 7 | Ascend closed-gripper ~1mm clearance causes physics noise | 🟡 Medium | Contributes to ascend tube drift |
| 8 | Transit object placed at goal XY — semantically confusing | 🟡 Medium | Mixed training signal |
| 9 | Test script doesn't verify arm position or tube compliance | 🟡 Medium | False confidence from passing tests |
| 10 | Soft-landing velocity units are m/step not m/s | 🟡 Medium | Documentation/understanding risk |
| 11 | `coordinate="local"` deprecated XML attribute | 🟡 Medium | Future compatibility |
| 12 | Object mass=2kg causes phantom forces in ascend | 🟡 Medium | Physics instability |

---

## Priority Fixes Before Restarting Training

1. **Fix Bug 1 (curriculum counter):** Remove `Value`/`CurriculumProgressCallback`. Let each env manage its own `total_env_steps`. The annealing will be per-env (each env independently progresses from 0→300k), which is fine since they all run ~equal episode counts.

2. **Fix Bug 5 (transit altitude):** Replace the `+0.5 * max(0, z - (SAFE_Z-0.02))` bonus with a proportional penalty: `-5.0 * max(0, SAFE_Z - z)`. This mirrors the tube penalty and creates immediate pressure to stay at altitude.

3. **Fix Bug 3 (soft landing double bonus):** Gate with `>= SUCCESS_THRESHOLD`.

4. **Fix Bug 7 (gripper clearance):** Change `CLOSED_FINGER_POS = 0.008`.

5. **Fix Bug 12 (object mass):** Change `mass="2"` to `mass="0.05"` in the XML.
