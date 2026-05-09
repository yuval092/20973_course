# Descend-to-HOVER_Z & State Machine Changes

The RL DESCEND scenario must target `HOVER_Z` (0.460m) instead of `GRASP_Z` (0.425m).
This requires changes in three places in `task.py`.

---

## Change 1: Descend Goal in `_reset_sim`

**File:** `src/chess_env/task.py`  
**Locate:** The `elif self.current_scenario == "descend":` block in `_reset_sim`.

**Current code:**
```python
elif self.current_scenario == "descend":
    self.tube_center_xy = start_xy.copy()
    arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
    self.goal_pos = np.array([start_xy[0], start_xy[1], self.GRASP_Z])
```

**Change to:**
```python
elif self.current_scenario == "descend":
    self.tube_center_xy = start_xy.copy()
    arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
    self.goal_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
```

---

## Change 2: Descend Goal in `soft_reset`

**File:** `src/chess_env/task.py`  
**Locate:** Inside `soft_reset`, the block that constructs `new_goal_pos` for a `"descend"` transition.

Find where `soft_reset` builds the descend goal and change the Z from `self.GRASP_Z` to `self.HOVER_Z`. The exact line depends on context — search for `GRASP_Z` inside `soft_reset` and update it.

**Current code (typical pattern):**
```python
new_goal_pos = np.array([nominal_xy[0], nominal_xy[1], self.GRASP_Z])
```

**Change to:**
```python
new_goal_pos = np.array([nominal_xy[0], nominal_xy[1], self.HOVER_Z])
```

---

## Change 3: Ascend Start in `_reset_sim`

**File:** `src/chess_env/task.py`  
**Locate:** The `elif self.current_scenario == "ascend":` block in `_reset_sim`.

**Current code:**
```python
elif self.current_scenario == "ascend":
    arm_start_pos = np.array([start_xy[0], start_xy[1], self.GRASP_Z])
    self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
```

**Change to:**
```python
elif self.current_scenario == "ascend":
    arm_start_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
    self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
```

**Why:** After `execute_grasp()` returns, `soft_reset` is called to transition to ASCEND. The arm is physically at HOVER_Z. If `_reset_sim` (for isolated scenario training) still places the arm at `GRASP_Z`, it creates a training inconsistency. ASCEND should start from HOVER_Z.

---

## Change 4: TUBE_BREACH Limit During Descend

**File:** `src/chess_env/task.py`  
**Locate:** `step()` method, the drift check for descend.

The TUBE_BREACH boundary monitors how far the arm drifts from `tube_center_xy` during descent. With the weld already stiff (`solref="0.01 1"`) and no crane-mode handcuffing, drift is reduced. Keep the current `TUBE_BREACH_GRACE = 0.003` (3mm). No change needed.

---

## Change 5: Evaluate Whether DESCEND Model Generalizes to HOVER_Z

The existing descend checkpoint (`best_model_combined.zip`) was trained to reach `GRASP_Z = 0.425m`. After the change, the target is `HOVER_Z = 0.460m`.

**Decision tree:**

```
Try existing model with HOVER_Z goal
        ↓
Evaluate 100 episodes of DESCEND only
        ↓
Success ≥ 95%?  ──YES──→  No retraining needed. Proceed.
        ↓ NO
Warm-start retrain from checkpoint:
  - Change only: goal_pos Z from GRASP_Z → HOVER_Z in env.yaml
  - Train for 200k steps from checkpoint
  - Verify ≥ 98% descend success
```

**Why the model likely generalizes:**  
The task shape is identical — fly from SAFE_Z to a target XY at a target Z, stopping within 10mm. The absolute Z changed by 35mm, but relative distances and reward gradients are the same. The policy learned the motion primitives, not the absolute altitude.

**If retraining is needed (backup plan):**
```bash
python train.py \
  --scenario descend \
  --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
  --steps 200000 \
  --fixed-drift
```

---

## State Machine Diagram (After Changes)

```
[TRANSIT]   goal: home → cube_XY at SAFE_Z
    ↓ soft_reset("descend", [cube_XY, HOVER_Z])
[DESCEND]   goal: SAFE_Z → HOVER_Z        ← NEW target (was GRASP_Z)
    ↓ execute_grasp()  [entirely scripted]
      Phase 0-6 runs; exits with arm at HOVER_Z, cube held
    ↓ soft_reset("ascend", [cube_XY, SAFE_Z])
[ASCEND]    goal: HOVER_Z → SAFE_Z        ← starts from HOVER_Z (was GRASP_Z)
    ↓ soft_reset("transit", [dst_XY, SAFE_Z])
[TRANSIT]   goal: cube_XY at SAFE_Z → dst_XY at SAFE_Z
    ↓ execute_place()  [scripted, future work]
[DESCEND]   goal: SAFE_Z → HOVER_Z over dst_XY
    ↓ execute_place() scripted release + retract
[ASCEND]    goal: HOVER_Z → SAFE_Z over dst_XY
    ↓ soft_reset("transit", [home_XY, SAFE_Z])
[TRANSIT]   goal: dst_XY → home at SAFE_Z
```
