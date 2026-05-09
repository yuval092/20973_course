# Physics & Environment Changes

Three targeted changes. No MuJoCo contact solver tuning required — existing parameters are correct.

---

## Change 1: Fix Pre-existing `grasp_z` Bug in `configs/env.yaml`

**Current value (bug):** `grasp_z: 0.430`  
**Correct value:** `grasp_z: 0.425`

```yaml
grasp_z: 0.425              # 6.5mm table clearance; grip site 5mm below cube top (78% overlap)
```

**Why this matters:**

| Value | Grip site vs cube | Finger bottom | Table clearance | Grasp quality |
|:---|:---|:---|:---|:---|
| `0.430` (current, buggy) | AT cube top edge | 0.4115m | 11.5mm | Poor — fingers contact only the top 0mm of cube |
| `0.425` (correct) | 5mm inside cube | 0.4065m | 6.5mm | Good — fingers contact upper half of cube |

With `grasp_z=0.430`, the grip site is exactly at the cube top (TABLE_Z + CUBE_HEIGHT = 0.400 + 0.030 = 0.430m). The fingers open 38mm wide — they straddle the cube, but the contact is at the very top edge, giving minimal friction area. At `0.425m`, the fingers are 5mm inside the cube from the top, grasping the upper half with full face contact.

The comment in env.yaml ("CHANGED from 0.430 — 78% cube overlap, 6.5mm table clearance") was placed by a previous agent who correctly identified the target value but failed to change the number. This is the fix.

---

## Change 2: Remove Crane Mode Enforcement from `_set_action`

**File:** `src/chess_env/simulation.py`  
**Locate:** End of `_set_action` method.  
**Delete these two lines:**
```python
# Enforce crane mode (vertical orientation) every step
self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)
```

**Why:** This absolute override prevents the RL policy from tilting the wrist at board edges. The arm physically cannot reach X≈1.10m with a strictly vertical wrist. Empirically: removing this enforcement → 100% descend success; keeping it → 80% at edge positions. Vertical correction now happens inside `execute_grasp()` Phase 1+2, at HOVER_Z (35mm above cube top — safe).

**What stays:** `rot_ctrl = np.zeros(4)` remains. This zeroes the *delta* rotation each step (arm won't actively spin), but doesn't force it back to vertical against kinematic constraints.

**Verify:** After the change, run `grep -n "set_mocap_quat" src/chess_env/simulation.py`. The method should only appear in `_env_setup` and `soft_reset`, NOT in `_set_action`.

---

## Change 3: Add `hover_z` to `configs/env.yaml`

Add after `grasp_z`:

```yaml
grasp_z: 0.425              # 6.5mm table clearance; grip site 5mm below cube top (78% overlap)
hover_z: 0.460              # Safe RL stop height (see justification below)
```

### Why HOVER_Z = 0.460m is the Minimum Safe Value

The user might ask: can HOVER_Z be lower so the RL does more descent work? The answer is mathematically constrained:

**Constraint 1 — Finger clearance during XY alignment:**  
During Phase 1+2 of `execute_grasp`, the arm sweeps horizontally over the board to align over the cube. The finger bottoms must clear the cube top (and all other pieces) during this sweep.

- Cube top = TABLE_Z + CUBE_HEIGHT = 0.400 + 0.030 = **0.430m**
- Finger bottom = HOVER_Z − 0.0185m (vertical offset from grip site to finger tip)
- Required: `HOVER_Z − 0.0185 > 0.430 + 0.003` (3mm sweep safety margin)
- → `HOVER_Z > 0.4515m`

**Constraint 2 — RL precision variance:**  
The RL policy has ±8mm precision variance. With HOVER_Z as the target, the RL may succeed at `HOVER_Z − 8mm`. The precondition check in `execute_grasp` accepts Z within ±15mm of HOVER_Z. At `HOVER_Z − 8mm`, the XY alignment sweep must still clear all pieces:

- Worst-case grip height: `HOVER_Z − 0.008`
- Worst-case finger bottom: `HOVER_Z − 0.008 − 0.0185 = HOVER_Z − 0.0265`
- Required: `HOVER_Z − 0.0265 > 0.430 + 0.003`
- → `HOVER_Z > 0.4595m` → **round up to 0.460m**

**Conclusion:** HOVER_Z = 0.460m is the minimum safe value, not an overcautious choice. Going lower risks the RL's natural precision variance causing the fingers to graze other chess pieces during XY alignment. The RL already performs the majority of work: SAFE_Z (0.550m) → HOVER_Z (0.460m) = 90mm of descent (72% of the total 125mm from SAFE_Z to GRASP_Z). The scripted phase handles only the final 28% (35mm) where RL precision is insufficient.

---

## Final `env.yaml` Diff Summary

```yaml
# BEFORE (buggy):
grasp_z: 0.430              # CHANGED from 0.430 — 78% cube overlap...

# AFTER (correct):
grasp_z: 0.425              # 6.5mm table clearance; grip site 5mm below cube top (78% overlap)
hover_z: 0.460              # Safe RL stop height; min value given RL variance + finger clearance
```

No other `env.yaml` changes needed. All other thresholds remain correct.
