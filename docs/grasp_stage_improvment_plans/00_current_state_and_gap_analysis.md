# Current State & Gap Analysis

**Date:** 2026-05-05  
**Goal:** 100% single-scenario success + 100% full pick-and-place chain.  
**Current:** 86% full pipeline, 94% grasp, 96% descend.

---

## 1. What Has Been Built (As-Built Summary)

The first version of the Grasp Stage is fully operational. Key achievements:

| Component | Status | Detail |
|:---|:---|:---|
| `grasp_mode` flag | ✅ Done | `simulation.py` branches on `self.grasp_mode` to enable contact physics |
| `execute_grasp()` | ✅ Done | Zeroes velocity, fine-descends to GRASP_Z, closes fingers, verifies |
| Contact physics XML | ✅ Done | `solref="0.002 1"`, `solimp="0.99 0.999 0.001"`, `condim=6`, `mass=0.05kg` |
| `Kp=150000`, `damping=5000` | ✅ Done | High stiffness + damping eliminates vibration |
| `force_start_pos`, `force_cube_pos` | ✅ Done | Eval overrides in `task.py` |
| Weld constraint | ✅ Already stiff | `shared.xml` weld at `solref="0.01 1"` — no change needed |
| TUBE_BREACH_GRACE = 3mm | ✅ Done | In `task.py step()` |
| Finger close to HOLD_TARGET=0.012 | ✅ Done | Not 0.0 — prevents 2000N explosion |
| `_check_cube_held()` | ✅ Done | Monitors cube in grasp_mode during ascend/transit |

---

## 2. Root Causes of Remaining Failures

### Failure 1: Descend TUBE_BREACH at Board Edges (~4% of episodes)

**File:** `src/chess_env/simulation.py`, line 169  
**Code:** `self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)`

The `_set_action` method forces perfect vertical orientation at every RL step. The Fetch arm **cannot physically reach board edges (X≈1.10m) while pointing straight down** — its maximum vertical reach is ~1.05m from the shoulder. The policy learned to tilt the wrist to reach far positions, but this enforcement overrides that learned behavior. Result: arm jerks at edge positions → TUBE_BREACH.

**Empirical evidence:** Removing the enforcement → 100.0% descend success. Keeping it → 80%.

### Failure 2: RL Descends to GRASP_Z (6.5mm table clearance) — Unsafe

**File:** `src/chess_env/task.py`, line 332  
**Code:** `self.goal_pos = np.array([start_xy[0], start_xy[1], self.GRASP_Z])`

The RL policy has ±8mm precision variance. With a 6.5mm table clearance, any overshoot causes a table collision. Asking the RL policy to precision-land at 0.425m is outside its design envelope.

### Failure 3: No Scripted Vertical Correction Before Plunge

Current `execute_grasp` descends from wherever the RL left the arm (potentially tilted). The contact approach goes straight to GRASP_Z without correcting the wrist to vertical first. A tilted arm at GRASP_Z can miss the cube or cause a physics explosion.

### Failure 4: No Scripted XY Alignment

Current flow relies on the RL policy's XY precision (±8mm). The precondition check already rejects cases with XY error >10mm. But 5-9mm XY error is accepted and leads to partial cube contact → marginal grasps → drops during ascent.

### Failure 5: No Scripted Retract

After grasp, control is handed to the RL `ASCEND` policy while the arm is at GRASP_Z (6.5mm above the table). The RL policy's first action may jerk sideways, scraping the cube on the table. A scripted slow lift to HOVER_Z before handoff eliminates this.

---

## 3. The Fix: Macro/Micro Hybrid Architecture

**Principle:** RL handles gross navigation through safe airspace. Scripts handle the "danger zone."

```
[RL TRANSIT]        Home → SAFE_Z over cube XY
      ↓
[RL DESCEND]        SAFE_Z → HOVER_Z (0.460m)      ← 60mm clearance, safe for RL
      ↓
[SCRIPT: execute_grasp()]
   Phase 0: Halt & settle
   Phase 1: Vertical correction (safe at HOVER_Z = 35mm above cube top)
   Phase 2: XY perfect align + rotation safety check
   Phase 3: Plunge HOVER_Z → GRASP_Z (1mm/step, perfect vertical piston)
   Phase 4: Finger close (linear ramp, 150 steps)
   Phase 5: Hold & verify
   Phase 6: Retract GRASP_Z → HOVER_Z (1mm/step, no wrist snap)
      ↓
[RL ASCEND]         HOVER_Z → SAFE_Z               ← safe handoff height
      ↓
[RL TRANSIT]        SAFE_Z over cube XY → SAFE_Z over dst XY
```

---

## 4. Implementation Files Required

| Document | File to Change | Change |
|:---|:---|:---|
| `01_physics_and_environment.md` | `configs/env.yaml` | Add `hover_z: 0.460` |
| `01_physics_and_environment.md` | `src/chess_env/simulation.py` | Remove `set_mocap_quat` from `_set_action` |
| `02_execute_grasp_rewrite.md` | `src/chess_env/task.py` | Add `_move_mocap_to()`, rewrite `execute_grasp()` |
| `03_descend_to_hover.md` | `src/chess_env/task.py` | Init `HOVER_Z`, change descend target in `_reset_sim` + `soft_reset` |
| `04_execute_place.md` | `src/chess_env/task.py` | Add `execute_place()` for full pick+place |
| `05_testing.md` | test scripts | Updated gates for HOVER_Z flow |

---

## 5. Expected Outcome After All Changes

| Failure Mode | Current Rate | Expected Rate After Fix |
|:---|:---|:---|
| Descend TUBE_BREACH (crane mode) | ~4% | ~0% (no orientation enforcement) |
| Grasp fail (proximity explosion/miss) | ~4% | ~0.5% (scripted plunge from HOVER_Z) |
| Grasp fail (XY drift) | ~2% | ~0.5% (scripted XY align) |
| Ascent drop (wrist snap) | ~1% | ~0% (scripted retract) |
| **Full pipeline success** | **~86%** | **~99%** |

> **Note on retraining:** The `DESCEND` RL model currently targets `GRASP_Z` (0.425m). After the change, it will target `HOVER_Z` (0.460m). The task shape is identical; only the absolute Z target changes. Attempt with the existing model first — if convergence fails, warm-start retrain from checkpoint with HOVER_Z goal (≤200k steps).
