# Grasp Stage: Executive Summary

## Objective

Execute the first complete physical chess pick: the robotic arm travels from a **Home
Position**, transits to hover above the cube, descends and **grasps** the cube, lifts
it, transits back to the Home Position. This is the first time the cube is physically
picked up and moved.

## The Six-Step Sequence

```
[1] HOME_POSITION (initial state — arm parked at safe height outside board)
        ↓
[2] TRANSIT: home → SAFE_Z (0.550m) above cube src_xy
        ↓
[3] DESCEND: SAFE_Z → GRASP_Z (0.425m) at src_xy (fingers open)
        ↓
[4] GRASP: scripted close fingers on cube (between DESCEND and ASCEND)
        ↓
[5] ASCEND: GRASP_Z → SAFE_Z at src_xy (fingers actuator-driven, cube held)
        ↓
[6] TRANSIT: src_xy → home at SAFE_Z (still holding cube)
```

No cube release in this stage. The sequence ends at the Home Position with the cube
held. Release will be the next stage.

---

## What Is New in This Stage

The previous stage (scenario chaining) kept the cube hidden (`hide_object=True`).
This stage **enables the cube** (`hide_object=False`) and adds the **GRASP** phase:
a scripted phase that closes the fingers onto the physical cube while the arm is
stationary at GRASP_Z.

New components:
1. **Physics hardening** of the XML assets (see doc 02)
2. **`grasp_mode`** flag in `simulation.py` and `task.py` — switches from teleport to
   actuator-driven finger control (see doc 03)
3. **`execute_grasp()`** scripted method in `task.py` (see doc 03)
4. **Home Position** definition and navigation (see doc 04)
5. **Cube position tracking** — cube hangs ~15mm below the grip site when held
6. **`eval_grasp.py`** — evaluates the full pick sequence
7. **`test_grasp_physics.py`** — verifies physics before any chain evaluation

---

## What Is NOT Changing

- The RL model is used as-is. **No retraining** is required for cube integration.
  The Phase 9 trick (finger state zeroed in observation) makes the policy blind to
  whether fingers are teleported or actuator-driven.
- The transit, descend, and ascend RL scenarios are unchanged.
- `eval_sequence.py` remains unchanged — it evaluates pure movement chains (no cube).
- `eval.py` remains the per-scenario evaluator.

---

## GRASP_Z Selection: 0.425m

The grip target height must satisfy two constraints:

```
Finger bottom = GRASP_Z + 0.020 − 0.0385 = GRASP_Z − 0.0185

Must clear table (0.400m) by ≥ 5mm (position noise budget):
GRASP_Z ≥ 0.400 + 0.005 + 0.0185 = 0.424m  →  use 0.425m
```

| GRASP_Z | Finger Bottom | Table Clearance | Cube Overlap | Verdict |
|:---|:---|:---|:---|:---|
| 0.415 | 0.3965m | **−3.5mm** | 100% | **TABLE COLLISION** |
| 0.420 | 0.4015m | 1.5mm | 95% | Too risky |
| **0.425** | **0.4065m** | **6.5mm** | **78%** | **✓ Use this** |
| 0.430 | 0.4115m | 11.5mm | 62% | Safe but poor overlap |

**GRASP_Z = 0.425m** gives 6.5mm of table clearance (safe against ~3mm position
noise) and 23.5mm of vertical finger-cube overlap (78% of cube height). The model
was trained with GRASP_Z=0.430 and reliably achieves ±9mm precision — a 5mm change
is within its capability.

---

## Cube Z-Offset During Hold

When the cube is grasped, its center-of-mass hangs **approximately 15mm below the
grip site** due to gravity pulling it down:

```
Cube CoM Z ≈ grip_site_Z − 0.015m
```

At SAFE_Z (0.550m) during transit: cube Z ≈ 0.535m, cube bottom ≈ 0.520m. This is
120mm above the table surface (0.400m) and 90mm above the tallest chess piece
(assuming 30mm pieces). No collision risk during transit.

This offset is accounted for in:
- `_check_cube_held()` Z threshold
- `evaluate_grasp_quality()` expected cube Z
- `CUBE_HELD_Z_LIMIT` (40mm tolerance around the expected offset)

---

## Critical Physics Findings (Challenging the Audit Document)

The `physics_audit_comparison.md` document contains errors. See doc 02 for the full
analysis. Summary:

### Wrong in audit:
- **"Deep 18.5mm overlap / cradling at GRASP_Z=0.430"** — 18.5mm overlap is 62%.
  We move to GRASP_Z=0.425 for 78% overlap.
- **"2mm elastic compression"** — Finger gap at joint=0: **1.8mm**. A 30mm cube
  stops finger closure at joint≈0.014. The fingers cannot physically reach joint=0
  with cube present. Hold force = Kp × joint_error = 150,000 × 0.014 = **2,100N
  per finger** (actuator force, not compression).
- **Absolute finger enforcement** — The current `_set_action` teleports fingers
  every step, completely bypassing MuJoCo contact physics.

### Confirmed correct in audit:
- Cube mass: 0.5kg → 0.05kg
- condim=6 and softened solref/solimp on cube and fingers
- Actuator Kp: 60,000 → 150,000

---

## Model for This Stage

Use `checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip`.

This is the latest training session (2026-05-04). It supersedes the earlier
`20260502_171928` checkpoint. See doc 07 for full training analysis and
certification protocol.

---

## Success Criteria

| Test | Target |
|:---|:---|
| Grasp success rate (cube stays through ascent) | ≥ 85% |
| Cube XY drift at home (vs nominal src_xy) | ≤ 15mm |
| Cube Z-rotation at home (twist during transit) | ≤ 15° |
| Cube drop during transit | 0% |
| Full sequence success (home→transit→descend→grasp→ascend→home) | ≥ 75% |

---

## Execution Order

1. **Training certification** — verify current model achieves ≥95% single-scenario
2. **Physics hardening** — apply XML changes, run `test_grasp_physics.py` (doc 09)
3. **Code changes** — implement `grasp_mode`, `execute_grasp()`, cube tracking
4. **Grasp isolation test** — `test_grasp_physics.py` all 4 tests pass
5. **Full sequence** — `eval_grasp.py` with 50 episodes
6. **Stress test** — 200-episode evaluation

---

*Documents: 02 Physics | 03 Grasp Scenario | 04 Sequence | 05 Implementation |
06 Risk Analysis | 07 Training | 08 Soft Reset Reference | 09 Test Plan*
