# Implementation Order & Risk Registry

---

## Recommended Implementation Order

Execute in this exact sequence to minimize debugging surface at each step.

### Step 1 — Environment Changes (30 min)
1. Add `hover_z: 0.460` to `configs/env.yaml`
2. Add `self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)` to `task.py __init__`
3. **Verify:** `python -c "from src.chess_env.task import ChessTaskEnv; print('OK')"`

### Step 2 — Remove Crane Mode Enforcement (5 min)
1. Delete the two `set_mocap_quat` lines from `simulation.py _set_action` (lines ~169)
2. **Verify:** `grep -n "set_mocap_quat" src/chess_env/simulation.py` — should only appear in `_env_setup` and `soft_reset`, NOT in `_set_action`

### Step 3 — Add `_move_mocap_to` Helper (10 min)
1. Add `_move_mocap_to()` method to `task.py` (see doc `02_execute_grasp_rewrite.md` Part A)
2. **Verify:** `python -c "from src.chess_env.task import ChessTaskEnv; t = ChessTaskEnv.__dict__; print('_move_mocap_to' in t)"`

### Step 4 — Update Descend/Ascend Goals (15 min)
1. Change descend goal in `_reset_sim` from `GRASP_Z` → `HOVER_Z`
2. Change ascend start in `_reset_sim` from `GRASP_Z` → `HOVER_Z`
3. Find and change descend goal in `soft_reset` from `GRASP_Z` → `HOVER_Z`
4. **Verify:** Run `python eval.py --scenario descend --episodes 20` — check success rate

### Step 5 — Rewrite `execute_grasp` (60 min)
1. Replace `execute_grasp()` entirely with the implementation from `02_execute_grasp_rewrite.md`
2. **Verify:** Run Gate 1 (static grasp physics, 20 trials)
3. **Verify:** Run Gate 3 (full grasp pipeline, 20 episodes)

### Step 6 — Add `execute_place` (45 min)
1. Add `execute_place()` from `04_execute_place.md`
2. **Verify:** Run Gate 4 (full chain, 10 episodes)

### Step 7 — Full Certification
1. Run all gates in `05_testing_and_certification.md` at full episode counts

---

## Risk Registry

### Risk A: DESCEND Model Doesn't Reach HOVER_Z
- **Probability:** Low (same task shape, 35mm Z shift)
- **Detection:** Gate 2 success < 95%
- **Mitigation:** Warm-start retrain from checkpoint (200k steps, HOVER_Z goal)
- **Do NOT:** Change HOVER_Z to be closer to GRASP_Z — the safety margin is required

### Risk B: VERTICAL_QUAT Removal Causes Arm Instability During RL
- **Probability:** Very low (arm was originally trained without this enforcement)
- **Detection:** TUBE_BREACH rate goes UP (not down) after removing enforcement
- **Mitigation:** If instability persists, add `rot_ctrl` damping in `_set_action` instead of absolute override
- **Fallback:** Limit enforcement to `_env_setup` and `soft_reset` only (not per-step)

### Risk C: Phase 1 Rotation Correction Fails at Kinematic Limits
- **Probability:** Low (arm is at safe HOVER_Z with no obstacles)
- **Detection:** `_move_mocap_to` hits `max_steps=80` without converging
- **Mitigation:** Increase `max_steps` to 150; add detection for kinematic failure (grip far from mocap after max steps) → abort with `ROTATION_FAILED` reason
- **Do NOT:** Reduce `tolerance` below 1mm — risks infinite loop at singularities

### Risk D: Plunge XY Drift During Descent
- **Probability:** Low (arm is stiff-welded to vertical mocap; no horizontal forces)
- **Detection:** `VERIFY_XY_FAILED` after plunge
- **Mitigation:** After each plunge step, re-lock `mocap_pos[:2]` to cube XY explicitly (already in the critical ordering — `_set_action` resets mocap, then we set target again)
- **If needed:** Add per-step XY re-centering in Phase 3 loop

### Risk E: Cube Drops During Retract (Phase 6)
- **Probability:** Low (300N hold force >> 0.49N gravity)
- **Detection:** `CUBE_DROPPED_DURING_RETRACT`
- **Mitigation:** Check finger stall position — should be ~0.014 with cube. If fingers are at 0.012 (full target), the ramp is working correctly. If cube drops at 10mm altitude, the initial grasp was marginal.
- **Diagnostic:** Print `l_finger` every 5 retract steps to detect slip

### Risk F: Cube Rotation > 25° Causes High Abort Rate
- **Probability:** Low (condim=6 torsional friction should prevent rotation)
- **Detection:** `CUBE_ROTATED` reason in grasp results > 2%
- **Mitigation:** The torsional friction coefficient in cube XML — increase the third element of `friction="2.0 0.005 0.0001"` from `0.0001` to `0.001`
- **Note:** The 25° threshold is correct; do NOT increase it. The fingertip stub problem is real.

### Risk G: `execute_place` Cube Slides on Placement
- **Probability:** Medium (releasing fingers while cube is at GRASP_Z may cause it to bounce)
- **Detection:** `PLACE_XY_FAILED` with drift > 20mm
- **Mitigation:** Release fingers more slowly (increase `release_steps` from 80 to 150); add a 30-step settle before retracting; check cube final Z (should be TABLE_Z + CUBE_HEIGHT/2)

---

## Do-Not-Touch List

These were empirically validated. Do not change them without new data:

| Parameter | Value | Why frozen |
|:---|:---|:---|
| Actuator Kp | 150,000 | kp=75k and kp=100k both caused FINGER_CLOSED_EMPTY spike |
| Cube mass | 0.05 kg | Heavier cube increases inertia → proximity explosions |
| `solref` (contacts) | `"0.002 1"` | Slower contact (0.005+) → ghosting at Kp=150k |
| `solimp` (contacts) | `"0.99 0.999 0.001"` | Softer (0.9 0.95) → ghosting at Kp=150k |
| Finger damping | 5000 | Default 50 → vibration against cube |
| GRASP_Z | 0.425 m | 6.5mm table clearance — minimum safe; 0.420 causes collisions |
| FINGER_CLOSED target | 0.012 | Target 0.0 → 2000N explosion; 0.012 → 300N stable hold |
| HOVER_Z | 0.460 m | 35mm above cube top — safe for XY sweep; 11.5mm clearance |
