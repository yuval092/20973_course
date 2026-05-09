# Testing & Certification Gates

Execute gates in order. Each gate must pass before proceeding to the next.

---

## Gate 0: Static Assertions

```bash
python verify_physics.py
```

Assert ALL of:

| Check | Expected Value | File |
|:---|:---|:---|
| Weld solref | `"0.01 1"` | `chess_env/assets/shared.xml` |
| `hover_z` config | `0.460` | `configs/env.yaml` |
| `grasp_z` config | `0.425` | `configs/env.yaml` |
| Cube mass | `0.05` | `chess_env/assets/pick_and_place.xml` |
| Cube `solref` | `"0.002 1"` | `chess_env/assets/pick_and_place.xml` |
| Cube `solimp` | `"0.99 0.999 0.001"` | `chess_env/assets/pick_and_place.xml` |
| Actuator Kp | `150000` | `chess_env/assets/pick_and_place.xml` |
| Finger damping | `5000` | `chess_env/assets/robot.xml` |
| No `set_mocap_quat` in `_set_action` | absent | `src/chess_env/simulation.py` |
| `HOVER_Z` initialized in `__init__` | present | `src/chess_env/task.py` |
| `_move_mocap_to` method exists | present | `src/chess_env/task.py` |

**Pass criteria:** All assertions pass. Zero failures.

---

## Gate 1: Static Grasp Physics (100 trials)

```bash
python test_grasp_physics.py --trials 100
```

Tests the physics of the grasp in isolation. Places arm at fixed HOVER_Z directly over a valid cube position, runs `execute_grasp()`, verifies cube is held.

**CRITICAL — Apply bounds fix before running:**  
`_sample_board_position()` returns invalid positions ~8% of the time (outside the robot's reach). These cause guaranteed failures unrelated to physics. Add a validation + retry loop:

```python
def _valid_board_position(env, max_retries=10):
    for _ in range(max_retries):
        pos = env._sample_board_position()[:2]
        # Check within valid range: x in [0.64, 1.12], y in [-0.02, 0.54]
        if 0.64 <= pos[0] <= 1.12 and -0.02 <= pos[1] <= 0.54:
            return pos
    raise RuntimeError("Could not sample a valid board position")
```

**Pass criteria:**
- Physics success ≥ 98% (on valid positions)
- FINGER_CLOSED_EMPTY rate = 0% (linear ramp prevents this)
- CUBE_EXPLOSION rate = 0%
- Zero table collisions during plunge

---

## Gate 2: DESCEND Scenario (200 episodes)

```bash
python eval.py --scenario descend --episodes 200
```

This gate verifies the RL model still achieves target HOVER_Z without crane mode enforcement.

**Pass criteria:**
- Descend success ≥ 97%
- TUBE_BREACH rate ≤ 1% (down from ~4%)
- No episodes truncated by table collision (impossible at HOVER_Z)

**If descend success < 95%:** The existing model may not generalize to HOVER_Z. See the retraining guide in `03_descend_to_hover_and_state_machine.md`.

---

## Gate 3: Full Grasp Pipeline (100 episodes)

```bash
python eval_grasp.py --episodes 100
```

Full sequence: TRANSIT → DESCEND → execute_grasp() → ASCEND.

**Pass criteria:**
- Per-stage success: Transit ≥99%, Descend ≥97%, Grasp ≥98%, Ascend ≥99%
- Full pipeline success ≥ 95%
- Breakdown by failure type: document rates for each failure reason

---

## Gate 4: Full Pick-and-Place Chain (50 episodes)

```bash
python eval_chain.py --episodes 50
```

Full 13-step sequence: TRANSIT → DESCEND → execute_grasp → ASCEND → TRANSIT → DESCEND → execute_place → ASCEND → TRANSIT(home).

**Pass criteria:**
- Full chain success ≥ 90%
- Cube XY placement error ≤ 15mm (p50), ≤ 25mm (p95)
- Zero physics explosions

---

## Gate 5: Stress Test — Board Edge Coverage (50 episodes)

```bash
python eval_grasp.py --episodes 50 --edge-positions-only
```

Force all episodes to use extreme board positions (x > 1.05m or x < 0.70m) to specifically test that removing crane mode enforcement enables edge reach.

**Pass criteria:**
- Full pipeline success at edge positions ≥ 90% (was ~80% with crane mode enforcement)
- No TUBE_BREACH from wrist handcuff behavior

---

## Regression Gate: Transit & Ascend (200 episodes each)

```bash
python eval.py --scenario transit --episodes 200
python eval.py --scenario ascend  --episodes 200
```

Verify that physics changes and state machine updates did not break existing RL scenarios.

**Pass criteria:**
- Transit success ≥ 99%
- Ascend success ≥ 99%

---

## Failure Mode Reference Table

| Reason Code | Phase | Root Cause | Fix |
|:---|:---|:---|:---|
| `PRECONDITION_SPEED` | Phase 0 | Residual RL momentum > 5mm/s | Increase halt settle to 25 steps |
| `PRECONDITION_Z` | Phase 0 | DESCEND model didn't reach HOVER_Z | Retrain descend for HOVER_Z |
| `PRECONDITION_FINGERS_NOT_OPEN` | Phase 0 | Finger teleport malfunction in _reset_sim | Check descend finger open loop |
| `CUBE_ROTATED` | Phase 2 | Cube yaw > 25° after transit vibration | Increase torsional friction in cube XML |
| `PLUNGE_FAILED` | Phase 3 | Arm didn't descend to GRASP_Z | Increase `_move_mocap_to` max_steps |
| `FINGER_CLOSED_EMPTY` | Phase 4 | Missed cube during plunge | Check Phase 2 XY alignment tolerance |
| `CUBE_EXPLOSION` | Phase 4 | Physics instability | Check cube solref/solimp and finger damping |
| `VERIFY_XY_FAILED` | Phase 5 | Cube moved during close | Reduce close ramp step size |
| `CUBE_DROPPED_DURING_RETRACT` | Phase 6 | Grasp lost during lift | Check finger stall position (should be ~0.014) |

---

## Success Rate Targets Summary

| Gate | Scenario | Target (After All Changes) |
|:---|:---|:---|
| 0 | Static assertions | 100% |
| 1 | Static grasp physics | ≥ 98% |
| 2 | DESCEND to HOVER_Z | ≥ 97% |
| 3 | Full grasp pipeline | ≥ 95% |
| 4 | Full pick+place chain | ≥ 90% |
| 5 | Edge positions stress test | ≥ 90% |
| Regression | Transit | ≥ 99% |
| Regression | Ascend | ≥ 99% |
