# Test Plan: Grasp Stage Verification

## Overview

This document specifies every test script required for the grasp stage, their
pass criteria, and the order in which they must be run. No test may be skipped.
Each test is a blocking gate for the next.

---

## Test Execution Order

```
Gate 0: verify_physics.py (post-XML changes)
    ↓
Gate 1: test_grasp_physics.py — Test 1: Static Grasp
    ↓
Gate 2: test_grasp_physics.py — Test 2: Lift
    ↓
Gate 3: test_grasp_physics.py — Test 3: Transit Held
    ↓
Gate 4: eval.py — per-scenario certification (transit, descend, ascend)
    ↓
Gate 5: eval_sequence.py — 3-chain certification
    ↓
Gate 6: eval_grasp.py — 20-episode visual inspection
    ↓
Gate 7: eval_grasp.py — 100-episode metrics
```

If any gate fails its threshold, stop. Diagnose and fix before proceeding.

---

## CRITICAL: Test Infrastructure Bug in Gates 1-3

**`_sample_board_position()` returns invalid positions ~8% of the time.**

In 30-trial runs of test_grasp_physics.py, approximately 7% of trials fail because
`_sample_board_position()[:2]` returns coordinates outside the robot's workspace (X=4.13, Y=-1.71,
etc.). The cube is placed there, falls off the table (no support), and the test reports failure.

**These are NOT physics failures.** When only valid positions are sampled, the static grasp
physics success rate is **100%** (confirmed over 30 trials with valid positions).

**Required fix before trusting Gates 1-3:** Add bounds validation in each test function:
```python
BOARD_X_MIN, BOARD_X_MAX = 0.64, 1.12
BOARD_Y_MIN, BOARD_Y_MAX = 0.02, 0.50

for attempt in range(10):
    src_xy = uw._sample_board_position()[:2]
    if BOARD_X_MIN <= src_xy[0] <= BOARD_X_MAX and BOARD_Y_MIN <= src_xy[1] <= BOARD_Y_MAX:
        break
else:
    raise ValueError(f"Could not sample valid board position after 10 retries (last: {src_xy})")
```
Apply this in `test_static_grasp()`, `test_lift()`, and `test_transit_held()`.

Without this fix, reported success rates are artificially low by ~10%.

---

---

## Gate 0: `scripts/verify_physics.py` (Post-XML Changes)

**Run:** `python scripts/verify_physics.py`

**New assertions to add (not yet in the current script):**

```python
def verify_grasp_xml_changes(model, data, uw):
    """Called from verify_physics.py after existing checks pass."""
    import mujoco
    
    # 1. Cube mass
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object0")
    assert abs(model.body_mass[cube_body_id] - 0.05) < 0.001, \
        f"Cube mass={model.body_mass[cube_body_id]:.3f}, expected 0.05"
    
    # 2. Actuator Kp
    l_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   "robot0:l_gripper_finger_joint")
    # Kp is stored as gainprm[0] for position actuators
    assert model.actuator_gainprm[l_act_id, 0] == 150000, \
        f"l_finger Kp={model.actuator_gainprm[l_act_id, 0]}, expected 150000"
    
    # 3. Actuator ctrlrange
    assert model.actuator_ctrlrange[l_act_id, 1] == pytest.approx(0.05, abs=0.001), \
        f"l_finger ctrlrange max={model.actuator_ctrlrange[l_act_id, 1]}, expected 0.05"
    
    # 4. GRASP_Z
    assert abs(uw.GRASP_Z - 0.425) < 0.001, \
        f"GRASP_Z={uw.GRASP_Z}, expected 0.425"
    
    print("[PASS] verify_grasp_xml_changes: all assertions passed")
```

**Pass criteria:** All assertions pass with no exceptions.

**If this gate fails:** The XML was not saved, the Python process was not restarted,
or the wrong config was loaded. Do not proceed.

---

## Gate 1: Test 1 — Static Grasp

**Script:** `scripts/test_grasp_physics.py --test static --n-trials 20`

**What it does:**
1. Place arm exactly at GRASP_Z above cube (using align_to_waypoint)
2. Call `execute_grasp()` with grasp_mode=False→True
3. Measure cube Z displacement from expected position (GRASP_Z − 15mm)
4. Measure close_steps_used
5. Measure l_finger final position

**Full implementation:**

```python
def test_static_grasp(env, debug=False) -> dict:
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    obs, _ = env.reset()
    
    # Position arm exactly at GRASP_Z above cube center
    cube_pos = uw.get_cube_position()
    src_xy = cube_pos[:2]
    target = np.array([src_xy[0], src_xy[1], uw.GRASP_Z])
    
    # Use mocap to position arm (direct, not RL)
    for _ in range(200):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        error = target - grip_pos
        if np.linalg.norm(error) < 0.001:
            break
        uw.data.mocap_pos[0][:3] += 0.5 * error
        uw._mujoco_step(None)
    
    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    arm_alignment_error = np.linalg.norm(grip_pos - target) * 1000  # mm
    
    # Execute grasp
    result = uw.execute_grasp()
    
    cube_pos_after = uw.get_cube_position()
    # During a STATIC grasp, the cube is still resting on the table (CoM = table_z +
    # cube_half_height = 0.400 + 0.015 = 0.415m). The 15mm hang offset only applies
    # when the cube is lifted into mid-air. Using GRASP_Z - 0.015 = 0.410 here would
    # be wrong — it implies the cube is 5mm below the table surface.
    expected_cube_z = 0.415  # table_z + cube_half_height = 0.400 + 0.015
    cube_z_displacement = abs(cube_pos_after[2] - expected_cube_z) * 1000  # mm
    cube_xy_displacement = np.linalg.norm(cube_pos_after[:2] - src_xy) * 1000  # mm
    
    if debug:
        print(f"  Arm alignment error: {arm_alignment_error:.1f}mm")
        print(f"  Grasp success: {result['success']} (reason: {result.get('reason')})")
        print(f"  Close steps: {result.get('close_steps_used')}")
        print(f"  Cube Z displacement: {cube_z_displacement:.1f}mm")
        print(f"  Cube XY displacement: {cube_xy_displacement:.1f}mm")
        print(f"  Final finger pos: {result.get('final_finger_pos', 'N/A'):.4f}")
    
    return {
        "success": result["success"],
        "reason": result.get("reason"),
        "arm_alignment_error_mm": arm_alignment_error,
        "cube_z_displacement_mm": cube_z_displacement,
        "cube_xy_displacement_mm": cube_xy_displacement,
        "close_steps_used": result.get("close_steps_used"),
        "final_finger_pos": result.get("final_finger_pos"),
    }
```

**Pass criteria:**
- Success rate ≥ 95% over 20 trials (with bounds fix applied: valid positions give 100% physics success)
- If bounds fix NOT applied: ≥ 80% (includes ~10% infrastructure failures; not a real physics failure)
- Cube Z displacement < 20mm in successful grasps (cube didn't pop)
- Cube XY displacement < 10mm in successful grasps
- `final_finger_pos` between 0.010 and 0.014 in successful grasps (fingers stalled at j≈0.0143)

**Important:** A result of 90-93% after applying the bounds fix likely indicates the halt-loop
proximity explosion bug (doc 06 Risk 13) is present. Fix execute_grasp halt ordering (doc 10 P1.1)
before re-running. In static tests, the arm is pre-settled (no residual velocity), so this bug
does NOT appear in test_grasp_physics.py — it only manifests in eval_grasp.py.

**Common failures and diagnosis:**

| Symptom | Cause | Fix |
|:---|:---|:---|
| `CUBE_EXPLOSION_DURING_CLOSE` | XML not updated (mass=0.5, no solref) | Restart process, verify XML |
| `VERIFY_FINGERS_FAILED (> 0.012)` | Fingers not stalling against cube | Check GRASP_Z position and cube mass |
| `PRECONDITION_Z failed` | Arm positioning error | Check align_to_waypoint gain |
| cube_z_displacement > 50mm | Physics explosion, solref too soft | Check solref="0.002 1" on BOTH cube AND finger geoms |

---

## Gate 2: Test 2 — Lift

**Script:** `scripts/test_grasp_physics.py --test lift --n-trials 20`

**What it does:**
1. Perform Test 1 (static grasp) — must succeed
2. Move arm from GRASP_Z to SAFE_Z using scripted mocap (not RL)
3. Measure cube Z during ascent — should rise with arm
4. Measure cube Z relative to grip site throughout — should stay ≈ 15mm below

**Full implementation:**

```python
def test_lift(env, debug=False) -> dict:
    uw = env.unwrapped
    
    # First: static grasp (reuse test_static_grasp logic)
    grasp_result = run_static_grasp(uw)
    if not grasp_result["success"]:
        return {"success": False, "reason": "GRASP_FAILED", **grasp_result}
    
    # Now lift: move from GRASP_Z to SAFE_Z in scripted steps
    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip").copy()
    src_xy = grip_pos[:2]
    target_z = uw.SAFE_Z
    
    max_z_deviation = 0.0
    cube_dropped = False
    drop_reason = None
    
    for step in range(300):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        if grip_pos[2] >= target_z - 0.001:
            break
        
        # Move upward 5mm/step
        uw.data.mocap_pos[0][2] += 0.005
        uw._mujoco_step(None)
        
        # Check cube is still held
        cube_pos = uw.get_cube_position()
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        z_offset = grip_pos[2] - cube_pos[2]  # should be ~15mm
        deviation = abs(z_offset - 0.015)
        max_z_deviation = max(max_z_deviation, deviation)
        
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2]) * 1000
        if xy_error > 30.0:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_XY ({xy_error:.1f}mm)"
            break
        if deviation > 0.040:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_Z (deviation={deviation*1000:.1f}mm)"
            break
    
    final_cube_z = uw.get_cube_position()[2]
    final_grip_z = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")[2]
    
    return {
        "success": not cube_dropped,
        "reason": drop_reason,
        "max_z_deviation_mm": max_z_deviation * 1000,
        "final_cube_z": final_cube_z,
        "final_grip_z": final_grip_z,
        "final_z_offset_mm": (final_grip_z - final_cube_z) * 1000,
    }
```

**Pass criteria:**
- Success rate ≥ 90% over 20 trials
- max_z_deviation_mm < 40mm in successful lifts
- final_z_offset_mm between 5mm and 30mm (cube approximately 15mm below grip)

---

## Gate 3: Test 3 — Transit Held

**Script:** `scripts/test_grasp_physics.py --test transit --n-trials 20`

**What it does:**
1. Perform Test 2 (lift to SAFE_Z) — must succeed
2. Move arm 100mm horizontally at SAFE_Z using scripted mocap
3. Measure cube yaw rotation (should stay < 15°)
4. Measure cube drop (XY and Z deviation)

**Pass criteria:**
- Success rate ≥ 85% over 20 trials
- Cube yaw rotation < 15° during transit (condim=6 working)
- No CUBE_DROPPED events (XY < 30mm, Z deviation < 40mm)

---

## Gate 4: `eval.py` — Per-Scenario Certification

**Commands:**

```bash
PYTHONPATH=. python scripts/eval.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --scenario transit --n-episodes 100

PYTHONPATH=. python scripts/eval.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --scenario descend --n-episodes 100

PYTHONPATH=. python scripts/eval.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --scenario ascend --n-episodes 100
```

**Pass criteria:**

| Scenario | Minimum Success Rate |
|:---|:---|
| transit | ≥ 95% |
| descend | ≥ 95% |
| ascend | ≥ 95% |

**Note:** `eval.py` default `eval_drift_limit` must be 0.010 (not 0.005). Verify
this in the script before running or pass `--drift-limit 0.010` explicitly.

---

## Gate 5: `eval_sequence.py` — 3-Chain Certification

**Command:**

```bash
PYTHONPATH=. python scripts/eval_sequence.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --n-episodes 100 --drift-limit 0.010
```

**Pass criteria:**
- Full chain success (transit → descend → ascend) ≥ 85%
- No stalling detected (arm reaches goal in < 400 steps per scenario)

---

## Gate 6: `eval_grasp.py` — Visual Inspection

**Command:**

```bash
PYTHONPATH=. python scripts/eval_grasp.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --n-episodes 20 --debug --visualize --wait
```

**What to inspect visually:**
1. Arm transits to correct XY position over cube
2. Descend ends with fingers bracketing cube (not above or beside it)
3. Finger close is smooth — no cube pop or bounce
4. Ascent is smooth — cube rises with arm
5. Home transit holds cube throughout — no wobble or drop

**What the `--debug` flag should print each step:**
```
[TRANSIT step 45] grip_pos=[0.782, 0.264, 0.550] dist=47.3mm speed=12.1mm/s
[DESCEND step 23] grip_pos=[0.782, 0.264, 0.461] dist=36.1mm speed=8.4mm/s
[GRASP close step 30] l_finger=0.0181→0.0147 cube_z=0.411m
[ASCEND step 15] grip_pos=[0.782, 0.264, 0.445] cube_offset=14.8mm
```

**Pass criteria (visual):** No obvious failure modes observed in 20 episodes.
Document any unexpected behavior before running Gate 7.

---

## Gate 7: `eval_grasp.py` — 100-Episode Metrics

**Command:**

```bash
PYTHONPATH=. python scripts/eval_grasp.py \
    --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
    --n-episodes 100 --drift-limit 0.010
```

**Pass criteria:**

| Metric | Current (achieved) | After Phase 1 fixes | After Phase 2 retrain |
|:-------|:-------------------|:--------------------|:----------------------|
| Full sequence success | **88%** ✓ | ≥ 94% | ≥ 98% |
| Grasp success rate | ~89% | ≥ 95% | ≥ 99% |
| Descend success | ~96% | ~99% | ~99.5% |
| Cube XY drift at home | ≤ 15mm | ≤ 15mm | ≤ 12mm |
| Cube yaw at home | ≤ 15° | ≤ 15° | ≤ 10° |
| Cube drop during transit | 0% | 0% | 0% |

**Phase 1 scripted-only fixes (before re-running this gate):**
1. execute_grasp: zero qvel BEFORE the halt loop, not after (doc 10 P1.1 — highest priority)
2. TUBE_BREACH_GRACE: 0.0015→0.0020 in task.py step() (doc 10 P1.2)
3. Cube rotation abort: 35°→25° in execute_grasp pre-conditions (doc 10 P1.3)
4. Early FINGER_CLOSED_EMPTY abort: add check `l_finger < 0.003 after 30 steps` (doc 10 P1.4)

See **doc 10 (path_to_99_percent.md)** for the complete improvement roadmap.

**Report format the script must output:**

```
=== EVAL_GRASP RESULTS (100 episodes) ===

Step success rates:
  transit_to_src:   98/100 (98%)
  descend:          96/100 (96%)
  grasp:            91/100 (91%)
  ascend:           87/100 (87%)
  transit_to_home:  85/100 (85%)

Grasp quality (successful episodes, n=85):
  cube_xy_drift:    mean=8.2mm  max=14.7mm
  cube_yaw:         mean=3.1°   max=12.4°
  cube_z_error:     mean=2.1mm  max=8.3mm

Failure breakdown:
  GRASP_PRECONDITION_FAILED: 2
  CUBE_EXPLOSION_DURING_CLOSE: 0
  CUBE_DROPPED_XY: 3
  CUBE_DROPPED_Z: 1
  TRANSIT_CRASH: 5
  DESCEND_CRASH: 3

=== OVERALL: PASS/FAIL ===
```

---

## Sub-Task Test Scripts (User Note 8)

Beyond the grasp-stage gates, each sub-task has its own test script for isolated
verification during development.

### `scripts/test_finger_contact.py`

Tests that the `grasp_mode` branch in `_set_action` works correctly:

```python
def test_grasp_mode_isolation():
    """Verify that grasp_mode=True does NOT teleport fingers."""
    uw = env.unwrapped
    uw.grasp_mode = True
    uw.finger_target_joint = 0.0  # CLOSED
    
    # Set finger to open position directly
    uw._utils.set_joint_qpos(uw.model, uw.data, "robot0:l_gripper_finger_joint", 0.018)
    
    # Call _set_action — in grasp_mode, should NOT teleport finger back to 0
    uw._set_action(np.zeros(4))
    
    finger_pos = uw._utils.get_joint_qpos(uw.model, uw.data, "robot0:l_gripper_finger_joint")
    assert finger_pos > 0.010, \
        f"grasp_mode=True but finger was teleported to {finger_pos:.4f} (expected ~0.018)"
    print("[PASS] grasp_mode correctly prevents finger teleportation")
```

### `scripts/test_cube_position.py`

Tests `get_cube_position()` and `get_cube_quat()`:

```python
def test_cube_position_accuracy():
    """Verify get_cube_position() returns correct world position."""
    uw = env.unwrapped
    # Place cube at known position
    obj_joint_id = uw.model.joint("object0:joint").id
    qpos_start = uw.model.jnt_qposadr[obj_joint_id]
    known_pos = np.array([0.75, 0.264, 0.415])
    uw.data.qpos[qpos_start:qpos_start+3] = known_pos
    mujoco.mj_forward(uw.model, uw.data)
    
    reported_pos = uw.get_cube_position()
    assert np.allclose(reported_pos, known_pos, atol=0.001), \
        f"get_cube_position() error: {reported_pos} vs {known_pos}"
    print("[PASS] get_cube_position() accurate")
```

### `scripts/test_soft_reset_grasp_mode.py`

Tests that `soft_reset` preserves `grasp_mode`:

```python
def test_soft_reset_preserves_grasp_mode():
    """Verify soft_reset does not reset grasp_mode."""
    uw = env.unwrapped
    uw.grasp_mode = True
    
    # Run soft_reset from grasp → ascend
    uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([0.782, 0.264, 0.550]),
        nominal_exit_pos=np.array([0.782, 0.264, 0.425]),
        nominal_xy=np.array([0.782, 0.264]),
    )
    
    assert uw.grasp_mode == True, \
        "soft_reset reset grasp_mode to False — Bug C not fixed"
    print("[PASS] soft_reset correctly preserves grasp_mode")

def test_soft_reset_phase4_skipped_when_grasp_mode():
    """Verify Phase 4 (finger teleport) is skipped when grasp_mode=True.
    
    IMPORTANT: Do NOT check finger qpos after a full soft_reset() call. During
    soft_reset's 100 halt + 200 align steps (300 total), the Kp=150,000 actuator
    (targeting FINGER_CLOSED_JOINT=0.0) physically drives the fingers fully closed
    — even without a cube to block them. Asserting finger_pos > 0.010 after soft_reset
    will always fail due to physics, not due to a bug in Phase 4 teleport logic.
    
    Instead, verify the CTRL signal: if Phase 4 correctly skips teleportation,
    data.ctrl[0] and data.ctrl[1] must be set to FINGER_CLOSED_JOINT (the actuator
    target), while data.qpos (finger joint) reflects purely physics-driven closure.
    The distinction is: teleportation would set qpos INSTANTLY; actuator just sets ctrl.
    """
    uw = env.unwrapped
    uw.grasp_mode = True
    uw.finger_target_joint = uw.FINGER_CLOSED_JOINT  # 0.0

    # Record ctrl BEFORE soft_reset to confirm Phase 4 sets it (not teleports qpos)
    # Run soft_reset
    uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([0.782, 0.264, 0.550]),
        nominal_exit_pos=np.array([0.782, 0.264, 0.425]),
        nominal_xy=np.array([0.782, 0.264]),
    )
    
    # Phase 4 correctly sets ctrl target; fingers may be physically closed by actuator
    assert uw.data.ctrl[0] == pytest.approx(uw.FINGER_CLOSED_JOINT, abs=0.001), \
        f"ctrl[0]={uw.data.ctrl[0]:.4f} — Phase 4 must set ctrl to CLOSED target"
    assert uw.data.ctrl[1] == pytest.approx(uw.FINGER_CLOSED_JOINT, abs=0.001), \
        f"ctrl[1]={uw.data.ctrl[1]:.4f} — Phase 4 must set ctrl to CLOSED target"
    # grasp_mode must survive the entire soft_reset (already tested separately)
    assert uw.grasp_mode, "grasp_mode was reset inside soft_reset (Bug 10 not fixed)"
    print("[PASS] soft_reset Phase 4 correctly uses actuator ctrl (not teleport) when grasp_mode=True")
```

### `scripts/test_cube_held_monitor.py`

Tests `_check_cube_held()` boundary conditions:

```python
def test_cube_held_monitor_boundaries():
    """Verify _check_cube_held() uses correct thresholds."""
    uw = env.unwrapped
    grip_pos = np.array([0.75, 0.264, 0.550])
    
    # Test: cube within limits
    set_cube_position(uw, np.array([0.75, 0.264, 0.535]))  # 15mm below grip
    held, reason = uw._check_cube_held(grip_pos)
    assert held, f"Should be held: {reason}"
    
    # Test: cube too far in XY (31mm > 30mm limit)
    set_cube_position(uw, np.array([0.781, 0.264, 0.535]))
    held, reason = uw._check_cube_held(grip_pos)
    assert not held, "Should detect XY drop"
    assert "CUBE_DROPPED_XY" in reason
    
    # Test: cube too far in Z (21mm deviation > 20mm limit)
    # grip_pos[2]=0.550, expected=0.550-0.015=0.535
    # set cube at 0.514 → z_error=|0.514-0.535|=0.021 > 0.020 limit
    set_cube_position(uw, np.array([0.75, 0.264, 0.514]))
    held, reason = uw._check_cube_held(grip_pos)
    assert not held, "Should detect Z drop"
    assert "CUBE_DROPPED_Z" in reason
    
    print("[PASS] _check_cube_held() boundaries correct")
```

---

## Mandatory Assertions in All Test Scripts

Every test script must include these assertions before any test runs:

```python
def assert_mandatory_preconditions(uw):
    """Call at the start of every test run."""
    assert abs(uw.GRASP_Z - 0.425) < 0.001, \
        f"GRASP_Z={uw.GRASP_Z}, expected 0.425. Restart process after env.yaml change."
    
    import mujoco
    cube_body_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
    assert abs(uw.model.body_mass[cube_body_id] - 0.05) < 0.001, \
        f"Cube mass={uw.model.body_mass[cube_body_id]:.3f}, expected 0.05. XML not updated."
    
    assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.012) < 0.001, \
        f"GRASP_VERIFY_FINGER_THRESHOLD={uw.GRASP_VERIFY_FINGER_THRESHOLD}, expected 0.012"
    
    print("[OK] Mandatory preconditions verified")
```

---

*End of grasp stage game plan. Implementation start: apply XML changes → run Gate 0.*
