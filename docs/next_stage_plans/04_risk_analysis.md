# Risk Analysis & Edge Cases for Scenario Chaining

## Risk Summary Table

| Risk | Probability | Severity | Mitigation |
|:---|:---|:---|:---|
| Alignment fails to converge | Low | High | 200-step budget; abort chain on failure |
| Arm still moving after halt + zeroing | Very Low | High | Active qvel zeroing + confirmation check |
| Finger state inconsistency | Medium | High | Scripted 50-step transition + validation |
| Transit precision too poor for alignment | Medium | Medium | Logged as quality metric; no cascade |
| Model confusion at scenario boundary | Low | Medium | Velocity gate + alignment = clean obs |
| Arm at wrong Z entering next scenario | Low | High | Z-band sanity check at transition |
| Object joint disturbed by qvel zeroing | N/A | High | Zero robot DOF only (indices 0–14), not object |
| "Too-Perfect Start" model overfitting | Low | Low | Inject optional Gaussian noise if observed |

---

## Risk 1: Alignment Convergence Failure

**The Problem:**
After halt, the arm is within ±10mm of the nominal waypoint. The scripted settle loop
must converge within 3mm in 200 steps. If the arm is in an unusual joint configuration
or near a joint limit, the gain-based approach may not converge.

**Analysis:** The settle loop (`_settle_arm_to_start`) is already proven in `_reset_sim`
for exactly this task. The arm starts within 10mm of target (well within the working
range of the loop) and the 200-step budget is generous — typical convergence is 20–50
steps for similar displacements.

**Mitigation:**
1. Set `ALIGN_TOLERANCE = 3mm` (well inside the 5mm drift limit)
2. Abort chain immediately if convergence fails — log position, steps used, final error
3. In debug logs, print the full settle trajectory to diagnose any stuck cases

**Residual risk:** If transit precision is consistently near 10mm (maximum allowed),
the alignment loop must move ~10mm to converge. This is fine for the loop but indicates
the model needs more training. The convergence failure itself is diagnostic.

---

## Risk 2: Arm Not Fully Halted After Active Zeroing

**The Problem:**
The halt phase issues hold actions AND then actively zeros `data.qvel[:ARM_JOINT_COUNT]`.
Could the arm still be moving after this?

**Analysis:** After `mj_forward`, the simulation fully recomputes arm kinematics from
the current `qpos`. Since `qvel` was just zeroed and `mocap_pos` is held fixed, the
arm is at rest by definition. The confirmation check (`speed < 0.5mm/s`) is a sanity
guard against implementation errors, not a real physical possibility.

**Mitigation:** The confirmation check exists for this reason. If it somehow fires, it
indicates a bug in the zeroing logic (e.g., wrong joint indices), which should fail
loudly rather than silently.

---

## Risk 3: Finger State Contamination

**The Problem:**
Between scenarios, the finger state must transition correctly:

| Transition | Required Action |
|:---|:---|
| Transit → Descend | Close → Open (50 scripted steps) |
| Descend → Ascend | Open → Close (50 scripted steps) |
| Ascend → Transit | No change needed |
| Transit → Transit | No change needed |
| Descend → Transit | Open → Close (50 scripted steps) |
| Ascend → Descend | Close → Open (50 scripted steps) |

If the scripted transition is skipped or incomplete, the next scenario begins with
wrong finger state. The `FINGER_FAULT` detector crashes immediately if fingers deviate
>3mm from target.

**Mitigation:**
1. `soft_reset` executes the same 50-step scripted loop as `_reset_sim` Phase 2
2. After scripted transition, perform same Phase 3 validation:
   ```python
   l_pos = get_joint_qpos("robot0:l_gripper_finger_joint")
   assert abs(l_pos - target_qpos) < 0.0005
   ```
3. Validation failure raises `RuntimeError` immediately — no silent propagation

---

## Risk 4: Transit Precision Too Poor for Reliable Alignment

**The Problem:**
The design requires alignment to converge from ±10mm → within 3mm. But if transit
systematically ends 9.5mm off nominal (near the 10mm success boundary), every chain
sees the alignment loop work at maximum stretch.

**Analysis:** This is a model quality metric, not a framework failure. The alignment
loop handles 10mm displacements reliably. What it reveals is that transit precision
is insufficient for consistent high-quality piece placement.

**Mitigation:**
- Log `align_final_error_mm` and `align_steps_used` per transition
- Track the distribution of errors across episodes
- Use this data to set minimum transit precision requirements for the V6 model
- No cascade: poor transit precision → slightly longer alignment → same tube quality

**Contrast with the rejected design:** If we had used `tube_center = actual arm position`,
poor transit precision would silently place pieces in the wrong location. The alignment
design converts position error into observable diagnostic data instead.

---

## Risk 5: Model Observation Discontinuity at Scenario Boundary

**The Problem:**
At the moment of `soft_reset`, the model sees a sudden change:
- `relative_dist_for_model` (obs[6:9]) changes as the new goal is set
- `scenario_id_vec` (obs[11:14]) changes from e.g. [1,0,0] to [0,1,0]

**Analysis:** Low risk with proper transition sequencing. After the complete halt and
alignment phases:
- `grip_velp` (obs[20:22]) ≈ [0,0,0] — arm is at rest
- `grip_pos` (obs[0:3]) is at the nominal waypoint
- The observation looks like a normal "step 1" of a fresh episode

The model was trained starting from the exact nominal waypoint positions with zero
velocity. The `soft_reset` recreates those exact conditions.

**Mitigation:** The halt + alignment phases ensure the observation presented to the
model is indistinguishable from a training-time episode start.

---

## Risk 6: Z-Height Mismatch at Transition

**The Problem:**
If the arm ends a scenario at the wrong Z height, the next scenario starts from an
unexpected height. Example: transit ends at z=0.541 instead of SAFE_Z=0.550 (9mm low).

**Analysis:** The alignment phase targets the exact nominal exit waypoint including Z.
So alignment corrects Z as well as XY. After alignment, the arm is at SAFE_Z within 3mm.

**Mitigation:** The alignment loop already handles this — it converges in all 3 dimensions,
not just XY. Add a Z-sanity diagnostic log at transition (not abort):

```python
expected_z = SCENARIO_EXIT_Z[current_scenario]
if abs(grip_pos[2] - expected_z) > 0.015:
    log.warning(f"Z mismatch before alignment: arm at z={grip_pos[2]:.4f}, "
                f"expected z={expected_z:.4f} (delta={abs(grip_pos[2]-expected_z)*1000:.1f}mm)")
```

---

## Risk 7: Object Joint Disrupted by Velocity Zeroing

**The Problem:**
In future scenarios, the arm will hold a chess piece. If the halt phase zeros
`data.qvel[:]` (all joints), it also zeros the cube's velocity. For a cube being held
while the arm is stationary, the cube's velocity should be ~zero anyway — but if
the arm is accelerating or the cube is mid-fall, zeroing its velocity would be a
violent physics discontinuity.

**Current state:** No cube in simulation yet. This risk is currently theoretical.

**Mitigation (built into the design now):**
```python
ARM_JOINT_COUNT = 7   # Fetch arm DOF
data.qvel[:ARM_JOINT_COUNT] = 0.0    # ARM JOINTS ONLY
data.qacc[:ARM_JOINT_COUNT] = 0.0    # ARM JOINTS ONLY
# data.qvel[ARM_JOINT_COUNT:] is NOT touched — object physics unaffected
```

The arm joints are always the first 7 DOF in Fetch MuJoCo. Object joints follow.
This design is safe for current (no cube) and future (with cube) scenarios.

---

## Risk 8: Model Version Mismatch (Pre-V6 Model)

**The Problem:**
The current `models/latest_model.zip` was trained before the 5mm drift limit.
It crashes immediately on descend/ascend (TUBE_BREACH on step 1, drift 6–8mm > 5mm).

**Mitigation:**
- Transit-only chains (`transit,transit`) can be tested now with the current model
- Full descend/ascend chains require the V6 model (training begun 2026-05-02)
- The eval_sequence.py script logs the model path and drift limit in every output
  header so there is no ambiguity about what was tested

**Priority ordering:**
1. Test transit-only chains with current model
2. Once V6 model certified (≥90% individual scenario success), test full chains
3. Full 6-scenario chain only when V6 passes Tier 1 and Tier 2 chains

---

## Risk 9: "Too-Perfect Start" — Model Overfitting to Noisy Training Distribution

**The Problem:**
During training, `env.reset()` slightly randomizes the initial arm position and adds
noise to the starting state. In chaining, `soft_reset` delivers a mathematically clean
start: arm exactly at the nominal waypoint, velocity exactly zero. The model has never
seen this exact condition in training — it has only seen states with small random offsets
and residual velocities.

Some RL models overfit to the noisy training distribution and behave erratically when
given "perfect" initial conditions, because the observation falls outside the support of
the training data.

**Assessment:** Low risk. The model is trained to navigate from varied positions toward
the goal — a zero-error starting observation is simply "already at the waypoint" which
the model should not find confusing. The larger concern is zero velocity, but the model
sees near-zero velocity in every successful episode and was trained with a braking reward
that encourages exactly this.

**Detection:** If chain Tier 1 (`transit,transit`) shows Step 2 performing measurably
worse than Step 1 (isolated) despite clean transitions, this is the likely cause.

**Mitigation (only if observed):**
Inject a small Gaussian noise into `mocap_pos` at the end of `soft_reset`:

```python
if noise_scale > 0:
    noise = np.random.normal(0, noise_scale, size=3)
    noise[2] = 0.0  # Do not perturb Z — preserve height invariant
    env_unwrapped.data.mocap_pos[0][:3] += noise
    mujoco.mj_forward(env_unwrapped.model, env_unwrapped.data)
```

`noise_scale = 0.002` (2mm sigma) is a reasonable starting point — large enough to
break the "too-perfect" condition, small enough not to add meaningful position error.
This should **not** be enabled by default; enable only if the performance regression is
observed and confirmed to be caused by this effect.

---
*Next: [05 — Implementation Plan](./05_implementation_plan.md)*
