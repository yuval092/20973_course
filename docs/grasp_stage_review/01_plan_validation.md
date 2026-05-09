# Grasp Stage Plan Validation: Bug Analysis & Errors Found

*Reviewer: Independent code/plan analysis*
*Date: 2026-05-04*

This document audits the **current updated plan** (`docs/grasp_stage_plans/`) for residual bugs,
inconsistencies, and logic errors after the author's fixes. It assumes the author's changes (Findings
A-H from the response document) have been incorporated. This review is **fresh eyes only** — it
identifies problems that even the author's response might not have fully addressed.

---

## 1. BUG IN `execute_grasp()`: Contact Approach Step Size Too Large (Sub-Critical)

### Location
`docs/grasp_stage_plans/05_implementation_plan.md`, Step 3e, Sub-Phase 1

### The Bug
The plan specifies `step_vec = 0.5 * error`, then clamps to `0.003m (3mm)` max step. However,
`step_vec` is applied to `mocap_pos` directly after `_set_action` resets it. The issue is the
initial step before the arm has caught up:

```python
step_vec = 0.5 * error       # half of (say) 9mm gap = 4.5mm
if np.linalg.norm(step_vec) > 0.003:
    step_vec = ... * 0.003    # clamped to 3mm/step
```

**3mm/step at 20ms/step = 150mm/s**, which is 3× faster than the arm's stable tracking speed of ~50mm/s.
The mocap is commanding a speed the weld constraint cannot instantly achieve. This causes the
arm to lag and the contact approach to overshoot → oscillate → time out.

### Evidence from Test Log
Lines 85–91 in the test log:
```
[move_arm] FAILED to reach [ 0.43068776 -0.35633542  0.55 ]. Ended at [ 0.45940808...]. Error: 56.7mm
```
Out-of-range targets cause the arm to fail entirely. In contact approach, a too-large step
size for the **Z-only movement** at close range is less catastrophic but still introduces
jitter as the contact forces build.

### Fix
For Sub-Phase 1 (fine-descent), reduce the max step to **1mm/step** (50mm/s):
```python
if np.linalg.norm(step_vec) > 0.001:  # 1mm max for fine-descent (not 3mm)
    step_vec = step_vec / np.linalg.norm(step_vec) * 0.001
```
The plan's 50-step budget with 1mm/step covers 50mm — still 5× the maximum 10mm gap.

---

## 2. BUG IN `_check_cube_held()`: Missing `grip_pos` Parameter in Signature

### Location
`docs/grasp_stage_plans/05_implementation_plan.md`, Step 3d; vs `docs/grasp_stage_plans/03_grasp_scenario.md`

### The Bug
The plan in doc 03 (authoritative) defines `_check_cube_held` with `grip_pos` as a parameter:
```python
def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
```

But the current **implemented code** (confirmed by git diff) defines it **without** that parameter:
```python
def _check_cube_held(self) -> bool:
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
```

And Step 3d in the implementation plan also passes `grip_pos` explicitly in the call from `step()`:
```python
cube_held, drop_reason = self._check_cube_held(gripper_pos)
```

This is a **direct incompatibility**: either the parameter-free version (currently implemented)
is correct OR the parameter-accepting version (in the plan and called from `step()`) is correct.
They cannot both be right.

The plan's version is better because it avoids a redundant MuJoCo query inside `_check_cube_held()`
when `grip_pos` is already available in `step()`. But if the new plan is re-implemented,
the call site in `step()` **must** pass `gripper_pos`.

### Verdict
The plan is internally consistent. The current *implementation* diverges from the plan.
The fix is straightforward but the plan should be explicit: **the call in `step()` must
pass `gripper_pos`**, and the test in `test_cube_held_monitor.py` must also pass `grip_pos`.

---

## 3. BUG IN `execute_grasp()`: GRASP_VERIFY_FINGER_THRESHOLD Check Is Inverted

### Location
`docs/grasp_stage_plans/05_implementation_plan.md`, Step 3e, Sub-Phase 3

### The Bug
The plan's verification check says:
```python
if l_finger > self.GRASP_VERIFY_FINGER_THRESHOLD:
    result["reason"] = f"VERIFY_FINGERS_FAILED (finger={l_finger:.4f} > threshold=...)"
    return result
```

A valid grasp stalls at `j≈0.0143`. The threshold is `0.012`. So:
- Valid grasp: `l_finger = 0.0143`, check: `0.0143 > 0.012` = **True** → FALSELY FAILS ✗

This is wrong. The plan reverses the semantics between the verification step and the threshold.

A valid grasp means **the finger is blocked by the cube** (i.e., `l_finger > threshold`).
An empty close means **the finger went all the way** (i.e., `l_finger ≈ 0.000 < threshold`).

The correct logic should be:
```python
if l_finger < self.GRASP_VERIFY_FINGER_THRESHOLD:  # < not >
    result["reason"] = f"FINGER_CLOSED_EMPTY (finger={l_finger:.4f} < threshold)"
    return result
```

This same inversion was in the original agent code and **partially corrected** in the
current implementation. However, the **plan text itself** in doc 05 Step 3e still has
the inverted check (`> threshold`). This will confuse the next implementer.

### Evidence
- The current implementation has it correct (`< threshold`)
- Doc 03 line 145 says: `if l_finger > GRASP_VERIFY_FINGER_THRESHOLD:` with the comment
  "Fingers above 0.016 means they haven't reached the cube" — this is **backwards**.
  If `l_finger > 0.016`, the finger is at its OPEN position and hasn't closed at all.
  If `l_finger < 0.012`, the finger went all the way through (cube not present).

### Fix Required in Plan
Update doc 03 Sub-Phase 3 and doc 05 Step 3e to use `if l_finger < threshold:`.

---

## 4. DOCUMENTATION INCONSISTENCY: `GRASP_VERIFY_FINGER_THRESHOLD` in Doc 03

### Location
`docs/grasp_stage_plans/03_grasp_scenario.md`, Sub-Phase 3, line 130

### The Problem
Doc 03 still shows `GRASP_VERIFY_FINGER_THRESHOLD = 0.016` despite the author's Finding C
updating it to `0.012`. The author says "Update doc 01, doc 05, doc 09 tests, memory, and env.yaml"
but does not mention doc 03. This is an oversight — doc 03 has the original wrong value.

---

## 5. BUG IN `soft_reset()` FINGER VALIDATION: Incorrect Guard Logic

### Location
Current implementation in `task.py` (confirmed by git diff)

### The Bug
The current implementation uses this guard:
```python
is_valid_grasp = (self.finger_target_joint == self.FINGER_CLOSED_JOINT and 
                  l_pos >= self.GRASP_VERIFY_FINGER_THRESHOLD)
if not is_valid_grasp:
    raise RuntimeError(...)
```

**This is wrong.** This `is_valid_grasp` check only runs when `abs(l_pos - finger_target_joint) > 0.0005`.
For the GRASP→ASCEND transition:
- `finger_target_joint = 0.0` (CLOSED)
- `l_pos ≈ 0.0143`
- `abs(0.0143 - 0.0) = 0.0143 > 0.0005` → enters the check
- `is_valid_grasp = (True and 0.0143 >= 0.012)` = `True`
- Does NOT raise → correct

But the plan (doc 06 Risk 10 and doc 08) specifies the **simpler and correct** fix:
```python
if not self.grasp_mode:  # skip validation entirely when holding cube
    if abs(l_pos - self.finger_target_joint) > 0.0005:
        raise RuntimeError(...)
```

The current implementation's approach is more fragile: it tries to "allow" a specific
configuration rather than simply "skip validation when in grasp mode." If `finger_target_joint`
is somehow not FINGER_CLOSED_JOINT during a grasp (e.g., future PLACE stage where fingers
open mid-hold), the current guard will wrongly raise.

The plan's `if not self.grasp_mode:` guard is cleaner and semantically correct.

---

## 6. BUG: `_check_cube_held()` Uses Wrong Z Reference During Grasping

### Location
Current implementation in `task.py` (git diff) vs. plan in doc 03

### The Bug
The plan says:
```python
z_error = abs(cube_pos[2] - (grip_pos[2] - 0.015))  # 15mm offset
```

The current implementation says:
```python
z_dist = abs(grip_pos[2] - cube_pos[2])  # No 15mm offset!
```

During the *grasp* phase (cube on table at `0.415m`, grip at `GRASP_Z=0.425m`):
- Expected gap = `0.425 - 0.415 = 0.010m = 10mm`
- `CUBE_HELD_Z_LIMIT = 0.020m = 20mm`
- Current check: `abs(0.425 - 0.415) = 0.010 < 0.020` → passes ✓ (accidentally correct)

During *ascent* (cube at `0.535m`, grip at `0.550m`):
- Expected gap if held = `0.550 - 0.535 = 0.015m = 15mm`
- But if cube *is actually dropping* and is at `0.520m`:
  - Plan check: `abs(0.520 - (0.550 - 0.015)) = abs(0.520 - 0.535) = 0.015 < 0.020` → **FALSE NEGATIVE** (not detected!)
  - Current check: `abs(0.550 - 0.520) = 0.030 > 0.020` → correctly detected ✓

Paradoxically, the **current implementation is more sensitive** than the plan in this case.
But the Z limit is using the wrong reference conceptually. A cube falling 15mm from the 
expected position should be detected; the plan's offset changes what "expected" means.

The real question is: what is `CUBE_HELD_Z_LIMIT` guarding against? The plan says:
> "20mm max cube Z deviation relative to grip-15mm"

If `z_error = |cube_z - (grip_z - 0.015)|` and limit = 0.020m, the cube is allowed to be
anywhere from `grip_z - 0.015 - 0.020 = grip_z - 0.035` to `grip_z - 0.015 + 0.020 = grip_z + 0.005`.
The lower bound `grip_z - 0.035` covers a significant drop; the upper bound `grip_z + 0.005`
would catch the cube being pushed *up* (unlikely but valid to catch).

### Verdict
The plan's version with the 15mm offset is semantically correct for monitoring mid-air holds.
The current version without the offset can produce false-passes for mild cube drops.
**Fix the implementation to use the plan's formula.**

---

## 7. MISSING IMPLEMENTATION: `force_start_pos` Override in `_reset_sim`

### Confirmed by Git Diff
The diff shows `force_start_pos` is declared in `__init__` but the `_reset_sim` override is
**not implemented**. The `if self.force_start_pos is not None:` branch does not appear
anywhere in the diff for `_reset_sim`. The author's analysis (Bug F) confirms this.

### Impact
Every `eval_grasp.py` run with `force_start_pos = home_pos` silently ignores the home position
and starts from a **random board square**. The reported "100% transit success" is therefore
meaningless — the arm is going from random square to random square, which is within training
distribution regardless. The HOME position at `[0.680, 0.264]` was **never actually tested**.

---

## 8. MISSING IMPLEMENTATION: Contact Approach Phase in `execute_grasp()`

### Confirmed by Git Diff
The current `execute_grasp()` skips Sub-Phase 1 entirely. It goes:
1. 50-step halt
2. Close fingers
3. Hold

The Contact Approach (scripted fine-descent to `GRASP_Z`) is completely absent.

### Impact
The author's analysis (Bug E) correctly identifies this as the **single biggest improvement
opportunity**. The 91% descend → 78% grasp drop (13-point gap) is directly attributable
to this missing step. When the descend succeeds at `GRASP_Z + 9mm = 0.434m`, the fingers
close onto **air above the cube** rather than properly bracketing it.

### Quantification
From the test log, arm alignment errors are consistently 0.3–1.0mm (excellent XY alignment)
but the critical gap is Z height: with success at up to 10mm above GRASP_Z, some attempts
close fingers with only 65% cube overlap. The contact approach would guarantee 78% overlap.

---

## 9. BUG IN `verify_physics.py`: Kp Assertion Checks Wrong Value

### Confirmed by Author's Analysis (Bug G)
The current `verify_physics.py` checks `kp != 500` when it should check `kp != 150000`.
This means Gate 0 **always fails** on the correctly-configured system.

### Impact
Critical: Gate 0 can never pass with the correctly-configured `kp=150000` XML, blocking
the entire verification pipeline.

---

## 10. BUG IN `execute_grasp()`: Missing Return Fields

### Confirmed by Author's Analysis (Bug H)
`result.get("close_steps_used")` and `result.get("final_finger_pos")` are called by
`test_grasp_physics.py` but not returned by the current `execute_grasp()`.

### Evidence from Test Log
Every trial shows:
```
Close steps: None
```
This confirms `close_steps_used` is never populated.

---

## Summary Table

| # | Issue | Severity | In Plan? | Fixed in Impl? |
|:--|:------|:---------|:---------|:---------------|
| 1 | Contact approach step size too large (3mm vs 1mm) | Medium | Plan says 3mm | N/A (not implemented) |
| 2 | `_check_cube_held()` parameter signature mismatch | Medium | Plan has param | Impl has no param — diverges |
| 3 | Finger threshold check inverted (`>` vs `<`) | **Critical** | **Plan still wrong** | Impl is correct |
| 4 | Doc 03 still has 0.016 threshold | Low | Missing update | N/A |
| 5 | `soft_reset` guard — fragile pattern vs clean `grasp_mode` guard | Low | Plan has correct pattern | Impl uses fragile pattern |
| 6 | `_check_cube_held()` missing 15mm Z offset | Medium | Plan has offset | Impl missing offset |
| 7 | `force_start_pos` not implemented in `_reset_sim` | **Critical** | Plan has it | **Missing** |
| 8 | Contact Approach Phase missing from `execute_grasp()` | **Critical** | Plan has it | **Missing** |
| 9 | `verify_physics.py` checks `kp != 500` not `kp != 150000` | **Critical** | Plan says 150000 | **Wrong** |
| 10 | `execute_grasp()` missing return fields | Low | Plan has them | **Missing** |

