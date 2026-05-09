# Implementation Autopsy: What Actually Happened vs. What Should Have Happened

*Reviewer: Independent analysis*
*Date: 2026-05-04*

This document reconstructs what went wrong during the agent's implementation session, explains
the causal chain for each failure, and validates whether the author's proposed fixes will actually
resolve them.

---

## Section 1: What the Agent Got Right

### 1.1 Physics XML Changes
The agent correctly applied **all 5 XML changes** from the plan:

| Change | Target | Result |
|:-------|:-------|:-------|
| `mass=0.05` on cube | `pick_and_place.xml` | ✓ Correct |
| `condim=6, solref=0.002 1, solimp=0.99 0.999 0.001` on cube | `pick_and_place.xml` | ✓ Correct |
| `kp=150000, ctrlrange=0 0.05` on actuators | `pick_and_place.xml` | ✓ Correct |
| `condim=6, solref=0.002 1, solimp=0.99 0.999 0.001` on fingers | `robot.xml` | ✓ Correct (verified by git diff) |
| `solref=0.02 1` on weld | `shared.xml` | ✓ Correct |

The agent's report claims `kp=100`, but the **actual diff shows `kp=150000`**. The agent's
written report was simply wrong; the implementation was correct. This is a documentation error,
not a code error.

### 1.2 grasp_mode Flag in `_set_action`
The simulation.py change correctly adds the `grasp_mode` branch that skips `set_joint_qpos`
teleportation. This is the fundamental enabler for actuator-driven grasping and was done correctly.

### 1.3 Cube Drop Detection in `step()`
The `_check_cube_held()` integration into `step()` with the `grasp_mode` guard is present and
functionally correct (modulo the Z-offset issue noted in 01_plan_validation.md).

### 1.4 `get_cube_position()` and `get_cube_quat()`
Both methods are implemented correctly and match the plan specification.

### 1.5 finger_target State Machine Guard
The `if not grasp_mode:` guard on the descend finger state machine is correctly applied.
This prevents the descend scenario from forcing fingers open when a cube is held (future
PLACE stage compatibility).

---

## Section 2: What the Agent Got Wrong (Root Causes)

### 2.1 Missing Contact Approach Phase

**What happened:** The agent implemented `execute_grasp()` with:
1. 50-step halt
2. Set `grasp_mode = True`
3. Close fingers for 150 steps
4. Hold for 50 steps

Sub-Phase 1 (Contact Approach: scripted fine-descent to GRASP_Z) was never implemented.

**Why it happened:** The plan document (doc 05 Step 3e) is long and detailed. The agent
implemented the simpler path: halt → close → verify. The contact approach requires a
mini-movement loop *before* `grasp_mode` is activated, which is a more subtle sequence.

**Why the plan author should have been more explicit:** The plan says the contact approach
is "the single most impactful step" and "CRITICAL: DO NOT SKIP THIS PHASE." Despite this
warning, the implementation skipped it. The lesson: **critical warnings in long documents
are not read.** The plan should have structured execute_grasp as a numbered checklist
with gate conditions, not prose.

**Consequence:** 13-point drop in grasp success rate (91% descend → 78% grasp).

### 2.2 Missing `force_start_pos` Override

**What happened:** The agent added `self.force_start_pos = None` to `__init__` but never
implemented the `if self.force_start_pos is not None: arm_start_pos = self.force_start_pos.copy()`
branch in `_reset_sim`.

**Why it happened:** The plan's description of this (doc 05, Step 3b) uses pseudocode and
describes where to insert the code rather than providing a drop-in patch. The agent
implemented the attribute declaration but missed the `_reset_sim` body.

**Why this is particularly insidious:** There is **no error**. The code runs. The test
"passes" (100% transit success) because the arm goes from one random board square to another
random board square — which is within training distribution. The silent failure is the worst
kind: it produces plausible-looking metrics that mask the real situation.

**Consequence:** The home→board transit has never been tested from the actual Home Position.
Every "100% transit success" result in eval_grasp.py is statistically meaningless for the
intended test.

### 2.3 Incorrect `verify_physics.py` Assertion

**What happened:** The agent wrote `if kp != 500:` instead of `if kp != 150000:`.

**Why it happened:** At some point during the session, a `kp=500` value was proposed
(the agent's walkthrough mentions "reducing Kp from 150,000 to 100" which was wrong, and
the agent was iterating on physics parameters). The verify script was updated with an
intermediate trial value rather than the final value.

**Consequence:** Gate 0 fails on a correctly-configured system, blocking the entire
verification pipeline.

### 2.4 Inverted Finger Threshold Logic

**What happened:** The original code checked `l_finger < threshold` to mean "FINGER_CLOSED_EMPTY"
(i.e., fingers closed fully without encountering the cube). This is correct. But at some point
during debugging, the logic became confused and the check became `l_finger > threshold`.

**Why it happened:** The semantics are genuinely counterintuitive:
- Cube present + fingers stalled at j=0.0143: **success** (finger is partially open)
- No cube + fingers fully closed at j=0.000: **failure** (finger is fully closed)

The natural reading of "FINGER_CLOSED_EMPTY" suggests the check should trigger when the
finger is "too closed" (j < threshold). The agent inverted the check during a debug session
and the wrong version propagated.

**Consequence:** All valid grasps were rejected as failures. The `test_grasp_physics.py`
results originally showed 0% success because this check was failing every successful grasp.
The agent then lowered the threshold to 0.012 to "fix" the failure, which actually fixed
the wrong bug (threshold change) for the right symptom (0% success rate).

**Current state:** The threshold is correctly set at 0.012 AND the check is correctly
inverted back to `< threshold`. This works. But the plan's doc 03 still documents the
inverted check.

---

## Section 3: The Agent Author's Proposed Fixes — Will They Work?

### Fix A: `solref="0.002 1"` on BOTH geoms
**Will it work?** Already implemented and confirmed effective (92% static grasp success,
91% in 25-trial log). ✓

### Fix B: `solimp="0.99 0.999 0.001"` on BOTH geoms
**Will it work?** Already implemented and confirmed effective. ✓

### Fix C: `GRASP_VERIFY_FINGER_THRESHOLD = 0.012`
**Will it work?** Already implemented. The threshold is correct. However, the doc 03
still shows 0.016 — documentation inconsistency. ✓ (in code), ✗ (in doc 03)

### Fix D: Friction analysis correction (8,571× not 42,000×)
**Will it work?** Informational update only. No code change. ✓

### Fix E: Contact Approach Phase implementation (Bug E)
**Will it work if correctly implemented?** Yes. The plan's specification is mathematically
sound. The 50-step budget with 3mm/step max covers 150mm but the author recommends 1mm/step
for fine descent. See Issue #1 in 01_plan_validation.md.

**Estimated impact:** Should recover approximately 8-10 percentage points of grasp success
by guaranteeing 78% cube overlap rather than the current 48-78% variable overlap.

### Fix F: `force_start_pos` override in `_reset_sim` (Bug F)
**Will it work if correctly implemented?** Yes. The plan's pseudocode is correct. The
implementation is straightforward: add 4 lines inside the transit scenario initialization.

**Estimated impact:** Primarily affects test validity. May slightly reduce transit success
rate since going from a fixed home position (always the same starting joint configuration)
may produce more consistent behavior than random starts. Unlikely to hurt success rate.

### Fix G: `verify_physics.py` kp assertion (Bug G)
**Will it work?** Yes, trivial one-line fix: `kp != 150000`. ✓

### Fix H: Missing return fields in `execute_grasp()` (Bug H)
**Will it work?** Yes. `close_steps_used` should be tracked (easy counter in the close loop).
`final_finger_pos` should be captured after the close loop. Both are diagnostic only. ✓

---

## Section 4: Remaining Risk After All Fixes Are Applied

Even after implementing all the fixes (Bugs A-H), the following risks remain:

### 4.1 The 8% Physics Explosion Rate Is Irreducible (Short Term)
Static grasp success at 92% means ~8% of grasps result in physics explosions regardless
of arm precision. This is a fundamental property of the `mass=0.05, Kp=150000` combination.
These 8% failures will persist in `eval_grasp.py` even after all code fixes.

**To eliminate:** Either increase the cube mass slightly (0.1kg → less Kp overpowering
contact) or reduce Kp to 75,000 (sacrificing some hold force). These require revalidating
the full physics suite.

### 4.2 Home Transit Distribution Shift
Once `force_start_pos` is correctly implemented, the arm starts at the Home Position
`[0.680, 0.264, 0.550]` — which is within but at the **near edge** of the training range
(`low_x = 0.640`). The model has seen this position during training, but rarely compared
to central board positions. Transit success rate from home may be 5-10% lower than random
board position.

**Estimated eval_grasp.py impact:** Transit success drops from "~100% (misleading)" to
perhaps 90-95% (accurate measurement). This will reduce the overall pipeline success rate
but reflects actual performance.

### 4.3 The `TUBE_BREACH` Crash in eval_grasp.py
From the previous eval_grasp.py run (100 episodes):
```
[EPISODE 100 END] Outcome=CRASH (TUBE_BREACH (center=0.0105 > limit=0.0100))
```
This suggests the RL model is occasionally using too-aggressive drift compensation during
the descend phase. This is not fixed by any of the current changes.

**Root cause hypothesis:** The descend model was trained with `drift_limit_end=0.010`
but the evaluation uses `drift_limit=0.010`. These match, so the TUBE_BREACH is a genuine
policy failure — the model occasionally exceeds 10mm drift. With 91% descend success,
this happens ~9% of the time.

**Fix:** Targeted retraining of the descend model with stricter drift limits, OR increase
the eval drift limit tolerance slightly (0.012) to accept near-miss approaches.

---

## Section 5: Expected Results After Full Implementation

| Metric | Current (broken impl) | After All Fixes |
|:-------|:----------------------|:----------------|
| Gate 0 (verify_physics) | ❌ FAIL (kp assertion wrong) | ✓ PASS |
| Static grasp success | 92% (test suite) | ~92% (unchanged) |
| Lift success | 88% | ~90% (marginal improvement) |
| Transit held success | 84% | ~88% (improved by slower step) |
| Transit to src (eval_grasp) | 100% (MISLEADING) | ~93% (from real Home Position) |
| Descend success | 91% | ~91% (unchanged) |
| Grasp success | 78% | **~88-91%** (contact approach implemented) |
| Ascend success | 78% | **~88-91%** (follow-on improvement) |
| Transit to home | 78% | **~88-91%** (follow-on improvement) |

The critical improvement is the contact approach phase, which should recover ~10 percentage
points by ensuring proper cube overlap at grasping time.

