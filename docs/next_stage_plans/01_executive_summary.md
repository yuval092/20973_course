# Next Stage Plan: Scenario Chaining — Executive Summary

## Objective

Validate and evaluate the robotic arm + RL model when multiple scenarios are executed
**back-to-back**, without a full simulation reset between them. This is the foundational
step toward the full game controller that orchestrates a complete chess move.

## Why This Is Non-Trivial

The current setup runs each scenario in isolation: `env.reset()` teleports the arm to
the canonical start of each episode. In a real chess move, there is no teleport — the
arm ends where it ends, and the next scenario must begin from that exact, correct position.

This creates two critical problems that must be solved:

### Problem 1: The Arm Is Still Moving After Success

The RL success condition requires `speed < 50mm/s`. After a "success" step, the arm
can still be coasting at 1–3mm/s. Before any transition can happen, the arm must reach
**complete halt**. The braking reward trained into the model helps — the arm is
already decelerating toward the goal — but a guaranteed halt requires active enforcement:

```
After success: issue scripted hold steps → actively zero arm velocities → confirm halt
```

The arm must be completely stationary before the next scenario begins. A coasting arm
entering a 5mm tube is a guaranteed TUBE_BREACH.

### Problem 2: The Arm Must Arrive at the Exact Waypoint

When a scenario "succeeds," the arm is within **10mm** of the nominal waypoint
(success_threshold = 10mm). The descend/ascend scenarios enforce a **5mm drift limit**.
The destination cell on the board does not change — descend must happen over the correct
cell, not "close enough to where transit happened to end."

The solution is **waypoint alignment**: after the RL episode succeeds, a short scripted
settle phase physically moves the arm from its ±10mm landing position to exactly the
nominal waypoint position. This is NOT a teleport — it is a physics-simulated smooth
movement using the same settle loop already proven in `_reset_sim`.

```
After success + halt:
  scripted settle loop → arm moves from ±10mm → within 3mm of exact waypoint
  → tube_center = nominal_xy (the arm is now close enough for a safe 5mm tube start)
```

This guarantees that descend/ascend always start from the correct cell position.

## The Full Transition Protocol

Every scenario-to-scenario boundary executes this strict sequence:

```
[1] RL SUCCESS (arm within 10mm, speed < 50mm/s)
        ↓
[2] COMPLETE HALT PHASE
    - Issue hold actions (zero movement) for up to 100 steps
    - Monitor velocity until speed < 0.5mm/s (near-zero, not just "slow")
    - If not settled: abort chain (log HALT_TIMEOUT)
        ↓
[3] WAYPOINT ALIGNMENT PHASE
    - Run scripted settle loop (same as _settle_arm_to_start)
    - Target = nominal waypoint for current scenario's exit
    - Arm physically moves from ±10mm → within 3mm of exact waypoint
    - Zeros arm velocities (not object joint)
        ↓
[4] GRIPPER TRANSITION PHASE
    - If next scenario requires a different finger state:
      execute scripted 50-step open/close loop while arm is held still
    - Validate finger state (same as _reset_sim Phase 3)
        ↓
[5] SOFT RESET
    - Update: scenario, goal_pos, tube_center = nominal_xy
    - Reset episode counter
    - Return new observation
        ↓
[6] NEXT SCENARIO BEGINS
```

## Scope

**Evaluation only.** No changes to training code, reward function, or model architecture.
A new script `scripts/eval_sequence.py` is created. The existing `task.py` gains one new
public method (`soft_reset`) and one helper (`transition_validate`).

## Success Criteria

These targets assume a fully trained V6 model (≥1M steps):

| Chain Length | Target Chain Success Rate |
|:---|:---|
| 2 scenarios | ≥ 90% |
| 3 scenarios | ≥ 85% |
| 4 scenarios | ≥ 80% |
| 6 scenarios (full move) | ≥ 70% |

---
*Next: [02 — Transition Architecture](./02_transition_architecture.md)*
