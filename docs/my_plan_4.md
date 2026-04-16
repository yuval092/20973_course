# Game Plan: Test Suite Hardening, Spare Pieces Cleanup & Critical Bug Fixes

## Executive Summary

After analyzing the full test suite (12 files), the spare-piece generation code, the source code, and both log files, I identified three workstreams with concrete recommendations for each.

---

## 1. Test Suite Audit & Improvement

### Current State

The test suite has **12 files** with reasonable coverage for the "happy path", but has major blindspots that let the bugs in `logs9`–`logs12` slip through undetected.

| Test File                    | What it Tests                                                    | Quality                                       |
| ---------------------------- | ---------------------------------------------------------------- | --------------------------------------------- |
| `test_chess_manager.py`      | Board logic (validate, push)                                     | ✅ Adequate, pure-logic                        |
| `test_operation_planner.py`  | Op generation (captures, castling, en passant, promotion)        | ✅ Good, but uses dummy mapping                |
| `test_runtime_guard.py`      | Board validation (coordinate, tilt, mapping mismatch)            | ⚠️ Fully mocked — never exercises real physics |
| `test_board_observer.py`     | Stability (tilt, displacement, below-table) + real physics drift | ✅ Good, has real-scene tests                  |
| `test_obs_wrapper.py`        | Observation vector reconstruction                                | ⚠️ Fully mocked, fragile                       |
| `test_health_checks.py`      | Check registry, arm home, workspace, NaN, etc.                   | ✅ Real scene + unit tests                     |
| `test_headless_execution.py` | Stage-level arm control (close, lift, place)                     | ✅ Good                                        |
| `test_headless_gameplay.py`  | Multi-turn sequences, captures, castling, validation errors      | ⚠️ **Missing critical coverage**               |
| `test_startup_runtime.py`    | Bootstrap, turn processing, viewer, main flow                    | ✅ Thorough                                    |
| `test_training_prep.py`      | Training env, task sampling, curriculum                          | ✅ Adequate                                    |

### Critical Gaps Identified

> [!CAUTION]
> The test suite currently has **zero tests** for graveyard capture operations — the exact scenario that crashed both `logs11.txt` and `logs12.txt`.

#### Gap 1: No Graveyard (Capture) Physical Tests
The `test_headless_gameplay.py` file has a capture test (`test_headless_human_capture_sends_captured_piece_to_black_graveyard`) but it only calls `execute_human_ops` which **teleports** pieces. There is no test that runs the RL-driven `execute_ai_ops` with a capture→graveyard operation.

**Proposed fix:** Add `test_headless_ai_capture_to_graveyard_succeeds` that:
1. Plays `e2e4, d7d5, e4d5` with the AI physically executing the capture
2. Asserts the captured piece lands near the graveyard origin
3. Asserts the capturing piece lands on `d5`

#### Gap 2: No Test for OUT_OF_POLICY_WORKSPACE Behavior
Every single log file shows `OUT_OF_POLICY_WORKSPACE` warnings. Most board squares exceed the policy's 0.15m trained radius, meaning the arm is *always* running outside its training distribution. Yet no test validates that the arm still succeeds under this condition, or documents which patterns fail.

**Proposed fix:** Add `test_out_of_policy_workspace_logs_warning_but_succeeds` and `test_extreme_distance_operation_fails_gracefully` that directly exercise moves at known distances (near center vs. corner-to-graveyard).

#### Gap 3: No Multi-Capture Sequence Test
No test exercises the graveyard counter incrementing properly across multiple captures.

**Proposed fix:** Add a test that performs 3+ captures in sequence and verifies graveyard grid positions don't overlap.

#### Gap 4: No Graveyard Reachability Integration Test
The graveyard origin `[0.93, 0.30, 0.425]` is 0.45m from the arm home — 3× the policy radius. `test_headless_graveyard_targets_are_reachable_and_raised` calls `_is_reachable` (which checks bounds) but never physically moves the arm there.

**Proposed fix:** Add a physical test that actually moves the arm to the graveyard position and back.

#### Gap 5: No Piece-Drop / Fumble Detection Test  
No test verifies that the system correctly detects and reports when the arm drops a piece mid-transit (which is the failure mode in both logs).

**Proposed fix:** Add a test with a monkeypatched `LIFT_VERIFY` that artificially drops the piece, confirming the stage returns `success=False`.

#### Gap 6: No Drift Test Under Physics Stepping *with* Arm Motion
`test_arm_motion_does_not_destabilize_board` sweeps the arm around, but only checks idle pieces. It doesn't check what happens when the arm collides with or passes near pieces.

**Proposed fix:** Add `test_arm_transit_near_piece_does_not_knock_it` that sweeps through critical paths.

#### Gap 7: Insufficient `test_chess_manager.py`
Only 4 tests for the chess logic. Missing: AI move integration, game-over detection, move history tracking.

**Proposed fix:** Add tests for `get_ai_move`, `is_game_over`, FEN consistency.

#### Gap 8: `test_runtime_guard.py` Uses Only Mocks
All 4 `validate_board_state` tests use `MagicMock`. The real-scene `validate_board_state` call in `test_headless_gameplay.py` compensates somewhat, but there's no isolated test of `validate_board_state` against the real MuJoCo scene.

**Proposed fix:** Add `test_validate_board_state_real_scene_initial_board_passes`.

### Proposed New Test Structure

```
tests/
├── test_board_observer.py          # Keep, good
├── test_chess_manager.py           # Extend with AI/game-over tests
├── test_execution_controller.py    # Keep
├── test_headless_execution.py      # Keep
├── test_headless_gameplay.py       # Extend with graveyard + capture tests  
├── test_health_checks.py           # Keep
├── test_obs_wrapper.py             # Keep
├── test_operation_planner.py       # Keep
├── test_runtime_guard.py           # Add real-scene test
├── test_startup_runtime.py         # Keep
├── test_training_prep.py           # Keep
├── test_graveyard_operations.py    # [NEW] Dedicated graveyard/capture tests
└── test_drift_and_stability.py     # [NEW] Extended drift + collision tests
```

---

## 2. Spare Pieces (Promotion Pieces)

### Current State

The spare pieces (`w_spare_queen`, `w_spare_rook`, etc.) are spawned at:
```python
x = BOARD_CENTER[0] + idx * 0.1 - 0.15   # ~1.03 to 1.33
y = BOARD_CENTER[1] ± 0.3                  # ~0.45 (black) / ~1.05 (white)
z = -0.0500                                # Below table (hidden)
```

The Z coordinate of `-0.05` means they start below the table. However, **they have free joints**, so after physics settling they may drift to visible locations on or under the table surface.

### Recommendation: Keep Spawning Below Table, But Freeze Them

> [!IMPORTANT]
> The comment says "remain hidden below the table until a pawn is swapped" — this is the correct design. The issue is that free-joint physics may cause them to fall or float.

**Proposed fix:**
1. **Remove free joints from spare pieces.** Since they're "storage" bodies that will be teleported into play during promotion, they don't need physics until activated. Making them static (no joint) keeps them completely frozen below the table.
2. **When a promotion occurs at runtime**, we teleport the spare piece to the board square and add the free joint programmatically (or, since MuJoCo doesn't support dynamic joint addition, we keep the joint but set the qpos directly via `teleport_piece`).

**Alternative (simpler):** Keep the free joints but add `<geom contype="0" conaffinity="0"/>` on the spare pieces' collision cylinders so they don't interact with anything. They'll sit frozen at Z=-0.05 since nothing pushes them. This is the simplest fix.

**My recommendation:** Go with the simpler alternative — just disable collision on spare pieces. They're already at Z=-0.05 and won't interact with anything. When promotion occurs, re-enable collision and teleport.

---

## 3. Critical Bugs from Log Analysis

### Bug #1: Graveyard Capture Operations Always Fail (CRITICAL)

> [!CAUTION]
> Both `logs11.txt` and `logs12.txt` crash on the **same operation type**: moving a captured piece to the graveyard.

**logs11.txt:** `w_pawn_5 e3→white_graveyard` → FAILED (xy_error=37.3mm, stages=6/13)
**logs12.txt:** `w_pawn_2 b4→white_graveyard` → FAILED (xy_error=54.7mm, stages=6/13)

#### Root Cause Analysis

The failure happens at `Stage PREHOVER_DEST` — the arm tries to carry the piece from its board square (~1.15, 0.85) to the graveyard (~0.93, 0.30).

The critical numbers:
- **Distance from arm home to graveyard:** `dest_xy_dist = 0.646m` (logs11)
- **RL policy trained radius:** `0.150m`
- The arm reaches the graveyard zone (grip reaches ~0.98, 0.36) but **can't converge** on the exact target (0.93, 0.30). The RL model outputs action `[-0.28, -0.10, ...]` for hundreds of steps without making progress — the remaining distance stays frozen at `0.015m`. The arm is **stuck in an RL output dead zone** far outside its training distribution.

After 150/250 max steps, the stage times out. **On 6 out of 13 stages completed** — meaning the piece was picked up successfully, but the arm can't deliver it.

#### Why This Passed All Tests
The test suite only tests graveyard captures via `execute_human_ops` (which teleports the piece) — never via the RL arm.

#### Proposed Fix

> [!IMPORTANT]
> Graveyard destinations are so far from the RL's training zone that the arm will likely **never** reliably deliver pieces there. The graveyard is ~0.65m from the arm home; the model was trained within a 0.15m radius.

**Option A (Recommended): Teleport captured pieces to the graveyard**
- In `execute_ai_ops`, when an op has `dest_square=*_graveyard`, skip RL execution and use `teleport_piece()` directly.
- This is logically correct: capturing means removing a piece from play. Whether the arm physically moves it is irrelevant to the game logic.
- This matches the behavior of `execute_human_ops` which already teleports.

**Option B: Move the graveyards closer to the board**
- Move them to ~0.1m from the board edge, within reach of the arm. 
- Risk: clutters the scene, graveyards collide with piece traffic.

**My recommendation:** Option A. Teleport captures to the graveyard. This is what human-move execution already does, it's the simplest and most robust solution, and it aligns with the game semantics (a captured piece is simply removed from the board).

### Bug #2: Stage `PREHOVER_DEST` Timeout Wastes 250 Steps Before Aborting

When the arm can't reach the target, it burns through all 250 steps outputting nearly identical actions. There's no early-termination for "not making progress."

**Proposed fix:** Add a *stall detector* to the waypoint loop. If `remaining` distance doesn't decrease by more than 1mm over 30 consecutive steps, abort the stage early as unreachable. This provides a faster fail and cleaner error message.

### Bug #3: Graveyard Counter Is a Class Variable (Minor, But Fragile)

```python
_white_graveyard_count = 0
_black_graveyard_count = 0
```

These are **class** variables on `ExecutionController`. The test setup manually resets them (`ExecutionController._white_graveyard_count = 0`). If two tests run in the same process without reset, the graveyard positions silently shift.

**Proposed fix:** Move to instance variables or add a proper `reset()` method called by `__init__`.

---

## Summary of Prioritized Actions

| Priority | Action                                               | Effort | Impact                     |
| -------- | ---------------------------------------------------- | ------ | -------------------------- |
| 🔴 P0     | Fix graveyard capture bug (teleport captured pieces) | Small  | Unblocks all captures      |
| 🟠 P1     | Add graveyard + capture physical tests               | Medium | Prevents regression        |
| 🟠 P1     | Add stall detector for waypoint convergence          | Small  | Better failure diagnostics |
| 🟡 P2     | Disable collision on spare pieces                    | Small  | Cleaner visual scene       |
| 🟡 P2     | Add OUT_OF_POLICY + drift integration tests          | Medium | Improves confidence        |
| 🟢 P3     | Extend chess_manager + runtime_guard tests           | Small  | General coverage           |
| 🟢 P3     | Fix graveyard counter to instance variable           | Tiny   | Code hygiene               |

## Open Questions

1. **For Bug #1:** Do you want me to implement Option A (teleport captures) or Option B (move graveyards closer)? I strongly recommend Option A.
2. **For spare pieces:** Should I simply disable their collision geometry so they stay hidden, or do you want a more elaborate approach?
3. **For the stall detector:** Do you want an early-abort at 30 steps of no progress, or a different threshold?
