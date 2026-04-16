# Fix Plan: Piece Drift & Board Validation Failures

## Problem Analysis

I analyzed both log files and ran an idle physics drift test. There are **two distinct problems**:

### Problem 1: Pieces drift (twitch) on the board — even with no arm movement

My drift test (`scripts/drift_test.py`) shows that **21 out of 32 pieces drift >5mm** after just 500 idle physics steps, with the worst drifting **30mm**. This drift happens with zero arm interaction.

**Root cause**: The XML free-joint damping is `0.05`, and the runtime code sets idle-piece damping to `20.0` — but only **after** `ChessPickPlaceEnv.__init__()` runs. The `generate_xml.py` spawns pieces at `Z_GRASP + 0.0006 = 0.4136`, but with the recent `board_collision_z` fix, the board collision top surface is at:

```
TABLE_HEIGHT + 0.001 - board_collision_half_thickness + board_collision_half_thickness
= TABLE_HEIGHT + 0.001 = 0.401
```

The pieces spawn at 0.4136 — that's **12.6mm above** the collision surface. During the initial settle, they drop onto the board with velocity, and even with `damping=20.0`, they bounce/slide laterally. The heavier pieces (king, queen) with taller meshes drift the most.

Additionally, **some pieces (rooks at corners, a couple of pawns) drift ~0mm** — these are the ones locked into tight positions by neighbours or walls. The asymmetric back-rank pieces (king, queen, knights) drift the most because their meshes have asymmetric mass distribution despite the cylinder hitbox.

### Problem 2: `BoardStateError` on `b_pawn_7` on square `g7`

Both logs crash with the same error:
```
Piece coordinate mismatch | square=g7 | piece=b_pawn_7
expected=[1.005 0.575 0.413] | actual=[0.9856 0.5556 0.4124]
xy_error=0.0275 (log9) / 0.0238 (log10)
```

> [!CAUTION]
> This is **not** a problem with the moved piece! The moved pieces (`b_pawn_5` in log9 at `e5`, `b_pawn_4` in log10 at `d6`) are placed correctly with <2mm error. The crash is on `b_pawn_7` at `g7` — **a piece that was never touched by the arm**. It drifted 24-28mm during the simulation from physics interaction/vibration alone.

The `placement_tolerance` is set to `0.02` (20mm). `b_pawn_7` at `g7` drifts 24-28mm, exceeding this threshold. The tolerance is checked in `validate_board_state()` against **all 32 pieces**, not just the moved one.

---

## Proposed Changes

### Phase 1: Fix idle piece drift (ROOT CAUSE of both problems)

#### [MODIFY] [generate_xml.py](file:///home/user/projects/robo_chess_2/scripts/generate_xml.py)

1. **Lower piece spawn height** — spawn at `Z_GRASP` (0.413) exactly instead of `Z_GRASP + 0.0006`. With the thickened board collision, pieces rest stably at z≈0.4122. Spawning them closer eliminates the initial drop energy that causes lateral drift.

2. **Increase free-joint damping in XML** from `0.05` to `5.0`. The runtime code already overrides to `20.0` for idle pieces, but this ensures that even before the environment initializes, pieces settle faster. The `5.0` value is chosen to be high enough to damp initial bounce but low enough that the active piece can still be manipulated by the RL policy.

3. **Regenerate** `chess_world.xml`.

#### [MODIFY] [chess_pick_place_env.py](file:///home/user/projects/robo_chess_2/src/env/chess_pick_place_env.py)

4. **Increase idle damping** from `20.0` to `200.0`. The current `20.0` was the "400×" multiplier over `0.05`, but it's clearly not enough — pieces still drift 30mm. With 200.0, pieces should be effectively frozen without any realistic motion concern (they're idle — they shouldn't move at all).

5. **Add an init-time settle loop**: After setting all pieces to idle damping, run a short settle (e.g., 200 `mj_step` calls) and then **snap each piece back to its initial position**. This absorbs the spawn-height settling energy and ensures pieces start at their exact expected coordinates. After snapping, do one `mj_forward()` to update all positions.

### Phase 2: Raise the board validation tolerance

#### [MODIFY] [runtime.yaml](file:///home/user/projects/robo_chess_2/src/settings/runtime.yaml)

6. **Increase `placement_tolerance`** from `0.02` (20mm) to `0.025` (25mm). This provides a more realistic margin for idle pieces, which will always have some sub-millimeter residual contact drift even with high damping. However, with the Phase 1 fixes, drift should be well under 5mm, making this a safety margin rather than a band-aid.

> [!WARNING]
> After Phase 1 fixes, this tolerance change may not be strictly necessary. But it's cheap insurance against edge cases. **If you'd prefer to keep it tight at 20mm**, let me know — the Phase 1 fixes alone may be sufficient.

### Phase 3: Verify

7. **Re-run the drift test** (`scripts/drift_test.py`) to confirm max drift < 5mm.
8. **Re-run a headless game test** to confirm no `BoardStateError`.
9. **Confirm arm manipulation still works** — the active piece damping (0.05) shouldn't change, so pick-and-place should work identically.

---

## Summary of Changes

| File                      | Change                                 | Why                                           |
| ------------------------- | -------------------------------------- | --------------------------------------------- |
| `generate_xml.py`         | Spawn at `Z_GRASP` exactly             | Eliminate drop energy → less lateral bounce   |
| `generate_xml.py`         | XML damping `0.05` → `5.0`             | Faster initial settling before env takes over |
| `chess_pick_place_env.py` | Idle damping `20.0` → `200.0`          | Actually freeze idle pieces                   |
| `chess_pick_place_env.py` | Init-time settle + snap                | Start pieces at exact rest positions          |
| `runtime.yaml`            | `placement_tolerance` `0.02` → `0.025` | Safety margin                                 |

## Open Questions

> [!IMPORTANT]
> **Q1**: The init-time settle + snap approach resets pieces to their spawn positions after physics settling. An alternative is to **not snap** and instead just accept wherever they settle physics-wise, then update `expected_pos` in the board controller to match. Which approach do you prefer? (I recommend snap — it's simpler and keeps positions clean.)

> [!IMPORTANT]
> **Q2**: Do you want me to increase the tolerance to 25mm, or would you prefer to keep it at 20mm and rely solely on the physics fixes?
