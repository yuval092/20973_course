# RoboChess — Hardening & Polish Plan

After reading every source file, test file, and the existing plan history, here is a comprehensive implementation plan.

---

## Phase 1: Bug Fixes & Edge Cases

These are concrete bugs or crash-paths I found during the code review.

---

### Bug 1: `board_panel.py` crashes on deselect when piece is `None`

In `board_panel.py`, the second click path calls `state.board.piece_at(self.selected_square).piece_type` without checking if the piece is still there. If between clicks the board state changes, `piece_at()` returns `None` and `.piece_type` throws `AttributeError`.

> [!WARNING]
> **Fix**: Guard with `piece = state.board.piece_at(self.selected_square); if piece is None: self.selected_square = None; self.refresh(); return`.

### Bug 2: `board_panel.py` uses `parse_san()` instead of `parse_uci()`

In `board_panel.py`, `state.board.parse_san(move_uci)` is called on a UCI string (like `"e2e4"`). `parse_san()` expects SAN notation (like `"e4"`). This will throw `ValueError` for most inputs, making the click-to-move completely non-functional.

> [!CAUTION]
> **Fix**: Replace `state.board.parse_san(move_uci) in state.board.legal_moves` with `chess.Move.from_uci(move_uci) in state.board.legal_moves`.

### Bug 3: `game_runtime.py` main() has duplicate `import time`

In `game_runtime.py`, `import time` appears at module body scope again after already being imported at the top.

> [!NOTE]
> **Fix**: Remove the duplicate `import time`.

### Bug 4: `game_runtime.py` main() has inline imports at module scope

In `game_runtime.py`, several imports (`tkinter`, `threading`, `GameLoop`, etc.) are placed at module body scope *after* function definitions. This causes them to execute on import of `game_runtime`, triggering a circular import with `game_loop.py`.

> [!IMPORTANT]
> **Fix**: Move all these imports strictly inside `main()`.

### Bug 5: `ChessGameManager` never closes the Stockfish engine

The Stockfish engine is opened in `ChessGameManager.__init__`, but **nothing in the codebase ever calls `.close()`**. The Stockfish subprocess leaks on every run.

> [!WARNING]
> **Fix**: Make `ChessGameManager` a context manager (`__enter__`/`__exit__`) and ensure it's closed during shutdown.

### Bug 6: `game_loop.py` `submit_move()` doesn't run health checks

Currently, `submit_move()` does not invoke `check_registry.run(CheckHook.TURN_END, ...)`. Turn-end validity checks (mapping integrity, piece stability) are silently skipped for human moves.

> [!WARNING]
> **Fix**: Add `self.systems.check_registry.run(CheckHook.TURN_END, ...)` after `push_move` in both `submit_move()` and `execute_ai_turn()`.

### Bug 7: `game_runtime.py` main() doesn't run `TURN_START` / `TURN_END` checks

The new `main()` worker loop bypasses `check_registry.run(CheckHook.TURN_START, ...)` completely. The entire health-check framework (mapping integrity, board agreement, arm home pose, piece observer, workspace check) is effectively dead at the turn boundary.

> [!CAUTION]
> **Fix**: Integrate turn-start checks into the `GameLoop`'s worker logic.

### Bug 8: Piece Damping and Clipping Bugs
The environment wrapper defines `IDLE_PIECE_FREEJOINT_DAMPING = [200.0, ...]` to prevent pieces from moving when they shouldn't. However, the collision geometry is explicitly stripped (`contype=0`) when a piece is not the active target. Furthermore, pieces could drop slightly inside the board or clip into other pieces if collision shapes are toggled incorrectly or if the damping values fail to overcome gravitational forces during teleport steps.

> [!CAUTION]
> **Fix**: We must ensure pieces are fully supported by collision geometry at all times to prevent them from dropping inside the board. The physics overrides must be hardened.

---

## Phase 2: Strict Physics Adherence & Stability (Addressing Your Tasks 1 & 2)

Per your specific request, we must guarantee that **we never override physics to 'cheat'** and that pieces never **move uncontrollably or drop into the board**.

### 2.1 Enforce Consistent Collision Geometry
Currently, in `ChessPickPlaceEnv._resolve_piece_mesh_geom_ids` and `set_piece_mesh_collision_enabled`, we manually toggle `contype` and `conaffinity` array values for the piece meshes. If a piece is "idle", its main mesh collision might be turned off, leaving only a small invisible collision cylinder (or nothing!), which allows pieces to clip through each other or sink into the board when teleported or dropped slightly off-center.

> [!CAUTION]
> **Plan:**
> - Remove dynamic toggling of `geom_contype` and `geom_conaffinity` for meshes entirely.
> - Pieces **must** maintain their standard mesh collision shapes throughout the entire simulation exactly as defined in the XML.
> - If the AI struggles to pick pieces due to collision volumes, we must adjust the `FETCH_POLICY_HORIZON` or `GRIP_CONTACT_TOLERANCE`, *not* disable collisions.

### 2.2 Fix Teleport Z-Drop (Sinking)
In `game_runtime.py::teleport_piece` and `handle_promotion`, pieces are placed precisely at the `Z_GRASP` or `TABLE_HEIGHT` thresholds. If teleported a millimeter too low, the physics engine resolves the penetration by springing the piece violently, or if it lacks collision, it falls through the board.

> [!IMPORTANT]
> **Plan:**
> - When a piece is teleported (e.g. at start or for promotion replacement), explicitly place its Z-coordinate slightly *above* the table level (e.g., `TABLE_HEIGHT + 0.05`), ensuring its Z-velocity is 0.
> - Let gravity naturally pull the piece down onto the board's collision shape. The `IDLE_PIECE_FREEJOINT_DAMPING` will settle it gently. No pieces will ever be placed clipped into the board.

### 2.3 Prevent Uncontrollable Movement
To prevent pieces from tipping or skidding uncontrollably during AI actions or gravity settling:

> [!TIP]
> **Plan:**
> - We will maintain the `ACTIVE_PIECE_FREEJOINT_DAMPING` and `IDLE_PIECE_FREEJOINT_DAMPING` mechanism for the freejoints, but ensure it is strictly applied before any step.
> - Idle damping stops gravity/small bumps from compounding into a fall.
> - The `PieceObserverCheck` runs on `TURN_END` to guarantee no piece has tipped or fallen before the game continues. (This was broken by Bug 7; fixing Bug 7 enables this guard).

---

## Phase 3: Tkinter GUI Polish & Game State

The current `board_panel.py` is a skeleton. This phase makes it fully functional and polished.

---

### 3.1 Fix Core GUI Functionality

#### [MODIFY] [board_panel.py](file:///home/user/projects/robo_chess_2/src/gui/board_panel.py)

- **Fix `parse_san` → `Move.from_uci`** (Bug 2)
- **Add null-safety for deselection** (Bug 1)
- **Add a UCI text input field** as fallback for click-to-move
- **Add promotion picker dialog** — when a pawn reaches the last rank, show a choice popup (Queen/Rook/Bishop/Knight) instead of silently choosing Queen
- **Add a "New Game" button** to reset without restarting the process
- **Show game-over reason** in a dialog (Checkmate, Stalemate, Draw by X)
- **Disable board interaction** when it's not the human's turn

### 3.2 Improve GUI–Worker Thread Communication

#### [MODIFY] [game_runtime.py](file:///home/user/projects/robo_chess_2/src/game_runtime.py) — `main()`

- Add proper Tkinter window title, minimum size, and close handling (`WM_DELETE_WINDOW`)
- When the viewer closes or the game ends, properly shut down the Tk mainloop
- Call `manager.close()` at shutdown to reclaim the Stockfish process

### 3.3 Game State Handling Verification

Verify all terminal conditions (Checkmate, Stalemate, 50-move Rule, Threefold Repetition, Insufficient Material) propagate correctly through the GUI's game-over display logic.

---

## Phase 4: Test Suite, Robustness & Production Hardening

---

### 4.1 Test Suite Optimization

The tests currently hang because `Config.py` and `_build_local_systems()` initialize a real Stockfish engine during pytest collection.

> [!TIP]
> **Plan**:
> - Fix test timeouts by mocking Stockfish during fast tests.
> - Add `test_game_loop.py` tests: `execute_ai_turn`, `submit_move_valid`, `request_hint`.
> - Add `test_board_panel.py` tests: UI interaction events.

### 4.2 Structural Robustness

#### A. Keep CLI UI
Per your request, `cli_ui.py` will be kept as a headless/SSH fallback.

#### B. Module Split
Per your request, `game_runtime.py` will be refactored into:
- `game_runtime.py`: Only `bootstrap_game_systems()` and `main()`
- `physics_ops.py`: `teleport_piece`, `execute_human_ops`, `execute_ai_ops`, `handle_promotion`
- `display.py`: Terminal display functions.

---

### 5 Extra

1. Physics Overrides: Phase 2 of the plan explicitly outlaws turning off geom_contype and conaffinity. The env.set_piece_mesh_collision_enabled(..., False) logic will be removed entirely. Pieces will maintain solid collision at all times.
2. Piece Stability (Z-Drop): Phase 2 details the fix for dropping pieces into the board or clipping. When pieces are spawned or promoted, we will teleport them slightly above the table (TABLE_HEIGHT + 0.05) with 0 velocity, allowing gravity to gently settle them. Furthermore, we will run the PieceObserverCheck at the end of each turn (which was currently broken due to a loop bug) to guarantee nothing is tipped or sunk before proceeding.
3. Module Split and CLI fallback: Your answers to the open questions are incorporated. cli_ui.py is kept for testing, and game_runtime.py is split.


## Summary

| Phase | Scope             | Key Deliverables                                                          |
| ----- | ----------------- | ------------------------------------------------------------------------- |
| **1** | Bug fixes         | 8 bugs fixed (GUI crash, missing checks, leaked engine, circular imports) |
| **2** | Physics Adherence | Remove collision mutators, fix Z-drop, verify damping stability           |
| **3** | GUI + Game state  | Polished Tkinter panel, promotion dialog, game-over UX                    |
| **4** | Tests + Refactor  | Optimize suite speed, split `game_runtime.py`, context manager            |

## Review & Next Steps
If this plan looks acceptable, I will begin implementing Phase 1 and Phase 2 immediately, ensuring physics integrity is ironclad before styling the UI.
