# RoboChess — Two-Phase Improvement Plan

After a thorough read of the entire codebase (all source modules, tests, configs, scripts, and runtime YAML), I've compiled my findings into two actionable implementation plans.

---

## Phase 1: Bug Fixes, Game State Handling, and Test Coverage

This phase focuses on correctness, edge-case hardening, and filling test gaps. No new features — only making what exists bulletproof.

---

### 1.1 Bugs and Edge Cases Found

#### Bug 1: `handle_promotion` sets wrong RGBA for non-mesh geoms

In [game_runtime.py:278](file:///home/user/projects/robo_chess_2/src/game_runtime.py#L274-L282), when re-enabling a spare piece during promotion, the cylinder (collision proxy) geom gets `rgba=[1.0, 0.0, 0.0, 0.0]` — the alpha is `0.0`, making the collision cylinder invisible. While this is intentional for collision-only geoms, the **mesh geom** line sets `rgba=[1.0, 1.0, 1.0, 1.0]` regardless of piece color. A promoted **black** spare piece would appear white.

> [!WARNING]
> **Fix**: Determine rgba from color prefix. If `color_prefix == "b"`, use the black piece material rgba, not pure white.

#### Bug 2: `process_human_turn` doesn't re-prompt after invalid input

In [game_runtime.py:376-408](file:///home/user/projects/robo_chess_2/src/game_runtime.py#L376-L408), when the user enters an illegal move or a hint command, the function returns `True` (continue), but the *caller* (`run_turn`) doesn't loop — it proceeds to `complete_turn`. This means:
- A `hint` command triggers `complete_turn` validation, which runs `BoardAgreementCheck`, `MappingIntegrityCheck`, etc. — all unnecessary since no move was made.
- An illegal move prints a message but the user doesn't get to re-enter. The turn just ends with no move played, and next turn is AI's move.

> [!IMPORTANT]
> **Fix**: `process_human_turn` should loop internally until a valid move is played or quit is requested. Only return `True` when a move was actually pushed onto the board.

#### Bug 3: Graveyard counters are class-level defaults leaked across instances

In [execution_controller.py:103-104](file:///home/user/projects/robo_chess_2/src/control/execution_controller.py#L103-L104), `_white_graveyard_count` and `_black_graveyard_count` are class variables. They are shadowed by instance attributes in `__init__` (lines 118-119), but `_build_game()` in [test_headless_gameplay.py:19-20](file:///home/user/projects/robo_chess_2/tests/test_headless_gameplay.py#L19-L20) resets them via the class directly (`ExecutionController._white_graveyard_count = 0`), which is fragile. If any test forgets to do this, stale counters leak.

> [!NOTE]
> **Fix**: Remove the class-level defaults. Only use instance variables set in `__init__`. Update any test that touches class variables to use the instance instead.

#### Bug 4: `_is_piece_body` has a broad false-positive pattern

In [game_runtime.py:92-93](file:///home/user/projects/robo_chess_2/src/game_runtime.py#L92-L93), the condition `any(... prefix in body_name ...)` uses `in` substring matching, so a body named `"w_squareboard"` would be excluded even though it starts with `"w_"`, because `"square"` is in the skip list. This is fragile — it works now only because the XML doesn't have such names.

> [!NOTE]
> **Fix**: Use `body_name.startswith(prefix)` for skip prefixes instead of `prefix in body_name`.

#### Bug 5: `BoardObserver.verify_stability` checks captured/graveyard pieces

In [board_observer.py:33-65](file:///home/user/projects/robo_chess_2/src/observation/board_observer.py#L33-L65), `verify_stability` iterates **all** bodies. Captured pieces teleported to the graveyard can have orientation drift or Z slightly below the displacement threshold, causing false `StabilityError` exceptions. The filter on line 31 skips `"graveyard"` bodies but not pieces whose names start with `w_` or `b_` that have been moved to the graveyard zone.

> [!WARNING]
> **Fix**: Accept a set of "active" piece names (from `square_to_piece.values()`) and skip pieces not in that set. Or add a positional guard: if the piece is outside the board bounding box, skip it.

#### Bug 6: En passant promotion with capture is not tested

The operation planner handles en passant and promotion capture separately, but there's no test for the edge case where en passant and promotion might coincide (impossible in chess, so this is fine). However, there is **no test for under-promotion** (e.g., promoting to a knight with `a7a8n`). The runtime `handle_promotion` supports it via the `promo_map`, but it's never validated.

---

### 1.2 Game State Handling Gaps

#### Current State
- **Game over** (`is_game_over(claim_draw=True)`) is checked in the main loop — ✅
- **Checkmate** — winner printed in `print_game_over` — ✅
- **Stalemate** — printed as draw — ✅
- **Check** — printed after each turn — ✅ (added in last session)
- **Insufficient Material / 50-move / Threefold** — handled by `claim_draw=True` in the main loop — ✅ (auto-claims draw)

> [!NOTE]
> The `claim_draw=True` argument to `is_game_over()` automatically claims draws for threefold repetition and the 50-move rule. This is correct behavior for a human-vs-AI game. No further changes needed.

#### Missing: Announce *reason* for draw

`print_game_over` currently prints `"Result: {board.result()}"` for non-checkmate, non-stalemate endings. This covers threefold/50-move, but the message is cryptic (e.g., `1/2-1/2`). We should add specific messages.

---

### 1.3 Test Coverage Gaps

| Area                                        | Current Coverage               | Gap                                                                              |
| ------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `hint`/`suggest` CLI commands               | **None**                       | No test for the hint command path added last session                             |
| `check` announcement                        | **None**                       | No test that `*** CHECK! ***` is printed                                         |
| Promotion (physical)                        | None (only planner ops tested) | `handle_promotion` never tested end-to-end                                       |
| Under-promotion (`a7a8n`)                   | None                           | Only queen promotion tested in planner                                           |
| `print_game_over` for 50-move/threefold     | None                           | Only checkmate and stalemate are tested                                          |
| `process_human_turn` with multiple inputs   | None                           | No test simulating re-entry after illegal move                                   |
| `BoardObserver` false positive on graveyard | None                           | No test placing a captured piece in the graveyard and running `verify_stability` |
| `teleport_piece` success path               | None                           | Only the error path is tested                                                    |
| `_is_piece_body` edge cases                 | None                           | No test for skip-list filtering                                                  |
| Queenside castling                          | None                           | Only kingside tested                                                             |

---

### 1.4 Proposed Changes for Phase 1

#### [MODIFY] [game_runtime.py](file:///home/user/projects/robo_chess_2/src/game_runtime.py)
- **`process_human_turn`**: Loop internally until a move is played or quit is requested. `hint`/`suggest` and illegal moves re-prompt.
- **`handle_promotion`**: Fix `rgba` to use correct piece color.
- **`print_game_over`**: Add specific messages for threefold repetition, 50-move rule, and insufficient material.
- **`_is_piece_body`**: Change skip-prefix matching from `in` to `startswith`.

#### [MODIFY] [board_observer.py](file:///home/user/projects/robo_chess_2/src/observation/board_observer.py)
- **`verify_stability`**: Accept optional `active_piece_names` set. Skip pieces not in that set if provided.

#### [MODIFY] [execution_controller.py](file:///home/user/projects/robo_chess_2/src/control/execution_controller.py)
- Remove class-level graveyard counter defaults (lines 103-104).

#### [MODIFY] [test_chess_manager.py](file:///home/user/projects/robo_chess_2/tests/test_chess_manager.py)
- Add: `test_is_game_over_stalemate`, `test_is_game_over_insufficient_material`

#### [MODIFY] [test_startup_runtime.py](file:///home/user/projects/robo_chess_2/tests/test_startup_runtime.py)
- Add: `test_process_human_turn_hint_prints_suggestion`, `test_process_human_turn_illegal_then_valid_move`, `test_run_turn_check_announced_after_move`, `test_print_game_over_fifty_move_draw`, `test_print_game_over_threefold_draw`, `test_print_game_banner_includes_hint`

#### [MODIFY] [test_operation_planner.py](file:///home/user/projects/robo_chess_2/tests/test_operation_planner.py)
- Add: `test_queenside_castling`, `test_pawn_under_promotion_to_knight`

#### [MODIFY] [test_headless_gameplay.py](file:///home/user/projects/robo_chess_2/tests/test_headless_gameplay.py)
- Add: `test_headless_human_promotion_swaps_piece`, `test_headless_teleport_piece_success_path`
- Fix: `_build_game` to not reset class-level counters.

#### [MODIFY] [test_board_observer.py](file:///home/user/projects/robo_chess_2/tests/test_board_observer.py)
- Add: `test_captured_piece_in_graveyard_does_not_trigger_stability_error`

---

## Phase 2: Interactive GUI, Code Modularity, and Robustness

This phase adds a companion GUI window and makes structural improvements for long-term maintainability.

---

### 2.1 Interactive GUI — Side-by-Side Board Panel

Instead of entering moves in the terminal, users will interact with a **Tkinter** window that runs alongside the MuJoCo viewer.

#### Why Tkinter (not PyQt)?
- Zero extra dependencies — Tkinter ships with Python.
- Lightweight — doesn't fight with MuJoCo's native GLFW window.
- Simple to maintain — this is a companion panel, not a full chess client.

#### GUI Features
| Feature              | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **2D Board View**    | An 8×8 canvas grid showing piece unicode symbols. Mirrors the logical `chess.Board`.                |
| **Click-to-Move**    | Click a piece to select it (highlights legal destinations), click a destination to submit the move. |
| **Text Input**       | A text field for UCI input as a fallback.                                                           |
| **Hint Button**      | Calls `get_ai_move()` and highlights the suggested move on the board.                               |
| **Auto-Play Toggle** | Let the AI play both sides with a checkbox. Useful for demos.                                       |
| **Status Bar**       | Shows current player, last move, check/mate status, move count.                                     |
| **Move History**     | Scrollable list of all moves played so far.                                                         |

#### Architecture
```
┌─────────────────────────────────────────────────┐
│                  main.py                        │
│  ┌───────────────────┐  ┌────────────────────┐ │
│  │  MuJoCo Viewer    │  │ Tkinter BoardPanel │ │
│  │  (GLFW thread)    │  │ (Tk mainloop)      │ │
│  └───────────────────┘  └────────────────────┘ │
│            ▲                      ▲             │
│            │                      │             │
│            ├──── GameLoop ────────┘             │
│            │   (decoupled from I/O)             │
│            ▼                                    │
│     GameSystems / ChessGameManager              │
└─────────────────────────────────────────────────┘
```

---

### 2.2 Code Modularity — Decouple CLI from Runtime

#### Current Problem
`game_runtime.py` is 490 lines mixing:
- Bootstrap logic (`bootstrap_game_systems`)
- Turn orchestration (`run_turn`, `process_human_turn`, `process_ai_turn`)
- Physical execution (`execute_human_ops`, `execute_ai_ops`, `teleport_piece`)
- I/O and display (`print_game_banner`, `print_game_over`, `input()`)

This makes it impossible to swap the terminal UI for a GUI without touching turn logic.

#### Proposed Split

| New File                 | Contents                                                                               | Purpose            |
| ------------------------ | -------------------------------------------------------------------------------------- | ------------------ |
| `src/game_loop.py`       | `GameLoop` class with `step()`, `get_board_state()`, `submit_move()`, `request_hint()` | Pure logic, no I/O |
| `src/cli_ui.py`          | Terminal-based UI (current `print_*` and `input()` calls)                              | CLI frontend       |
| `src/gui/board_panel.py` | Tkinter GUI window                                                                     | GUI frontend       |
| `src/game_runtime.py`    | Slimmed to bootstrap + launch frontends                                                | Wiring only        |

The `GameLoop` class will expose a clean API:
```python
class GameLoop:
    def __init__(self, systems: GameSystems): ...
    def get_state(self) -> GameState: ...    # board, turn, is_check, is_over, etc.
    def submit_move(self, uci: str) -> MoveResult: ...
    def request_hint(self) -> chess.Move: ...
    def execute_ai_turn(self, viewer=None) -> MoveResult: ...
```

Both `cli_ui.py` and `board_panel.py` consume this API. The GUI can call `submit_move()` on click, and the CLI can call it on `input()`.

---

### 2.3 Robustness Improvements

#### A. Stockfish Connection Pooling
Currently, `ChessGameManager.get_ai_move()` opens and closes a Stockfish process on **every single call** ([chess_manager.py:50-62](file:///home/user/projects/robo_chess_2/src/logic/chess_manager.py#L50-L62)). This is expensive and flaky under load. 

> [!TIP]
> **Fix**: Open the engine once in `__init__` and reuse it. Close it in a `close()` method called at shutdown.

#### B. Structured Logging
The current logging uses `log_event()` which produces key=value strings. This is fine for reading but hard to query programmatically.

> [!TIP]
> **Fix**: Switch to `structlog` or at minimum emit JSON-formatted log records. This lets you `jq` the logs for post-mortem analysis.

#### C. Configuration Validation at Startup
`config.py` does raw dictionary access with no validation. If someone misspells a key in `runtime.yaml`, they get a `KeyError` at import time with no indication of which setting is wrong.

> [!TIP]
> **Fix**: Add a `_validate_config()` function that checks all required keys exist and have sane types/ranges before any other module imports them.

#### D. Graceful Degradation When Stockfish is Missing
Currently, if Stockfish is not installed, `get_ai_move()` raises `AIEngineError` and the game crashes. For a better experience, the game could fall back to a random legal move with a warning.

---

### 2.4 Proposed Changes for Phase 2

#### [NEW] [src/game_loop.py](file:///home/user/projects/robo_chess_2/src/game_loop.py)
- `GameLoop` class: `get_state()`, `submit_move()`, `request_hint()`, `execute_ai_turn()`
- `GameState` and `MoveResult` dataclasses

#### [NEW] [src/gui/board_panel.py](file:///home/user/projects/robo_chess_2/src/gui/board_panel.py)
- `BoardPanel` Tkinter widget: 2D board, click-to-move, hint button, status bar, move history

#### [MODIFY] [src/game_runtime.py](file:///home/user/projects/robo_chess_2/src/game_runtime.py)
- Slim down to bootstrap + launch. Delegate turn logic to `GameLoop`.

#### [NEW] [src/cli_ui.py](file:///home/user/projects/robo_chess_2/src/cli_ui.py)
- Extract terminal `print_*` and `input()` functions as a CLI frontend consuming `GameLoop`.

#### [MODIFY] [src/logic/chess_manager.py](file:///home/user/projects/robo_chess_2/src/logic/chess_manager.py)
- Open Stockfish once at init, reuse across calls, add `close()` method.

#### [MODIFY] [src/config.py](file:///home/user/projects/robo_chess_2/src/config.py)
- Add `_validate_config()` with type and range checks.

#### [NEW] [tests/test_game_loop.py](file:///home/user/projects/robo_chess_2/tests/test_game_loop.py)
- Tests for the decoupled `GameLoop` API.

#### [NEW] [tests/test_board_panel.py](file:///home/user/projects/robo_chess_2/tests/test_board_panel.py)
- Headless tests for the Tkinter widget (using Tk's virtual event system).

---

## Summary

| Phase | Scope                         | Key Deliverables                                                                 | Estimated New/Modified Files  |
| ----- | ----------------------------- | -------------------------------------------------------------------------------- | ----------------------------- |
| **1** | Bugs + Game States + Tests    | 6 bug fixes, ~15 new tests, draw-reason messages                                 | 8 modified files              |
| **2** | GUI + Modularity + Robustness | Tkinter board panel, `GameLoop` extraction, Stockfish pooling, config validation | 4 new files, 4 modified files |

## Open Questions

> [!IMPORTANT]
> 1. **GUI library**: I proposed Tkinter for zero-dependency simplicity. Would you prefer something richer like **Dear PyGui** or **PyQt6**? Dear PyGui is interesting because it can render inside a game-like loop and might feel more natural alongside MuJoCo.
> 2. **Auto-play mode**: Should the GUI support a mode where the AI plays both sides automatically (useful for demos/debugging)? I included it in the plan but want to confirm.
> 3. **Stockfish fallback**: When Stockfish is not installed, should the game (a) crash with a clear error, (b) fall back to random legal moves, or (c) fall back to a simple minimax evaluator?

## Verification Plan

### Automated Tests
```bash
PYTHONPATH=. pytest tests/ -v --tb=short
```

### Manual Verification
- Launch with `python main.py` and visually confirm the GUI board panel appears alongside the MuJoCo viewer.
- Enter moves via click-to-move and verify the physical robot responds.
- Test the hint button.
- Play to checkmate and verify the game-over state is displayed in both CLI and GUI.
