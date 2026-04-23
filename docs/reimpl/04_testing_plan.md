# 04 — Testing Plan

## Philosophy

- **Unit tests** are pure Python with no MuJoCo dependency. They use `DummyEnv`, `DummyModel`, and `DummyData` shim classes to simulate MuJoCo responses. Test one class in isolation.
- **Integration tests** require a real MuJoCo scene (`chess_world.xml`). They run headlessly (no viewer). They verify that the env, controller, and bootstrap work correctly with real physics.
- **System tests** exercise multiple components together in a full turn cycle, headlessly.
- **Regression tests** verify that specific known bugs cannot reappear.
- All tests are run with `PYTHONPATH=.` from the project root.
- Stockfish is optional: tests that call `get_ai_move()` should skip gracefully if Stockfish is not installed.
- The RL model is optional for controller unit tests: pass `rl_model=None` to `ExecutionController`.

---

## Test File Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── test_config.py
├── test_exceptions.py
├── test_logging_utils.py
├── test_models.py
├── test_chess_manager.py
├── test_operation_planner.py
├── test_obs_wrapper.py
├── test_board_observer.py
├── test_execution_controller.py     # Pure unit tests (DummyEnv)
├── test_runtime_guard.py
├── test_health_checks.py
├── test_board_panel.py
├── test_bootstrap.py                # Integration (real MuJoCo)
├── test_chess_pick_place_env.py     # Integration (real MuJoCo)
├── test_headless_execution.py       # Integration (real arm)
├── test_headless_gameplay.py        # System test
├── test_graveyard_operations.py     # Integration
├── test_startup_runtime.py          # System test
├── test_drift_and_stability.py      # Integration
└── test_training_prep.py            # Training config sanity
```

---

## `tests/conftest.py`

Shared pytest fixtures:

```python
@pytest.fixture
def manager():
    """ChessGameManager with Stockfish (skips gracefully if missing)."""
    ...

@pytest.fixture
def scene():
    """(MjModel, MjData) loaded from SCENE_XML."""
    ...

@pytest.fixture
def env(scene):
    """ChessPickPlaceEnv wrapping a real scene."""
    ...

@pytest.fixture
def controller(env):
    """ExecutionController(rl_model=None, env=env)."""
    ...
```

---

## `tests/test_config.py`

| Test | Verifies |
|------|----------|
| `test_loads_without_error` | `from src.config import BOARD_CENTER` does not raise |
| `test_board_center_is_list_of_3` | `len(BOARD_CENTER) == 3` |
| `test_square_size_positive` | `SQUARE_SIZE > 0` |
| `test_z_safe_above_z_grasp` | `Z_SAFE > Z_GRASP` |
| `test_gripper_open_positive` | `GRIPPER_OPEN > 0` |
| `test_local_model_path_is_string` | `LOCAL_MODEL_PATH` is a `str` |
| `test_missing_key_raises_value_error` | `ConfigManager` with YAML missing required key raises `ValueError` |
| `test_wrong_type_raises_value_error` | `ConfigManager` with wrong type for a required key raises `ValueError` |
| `test_get_returns_value` | `_MANAGER.get("board_center")` equals `BOARD_CENTER` |

---

## `tests/test_exceptions.py`

| Test | Verifies |
|------|----------|
| `test_hierarchy` | `ArmStateError` is instance of `RuntimeCheckError` and `RoboChessError` |
| `test_all_exceptions_catchable_as_base` | All exception types can be caught as `RoboChessError` |
| `test_exceptions_have_message` | Each exception preserves the message string |

---

## `tests/test_logging_utils.py`

| Test | Verifies |
|------|----------|
| `test_format_value_float` | `format_value(1.23456)` returns `"1.2346"` |
| `test_format_value_str` | `format_value("hello")` returns `"hello"` |
| `test_format_fields_sorted` | Fields appear in alphabetical order |
| `test_format_fields_skips_none` | Keys with `None` values are omitted |
| `test_log_event_no_fields` | Message is just the event name |
| `test_log_event_with_fields` | Message is `"event | key=value"` format |

---

## `tests/test_models.py`

| Test | Verifies |
|------|----------|
| `test_game_systems_stores_fields` | Constructor assigns all fields |
| `test_captured_count_defaults_to_zero` | `captured_count == {"white": 0, "black": 0}` when not provided |
| `test_move_result_fields` | `success`, `message`, `error` accessible |
| `test_game_state_flags` | All boolean flags accessible |

---

## `tests/test_chess_manager.py`

| Test | Verifies |
|------|----------|
| `test_initial_board` | White to move, 20 legal moves |
| `test_validate_move_legal` | `"e2e4"` returns `True` |
| `test_validate_move_illegal` | `"e2e5"` returns `False` |
| `test_validate_move_invalid_uci` | `"invalid"` returns `False` |
| `test_push_move_updates_board` | After `push_move("e2e4")`, pawn on e4, None on e2, Black to move |
| `test_push_move_illegal_raises` | `push_move("e2e5")` raises `ValueError` |
| `test_push_move_invalid_raises` | `push_move("zzz")` raises `ValueError` |
| `test_is_game_over_false_at_start` | `is_game_over()` returns `False` on fresh board |
| `test_is_game_over_fools_mate` | After Fool's Mate moves, `is_game_over()` returns `True` |
| `test_is_game_over_stalemate` | Set stalemate FEN, `is_game_over()` returns `True` |
| `test_is_game_over_insufficient_material` | KvK FEN, `is_game_over()` returns `True` |
| `test_get_ai_move_returns_legal_move` | If Stockfish available, returned move is legal (skip if not) |
| `test_context_manager_closes_engine` | `with ChessGameManager() as m: pass` doesn't raise |

---

## `tests/test_operation_planner.py`

Uses a chess board set to specific positions; `square_to_piece` is manually crafted.

| Test | Verifies |
|------|----------|
| `test_simple_move` | Single op: `target_square=e2`, `dest_square=e4`, correct `piece_name` |
| `test_simple_capture` | Two ops: capture op (to graveyard) first, then moving piece |
| `test_capture_graveyard_color` | White captured piece goes to `"white_graveyard"` |
| `test_en_passant` | Captured pawn is at `(dest_file, src_rank)`, not `(dest_file, dest_rank)` |
| `test_castling_kingside` | Two ops: king e1→g1, rook h1→f1 |
| `test_castling_queenside` | Two ops: king e1→c1, rook a1→d1 |
| `test_promotion_flag` | Op has `promotion=chess.QUEEN` (or other type) |
| `test_missing_piece_raises` | `_require_piece` raises `KeyError` when square not in mapping |
| `test_en_passant_captured_square_not_in_mapping_raises` | Captured en passant square missing → `ValueError` |

---

## `tests/test_obs_wrapper.py`

Uses a real MuJoCo env (requires scene XML).

| Test | Verifies |
|------|----------|
| `test_output_shape` | `reconstruct_obs(...)` returns shape `(26,)` |
| `test_output_dtype` | Output dtype is `float32` |
| `test_time_feature_clipped` | `time_feature=2.0` → output[25] == 1.0; `time_feature=-1.0` → output[25] == 0.0 |
| `test_grip_pos_in_output` | First 3 elements match `data.site(GRIP_SITE).xpos` |

---

## `tests/test_board_observer.py`

Uses `DummyModel`/`DummyData` shims.

| Test | Verifies |
|------|----------|
| `test_upright_piece_passes` | `up_z = 1.0` → no exception |
| `test_tipped_piece_raises` | `up_z < TILT_THRESHOLD_COS` → `StabilityError` |
| `test_buried_piece_raises` | `pos[2] < TABLE_HEIGHT - 0.01` → `StabilityError` |
| `test_displaced_piece_raises` | Piece near board but > `SQUARE_SIZE * DISPLACEMENT_THRESHOLD` from square center → `StabilityError` |
| `test_ignored_bodies_skipped` | Bodies named "world_body", "table_top" etc. → no check |
| `test_active_piece_names_filter` | Only pieces in `active_piece_names` are checked |
| `test_spare_pieces_skipped` | Bodies containing "spare" are ignored |

---

## `tests/test_execution_controller.py`

All tests use `DummyEnv` (no real MuJoCo needed for unit tests).

### DummyEnv spec

Must implement: `get_piece_pos()`, `get_grip_pos()`, `get_piece_linear_velocity()`, `get_gripper_finger_qpos()`, `get_obs()`, `force_gripper_open()`, `set_gripper_target()`, `set_target()`, `step()`, `home_grip_pos` property, `data.body()`, `model`.

| Test | Verifies |
|------|----------|
| `test_reachability_accepts_in_bounds` | `_is_reachable([1.20, 0.60, 0.50])` → True |
| `test_reachability_rejects_out_of_bounds` | `_is_reachable([1.60, 0.60, 0.50])` → False |
| `test_policy_issues_far_square` | Source far from policy center → issues list non-empty |
| `test_policy_issues_mid_board_clean` | Mid-board move → issues list empty |
| `test_stage_targets_all_keys_present` | `_build_stage_targets` returns dict with all 10 stage keys |
| `test_stage_targets_prehover_src_z_is_z_safe` | `PREHOVER_SRC[2] == Z_SAFE` |
| `test_stage_targets_descend_dest_uses_close_descend_offset` | `DESCEND_DEST[2] == dest[2] - CLOSE_DESCEND_OFFSET` |
| `test_resolve_stage_goal_descend_dest` | Returns `[dest_x, dest_y, dest_z - CLOSE_DESCEND_OFFSET]` |
| `test_resolve_stage_goal_descend_src_uses_live_piece_z` | Reads live piece Z, adds `_source_descend_offset_z` |
| `test_compose_guided_action_scripted_only` | `use_rl=False` returns clipped delta-based action |
| `test_compose_guided_action_rl_primary_returns_rl_action` | `use_rl=True` with a loaded model returns the raw RL action (not a blend); gripper dim is zeroed |
| `test_compose_guided_action_rl_fallback_when_no_model` | `use_rl=True` but `rl_model=None` falls back to scripted action |
| `test_lift_verify_calls_set_target_and_resets_steps` | LIFT_VERIFY stage calls `env.set_target(piece_name, goal)` and `env.reset_elapsed_steps()` before motion |
| `test_get_square_pos_a1_matches_xml` | `get_square_pos("a1")[:2]` matches expected XYZ from XML (locks down coordinate system) |
| `test_finalize_placement_fails_when_piece_off_target` | Piece far from goal → `success=False` |
| `test_finalize_placement_succeeds_when_settled` | Piece at goal with zero velocity → `success=True` |
| `test_released_piece_issue_xy_error` | XY too large → non-None issue string |
| `test_released_piece_issue_z_error` | Z too large → non-None issue string |
| `test_released_piece_issue_speed` | Speed too large (when jitter not tolerated) → non-None |
| `test_released_piece_issue_clean` | Within all tolerances → `None` |
| `test_settle_released_piece_success` | Piece at goal, zero velocity → `(True, N)` |
| `test_settle_released_piece_timeout` | Piece never settles → `(False, max_steps)` |
| `test_placement_stability_issue_tipped` | `up_z < 0.966` → "tipped" string |
| `test_placement_stability_issue_elevated` | Too high → "elevated" string |
| `test_placement_stability_issue_sunk` | Too low → "sunk" string |
| `test_placement_stability_issue_ok` | Correct position and upright → `None` |
| `test_release_clearance_z_above_z_safe` | Result always ≥ `Z_SAFE + RELEASE_CLEARANCE_MARGIN` |
| `test_retract_arm_calls_force_gripper_open` | `_retract_arm` calls `force_gripper_open` exactly once |
| `test_retract_arm_moves_to_home` | Final move target is `home_grip_pos` |
| `test_preflight_unreachable_produces_failure_result` | Stage target outside bounds → `failure_result` not None |
| `test_preflight_reachable_no_failure_result` | All targets in bounds → `failure_result` is None |

---

## `tests/test_runtime_guard.py`

| Test | Verifies |
|------|----------|
| `test_expected_board_squares_initial` | Returns 32 entries from starting position |
| `test_validate_piece_identity_correct` | `w_pawn_1` for white pawn on e2 → no exception |
| `test_validate_piece_identity_wrong_color` | `b_pawn_1` for white pawn → `BoardStateError` |
| `test_validate_piece_identity_wrong_type` | `w_rook_1` for knight → `BoardStateError` |
| `test_validate_board_state_mismatch` | Square_to_piece has extra entry → `BoardStateError` |
| `test_validate_board_state_tipped_piece` | Piece with bad quaternion → `StabilityError` |
| `test_freeze_loop_runs_limited_cycles` | `run_freeze_loop(max_cycles=3)` terminates after 3 cycles |

---

## `tests/test_health_checks.py`

| Test | Verifies |
|------|----------|
| `test_registry_runs_checks_for_hook` | Check registered for hook is called |
| `test_registry_skips_checks_for_wrong_hook` | Check registered for other hook is not called |
| `test_scene_assets_check_missing_body_raises` | Missing `w_king` → `SceneIntegrityError` |
| `test_scene_assets_check_passes_on_valid_scene` | Real scene → no exception |
| `test_mapping_integrity_check_duplicate_pieces` | Two squares → same body name → `MappingIntegrityError` |
| `test_mapping_integrity_check_count_mismatch` | Mapping has 31 entries, board has 32 → `MappingIntegrityError` |
| `test_finite_state_check_nan_raises` | NaN in qpos → `NumericalStabilityError` |
| `test_finite_state_check_inf_raises` | Inf in qvel → `NumericalStabilityError` |
| `test_finite_state_check_passes_on_clean_state` | Normal state → no exception |
| `test_arm_home_pose_check_displaced_raises` | Gripper far from home → `ArmStateError` |
| `test_arm_home_pose_check_fires_at_turn_start` | Hook is TURN_START → check runs (not skipped) |
| `test_robot_workspace_check_outside_raises` | Gripper outside bounds → `ArmStateError` |
| `test_stage_goal_reachability_check_unreachable_raises` | Goal outside bounds → `ArmStateError` |
| `test_stage_goal_reachability_check_none_goal_passes` | `goal=None` → no check, no exception |
| `test_stage_outcome_check_failure_raises` | `success=False` in extra → `ExecutionError` |
| `test_stage_outcome_check_success_passes` | `success=True` → no exception |
| `test_stage_grip_attachment_check_detached_raises` | Big piece-grip distance after LIFT_VERIFY → `ArmStateError` |
| `test_build_default_registry_has_ten_checks` | `build_default_check_registry()` returns registry with 10 checks |

---

## `tests/test_board_panel.py`

Uses `unittest.mock` to mock `GameLoop`.

| Test | Verifies |
|------|----------|
| `test_panel_builds_64_squares` | `len(panel.squares) == 64` |
| `test_refresh_shows_pieces` | After refresh, squares with pieces have non-empty text |
| `test_lock_board_disables_buttons` | All buttons have state `DISABLED` after `lock_board()` |
| `test_unlock_board_enables_buttons` | All buttons have state `NORMAL` after `unlock_board()` |
| `test_append_history_adds_to_listbox` | `append_history("e2e4")` → listbox has 1 entry |
| `test_set_status_updates_var` | `set_status("hello")` → `status_var.get() == "hello"` |
| `test_on_square_click_selects_own_piece` | Click white pawn → `selected_square` is set |
| `test_on_square_click_calls_callback` | Two-click sequence with legal move → `on_move_callback` called with UCI |
| `test_on_square_click_ignores_illegal_move` | Illegal two-click → `on_move_callback` NOT called |

---

## `tests/test_chess_pick_place_env.py` (Integration)

Requires real MuJoCo scene.

| Test | Verifies |
|------|----------|
| `test_env_initializes` | Constructor completes, `observation_space` and `action_space` defined |
| `test_home_grip_pos_set` | `env.home_grip_pos` is a 3-element array |
| `test_set_target_changes_target` | `set_target("w_pawn_1", pos)` → `target_body_name == "w_pawn_1"` |
| `test_step_returns_correct_shapes` | obs["observation"].shape == (26,), achieved_goal.shape == (3,) |
| `test_step_terminated_is_false` | `terminated` is always `False` |
| `test_gripper_target_applies` | `set_gripper_target(0.05)` → finger positions approach open state |
| `test_get_grip_pos_3d` | `get_grip_pos()` returns shape (3,) |
| `test_get_piece_pos_3d` | After `set_target`, `get_piece_pos()` returns shape (3,) |
| `test_piece_mesh_collision_disabled_by_default` | All piece mesh geom contype == 0 at init (including spare pieces) |
| `test_spare_pieces_in_dof_cache` | Spare piece body names appear in `_piece_joint_dofadrs` |
| `test_remaining_time_feature_decrements` | After 10 steps, `remaining_time_feature < 1.0` |
| `test_remaining_time_feature_clipped` | Feature never exceeds 1.0 or drops below 0.0 |
| `test_reset_robot_pose_restores_home` | After movement, `reset_robot_pose()` → grip near original home |

---

## `tests/test_headless_execution.py` (Integration)

Requires real MuJoCo scene. Uses `ExecutionController(rl_model=None, env=env)` (scripted mode only).

| Test | Verifies |
|------|----------|
| `test_execute_op_places_piece_near_dest` | After `execute_op`, piece XY within 20mm of dest |
| `test_execute_op_piece_upright` | Piece `up_z > 0.95` after placement |
| `test_execute_op_success_flag` | `result.success == True` |
| `test_execute_op_sequential_two_moves` | Move piece, then move same piece again; both succeed |
| `test_execute_op_unreachable_returns_failure` | Op with dest outside reachability → `success=False`, no crash |

---

## `tests/test_graveyard_operations.py` (Integration)

| Test | Verifies |
|------|----------|
| `test_physical_move_to_graveyard` | Piece body ends up near graveyard origin after physical execution |
| `test_graveyard_grid_increments` | Two captures → pieces at different grid positions |
| `test_graveyard_count_tracked` | `systems.captured_count["white"]` increments after white piece captured |

---

## `tests/test_startup_runtime.py` (System)

| Test | Verifies |
|------|----------|
| `test_bootstrap_returns_game_systems` | `bootstrap_game_systems()` returns `GameSystems` instance |
| `test_bootstrap_square_to_piece_32_entries` | `len(systems.square_to_piece) == 32` |
| `test_bootstrap_no_duplicate_pieces` | `len(set(square_to_piece.values())) == 32` |
| `test_bootstrap_all_checks_pass` | No exception raised during bootstrap |

---

## `tests/test_headless_gameplay.py` (System)

| Test | Verifies |
|------|----------|
| `test_human_move_updates_board` | After `game_loop.submit_move("e2e4")`, pawn is on e4 in logical board |
| `test_human_move_updates_mapping` | `square_to_piece["e4"]` is set, `"e2"` is removed |
| `test_ai_move_executes` | `game_loop.execute_ai_turn()` returns `MoveResult(success=True)` (skip if Stockfish absent) |
| `test_get_state_reflects_turn` | After white move, `get_state().board.turn == chess.BLACK` |
| `test_get_state_game_not_over_at_start` | `get_state().is_game_over == False` on fresh game |

---

## `tests/test_drift_and_stability.py` (Integration)

| Test | Verifies |
|------|----------|
| `test_idle_scene_stable_after_500_steps` | 500 idle physics steps → `BoardObserver.verify_stability` passes |
| `test_pieces_not_tipped_at_init` | All pieces upright at scene load |
| `test_pieces_on_table_at_init` | All piece Z values ≥ `TABLE_HEIGHT` |

---

## Coverage Expectations

| Module | Target |
|--------|--------|
| `src/exceptions.py` | 100% |
| `src/logging_utils.py` | 100% |
| `src/config.py` | 90%+ |
| `src/models.py` | 100% |
| `src/logic/chess_manager.py` | 90%+ |
| `src/logic/operation_planner.py` | 95%+ |
| `src/env/obs_wrapper.py` | 95%+ |
| `src/observation/board_observer.py` | 90%+ |
| `src/control/execution_controller.py` | 80%+ |
| `src/runtime_guard.py` | 85%+ |
| `src/health_checks.py` | 90%+ |
| `src/bootstrap.py` | 75%+ |
| `src/game_runtime.py` | 75%+ |
| `src/game_loop.py` | 80%+ |
| `src/gui/board_panel.py` | 70%+ |
| `src/app.py` | 60%+ (threading is hard to test) |

---

## Regression Test Checklist

These specific bugs from the original project must not reappear:

| Bug | Regression Test |
|-----|----------------|
| Double-drop (piece placed 10mm too low) | `test_stage_targets_descend_dest_uses_close_descend_offset`: verify `DESCEND_DEST[2] == dest[2] - CLOSE_DESCEND_OFFSET` (not `dest[2] - 2*offset`) |
| Ghost success (missed piece counted as success) | `test_execute_op_success_flag`: physical placement miss → `success=False` |
| Floating graveyards | `test_teleport_to_graveyard`: piece Z is at graveyard platform height, not above it |
| NaN instability | `test_idle_scene_stable_after_500_steps`: no NaN after stepping |
| Piece collision causing phantom forces | `test_pieces_not_tipped_at_init`: scene stable after mesh collision disabled |
| fingers_closed inversion (valid grasps fail) | `test_close_gripper_success_on_piece_contact`: `_close_gripper_with_descent` returns success when `piece_to_grip < desired_gap`, even when `max(finger_qpos) > 0.002` |
| Spare pieces excluded from DOF cache | `test_spare_pieces_in_dof_cache`: spare piece body names present in `_piece_joint_dofadrs` after init |
| Promotion mesh contype=2 phantom contact | `test_promotion_spare_mesh_contype_zero`: after `handle_promotion`, promoted spare mesh geom has `contype=0` |
| desired_goal set to waypoints (policy out-of-distribution) | `test_execute_op_desired_goal_is_dest_pos`: after `execute_op` starts, `env._goal` equals `dest_pos` (not `src_pos` or hover height) |
| Wrong coordinate sign (rank direction) | `test_get_square_pos_a1_matches_xml`: `get_square_pos("a1")` matches XML-derived expected value |
| Box hitboxes (cylinder rolling) | `test_piece_hitbox_is_box`: piece hitbox geoms in XML are type `box`, not `cylinder` |
