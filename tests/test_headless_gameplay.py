import chess
import mujoco
import numpy as np
import pytest
import re

from scripts.generate_xml import generate_chess_world
from src.config import BLACK_GRAVEYARD_ORIGIN, N_SUBSTEPS, SCENE_XML, SQUARE_SIZE, WHITE_GRAVEYARD_ORIGIN, Z_GRASP
from src.control.execution_controller import ExecutionController, OpResult
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import BoardStateError, ExecutionError, PieceLookupError, StabilityError
from src.game_runtime import execute_ai_ops, execute_human_ops, initialize_square_to_piece, teleport_piece
from src.logic.chess_manager import ChessGameManager
from src.logic.operation_planner import OperationPlanner, PickPlaceOp
from src.runtime_guard import validate_board_state


def _build_game():
    ExecutionController._white_graveyard_count = 0
    ExecutionController._black_graveyard_count = 0

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    manager = ChessGameManager()
    planner = OperationPlanner()
    square_to_piece = initialize_square_to_piece(model, data)

    game = {
        "model": model,
        "data": data,
        "env": env,
        "controller": controller,
        "manager": manager,
        "planner": planner,
        "square_to_piece": square_to_piece,
        "home_grip": env.get_grip_pos().copy(),
        "check_registry": None,
        "arm_home_grip": env.get_grip_pos().copy(),
    }
    validate_board_state(model, data, manager.board, square_to_piece, controller)
    return game


def _play_turn(game, uci, physical_black=False):
    manager = game["manager"]
    planner = game["planner"]
    controller = game["controller"]
    square_to_piece = game["square_to_piece"]

    assert manager.validate_move(uci), f"Illegal scripted test move: {uci}"
    move = chess.Move.from_uci(uci)
    ops = planner.generate_operations(move, manager.board, square_to_piece)

    if manager.board.turn == chess.BLACK and physical_black:
        execute_ai_ops(
            ops,
            controller,
            square_to_piece,
            game["model"],
            game["data"],
            viewer=None,
        )
    else:
        execute_human_ops(
            game["model"],
            game["data"],
            ops,
            controller,
            square_to_piece,
        )

    manager.push_move(uci)
    validate_board_state(
        game["model"],
        game["data"],
        manager.board,
        square_to_piece,
        controller,
    )
    return ops


def _assert_arm_home(game, atol=0.01):
    np.testing.assert_allclose(
        game["env"].get_grip_pos(),
        game["home_grip"],
        atol=atol,
    )


def _assert_piece_on_square(game, square_name, expected_piece_name):
    assert game["square_to_piece"].get(square_name) == expected_piece_name
    expected_pos = game["controller"].get_square_pos(square_name)
    actual_pos = game["data"].body(expected_piece_name).xpos.copy()
    xy_error = np.linalg.norm(actual_pos[:2] - expected_pos[:2])
    z_error = abs(actual_pos[2] - expected_pos[2])
    assert xy_error < 0.02
    assert z_error < 0.02


def _set_piece_tilt(game, piece_name, quat):
    body = game["model"].body(piece_name)
    joint_id = body.jntadr[0]
    game["data"].joint(joint_id).qpos[3:7] = quat
    mujoco.mj_forward(game["model"], game["data"])


def test_headless_initial_board_is_valid_and_arm_is_home():
    game = _build_game()
    assert len(game["square_to_piece"]) == 32
    validate_board_state(
        game["model"],
        game["data"],
        game["manager"].board,
        game["square_to_piece"],
        game["controller"],
    )
    _assert_arm_home(game)


def test_headless_human_e2e4_updates_board_scene_and_mapping():
    game = _build_game()
    _play_turn(game, "e2e4")

    assert "e2" not in game["square_to_piece"]
    _assert_piece_on_square(game, "e4", "w_pawn_5")
    _assert_arm_home(game)


def test_headless_ai_g8f6_logs_success_and_workspace_warning(caplog):
    game = _build_game()
    _play_turn(game, "e2e4")

    with caplog.at_level("INFO"):
        _play_turn(game, "g8f6", physical_black=True)

    assert any("OUT_OF_POLICY_WORKSPACE" in rec.message for rec in caplog.records)
    assert any("Result: SUCCESS" in rec.message for rec in caplog.records)
    _assert_piece_on_square(game, "f6", "b_knight_2")
    _assert_arm_home(game)


def test_headless_multi_turn_opening_sequence_one_stays_valid_each_turn():
    game = _build_game()

    for uci, physical_black in [
        ("e2e4", False),
        ("e7e5", True),
        ("g1f3", False),
        ("d7d6", True),
    ]:
        _play_turn(game, uci, physical_black=physical_black)
        validate_board_state(
            game["model"],
            game["data"],
            game["manager"].board,
            game["square_to_piece"],
            game["controller"],
        )
        _assert_arm_home(game)

    _assert_piece_on_square(game, "e4", "w_pawn_5")
    _assert_piece_on_square(game, "f3", "w_knight_2")
    _assert_piece_on_square(game, "e5", "b_pawn_5")
    _assert_piece_on_square(game, "d6", "b_pawn_4")


def test_headless_multi_turn_opening_sequence_two_stays_valid_each_turn():
    game = _build_game()

    for uci, physical_black in [
        ("c2c4", False),
        ("e7e5", True),
        ("b1c3", False),
        ("d7d6", True),
    ]:
        _play_turn(game, uci, physical_black=physical_black)
        validate_board_state(
            game["model"],
            game["data"],
            game["manager"].board,
            game["square_to_piece"],
            game["controller"],
        )
        _assert_arm_home(game)

    _assert_piece_on_square(game, "c4", "w_pawn_3")
    _assert_piece_on_square(game, "c3", "w_knight_1")
    _assert_piece_on_square(game, "e5", "b_pawn_5")
    _assert_piece_on_square(game, "d6", "b_pawn_4")


def test_headless_human_capture_sends_captured_piece_to_black_graveyard():
    game = _build_game()

    _play_turn(game, "e2e4")
    _play_turn(game, "d7d5")
    _play_turn(game, "e4d5")

    _assert_piece_on_square(game, "d5", "w_pawn_5")
    captured_pos = game["data"].body("b_pawn_4").xpos.copy()
    np.testing.assert_allclose(captured_pos, BLACK_GRAVEYARD_ORIGIN, atol=0.002)
    assert "d7" not in game["square_to_piece"]


def test_headless_human_castling_moves_king_and_rook_to_expected_squares():
    game = _build_game()

    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1"]:
        _play_turn(game, uci)

    _assert_piece_on_square(game, "g1", "w_king")
    _assert_piece_on_square(game, "f1", "w_rook_2")
    assert "e1" not in game["square_to_piece"]
    assert "h1" not in game["square_to_piece"]
    _assert_arm_home(game)


def test_headless_two_white_pawn_pushes_keep_board_consistent():
    game = _build_game()

    for uci in ["e2e4", "e7e5", "d2d4", "d7d6"]:
        _play_turn(game, uci)
        validate_board_state(
            game["model"],
            game["data"],
            game["manager"].board,
            game["square_to_piece"],
            game["controller"],
        )

    _assert_piece_on_square(game, "e4", "w_pawn_5")
    _assert_piece_on_square(game, "d4", "w_pawn_4")
    _assert_piece_on_square(game, "e5", "b_pawn_5")
    _assert_piece_on_square(game, "d6", "b_pawn_4")


def test_headless_direct_physical_knight_sequence_keeps_positions_valid():
    game = _build_game()
    controller = game["controller"]

    first = controller.execute_op(PickPlaceOp("g8", "f6", "b_knight_2"), viewer=None)
    assert first.success, first.details
    _assert_arm_home(game)

    second = controller.execute_op(PickPlaceOp("f6", "e4", "b_knight_2"), viewer=None)
    assert second.success, second.details
    _assert_arm_home(game)

    piece = game["data"].body("b_knight_2")
    dest = controller.get_square_pos("e4")
    xy_error = np.linalg.norm(piece.xpos[:2] - dest[:2])
    z_error = abs(piece.xpos[2] - dest[2])
    assert xy_error < 0.02
    assert z_error < 0.02


def test_headless_graveyard_targets_are_reachable_and_raised():
    game = _build_game()
    white_target = game["controller"].get_pos("white_graveyard")
    black_target = game["controller"].get_pos("black_graveyard")

    assert ExecutionController._is_reachable(white_target)
    assert ExecutionController._is_reachable(black_target)
    assert white_target[2] > 0.413
    assert black_target[2] > 0.413
    np.testing.assert_allclose(white_target, WHITE_GRAVEYARD_ORIGIN, atol=1e-6)
    np.testing.assert_allclose(black_target, BLACK_GRAVEYARD_ORIGIN, atol=1e-6)


def test_generated_xml_contains_rank_and_file_labels():
    xml = generate_chess_world()

    assert "label_file_a_" in xml
    assert "label_file_h_" in xml
    assert "label_file_near_a_" in xml
    assert "label_file_near_h_" in xml
    assert "label_rank_1_" in xml
    assert "label_rank_8_" in xml
    assert 'mesh="chess_pawn_mesh"' in xml
    assert 'mesh="chess_king_mesh"' in xml
    assert '<body name="w_pawn_1"' in xml
    assert '<geom type="mesh" mesh="chess_pawn_mesh" material="white_piece"' in xml


def test_generated_xml_file_labels_read_a_to_h_on_near_edge():
    xml = generate_chess_world()

    def _x_for(label_prefix):
        match = re.search(rf'{label_prefix}_0" type="box" size="[^"]+" pos="([0-9.]+) ', xml)
        assert match is not None, label_prefix
        return float(match.group(1))

    far_a_x = _x_for("label_file_a")
    far_h_x = _x_for("label_file_h")
    near_a_x = _x_for("label_file_near_a")
    near_h_x = _x_for("label_file_near_h")

    assert far_a_x > far_h_x
    assert near_a_x > near_h_x


def test_generated_xml_board_has_visible_cell_gaps_and_frame():
    xml = generate_chess_world()

    assert 'geom name="board_underlay"' in xml
    assert 'geom name="board_frame_north"' in xml
    assert 'geom name="board_frame_south"' in xml
    assert 'geom name="board_frame_west"' in xml
    assert 'geom name="board_frame_east"' in xml
    assert 'size="0.033 0.033 0.0005"' in xml


def test_generated_xml_board_underlay_sits_below_playable_squares():
    xml = generate_chess_world()

    underlay_match = re.search(r'geom name="board_underlay".*pos="[0-9.]+ [0-9.]+ ([0-9.]+)"', xml)
    square_match = re.search(r'geom name="square_0_0".*pos="[0-9.]+ [0-9.]+ ([0-9.]+)"', xml)

    assert underlay_match is not None
    assert square_match is not None
    assert float(underlay_match.group(1)) < float(square_match.group(1))


def test_headless_graveyard_trays_are_separated_from_board_edge():
    game = _build_game()
    table_geom = game["model"].geom("table")
    table_min_y = float(table_geom.pos[1] - table_geom.size[1])
    board_min_y = float(game["controller"].get_square_pos("a8")[1] - 0.5 * SQUARE_SIZE)

    white_body = game["model"].body("white_graveyard")
    black_body = game["model"].body("black_graveyard")
    white_geom = game["model"].geom(white_body.geomadr[0])
    black_geom = game["model"].geom(black_body.geomadr[0])

    white_max_y = float(white_body.pos[1] + white_geom.size[1])
    black_max_y = float(black_body.pos[1] + black_geom.size[1])

    assert table_min_y <= white_max_y < board_min_y
    assert table_min_y <= black_max_y < board_min_y


def test_headless_nudged_piece_raises_board_state_error():
    game = _build_game()
    _play_turn(game, "e2e4")

    body = game["data"].body("w_pawn_5")
    joint_id = game["model"].body("w_pawn_5").jntadr[0]
    game["data"].joint(joint_id).qpos[0] = body.xpos[0] + 0.03
    mujoco.mj_forward(game["model"], game["data"])

    with pytest.raises(BoardStateError, match="coordinate mismatch"):
        validate_board_state(game["model"], game["data"], game["manager"].board, game["square_to_piece"], game["controller"])


def test_headless_tilted_piece_raises_stability_error():
    game = _build_game()
    _play_turn(game, "e2e4")

    _set_piece_tilt(game, "w_pawn_5", np.array([0.923, 0.382, 0.0, 0.0]))

    with pytest.raises(StabilityError, match="tilted"):
        validate_board_state(game["model"], game["data"], game["manager"].board, game["square_to_piece"], game["controller"])


def test_headless_invalid_piece_teleport_raises_piece_lookup_error():
    game = _build_game()

    with pytest.raises(PieceLookupError, match="not found"):
        teleport_piece(
            game["model"],
            game["data"],
            "not_a_real_piece",
            np.array([1.0, 1.0, Z_GRASP]),
        )


def test_headless_corrupted_square_mapping_raises_board_state_error():
    game = _build_game()
    _play_turn(game, "e2e4")
    game["square_to_piece"]["e4"] = "w_pawn_4"

    with pytest.raises(BoardStateError):
        validate_board_state(
            game["model"],
            game["data"],
            game["manager"].board,
            game["square_to_piece"],
            game["controller"],
        )


def test_headless_execute_ai_ops_raises_on_unsuccessful_controller_result(monkeypatch):
    game = _build_game()
    ops = [PickPlaceOp("g8", "f6", "b_knight_2")]

    def fake_execute_op(op, viewer=None):
        return OpResult(False, 12, float("inf"), 3, "FAILED: synthetic test failure")

    monkeypatch.setattr(game["controller"], "execute_op", fake_execute_op)

    with pytest.raises(ExecutionError, match="synthetic test failure"):
        execute_ai_ops(
            ops,
            game["controller"],
            game["square_to_piece"],
            game["model"],
            game["data"],
            viewer=None,
        )
