import chess
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.exceptions import BoardStateError, StabilityError
from src.runtime_guard import run_freeze_loop, validate_board_state


class _DummyController:
    def __init__(self, positions):
        self.positions = positions

    def get_pos(self, square_name):
        return np.array(self.positions[square_name], dtype=np.float64)


def test_validate_board_state_accepts_matching_board():
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    square_to_piece = {"e4": "w_pawn_5"}
    controller = _DummyController({"e4": [1.0, 2.0, 3.0]})

    mj_data = MagicMock()
    body = MagicMock()
    body.xpos = np.array([1.0, 2.0, 3.0])
    body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
    mj_data.body.return_value = body

    with patch("src.runtime_guard.BoardObserver.verify_stability"):
        validate_board_state(MagicMock(), mj_data, board, square_to_piece, controller)


def test_validate_board_state_raises_for_coordinate_mismatch():
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    square_to_piece = {"e4": "w_pawn_5"}
    controller = _DummyController({"e4": [1.0, 2.0, 3.0]})

    mj_data = MagicMock()
    body = MagicMock()
    body.xpos = np.array([1.03, 2.0, 3.0])
    body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
    mj_data.body.return_value = body

    with patch("src.runtime_guard.BoardObserver.verify_stability"):
        with pytest.raises(BoardStateError, match="coordinate mismatch"):
            validate_board_state(MagicMock(), mj_data, board, square_to_piece, controller)


def test_validate_board_state_raises_for_tilted_piece():
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    square_to_piece = {"e4": "w_pawn_5"}
    controller = _DummyController({"e4": [1.0, 2.0, 3.0]})

    mj_data = MagicMock()
    body = MagicMock()
    body.xpos = np.array([1.0, 2.0, 3.0])
    body.xquat = np.array([0.923, 0.382, 0.0, 0.0])
    mj_data.body.return_value = body

    with patch("src.runtime_guard.BoardObserver.verify_stability"):
        with pytest.raises(StabilityError, match="tilted at turn start"):
            validate_board_state(MagicMock(), mj_data, board, square_to_piece, controller)


def test_validate_board_state_raises_for_mapping_mismatch():
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    square_to_piece = {}
    controller = _DummyController({})

    with patch("src.runtime_guard.BoardObserver.verify_stability"):
        with pytest.raises(BoardStateError, match="mapping mismatch"):
            validate_board_state(MagicMock(), MagicMock(), board, square_to_piece, controller)


def test_run_freeze_loop_syncs_until_viewer_stops():
    viewer = MagicMock()
    viewer.is_running.side_effect = [True, True, False]
    mj_model = MagicMock()
    mj_data = MagicMock()

    with patch("src.runtime_guard.time.sleep"):
        with patch("src.runtime_guard.mujoco.mj_forward") as mj_forward:
            run_freeze_loop(viewer=viewer, mj_model=mj_model, mj_data=mj_data, max_cycles=10)

    assert viewer.sync.call_count == 2
    assert mj_forward.call_count == 2

def test_validate_board_state_real_scene_initial_board_passes():
    import mujoco
    from src.config import N_SUBSTEPS, SCENE_XML
    from src.env.chess_pick_place_env import ChessPickPlaceEnv
    from src.control.execution_controller import ExecutionController
    from src.game_runtime import initialize_square_to_piece
    
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    square_to_piece = initialize_square_to_piece(model, data)
    board = chess.Board()
    
    # Should not raise an exception
    validate_board_state(model, data, board, square_to_piece, controller)

