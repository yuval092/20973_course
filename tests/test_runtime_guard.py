"""
Unit tests for the runtime guard and board state validation.

This module verifies that the system can correctly detect discrepancies
between the logical chess board state and the physical Mujoco simulation,
including coordinate mismatches, tilted pieces, and mapping inconsistencies.
"""

from unittest.mock import MagicMock, patch
import chess
import numpy as np
import pytest

from src.exceptions import BoardStateError, StabilityError
from src.runtime_guard import run_freeze_loop, validate_board_state


class _DummyController:
    """
    A minimal mock controller for providing square positions.
    """
    def __init__(self, positions):
        self.positions = positions

    def get_pos(self, square_name):
        """
        Return the pre-defined position for a square.
        """
        return np.array(self.positions[square_name], dtype=np.float64)


class TestRuntimeGuard:
    """
    Test suite for runtime state validation and monitoring.
    """

    def test_validate_board_state_accepts_matching_board(self):
        """
        Verify that validation succeeds when the physical and logical states match.
        """
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
            validate_board_state(
                MagicMock(), mj_data, board, square_to_piece, controller
            )

    def test_validate_board_state_raises_for_coordinate_mismatch(self):
        """
        Verify that validation fails when a piece is not at its expected coordinates.
        """
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        square_to_piece = {"e4": "w_pawn_5"}
        controller = _DummyController({"e4": [1.0, 2.0, 3.0]})

        mj_data = MagicMock()
        body = MagicMock()
        # Offset X coordinate by 3cm
        body.xpos = np.array([1.03, 2.0, 3.0])
        body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
        mj_data.body.return_value = body

        with patch("src.runtime_guard.BoardObserver.verify_stability"):
            with pytest.raises(BoardStateError, match="coordinate mismatch"):
                validate_board_state(
                    MagicMock(), mj_data, board, square_to_piece, controller
                )

    def test_validate_board_state_raises_for_tilted_piece(self):
        """
        Verify that validation fails when a piece is tilted beyond the stability threshold.
        """
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        square_to_piece = {"e4": "w_pawn_5"}
        controller = _DummyController({"e4": [1.0, 2.0, 3.0]})

        mj_data = MagicMock()
        body = MagicMock()
        body.xpos = np.array([1.0, 2.0, 3.0])
        # Tilted quaternion
        body.xquat = np.array([0.923, 0.382, 0.0, 0.0])
        mj_data.body.return_value = body

        with patch("src.runtime_guard.BoardObserver.verify_stability"):
            with pytest.raises(StabilityError, match="tilted at turn start"):
                validate_board_state(
                    MagicMock(), mj_data, board, square_to_piece, controller
                )

    def test_validate_board_state_raises_for_mapping_mismatch(self):
        """
        Verify that validation fails when the piece mapping is inconsistent with the board.
        """
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        # Empty mapping where a pawn should be
        square_to_piece = {}
        controller = _DummyController({})

        with patch("src.runtime_guard.BoardObserver.verify_stability"):
            with pytest.raises(BoardStateError, match="mapping mismatch"):
                validate_board_state(
                    MagicMock(), MagicMock(), board, square_to_piece, controller
                )

    def test_run_freeze_loop_syncs_until_viewer_stops(self):
        """
        Verify that the freeze loop correctly synchronizes the viewer and physics.
        """
        viewer = MagicMock()
        viewer.is_running.side_effect = [True, True, False]
        mj_model = MagicMock()
        mj_data = MagicMock()

        with patch("src.runtime_guard.time.sleep"):
            with patch("src.runtime_guard.mujoco.mj_forward") as mj_forward:
                run_freeze_loop(
                    viewer=viewer, mj_model=mj_model, 
                    mj_data=mj_data, max_cycles=10
                )

        assert viewer.sync.call_count == 2
        assert mj_forward.call_count == 2

    def test_validate_board_state_real_scene_initial_board_passes(self):
        """
        Verify that the standard initial board setup passes validation in the real scene.
        """
        import mujoco
        from src.config import N_SUBSTEPS, SCENE_XML
        from src.env.chess_pick_place_env import ChessPickPlaceEnv
        from src.control.execution_controller import ExecutionController
        from src.bootstrap import SystemBootstrapper
        
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)
        board = chess.Board()
        
        # Should not raise any exceptions
        validate_board_state(model, data, board, square_to_piece, controller)
