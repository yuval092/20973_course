"""
Tests for piece graveyard operations and capture logic.

This module verifies that captured pieces are correctly moved to their
respective graveyards (white or black) and that the graveyard counters are
accurately maintained.
"""

import chess
import mujoco
import numpy as np
from src.config import N_SUBSTEPS, SCENE_XML, WHITE_GRAVEYARD_ORIGIN, BLACK_GRAVEYARD_ORIGIN
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.bootstrap import SystemBootstrapper
from src.game_runtime import GameOrchestrator
from src.logic.operation_planner import OperationPlanner


class TestGraveyardOperations:
    """
    Test suite for piece captures and graveyard placement logic.
    """

    def _build_test_env(self):
        """
        Internal helper to initialize a full test environment.

        Returns:
            tuple: (model, data, env, controller, planner, square_to_piece, board)
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        planner = OperationPlanner()
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)
        board = chess.Board()
        return model, data, env, controller, planner, square_to_piece, board

    def _execute_ai_move(self, uci, model, data, controller, planner, square_to_piece, board):
        """
        Internal helper to generate and execute AI operations for a UCI move.
        """
        move = chess.Move.from_uci(uci)
        ops = planner.generate_operations(move, board, square_to_piece)
        GameOrchestrator.execute_ops(model, data, ops, controller, square_to_piece, viewer=None)
        board.push(move)

    def test_headless_ai_capture_to_graveyard_succeeds(self):
        """
        Verify that a single capture correctly moves a piece to the graveyard.
        """
        model, data, _, controller, planner, square_to_piece, board = self._build_test_env()

        # Execute opening
        self._execute_ai_move("e2e4", model, data, controller, planner, square_to_piece, board)
        self._execute_ai_move("d7d5", model, data, controller, planner, square_to_piece, board)

        # White captures black d5 pawn
        self._execute_ai_move("e4d5", model, data, controller, planner, square_to_piece, board)

        # The captured piece was b_pawn_4
        assert "d7" not in square_to_piece
        assert square_to_piece["d5"] == "w_pawn_5"

        # Black graveyard count should be 1
        assert controller._black_graveyard_count == 1

        # b_pawn_4 should be at black graveyard origin
        captured_pos = data.body("b_pawn_4").xpos.copy()
        np.testing.assert_allclose(captured_pos[:2], BLACK_GRAVEYARD_ORIGIN[:2], atol=0.01)

    def test_multi_capture_sequence_increments_graveyard_counter(self):
        """
        Verify that multiple captures correctly increment graveyard counters.
        """
        model, data, _, controller, planner, square_to_piece, board = self._build_test_env()

        moves = [
            "e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d2", "c1d2"
        ]
        for uci in moves:
            self._execute_ai_move(uci, model, data, controller, planner, square_to_piece, board)

        assert controller._black_graveyard_count == 2
        assert controller._white_graveyard_count == 2

        gy_w = np.array(WHITE_GRAVEYARD_ORIGIN) + np.array([0.05, 0.0, 0.0])
        w_pawn_pos = data.body("w_pawn_4").xpos.copy()
        np.testing.assert_allclose(w_pawn_pos[:2], gy_w[:2], atol=0.01)

    def test_stall_detector_fails_on_unreachable_target(self):
        """
        Verify that the stall detector aborts operations for unreachable targets.
        """
        model, data, env, controller, _, _, _ = self._build_test_env()

        # Try reaching a target 10 meters away
        goal = np.array([10.0, 10.0, 1.0])
        env.set_target("w_pawn_1", data.body("w_pawn_1").xpos.copy())
        success, steps = controller._move_gripper_to(goal, use_rl=True)

        assert success is False
        assert steps > 50
        # It should hit the stall limit of 50 steps on the distant waypoint:
        assert steps == 78
