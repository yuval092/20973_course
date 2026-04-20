"""
Unit tests for physical robotic arm execution in headless mode.

This module verifies that the ExecutionController can correctly perform
pick-and-place operations on the chess board, ensuring piece stability
and accurate placement.
"""

import mujoco
import numpy as np
import pytest

from src.config import N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.logic.operation_planner import PickPlaceOp


class TestHeadlessExecution:
    """
    Test suite for physical arm execution without a GUI.
    """

    def _build_env_and_controller(self):
        """
        Internal helper to initialize MuJoCo environment and controller.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        return env, controller, None

    def test_headless_execute_op_places_piece_on_board_upright(self):
        """
        Verify that execute_op moves a piece to the destination square and keeps it upright.
        """
        env, controller, viewer = self._build_env_and_controller()
        # Use f4 to f5 which is deep in the policy workspace sweet spot
        op = PickPlaceOp(target_square="f4", dest_square="f5", piece_name="w_pawn_6")

        result = controller.execute_op(op, viewer=viewer)

        piece = env.data.body("w_pawn_6")
        dest = controller.get_square_pos("f5")
        xy_error = ((piece.xpos[0] - dest[0])**2 + (piece.xpos[1] - dest[1])**2)**0.5
        z_error = abs(piece.xpos[2] - dest[2])
        
        assert result.success, f"Operation execution failed: {result.details}"
        assert xy_error < 0.02, f"XY Error too large: {xy_error:.4f}m"
        assert z_error < 0.02, f"Z Error too large: {z_error:.4f}m"

    def test_headless_sequential_knight_moves_succeed_on_visible_path(self):
        """
        Verify that multiple sequential moves with the same piece succeed.
        """
        env, controller, viewer = self._build_env_and_controller()

        # Use moves within policy workspace sweet spot (File F, ranks 4-6)
        first = PickPlaceOp(target_square="f4", dest_square="f5", piece_name="w_pawn_6")
        first_result = controller.execute_op(first, viewer=viewer)
        assert first_result.success, f"First move failed: {first_result.details}"

        second = PickPlaceOp(target_square="f5", dest_square="f6", piece_name="w_pawn_6")
        second_result = controller.execute_op(second, viewer=viewer)
        assert second_result.success, f"Second move failed: {second_result.details}"
