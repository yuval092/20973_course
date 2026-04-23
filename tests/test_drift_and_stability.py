"""
Tests for piece drift and robot arm stability during operations.

This module verifies that the robot arm can perform operations without
unintentionally disturbing other pieces on the board and that it handles
out-of-workspace or unreachable targets gracefully.
"""

import chess
import mujoco
import numpy as np
from src.config import N_SUBSTEPS, SCENE_XML, Z_GRASP
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.bootstrap import SystemBootstrapper
from src.game_runtime import GameOrchestrator
from src.logic.operation_planner import OperationPlanner, PickPlaceOp


class TestDriftAndStability:
    """
    Test suite for monitoring drift and stability during arm operations.
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

    def test_out_of_policy_workspace_logs_warning_but_succeeds(self, caplog):
        """
        Verify that moves outside the policy workspace log a warning but still succeed via teleportation.
        """
        (model, data, _, controller, _, square_to_piece, _) = self._build_test_env()

        # Mock a4 to be very far away to trigger the radius warning
        def mock_get_pos(sq):
            if sq == "a4":
                return np.array([0.9, 1.18, Z_GRASP])
            return controller.get_square_pos(sq)
        controller.get_pos = mock_get_pos

        op = PickPlaceOp("a2", "a4", "w_pawn_1")

        with caplog.at_level("WARNING"):
            # Use GameOrchestrator to trigger the teleportation fallback logic
            GameOrchestrator.execute_ops(model, data, [op], controller, square_to_piece)

        # Should log a warning about falling back to teleportation
        assert any("outside policy workspace" in rec.message for rec in caplog.records)
        # Physical position should be updated (teleported)
        actual_pos = data.body("w_pawn_1").xpos.copy()
        expected_pos = controller.get_pos("a4")
        np.testing.assert_allclose(actual_pos[:2], expected_pos[:2], atol=0.01)

    def test_extreme_distance_operation_fails_gracefully(self, caplog):
        """
        Verify that unreachable operations fail gracefully without crashing.
        """
        (_, _, _, controller, _, _, _) = self._build_test_env()

        # Intentionally target a location way out of bounds (10, 10)
        op = PickPlaceOp("a2", "a4", "w_pawn_1")

        def fake_get_square_pos(sq):
            if sq == "a4":
                return np.array([10.0, 10.0, Z_GRASP])
            return np.array([1.18, 0.75, Z_GRASP])

        controller.get_square_pos = fake_get_square_pos

        result = controller.execute_op(op)
        # The stall detector should eventually kick in and abort the operation early.
        assert result.success is False
        assert "UNREACHABLE" in result.details or "FAILED" in result.details

    def test_arm_transit_near_piece_does_not_knock_it_over(self):
        """
        Verify that moving the arm near pieces does not cause them to drift or fall.
        """
        (_, data, env, controller, _, _, _) = self._build_test_env()

        # d2 is directly in front of white king/queen
        # We move arm back and forth directly over d2 to see if it hits anything
        transit_z = Z_GRASP + 0.15
        env.set_target("w_pawn_4", data.body("w_pawn_4").xpos.copy())
        for waypoint in [
            np.array([1.18, 0.75, transit_z]),
            np.array([1.18, 0.85, transit_z]),
            np.array([1.18, 0.75, transit_z])
        ]:
            controller._move_gripper_to(waypoint, use_rl=False)

        # Pieces should not have drifted
        d2_pawn_pos = data.body("w_pawn_4").xpos.copy()
        expected_d2 = controller.get_square_pos("d2")

        assert np.linalg.norm(d2_pawn_pos[:2] - expected_d2[:2]) < 0.01
