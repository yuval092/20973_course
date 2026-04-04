import chess
import mujoco
import numpy as np
import pytest

from src.config import N_SUBSTEPS, SCENE_XML, TABLE_HEIGHT, Z_GRASP
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.game_runtime import execute_ai_ops, initialize_square_to_piece
from src.logic.operation_planner import OperationPlanner, PickPlaceOp

def _build_test_env():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    planner = OperationPlanner()
    square_to_piece = initialize_square_to_piece(model, data)
    board = chess.Board()
    return model, data, env, controller, planner, square_to_piece, board

def test_out_of_policy_workspace_logs_warning_but_succeeds(caplog):
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
    # White a2 to a4 is very far left, usually out of policy radius
    op = PickPlaceOp("a2", "a4", "w_pawn_1")
    
    with caplog.at_level("WARNING"):
        result = controller.execute_op(op)
        
    assert result.success
    # Should flag out of workspace
    assert any("OUT_OF_POLICY_WORKSPACE" in rec.message for rec in caplog.records)

def test_extreme_distance_operation_fails_gracefully(caplog):
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
    # Intentionally target a location way out of bounds (10, 10)
    op = PickPlaceOp("a2", "a4", "w_pawn_1")
    controller.get_square_pos = lambda sq: np.array([10.0, 10.0, Z_GRASP]) if sq == "a4" else np.array([1.18, 0.75, Z_GRASP])
    
    result = controller.execute_op(op)
    # The stall detector should eventually kick in and abort the operation early.
    assert result.success is False
    assert "UNREACHABLE" in result.details or "FAILED" in result.details

def test_arm_transit_near_piece_does_not_knock_it_over():
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
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
