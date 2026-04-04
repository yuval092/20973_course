import chess
import mujoco
import numpy as np
import pytest

from src.config import N_SUBSTEPS, SCENE_XML, WHITE_GRAVEYARD_ORIGIN, BLACK_GRAVEYARD_ORIGIN
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.game_runtime import execute_ai_ops, initialize_square_to_piece
from src.logic.operation_planner import OperationPlanner

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

def _execute_ai_move(uci, model, data, controller, planner, square_to_piece, board):
    move = chess.Move.from_uci(uci)
    ops = planner.generate_operations(move, board, square_to_piece)
    execute_ai_ops(ops, controller, square_to_piece, model, data, viewer=None)
    board.push(move)

def test_headless_ai_capture_to_graveyard_succeeds():
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
    # Execute opening
    _execute_ai_move("e2e4", model, data, controller, planner, square_to_piece, board)
    _execute_ai_move("d7d5", model, data, controller, planner, square_to_piece, board)
    
    # White captures black d5 pawn
    _execute_ai_move("e4d5", model, data, controller, planner, square_to_piece, board)
    
    # The captured piece was b_pawn_4
    assert "d7" not in square_to_piece
    assert square_to_piece["d5"] == "w_pawn_5"
    
    # Black graveyard count should be 1
    assert controller._black_graveyard_count == 1
    
    # b_pawn_4 should be at black graveyard origin
    captured_pos = data.body("b_pawn_4").xpos.copy()
    np.testing.assert_allclose(captured_pos[:2], BLACK_GRAVEYARD_ORIGIN[:2], atol=0.01)
    
    
def test_multi_capture_sequence_increments_graveyard_counter():
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
    moves = [
        "e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d2", "c1d2"
    ]
    for uci in moves:
        _execute_ai_move(uci, model, data, controller, planner, square_to_piece, board)
        
    assert controller._black_graveyard_count == 2
    assert controller._white_graveyard_count == 2
    
    gy_w = np.array(WHITE_GRAVEYARD_ORIGIN) + np.array([0.05, 0.0, 0.0])
    w_pawn_pos = data.body("w_pawn_4").xpos.copy()
    np.testing.assert_allclose(w_pawn_pos[:2], gy_w[:2], atol=0.01)

def test_stall_detector_fails_on_unreachable_target():
    model, data, env, controller, planner, square_to_piece, board = _build_test_env()
    
    # Try reaching a target 10 meters away
    # the stall detector should abort early because it makes zero progress physically
    # (since the arm cannot reach 10m)
    
    goal = np.array([10.0, 10.0, 1.0])
    env.set_target("w_pawn_1", data.body("w_pawn_1").xpos.copy())
    success, steps = controller._move_gripper_to(goal, use_rl=True)
    
    assert success is False
    assert steps > 50
    # It should hit the stall limit of 50 steps on the distant waypoint:
    assert steps == 78
    
