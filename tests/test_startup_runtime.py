from contextlib import contextmanager
from unittest.mock import MagicMock

import chess
import mujoco
import numpy as np
import pytest

import src.game_runtime as game_runtime
from src.game_runtime import (
    GameSystems,
    bootstrap_game_systems,
    initialize_square_to_piece,
    load_rl_policy,
    load_scene,
    print_game_banner,
    print_game_over,
    sync_viewer,
    validate_initial_mapping,
)
from src.config import N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import AIEngineError, BoardStateError, ExecutionError, RLModelError, SceneLoadError
from src.health_checks import build_default_check_registry
from src.logic.chess_manager import ChessGameManager
from src.logic.operation_planner import OperationPlanner


def _build_local_systems(manager):
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    planner = OperationPlanner()
    square_to_piece = initialize_square_to_piece(model, data)
    return GameSystems(
        mj_model=model,
        mj_data=data,
        env=env,
        rl_model=None,
        manager=manager,
        planner=planner,
        controller=controller,
        square_to_piece=square_to_piece,
        arm_home_grip=env.get_grip_pos().copy(),
        check_registry=build_default_check_registry(),
    )


def test_load_scene_successfully_builds_model_and_data():
    model, data = load_scene(SCENE_XML)
    assert model.nbody > 0
    assert data.qpos.shape[0] == model.nq


def test_load_scene_missing_file_raises_scene_load_error():
    with pytest.raises(SceneLoadError, match="Scene not found"):
        load_scene("/tmp/definitely_missing_scene.xml")


def test_load_rl_policy_wraps_hub_failure(monkeypatch):
    env = MagicMock()
    monkeypatch.setattr(game_runtime, "load_from_hub", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hub down")))

    with pytest.raises(RLModelError, match="hub down"):
        load_rl_policy(env, repo_id="fake/repo", filename="fake.zip")


def test_validate_initial_mapping_rejects_wrong_piece_count():
    with pytest.raises(BoardStateError, match="Expected 32 pieces"):
        validate_initial_mapping({"e2": "w_pawn_5"})


def test_bootstrap_game_systems_success_with_mocked_policy(monkeypatch):
    fake_policy = object()
    monkeypatch.setattr(game_runtime, "load_rl_policy", lambda env, repo_id, filename: fake_policy)

    # bootstrap_game_systems already calls initialize_square_to_piece
    systems = bootstrap_game_systems()
    print(f"TEST DEBUG: b_rook_1 xml pos = {systems.mj_model.body('b_rook_1').pos}")

    try:
        assert systems.rl_model is fake_policy
        assert len(systems.square_to_piece) == 32
        assert systems.square_to_piece["e2"] == "w_pawn_5"
    finally:
        systems.manager.close()


def test_bootstrap_game_systems_propagates_model_load_failure(monkeypatch):
    monkeypatch.setattr(game_runtime, "load_rl_policy", lambda env, repo_id, filename: (_ for _ in ()).throw(RLModelError("bad model")))
    monkeypatch.setattr(game_runtime, "ChessGameManager", MagicMock())

    with pytest.raises(RLModelError, match="bad model"):
        bootstrap_game_systems()




def test_sync_viewer_updates_forward_and_sync(manager):
    systems = _build_local_systems(manager)
    viewer = MagicMock()

    sync_viewer(viewer, systems.mj_model, systems.mj_data)

    viewer.sync.assert_called_once()


def test_print_game_banner_contains_expected_instructions(capsys):
    print_game_banner()
    out = capsys.readouterr().out
    assert "ROBOCHESS" in out
    assert "Enter moves in UCI format" in out
    assert "quit, exit" in out


def test_print_game_banner_includes_hint(capsys):
    print_game_banner()
    out = capsys.readouterr().out
    assert "hint, suggest" in out


def test_print_game_over_checkmate_reports_winner(capsys):
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(uci)

    print_game_over(board)
    out = capsys.readouterr().out
    assert "Checkmate!" in out
    assert "Black wins." in out


def test_print_game_over_stalemate_reports_draw(capsys):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    print_game_over(board)
    out = capsys.readouterr().out
    assert "Stalemate" in out
    
def test_print_game_over_fifty_move_draw(capsys):
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 50")
    print_game_over(board)
    out = capsys.readouterr().out
    assert "50-move rule" in out

def test_print_game_over_threefold_draw(capsys):
    board = chess.Board()
    for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(m)
    print_game_over(board)
    out = capsys.readouterr().out
    assert "threefold repetition" in out



