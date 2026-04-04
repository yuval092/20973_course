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
    complete_turn,
    initialize_square_to_piece,
    load_rl_policy,
    load_scene,
    print_game_banner,
    print_game_over,
    process_ai_turn,
    process_human_turn,
    run_turn,
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


def _build_local_systems():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    manager = ChessGameManager()
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

    systems = bootstrap_game_systems()

    assert systems.rl_model is fake_policy
    assert len(systems.square_to_piece) == 32
    assert systems.square_to_piece["e2"] == "w_pawn_5"


def test_bootstrap_game_systems_propagates_model_load_failure(monkeypatch):
    monkeypatch.setattr(game_runtime, "load_rl_policy", lambda env, repo_id, filename: (_ for _ in ()).throw(RLModelError("bad model")))

    with pytest.raises(RLModelError, match="bad model"):
        bootstrap_game_systems()


def test_process_human_turn_quit_returns_false_and_does_not_move():
    systems = _build_local_systems()
    before = systems.manager.board.fen()

    assert process_human_turn(systems, input_func=lambda prompt: "quit") is False
    assert systems.manager.board.fen() == before


def test_process_human_turn_illegal_move_returns_true_without_mutation(capsys):
    systems = _build_local_systems()
    before = systems.manager.board.fen()

    assert process_human_turn(systems, input_func=lambda prompt: "e2e5") is True
    assert systems.manager.board.fen() == before
    assert "Illegal move" in capsys.readouterr().out


def test_process_human_turn_legal_move_updates_board_and_mapping():
    systems = _build_local_systems()

    assert process_human_turn(systems, input_func=lambda prompt: "e2e4") is True
    assert systems.manager.board.piece_at(chess.E4) is not None
    assert systems.square_to_piece["e4"] == "w_pawn_5"
    assert "e2" not in systems.square_to_piece


def test_process_ai_turn_raises_when_engine_returns_none(monkeypatch):
    systems = _build_local_systems()
    systems.manager.push_move("e2e4")
    monkeypatch.setattr(systems.manager, "get_ai_move", lambda: None)

    with pytest.raises(AIEngineError, match="AI returned no move"):
        process_ai_turn(systems, viewer=None)


def test_process_ai_turn_executes_move_and_updates_board(monkeypatch):
    systems = _build_local_systems()
    systems.manager.push_move("e2e4")
    move = chess.Move.from_uci("e7e5")
    monkeypatch.setattr(systems.manager, "get_ai_move", lambda: move)

    called = {}

    def fake_execute_ai_ops(ops, controller, square_to_piece, mj_model, mj_data, viewer):
        called["ops"] = ops
        square_to_piece["e5"] = square_to_piece.pop("e7")

    monkeypatch.setattr(game_runtime, "execute_ai_ops", fake_execute_ai_ops)

    assert process_ai_turn(systems, viewer=None) is True
    assert systems.manager.board.piece_at(chess.E5) is not None
    assert systems.square_to_piece["e5"] == "b_pawn_5"
    assert called["ops"][0].dest_square == "e5"


def test_complete_turn_runs_post_move_validation(monkeypatch):
    systems = _build_local_systems()
    calls = {"checked": 0}
    systems.check_registry.run = lambda hook, context: calls.__setitem__("checked", calls["checked"] + 1)
    monkeypatch.setattr(game_runtime.time, "sleep", lambda _sec: None)

    complete_turn(systems, viewer=None, sleep_sec=0.0)

    assert calls["checked"] == 1


def test_run_turn_human_path_calls_complete_turn(monkeypatch):
    systems = _build_local_systems()
    monkeypatch.setattr(game_runtime, "process_human_turn", lambda systems, input_func=input: True)
    calls = {"complete": 0}
    monkeypatch.setattr(game_runtime, "complete_turn", lambda systems, viewer=None, sleep_sec=0.5: calls.__setitem__("complete", calls["complete"] + 1))
    systems.check_registry.run = lambda hook, context: None

    assert run_turn(systems, viewer=None, sleep_sec=0.0) is True
    assert calls["complete"] == 1


def test_run_turn_ai_path_calls_complete_turn(monkeypatch):
    systems = _build_local_systems()
    systems.manager.push_move("e2e4")
    monkeypatch.setattr(game_runtime, "process_ai_turn", lambda systems, viewer=None: True)
    calls = {"complete": 0}
    monkeypatch.setattr(game_runtime, "complete_turn", lambda systems, viewer=None, sleep_sec=0.5: calls.__setitem__("complete", calls["complete"] + 1))
    systems.check_registry.run = lambda hook, context: None

    assert run_turn(systems, viewer=None, sleep_sec=0.0) is True
    assert calls["complete"] == 1


def test_sync_viewer_updates_forward_and_sync():
    systems = _build_local_systems()
    viewer = MagicMock()

    sync_viewer(viewer, systems.mj_model, systems.mj_data)

    viewer.sync.assert_called_once()


def test_print_game_banner_contains_expected_instructions(capsys):
    print_game_banner()
    out = capsys.readouterr().out
    assert "ROBOCHESS" in out
    assert "Enter moves in UCI format" in out
    assert "quit, exit" in out


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


def test_main_starts_viewer_and_quits_cleanly(monkeypatch, capsys):
    systems = _build_local_systems()
    monkeypatch.setattr(game_runtime, "bootstrap_game_systems", lambda: systems)
    monkeypatch.setattr(game_runtime.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(game_runtime, "run_turn", lambda systems, viewer=None: False)

    class FakeViewer:
        def __init__(self):
            self.sync_calls = 0

        def is_running(self):
            return True

        def sync(self):
            self.sync_calls += 1

    @contextmanager
    def fake_launch_passive(model, data):
        yield FakeViewer()

    monkeypatch.setattr(game_runtime.mujoco.viewer, "launch_passive", fake_launch_passive)

    game_runtime.main()
    out = capsys.readouterr().out
    assert "ROBOCHESS" in out
    assert "GAME OVER" in out


def test_main_freezes_on_turn_exception(monkeypatch):
    systems = _build_local_systems()
    monkeypatch.setattr(game_runtime, "bootstrap_game_systems", lambda: systems)
    monkeypatch.setattr(game_runtime.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(game_runtime, "run_turn", lambda systems, viewer=None: (_ for _ in ()).throw(RuntimeError("boom")))

    class FakeViewer:
        def is_running(self):
            return True

        def sync(self):
            pass

    @contextmanager
    def fake_launch_passive(model, data):
        yield FakeViewer()

    called = {"freeze": 0}

    def fake_freeze(exc, viewer=None, mj_model=None, mj_data=None):
        called["freeze"] += 1
        assert str(exc) == "boom"

    monkeypatch.setattr(game_runtime.mujoco.viewer, "launch_passive", fake_launch_passive)
    monkeypatch.setattr(game_runtime, "freeze_on_exception", fake_freeze)

    game_runtime.main()
    assert called["freeze"] == 1


def test_main_does_not_freeze_on_operational_execution_error(monkeypatch, capsys):
    systems = _build_local_systems()
    monkeypatch.setattr(game_runtime, "bootstrap_game_systems", lambda: systems)
    monkeypatch.setattr(game_runtime.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(
        game_runtime,
        "run_turn",
        lambda systems, viewer=None: (_ for _ in ()).throw(ExecutionError("out-of-policy capture")),
    )

    class FakeViewer:
        def is_running(self):
            return True

        def sync(self):
            pass

    @contextmanager
    def fake_launch_passive(model, data):
        yield FakeViewer()

    called = {"freeze": 0}

    def fake_freeze(exc, viewer=None, mj_model=None, mj_data=None):
        called["freeze"] += 1

    monkeypatch.setattr(game_runtime.mujoco.viewer, "launch_passive", fake_launch_passive)
    monkeypatch.setattr(game_runtime, "freeze_on_exception", fake_freeze)

    game_runtime.main()

    out = capsys.readouterr().out
    assert "GAME ABORTED" in out
    assert "out-of-policy capture" in out
    assert called["freeze"] == 0
