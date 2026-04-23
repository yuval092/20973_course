"""
Unit tests for system startup and game runtime initialization.

This module verifies the bootstrapping process, including scene loading,
RL policy loading, initial mapping validation, and utility functions for
displaying game information and results.
"""

from unittest.mock import MagicMock
import chess
import mujoco
import numpy as np
import pytest

from src.models import GameSystems
from src.bootstrap import SystemBootstrapper
from src.game_runtime import GameOrchestrator
from src.app import RoboChessApp
import src.bootstrap as bootstrap_module

from src.config import N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import (
    BoardStateError, RLModelError, SceneLoadError
)
from src.health_checks import build_default_check_registry
from src.logic.operation_planner import OperationPlanner


class TestStartupRuntime:
    """
    Test suite for system initialization and runtime utilities.
    """

    def _build_local_systems(self, manager):
        """
        Helper to initialize game systems for local testing.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        planner = OperationPlanner()
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)
        
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

    def test_load_scene_successfully_builds_model_and_data(self):
        """
        Verify that load_scene correctly initializes Mujoco model and data.
        """
        model, data = SystemBootstrapper.load_scene(SCENE_XML)
        assert model.nbody > 0
        assert data.qpos.shape[0] == model.nq

    def test_load_scene_missing_file_raises_scene_load_error(self):
        """
        Verify that load_scene raises an error if the XML file is missing.
        """
        with pytest.raises(SceneLoadError, match="Scene not found"):
            SystemBootstrapper.load_scene("/tmp/definitely_missing_scene.xml")

    def test_load_rl_policy_wraps_local_failure(self, monkeypatch):
        """
        Verify that load_rl_policy wraps downstream disk errors.
        """
        env = MagicMock()
        # Mock SAC.load to raise an error
        def mock_sac_fail(*args, **kwargs):
            raise RuntimeError("disk error")
        monkeypatch.setattr(bootstrap_module.SAC, "load", mock_sac_fail)

        with pytest.raises(RLModelError, match="disk error"):
            SystemBootstrapper.load_rl_policy(env, model_path="fake.zip")

    def test_validate_initial_mapping_rejects_wrong_piece_count(self):
        """
        Verify that validate_initial_mapping detects an incorrect piece count.
        """
        with pytest.raises(BoardStateError, match="Expected 32 pieces"):
            SystemBootstrapper.validate_initial_mapping({"e2": "w_pawn_5"})

    def test_bootstrap_game_systems_success_with_mocked_policy(self, monkeypatch):
        """
        Verify that bootstrap_game_systems correctly assembles all components.
        """
        fake_policy = object()
        monkeypatch.setattr(
            SystemBootstrapper, "load_rl_policy", 
            lambda *args, **kwargs: fake_policy
        )

        systems = SystemBootstrapper.bootstrap_game_systems()

        try:
            assert systems.rl_model is fake_policy
            assert len(systems.square_to_piece) == 32
            assert systems.square_to_piece["e2"] == "w_pawn_5"
        finally:
            systems.manager.close()

    def test_bootstrap_game_systems_propagates_model_load_failure(self, monkeypatch):
        """
        Verify that bootstrap_game_systems propagates RL model load errors.
        """
        def mock_load_fail(*args, **kwargs):
            raise RLModelError("bad model")
        monkeypatch.setattr(SystemBootstrapper, "load_rl_policy", mock_load_fail)

        with pytest.raises(RLModelError, match="bad model"):
            SystemBootstrapper.bootstrap_game_systems()

    def test_sync_viewer_updates_forward_and_sync(self, manager):
        """
        Verify that sync_viewer calls the viewer's sync method.
        """
        systems = self._build_local_systems(manager)
        viewer = MagicMock()

        GameOrchestrator.sync_viewer(viewer, systems.mj_model, systems.mj_data)

        viewer.sync.assert_called_once()

    def test_print_game_banner_contains_expected_instructions(self, capsys):
        """
        Verify that the game banner displays basic usage instructions.
        """
        app = RoboChessApp()
        app.print_game_banner()
        out = capsys.readouterr().out
        assert "ROBOCHESS" in out
        assert "Enter moves in UCI format" in out
        assert "quit, exit" in out
        assert "hint, suggest" in out

    def test_print_game_over_checkmate_reports_winner(self, capsys):
        """
        Verify the end-of-game report for a checkmate scenario.
        """
        board = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            board.push_uci(uci)

        app = RoboChessApp()
        app.print_game_over(board)
        out = capsys.readouterr().out
        assert "Checkmate!" in out
        assert "Black wins." in out

    def test_print_game_over_stalemate_reports_draw(self, capsys):
        """
        Verify the end-of-game report for a stalemate.
        """
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        app = RoboChessApp()
        app.print_game_over(board)
        out = capsys.readouterr().out
        assert "Stalemate" in out
        
    def test_print_game_over_fifty_move_draw(self, capsys):
        """
        Verify the end-of-game report for the fifty-move rule.
        """
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 50"
        )
        app = RoboChessApp()
        app.print_game_over(board)
        out = capsys.readouterr().out
        assert "50-move rule" in out

    def test_print_game_over_threefold_draw(self, capsys):
        """
        Verify the end-of-game report for threefold repetition.
        """
        board = chess.Board()
        # Repeat moves to trigger threefold repetition
        sequence = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
        for m in sequence:
            board.push_uci(m)
            
        app = RoboChessApp()
        app.print_game_over(board)
        out = capsys.readouterr().out
        assert "threefold repetition" in out
