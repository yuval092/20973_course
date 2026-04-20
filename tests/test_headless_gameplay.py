"""
End-to-end headless gameplay and system integration tests.

This module verifies the integration between the chess manager, operation planner,
execution controller, and the Mujoco physics environment. It covers full game turns,
special moves (castling, promotion, capture), and board state validation.
"""

import re
import chess
import mujoco
import numpy as np
import pytest

from scripts.generate_xml import XMLGenerator
from src.config import (
    BLACK_GRAVEYARD_ORIGIN, N_SUBSTEPS, SCENE_XML, 
    SQUARE_SIZE, WHITE_GRAVEYARD_ORIGIN, Z_GRASP, PLACEMENT_TOLERANCE
)
from src.control.execution_controller import ExecutionController, OpResult
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import (
    BoardStateError, ExecutionError, PieceLookupError, StabilityError
)
from src.bootstrap import SystemBootstrapper
from src.game_runtime import GameOrchestrator
from src.logic.operation_planner import OperationPlanner, PickPlaceOp
from src.runtime_guard import validate_board_state, RuntimeGuard


class TestHeadlessGameplay:
    """
    Tests for end-to-end gameplay logic and multi-turn sequences.
    """

    def _build_game(self, manager):
        """
        Helper to initialize a game session.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        planner = OperationPlanner()
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)

        game = {
            "model": model,
            "data": data,
            "env": env,
            "controller": controller,
            "manager": manager,
            "planner": planner,
            "square_to_piece": square_to_piece,
            "home_grip": env.get_grip_pos().copy(),
            "check_registry": None,
            "arm_home_grip": env.get_grip_pos().copy(),
        }
        validate_board_state(model, data, manager.board, square_to_piece, controller)
        return game

    def _play_turn(self, game, uci, physical_black=False):
        """
        Helper to play a turn and update the state.
        """
        manager = game["manager"]
        planner = game["planner"]
        controller = game["controller"]
        square_to_piece = game["square_to_piece"]

        assert manager.validate_move(uci), f"Illegal scripted test move: {uci}"
        move = chess.Move.from_uci(uci)
        ops = planner.generate_operations(move, manager.board, square_to_piece)

        if manager.board.turn == chess.BLACK and physical_black:
            GameOrchestrator.execute_ops(
                game["model"],
                game["data"],
                ops,
                controller,
                square_to_piece,
                viewer=None,
                physical_arm=True,
            )
        else:
            GameOrchestrator.execute_ops(
                game["model"],
                game["data"],
                ops,
                controller,
                square_to_piece,
                viewer=None,
                physical_arm=False,
            )

        manager.push_move(uci)
        validate_board_state(
            game["model"],
            game["data"],
            manager.board,
            square_to_piece,
            controller,
        )
        return ops

    def _assert_arm_home(self, game, atol=0.01):
        """
        Assert the robot arm has returned to its home position.
        """
        np.testing.assert_allclose(
            game["env"].get_grip_pos(),
            game["home_grip"],
            atol=atol,
        )

    def _assert_piece_on_square(self, game, square_name, expected_piece_name):
        """
        Assert that a piece is physically located at the expected square.
        """
        assert game["square_to_piece"].get(square_name) == expected_piece_name
        expected_pos = game["controller"].get_square_pos(square_name)
        actual_pos = game["data"].body(expected_piece_name).xpos.copy()
        xy_error = np.linalg.norm(actual_pos[:2] - expected_pos[:2])
        z_error = abs(actual_pos[2] - expected_pos[2])
        assert xy_error < 0.02
        assert z_error < 0.02

    def test_headless_initial_board_is_valid_and_arm_is_home(self, manager):
        """
        Verify the starting state of the board and arm.
        """
        game = self._build_game(manager)
        assert len(game["square_to_piece"]) == 32
        validate_board_state(
            game["model"],
            game["data"],
            game["manager"].board,
            game["square_to_piece"],
            game["controller"],
        )
        self._assert_arm_home(game)

    def test_headless_human_e2e4_updates_board_scene_and_mapping(self, manager):
        """
        Verify a standard human move updates the mapping and physical scene.
        """
        game = self._build_game(manager)
        self._play_turn(game, "e2e4")

        assert "e2" not in game["square_to_piece"]
        self._assert_piece_on_square(game, "e4", "w_pawn_5")
        self._assert_arm_home(game)

    def test_headless_ai_f6f5_logs_success(self, manager, caplog):
        """
        Verify an AI move succeeds and logs the expected output.
        """
        game = self._build_game(manager)
        # Setup board for a move in the sweet spot (f6 to f5)
        game["manager"].board.set_fen("rnbqkbnr/ppppp1pp/5p2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        game["square_to_piece"].pop("f7")
        game["square_to_piece"]["f6"] = "b_pawn_6"

        with caplog.at_level("INFO", logger="robo_chess"):
            self._play_turn(game, "f6f5", physical_black=True)

        assert any("Result: SUCCESS" in rec.message for rec in caplog.records)
        self._assert_piece_on_square(game, "f5", "b_pawn_6")
        self._assert_arm_home(game)

    def test_headless_multi_turn_opening_sequence_one_stays_valid_each_turn(self, manager):
        """
        Test a sequence of four moves with interleaved human/AI play.
        """
        game = self._build_game(manager)

        sequence = [
            ("e2e4", False),
            ("e7e5", True),
            ("g1f3", False),
            ("d7d6", True),
        ]
        for uci, physical_black in sequence:
            self._play_turn(game, uci, physical_black=physical_black)
            validate_board_state(
                game["model"],
                game["data"],
                game["manager"].board,
                game["square_to_piece"],
                game["controller"],
            )
            self._assert_arm_home(game)

        self._assert_piece_on_square(game, "e4", "w_pawn_5")
        self._assert_piece_on_square(game, "f3", "w_knight_2")
        self._assert_piece_on_square(game, "e5", "b_pawn_5")
        self._assert_piece_on_square(game, "d6", "b_pawn_4")


class TestHeadlessMoveTypes:
    """
    Tests for specific chess move types (captures, castling, promotion).
    """

    def _build_game(self, manager):
        return TestHeadlessGameplay()._build_game(manager)

    def _play_turn(self, game, uci, physical_black=False):
        return TestHeadlessGameplay()._play_turn(game, uci, physical_black)

    def _assert_piece_on_square(self, game, square_name, expected_piece_name):
        return TestHeadlessGameplay()._assert_piece_on_square(game, square_name, expected_piece_name)

    def _assert_arm_home(self, game):
        return TestHeadlessGameplay()._assert_arm_home(game)

    def test_headless_human_capture_sends_captured_piece_to_black_graveyard(self, manager):
        """
        Verify that captured pieces are moved to the correct graveyard area.
        """
        game = self._build_game(manager)

        self._play_turn(game, "e2e4")
        self._play_turn(game, "d7d5")
        self._play_turn(game, "e4d5")

        self._assert_piece_on_square(game, "d5", "w_pawn_5")
        captured_pos = game["data"].body("b_pawn_4").xpos.copy()
        
        # b_pawn_4 is the first black piece captured in this test
        expected_pos = GameOrchestrator._get_graveyard_grid_pos(BLACK_GRAVEYARD_ORIGIN, 0)
        np.testing.assert_allclose(captured_pos, expected_pos, atol=0.01)
        assert "d7" not in game["square_to_piece"]

    def test_headless_human_castling_moves_king_and_rook_to_expected_squares(self, manager):
        """
        Verify that castling correctly moves both the king and the rook.
        """
        game = self._build_game(manager)

        sequence = ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1"]
        for uci in sequence:
            self._play_turn(game, uci)

        self._assert_piece_on_square(game, "g1", "w_king")
        self._assert_piece_on_square(game, "f1", "w_rook_2")
        assert "e1" not in game["square_to_piece"]
        assert "h1" not in game["square_to_piece"]
        self._assert_arm_home(game)

    def test_headless_human_promotion_swaps_piece(self, manager):
        """
        Verify that pawn promotion replaces the pawn with the selected piece.
        """
        game = self._build_game(manager)
        board = game["manager"].board
        
        # Setup promotion scenario
        board.set_fen("8/P7/8/8/8/8/8/k6K w - - 0 1")
        game["square_to_piece"].clear()
        game["square_to_piece"]["a7"] = "w_pawn_1"
        game["square_to_piece"]["a1"] = "b_king"
        game["square_to_piece"]["h1"] = "w_king"
        
        gy_pos = [1.5, 0.5, 0.4]
        for body_id in range(game["model"].nbody):
            name = mujoco.mj_id2name(game["model"], mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith(("w_", "b_")) and \
               name not in {"w_pawn_1", "b_king", "w_king", "w_spare_queen_1"}:
                GameOrchestrator.teleport_piece(game["model"], game["data"], name, gy_pos)
                
        GameOrchestrator.teleport_piece(game["model"], game["data"], "w_pawn_1", 
                       game["controller"].get_square_pos("a7"))
        GameOrchestrator.teleport_piece(game["model"], game["data"], "b_king", 
                       game["controller"].get_square_pos("a1"))
        GameOrchestrator.teleport_piece(game["model"], game["data"], "w_king", 
                       game["controller"].get_square_pos("h1"))
        
        mujoco.mj_forward(game["model"], game["data"])
        
        self._play_turn(game, "a7a8q")
        
        assert "a7" not in game["square_to_piece"]
        assert game["square_to_piece"]["a8"] == "w_spare_queen_1"


class TestHeadlessValidation:
    """
    Tests for system validation, error handling, and stability checks.
    """

    def _build_game(self, manager):
        return TestHeadlessGameplay()._build_game(manager)

    def _play_turn(self, game, uci, physical_black=False):
        return TestHeadlessGameplay()._play_turn(game, uci, physical_black)

    def _set_piece_tilt(self, game, piece_name, quat):
        """
        Helper to artificially tilt a piece.
        """
        body = game["model"].body(piece_name)
        joint_id = body.jntadr[0]
        game["data"].joint(joint_id).qpos[3:7] = quat
        mujoco.mj_forward(game["model"], game["data"])

    def test_headless_nudged_piece_raises_board_state_error(self, manager):
        """
        Verify that moving a piece manually triggers a validation error.
        """
        game = self._build_game(manager)
        # Use f4 which is within policy workspace
        game["manager"].board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        game["square_to_piece"].clear()
        # Minimal mapping for validation test
        for sq in ["a1", "a2", "a7", "a8", "e4", "h1", "h8"]:
             game["square_to_piece"][sq] = "dummy"
        
        game["square_to_piece"]["e4"] = "w_pawn_5"
        game["square_to_piece"]["h1"] = "w_king"
        game["square_to_piece"]["a1"] = "b_king"
        
        # Reset board to match mapping
        game["manager"].board.set_fen("k7/8/8/8/4P3/8/8/7K w - - 0 1")
        game["square_to_piece"] = {
            "a8": "b_king",
            "e4": "w_pawn_5",
            "h1": "w_king"
        }

        # Nudge piece
        joint_id = game["model"].body("w_pawn_5").jntadr[0]
        game["data"].joint(joint_id).qpos[0] += 0.05
        mujoco.mj_forward(game["model"], game["data"])

        with pytest.raises(BoardStateError, match="coordinate mismatch"):
            RuntimeGuard.validate_board_state(
                game["model"], game["data"], game["manager"].board, 
                game["square_to_piece"], game["controller"]
            )

    def test_headless_tilted_piece_raises_stability_error(self, manager):
        """
        Verify that a tilted piece triggers a stability validation error.
        """
        game = self._build_game(manager)
        game["manager"].board.set_fen("k7/8/8/8/4P3/8/8/7K w - - 0 1")
        game["square_to_piece"] = {
            "a8": "b_king",
            "e4": "w_pawn_5",
            "h1": "w_king"
        }

        self._set_piece_tilt(game, "w_pawn_5", np.array([0.923, 0.382, 0.0, 0.0]))

        with pytest.raises(StabilityError, match="tilted"):
             RuntimeGuard.validate_board_state(
                game["model"], game["data"], game["manager"].board, 
                game["square_to_piece"], game["controller"]
            )

    def test_headless_invalid_piece_teleport_raises_piece_lookup_error(self, manager):
        """
        Verify that attempting to teleport a non-existent piece raises an error.
        """
        game = self._build_game(manager)

        with pytest.raises(PieceLookupError, match="not found"):
            GameOrchestrator.teleport_piece(
                game["model"],
                game["data"],
                "not_a_real_piece",
                np.array([1.0, 1.0, Z_GRASP]),
            )

    def test_headless_corrupted_square_mapping_raises_board_state_error(self, manager):
        """
        Verify that an inconsistent mapping triggers a validation error.
        """
        game = self._build_game(manager)
        game["square_to_piece"]["e4"] = "w_pawn_4"

        with pytest.raises(BoardStateError):
            RuntimeGuard.validate_board_state(
                game["model"],
                game["data"],
                game["manager"].board,
                game["square_to_piece"],
                game["controller"],
            )

    def test_headless_execute_ai_ops_raises_on_unsuccessful_result(self, manager, monkeypatch):
        """
        Verify that AI operation failure propagates as an ExecutionError.
        """
        game = self._build_game(manager)
        # Use f4f5 which is in the sweet spot
        ops = [PickPlaceOp("f4", "f5", "w_pawn_6")]

        def fake_execute_op(op, viewer=None):
            return OpResult(False, 12, float("inf"), 3, "FAILED: synthetic test failure")

        monkeypatch.setattr(game["controller"], "execute_op", fake_execute_op)

        with pytest.raises(ExecutionError, match="synthetic test failure"):
            GameOrchestrator.execute_ops(
                game["model"],
                game["data"],
                ops,
                game["controller"],
                game["square_to_piece"],
                viewer=None,
                physical_arm=True
            )

    def test_headless_teleport_piece_success_path(self, manager):
        """
        Verify the successful teleportation of a piece.
        """
        game = self._build_game(manager)
        GameOrchestrator.teleport_piece(game["model"], game["data"], "w_pawn_1", [1.0, 1.0, 1.5])
        mujoco.mj_forward(game["model"], game["data"])
        pos = game["data"].body("w_pawn_1").xpos
        np.testing.assert_allclose(pos, [1.0, 1.0, 1.5], atol=0.01)


class TestXMLGeneration:
    """
    Tests for the dynamic generation of the Mujoco XML world.
    """

    def test_generated_xml_contains_rank_and_file_labels(self):
        """
        Verify that the generated XML includes expected labels and meshes.
        """
        xml = XMLGenerator().generate()

        assert "label_file_a_" in xml
        assert "label_file_h_" in xml
        assert "label_rank_1_" in xml
        assert "label_rank_8_" in xml
        assert 'mesh="chess_pawn_mesh"' in xml
        assert '<body name="w_pawn_1"' in xml

    def test_generated_xml_file_labels_read_a_to_h_on_near_edge(self):
        """
        Verify spatial ordering of file labels.
        """
        xml = XMLGenerator().generate()

        def _x_for(label_prefix):
            pattern = rf'{label_prefix}_0" type="box" size="[^"]+" pos="([0-9.]+) '
            match = re.search(pattern, xml)
            assert match is not None, label_prefix
            return float(match.group(1))

        far_a_x = _x_for("label_file_a")
        far_h_x = _x_for("label_file_h")
        near_a_x = _x_for("label_file_near_a")
        near_h_x = _x_for("label_file_near_h")

        assert far_a_x > far_h_x
        assert near_a_x > near_h_x

    def test_generated_xml_board_underlay_sits_below_playable_squares(self):
        """
        Verify vertical ordering of board components.
        """
        xml = XMLGenerator().generate()

        underlay_pat = r'geom name="board_underlay".*pos="[0-9.]+ [0-9.]+ ([0-9.]+)"'
        square_pat = r'geom name="square_0_0".*pos="[0-9.]+ [0-9.]+ ([0-9.]+)"'
        
        underlay_match = re.search(underlay_pat, xml)
        square_match = re.search(square_pat, xml)

        assert underlay_match is not None
        assert square_match is not None
        assert float(underlay_match.group(1)) < float(square_match.group(1))
