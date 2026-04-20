"""
Tests for the BoardObserver class and board stability during simulation.

This module contains tests that verify the BoardObserver's ability to detect
unstable board states (e.g., tipped or displaced pieces) and ensures that
physics simulation and robot arm movements do not unintentionally destabilize
the board.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import mujoco
from gymnasium_robotics.utils import mujoco_utils
from src.observation.board_observer import BoardObserver
from src.exceptions import StabilityError
from src.config import TABLE_HEIGHT, BOARD_CENTER, SQUARE_SIZE, SCENE_XML, ACTION_SCALE
from src.env.chess_pick_place_env import ChessPickPlaceEnv


class TestBoardObserver:
    """
    Test suite for BoardObserver functionality and board stability.
    """

    def test_board_observer_stability_pass(self):
        """
        Verify that BoardObserver passes when pieces are in stable positions.
        """
        model = MagicMock()
        model.nbody = 2
        with patch('mujoco.mj_id2name') as mock_name2id:
            mock_name2id.side_effect = lambda m, t, i: "white_pawn_a2" if i == 1 else "world"

            data = MagicMock()
            mock_body = MagicMock()
            mock_body.xquat = np.array([1.0, 0.0, 0.0, 0.0])  # Upright
            pos_x = BOARD_CENTER[0] + 0.5 * SQUARE_SIZE
            pos_y = BOARD_CENTER[1] + 0.5 * SQUARE_SIZE
            mock_body.xpos = np.array([pos_x, pos_y, TABLE_HEIGHT + 0.01])
            data.body.side_effect = lambda i: mock_body if i == 1 else MagicMock()

            observer = BoardObserver(model, data)
            observer.verify_stability()

    def test_board_observer_tilt_fail(self):
        """
        Verify that BoardObserver raises StabilityError when a piece is tilted.
        """
        model = MagicMock()
        model.nbody = 2
        with patch('mujoco.mj_id2name') as mock_name2id:
            mock_name2id.side_effect = lambda m, t, i: "white_pawn_a2" if i == 1 else "world"

            data = MagicMock()
            mock_body = MagicMock()
            # Tipped over (45 degrees tilt roughly)
            mock_body.xquat = np.array([0.923, 0.382, 0.0, 0.0])
            mock_body.xpos = np.array([BOARD_CENTER[0], BOARD_CENTER[1], TABLE_HEIGHT + 0.01])
            data.body.side_effect = lambda i: mock_body if i == 1 else MagicMock()

            observer = BoardObserver(model, data)
            with pytest.raises(StabilityError, match="tipped over"):
                observer.verify_stability()

    def test_board_observer_displacement_fail(self):
        """
        Verify that BoardObserver raises StabilityError when a piece is displaced.
        """
        model = MagicMock()
        model.nbody = 2
        with patch('mujoco.mj_id2name') as mock_name2id:
            mock_name2id.side_effect = lambda m, t, i: "white_pawn_a2" if i == 1 else "world"

            data = MagicMock()
            mock_body = MagicMock()
            mock_body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
            # Displaced by 0.5*SQUARE_SIZE (too much)
            mock_body.xpos = np.array([BOARD_CENTER[0] + 0.5 * SQUARE_SIZE, BOARD_CENTER[1], TABLE_HEIGHT + 0.01])
            data.body.side_effect = lambda i: mock_body if i == 1 else MagicMock()

            observer = BoardObserver(model, data)
            with pytest.raises(StabilityError, match="significantly displaced"):
                observer.verify_stability()

    def test_board_observer_below_table_fail(self):
        """
        Verify that BoardObserver raises StabilityError when a piece is below table level.
        """
        model = MagicMock()
        model.nbody = 2
        with patch('mujoco.mj_id2name') as mock_name2id:
            mock_name2id.side_effect = lambda m, t, i: "white_pawn_a2" if i == 1 else "world"

            data = MagicMock()
            mock_body = MagicMock()
            mock_body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
            mock_body.xpos = np.array([BOARD_CENTER[0], BOARD_CENTER[1], TABLE_HEIGHT - 0.05])
            data.body.side_effect = lambda i: mock_body if i == 1 else MagicMock()

            observer = BoardObserver(model, data)
            with pytest.raises(StabilityError, match="below the table level"):
                observer.verify_stability()

    def test_real_scene_remains_stable_under_passive_physics(self):
        """
        Verify that the scene remains stable under passive physics for 300 steps.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        initial_positions = {}
        for body_idx in range(model.nbody):
            name = model.body(body_idx).name
            if name and name.startswith(("w_", "b_")) and "spare" not in name:
                initial_positions[name] = data.body(name).xpos.copy()

        for _ in range(300):
            mujoco.mj_step(model, data)

        drifts = [
            np.linalg.norm(data.body(name).xpos[:2] - pos[:2])
            for name, pos in initial_positions.items()
        ]

        assert max(drifts) < 0.004

    def test_env_initialization_does_not_destabilize_board(self):
        """
        Verify that environment initialization does not destabilize the board.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        # Initialization check
        ChessPickPlaceEnv(model, data)

        initial_positions = {}
        for body_idx in range(model.nbody):
            name = model.body(body_idx).name
            if name and name.startswith(("w_", "b_")) and "spare" not in name:
                initial_positions[name] = data.body(name).xpos.copy()

        for _ in range(120):
            mujoco.mj_step(model, data)

        drifts = [
            np.linalg.norm(data.body(name).xpos[:2] - pos[:2])
            for name, pos in initial_positions.items()
        ]
        min_piece_z = min(float(data.body(name).xpos[2]) for name in initial_positions)

        assert max(drifts) < 0.003
        assert min_piece_z > TABLE_HEIGHT + 0.008

    def _raw_step(self, env, model, data, action):
        """
        Helper method to perform a raw simulation step with a given action.
        """
        fetch_action = env._expand_fetch_action(np.asarray(action, dtype=np.float64))
        mujoco_utils.ctrl_set_action(model, data, fetch_action)
        mujoco_utils.mocap_set_action(model, data, fetch_action)
        data.mocap_pos[0] = np.clip(data.mocap_pos[0], env._mocap_min, env._mocap_max)
        for _ in range(env.n_substeps):
            mujoco.mj_step(model, data)

    def test_arm_motion_does_not_destabilize_board(self):
        """
        Verify that arm motion across sweep targets does not destabilize pieces.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        env = ChessPickPlaceEnv(model, data)
        env.set_target("w_spare_queen_1", data.body("w_spare_queen_1").xpos.copy())

        initial_positions = {}
        for body_idx in range(model.nbody):
            name = model.body(body_idx).name
            if name and name.startswith(("w_", "b_")) and "spare" not in name:
                initial_positions[name] = data.body(name).xpos.copy()

        sweep_targets = [
            np.array([1.18, 0.75, 0.56]),
            np.array([1.18, 0.75, 0.48]),
            np.array([1.34, 0.95, 0.48]),
            np.array([1.02, 0.55, 0.48]),
            np.array([1.34, 0.95, 0.56]),
        ]
        for target in sweep_targets:
            for _ in range(50):
                grip = env.get_grip_pos()
                delta = target - grip
                action = np.zeros(4, dtype=np.float32)
                action[:3] = np.clip(delta / ACTION_SCALE, -1.0, 1.0)
                action[3] = 1.0
                self._raw_step(env, model, data, action)

        drifts = [
            np.linalg.norm(data.body(name).xpos[:2] - pos[:2])
            for name, pos in initial_positions.items()
        ]
        min_piece_z = min(float(data.body(name).xpos[2]) for name in initial_positions)

        assert max(drifts) < 0.004
        assert min_piece_z > TABLE_HEIGHT + 0.009

    def test_captured_piece_in_graveyard_does_not_trigger_stability_error(self):
        """
        Verify that pieces in the graveyard are ignored by stability checks.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        from src.bootstrap import SystemBootstrapper
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)
        active_pieces = set(square_to_piece.values())

        # Mark a piece as inactive (captured) by removing it from active pieces
        captured_piece = "w_pawn_1"
        active_pieces.discard(captured_piece)

        # Intentionally submerge it below the table to simulate false positive
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, captured_piece)
        jnt_adr = model.body_jntadr[body_id]

        # Set pos below table limit
        data.qpos[jnt_adr:jnt_adr + 3] = [0.0, 0.0, 0.2]
        # Set orientation tipped
        data.qpos[jnt_adr + 3:jnt_adr + 7] = [0.0, 1.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        observer = BoardObserver(model, data)

        # Calling without active_pieces filter should raise exception
        with pytest.raises(StabilityError):
            observer.verify_stability()

        # Calling with filter should pass because it skips captured pieces
        observer.verify_stability(active_piece_names=active_pieces)
