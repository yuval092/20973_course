"""
Bootstrapping and system initialization for RoboChess.

This module provides tools for loading the MuJoCo scene, RL policies, and 
initializing the core game systems.
"""

import logging
import os
import chess
import mujoco
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer

from src.config import LOCAL_MODEL_PATH, N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import (
    BoardStateError,
    RLModelError,
    SceneLoadError,
)
from src.health_checks import CheckContext, CheckHook, build_default_check_registry
from src.logging_utils import log_event
from src.logic.chess_manager import ChessGameManager
from src.logic.operation_planner import OperationPlanner
from src.models import GameSystems

logger = logging.getLogger("robo_chess")


class SystemBootstrapper:
    """Handles the initialization of all RoboChess runtime systems."""

    @staticmethod
    def load_scene(scene_xml=SCENE_XML):
        """
        Load the MuJoCo scene and return the loaded model/data pair.
        
        Args:
            scene_xml: Path to the scene XML file.
            
        Returns:
            A tuple of (MjModel, MjData).
            
        Raises:
            SceneLoadError: If the scene cannot be loaded.
        """
        if not os.path.exists(scene_xml):
            raise SceneLoadError(f"Scene not found: {scene_xml}")

        try:
            mj_model = mujoco.MjModel.from_xml_path(scene_xml)
            mj_data = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, mj_data)
            return mj_model, mj_data
        except Exception as exc:
            raise SceneLoadError(
                f"Failed to load MuJoCo scene '{scene_xml}': {exc}") from exc

    @staticmethod
    def load_rl_policy(env, model_path=LOCAL_MODEL_PATH):
        """
        Load the pretrained RL policy from a local file.
        
        Args:
            env: The environment the policy was trained for.
            model_path: Path to the .zip model file.
            
        Returns:
            The loaded RL model.
            
        Raises:
            RLModelError: If the model fails to load.
        """
        logger.info("Loading local RL model: %s...", model_path)
        if not os.path.exists(model_path):
            # Try relative to project root if not absolute
            from src.config import PACKAGE_DIR
            alt_path = os.path.join(PACKAGE_DIR.parent, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        try:
            rl_model = SAC.load(model_path, env=env)
            logger.info("SAC model loaded successfully from %s", model_path)
            return rl_model
        except Exception as exc:
            raise RLModelError(f"Failed to load RL model from {model_path}: {exc}") from exc

    @staticmethod
    def _is_piece_body(body_name):
        """
        Check if the given body name represents a chess piece.
        
        Args:
            body_name: The name of the body.
            
        Returns:
            True if it's a piece, False otherwise.
        """
        if not body_name:
            return False

        skip_prefixes = ("world", "table", "spare", "graveyard", "robot", "square")
        if any(body_name.startswith(prefix) for prefix in skip_prefixes):
            return False

        return body_name.startswith(("w_", "b_"))

    @classmethod
    def initialize_square_to_piece(cls, mj_model, mj_data):
        """
        Build the initial square-to-piece mapping from the physical scene.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            
        Returns:
            A dictionary mapping square names to body names.
        """
        square_to_piece = {}
        a1_pos = ExecutionController.get_square_pos("a1")
        b1_pos = ExecutionController.get_square_pos("b1")
        a2_pos = ExecutionController.get_square_pos("a2")
        file_step = b1_pos[0] - a1_pos[0]
        rank_step = a2_pos[1] - a1_pos[1]

        for i in range(mj_model.nbody):
            body = mj_model.body(i)
            body_name = body.name
            if not cls._is_piece_body(body_name):
                continue

            pos = mj_data.body(i).xpos
            file_idx = int(round((pos[0] - a1_pos[0]) / file_step))
            rank_idx = int(round((pos[1] - a1_pos[1]) / rank_step))

            if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                square_name = chess.square_name(chess.square(file_idx, rank_idx))
                square_to_piece[square_name] = body_name

        return square_to_piece

    @staticmethod
    def validate_initial_mapping(square_to_piece):
        """
        Validate that the scene booted with a complete 32-piece board mapping.
        
        Args:
            square_to_piece: The mapping to validate.
            
        Raises:
            BoardStateError: If piece count is not 32.
        """
        if len(square_to_piece) != 32:
            raise BoardStateError(
                f"Expected 32 pieces, found {len(square_to_piece)}. "
                f"Mapped squares: {sorted(square_to_piece.keys())}"
            )

    @classmethod
    def bootstrap_game_systems(cls, scene_xml=SCENE_XML, model_path=LOCAL_MODEL_PATH):
        """
        Build all runtime systems into a unified manager payload.
        
        Args:
            scene_xml: Path to MuJoCo scene.
            model_path: Path to local RL model.
            
        Returns:
            A GameSystems instance.
        """
        # 1. Start Environment & Initial Physics Payload
        mj_model, mj_data = cls.load_scene(scene_xml)
        env = ChessPickPlaceEnv(mj_model, mj_data, n_substeps=N_SUBSTEPS)
        arm_home_grip = env.get_grip_pos().copy()
        
        dummy_systems = GameSystems(
            mj_model=mj_model,
            mj_data=mj_data,
            env=env,
            rl_model=None,
            manager=ChessGameManager(),
            planner=OperationPlanner(),
            controller=ExecutionController(None, env),
            square_to_piece={},
            arm_home_grip=arm_home_grip,
            check_registry=build_default_check_registry(),
        )
        
        # Run initial post-load checks
        dummy_systems.check_registry.run(
            CheckHook.POST_SCENE_LOAD, 
            CheckContext(hook=CheckHook.POST_SCENE_LOAD, systems=dummy_systems)
        )
        log_event(logger, logging.INFO, "scene_loaded", scene_xml=scene_xml)

        # 2. Download/Mount Pretrained RL Weights
        rl_model = cls.load_rl_policy(env, model_path=model_path)
        manager = dummy_systems.manager
        planner = dummy_systems.planner
        controller = dummy_systems.controller
        controller.rl_model = rl_model

        # 3. Synchronize Game Logic Board to Physical Position
        square_to_piece = cls.initialize_square_to_piece(mj_model, mj_data)
        log_event(logger, logging.INFO, "board_initialized",
                  mapped_pieces=len(square_to_piece))
        cls.validate_initial_mapping(square_to_piece)
        
        systems = GameSystems(
            mj_model=mj_model,
            mj_data=mj_data,
            env=env,
            rl_model=rl_model,
            manager=manager,
            planner=planner,
            controller=controller,
            square_to_piece=square_to_piece,
            arm_home_grip=arm_home_grip,
            check_registry=dummy_systems.check_registry,
        )

        # 4. Attach Event Hooks for Policy Validations
        def _run_stage_checks(phase, **payload):
            """Adapt controller stage payloads into check-context."""
            context = CheckContext(
                hook=phase,
                systems=systems,
                viewer=payload.get("viewer"),
                stage=payload.get("stage"),
                op=payload.get("op"),
                extra={
                    key: value
                    for key, value in payload.items()
                    if key not in {"viewer", "stage", "op"}
                },
            )
            systems.check_registry.run(phase, context)

        controller.set_stage_event_handler(_run_stage_checks)
        
        systems.check_registry.run(
            CheckHook.PROGRAM_START, 
            CheckContext(hook=CheckHook.PROGRAM_START, systems=systems)
        )
        
        return systems
