"""Runtime orchestration for RoboChess.

This module contains the bootstrapping and turn-processing logic for the
interactive game. Keeping it outside the CLI entrypoint makes the runtime
easier to test, reuse, and evolve without coupling behavior to `__main__`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import chess
import mujoco
import mujoco.viewer
from huggingface_sb3 import load_from_hub
from sb3_contrib import TQC
from stable_baselines3.common.buffers import DictReplayBuffer

from src.config import HF_FILENAME, HF_REPO_ID, N_SUBSTEPS, SCENE_XML, Z_GRASP
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import (
    AIEngineError,
    BoardStateError,
    ExecutionError,
    PieceLookupError,
    RLModelError,
    SceneLoadError,
)
from src.health_checks import CheckContext, CheckHook, RuntimeCheckRegistry, build_default_check_registry
from src.logging_utils import log_event
from src.logic.chess_manager import ChessGameManager
from src.logic.operation_planner import OperationPlanner
from src.runtime_guard import (
    freeze_on_exception,
    validate_board_state,
)

logger = logging.getLogger("robo_chess")

SquareMap = dict[str, str]
InputFunc = Callable[[str], str]


@dataclass
class GameSystems:
    """Bootstrapped runtime objects required to run a RoboChess game."""

    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    env: ChessPickPlaceEnv
    rl_model: Any
    manager: ChessGameManager
    planner: OperationPlanner
    controller: ExecutionController
    square_to_piece: SquareMap
    arm_home_grip: Any
    check_registry: RuntimeCheckRegistry


def teleport_piece(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    piece_name: str,
    target_pos,
) -> None:
    """Teleport a piece to a target pose and reset its velocities."""
    try:
        body = mj_model.body(piece_name)
        jnt_adr = body.jntadr[0]
        if jnt_adr == -1:
            raise PieceLookupError(f"Piece '{piece_name}' has no joint and cannot be teleported.")

        mj_data.joint(jnt_adr).qpos[:3] = target_pos
        mj_data.joint(jnt_adr).qpos[3:7] = [1, 0, 0, 0]
        mj_data.joint(jnt_adr).qvel[:] = 0
    except KeyError as exc:
        raise PieceLookupError(f"Piece '{piece_name}' not found in model.") from exc
    except Exception as exc:
        raise ExecutionError(f"Error teleporting {piece_name}: {exc}") from exc


def _is_piece_body(body_name: str) -> bool:
    """Return True when the given MuJoCo body name represents a chess piece."""
    if not body_name:
        return False

    skip_prefixes = ("world", "table", "spare", "graveyard", "robot", "square")
    if any(body_name.startswith(prefix) or prefix in body_name for prefix in skip_prefixes):
        return False

    return body_name.startswith(("w_", "b_"))


def initialize_square_to_piece(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> SquareMap:
    """Build the initial square-to-piece mapping from the physical scene."""
    square_to_piece: SquareMap = {}
    a1_pos = ExecutionController.get_square_pos("a1")
    b1_pos = ExecutionController.get_square_pos("b1")
    a2_pos = ExecutionController.get_square_pos("a2")
    file_step = b1_pos[0] - a1_pos[0]
    rank_step = a2_pos[1] - a1_pos[1]

    for i in range(mj_model.nbody):
        body = mj_model.body(i)
        body_name = body.name
        if not _is_piece_body(body_name):
            continue

        pos = mj_data.body(i).xpos
        file_idx = int(round((pos[0] - a1_pos[0]) / file_step))
        rank_idx = int(round((pos[1] - a1_pos[1]) / rank_step))

        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            square_name = chess.square_name(chess.square(file_idx, rank_idx))
            square_to_piece[square_name] = body_name

    return square_to_piece


def validate_initial_mapping(square_to_piece: SquareMap) -> None:
    """Validate that the scene booted with a complete 32-piece board mapping."""
    if len(square_to_piece) != 32:
        raise BoardStateError(
            f"Expected 32 pieces, found {len(square_to_piece)}. "
            f"Mapped squares: {sorted(square_to_piece.keys())}"
        )


def load_scene(scene_xml: str = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MuJoCo scene and return model/data pair."""
    if not os.path.exists(scene_xml):
        raise SceneLoadError(f"Scene not found: {scene_xml}")

    try:
        mj_model = mujoco.MjModel.from_xml_path(scene_xml)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        return mj_model, mj_data
    except Exception as exc:
        raise SceneLoadError(f"Failed to load MuJoCo scene '{scene_xml}': {exc}") from exc


def load_rl_policy(env: ChessPickPlaceEnv, repo_id: str = HF_REPO_ID, filename: str = HF_FILENAME) -> Any:
    """Load the pretrained TQC checkpoint used by the runtime."""
    logger.info("Loading pretrained model: %s...", repo_id)
    try:
        checkpoint = load_from_hub(repo_id, filename)
        rl_model = TQC.load(
            checkpoint,
            env=env,
            custom_objects={
                "learning_rate": 0.001,
                "lr_schedule": lambda _: 0.001,
                "replay_buffer_kwargs": {},
                "replay_buffer_class": DictReplayBuffer,
            },
        )
        logger.info("TQC model loaded successfully.")
        return rl_model
    except Exception as exc:
        raise RLModelError(f"Failed to load RL model: {exc}") from exc


def bootstrap_game_systems(
    scene_xml: str = SCENE_XML,
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
) -> GameSystems:
    """Build all runtime systems and validate the initial physical board state."""
    mj_model, mj_data = load_scene(scene_xml)
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
    dummy_systems.check_registry.run(CheckHook.POST_SCENE_LOAD, CheckContext(hook=CheckHook.POST_SCENE_LOAD, systems=dummy_systems))
    log_event(logger, logging.INFO, "scene_loaded", scene_xml=scene_xml)

    rl_model = load_rl_policy(env, repo_id=repo_id, filename=filename)
    manager = dummy_systems.manager
    planner = dummy_systems.planner
    controller = dummy_systems.controller
    controller.rl_model = rl_model

    square_to_piece = initialize_square_to_piece(mj_model, mj_data)
    log_event(logger, logging.INFO, "board_initialized", mapped_pieces=len(square_to_piece))
    validate_initial_mapping(square_to_piece)
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

    def _run_stage_checks(phase: CheckHook, **payload: Any) -> None:
        """Adapt controller stage payloads into the stable check-context shape."""
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

    controller.set_stage_event_handler(
        _run_stage_checks
    )
    systems.check_registry.run(CheckHook.PROGRAM_START, CheckContext(hook=CheckHook.PROGRAM_START, systems=systems))
    return systems


def _update_square_mapping(square_to_piece: SquareMap, target_square: str, dest_square: str) -> None:
    """Update the logical square mapping after a successful piece move."""
    if "graveyard" in dest_square:
        square_to_piece.pop(target_square, None)
        return

    if target_square in square_to_piece:
        square_to_piece[dest_square] = square_to_piece.pop(target_square)


def handle_promotion(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    op,
    controller: ExecutionController,
    square_to_piece: SquareMap,
    is_white: bool,
) -> None:
    """Handle pawn promotion by swapping the pawn with a spare piece."""
    color_prefix = "w" if is_white else "b"
    promo_map = {
        chess.QUEEN: "queen",
        chess.ROOK: "rook",
        chess.BISHOP: "bishop",
        chess.KNIGHT: "knight",
    }
    promo_type = promo_map.get(op.promotion, "queen")
    spare_name = f"{color_prefix}_spare_{promo_type}"

    gy_pos = controller.get_pos("white_graveyard" if is_white else "black_graveyard")
    teleport_piece(mj_model, mj_data, op.piece_name, gy_pos)

    # Re-enable collision and visibility for the spare piece
    spare_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, spare_name)
    geom_id_start = mj_model.body_geomadr[spare_body_id]
    geom_id_end = geom_id_start + mj_model.body_geomnum[spare_body_id]
    for geom_id in range(geom_id_start, geom_id_end):
        if mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mj_model.geom_contype[geom_id] = 1
            mj_model.geom_conaffinity[geom_id] = 1
            mj_model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 0.0]
        elif mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[geom_id] = 2
            mj_model.geom_conaffinity[geom_id] = 2
            mj_model.geom_rgba[geom_id] = [1.0, 1.0, 1.0, 1.0]

    dest_pos = controller.get_pos(op.dest_square)
    teleport_piece(mj_model, mj_data, spare_name, dest_pos)

    square_to_piece[op.dest_square] = spare_name
    logger.info("Promoted: %s → %s at %s", op.piece_name, spare_name, op.dest_square)


def execute_human_ops(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ops,
    controller: ExecutionController,
    square_to_piece: SquareMap,
) -> None:
    """Execute human move operations via teleportation."""
    for op in ops:
        target_pos = controller.get_pos(op.dest_square)
        teleport_piece(mj_model, mj_data, op.piece_name, target_pos)
        _update_square_mapping(square_to_piece, op.target_square, op.dest_square)

        if op.promotion:
            handle_promotion(mj_model, mj_data, op, controller, square_to_piece, is_white=True)
        mujoco.mj_forward(mj_model, mj_data)


def execute_ai_ops(
    ops,
    controller: ExecutionController,
    square_to_piece: SquareMap,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    viewer=None,
) -> None:
    """Execute AI move operations through the physical controller."""
    for op in ops:
        if "graveyard" in op.dest_square:
            log_event(logger, logging.INFO, "ai_teleport_capture", piece=op.piece_name, dest=op.dest_square)
            teleport_piece(mj_model, mj_data, op.piece_name, controller.get_pos(op.dest_square))
        else:
            log_event(logger, logging.INFO, "ai_execute_op", piece=op.piece_name, src=op.target_square, dest=op.dest_square)
            result = controller.execute_op(op, viewer=viewer)
            if not result.success:
                raise ExecutionError(
                    f"RL execution failed for {op.piece_name} {op.target_square}->{op.dest_square}: {result.details}"
                )

        _update_square_mapping(square_to_piece, op.target_square, op.dest_square)
        if op.promotion:
            handle_promotion(mj_model, mj_data, op, controller, square_to_piece, is_white=False)

        sync_viewer(viewer, mj_model, mj_data)


def sync_viewer(viewer, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    """Refresh the viewer if one is available."""
    if viewer is None:
        return
    mujoco.mj_forward(mj_model, mj_data)
    viewer.sync()


def print_game_banner() -> None:
    """Print the startup banner shown when the interactive game launches."""
    print("\n" + "=" * 50)
    print("  ROBOCHESS — Human (White) vs AI (Black)")
    print("  Enter moves in UCI format (e.g., e2e4)")
    print("  Commands: quit, exit")
    print("=" * 50 + "\n")


def print_game_over(board: chess.Board) -> None:
    """Print the game-over summary for the current board result."""
    print("\n" + "=" * 50)
    print("  GAME OVER")
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"  Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("  Stalemate — draw.")
    else:
        print(f"  Result: {board.result()}")
    print("=" * 50)


def print_runtime_abort(exc: Exception) -> None:
    """Print a concise abort summary for expected operational failures."""
    print("\n" + "=" * 50)
    print("  GAME ABORTED")
    print(f"  Reason: {exc}")
    print("=" * 50)


def process_human_turn(systems: GameSystems, input_func: InputFunc = input) -> bool:
    """Process one human turn. Return False when the user requests quit."""
    print("\nWhite's turn (Human)")
    print(systems.manager.board)
    move_uci = input_func("Enter move: ").strip()

    if move_uci.lower() in ("quit", "exit"):
        logger.info("Player quit the game.")
        return False

    if not systems.manager.validate_move(move_uci):
        print(f"Illegal move: {move_uci}. Try again.")
        return True

    move = chess.Move.from_uci(move_uci)
    ops = systems.planner.generate_operations(move, systems.manager.board, systems.square_to_piece)
    execute_human_ops(
        systems.mj_model,
        systems.mj_data,
        ops,
        systems.controller,
        systems.square_to_piece,
    )
    systems.manager.push_move(move_uci)
    return True


def process_ai_turn(systems: GameSystems, viewer=None) -> bool:
    """Process one AI turn and update the logical and physical board."""
    print("\nBlack's turn (AI)")
    move = systems.manager.get_ai_move()
    if not move:
        raise AIEngineError("AI returned no move.")

    move_uci = move.uci()
    print(f"AI plays: {move_uci}")
    ops = systems.planner.generate_operations(move, systems.manager.board, systems.square_to_piece)
    execute_ai_ops(
        ops,
        systems.controller,
        systems.square_to_piece,
        systems.mj_model,
        systems.mj_data,
        viewer,
    )
    systems.manager.push_move(move_uci)
    return True


def complete_turn(systems: GameSystems, viewer=None, sleep_sec: float = 0.5) -> None:
    """Run post-turn validation and viewer synchronization."""
    systems.check_registry.run(CheckHook.TURN_END, CheckContext(hook=CheckHook.TURN_END, systems=systems, viewer=viewer))
    sync_viewer(viewer, systems.mj_model, systems.mj_data)
    time.sleep(sleep_sec)


def run_turn(
    systems: GameSystems,
    viewer=None,
    input_func: InputFunc = input,
    sleep_sec: float = 0.5,
) -> bool:
    """Run exactly one turn for the side to move."""
    systems.check_registry.run(CheckHook.TURN_START, CheckContext(hook=CheckHook.TURN_START, systems=systems, viewer=viewer))

    should_continue = (
        process_human_turn(systems, input_func=input_func)
        if systems.manager.board.turn == chess.WHITE
        else process_ai_turn(systems, viewer=viewer)
    )
    if not should_continue:
        return False

    complete_turn(systems, viewer=viewer, sleep_sec=sleep_sec)
    return True


def main() -> None:
    """Run the interactive RoboChess application."""
    systems = bootstrap_game_systems()
    log_event(logger, logging.INFO, "viewer_launch")

    with mujoco.viewer.launch_passive(systems.mj_model, systems.mj_data) as viewer:
        sync_viewer(viewer, systems.mj_model, systems.mj_data)
        print_game_banner()

        while viewer.is_running() and not systems.manager.board.is_game_over(claim_draw=True):
            try:
                if not run_turn(systems, viewer=viewer):
                    break
            except EOFError:
                logger.info("EOF received, ending game.")
                break
            except (ExecutionError, AIEngineError) as exc:
                logger.error("Operational RoboChess failure: %s", exc)
                sync_viewer(viewer, systems.mj_model, systems.mj_data)
                print_runtime_abort(exc)
                return
            except Exception as exc:
                freeze_on_exception(exc, viewer=viewer, mj_model=systems.mj_model, mj_data=systems.mj_data)
                return

        print_game_over(systems.manager.board)
        time.sleep(5.0)
