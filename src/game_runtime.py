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

from src.config import HF_FILENAME, HF_REPO_ID, N_SUBSTEPS, SCENE_XML, Z_GRASP, Z_SAFE
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

from src.models import GameSystems, SquareMap

logger = logging.getLogger("robo_chess")

InputFunc = Callable[[str], str]


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
    if any(body_name.startswith(prefix) for prefix in skip_prefixes):
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

        # Use the model's initial position (XML) instead of live data xpos
        # to be immune to any initial physics settling or homing collisions.
        pos = body.pos
        f_val = (pos[0] - a1_pos[0]) / file_step
        r_val = (pos[1] - a1_pos[1]) / rank_step
        file_idx = int(round(f_val))
        rank_idx = int(round(r_val))

        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            square_name = chess.square_name(chess.square(file_idx, rank_idx))
            square_to_piece[square_name] = body_name
        else:
            logger.debug(f"Piece {body_name} at {pos} mapped to OUT: f_val={f_val:.4f}, r_val={r_val:.4f}")

    return square_to_piece


def validate_initial_mapping(square_to_piece: SquareMap) -> None:
    """Validate that the scene booted with a complete 32-piece board mapping."""
    if len(square_to_piece) != 32:
        # Sort keys for deterministic error messages
        missing = []
        for f in range(8):
            for r in [0, 1, 6, 7]:
                sq = chess.square_name(chess.square(f, r))
                if sq not in square_to_piece:
                    missing.append(sq)
        
        raise BoardStateError(
            f"Expected 32 pieces, found {len(square_to_piece)}. "
            f"Missing squares: {missing}. "
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
    """Load the TQC checkpoint. Prioritizes local fine-tuned model if available."""
    from src.config import FINETUNED_MODEL_PATH
    
    model_path = None
    if FINETUNED_MODEL_PATH and os.path.exists(FINETUNED_MODEL_PATH + ".zip"):
        model_path = FINETUNED_MODEL_PATH
        logger.info("Loading local fine-tuned model: %s...", model_path)
    elif FINETUNED_MODEL_PATH and os.path.exists(FINETUNED_MODEL_PATH):
        # Handle case without .zip extension if it was provided fully
        model_path = FINETUNED_MODEL_PATH
        logger.info("Loading local fine-tuned model: %s...", model_path)
    
    try:
        if model_path:
            rl_model = TQC.load(model_path, env=env)
        else:
            logger.info("Loading pretrained model from hub: %s...", repo_id)
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
    
    # Initialize square_to_piece mapping BEFORE creating the env
    # because the env constructor triggers robot homing which might hit pieces.
    square_to_piece = initialize_square_to_piece(mj_model, mj_data)
    log_event(logger, logging.INFO, "board_initialized", mapped_pieces=len(square_to_piece))
    validate_initial_mapping(square_to_piece)

    env = ChessPickPlaceEnv(mj_model, mj_data, n_substeps=N_SUBSTEPS)
    arm_home_grip = env.get_grip_pos().copy()
    
    manager = ChessGameManager()
    try:
        dummy_systems = GameSystems(
            mj_model=mj_model,
            mj_data=mj_data,
            env=env,
            rl_model=None,
            manager=manager,
            planner=OperationPlanner(),
            controller=ExecutionController(None, env),
            square_to_piece=square_to_piece,
            arm_home_grip=arm_home_grip,
            check_registry=build_default_check_registry(),
            captured_count={"white": 0, "black": 0},
        )
        dummy_systems.check_registry.run(CheckHook.POST_SCENE_LOAD, CheckContext(hook=CheckHook.POST_SCENE_LOAD, systems=dummy_systems))
        log_event(logger, logging.INFO, "scene_loaded", scene_xml=scene_xml)

        rl_model = load_rl_policy(env, repo_id=repo_id, filename=filename)
        planner = dummy_systems.planner
        controller = dummy_systems.controller
        controller.rl_model = rl_model

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
            captured_count={"white": 0, "black": 0},
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
    except Exception:
        manager.close()
        raise


def _update_square_mapping(systems: GameSystems, target_square: str, dest_square: str) -> None:
    """Update the logical square mapping after a successful piece move."""
    square_to_piece = systems.square_to_piece
    if "graveyard" in dest_square:
        square_to_piece.pop(target_square, None)
        # Increment capture counter
        color = "white" if "white" in dest_square else "black"
        systems.captured_count[color] += 1
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
    
    # Find an available spare piece of the requested type
    # We generated 2 of each in generate_xml.py
    spare_name = None
    for i in (1, 2):
        candidate = f"{color_prefix}_spare_{promo_type}_{i}"
        if candidate not in square_to_piece.values():
            spare_name = candidate
            break
            
    if spare_name is None:
        logger.warning("No more spare %s pieces available for %s", promo_type, color_prefix)
        # Fallback to first one if all are "used", though this shouldn't happen in normal games
        spare_name = f"{color_prefix}_spare_{promo_type}_1"

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
            # Make it visible with correct color
            mj_model.geom_rgba[geom_id] = [1.0, 1.0, 1.0, 1.0] if is_white else [0.1, 0.1, 0.1, 1.0]

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
    systems: GameSystems = None,
    viewer=None,
    use_arm: bool = False,
) -> None:
    """Execute human move operations via teleportation (default) or robotic arm.
    
    Note: Robotic arm execution for board moves requires fine-tuned policy 
    to handle the full board workspace.
    """
    for op in ops:
        if "graveyard" in op.dest_square or not use_arm:
            target_pos = controller.get_pos(op.dest_square)
            if systems and "graveyard" in op.dest_square:
                color = "white" if "white" in op.dest_square else "black"
                target_pos = _get_graveyard_grid_pos(target_pos, systems.captured_count[color])
            teleport_piece(mj_model, mj_data, op.piece_name, target_pos)
            
            if systems:
                _update_square_mapping(systems, op.target_square, op.dest_square)
            else:
                if "graveyard" in op.dest_square:
                    square_to_piece.pop(op.target_square, None)
                elif op.target_square in square_to_piece:
                    square_to_piece[op.dest_square] = square_to_piece.pop(op.target_square)
        else:
            log_event(logger, logging.INFO, "human_execute_op", piece=op.piece_name, src=op.target_square, dest=op.dest_square)
            result = controller.execute_op(op, viewer=viewer)
            if not result.success:
                raise ExecutionError(
                    f"Physical execution failed for human move {op.piece_name} {op.target_square}->{op.dest_square}: {result.details}"
                )
            if systems:
                _update_square_mapping(systems, op.target_square, op.dest_square)

        if op.promotion:
            is_white = op.piece_name.startswith("w_")
            handle_promotion(mj_model, mj_data, op, controller, square_to_piece, is_white=is_white)
        mujoco.mj_forward(mj_model, mj_data)
        sync_viewer(viewer, mj_model, mj_data)



def execute_ai_ops(
    ops,
    controller: ExecutionController,
    square_to_piece: SquareMap,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    viewer=None,
    systems: GameSystems = None,
) -> None:
    """Execute AI move operations through the physical controller."""
    for op in ops:
        if "graveyard" in op.dest_square:
            log_event(logger, logging.INFO, "ai_teleport_capture", piece=op.piece_name, dest=op.dest_square)
            target_pos = controller.get_pos(op.dest_square)
            if systems:
                color = "white" if "white" in op.dest_square else "black"
                target_pos = _get_graveyard_grid_pos(target_pos, systems.captured_count[color])
            teleport_piece(mj_model, mj_data, op.piece_name, target_pos)
        else:
            log_event(logger, logging.INFO, "ai_execute_op", piece=op.piece_name, src=op.target_square, dest=op.dest_square)
            result = controller.execute_op(op, viewer=viewer)
            if not result.success:
                raise ExecutionError(
                    f"RL execution failed for {op.piece_name} {op.target_square}->{op.dest_square}: {result.details}"
                )

        if systems:
            _update_square_mapping(systems, op.target_square, op.dest_square)
        else:
            # Fallback
            if "graveyard" in op.dest_square:
                square_to_piece.pop(op.target_square, None)
            elif op.target_square in square_to_piece:
                square_to_piece[op.dest_square] = square_to_piece.pop(op.target_square)

        if op.promotion:
            handle_promotion(mj_model, mj_data, op, controller, square_to_piece, is_white=False)

        sync_viewer(viewer, mj_model, mj_data)


def _get_graveyard_grid_pos(origin, count: int):
    """Compute a grid offset for captured pieces in the graveyard."""
    rows = 4
    spacing = 0.04
    row = count % rows
    col = count // rows
    offset = [col * spacing, row * spacing, 0]
    return [origin[i] + offset[i] for i in range(3)]


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
    print("  Commands: quit, exit, hint, suggest")
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
    elif board.is_insufficient_material():
        print("  Draw — insufficient material.")
    elif board.can_claim_fifty_moves():
        print("  Draw — 50-move rule.")
    elif board.can_claim_threefold_repetition():
        print("  Draw — threefold repetition.")
    else:
        print(f"  Result: {board.result()}")
    print("=" * 50)


def print_runtime_abort(exc: Exception) -> None:
    """Print a concise abort summary for expected operational failures."""
    print("\n" + "=" * 50)
    print("  GAME ABORTED")
    print(f"  Reason: {exc}")
    print("=" * 50)


import tkinter as tk
import threading
import queue
from src.game_loop import GameLoop
from src.gui.board_panel import BoardPanel
import traceback

def main() -> None:
    """Run the interactive RoboChess application with GUI."""
    systems = bootstrap_game_systems()
    log_event(logger, logging.INFO, "viewer_launch")
    
    game_loop = GameLoop(systems)
    root = tk.Tk()
    
    move_queue = queue.Queue()
    stop_event = threading.Event()
    
    def on_gui_move(uci):
        move_queue.put(uci)
        
    panel = BoardPanel(root, game_loop, on_move_callback=on_gui_move)
    panel.pack(fill=tk.BOTH, expand=True)

    def game_worker():
        try:
            with mujoco.viewer.launch_passive(systems.mj_model, systems.mj_data) as viewer:
                sync_viewer(viewer, systems.mj_model, systems.mj_data)
                print_game_banner()
                
                while not stop_event.is_set() and viewer.is_running() and not systems.manager.board.is_game_over(claim_draw=True):
                    state = game_loop.get_state()
                    try:
                        if state.board.turn == chess.WHITE:
                            # Use a timeout to allow checking stop_event
                            try:
                                uci = move_queue.get(timeout=1.0)
                            except queue.Empty:
                                continue
                                
                            if uci is None:
                                # Auto play step requested
                                if not panel.auto_play.get():
                                    # Meaningless button click, continue
                                    continue
                                # Use AI for white
                                res = game_loop.execute_ai_turn(viewer)
                            else:
                                res = game_loop.submit_move(uci, viewer=viewer)
                        else: # black turn
                            if panel.auto_play.get():
                                # wait for user to click next turn
                                try:
                                    uci = move_queue.get(timeout=1.0)
                                except queue.Empty:
                                    continue
                                if uci is not None:
                                    continue # Ignore explicit moves on AI's turn
                            res = game_loop.execute_ai_turn(viewer)
                    except Exception as e:
                        print(f"Loop Error: {traceback.format_exc()}")
                        break
                        
                    root.after(0, panel.refresh)
                    
                    if res.success:
                        root.after(0, panel.append_history, res.message)
                    else:
                        if res.error:
                            print(f"Execution Error: {traceback.format_exc()}")
                            freeze_on_exception(res.error, viewer=viewer, mj_model=systems.mj_model, mj_data=systems.mj_data)
                            break
                        else:
                            print(f"Failed: {res.message}")

                if stop_event.is_set():
                    return

                if not viewer.is_running():
                    return
                    
                print_game_over(systems.manager.board)
                root.after(0, panel.refresh)
                
                while not stop_event.is_set() and viewer.is_running():
                    time.sleep(1)
        finally:
            systems.manager.close()

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    thread = threading.Thread(target=game_worker, daemon=True)
    thread.start()
    
    root.mainloop()
