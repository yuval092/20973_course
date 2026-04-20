"""
Runtime orchestration and execution for RoboChess.

This module provides the core execution logic for chess moves,
including teleportation for captures and physical moves for standard turns.
"""

import logging
import chess
import mujoco
from src.config import BOARD_CENTER, PIECE_HIDE_Z, GRAVEYARD_SPACING, GRAVEYARD_COLS
from src.exceptions import ExecutionError, PieceLookupError

logger = logging.getLogger("robo_chess")


class GameOrchestrator:
    """Orchestrates move execution and physical board updates."""

    @staticmethod
    def teleport_piece(mj_model, mj_data, piece_name, target_pos):
        """
        Teleport a piece to a target pose and reset its velocities.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            piece_name: Name of the piece body to teleport.
            target_pos: Target [X, Y, Z] position.
            
        Raises:
            PieceLookupError: If the piece or its joint is not found.
            ExecutionError: If any other error occurs during teleportation.
        """
        try:
            body = mj_model.body(piece_name)
            jnt_adr = body.jntadr[0]
            if jnt_adr == -1:
                raise PieceLookupError(
                    f"Piece '{piece_name}' has no joint and cannot be teleported.")

            mj_data.joint(jnt_adr).qpos[:3] = target_pos
            mj_data.joint(jnt_adr).qpos[3:7] = [1, 0, 0, 0]
            mj_data.joint(jnt_adr).qvel[:] = 0
        except KeyError as exc:
            raise PieceLookupError(
                f"Piece '{piece_name}' not found in model.") from exc
        except PieceLookupError:
            raise
        except Exception as exc:
            raise ExecutionError(f"Error teleporting {piece_name}: {exc}") from exc

    @staticmethod
    def sync_viewer(viewer, mj_model, mj_data):
        """Refresh the viewer if one is available."""
        if viewer is None:
            return
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

    @staticmethod
    def _get_graveyard_grid_pos(origin, count):
        """
        Compute a grid offset for captured pieces in the graveyard.
        
        Args:
            origin: The origin [X, Y, Z] of the graveyard.
            count: Number of pieces already in the graveyard.
            
        Returns:
            The offset [X, Y, Z] position.
        """
        cols = GRAVEYARD_COLS or 4
        spacing = GRAVEYARD_SPACING or 0.04
        row = count // cols
        col = count % cols
        offset = [col * spacing, row * spacing, 0]
        return [origin[i] + offset[i] for i in range(3)]

    @staticmethod
    def _update_square_mapping(systems, target_square, dest_square):
        """
        Update the logical square mapping after a successful piece move.
        
        Args:
            systems: The GameSystems instance.
            target_square: The source square name.
            dest_square: The destination square name.
        """
        square_to_piece = systems.square_to_piece
        if "graveyard" in dest_square:
            square_to_piece.pop(target_square, None)
            return

        if target_square in square_to_piece:
            square_to_piece[dest_square] = square_to_piece.pop(target_square)

    @classmethod
    def handle_promotion(cls, mj_model, mj_data, op, controller,
                         square_to_piece, is_white):
        """
        Handle pawn promotion by swapping the pawn with a spare piece.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            op: The PickPlaceOp being executed.
            controller: The execution controller for position lookups.
            square_to_piece: Mapping of squares to piece body names.
            is_white: True if the promoting piece is white.
        """
        color_prefix = "w" if is_white else "b"
        promo_map = {
            chess.QUEEN: "queen",
            chess.ROOK: "rook",
            chess.BISHOP: "bishop",
            chess.KNIGHT: "knight",
        }
        promo_type = promo_map.get(op.promotion, "queen")
        
        # Find an available spare piece of the requested type
        spare_name = None
        for i in (1, 2):
            candidate = f"{color_prefix}_spare_{promo_type}_{i}"
            if candidate not in square_to_piece.values():
                spare_name = candidate
                break
                
        if spare_name is None:
            logger.warning("No more spare %s pieces available for %s",
                           promo_type, color_prefix)
            spare_name = f"{color_prefix}_spare_{promo_type}_1"

        hidden_pos = [BOARD_CENTER[0], BOARD_CENTER[1], PIECE_HIDE_Z]
        cls.teleport_piece(mj_model, mj_data, op.piece_name, hidden_pos)

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
                mj_model.geom_rgba[geom_id] = ([1.0, 1.0, 1.0, 1.0] if is_white
                                             else [0.1, 0.1, 0.1, 1.0])

        dest_pos = controller.get_pos(op.dest_square)
        cls.teleport_piece(mj_model, mj_data, spare_name, dest_pos)

        square_to_piece[op.dest_square] = spare_name
        logger.info("Promoted: %s → %s at %s",
                    op.piece_name, spare_name, op.dest_square)

    @classmethod
    def execute_ops(cls, mj_model, mj_data, ops, controller, square_to_piece,
                    systems=None, viewer=None, physical_arm=True):
        """
        Execute a sequence of PickPlaceOps.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            ops: List of PickPlaceOp instances to execute.
            controller: The execution controller for the robot arm.
            square_to_piece: Mapping of squares to piece body names.
            systems: Optional GameSystems instance for state updates.
            viewer: Optional MuJoCo viewer instance.
            physical_arm: If True, uses the robot arm for moves; else teleports.
        """
        for op in ops:
            is_graveyard = "graveyard" in str(op.dest_square).lower()
            
            # Check for policy workspace compatibility if physical arm is requested
            use_teleport = not physical_arm
            if physical_arm and not is_graveyard:
                src_pos = controller.get_pos(op.target_square)
                dest_pos = controller.get_pos(op.dest_square)
                issues = controller._policy_compatibility_issues(src_pos, dest_pos)
                if issues:
                    logger.warning(
                        "Move %s->%s is outside policy workspace. Falling back to teleportation. Issues: %s",
                        op.target_square, op.dest_square, "; ".join(issues)
                    )
                    use_teleport = True

            if is_graveyard:
                target_pos = controller.get_pos(op.dest_square)
                if systems:
                    color = "white" if "white" in op.dest_square else "black"
                    target_pos = cls._get_graveyard_grid_pos(
                        target_pos, systems.captured_count[color])
                    systems.captured_count[color] += 1
                cls.teleport_piece(mj_model, mj_data, op.piece_name, target_pos)
            elif use_teleport:
                cls.teleport_piece(mj_model, mj_data, op.piece_name,
                                 controller.get_pos(op.dest_square))
            elif physical_arm:
                result = controller.execute_op(op, viewer=viewer)
                if not result.success:
                    raise ExecutionError(
                        f"Physical execution failed for {op.piece_name} "
                        f"{op.target_square}→{op.dest_square}: {result.details}"
                    )
            
            if systems:
                cls._update_square_mapping(systems, op.target_square, op.dest_square)
            else:
                if "graveyard" in op.dest_square:
                    square_to_piece.pop(op.target_square, None)
                elif op.target_square in square_to_piece:
                    square_to_piece[op.dest_square] = square_to_piece.pop(
                        op.target_square)

            if op.promotion:
                is_white = op.piece_name.startswith("w_")
                cls.handle_promotion(mj_model, mj_data, op, controller,
                                    square_to_piece, is_white=is_white)
            
            mujoco.mj_forward(mj_model, mj_data)
            cls.sync_viewer(viewer, mj_model, mj_data)
