"""
Runtime validation and freeze-on-failure helpers for RoboChess.

This module provides tools for verifying the physical board state against
the logical chess board and handling fatal simulation errors.
"""

import logging
import time
import chess
import mujoco
import numpy as np

from src.config import PLACEMENT_TOLERANCE, TILT_THRESHOLD_COS
from src.exceptions import BoardStateError, PieceLookupError, StabilityError
from src.observation.board_observer import BoardObserver

logger = logging.getLogger(__name__)


class RuntimeGuard:
    """Orchestrates runtime validations and failure handling."""

    @staticmethod
    def expected_board_squares(board):
        """
        Return the expected square occupancy from the logical chess board.
        
        Args:
            board: The chess.Board object.
            
        Returns:
            A dictionary mapping square names to chess.Piece objects.
        """
        return {
            chess.square_name(square): piece
            for square, piece in board.piece_map().items()
        }

    @staticmethod
    def _piece_type_token(piece):
        """
        Map a python-chess piece to the naming token used in MuJoCo bodies.
        
        Args:
            piece: A chess.Piece object.
            
        Returns:
            A string token (e.g., 'pawn').
        """
        return {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }[piece.piece_type]

    @staticmethod
    def _body_by_name(mj_data, piece_name):
        """
        Resolve a MuJoCo body by name with a domain-specific exception.
        
        Args:
            mj_data: The MuJoCo data object.
            piece_name: The name of the piece body.
            
        Returns:
            The MuJoCo body object.
            
        Raises:
            PieceLookupError: If the body is not found.
        """
        try:
            return mj_data.body(piece_name)
        except KeyError as exc:
            raise PieceLookupError(f"MuJoCo body not found for piece '{piece_name}'.") from exc

    @staticmethod
    def _up_z_from_quat(quat):
        """
        Compute the world-space Z component of a body's local up vector.
        
        Args:
            quat: A quaternion array.
            
        Returns:
            The Z component of the up vector.
        """
        return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))

    @classmethod
    def validate_piece_identity(cls, square_name, piece_name, piece):
        """
        Verify that the mapped MuJoCo body matches the logical chess piece.
        
        Args:
            square_name: Name of the square.
            piece_name: Name of the MuJoCo piece body.
            piece: The chess.Piece object expected at the square.
            
        Raises:
            BoardStateError: If the identity does not match.
        """
        color_prefix = "w_" if piece.color == chess.WHITE else "b_"
        if not piece_name.startswith(color_prefix):
            raise BoardStateError(
                f"Square {square_name} expects color prefix {color_prefix!r}, got piece '{piece_name}'."
            )

        expected_token = cls._piece_type_token(piece)
        if expected_token not in piece_name:
            raise BoardStateError(
                f"Square {square_name} expects a {expected_token}, got piece '{piece_name}'."
            )

    @classmethod
    def validate_board_state(cls, mj_model, mj_data, board, square_to_piece,
                             controller, position_tolerance=PLACEMENT_TOLERANCE):
        """
        Assert that the physical board matches the logical chess state.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            board: The chess.Board object.
            square_to_piece: Mapping of squares to piece body names.
            controller: The execution controller for position lookups.
            position_tolerance: Allowed displacement before erroring.
            
        Raises:
            BoardStateError: If a state mismatch is detected.
            StabilityError: If a piece is tilted.
        """
        expected = cls.expected_board_squares(board)
        mapped = dict(square_to_piece)

        missing = sorted(set(expected) - set(mapped))
        extra = sorted(set(mapped) - set(expected))
        if missing or extra:
            raise BoardStateError(
                "Square mapping mismatch | "
                f"missing={missing} | extra={extra}"
            )

        for square_name, piece in expected.items():
            piece_name = mapped[square_name]
            cls.validate_piece_identity(square_name, piece_name, piece)

            body = cls._body_by_name(mj_data, piece_name)
            actual_pos = np.array(body.xpos, dtype=np.float64)
            actual_quat = np.array(body.xquat, dtype=np.float64)
            expected_pos = np.array(controller.get_pos(square_name), dtype=np.float64)

            xy_error = float(np.linalg.norm(actual_pos[:2] - expected_pos[:2]))
            z_error = abs(float(actual_pos[2] - expected_pos[2]))
            up_z = cls._up_z_from_quat(actual_quat)

            if xy_error > position_tolerance or z_error > position_tolerance:
                raise BoardStateError(
                    "Piece coordinate mismatch | "
                    f"square={square_name} | piece={piece_name} | "
                    f"expected={expected_pos.round(4)} | actual={actual_pos.round(4)} | "
                    f"xy_error={xy_error:.4f} | z_error={z_error:.4f}"
                )

            if up_z < TILT_THRESHOLD_COS:
                raise StabilityError(
                    "Piece tilted at turn start | "
                    f"square={square_name} | piece={piece_name} | up_z={up_z:.4f}"
                )

        BoardObserver(mj_model, mj_data).verify_stability(
            active_piece_names=set(square_to_piece.values())
        )

    @staticmethod
    def run_freeze_loop(viewer=None, mj_model=None, mj_data=None,
                        sleep_sec=0.1, max_cycles=None):
        """
        Keep the simulator open at the failing state for manual inspection.
        
        Args:
            viewer: The MuJoCo viewer instance.
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            sleep_sec: Time to sleep between cycles.
            max_cycles: Optional maximum number of cycles to run.
        """
        cycles = 0
        while True:
            if viewer is not None:
                if not viewer.is_running():
                    break
                if mj_model is not None and mj_data is not None:
                    mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

            time.sleep(sleep_sec)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break

    @classmethod
    def freeze_on_exception(cls, exc, viewer=None, mj_model=None, mj_data=None):
        """
        Log a fatal exception and freeze the simulator state for inspection.
        
        Args:
            exc: The exception that occurred.
            viewer: The MuJoCo viewer instance.
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
        """
        logger.error(
            "Fatal RoboChess error. Freezing simulation for inspection.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cls.run_freeze_loop(viewer=viewer, mj_model=mj_model, mj_data=mj_data)
