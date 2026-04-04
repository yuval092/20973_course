"""Runtime validation and freeze-on-failure helpers."""

from __future__ import annotations

import logging
import time
from typing import Optional

import chess
import mujoco
import numpy as np

from src.config import PLACEMENT_TOLERANCE, TILT_THRESHOLD_COS
from src.exceptions import BoardStateError, PieceLookupError, StabilityError
from src.observation.board_observer import BoardObserver

logger = logging.getLogger(__name__)


def expected_board_squares(board: chess.Board) -> dict[str, chess.Piece]:
    """Return the expected square occupancy from the logical chess board."""
    return {
        chess.square_name(square): piece
        for square, piece in board.piece_map().items()
    }


def _piece_type_token(piece: chess.Piece) -> str:
    """Map a python-chess piece to the naming token used in MuJoCo bodies."""
    return {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }[piece.piece_type]


def _body_by_name(mj_data: mujoco.MjData, piece_name: str):
    """Resolve a MuJoCo body by name with a domain-specific exception."""
    try:
        return mj_data.body(piece_name)
    except KeyError as exc:
        raise PieceLookupError(f"MuJoCo body not found for piece '{piece_name}'.") from exc


def _up_z_from_quat(quat: np.ndarray) -> float:
    """Compute the world-space Z component of a body's local up vector."""
    return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))


def validate_piece_identity(square_name: str, piece_name: str, piece: chess.Piece) -> None:
    """Verify that the mapped MuJoCo body matches the logical chess piece."""
    color_prefix = "w_" if piece.color == chess.WHITE else "b_"
    if not piece_name.startswith(color_prefix):
        raise BoardStateError(
            f"Square {square_name} expects color prefix {color_prefix!r}, got piece '{piece_name}'."
        )

    expected_token = _piece_type_token(piece)
    if expected_token not in piece_name:
        raise BoardStateError(
            f"Square {square_name} expects a {expected_token}, got piece '{piece_name}'."
        )


def validate_board_state(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    board: chess.Board,
    square_to_piece: dict[str, str],
    controller,
    position_tolerance: float = PLACEMENT_TOLERANCE,
) -> None:
    """Assert that the physical board matches the logical chess state."""
    expected = expected_board_squares(board)
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
        validate_piece_identity(square_name, piece_name, piece)

        body = _body_by_name(mj_data, piece_name)
        actual_pos = np.array(body.xpos, dtype=np.float64)
        actual_quat = np.array(body.xquat, dtype=np.float64)
        expected_pos = np.array(controller.get_pos(square_name), dtype=np.float64)

        xy_error = float(np.linalg.norm(actual_pos[:2] - expected_pos[:2]))
        z_error = abs(float(actual_pos[2] - expected_pos[2]))
        up_z = _up_z_from_quat(actual_quat)

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

    BoardObserver(mj_model, mj_data).verify_stability()


def detect_physical_instability(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    board: chess.Board,
    square_to_piece: dict[str, str],
    controller,
) -> None:
    """Raise if the scene no longer matches the expected logical state."""
    validate_board_state(mj_model, mj_data, board, square_to_piece, controller)


def run_freeze_loop(
    viewer=None,
    mj_model: Optional[mujoco.MjModel] = None,
    mj_data: Optional[mujoco.MjData] = None,
    sleep_sec: float = 0.1,
    max_cycles: Optional[int] = None,
) -> None:
    """Keep the simulator open at the failing state for manual inspection."""
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


def freeze_on_exception(exc: Exception, viewer=None, mj_model=None, mj_data=None) -> None:
    """Log a fatal exception and freeze the simulator state for inspection."""
    logger.error(
        "Fatal RoboChess error. Freezing simulation for inspection.",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    run_freeze_loop(viewer=viewer, mj_model=mj_model, mj_data=mj_data)
