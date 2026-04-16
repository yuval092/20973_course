"""Runtime safety guards and physical-logical state validation."""

from __future__ import annotations

import logging
import time

import chess
import mujoco
import numpy as np

from src.config import (
    PLACEMENT_TOLERANCE,
    TILT_THRESHOLD_COS,
)
from src.exceptions import BoardStateError, StabilityError
from src.observation.board_observer import BoardObserver


logger = logging.getLogger(__name__)


def expected_board_squares(board: chess.Board) -> dict[str, chess.Piece]:
    """Return a map of square names to logical piece objects for current state."""
    return {
        chess.square_name(sq): board.piece_at(sq)
        for sq in chess.SQUARES
        if board.piece_at(sq) is not None
    }


def _body_by_name(data: mujoco.MjData, name: str):
    """Return the MuJoCo body object or raise clearly."""
    try:
        return data.body(name)
    except KeyError as exc:
        raise BoardStateError(f"Piece body '{name}' not found in scene.") from exc


def _up_z_from_quat(quat: np.ndarray) -> float:
    """Compute the Z-component of the body's UP vector from its quaternion."""
    # quat is [w, x, y, z]
    # The UP vector in body frame is [0, 0, 1]
    # Rotated UP vector z component is 1 - 2*x^2 - 2*y^2
    return float(1.0 - 2.0 * (quat[1]**2 + quat[2]**2))


def validate_piece_identity(square_name: str, piece_name: str, expected_piece: chess.Piece) -> None:
    """Assert that a physical body name matches the expected chess piece type/color."""
    color_prefix = "w_" if expected_piece.color == chess.WHITE else "b_"
    expected_token = f"{color_prefix}{chess.piece_name(expected_piece.piece_type)}"
    spare_token = f"{color_prefix}spare_{chess.piece_name(expected_piece.piece_type)}"

    if not piece_name.startswith(expected_token) and not piece_name.startswith(spare_token):
        raise BoardStateError(
            f"Square {square_name} expects a {expected_token}, got piece '{piece_name}'."
        )


def validate_board_state(
    model: mujoco.MjModel,
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

    BoardObserver(model, mj_data).verify_stability(
        active_piece_names=set(square_to_piece.values())
    )


def detect_physical_instability(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    square_to_piece: dict[str, str],
) -> list[str]:
    """Return a list of pieces that are physically unstable (tilted or fallen)."""
    unstable = []
    active_names = set(square_to_piece.values())

    for name in active_names:
        try:
            body = mj_data.body(name)
            up_z = _up_z_from_quat(body.xquat)
            if up_z < TILT_THRESHOLD_COS:
                unstable.append(name)
        except KeyError:
            continue
    return unstable


def run_freeze_loop(viewer, mj_model, mj_data, max_cycles: int | None = None) -> None:
    """Keep the simulation frozen and the viewer open for inspection."""
    if viewer is None:
        return

    logger.info("Simulation frozen for inspection. Close viewer to exit.")
    cycles = 0
    while viewer.is_running():
        if max_cycles is not None and cycles >= max_cycles:
            break
        
        # Keep piece joints locked via high damping or zero velocity
        # (Though in passive viewer, we just don't call mj_step)
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()
        time.sleep(0.1)
        cycles += 1


def freeze_on_exception(
    exc: Exception,
    viewer=None,
    mj_model=None,
    mj_data=None,
) -> None:
    """Abort gameplay, log the error, and freeze the scene for visual debugging."""
    logger.error("FATAL RUNTIME EXCEPTION: %s", exc, exc_info=True)

    if viewer:
        run_freeze_loop(viewer, mj_model, mj_data)
    
    # Do not exit(1) during tests
    import sys
    if "pytest" not in sys.modules:
        sys.exit(1)
