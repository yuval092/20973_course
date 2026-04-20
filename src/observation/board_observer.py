"""
Physical board stability checks for MuJoCo chess scenes.

This module provides the BoardObserver class, which is responsible for
monitoring the physical state of the chess pieces in the MuJoCo simulation.
It checks for pieces that have tipped over, fallen below the table, or
drifted significantly from their expected positions.
"""

import numpy as np
import mujoco

from src.exceptions import StabilityError
from src.config import (
    BOARD_CENTER,
    SQUARE_SIZE,
    TABLE_HEIGHT,
    TILT_THRESHOLD_COS,
    DISPLACEMENT_THRESHOLD
)

# Bodies that should be ignored during stability checks.
IGNORED_BODY_TOKENS = (
    "world", "table", "spare", "graveyard", "base", "fetch", "robot", "mocap",
)


class BoardObserver:
    """
    Inspect the scene for tipped, buried, or displaced pieces.

    The observer uses the MuJoCo model and data to validate that all chess
    pieces are in a stable and expected physical state.
    """

    def __init__(self, model, data):
        """
        Initialize the BoardObserver.

        Args:
            model: The MuJoCo model (MjModel).
            data: The MuJoCo data (MjData).
        """
        self.model = model
        self.data = data

    @staticmethod
    def _up_z(quat):
        """
        Compute the world-space Z component of the body's local up axis.

        Args:
            quat: MuJoCo quaternion (w, x, y, z).

        Returns:
            float: The Z component of the local up axis in world space.
        """
        return 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)

    @staticmethod
    def _is_ignored_body(name):
        """
        Return True when a MuJoCo body should not be treated as a chess piece.

        Checks if the body name is empty or contains any of the ignored tokens.

        Args:
            name: The name of the MuJoCo body.

        Returns:
            bool: True if the body should be ignored, False otherwise.
        """
        return not name or any(token in name for token in IGNORED_BODY_TOKENS)

    def verify_stability(self, active_piece_names=None):
        """
        Check every relevant piece for tilt, burial, and large displacement.

        Iterates through all bodies in the MuJoCo model and performs physical
        sanity checks on those identified as chess pieces.

        Args:
            active_piece_names: Optional list of piece names to exclusively check.
                               If None, all non-ignored pieces are checked.

        Raises:
            StabilityError: If a piece is tipped, buried, or displaced.
        """
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if self._is_ignored_body(name):
                continue
            if active_piece_names is not None and name not in active_piece_names:
                continue

            quat = self.data.body(i).xquat
            # MuJoCo quaternions are stored as w, x, y, z. For an upright
            # piece, its local +Z axis should still point mostly upward.
            up_z = self._up_z(quat)
            if up_z < TILT_THRESHOLD_COS:
                raise StabilityError(
                    f"Piece '{name}' has tipped over (up_z={up_z:.3f})"
                )

            pos = self.data.body(i).xpos
            if pos[2] < TABLE_HEIGHT - 0.01:
                raise StabilityError(
                    f"Piece '{name}' is below the table level (z={pos[2]:.3f})"
                )

            dx = abs(pos[0] - BOARD_CENTER[0])
            dy = abs(pos[1] - BOARD_CENTER[1])
            if dx < 0.35 and dy < 0.35:
                if pos[2] < TABLE_HEIGHT + 0.05:
                    # Convert world-space XY back into board-cell coordinates
                    # to estimate drift away from the nearest square center.
                    rel_x = (pos[0] - BOARD_CENTER[0]) / SQUARE_SIZE + 3.5
                    rel_y = (pos[1] - BOARD_CENTER[1]) / SQUARE_SIZE + 3.5

                    dist_x = abs(rel_x - round(rel_x))
                    dist_y = abs(rel_y - round(rel_y))
                    err_m = np.sqrt(
                        (dist_x * SQUARE_SIZE) ** 2 + (dist_y * SQUARE_SIZE) ** 2
                    )

                    if err_m > SQUARE_SIZE * DISPLACEMENT_THRESHOLD:
                        raise StabilityError(
                            f"Piece '{name}' is significantly displaced from "
                            f"square center (err={err_m:.3f}m)"
                        )
