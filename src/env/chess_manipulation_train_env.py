"""Training-oriented task wrapper around the existing chess MuJoCo environment."""

from __future__ import annotations

import mujoco

from src.config import N_SUBSTEPS
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import BoardStateError
from src.training.tasks import ManipulationTask


class ChessManipulationTrainEnv:
    """Bind manipulation tasks onto the existing low-level chess environment.

    This class does not implement a full training reset pipeline yet. It
    prepares a task, exposes the expected target/goal wiring, and provides a
    stable place for future fine-tuning logic to grow.
    """

    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, square_to_piece: dict[str, str], n_substeps: int = N_SUBSTEPS):
        self.env = ChessPickPlaceEnv(mj_model, mj_data, n_substeps=n_substeps)
        self.controller = ExecutionController(None, self.env)
        self.square_to_piece = dict(square_to_piece)
        self.current_task: ManipulationTask | None = None

    def configure_task(self, task: ManipulationTask):
        """Bind a task to the low-level env and return the initial observation."""
        if self.square_to_piece.get(task.source_square) != task.piece_name:
            raise BoardStateError(
                f"Task piece mismatch for {task.source_square}: expected {self.square_to_piece.get(task.source_square)!r}, got {task.piece_name!r}."
            )
        self.current_task = task
        self.env.set_target(task.piece_name, self.controller.get_square_pos(task.dest_square))
        return self.env.get_obs()

    def get_task_info(self) -> dict[str, object]:
        """Return metadata for the currently bound task."""
        if self.current_task is None:
            raise BoardStateError("No manipulation task configured.")
        return {
            "move_uci": self.current_task.move_uci,
            "piece_name": self.current_task.piece_name,
            "source_square": self.current_task.source_square,
            "dest_square": self.current_task.dest_square,
            "is_capture": self.current_task.is_capture,
        }
