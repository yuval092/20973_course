"""
Training-oriented task wrapper for the chess MuJoCo environment.

This module provides a high-level environment wrapper that binds specific
manipulation tasks to the underlying MuJoCo physics simulation.
"""

from src.config import N_SUBSTEPS
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import BoardStateError


class ChessManipulationTrainEnv:
    """
    Binds manipulation tasks onto the low-level chess environment.

    This class prepares tasks, exposes target/goal configurations, and
    provides a platform for training and fine-tuning manipulation policies.
    """

    def __init__(self, mj_model, mj_data, square_to_piece, n_substeps=N_SUBSTEPS):
        """
        Initialize the training environment.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            square_to_piece: Dictionary mapping square names to piece names.
            n_substeps: Number of physics sub-steps per action step.
        """
        self.env = ChessPickPlaceEnv(mj_model, mj_data, n_substeps=n_substeps)
        self.controller = ExecutionController(None, self.env)
        self.square_to_piece = dict(square_to_piece)
        self.current_task = None

    def configure_task(self, task):
        """
        Bind a task to the low-level environment.

        Args:
            task: ManipulationTask instance to configure.

        Returns:
            The initial observation after task configuration.

        Raises:
            BoardStateError: If the task piece does not match the board state.
        """
        if self.square_to_piece.get(task.source_square) != task.piece_name:
            raise BoardStateError(
                f"Task piece mismatch for {task.source_square}: "
                f"expected {self.square_to_piece.get(task.source_square)!r}, "
                f"got {task.piece_name!r}."
            )
        self.current_task = task
        self.env.set_target(task.piece_name, self.controller.get_square_pos(task.dest_square))
        return self.env.get_obs()

    def get_task_info(self):
        """
        Return metadata for the currently bound task.

        Returns:
            Dictionary containing task details like UCI move, piece name, etc.

        Raises:
            BoardStateError: If no task has been configured.
        """
        if self.current_task is None:
            raise BoardStateError("No manipulation task configured.")
        return {
            "move_uci": self.current_task.move_uci,
            "piece_name": self.current_task.piece_name,
            "source_square": self.current_task.source_square,
            "dest_square": self.current_task.dest_square,
            "is_capture": self.current_task.is_capture,
        }
