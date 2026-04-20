"""
Run a single headless manipulation op for smoke testing.

This script allows for the execution of a single chess piece manipulation
operation without a GUI. It's useful for testing the logic and physics
in a headless environment.
"""

import logging
import sys
from pathlib import Path

import mujoco

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.logic.operation_planner import PickPlaceOp


class HeadlessRunner:
    """
    A class to execute chess operations in a headless MuJoCo environment.
    """

    def __init__(self, scene_xml=SCENE_XML):
        """
        Initialize the HeadlessRunner.

        Args:
            scene_xml: Path to the MuJoCo scene XML file.
        """
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.logger = logging.getLogger("headless_run")

    def run_op(self, op):
        """
        Execute one operation without launching the interactive viewer.

        Args:
            op: A PickPlaceOp instance describing the move.
        """
        mujoco.mj_forward(self.model, self.data)

        env = ChessPickPlaceEnv(self.model, self.data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)

        self.logger.info(
            "Starting headless op | "
            f"piece={op.piece_name} | "
            f"src={op.target_square} | "
            f"dest={op.dest_square}"
        )
        
        result = controller.execute_op(op, viewer=None)
        
        body = self.data.body(op.piece_name)
        quat = body.xquat.copy()
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        
        self.logger.info(
            "Completed headless op | "
            f"success={result.success} | "
            f"steps={result.steps_taken} | "
            f"error_mm={result.final_error_mm:.1f} | "
            f"details={result.details}"
        )
        self.logger.info(
            "Post-op piece state | "
            f"piece={op.piece_name} | "
            f"pos={body.xpos.round(4)} | "
            f"quat={quat.round(4)} | "
            f"up_z={up_z:.4f}"
        )


def main():
    """
    Main entry point for the headless run script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    runner = HeadlessRunner()
    test_op = PickPlaceOp(
        target_square="c7",
        dest_square="c5",
        piece_name="b_pawn_3"
    )
    runner.run_op(test_op)


if __name__ == "__main__":
    main()
