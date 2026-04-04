"""Run a single headless manipulation op for smoke testing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import N_SUBSTEPS, SCENE_XML  # noqa: E402
from src.control.execution_controller import ExecutionController  # noqa: E402
from src.env.chess_pick_place_env import ChessPickPlaceEnv  # noqa: E402
from src.logic.operation_planner import PickPlaceOp  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("headless_run")


def run_op(op: PickPlaceOp) -> None:
    """Execute one operation without launching the interactive viewer."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)

    logger.info(
        "Starting headless op | "
        f"piece={op.piece_name} | "
        f"src={op.target_square} | "
        f"dest={op.dest_square}"
    )
    result = controller.execute_op(op, viewer=None)
    body = data.body(op.piece_name)
    quat = body.xquat.copy()
    up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
    logger.info(
        "Completed headless op | "
        f"success={result.success} | "
        f"steps={result.steps_taken} | "
        f"error_mm={result.final_error_mm:.1f} | "
        f"details={result.details}"
    )
    logger.info(
        "Post-op piece state | "
        f"piece={op.piece_name} | "
        f"pos={body.xpos.round(4)} | "
        f"quat={quat.round(4)} | "
        f"up_z={up_z:.4f}"
    )


if __name__ == "__main__":
    run_op(PickPlaceOp(target_square="c7", dest_square="c5", piece_name="b_pawn_3"))
