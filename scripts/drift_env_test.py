"""Measure piece drift over idle physics steps using the actual environment."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.config import SCENE_XML, N_SUBSTEPS

def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Init env to trigger the settle logic
    env = ChessPickPlaceEnv(model, data)
    mujoco.mj_forward(model, data)

    # Record initial positions of all pieces AFTER env init
    piece_names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and (name.startswith("w_") or name.startswith("b_")) and "spare" not in name:
            piece_names.append(name)

    initial = {}
    for name in piece_names:
        body = data.body(name)
        initial[name] = body.xpos.copy()

    # Step the simulation 500 times (no actions, no arm movement)
    for _ in range(500):
        mujoco.mj_step(model, data, N_SUBSTEPS)

    # Check final positions
    print(f"{'Piece':<20} {'Init XY':>20} {'Final XY':>20} {'XY Drift (mm)':>15} {'Z Drift (mm)':>15}")
    print("-" * 95)
    drifts = []
    for name in sorted(piece_names):
        body = data.body(name)
        final = body.xpos.copy()
        init = initial[name]
        xy_drift = np.linalg.norm(final[:2] - init[:2]) * 1000
        z_drift = abs(final[2] - init[2]) * 1000
        drifts.append(xy_drift)
        flag = " *** DRIFT!" if xy_drift > 5.0 else ""
        print(
            f"{name:<20} "
            f"({init[0]:.4f}, {init[1]:.4f}) "
            f"({final[0]:.4f}, {final[1]:.4f}) "
            f"{xy_drift:>12.2f}   "
            f"{z_drift:>12.2f}"
            f"{flag}"
        )

    print(f"\nMax XY drift: {max(drifts):.2f} mm")
    print(f"Mean XY drift: {np.mean(drifts):.2f} mm")
    print(f"Pieces with >5mm drift: {sum(1 for d in drifts if d > 5.0)}")

if __name__ == "__main__":
    main()
