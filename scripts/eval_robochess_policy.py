"""
Evaluation script for the fine-tuned RoboChess policy.
"""

import argparse
import json
import numpy as np
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv

def run_eval_suite(
    model,
    stage:      int = 3,
    n_episodes: int = 100,
    seed:       int = 9999,
) -> dict:
    metrics = {}
    metrics.update(_run_unit_suite(model, stage, n_episodes, seed))
    return metrics

def _run_unit_suite(model, stage, n_episodes, seed) -> dict:
    """
    Fixed source/dest tasks or sampled from specific stage.
    Always uses deterministic=True.
    """
    env = ChessManipulationTrainEnv(curriculum_stage=stage, seed=seed)
    
    successes, grasps, drops, xy_errs, z_errs = [], [], [], [], []

    obs, _ = env.reset()
    for _ in range(n_episodes):
        done = False
        grasped_this_ep = False
        info_final = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            grasped_this_ep = grasped_this_ep or info.get("grasped", False)
            done = terminated or truncated
            info_final = info

        successes.append(float(info_final.get("is_success", False)))
        grasps.append(float(grasped_this_ep))
        # Note: piece_dropped and other metrics should be in info
        drops.append(float(info_final.get("piece_dropped", False)))
        
        err_m = info_final.get("placement_error_m", 0.0)
        xy_errs.append(err_m * 1000)   # to mm
        
        # Reset for next episode
        obs, _ = env.reset()

    return {
        "grasp_rate":    float(np.mean(grasps)),
        "place_rate":    float(np.mean(successes)),
        "drop_rate":     float(np.mean(drops)),
        "mean_xy_err_mm": float(np.mean(xy_errs)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    model = TQC.load(args.checkpoint)
    metrics = run_eval_suite(model, args.stage, args.n_episodes, args.seed)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
