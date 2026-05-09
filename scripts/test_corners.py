#!/usr/bin/env python3
"""
Geometric Stress Test: Targets the corners and board edges.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from scripts.eval_sequence import ChainEpisodeRunner, print_summary

def main():
    # Board limits for Fetch (roughly)
    # x: [0.6, 1.2], y: [-0.3, 0.3]
    CORNERS = [
        np.array([0.6, -0.3]),
        np.array([0.6,  0.3]),
        np.array([1.2, -0.3]),
        np.array([1.2,  0.3]),
        np.array([0.9,  0.0]), # Center
    ]

    env = gym.make("ChessFetchTask-v0", force_scenario="transit")
    model = SAC.load("checkpoints/pure_movement_v6_20260502_171928/best_model_combined.zip", env=env)
    
    chain = ["transit", "descend", "ascend"]
    runner = ChainEpisodeRunner(env, model, chain, drift_limit=0.010, debug=True)

    all_results = []
    for i, corner in enumerate(CORNERS):
        print(f"\nTESTING CORNER {i+1}: {corner}")
        # waypoint logic: pick at the corner
        waypoints = [corner, corner, corner]
        res = runner.run_one_chain(waypoints)
        all_results.append(res)

    print_summary(chain, all_results, 0.010)
    env.close()

if __name__ == "__main__":
    main()
