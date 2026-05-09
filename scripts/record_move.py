#!/usr/bin/env python3
"""
Visual Recorder: Captures a full move and saves frames to disk.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from src.chess_env.waypoints import derive_goal_pos, exit_waypoint
import PIL.Image

def main():
    frame_dir = "logs/visual_recording"
    os.makedirs(frame_dir, exist_ok=True)

    # Use offscreen rendering
    env = gym.make("ChessFetchTask-v0", render_mode="rgb_array", width=640, height=480)
    model = SAC.load("checkpoints/pure_movement_v6_20260502_171928/best_model_combined.zip", env=env)
    
    chain = ["transit", "descend", "ascend", "transit", "descend", "ascend"]
    
    # Pre-sample standard board positions
    src_xy = np.array([0.8, 0.1])
    dst_xy = np.array([0.9, -0.1])
    waypoints = [src_xy]*3 + [dst_xy]*3

    obs, _ = env.reset()
    uw = env.unwrapped
    uw.force_drift_limit = 0.010
    
    frame_idx = 0
    
    for i, scenario in enumerate(chain):
        cell_xy = waypoints[i]
        goal = derive_goal_pos(scenario, cell_xy)
        
        if i > 0:
            nom_exit = exit_waypoint(chain[i-1], waypoints[i-1])
            obs, _ = uw.soft_reset(scenario, goal, nom_exit, cell_xy)
        else:
            uw.current_scenario = scenario
            uw.goal_pos = goal
            uw.goal = goal
            if scenario in {"descend", "ascend"}: uw.tube_center_xy = cell_xy
            obs = uw._get_obs()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Capture frame every 2 simulation steps to keep it smooth but light
            if frame_idx % 2 == 0:
                frame = env.render()
                img = PIL.Image.fromarray(frame)
                img.save(os.path.join(frame_dir, f"frame_{frame_idx:04d}.png"))
            frame_idx += 1
            
        print(f"Finished scenario {i+1}: {scenario}")

    print(f"\nRecording complete! {frame_idx} frames saved to {frame_dir}")
    env.close()

if __name__ == "__main__":
    main()
