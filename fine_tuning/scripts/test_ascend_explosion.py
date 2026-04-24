import gymnasium as gym
import numpy as np
import chess_env

env = gym.make('ChessFetchDense-v0', force_scenario='ascend')
obs, _ = env.reset()
print(f"Start pos: {obs['observation'][:3]}")

# Take 1 step
obs, reward, terminated, truncated, info = env.step(np.array([0,0,0,0], dtype=np.float32))
print(f"Step 1: pos: {obs['observation'][:3]}, reward: {reward}, terminated: {terminated}")
if terminated:
    print(f"Info: {info}")
