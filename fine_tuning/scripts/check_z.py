import chess_env
import gymnasium as gym
import numpy as np

try:
    env = gym.make('ChessFetch-v0')
    obs, _ = env.reset()
    # object_pos is typically obs['observation'][3:6]
    # Fetch env observation:
    # 0:3 gripper pos
    # 3:6 object pos
    # 6:9 object rel pos
    # ...
    block_z = obs['observation'][5]
    print(f'Block Z after reset: {block_z}')
    env.close()
except Exception as e:
    print(f'Error: {e}')
    exit(1)
