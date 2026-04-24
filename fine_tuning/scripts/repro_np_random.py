import gymnasium as gym
import chess_env
import numpy as np

env = gym.make('ChessFetch-v0').unwrapped
seed = 42
obs, info = env.reset(seed=seed)
state_after_reset = env.np_random.bit_generator.state

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
state_after_step = env.np_random.bit_generator.state

if state_after_reset == state_after_step:
    print("np_random state did NOT change after step.")
else:
    print("np_random state changed after step.")

# Gymnasium check_env check:
from gymnasium.utils.env_checker import check_env
try:
    check_env(env)
except Exception as e:
    print(f"check_env failed: {e}")
