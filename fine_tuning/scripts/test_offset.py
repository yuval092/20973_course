import gymnasium as gym
import numpy as np
import chess_env
import mujoco

env = gym.make('ChessFetchDense-v0', force_scenario='transit')

for i in range(5):
    env.reset()
    grip = env.unwrapped._utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, "robot0:grip")
    mocap = env.unwrapped.data.mocap_pos[0]
    print(f"Diff: {mocap - grip}")

