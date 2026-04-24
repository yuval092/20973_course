import gymnasium as gym
import numpy as np
import chess_env
import mujoco

env = gym.make('ChessFetchDense-v0', force_scenario='transit')
env.reset()
print("Final grip pos:", env.unwrapped._utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, "robot0:grip"))
print("Target mocap pos:", env.unwrapped.data.mocap_pos[0])

