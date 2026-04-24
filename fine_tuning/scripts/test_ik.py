import gymnasium as gym
import numpy as np
import chess_env
import mujoco

env = gym.make('ChessFetchDense-v0', force_scenario='transit')

def test_settle():
    env.reset()
    start_pos = env.unwrapped.goal_pos.copy() # using goal as a random point to go to
    
    # zero out mocap
    for _ in range(100):
        grip_pos = env.unwrapped._utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, "robot0:grip")
        error = start_pos - grip_pos
        env.unwrapped.data.mocap_pos[0] += error * 0.5
        env.unwrapped._mujoco.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.n_substeps)
        
    grip_pos = env.unwrapped._utils.get_site_xpos(env.unwrapped.model, env.unwrapped.data, "robot0:grip")
    print(f"Error after 100 steps: {np.linalg.norm(start_pos - grip_pos)}")
    print(f"Grip: {grip_pos}, Target: {start_pos}")

for _ in range(5):
    test_settle()
