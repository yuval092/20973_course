import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)
env = gym.make('FetchPickAndPlace-v4').unwrapped
seed = 42
obs, info = env.reset(seed=seed)
state_before = env.np_random.bit_generator.state

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
state_after = env.np_random.bit_generator.state

if state_before == state_after:
    print("Original env np_random state did NOT change after step.")
else:
    print("Original env np_random state changed after step.")
