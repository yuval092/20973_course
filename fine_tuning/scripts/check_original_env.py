import gymnasium as gym
import gymnasium_robotics
from gymnasium.utils.env_checker import check_env

gym.register_envs(gymnasium_robotics)
env = gym.make('FetchPickAndPlace-v4').unwrapped
print("Checking FetchPickAndPlace-v4...")
try:
    check_env(env)
    print("FetchPickAndPlace-v4 passed.")
except Exception as e:
    print(f"FetchPickAndPlace-v4 failed: {e}")
