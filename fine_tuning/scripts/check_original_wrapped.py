import gymnasium as gym
import gymnasium_robotics
from gymnasium.utils.env_checker import check_env

gym.register_envs(gymnasium_robotics)
env = gym.make('FetchPickAndPlace-v4')
print("Checking FetchPickAndPlace-v4 (wrapped)...")
try:
    check_env(env)
    print("FetchPickAndPlace-v4 (wrapped) passed.")
except Exception as e:
    print(f"FetchPickAndPlace-v4 (wrapped) failed: {e}")
