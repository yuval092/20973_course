import gymnasium as gym
import gymnasium_robotics
import os
from stable_baselines3 import SAC

# Ensure gymnasium-robotics envs are registered
gym.register_envs(gymnasium_robotics)

model_path = 'models/sac-FetchPickAndPlace-v4.zip'

if not os.path.exists(model_path):
    print(f'Model not found at {model_path}')
    exit(1)

try:
    env = gym.make('FetchPickAndPlace-v4')
    model = SAC.load(model_path, env=env)
    print('Action space:', env.action_space)
    print('Obs space:   ', env.observation_space)
    env.close()
    print('Model loaded successfully.')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)
