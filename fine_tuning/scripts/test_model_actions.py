import gymnasium as gym
import numpy as np
import chess_env
from stable_baselines3 import SAC

env = gym.make('ChessFetchDense-v0', force_scenario='transit')
model = SAC.load('models/final_grandmaster_model.zip', env=env)

for i in range(2):
    obs, _ = env.reset()
    done = False
    step = 0
    print(f"\nEpisode {i+1} Start: pos={obs['observation'][:3]}, goal={obs['desired_goal']}")
    while not done and step < 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        print(f"  Step {step}: action={action}, pos={obs['observation'][:3]}, reward={reward}, term={term}")
        done = term or trunc
