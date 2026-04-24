import gymnasium as gym
import numpy as np
import chess_env

env = gym.make('ChessFetchDense-v0', force_scenario='transit')

for i in range(5):
    obs, _ = env.reset()
    print(f"\nEpisode {i+1} Start: {obs['observation'][:3]}")
    done = False
    step = 0
    while not done and step < 5:
        # Action that does nothing
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        print(f"  Step {step}: pos={obs['observation'][:3]}, reward={reward}, term={term}, trunc={trunc}, success={info.get('is_success')}")
        done = term or trunc

