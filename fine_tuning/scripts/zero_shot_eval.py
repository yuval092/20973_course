import chess_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

model_path = 'models/sac-FetchPickAndPlace-v4.zip'

def run_zero_shot(n_episodes=100):
    print(f"Running zero-shot evaluation for {n_episodes} episodes...")
    env = gym.make('ChessFetch-v0')
    model = SAC.load(model_path, env=env)
    
    successes, returns = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        successes.append(info.get('is_success', 0))
        returns.append(total_r)
    
    env.close()
    mean_success = np.mean(successes)
    mean_return = np.mean(returns)
    print(f'Zero-shot success rate: {mean_success*100:.1f}%')
    print(f'Zero-shot mean return:  {mean_return:.2f}')
    return mean_success

if __name__ == "__main__":
    run_zero_shot()
