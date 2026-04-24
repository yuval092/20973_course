import chess_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

def evaluate_model(model_path='checkpoints/best_model.zip', n_episodes=200):
    print(f"Loading model from {model_path}...")
    eval_env = gym.make('ChessFetch-v0')
    model = SAC.load(model_path, env=eval_env)
    
    successes, returns, goal_positions = [], [], []
    
    print(f"Running evaluation for {n_episodes} episodes...")
    # Get max steps from environment
    if hasattr(eval_env, 'spec') and eval_env.spec is not None:
        max_steps = eval_env.spec.max_episode_steps
    else:
        max_steps = 175 # Fallback to our new default
    print(f"Evaluating with {max_steps} step limit per episode.")
    
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        total_r = 0
        goal_positions.append(eval_env.unwrapped.goal[:2].copy())
        
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            total_r += r
            if term or trunc:
                break
        
        successes.append(info.get('is_success', 0))
        returns.append(total_r)
        
    eval_env.close()
    
    goal_positions = np.array(goal_positions)
    successes = np.array(successes)
    
    print(f"\nOverall Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"Mean Return:          {np.mean(returns):.2f}")
    
    # Regional Breakdown
    cx, cy = eval_env.unwrapped.TABLE_CENTER_XY
    hx, hy = eval_env.unwrapped.TABLE_HALF_X, eval_env.unwrapped.TABLE_HALF_Y
    
    center_thresh = 0.4    # inner 40%
    corner_thresh = 0.7    # outer 30%
    
    def get_region(pos):
        rx = abs(pos[0] - cx) / hx
        ry = abs(pos[1] - cy) / hy
        if rx < center_thresh and ry < center_thresh: return 'centre'
        if rx > corner_thresh and ry > corner_thresh: return 'corner'
        return 'edge'
        
    regions = [get_region(p) for p in goal_positions]
    print("\nRegional Performance:")
    for reg in ['centre', 'edge', 'corner']:
        mask = np.array([r == reg for r in regions])
        if mask.sum() > 0:
            rate = successes[mask].mean() * 100
            print(f"  {reg:8s}: {rate:5.1f}%  ({mask.sum()} episodes)")
        else:
            print(f"  {reg:8s}: N/A")

def get_latest_model():
    import glob
    # Check for best_model in timestamped run folders
    runs = glob.glob('checkpoints/fine_tune_*/best_model.zip')
    if runs:
        # Sort by folder name (which contains the timestamp)
        runs.sort()
        return runs[-1]
    return 'checkpoints/best_model.zip'

if __name__ == "__main__":
    import os
    best_model = get_latest_model()
    if os.path.exists(best_model):
        evaluate_model(best_model)
    else:
        print(f"No model found at {best_model}")
