import chess_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

def diagnose_failures(model_path='checkpoints/best_model.zip', n_episodes=100):
    print(f"Loading model from {model_path}...")
    eval_env = gym.make('ChessFetch-v0')
    model = SAC.load(model_path, env=eval_env)
    
    cx, cy = eval_env.unwrapped.TABLE_CENTER_XY
    hx, hy = eval_env.unwrapped.TABLE_HALF_X, eval_env.unwrapped.TABLE_HALF_Y
    
    # We want to specifically sample corner goals to test them
    corner_thresh = 0.7
    
    failures = 0
    truncations = 0
    terminations = 0
    
    # Get max steps from environment
    if hasattr(eval_env, 'spec') and eval_env.spec is not None:
        max_steps = eval_env.spec.max_episode_steps
    else:
        max_steps = 175
    print(f"Diagnosing with {max_steps} step limit.")
    
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        
        # --- FORCED CORNER SAMPLING ---
        # Instead of waiting for a corner, we force the goal into a corner
        # for every single episode to gather data quickly.
        hx_m = eval_env.unwrapped.TABLE_HALF_X - eval_env.unwrapped.EDGE_MARGIN
        hy_m = eval_env.unwrapped.TABLE_HALF_Y - eval_env.unwrapped.EDGE_MARGIN
        
        # Pick one of the 4 corners randomly
        sign_x = 1 if np.random.random() > 0.5 else -1
        sign_y = 1 if np.random.random() > 0.5 else -1
        
        # Add a tiny bit of noise so it's not the exact same point
        corner_goal = np.array([
            cx + (sign_x * hx_m) + np.random.uniform(-0.02, 0.02),
            cy + (sign_y * hy_m) + np.random.uniform(-0.02, 0.02),
            eval_env.unwrapped.TABLE_SURFACE_Z
        ])
        
        eval_env.unwrapped.goal = corner_goal.copy()
        obs['desired_goal'] = corner_goal.copy()
        # ------------------------------
            
        total_r = 0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            total_r += r
            if term or trunc:
                break
        
        success = info.get('is_success', 0)
        if not success:
            failures += 1
            if trunc and not term:
                truncations += 1
            elif term:
                terminations += 1
                
    eval_env.close()
    print(f"Total corner episodes: {n_episodes}")
    print(f"Analyzed {failures} corner failures.")
    if failures > 0:
        print(f" - Failed due to truncation (ran out of time): {truncations} ({(truncations/failures)*100:.1f}%)")
        print(f" - Failed due to termination (dropped/crashed): {terminations} ({(terminations/failures)*100:.1f}%)")
    print(f"Corner Success Rate: {((n_episodes - failures)/n_episodes)*100:.1f}%")

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
    # Prefer the final model if it exists, otherwise get latest best
    if os.path.exists('chess_fetch_20260421_095337.zip'):
        diagnose_failures('chess_fetch_20260421_095337.zip', n_episodes=50)
    else:
        latest = get_latest_model()
        diagnose_failures(latest, n_episodes=50)
