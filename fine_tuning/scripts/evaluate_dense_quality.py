import numpy as np
import gymnasium as gym
import chess_env
from stable_baselines3 import SAC

def evaluate_quality(model_path, n_episodes=10):
    print(f"Evaluating quality metrics for: {model_path}")
    
    # Robust load: load without env first
    model = SAC.load(model_path)
    env = gym.make('ChessFetchDense-v0')
    model.set_env(env)
    model.tensorboard_log = None
    
    accuracies = []
    velocities = []
    collisions = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 175:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            if info.get('is_floor_collision'):
                collisions += 1
                
        # Calculate final precision
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        accuracies.append(dist)
        
        # Capture velocity at the end (should be low)
        obj_vel = info.get('obj_vel', np.array([0,0,0]))
        velocities.append(np.linalg.norm(obj_vel))
        
    env.close()
    
    print("\n--- Phase 8 Quality Report ---")
    print(f"Success Rate (20mm): {np.mean(np.array(accuracies) < 0.02) * 100:.1f}%")
    print(f"Mean Accuracy:       {np.mean(accuracies)*1000:.2f} mm")
    print(f"Best Accuracy:       {np.min(accuracies)*1000:.2f} mm")
    print(f"Mean Impact Vel:     {np.mean(velocities):.4f} m/s")
    print(f"Floor Collisions:    {collisions}")
    print("------------------------------\n")

if __name__ == "__main__":
    evaluate_quality('checkpoints/dense_refinement_20260423_124752/best_model.zip')
