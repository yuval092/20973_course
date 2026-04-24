import os
import sys
import numpy as np
import gymnasium as gym
import chess_env
from stable_baselines3 import SAC

def evaluate_scenario(model, scenario_name, num_episodes=100):
    env = gym.make('ChessFetchDense-v0', force_scenario=scenario_name)
    successes = 0
    collisions = 0
    
    print(f"Evaluating Scenario: {scenario_name}")
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        success = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if info.get('is_success') == 1.0:
                success = True
                
        if success:
            successes += 1
        elif reward <= -10.0:  # Terminal penalty indicating a collision/breach
            collisions += 1
            
    success_rate = (successes / num_episodes) * 100
    collision_rate = (collisions / num_episodes) * 100
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Collision Rate: {collision_rate:.1f}%")
    
    return success_rate, collision_rate

def evaluate_model(model_path, num_episodes=100):
    print(f"Loading model from {model_path}...")
    # Just need any valid dummy env to load the model
    dummy_env = gym.make('ChessFetchDense-v0')
    
    # Try multiple custom_objects configurations for robust loading
    model = None
    try:
        # Standard load for Phase 9 models
        custom_objects = {'ent_coef': 0.001}
        model = SAC.load(model_path, env=dummy_env, custom_objects=custom_objects)
    except Exception as e:
        print(f"Initial load failed: {e}. Trying fallback...")
        try:
            # Fallback for models without ent_coef_optimizer saved
            custom_objects = {'ent_coef': 'auto'}
            model = SAC.load(model_path, env=dummy_env, custom_objects=custom_objects)
        except Exception as e2:
            print(f"Fallback load failed: {e2}. Final attempt with minimal objects...")
            model = SAC.load(model_path, env=dummy_env)

    if model:
        model.ent_coef = 0.001
    else:
        print("Failed to load model.")
        return

    
    scenarios = ['transit', 'descend', 'ascend']
    results = {}
    
    for s in scenarios:
        s_rate, c_rate = evaluate_scenario(model, s, num_episodes)
        results[s] = {'success': s_rate, 'collision': c_rate}
        
    print("\n--- FINAL EVALUATION SUMMARY ---")
    for s, res in results.items():
        print(f"{s.upper()}: Success {res['success']:.1f}%, Collisions {res['collision']:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_phase9.py <path_to_model.zip>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
        
    evaluate_model(model_path, num_episodes=100)
