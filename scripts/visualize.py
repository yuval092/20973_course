import sys
import os
import argparse
import time
import gymnasium as gym
from stable_baselines3 import SAC

# Ensure src is importable
sys.path.append(os.getcwd())

import src.chess_env

def main():
    parser = argparse.ArgumentParser(description="Qualitative visualization for RoboChess.")
    parser.add_argument("--model", type=str, default="models/latest_model.zip", help="Path to the model zip file (optional, defaults to latest_model.zip).")
    parser.add_argument("--scenario", type=str, default="transit", choices=["transit", "descend", "ascend", "random"], help="Scenario to visualize.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps in seconds.")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 for infinite).")
    parser.add_argument("--wait", action="store_true", help="Wait for keypress after each episode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging in the environment.")
    
    args = parser.parse_args()
    
    force_scenario = args.scenario if args.scenario != "random" else None
    
    print(f"Starting visualization for scenario: {args.scenario}")
    env = gym.make("ChessFetchTask-v0", force_scenario=force_scenario, render_mode="human", debug=args.debug)
    
    model = None
    model_path = args.model
    if model_path:
        if os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            model = SAC.load(model_path, env=env)
        else:
            print(f"Warning: Model file {model_path} not found. Using random actions.")
            
    ep = 0
    try:
        while True:
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                    
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                
                env.render()
                if args.delay > 0:
                    time.sleep(args.delay)
            
            ep += 1
            print(f"Episode {ep} finished. Reward: {ep_reward:.2f}, Success: {info.get('is_success')}")
            
            if args.wait:
                input("Press Enter to continue to next episode...")
                
            if args.episodes > 0 and ep >= args.episodes:
                break
                
    except KeyboardInterrupt:
        print("\nExiting visualization.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
