import time
import sys
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

# Ensure src is importable
sys.path.append(os.getcwd())

import src.chess_env

def evaluate_scenario(model, scenario, n_episodes, render=False, deterministic=True, debug=False, delay=0.0):
    # Load env config for evaluation parameters
    from src.utils.config import load_config
    env_cfg = load_config("env")
    drift_limit = env_cfg.get("eval_drift_limit", 0.005)
    CRASH_PENALTY = env_cfg.get("crash_penalty", -500.0)

    # Using 'human' for render_mode if requested
    render_mode = "human" if render else None
    env = gym.make("ChessFetchTask-v0", 
                   force_scenario=scenario, 
                   force_drift_limit=drift_limit,
                   render_mode=render_mode, 
                   debug=debug)
    
    successes = 0
    total_reward = 0
    crashes = 0
    timeouts = 0

    if debug:
        print(f"\n[EVAL] Starting {n_episodes} episodes for scenario: {scenario}")

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_success = False
        ep_crashed = False
        ep_crash_reason = None
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

            if render:
                env.render()
                if delay > 0:
                    time.sleep(delay)

            if terminated:
                if info.get("is_success") == 1.0:
                    ep_success = True
                    successes += 1
                elif reward <= CRASH_PENALTY + 1e-3:
                    ep_crashed = True
                    crashes += 1
                    ep_crash_reason = info.get("crash_reason")
            elif truncated:
                timeouts += 1

        total_reward += ep_reward

        if debug:
            ep_num = i + 1
            if ep_success:
                outcome = "SUCCESS"
            elif ep_crashed:
                outcome = f"CRASH ({ep_crash_reason})"
            elif truncated:
                outcome = "TIMEOUT"
            else:
                outcome = "FAIL"
            print(f"[EVAL] Episode {ep_num:3d}/{n_episodes} | {scenario:<8} | {outcome:<8} | steps={ep_steps:3d} | reward={ep_reward:8.2f}")

    env.close()
    return {
        "success_rate": successes / n_episodes,
        "crash_rate": crashes / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "avg_reward": total_reward / n_episodes
    }

def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation suite for RoboChess.")
    parser.add_argument("--model", type=str, default="models/latest_model.zip", help="Path to the model zip file (defaults to models/latest_model.zip).")
    parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenarios (transit,descend,ascend) or 'all'.")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes per scenario.")
    parser.add_argument("--visualize", action="store_true", help="Enable human rendering during evaluation.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps in seconds during visualization.")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging in the environment.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)
        
    print(f"Loading model: {args.model}")
    temp_env = gym.make("ChessFetchTask-v0")
    model = SAC.load(args.model, env=temp_env)
    temp_env.close()
    
    if args.scenarios == "all":
        scenarios = ["transit", "descend", "ascend"]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]
        
    print(f"\n{'Scenario':<12} | {'Success':<8} | {'Crash':<8} | {'Timeout':<8} | {'Avg Reward':<12}")
    print("-" * 58)

    for scenario in scenarios:
        results = evaluate_scenario(model, scenario, args.n_episodes, args.visualize, args.deterministic, debug=args.debug, delay=args.delay)
        print(f"{scenario:<12} | {results['success_rate']:>8.1%} | {results['crash_rate']:>8.1%} | {results['timeout_rate']:>8.1%} | {results['avg_reward']:>12.2f}")

if __name__ == "__main__":
    main()
