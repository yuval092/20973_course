import os
import numpy as np
import gymnasium as gym
import chess_env
from gymnasium.utils.env_checker import check_env

def verify_check_env():
    print("Running check_env...")
    env = gym.make('ChessFetch-v0').unwrapped
    check_env(env, warn=True)
    env.close()
    print("check_env passed.")

def verify_goal_boundaries():
    print("Verifying goal boundaries...")
    env = gym.make('ChessFetch-v0').unwrapped
    goals = []
    for _ in range(200):
        env.reset()
        goals.append(env.goal.copy())
    goals = np.array(goals)
    
    cx, cy = env.TABLE_CENTER_XY
    hx = env.TABLE_HALF_X - env.EDGE_MARGIN
    hy = env.TABLE_HALF_Y - env.EDGE_MARGIN
    
    x_ok = np.all(goals[:, 0] >= cx - hx) and np.all(goals[:, 0] <= cx + hx)
    y_ok = np.all(goals[:, 1] >= cy - hy) and np.all(goals[:, 1] <= cy + hy)
    z_ok = np.allclose(goals[:, 2], env.TABLE_SURFACE_Z, atol=1e-3)
    
    if not x_ok:
        print(f"X out of range: min {goals[:, 0].min()}, max {goals[:, 0].max()}, expected [{cx-hx}, {cx+hx}]")
    if not y_ok:
        print(f"Y out of range: min {goals[:, 1].min()}, max {goals[:, 1].max()}, expected [{cy-hy}, {cy+hy}]")
    if not z_ok:
        print(f"Z not on surface: mean {goals[:, 2].mean()}, expected {env.TABLE_SURFACE_Z}")
        
    assert x_ok and y_ok and z_ok, "Goal boundaries verification failed"
    print("All goals within board bounds.")
    env.close()

def verify_arm_reachability():
    print("Verifying arm reachability (target >= 90%)...")
    env = gym.make('ChessFetch-v0')
    reached = 0
    total = 50
    for i in range(total):
        obs, _ = env.reset()
        target = obs['desired_goal']
        for step in range(150):
            grip_pos = obs['observation'][:3]
            delta = (target - grip_pos) * 5.0
            delta = np.clip(delta, -1, 1)
            action = np.append(delta, 0.0)
            obs, _, term, trunc, _ = env.step(action)
            if np.linalg.norm(obs['observation'][:3] - target) < 0.05:
                reached += 1
                break
    success_rate = reached / total
    print(f"Reachability: {reached}/{total} = {100*success_rate:.1f}%")
    assert success_rate >= 0.90, "Board too large. Reachability below 90%."
    env.close()

if __name__ == "__main__":
    try:
        verify_check_env()
        verify_goal_boundaries()
        verify_arm_reachability()
        print("\nAll verification steps PASSED!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        exit(1)
