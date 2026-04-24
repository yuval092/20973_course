import os
import time
import numpy as np
import gymnasium as gym
import mujoco.viewer
import chess_env
from scripts.coordinate_mapping import ChessCoordinateMapper
from stable_baselines3 import SAC

def play_move(start_sq='e2', end_sq='e4', model_path='chess_fetch_20260421_152804.zip', speed=0.02):
    print(f"--- RoboChess Interactive Move: {start_sq} -> {end_sq} ---")
    
    # 1. Setup Mapper and Coordinates
    mapper = ChessCoordinateMapper()
    
    # Logic for air positions (suffix '^')
    def get_pos(sq_not):
        is_air = '^' in sq_not
        base_sq = sq_not.replace('^', '')
        pos = mapper.notation_to_world(base_sq)
        if is_air:
            pos[2] += 0.15 # Hover 15cm above table
        return pos, is_air

    start_pos, start_is_air = get_pos(start_sq)
    end_pos, end_is_air = get_pos(end_sq)
    
    # 2. Setup Environment
    env = gym.make('ChessFetch-v0', render_mode='human')
    model = SAC.load(model_path, env=env)
    
    # 3. Reset and Manually Position Object/Goal
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    
    # Set Object position
    obj_joint_id = unwrapped.model.joint("object0:joint").id
    qpos_start = unwrapped.model.jnt_qposadr[obj_joint_id]
    unwrapped.data.qpos[qpos_start : qpos_start + 3] = start_pos
    unwrapped.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0] # Upright
    
    # Set Goal position
    unwrapped.goal = end_pos.copy()
    
    # Update physics
    import mujoco
    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    obs, _, _, _, _ = env.step(np.zeros(4)) 
    
    print(f"Viewer launching... waiting 2 seconds before movement starts.")
    
    # 4. Run movement loop
    with mujoco.viewer.launch_passive(unwrapped.model, unwrapped.data) as viewer:
        # 2 Second Delay
        time.sleep(2)
        
        print(f"Starting movement. Speed delay: {speed}s")
        terminated = False
        truncated = False
        step_count = 0
        
        while viewer.is_running() and not (terminated or truncated):
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Sync viewer
            viewer.sync()
            
            # Artificial delay to make it look smooth and slow
            time.sleep(speed)
            
            step_count += 1
            dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            if step_count % 20 == 0:
                print(f" Step {step_count}: Distance to goal = {dist:.3f}m")

        # Final verification: Check if piece is within 2cm of square center
        final_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        if final_dist < 0.02:
            print(f"\nSUCCESS! Move {start_sq} -> {end_sq} completed perfectly.")
            print(f"Final accuracy: {final_dist*1000:.1f} mm")
        elif info.get('is_success'):
            print(f"\nNEAR SUCCESS: Piece is on the square but slightly offset.")
            print(f"Final accuracy: {final_dist*1000:.1f} mm")
        else:
            print(f"\nMove failed. Final distance: {final_dist*1000:.1f} mm")
            
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='e2')
    parser.add_argument('--end', type=str, default='e4')
    parser.add_argument('--speed', type=float, default=0.03, help="Seconds delay between steps")
    args = parser.parse_args()
    
    try:
        play_move(args.start, args.end, speed=args.speed)
    except Exception as e:
        print(f"Error: {e}")
